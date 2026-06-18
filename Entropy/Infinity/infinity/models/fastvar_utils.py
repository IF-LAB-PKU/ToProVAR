import os
from contextlib import contextmanager
import math
from typing import Tuple

import torch
import torch.nn.functional as F

import time


def do_nothing(x: torch.Tensor, *args, **kwargs):
    return x

def skip_layer(x: torch.Tensor, *args, **kwargs):
    return 0




size_to_width_need_cache = {
    1: 0,
    2: 1,
    3: 2,
    4: 4,
    5: 6,
    6: 8,
    7: 12,
    8: 16,
    9: 20,
    10: 24,
    11: 32,
    12: 40,
    100: 100000, # effectively disables cache-width pruning
}


def _normalize_toprovar_ratio(ratio: float) -> float:
    ratio = float(ratio)
    if not math.isfinite(ratio):
        raise ValueError(f'ToProVAR ratio must be finite, got {ratio!r}')
    return min(max(ratio, 0.0), 0.999)


def _normalize_toprovar_fraction(fraction: float) -> float:
    fraction = float(fraction)
    if not math.isfinite(fraction):
        raise ValueError(f'ToProVAR fraction must be finite, got {fraction!r}')
    return min(max(fraction, 0.0), 1.0)


def parse_toprovar_layer_ratio_schedule(raw_schedule: str):
    schedule = {}
    if raw_schedule is None or str(raw_schedule).strip() == '':
        return schedule

    for item in str(raw_schedule).split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(
                f'Invalid TOPROVAR_LAYER_RATIO_SCHEDULE item {item!r}; '
                'expected scale:ratio'
            )
        scale_text, ratio_text = item.split(':', 1)
        scale_ind = int(scale_text.strip())
        if scale_ind < 0:
            raise ValueError(f'ToProVAR scale index must be non-negative, got {scale_ind}')
        schedule[scale_ind] = _normalize_toprovar_ratio(ratio_text.strip())
    return schedule


def parse_toprovar_token_ratio_schedule(raw_schedule: str):
    schedule = {}
    if raw_schedule is None or str(raw_schedule).strip() == '':
        return schedule

    for item in str(raw_schedule).split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item or '|' not in item:
            raise ValueError(
                f'Invalid TOPROVAR_TOKEN_RATIO_SCHEDULE item {item!r}; '
                'expected scale:low|high'
            )
        scale_text, ratio_text = item.split(':', 1)
        low_text, high_text = ratio_text.split('|', 1)
        scale_ind = int(scale_text.strip())
        if scale_ind < 0:
            raise ValueError(f'ToProVAR scale index must be non-negative, got {scale_ind}')
        schedule[scale_ind] = (
            _normalize_toprovar_ratio(low_text.strip()),
            _normalize_toprovar_ratio(high_text.strip()),
        )
    return schedule


def parse_toprovar_token_high_entropy_fraction_schedule(raw_schedule: str):
    schedule = {}
    if raw_schedule is None or str(raw_schedule).strip() == '':
        return schedule

    for item in str(raw_schedule).split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(
                f'Invalid TOPROVAR_TOKEN_HIGH_ENTROPY_FRACTION_SCHEDULE item {item!r}; '
                'expected scale:fraction'
            )
        scale_text, fraction_text = item.split(':', 1)
        scale_ind = int(scale_text.strip())
        if scale_ind < 0:
            raise ValueError(f'ToProVAR scale index must be non-negative, got {scale_ind}')
        schedule[scale_ind] = _normalize_toprovar_fraction(fraction_text.strip())
    return schedule


def parse_toprovar_scale_set(raw_scales):
    if raw_scales is None:
        return set()
    if isinstance(raw_scales, (list, tuple, set)):
        items = raw_scales
    else:
        raw_text = str(raw_scales).strip()
        if raw_text == '':
            return set()
        items = raw_text.split(',')
    scale_set = set()
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        scale_ind = int(text)
        if scale_ind < 0:
            raise ValueError(f'ToProVAR scale index must be non-negative, got {scale_ind}')
        scale_set.add(scale_ind)
    return scale_set


class ToProVARControllerForExperiment:
    '''
    Controller for scale skipping, SVD layer grouping, and token pruning.

    The controller centralizes pruning decisions from CLI/runtime flags:
    `only_scale_skip` controls direct scale skipping, `use_adaptive_scale`
    enables entropy-based scale triggering, `use_svd_depart_layers` selects
    layer groups from SVD statistics, and `use_token_pruning` switches between
    token pruning and full layer skipping once a scale/layer is selected.
    '''
    def __init__(
        self,
        only_scale_skip: bool,
        use_adaptive_scale: bool,
        scale_prop: float,
        entropy_boundary_line: float,
        start_scale_index: int,
        direct_skip_start_scale_index: int,
        use_svd_depart_layers: bool,
        structure_layer_ration: float,
        use_token_pruning: bool,
        svd_rho_threshold=None,
        svd_source_scale_index: int = 7,
        svd_split_mode: str = 'ratio',
        protected_global_layers=None,
        adaptive_entropy_s_min=None,
        adaptive_entropy_collect_start_scale_index=7,
        adaptive_entropy_trigger_start_scale_index=8,
        adaptive_entropy_gamma: float = 1.4e-6,
        adaptive_entropy_epsilon: float = 1e-5,
        adaptive_entropy_lambda: float = 1.0,
        adaptive_entropy_score_mode: str = 'delta_h_width_power',
        adaptive_entropy_signal_mode: str = 'layer_mean_delta',
        adaptive_entropy_signal_layer_idx: int = 5,
        adaptive_max_raw_start_scale_index: int = 10,
        force_entropy_collection_scales=None,
    ) -> None:
        self.only_scale_skip = only_scale_skip
        self.use_adaptive_scale = use_adaptive_scale
        self.scale_prop = scale_prop 
        self.entropy_boundary_line = entropy_boundary_line  
        self.configured_start_scale_index = int(start_scale_index)
        self.adaptive_min_effective_start_scale_index = 9
        self.start_scale_index = (
            self.adaptive_min_effective_start_scale_index
            if self.use_adaptive_scale
            else self.configured_start_scale_index
        )
        self.direct_skip_start_scale_index = direct_skip_start_scale_index
        self.use_svd_depart_layers = use_svd_depart_layers 
        self.structure_layer_ration = structure_layer_ration 
        self.use_token_pruning = use_token_pruning 
        legacy_adaptive_s_min = None if adaptive_entropy_s_min is None else int(adaptive_entropy_s_min)
        if adaptive_entropy_collect_start_scale_index is None:
            adaptive_entropy_collect_start_scale_index = 7
        if adaptive_entropy_trigger_start_scale_index is None:
            adaptive_entropy_trigger_start_scale_index = 8
        if legacy_adaptive_s_min is not None:
            if (
                adaptive_entropy_collect_start_scale_index == 7
                and adaptive_entropy_trigger_start_scale_index == 8
            ):
                adaptive_entropy_trigger_start_scale_index = legacy_adaptive_s_min
                adaptive_entropy_collect_start_scale_index = max(0, legacy_adaptive_s_min - 1)
            elif legacy_adaptive_s_min != int(adaptive_entropy_trigger_start_scale_index):
                print(
                    '[Adaptive Pruning] Ignoring legacy adaptive_entropy_s_min='
                    f'{legacy_adaptive_s_min} because explicit collect/trigger indices were provided.'
                )

        self.adaptive_entropy_collect_start_scale_index = int(adaptive_entropy_collect_start_scale_index)
        self.adaptive_entropy_trigger_start_scale_index = int(adaptive_entropy_trigger_start_scale_index)
        self.adaptive_entropy_s_min = int(self.adaptive_entropy_trigger_start_scale_index)
        self.adaptive_entropy_gamma = float(adaptive_entropy_gamma)
        self.adaptive_entropy_epsilon = float(adaptive_entropy_epsilon)
        self.adaptive_entropy_lambda = float(adaptive_entropy_lambda)
        self.adaptive_entropy_score_mode = str(adaptive_entropy_score_mode or 'delta_h_width_power').strip().lower()
        self.adaptive_entropy_signal_mode = str(adaptive_entropy_signal_mode or 'layer_mean_delta').strip().lower()
        self.adaptive_entropy_signal_layer_idx = int(adaptive_entropy_signal_layer_idx)
        self.adaptive_max_raw_start_scale_index = int(adaptive_max_raw_start_scale_index)
        self.force_entropy_collection_scale_inds = parse_toprovar_scale_set(force_entropy_collection_scales)
        if self.adaptive_entropy_collect_start_scale_index < 0:
            raise ValueError(
                'adaptive_entropy_collect_start_scale_index must be >= 0, '
                f'got {self.adaptive_entropy_collect_start_scale_index}'
            )
        if self.adaptive_entropy_trigger_start_scale_index <= self.adaptive_entropy_collect_start_scale_index:
            raise ValueError(
                'adaptive_entropy_trigger_start_scale_index must be greater than '
                'adaptive_entropy_collect_start_scale_index, '
                f'got {self.adaptive_entropy_trigger_start_scale_index} <= '
                f'{self.adaptive_entropy_collect_start_scale_index}'
            )
        if self.adaptive_max_raw_start_scale_index <= self.adaptive_entropy_trigger_start_scale_index:
            raise ValueError(
                'adaptive_max_raw_start_scale_index must be greater than '
                'adaptive_entropy_trigger_start_scale_index, '
                f'got {adaptive_max_raw_start_scale_index} <= '
                f'{self.adaptive_entropy_trigger_start_scale_index}'
            )
        if not math.isfinite(self.adaptive_entropy_gamma):
            raise ValueError(f'adaptive_entropy_gamma must be finite, got {adaptive_entropy_gamma!r}')
        if not math.isfinite(self.adaptive_entropy_epsilon) or self.adaptive_entropy_epsilon <= 0.0:
            raise ValueError(f'adaptive_entropy_epsilon must be positive and finite, got {adaptive_entropy_epsilon!r}')
        if not math.isfinite(self.adaptive_entropy_lambda):
            raise ValueError(f'adaptive_entropy_lambda must be finite, got {adaptive_entropy_lambda!r}')
        if self.adaptive_entropy_score_mode not in {'delta_h_width_power', 'legacy_delta_cost_ratio'}:
            raise ValueError(
                f'Unsupported adaptive_entropy_score_mode={adaptive_entropy_score_mode!r}, '
                'expected delta_h_width_power or legacy_delta_cost_ratio'
            )
        if self.adaptive_entropy_signal_mode not in {'layer_mean_delta', 'global_mean_delta'}:
            raise ValueError(
                f'Unsupported adaptive_entropy_signal_mode={adaptive_entropy_signal_mode!r}, '
                'expected layer_mean_delta or global_mean_delta'
            )
        if self.adaptive_entropy_signal_layer_idx < 0:
            raise ValueError(
                'adaptive_entropy_signal_layer_idx must be >= 0, '
                f'got {self.adaptive_entropy_signal_layer_idx}'
            )
        self.scale_index_to_hw = {}
        self.low_entropy_prune_ratio = min(max(float(os.environ.get('TOPROVAR_LOW_ENTROPY_PRUNE_RATIO', '0.3')), 0.0), 0.999)
        self.high_entropy_prune_ratio = min(max(float(os.environ.get('TOPROVAR_HIGH_ENTROPY_PRUNE_RATIO', '0.9')), 0.0), 0.999)
        self.env_layer_ratio_schedule = parse_toprovar_layer_ratio_schedule(
            os.environ.get('TOPROVAR_LAYER_RATIO_SCHEDULE')
        )
        self.env_token_prune_ratio_schedule = parse_toprovar_token_ratio_schedule(
            os.environ.get('TOPROVAR_TOKEN_RATIO_SCHEDULE')
        )
        self.env_token_high_entropy_fraction_schedule = parse_toprovar_token_high_entropy_fraction_schedule(
            os.environ.get('TOPROVAR_TOKEN_HIGH_ENTROPY_FRACTION_SCHEDULE')
        )
        self.per_scale_detail_ratio_overrides = {
            9: 0.60,
            10: 0.60,
        }
        self.per_scale_token_prune_ratio_overrides = {
            7: {
                8: (0.0, 0.3),
                9: (0.3, 0.9),
                10: (0.3, 0.9),
            },
        }
        requested_source_scale_ind = int(svd_source_scale_index)
        if self.use_adaptive_scale and requested_source_scale_ind != 8:
            print(
                '[Adaptive Pruning] Normalizing svd_source_scale_index '
                f'from {requested_source_scale_ind} to 8 for adaptive SEGR.'
            )
            requested_source_scale_ind = 8
        self.svd_source_scale_ind = requested_source_scale_ind
        self.svd_source_scale_index = requested_source_scale_ind
        if self.use_svd_depart_layers and self.svd_source_scale_ind >= self.start_scale_index:
            raise ValueError(
                f'svd_source_scale_index={self.svd_source_scale_ind} must be smaller than '
                f'start_scale_index={self.start_scale_index}'
            )
        self.svd_rho_threshold = None if svd_rho_threshold is None or float(svd_rho_threshold) <= 0.0 else float(svd_rho_threshold)
        self.svd_split_mode = str(svd_split_mode or 'ratio').strip().lower()
        if self.svd_split_mode not in {'ratio', 'threshold', 'gap_raw', 'gap_log', 'gap_prior'}:
            raise ValueError(
                f'Unsupported svd_split_mode={svd_split_mode!r}, '
                'expected one of ratio, threshold, gap_raw, gap_log, gap_prior'
            )
        protected_layers = [] if protected_global_layers is None else protected_global_layers
        self.protected_global_layers = {int(layer_idx) for layer_idx in protected_layers}
        self.svd_auto_min_detail_layers = 10
        self.svd_auto_max_detail_layers = 18
        self.svd_auto_fallback_detail_ratio = 0.40
        self.svd_layer_rho = {}
        self.svd_layer_score = {}
        self.svd_ratio_sorted_layers = None
        self._svd_layer_values_materialized = True
        self.svd_layer_sets_by_scale = {}
        self.svd_fixed_layer_sets = None
        self.structure_layers = None
        self.semantic_layers = None
        self.svd_global_layers = None
        self.svd_detail_layers = None
        self.svd_pruning_layers = None
        self.svd_split_method = None
        self.svd_split_gap = None
        self.svd_effective_rho_threshold = None
        self.svd_gap_metric = None
        self.svd_gap_value = None
        self.svd_fallback_reason = None
        self.effective_layer_detail_ratio_by_scale = {}
        self.effective_token_prune_ratio_by_scale = {}
        self.effective_token_high_entropy_fraction_by_scale = {}
        self.enable_timing_stats = os.environ.get('TOPROVAR_ENABLE_TIMING_STATS', '0') != '0'
        self._timing_event_records = []
        self._timing_cpu_totals = {}
        self._token_index_cache = {}
        self.reset_adaptive_state()
        
        self.dict = {}

    def reset_adaptive_state(self) -> None:
        if self.use_adaptive_scale:
            self.start_scale_index = self.adaptive_min_effective_start_scale_index
            self.raw_start_scale_index = None
            self.effective_start_scale_index = None
        else:
            self.raw_start_scale_index = int(self.start_scale_index)
            self.effective_start_scale_index = int(self.start_scale_index)
        self.first_pruned_scale_index = None
        self.adaptive_h_mean_prev = None
        self.adaptive_signal_h_mean_prev = None
        self.adaptive_entropy_stats_by_scale = {}
        self.adaptive_entropy_layer_stats_by_scale = {}
        self.adaptive_segr_trace = []
        self._adaptive_finalized_scales = set()

    def set_scale_schedule(self, scale_schedule) -> None:
        scale_index_to_hw = {}
        if scale_schedule is not None:
            for scale_idx, item in enumerate(scale_schedule):
                if isinstance(item, (list, tuple)):
                    if len(item) == 3:
                        _, h, w = item
                    elif len(item) == 2:
                        h, w = item
                    elif len(item) == 1:
                        h = w = item[0]
                    else:
                        raise ValueError(
                            f'Unsupported scale_schedule item at index {scale_idx}: {item!r}'
                        )
                else:
                    h = w = item
                scale_index_to_hw[int(scale_idx)] = (int(h), int(w))
        self.scale_index_to_hw = scale_index_to_hw

    def _adaptive_monitor_start_scale_ind(self) -> int:
        return int(self.adaptive_entropy_collect_start_scale_index)

    def _adaptive_trigger_start_scale_ind(self) -> int:
        return int(self.adaptive_entropy_trigger_start_scale_index)

    def _adaptive_monitor_end_scale_ind(self) -> int:
        return int(self.adaptive_max_raw_start_scale_index) - 1

    def _adaptive_trigger_end_scale_ind(self) -> int:
        return int(self.adaptive_max_raw_start_scale_index) - 1

    def _is_adaptive_monitor_scale(self, scale_ind: int) -> bool:
        scale_ind = int(scale_ind)
        return (
            self.use_adaptive_scale
            and self.raw_start_scale_index is None
            and self._adaptive_monitor_start_scale_ind() <= scale_ind <= self._adaptive_monitor_end_scale_ind()
        )

    def _is_adaptive_trigger_scale(self, scale_ind: int) -> bool:
        scale_ind = int(scale_ind)
        return (
            self.use_adaptive_scale
            and self.raw_start_scale_index is None
            and self._adaptive_trigger_start_scale_ind() <= scale_ind <= self._adaptive_trigger_end_scale_ind()
        )

    def _resolve_cost_n(self, cost_scale_index: int):
        hw = self.scale_index_to_hw.get(int(cost_scale_index))
        if hw is None:
            return None
        h, w = hw
        if h <= 0 or w <= 0:
            return None
        if h == w:
            return float(h)
        return math.sqrt(float(h) * float(w))

    def _set_adaptive_start(
        self,
        raw_start_scale_index: int,
        *,
        trigger_scale_ind=None,
        delta_h=None,
        cost_scale_index=None,
        cost_n=None,
        cost_value=None,
        delta_cost_ratio=None,
        adaptive_score=None,
        adaptive_score_n=None,
        reason='threshold',
    ) -> None:
        if not self.use_adaptive_scale or self.raw_start_scale_index is not None:
            return
        raw_start_scale_index = int(raw_start_scale_index)
        raw_start_scale_index = min(raw_start_scale_index, int(self.adaptive_max_raw_start_scale_index))
        effective_start_scale_index = max(raw_start_scale_index, int(self.adaptive_min_effective_start_scale_index))
        if self.use_svd_depart_layers and int(self.svd_source_scale_ind) >= effective_start_scale_index:
            raise ValueError(
                f'svd_source_scale_index={self.svd_source_scale_ind} must be smaller than '
                f'effective_start_scale_index={effective_start_scale_index}'
            )
        self.raw_start_scale_index = raw_start_scale_index
        self.effective_start_scale_index = effective_start_scale_index
        self.start_scale_index = effective_start_scale_index
        delta_h_text = 'n/a' if delta_h is None else f'{float(delta_h):.6e}'
        cost_n_text = 'n/a' if cost_n is None else f'{float(cost_n):.6f}'
        cost_value_text = 'n/a' if cost_value is None else f'{float(cost_value):.6e}'
        delta_cost_ratio_text = 'n/a' if delta_cost_ratio is None else f'{float(delta_cost_ratio):.6e}'
        adaptive_score_text = 'n/a' if adaptive_score is None else f'{float(adaptive_score):.6e}'
        adaptive_score_n_text = 'n/a' if adaptive_score_n is None else f'{float(adaptive_score_n):.6f}'
        print(
            '[Adaptive Pruning] Triggered raw_start='
            f'{raw_start_scale_index}, effective_start={effective_start_scale_index}, '
            f'trigger_scale={trigger_scale_ind}, delta_h={delta_h_text}, '
            f'cost_scale_index={cost_scale_index}, cost_n={cost_n_text}, '
            f'cost_value={cost_value_text}, delta_cost_ratio={delta_cost_ratio_text}, '
            f'adaptive_score_n={adaptive_score_n_text}, adaptive_score={adaptive_score_text}, '
            f'adaptive_entropy_lambda={float(self.adaptive_entropy_lambda):.6f}, '
            f'adaptive_entropy_score_mode={self.adaptive_entropy_score_mode}, reason={reason}'
        )

    def _record_first_pruned_scale(self, scale_ind: int) -> None:
        if self.first_pruned_scale_index is None:
            self.first_pruned_scale_index = int(scale_ind)

    def should_collect_attention_entropy(self, scale_ind: int) -> bool:
        scale_ind = int(scale_ind)
        if scale_ind in self.force_entropy_collection_scale_inds:
            return True
        if scale_ind == int(self.svd_source_scale_ind):
            return self.use_svd_depart_layers or self.use_token_pruning or self.use_adaptive_scale
        return self._is_adaptive_monitor_scale(scale_ind)

    def should_record_adaptive_entropy(self, scale_ind: int) -> bool:
        return self._is_adaptive_monitor_scale(scale_ind)

    def record_adaptive_entropy(self, scale_ind: int, layer_idx: int, attn_entropy) -> None:
        if not self.should_record_adaptive_entropy(scale_ind=scale_ind):
            return
        if attn_entropy is None:
            return
        entropy = attn_entropy.detach()
        token_count = int(entropy.numel())
        if token_count <= 0:
            return
        entropy_sum_tensor = entropy.float().sum()
        bucket = self.adaptive_entropy_stats_by_scale.setdefault(
            int(scale_ind),
            {
                'entropy_sum_tensor': None,
                'token_count': 0,
                'layer_count': 0,
            },
        )
        if bucket['entropy_sum_tensor'] is None:
            bucket['entropy_sum_tensor'] = entropy_sum_tensor
        else:
            bucket['entropy_sum_tensor'].add_(entropy_sum_tensor)
        bucket['token_count'] += token_count
        bucket['layer_count'] += 1

        layer_buckets = self.adaptive_entropy_layer_stats_by_scale.setdefault(int(scale_ind), {})
        layer_bucket = layer_buckets.setdefault(
            int(layer_idx),
            {
                'entropy_sum_tensor': None,
                'token_count': 0,
                'layer_count': 0,
            },
        )
        layer_entropy_sum_tensor = entropy.float().sum()
        if layer_bucket['entropy_sum_tensor'] is None:
            layer_bucket['entropy_sum_tensor'] = layer_entropy_sum_tensor
        else:
            layer_bucket['entropy_sum_tensor'].add_(layer_entropy_sum_tensor)
        layer_bucket['token_count'] += token_count
        layer_bucket['layer_count'] += 1

    def _entropy_bucket_mean(self, stats):
        if stats is None or int(stats.get('token_count', 0)) <= 0:
            return None
        entropy_sum_tensor = stats.get('entropy_sum_tensor')
        if entropy_sum_tensor is not None:
            entropy_sum = float(entropy_sum_tensor.item())
        else:
            entropy_sum = float(stats.get('entropy_sum', 0.0))
        return entropy_sum / float(stats['token_count'])

    def _select_adaptive_signal_stats(self, scale_ind: int):
        if self.adaptive_entropy_signal_mode == 'layer_mean_delta':
            layer_stats = self.adaptive_entropy_layer_stats_by_scale.get(int(scale_ind), {})
            stats = layer_stats.get(int(self.adaptive_entropy_signal_layer_idx))
            return stats, bool(stats is None)
        stats = self.adaptive_entropy_stats_by_scale.get(int(scale_ind))
        return stats, bool(stats is None)

    def finalize_adaptive_scale(self, scale_ind: int) -> None:
        if not self.use_adaptive_scale:
            return
        scale_ind = int(scale_ind)
        if scale_ind in self._adaptive_finalized_scales:
            return
        self._adaptive_finalized_scales.add(scale_ind)

        stats = self.adaptive_entropy_stats_by_scale.get(scale_ind)
        h_mean = None
        signal_stats, signal_missing = self._select_adaptive_signal_stats(scale_ind)
        signal_expected = self._is_adaptive_monitor_scale(scale_ind)
        signal_missing = bool(signal_expected and signal_missing)
        signal_h_mean = None
        signal_delta_h_signed = None
        signal_delta_h_for_score = None
        delta_h = None
        cost_scale_index = None
        cost_n = None
        cost_value = None
        delta_cost_ratio = None
        adaptive_score_n = None
        adaptive_score = None
        delta_rel = None
        trigger_eligible = self._is_adaptive_trigger_scale(scale_ind)
        candidate_raw_start = scale_ind + 1 if trigger_eligible else None
        triggered = False
        prev_h_mean = self.adaptive_h_mean_prev
        prev_signal_h_mean = self.adaptive_signal_h_mean_prev

        h_mean = self._entropy_bucket_mean(stats)
        signal_h_mean = self._entropy_bucket_mean(signal_stats)
        if h_mean is not None:
            if prev_h_mean is not None:
                delta_rel = (
                    h_mean - float(prev_h_mean)
                ) / (float(prev_h_mean) + float(self.adaptive_entropy_epsilon))
            self.adaptive_h_mean_prev = h_mean

        if signal_h_mean is not None:
            if prev_signal_h_mean is not None:
                signal_delta_h_signed = signal_h_mean - float(prev_signal_h_mean)
                if self.adaptive_entropy_signal_mode == 'layer_mean_delta':
                    signal_delta_h_for_score = max(float(signal_delta_h_signed), 0.0)
                else:
                    signal_delta_h_for_score = abs(float(signal_delta_h_signed))
                delta_h = float(signal_delta_h_for_score)
                cost_scale_index = scale_ind + 1
                cost_n = self._resolve_cost_n(cost_scale_index)
                if cost_n is not None:
                    cost_value = float(cost_n) ** 4
                    if cost_value > 0.0:
                        delta_cost_ratio = float(delta_h) / float(cost_value)
                    adaptive_score_n = float(cost_n)
                    if self.adaptive_entropy_score_mode == 'delta_h_width_power':
                        adaptive_score = float(delta_h) * (adaptive_score_n ** float(self.adaptive_entropy_lambda))
                    elif self.adaptive_entropy_score_mode == 'legacy_delta_cost_ratio':
                        adaptive_score = delta_cost_ratio
            if self.adaptive_entropy_score_mode == 'legacy_delta_cost_ratio':
                threshold_passed = (
                    adaptive_score is not None
                    and adaptive_score < float(self.adaptive_entropy_gamma)
                )
                threshold_reason = 'legacy_delta_cost_ratio_threshold'
            else:
                threshold_passed = (
                    adaptive_score is not None
                    and adaptive_score > float(self.adaptive_entropy_gamma)
                )
                threshold_reason = 'delta_h_width_power_threshold'
            if (
                trigger_eligible
                and self.raw_start_scale_index is None
                and threshold_passed
            ):
                self._set_adaptive_start(
                    candidate_raw_start,
                    trigger_scale_ind=scale_ind,
                    delta_h=delta_h,
                    cost_scale_index=cost_scale_index,
                    cost_n=cost_n,
                    cost_value=cost_value,
                    delta_cost_ratio=delta_cost_ratio,
                    adaptive_score=adaptive_score,
                    adaptive_score_n=adaptive_score_n,
                    reason=threshold_reason,
                )
                triggered = True
            self.adaptive_signal_h_mean_prev = signal_h_mean

        if (
            self.raw_start_scale_index is None
            and scale_ind >= self._adaptive_trigger_end_scale_ind()
        ):
            self._set_adaptive_start(
                self.adaptive_max_raw_start_scale_index,
                trigger_scale_ind=scale_ind,
                delta_h=delta_h,
                cost_scale_index=cost_scale_index,
                cost_n=cost_n,
                cost_value=cost_value,
                delta_cost_ratio=delta_cost_ratio,
                adaptive_score=adaptive_score,
                adaptive_score_n=adaptive_score_n,
                reason='fallback_max_raw_start',
            )
            triggered = True

        self.adaptive_segr_trace.append(
            {
                'scale_idx': scale_ind,
                'h_mean': h_mean,
                'signal_h_mean': signal_h_mean,
                'signal_delta_h_signed': signal_delta_h_signed,
                'signal_delta_h_for_score': signal_delta_h_for_score,
                'signal_expected': bool(signal_expected),
                'signal_missing': bool(signal_missing),
                'delta_h': delta_h,
                'cost_scale_index': cost_scale_index,
                'cost_n': cost_n,
                'cost_value': cost_value,
                'delta_cost_ratio': delta_cost_ratio,
                'adaptive_score_n': adaptive_score_n,
                'adaptive_score': adaptive_score,
                'adaptive_entropy_lambda': float(self.adaptive_entropy_lambda),
                'adaptive_entropy_score_mode': self.adaptive_entropy_score_mode,
                'adaptive_entropy_signal_mode': self.adaptive_entropy_signal_mode,
                'adaptive_entropy_signal_layer_idx': int(self.adaptive_entropy_signal_layer_idx),
                'trigger_eligible': bool(trigger_eligible),
                'delta_rel': delta_rel,
                'candidate_raw_start_scale_index': candidate_raw_start,
                'triggered': triggered,
                'raw_start_scale_index': self.raw_start_scale_index,
                'effective_start_scale_index': self.effective_start_scale_index,
                'entropy_token_count': None if stats is None else int(stats.get('token_count', 0)),
                'entropy_layer_count': None if stats is None else int(stats.get('layer_count', 0)),
                'signal_entropy_token_count': None if signal_stats is None else int(signal_stats.get('token_count', 0)),
                'signal_entropy_layer_count': None if signal_stats is None else int(signal_stats.get('layer_count', 0)),
            }
        )

    def get_entropy_extract_scale_inds(self):
        scales = {int(self.svd_source_scale_ind)}
        if self.use_adaptive_scale:
            scales.update(range(self._adaptive_monitor_start_scale_ind(), self._adaptive_monitor_end_scale_ind() + 1))
        return sorted(scales)

    def get_adaptive_segr_report(self):
        return {
            'enabled': bool(self.use_adaptive_scale),
            'source_scale_index': int(self.svd_source_scale_ind),
            'collect_start_scale_index': int(self.adaptive_entropy_collect_start_scale_index),
            'trigger_start_scale_index': int(self.adaptive_entropy_trigger_start_scale_index),
            'trigger_end_scale_index': int(self._adaptive_trigger_end_scale_ind()),
            's_min': int(self.adaptive_entropy_s_min),
            'gamma': float(self.adaptive_entropy_gamma),
            'epsilon': float(self.adaptive_entropy_epsilon),
            'lambda': float(self.adaptive_entropy_lambda),
            'score_mode': self.adaptive_entropy_score_mode,
            'adaptive_entropy_signal_mode': self.adaptive_entropy_signal_mode,
            'adaptive_entropy_signal_layer_idx': int(self.adaptive_entropy_signal_layer_idx),
            'signal_mode': self.adaptive_entropy_signal_mode,
            'signal_layer_idx': int(self.adaptive_entropy_signal_layer_idx),
            'max_raw_start_scale_index': int(self.adaptive_max_raw_start_scale_index),
            'min_effective_start_scale_index': int(self.adaptive_min_effective_start_scale_index),
            'raw_start_scale_index': None if self.raw_start_scale_index is None else int(self.raw_start_scale_index),
            'effective_start_scale_index': None if self.effective_start_scale_index is None else int(self.effective_start_scale_index),
            'first_pruned_scale_index': None if self.first_pruned_scale_index is None else int(self.first_pruned_scale_index),
            'scale_index_to_hw': {
                str(int(scale_idx)): [int(hw[0]), int(hw[1])]
                for scale_idx, hw in sorted(self.scale_index_to_hw.items())
            },
            'trace': list(self.adaptive_segr_trace),
        }

    def set_fixed_start_scale_index(self, start_scale_index: int) -> None:
        if self.use_adaptive_scale:
            raise ValueError('set_fixed_start_scale_index is only valid when use_adaptive_scale=False')
        start_scale_index = int(start_scale_index)
        if self.use_svd_depart_layers and int(self.svd_source_scale_ind) >= start_scale_index:
            raise ValueError(
                f'svd_source_scale_index={self.svd_source_scale_ind} must be smaller than '
                f'start_scale_index={start_scale_index}'
            )
        self.configured_start_scale_index = start_scale_index
        self.start_scale_index = start_scale_index
        self.raw_start_scale_index = start_scale_index
        self.effective_start_scale_index = start_scale_index
    
    def _sacle_start(self, scale_ind, attn_entropy=None):
        if self.use_adaptive_scale:
            if self.effective_start_scale_index is None:
                if int(scale_ind) >= int(self.adaptive_max_raw_start_scale_index):
                    self._set_adaptive_start(
                        self.adaptive_max_raw_start_scale_index,
                        trigger_scale_ind=int(scale_ind) - 1,
                        reason='late_fallback_max_raw_start',
                    )
                else:
                    return False
            return scale_ind >= int(self.effective_start_scale_index)
        else:
            return scale_ind >= self.start_scale_index
    
    def _layer_start(self, layer_idx, scale_ind=None, attn_entropy=None):
        if self.use_svd_depart_layers:
            layer_sets = self.get_svd_layer_sets_for_scale(scale_ind=scale_ind)
            if layer_sets is None:
                return False
            return layer_idx in layer_sets['pruning_layers']
        return layer_idx >= 2 and layer_idx <= 18

    def should_record_svd_layer(self, scale_ind: int, layer_idx: int) -> bool:
        return (
            self.use_svd_depart_layers
            and scale_ind == self.svd_source_scale_ind
            and layer_idx not in self.svd_layer_rho
        )

    def get_entropy_extract_scale_ind(self) -> int:
        if self.use_svd_depart_layers:
            return self.svd_source_scale_ind
        return 8

    def should_direct_skip_scale(self, scale_ind: int) -> bool:
        return (
            self.direct_skip_start_scale_index is not None
            and self.direct_skip_start_scale_index != 100
            and scale_ind >= self.direct_skip_start_scale_index
        )

    def _start_timing(self, name: str):
        if not self.enable_timing_stats:
            return None
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            return ('cuda', name, start_event, end_event)
        return ('cpu', name, time.perf_counter(), None)

    def _stop_timing(self, handle) -> None:
        if handle is None:
            return

        mode, name, start_token, end_token = handle
        if mode == 'cuda':
            end_token.record()
            self._timing_event_records.append((name, start_token, end_token))
            return

        duration_ms = (time.perf_counter() - start_token) * 1000.0
        if name not in self._timing_cpu_totals:
            self._timing_cpu_totals[name] = {'total_ms': 0.0, 'count': 0}
        self._timing_cpu_totals[name]['total_ms'] += duration_ms
        self._timing_cpu_totals[name]['count'] += 1

    @contextmanager
    def timing_scope(self, name: str):
        handle = self._start_timing(name)
        try:
            yield
        finally:
            self._stop_timing(handle)

    def get_timing_stats(self):
        totals = {}

        if self._timing_event_records:
            torch.cuda.synchronize()
            for name, start_event, end_event in self._timing_event_records:
                if name not in totals:
                    totals[name] = {'total_ms': 0.0, 'count': 0}
                totals[name]['total_ms'] += start_event.elapsed_time(end_event)
                totals[name]['count'] += 1

        for name, bucket in self._timing_cpu_totals.items():
            if name not in totals:
                totals[name] = {'total_ms': 0.0, 'count': 0}
            totals[name]['total_ms'] += bucket['total_ms']
            totals[name]['count'] += bucket['count']

        if not totals:
            return []

        tracked_total_ms = sum(item['total_ms'] for item in totals.values())
        stats = []
        for name, item in totals.items():
            count = item['count']
            total_ms = item['total_ms']
            avg_ms = total_ms / count if count > 0 else 0.0
            pct = (total_ms / tracked_total_ms * 100.0) if tracked_total_ms > 0 else 0.0
            stats.append((name, total_ms, avg_ms, count, pct))

        stats.sort(key=lambda item: item[1], reverse=True)
        return stats

    def print_timing_stats(self) -> None:
        stats = self.get_timing_stats()
        if not stats:
            return

        tracked_total_ms = sum(item[1] for item in stats)
        print(f'[ToProVAR timing] tracked_total_ms={tracked_total_ms:.3f}')
        for name, total_ms, avg_ms, count, pct in stats:
            print(f'  {name}: total_ms={total_ms:.3f}, avg_ms={avg_ms:.3f}, count={count}, pct={pct:.2f}')

    def record_svd_layer(self, scale_ind: int, layer_idx: int, rho) -> None:
        if not self.should_record_svd_layer(scale_ind=scale_ind, layer_idx=layer_idx):
            return
        if rho is None:
            return
        if isinstance(rho, torch.Tensor):
            if rho.numel() != 1:
                return
            self.svd_layer_rho[layer_idx] = rho.detach()
            self._svd_layer_values_materialized = False
        else:
            rho = float(rho)
            if not math.isfinite(rho):
                return
            self.svd_layer_rho[layer_idx] = rho
            self.svd_layer_score[layer_idx] = math.exp(-rho)
        self.svd_ratio_sorted_layers = None
        self.svd_layer_sets_by_scale = {}
        self.svd_fixed_layer_sets = None

    def _materialize_svd_layer_values(self) -> None:
        if self._svd_layer_values_materialized:
            return

        tensor_items_by_device = {}
        materialized = {}
        for layer_idx, rho in self.svd_layer_rho.items():
            if isinstance(rho, torch.Tensor):
                if rho.numel() == 1:
                    tensor = rho.detach().reshape(())
                    tensor_items_by_device.setdefault(tensor.device, []).append((int(layer_idx), tensor))
                continue
            rho = float(rho)
            if math.isfinite(rho):
                materialized[int(layer_idx)] = rho

        for tensor_items in tensor_items_by_device.values():
            layer_indices, tensors = zip(*tensor_items)
            values = torch.stack([tensor.to(dtype=torch.float32) for tensor in tensors]).cpu().tolist()
            for layer_idx, rho in zip(layer_indices, values):
                rho = float(rho)
                if math.isfinite(rho):
                    materialized[layer_idx] = rho

        self.svd_layer_rho = materialized
        self.svd_layer_score = {
            layer_idx: math.exp(-rho)
            for layer_idx, rho in materialized.items()
        }
        self.svd_ratio_sorted_layers = None
        self._svd_layer_values_materialized = True

    def _normalize_prune_ratio(self, ratio: float) -> float:
        return _normalize_toprovar_ratio(ratio)

    def get_effective_layer_detail_ratio(self, scale_ind: int) -> float:
        scale_ind = int(scale_ind)
        if scale_ind in self.effective_layer_detail_ratio_by_scale:
            return float(self.effective_layer_detail_ratio_by_scale[scale_ind])
        if scale_ind in self.env_layer_ratio_schedule:
            ratio = self.env_layer_ratio_schedule[scale_ind]
        else:
            ratio = self.per_scale_detail_ratio_overrides.get(scale_ind, self.structure_layer_ration)
        ratio = self._normalize_prune_ratio(ratio)
        self.effective_layer_detail_ratio_by_scale[scale_ind] = ratio
        return ratio

    def get_effective_token_prune_ratios(self, scale_ind: int):
        scale_ind = int(scale_ind)
        scale_overrides = self.per_scale_token_prune_ratio_overrides.get(int(self.svd_source_scale_ind), {})
        if scale_ind in self.env_token_prune_ratio_schedule:
            low_ratio, high_ratio = self.env_token_prune_ratio_schedule[scale_ind]
        else:
            low_ratio, high_ratio = scale_overrides.get(
                scale_ind,
                (self.low_entropy_prune_ratio, self.high_entropy_prune_ratio),
            )
        ratios = (
            self._normalize_prune_ratio(low_ratio),
            self._normalize_prune_ratio(high_ratio),
        )
        self.effective_token_prune_ratio_by_scale[scale_ind] = ratios
        return ratios

    def get_effective_token_high_entropy_fraction(self, scale_ind: int):
        scale_ind = int(scale_ind)
        if scale_ind in self.effective_token_high_entropy_fraction_by_scale:
            return float(self.effective_token_high_entropy_fraction_by_scale[scale_ind])
        if scale_ind not in self.env_token_high_entropy_fraction_schedule:
            return None
        high_fraction = _normalize_toprovar_fraction(
            self.env_token_high_entropy_fraction_schedule[scale_ind]
        )
        self.effective_token_high_entropy_fraction_by_scale[scale_ind] = high_fraction
        return high_fraction

    def get_effective_layer_detail_ratio_by_scale_report(self):
        report = {}
        if self.env_layer_ratio_schedule:
            report.update(self.env_layer_ratio_schedule)
        report.update(self.effective_layer_detail_ratio_by_scale)
        return {
            str(int(scale_ind)): float(ratio)
            for scale_ind, ratio in sorted(report.items())
        }

    def get_effective_token_prune_ratio_by_scale_report(self):
        report = {}
        if self.env_token_prune_ratio_schedule:
            report.update(self.env_token_prune_ratio_schedule)
        else:
            for scale_ind, ratios in self.per_scale_token_prune_ratio_overrides.get(int(self.svd_source_scale_ind), {}).items():
                low_ratio, high_ratio = ratios
                report[int(scale_ind)] = (
                    self._normalize_prune_ratio(low_ratio),
                    self._normalize_prune_ratio(high_ratio),
                )
        for scale_ind, ratios in self.effective_token_prune_ratio_by_scale.items():
            report[int(scale_ind)] = ratios
        return {
            str(int(scale_ind)): {
                'low': float(ratios[0]),
                'high': float(ratios[1]),
            }
            for scale_ind, ratios in sorted(report.items())
        }

    def get_effective_token_high_entropy_fraction_by_scale_report(self):
        return {
            str(int(scale_ind)): float(fraction)
            for scale_ind, fraction in sorted(self.effective_token_high_entropy_fraction_by_scale.items())
        }

    def get_svd_pruning_layers_by_scale_report(self):
        return {
            str(int(scale_ind)): sorted(int(x) for x in layer_sets['pruning_layers'])
            for scale_ind, layer_sets in sorted(self.svd_layer_sets_by_scale.items())
        }

    def _build_svd_layer_sets(self, global_layers, detail_layers):
        if not global_layers or not detail_layers:
            return None
        global_layers = {int(layer_idx) for layer_idx in global_layers}
        detail_layers = {int(layer_idx) for layer_idx in detail_layers}
        protected_layers = {
            int(layer_idx)
            for layer_idx in self.protected_global_layers
            if int(layer_idx) in detail_layers
        }
        if protected_layers:
            detail_layers -= protected_layers
            global_layers |= protected_layers
        pruning_layers = set(detail_layers)
        if not pruning_layers:
            return None
        return {
            'global_layers': global_layers,
            'detail_layers': detail_layers,
            'pruning_layers': pruning_layers,
        }

    def _assign_legacy_svd_snapshot(self, layer_sets) -> None:
        self.svd_global_layers = set(layer_sets['global_layers'])
        self.svd_detail_layers = set(layer_sets['detail_layers'])
        self.svd_pruning_layers = set(layer_sets['pruning_layers'])
        self.structure_layers = set(self.svd_pruning_layers)
        self.semantic_layers = set(self.svd_global_layers)

    def _cache_svd_layer_sets_for_scale(self, scale_ind: int, layer_sets, detail_ratio=None):
        cached_layer_sets = {
            'global_layers': set(layer_sets['global_layers']),
            'detail_layers': set(layer_sets['detail_layers']),
            'pruning_layers': set(layer_sets['pruning_layers']),
        }
        self.svd_layer_sets_by_scale[int(scale_ind)] = cached_layer_sets
        if detail_ratio is not None:
            self.effective_layer_detail_ratio_by_scale[int(scale_ind)] = float(detail_ratio)
        if int(scale_ind) == int(self.start_scale_index):
            self._assign_legacy_svd_snapshot(cached_layer_sets)
        return cached_layer_sets

    def _get_ratio_sorted_layers(self):
        self._materialize_svd_layer_values()
        if self.svd_ratio_sorted_layers is None:
            self.svd_ratio_sorted_layers = sorted(self.svd_layer_score.items(), key=lambda item: item[1])
        return self.svd_ratio_sorted_layers

    def _build_ratio_svd_layer_sets_for_scale(self, scale_ind: int):
        sorted_layers = self._get_ratio_sorted_layers()
        if not sorted_layers:
            return None, None

        num_layers = len(sorted_layers)
        detail_ratio = self.get_effective_layer_detail_ratio(scale_ind=scale_ind)
        num_detail_layers = int(round(num_layers * detail_ratio))
        if num_layers > 1:
            num_detail_layers = min(max(num_detail_layers, 1), num_layers - 1)
        num_global_layers = num_layers - num_detail_layers
        global_layers = {layer_idx for layer_idx, _ in sorted_layers[:num_global_layers]}
        detail_layers = {layer_idx for layer_idx, _ in sorted_layers[num_global_layers:]}
        return self._build_svd_layer_sets(global_layers=global_layers, detail_layers=detail_layers), detail_ratio

    def _materialize_ratio_svd_layer_sets_for_scale(self, scale_ind: int):
        scale_ind = int(scale_ind)
        if scale_ind in self.svd_layer_sets_by_scale:
            return self.svd_layer_sets_by_scale[scale_ind]
        layer_sets, detail_ratio = self._build_ratio_svd_layer_sets_for_scale(scale_ind=scale_ind)
        if layer_sets is None:
            return None
        self.svd_split_method = 'ratio_exp_rho'
        self.svd_split_gap = None
        self.svd_fallback_reason = None
        return self._cache_svd_layer_sets_for_scale(scale_ind=scale_ind, layer_sets=layer_sets, detail_ratio=detail_ratio)

    def get_svd_layer_sets_for_scale(self, scale_ind: int):
        if not self.use_svd_depart_layers or scale_ind is None:
            return None
        scale_ind = int(scale_ind)
        if scale_ind < self.start_scale_index:
            return None
        if scale_ind <= self.svd_source_scale_ind:
            return None
        if self.svd_split_mode == 'ratio':
            return self._materialize_ratio_svd_layer_sets_for_scale(scale_ind=scale_ind)
        if self.svd_fixed_layer_sets is None:
            self._maybe_finalize_svd_layers(scale_ind=scale_ind)
        if self.svd_fixed_layer_sets is None:
            return None
        if scale_ind not in self.svd_layer_sets_by_scale:
            self._cache_svd_layer_sets_for_scale(scale_ind=scale_ind, layer_sets=self.svd_fixed_layer_sets)
        return self.svd_layer_sets_by_scale[scale_ind]

    def get_svd_snapshot_for_scale(self, scale_ind: int):
        layer_sets = self.get_svd_layer_sets_for_scale(scale_ind=scale_ind)
        if layer_sets is None:
            return {
                'global_layers': set(),
                'detail_layers': set(),
                'pruning_layers': set(),
            }
        return {
            'global_layers': set(layer_sets['global_layers']),
            'detail_layers': set(layer_sets['detail_layers']),
            'pruning_layers': set(layer_sets['pruning_layers']),
        }

    def _finalize_svd_layers_from_sets(self, global_layers, detail_layers, split_method: str, split_gap=None) -> bool:
        layer_sets = self._build_svd_layer_sets(global_layers=global_layers, detail_layers=detail_layers)
        if layer_sets is None:
            return False

        self.svd_fixed_layer_sets = {
            'global_layers': set(layer_sets['global_layers']),
            'detail_layers': set(layer_sets['detail_layers']),
            'pruning_layers': set(layer_sets['pruning_layers']),
        }
        self._assign_legacy_svd_snapshot(self.svd_fixed_layer_sets)
        self.svd_split_method = split_method
        self.svd_split_gap = split_gap
        return True

    def _finalize_svd_layers_by_ratio(self, detail_ratio=None) -> bool:
        scale_ind = self.start_scale_index
        if detail_ratio is not None:
            self.effective_layer_detail_ratio_by_scale[int(scale_ind)] = self._normalize_prune_ratio(detail_ratio)
        layer_sets, resolved_detail_ratio = self._build_ratio_svd_layer_sets_for_scale(scale_ind=scale_ind)
        if layer_sets is None:
            return False
        self.svd_fixed_layer_sets = {
            'global_layers': set(layer_sets['global_layers']),
            'detail_layers': set(layer_sets['detail_layers']),
            'pruning_layers': set(layer_sets['pruning_layers']),
        }
        self._cache_svd_layer_sets_for_scale(
            scale_ind=scale_ind,
            layer_sets=self.svd_fixed_layer_sets,
            detail_ratio=resolved_detail_ratio,
        )
        self.svd_split_method = 'ratio_exp_rho'
        self.svd_split_gap = None
        return True

    def _get_gap_detail_layer_bounds(self, num_layers: int):
        if num_layers <= 1:
            return None

        upper_bound = num_layers - 1
        min_detail = min(max(int(self.svd_auto_min_detail_layers), 1), upper_bound)
        max_detail = min(max(int(self.svd_auto_max_detail_layers), min_detail), upper_bound)
        if min_detail > max_detail:
            return None
        return min_detail, max_detail

    def _finalize_svd_layers_by_gap(self, *, use_log_gap: bool, use_position_prior: bool = False) -> bool:
        self._materialize_svd_layer_values()
        sorted_layers = sorted(self.svd_layer_rho.items(), key=lambda item: item[1])
        if len(sorted_layers) <= 1:
            return False

        detail_bounds = self._get_gap_detail_layer_bounds(len(sorted_layers))
        if detail_bounds is None:
            return False

        transformed_rhos = []
        for _, rho in sorted_layers:
            if not math.isfinite(rho):
                return False
            if use_log_gap:
                if rho <= 0.0:
                    return False
                transformed_rhos.append(math.log(rho))
            else:
                transformed_rhos.append(float(rho))

        min_detail_layers, max_detail_layers = detail_bounds
        preferred_detail_layers = int(round(len(sorted_layers) * float(self.svd_auto_fallback_detail_ratio)))
        preferred_detail_layers = min(max(preferred_detail_layers, min_detail_layers), max_detail_layers)
        best_candidate = None
        sigma = max(1.5, (max_detail_layers - min_detail_layers) / 3.0)
        for idx in range(len(sorted_layers) - 1):
            detail_count = idx + 1
            if detail_count < min_detail_layers or detail_count > max_detail_layers:
                continue

            gap_value = transformed_rhos[idx + 1] - transformed_rhos[idx]
            if not math.isfinite(gap_value) or gap_value <= 0.0:
                continue

            if use_position_prior:
                prev_gap = gap_value if idx == 0 else (transformed_rhos[idx] - transformed_rhos[idx - 1])
                next_gap = gap_value if idx + 2 >= len(transformed_rhos) else (transformed_rhos[idx + 2] - transformed_rhos[idx + 1])
                if not math.isfinite(prev_gap) or prev_gap <= 0.0 or not math.isfinite(next_gap) or next_gap <= 0.0:
                    continue
                local_context = max((prev_gap + next_gap) * 0.5, 1e-6)
                valley_score = gap_value / local_context
                position_prior = math.exp(-0.5 * ((detail_count - preferred_detail_layers) / sigma) ** 2)
                split_score = valley_score * position_prior
                if not math.isfinite(split_score) or split_score <= 0.0:
                    continue
                sort_key = (split_score, gap_value, -abs(detail_count - preferred_detail_layers))
            else:
                sort_key = (gap_value, -abs(detail_count - preferred_detail_layers))
            threshold = (
                math.exp((transformed_rhos[idx] + transformed_rhos[idx + 1]) * 0.5)
                if use_log_gap else
                (float(sorted_layers[idx][1]) + float(sorted_layers[idx + 1][1])) * 0.5
            )
            candidate = (sort_key, detail_count, gap_value, threshold)
            if best_candidate is None or candidate[0] > best_candidate[0]:
                best_candidate = candidate

        if best_candidate is None:
            return False

        _, detail_count, gap_value, threshold = best_candidate
        detail_layers = {int(layer_idx) for layer_idx, _ in sorted_layers[:detail_count]}
        global_layers = {int(layer_idx) for layer_idx, _ in sorted_layers[detail_count:]}
        split_method = 'gap_prior_rho' if use_position_prior else ('gap_log_rho' if use_log_gap else 'gap_raw_rho')
        gap_metric = 'rho_valley_prior' if use_position_prior else ('log_rho' if use_log_gap else 'rho')
        gap_score = best_candidate[0][0] if use_position_prior else gap_value
        if not self._finalize_svd_layers_from_sets(
            global_layers=global_layers,
            detail_layers=detail_layers,
            split_method=split_method,
            split_gap=gap_score,
        ):
            return False

        self.svd_effective_rho_threshold = float(threshold)
        self.svd_gap_metric = gap_metric
        self.svd_gap_value = float(gap_score)
        return True

    def _finalize_svd_layers_by_threshold(self) -> bool:
        if self.svd_rho_threshold is None:
            return False
        self._materialize_svd_layer_values()

        global_layers = {layer_idx for layer_idx, rho in self.svd_layer_rho.items() if rho >= self.svd_rho_threshold}
        detail_layers = {layer_idx for layer_idx, rho in self.svd_layer_rho.items() if rho < self.svd_rho_threshold}
        if not global_layers or not detail_layers:
            return False

        split_gap = min(abs(rho - self.svd_rho_threshold) for rho in self.svd_layer_rho.values())
        if not self._finalize_svd_layers_from_sets(
            global_layers=global_layers,
            detail_layers=detail_layers,
            split_method='threshold_rho',
            split_gap=split_gap,
        ):
            return False
        self.svd_effective_rho_threshold = float(self.svd_rho_threshold)
        return True

    def _maybe_finalize_svd_layers(self, scale_ind: int) -> None:
        if not self.use_svd_depart_layers:
            return
        if scale_ind < self.start_scale_index:
            return
        if scale_ind <= self.svd_source_scale_ind:
            return
        self._materialize_svd_layer_values()
        if not self.svd_layer_rho:
            return
        if self.svd_split_mode == 'ratio':
            self._materialize_ratio_svd_layer_sets_for_scale(scale_ind=scale_ind)
            return
        if self.svd_fixed_layer_sets is not None:
            if scale_ind not in self.svd_layer_sets_by_scale:
                self._cache_svd_layer_sets_for_scale(scale_ind=scale_ind, layer_sets=self.svd_fixed_layer_sets)
            return

        self.svd_fallback_reason = None
        if self.svd_split_mode == 'threshold':
            if self._finalize_svd_layers_by_threshold():
                self._cache_svd_layer_sets_for_scale(scale_ind=scale_ind, layer_sets=self.svd_fixed_layer_sets)
                return
            self.svd_fallback_reason = 'threshold_degenerate_or_empty'
        elif self.svd_split_mode == 'gap_raw':
            if self._finalize_svd_layers_by_gap(use_log_gap=False):
                self._cache_svd_layer_sets_for_scale(scale_ind=scale_ind, layer_sets=self.svd_fixed_layer_sets)
                return
            self.svd_fallback_reason = 'gap_raw_degenerate_or_out_of_bounds'
        elif self.svd_split_mode == 'gap_log':
            if self._finalize_svd_layers_by_gap(use_log_gap=True):
                self._cache_svd_layer_sets_for_scale(scale_ind=scale_ind, layer_sets=self.svd_fixed_layer_sets)
                return
            self.svd_fallback_reason = 'gap_log_degenerate_or_out_of_bounds'
        elif self.svd_split_mode == 'gap_prior':
            if self._finalize_svd_layers_by_gap(use_log_gap=False, use_position_prior=True):
                self._cache_svd_layer_sets_for_scale(scale_ind=scale_ind, layer_sets=self.svd_fixed_layer_sets)
                return
            self.svd_fallback_reason = 'gap_prior_degenerate_or_out_of_bounds'
        fallback_detail_ratio = (
            self.svd_auto_fallback_detail_ratio
            if self.svd_split_mode in {'gap_raw', 'gap_log', 'gap_prior'}
            else None
        )
        self._finalize_svd_layers_by_ratio(detail_ratio=fallback_detail_ratio)
        if self.svd_fixed_layer_sets is not None:
            self._cache_svd_layer_sets_for_scale(scale_ind=scale_ind, layer_sets=self.svd_fixed_layer_sets)
        if self.svd_split_mode == 'threshold':
            self.svd_split_method = 'threshold_fallback_to_ratio'
        elif self.svd_split_mode in {'gap_raw', 'gap_log', 'gap_prior'}:
            self.svd_split_method = f'{self.svd_split_mode}_fallback_to_ratio'

    def _get_tokens_ratios_for_tokens_by_entropy(self, attn_entropy, entropy_boundary_line, L, device,  scale_ind: int, layer_idx: int):
        # Ratios are ordered as low-entropy, high-entropy.
        entropy_len = int(attn_entropy.numel())
        entropy_w = math.isqrt(entropy_len)
        if entropy_w * entropy_w != entropy_len:
            return None
        current_w = int(math.sqrt(L))
        attn_entropy = attn_entropy.view(entropy_w, entropy_w)
        with self.timing_scope('entropy_interpolate'):
            full_entropy = F.interpolate(attn_entropy.unsqueeze(0).unsqueeze(0), size=(current_w, current_w), mode='bilinear').squeeze(0).squeeze(0)
            entropy = full_entropy.view(-1).contiguous()

        with self.timing_scope('token_select'):
            all_indices = self._get_token_indices(L=L, device=device)
            high_fraction = self.get_effective_token_high_entropy_fraction(scale_ind=scale_ind)
            if high_fraction is None:
                threshold = torch.mean(entropy)
                mask_low_entropy = entropy <= threshold
                low_entropy_indices = all_indices[mask_low_entropy]
                high_entropy_indices = all_indices[~mask_low_entropy]
            else:
                high_count = int(round(L * high_fraction))
                high_count = min(max(high_count, 0), L)
                if high_count <= 0:
                    high_entropy_indices = all_indices[:0]
                    low_entropy_indices = all_indices
                elif high_count >= L:
                    high_entropy_indices = all_indices
                    low_entropy_indices = all_indices[:0]
                else:
                    _, high_rel = torch.topk(entropy, k=high_count, dim=0, largest=True, sorted=False)
                    high_mask = torch.zeros(L, dtype=torch.bool, device=device)
                    high_mask[high_rel] = True
                    high_entropy_indices = all_indices[high_mask]
                    low_entropy_indices = all_indices[~high_mask]
            token_sets = [low_entropy_indices, high_entropy_indices]
            ratios = list(self.get_effective_token_prune_ratios(scale_ind=scale_ind))
            kept_indices_list = []
            for ids, ratio in zip(token_sets, ratios):
                kgroup = ids.numel()
                sub_entropy = entropy[ids]
                num_keep = int(kgroup*(1.0 - ratio))
                if num_keep == 0:
                    continue
                _, topk_rel = torch.topk(sub_entropy, k=num_keep, dim=0, largest=False, sorted=False)
                global_idx = ids[topk_rel]
                kept_indices_list.append(global_idx)
        
        if not kept_indices_list:
            return None
        final_keep_indices = torch.cat(kept_indices_list, dim=0)
        final_keep_indices = torch.sort(final_keep_indices.clamp_(0, L - 1), dim=0).values
        return final_keep_indices

    def _get_token_indices(self, L: int, device):
        key = (str(device), int(L))
        cached = self._token_index_cache.get(key)
        if cached is None or cached.device != device:
            cached = torch.arange(int(L), device=device)
            self._token_index_cache[key] = cached
        return cached

    
    def compute_merge_from_entropy(self, x, attn_entropy_fn, x_shape, layer_idx, scale_ind):
        B, L, C = x.shape
        attn_entropy = None
        if self.should_direct_skip_scale(scale_ind=scale_ind):
            return skip_layer, skip_layer, skip_layer
        self._maybe_finalize_svd_layers(scale_ind=scale_ind)
        
        if self.only_scale_skip:
            if self._sacle_start(scale_ind=scale_ind, attn_entropy=attn_entropy):
                self._record_first_pruned_scale(scale_ind)
                return skip_layer, skip_layer, skip_layer
            else:
                return do_nothing, do_nothing, do_nothing
        else:
            if not self._sacle_start(scale_ind=scale_ind, attn_entropy=attn_entropy):
                return do_nothing, do_nothing, do_nothing
                       
            if not self._layer_start(layer_idx=layer_idx, scale_ind=scale_ind, attn_entropy=attn_entropy):
                return do_nothing, do_nothing, do_nothing
            
            if not self.use_token_pruning:
                # Skip the selected layer when token pruning is disabled.
                self._record_first_pruned_scale(scale_ind)
                return skip_layer, skip_layer, skip_layer
            
            # Apply token pruning for the selected scale and layer.
            if attn_entropy is None:
                attn_entropy = attn_entropy_fn()
            
            device = x.device
            current_h, current_w = x_shape[1], x_shape[2] 
            
            final_keep_indices = self._get_tokens_ratios_for_tokens_by_entropy(attn_entropy=attn_entropy, entropy_boundary_line=self.entropy_boundary_line, L=L, device=device, scale_ind=scale_ind, layer_idx=layer_idx)
            if final_keep_indices is None:
                self._record_first_pruned_scale(scale_ind)
                return skip_layer, skip_layer, skip_layer
            self._record_first_pruned_scale(scale_ind)
            final_keep_indices = final_keep_indices.unsqueeze(0).expand(B, -1).unsqueeze(-1)

            def merge_fn(merged_cur_x: torch.Tensor) -> torch.Tensor:
                idx = final_keep_indices.expand(-1, -1, merged_cur_x.size(-1))
                res = torch.gather(merged_cur_x, 1, idx)
                return res

            def unmerge_fn(unmerged_cur_x: torch.Tensor,
                        unmerged_cache_x: torch.Tensor,
                        cached_hw: Tuple[int, int]) -> torch.Tensor:
                H_cache, W_cache = cached_hw
                bg = unmerged_cache_x.view(B, H_cache, W_cache, C).permute(0, 3, 1, 2)
                bg = F.interpolate(bg, size=(current_h, current_w), mode='area')
                bg = bg.permute(0, 2, 3, 1).reshape(B, L, C)
                idx = final_keep_indices.expand(-1, -1, unmerged_cur_x.size(-1))
                bg.scatter_(1, idx, unmerged_cur_x)
                return bg

            def get_src_tgt_idx_fn() -> torch.Tensor:
                return final_keep_indices

            return merge_fn, unmerge_fn, get_src_tgt_idx_fn
                
                
    
    def clear_cache(self):
        self.dict = {}
        self.svd_layer_rho = {}
        self.svd_layer_score = {}
        self.svd_ratio_sorted_layers = None
        self._svd_layer_values_materialized = True
        self.svd_layer_sets_by_scale = {}
        self.svd_fixed_layer_sets = None
        self.structure_layers = None
        self.semantic_layers = None
        self.svd_global_layers = None
        self.svd_detail_layers = None
        self.svd_pruning_layers = None
        self.svd_split_method = None
        self.svd_split_gap = None
        self.svd_effective_rho_threshold = None
        self.svd_gap_metric = None
        self.svd_gap_value = None
        self.svd_fallback_reason = None
        self.effective_layer_detail_ratio_by_scale = {}
        self.effective_token_prune_ratio_by_scale = {}
        self.effective_token_high_entropy_fraction_by_scale = {}
        self._timing_event_records = []
        self._timing_cpu_totals = {}
        self.reset_adaptive_state()

    def get_cache_scale_len_start(self):
        if self.use_adaptive_scale:
            return size_to_width_need_cache[self.adaptive_min_effective_start_scale_index]
        else:
            return size_to_width_need_cache[self.start_scale_index]
