"""
Definitions of blocks of VAR transformer model.
"""

from tools.nvtx_profiler import nvtx_profiler as nvtx
import math
import os
import json
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath

# Import flash_attn's attention
from flash_attn import flash_attn_func, flash_attn_func_with_entropy                  # q, k, or v: BLHc, ret: BLHc
from flash_attn import flash_attn_varlen_kvpacked_func  # qkv: N3Hc, ret: NHc


_KERNEL_DIFF_PROBE_CONFIG = None
_KERNEL_DIFF_PROBE_RECORDS = 0
_ADAPTIVE_ENTROPY_SIDE_CHANNEL_LOGGED = set()


def _env_flag(name: str, default: str = '0') -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {'1', 'true', 'yes', 'on'}


def _parse_int_set(raw_value):
    raw_value = '' if raw_value is None else str(raw_value).strip()
    if raw_value == '' or raw_value.lower() in {'none', 'all'}:
        return None if raw_value.lower() == 'all' else set()
    return {int(item.strip()) for item in raw_value.split(',') if item.strip()}


def _get_kernel_diff_probe_config():
    global _KERNEL_DIFF_PROBE_CONFIG
    if _KERNEL_DIFF_PROBE_CONFIG is not None:
        return _KERNEL_DIFF_PROBE_CONFIG
    enabled = os.environ.get('TOPROVAR_KERNEL_DIFF_PROBE', '0') == '1'
    _KERNEL_DIFF_PROBE_CONFIG = {
        'enabled': enabled,
        'scales': _parse_int_set(os.environ.get('TOPROVAR_KERNEL_DIFF_PROBE_SCALES', '6,7,8,9')),
        'layers': _parse_int_set(os.environ.get('TOPROVAR_KERNEL_DIFF_PROBE_LAYERS', '0,10,20,39')),
        'output': os.environ.get('TOPROVAR_KERNEL_DIFF_PROBE_OUTPUT', '/tmp/toprovar_kernel_diff_probe.jsonl'),
        'max_records': int(os.environ.get('TOPROVAR_KERNEL_DIFF_PROBE_MAX_RECORDS', '256')),
        'use_plain_output': os.environ.get('TOPROVAR_KERNEL_DIFF_PROBE_USE_PLAIN_OUTPUT', '1') == '1',
    }
    return _KERNEL_DIFF_PROBE_CONFIG


def _should_probe_kernel_diff(scale_ind, layer_idx):
    cfg = _get_kernel_diff_probe_config()
    if not cfg['enabled']:
        return False
    global _KERNEL_DIFF_PROBE_RECORDS
    if cfg['max_records'] > 0 and _KERNEL_DIFF_PROBE_RECORDS >= cfg['max_records']:
        return False
    scale_ind = int(scale_ind)
    layer_idx = -1 if layer_idx is None else int(layer_idx)
    if cfg['scales'] is not None and scale_ind not in cfg['scales']:
        return False
    if cfg['layers'] is not None and layer_idx not in cfg['layers']:
        return False
    return True


def _record_kernel_diff_probe(record):
    cfg = _get_kernel_diff_probe_config()
    output = cfg['output']
    if not output:
        return
    global _KERNEL_DIFF_PROBE_RECORDS
    _KERNEL_DIFF_PROBE_RECORDS += 1
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, sort_keys=True) + '\n')


def _maybe_log_adaptive_entropy_side_channel(scale_ind, layer_idx, source_scale_ind, output_mode):
    if not _env_flag('TOPROVAR_ADAPTIVE_ENTROPY_SIDE_CHANNEL_DEBUG'):
        return
    global _ADAPTIVE_ENTROPY_SIDE_CHANNEL_LOGGED
    layer_key = -1 if layer_idx is None else int(layer_idx)
    key = (os.getpid(), int(scale_ind), layer_key, int(source_scale_ind), str(output_mode))
    if key in _ADAPTIVE_ENTROPY_SIDE_CHANNEL_LOGGED:
        return
    if len(_ADAPTIVE_ENTROPY_SIDE_CHANNEL_LOGGED) >= 128:
        return
    _ADAPTIVE_ENTROPY_SIDE_CHANNEL_LOGGED.add(key)
    print(
        '[Adaptive Entropy SideChannel] '
        f'scale={int(scale_ind)} layer={layer_key} source_scale={int(source_scale_ind)} '
        f'output={output_mode}',
        flush=True,
    )


def _use_entropy_side_channel_for_output(scale_ind, source_scale_ind, collect_attn_entropy=False) -> bool:
    if not collect_attn_entropy:
        return False
    if _env_flag('TOPROVAR_ENTROPY_SIDE_CHANNEL_ALL'):
        return True
    return True


from infinity.models.fastvar_utils import do_nothing, skip_layer

# Import flash_attn's fused ops
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.rms_norm import dropout_add_rms_norm
    from flash_attn.ops.rms_norm import rms_norm as rms_norm_impl
    from flash_attn.ops.fused_dense import fused_mlp_func
    flash_fused_op_installed = True
except ImportError:
    dropout_add_layer_norm = dropout_add_rms_norm = fused_mlp_func = None
    flash_fused_op_installed = False
    
    def rms_norm_impl(x, weight, epsilon):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(epsilon))) * weight

def apply_rotary_emb_flash_attn(q, k, scale_schedule, rope2d_freqs_grid, pad_to_multiplier, rope2d_normalized_by_hw, scale_ind, idx_fn=None, seq_len=0):
    
    qk = torch.stack((q, k), dim=0)  #(2, batch_size, seq_len, head_num, head_dim)
    qk = qk.transpose(2, 3)  #(2, batch_size, head_num, seq_len, head_dim)
    device_type = qk.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        seq_len = qk.shape[3]
        start = 0
        if scale_ind >= 1:
            assert len(scale_schedule[0]) == 3
            start = np.sum([item[0] * item[1] * item[2] for item in scale_schedule[:scale_ind]])
        rope2d_freqs_grid[str(tuple(scale_schedule))] = rope2d_freqs_grid[str(tuple(scale_schedule))].to(qk.device)
        assert start+seq_len <= rope2d_freqs_grid[str(tuple(scale_schedule))].shape[4]
        rope_cache = rope2d_freqs_grid[str(tuple(scale_schedule))][:, :, :, :, start:start+seq_len] # rope_cache shape: [2, 1, 1, 1, seq_len, half_head_dim]
        if idx_fn is not None and idx_fn.__name__ != 'do_nothing':
            rope_idx = idx_fn().clamp(min=0, max=seq_len - 1) # [B, len, 1]
            rope_cache = rope_cache.repeat(1,1,rope_idx.shape[0],1,1,1)
            rope_cache = torch.gather(rope_cache,
                                      index=rope_idx.reshape(1,1,rope_idx.shape[0],1,rope_idx.shape[-2],
                                      rope_idx.shape[-1]).repeat(2,1,1,1,1,rope_cache.shape[-1]), dim=4)

        qk = qk.reshape(*qk.shape[:-1], -1, 2) #(2, batch_size, head_num, seq_len, half_head_dim, 2)
        qk = torch.stack([
            rope_cache[0] * qk[...,0] - rope_cache[1] * qk[...,1],
            rope_cache[1] * qk[...,0] + rope_cache[0] * qk[...,1],
        ], dim=-1) # (2, batch_size, head_num, seq_len, half_head_dim, 2)
        qk = qk.reshape(*qk.shape[:-2], -1) #(2, batch_size, head_num, seq_len, head_dim)
        q, k = qk.unbind(dim=0) # (batch_size, head_num, seq_len, head_dim)
    return q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous()  # (batch_size, seq_len, heads, head_dim)


class FastRMSNorm(nn.Module):
    def __init__(self, C, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.C = C
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(C))
        else:
            self.register_buffer('weight', torch.ones(C))
    
    def forward(self, x):
        src_type = x.dtype
        return rms_norm_impl(x.float(), self.weight, epsilon=self.eps).to(src_type)
    
    def extra_repr(self) -> str:
        return f'C={self.C}, eps={self.eps:g}, elementwise_affine={self.elementwise_affine}'


def get_dropout_layer(p):
    return nn.Dropout(p, inplace=True) if p > 0 else nn.Identity()


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., fused_mlp=False):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_mlp else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = get_dropout_layer(drop)
        self.heuristic = -1
    
    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x,
                weight1=self.fc1.weight,
                weight2=self.fc2.weight,
                bias1=self.fc1.bias,
                bias2=self.fc2.bias,
                activation='gelu_approx',
                save_pre_act=self.training,
                return_residual=False,
                checkpoint_lvl=0,
                heuristic=self.heuristic,
                process_group=None,
            ))
        else:
            return self.drop(self.fc2( self.act(self.fc1(x)) ))
    
    def extra_repr(self) -> str:
        return f'fused_mlp={self.fused_mlp_func is not None}'


class FFNSwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, drop=0., fused_mlp=False):
        super().__init__()
        self.fused_mlp_func = None
        hidden_features = round(2 * hidden_features / 3 / 256) * 256
        
        out_features = out_features or in_features
        self.fcg = nn.Linear(in_features, hidden_features, bias=False)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = get_dropout_layer(drop)
    
    def forward(self, x):
        return self.drop(self.fc2( F.silu(self.fcg(x), inplace=True).mul_(self.fc1(x)) ))
    
    def extra_repr(self) -> str:
        return f'fused_mlp={self.fused_mlp_func is not None}'




class FastVARSelfAttention(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12,
        proj_drop=0., tau=1, cos_attn=False, customized_flash_attn=True, use_flex_attn=False, 
        batch_size=2, pad_to_multiplier=1, rope2d_normalized_by_hw=0,
    ):
        """
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        :param proj_drop: always 0 for testing
        :param tau: always 1
        :param cos_attn: when enabled, q and k are L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        :param customized_flash_attn:
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.using_flash = customized_flash_attn
        
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.tau, self.cos_attn = tau, cos_attn
        if self.cos_attn:
            self.scale = 1

            size = (1, self.num_heads, 1, 1)
            
            # size: 11H1 or 1H11
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=size, fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim) / self.tau
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
        
        self.caching = False    # kv caching: only used during inference
        self.cached_k = None    # kv caching: only used during inference
        self.cached_v = None    # kv caching: only used during inference

        self.batch_size = batch_size
        self.use_flex_attn = use_flex_attn
        self.pad_to_multiplier = pad_to_multiplier

        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw

        # Attention entropy is captured for ToProVAR pruning decisions.
        self.attn_entropy = None
        self.latest_attn_entropy = None
        self.latest_attn_entropy_scale_ind = None
        self.latest_attn_entropy_per_head = None
        self.latest_attn_entropy_per_head_scale_ind = None
        self.source_entropy_scale_ind = 8
        self.entropy_extract_scale_ind = 8

    
    def kv_caching(self, enable: bool): # kv caching: only used during inference
        self.caching = enable
        self.cached_k = None
        self.cached_v = None

    def _set_latest_attn_entropy(self, attn_entropy, scale_ind, source_scale_ind):
        per_head_entropy = attn_entropy[0]
        reduced_entropy = per_head_entropy.mean(dim=0)
        self.latest_attn_entropy = reduced_entropy
        self.latest_attn_entropy_scale_ind = int(scale_ind)
        self.latest_attn_entropy_per_head = per_head_entropy
        self.latest_attn_entropy_per_head_scale_ind = int(scale_ind)
        if int(scale_ind) == int(source_scale_ind):
            self.attn_entropy = reduced_entropy
        return reduced_entropy
    
    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, scale_ind=0,rope_idx=None,ori_len=0, return_q_k: bool = False, collect_attn_entropy: bool = False, layer_idx=None):
        """
        :param (fp32) x: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        :param (fp32) attn_bias_or_two_vector:
                if not using_flash:
                    a block-wise, lower-triangle matrix, like:
                    [[[[0, -, -, -, -, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
                    where 0 means visible and - means invisible (-inf)
                else:
                    a tuple of two 1-dim int vector (VAR_visible_kvlen, VAR_invisible_qlen)
        :return: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        """
        # q_t, k_t = self.get_qk(x, scale_schedule, rope2d_freqs_grid, scale_ind, rope_idx, ori_len)
        
        
        # x: fp32
        B, L, C = x.shape
        with nvtx.range('compute qkv'):
            # qkv: amp, bf16
            qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)  # BL3Hc
            if self.using_flash: q, k, v = qkv.unbind(dim=2); L_dim = 1           # q or k or v: all are shaped in (B:batch_size, L:seq_len, H:heads, c:head_dim)
            else: q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); L_dim = 2   # q or k or v: all are shaped in (B:batch_size, H:heads, L:seq_len, c:head_dim)
        
    
        if self.cos_attn:
            
            # Reshape the learned attention scale for the FlashAttention layout.
            with nvtx.range('reshape one head'):
                if self.using_flash and self.scale_mul_1H11.shape == (1, self.num_heads, 1, 1):
                    self.scale_mul_1H11 = nn.Parameter(self.scale_mul_1H11.reshape(1, 1, self.num_heads, 1))
                
                
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp() # 11H1 (flash), or 1H11 (not flash)
            with nvtx.range('qk normalize'):
                q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()   # fp32
                k = F.normalize(k, dim=-1, eps=1e-12).contiguous()                  # fp32
            with nvtx.range('v contiguous'):
                v = v.contiguous()                                                  # bf16
        else:   # be contiguous, to make kernel happy
            with nvtx.range('qkv contiguous'):
                q = q.contiguous()      # bf16
                k = k.contiguous()      # bf16
                v = v.contiguous()      # bf16
        if rope2d_freqs_grid is not None:
            with nvtx.range(f'rope for sa at scale {scale_ind}'):
                q, k = apply_rotary_emb_flash_attn(q, k, scale_schedule, rope2d_freqs_grid, self.pad_to_multiplier, self.rope2d_normalized_by_hw, scale_ind, idx_fn=rope_idx, seq_len=ori_len) #, freqs_cis=freqs_cis) 
            
        if self.caching:    # kv caching: only used during inference
            if self.cached_k is None: self.cached_k = k; self.cached_v = v
            else: k = self.cached_k = torch.cat((self.cached_k, k), dim=L_dim); v = self.cached_v = torch.cat((self.cached_v, v), dim=L_dim) # 10,521
        
        if self.using_flash:
            if attn_bias_or_two_vector is not None: # training
                kw = dict(VAR_visible_kvlen=attn_bias_or_two_vector[0], VAR_invisible_qlen=attn_bias_or_two_vector[1])
            else:                                   # inference (autoregressive sampling)
                kw = dict()
            
            source_scale_ind = int(self.source_entropy_scale_ind)
            should_probe_kernel_diff = _should_probe_kernel_diff(scale_ind=scale_ind, layer_idx=layer_idx)
            if should_probe_kernel_diff:
                oup_plain = flash_attn_func(q.to(v.dtype), k.to(v.dtype), v, dropout_p=0, softmax_scale=self.scale, **kw)
                oup_entropy, attn_entropy = flash_attn_func_with_entropy(q.to(v.dtype), k.to(v.dtype), v, softmax_scale=self.scale)
                diff = (oup_plain.float() - oup_entropy.float()).abs()
                record = {
                    'pid': os.getpid(),
                    'scale_idx': int(scale_ind),
                    'layer_idx': None if layer_idx is None else int(layer_idx),
                    'shape': list(oup_plain.shape),
                    'dtype': str(oup_plain.dtype),
                    'max_abs_diff': float(diff.max().item()) if diff.numel() else 0.0,
                    'mean_abs_diff': float(diff.mean().item()) if diff.numel() else 0.0,
                    'bitwise_equal': bool(torch.equal(oup_plain, oup_entropy)),
                    'allclose_rtol1e-03_atol1e-05': bool(torch.allclose(oup_plain, oup_entropy, rtol=1e-3, atol=1e-5)),
                    'collect_attn_entropy': bool(collect_attn_entropy),
                    'used_plain_output': bool(_get_kernel_diff_probe_config()['use_plain_output']),
                }
                _record_kernel_diff_probe(record)
                if collect_attn_entropy:
                    self._set_latest_attn_entropy(attn_entropy, scale_ind, source_scale_ind)
                use_plain_output = (
                    _get_kernel_diff_probe_config()['use_plain_output']
                    or (
                        collect_attn_entropy
                        and _use_entropy_side_channel_for_output(
                            scale_ind,
                            source_scale_ind,
                            collect_attn_entropy=collect_attn_entropy,
                        )
                    )
                )
                oup = oup_plain if use_plain_output else oup_entropy
            elif not collect_attn_entropy:
                oup = flash_attn_func(q.to(v.dtype), k.to(v.dtype), v, dropout_p=0, softmax_scale=self.scale, **kw)
            else:
                oup_entropy, attn_entropy = flash_attn_func_with_entropy(q.to(v.dtype), k.to(v.dtype), v, softmax_scale=self.scale)
                # Keep source entropy stable for token pruning; SEGR reads latest_attn_entropy only.
                self._set_latest_attn_entropy(attn_entropy, scale_ind, source_scale_ind)
                if _use_entropy_side_channel_for_output(
                    scale_ind,
                    source_scale_ind,
                    collect_attn_entropy=collect_attn_entropy,
                ):
                    oup = flash_attn_func(q.to(v.dtype), k.to(v.dtype), v, dropout_p=0, softmax_scale=self.scale, **kw)
                    _maybe_log_adaptive_entropy_side_channel(
                        scale_ind=scale_ind,
                        layer_idx=layer_idx,
                        source_scale_ind=source_scale_ind,
                        output_mode='plain',
                    )
                else:
                    oup = oup_entropy
                    _maybe_log_adaptive_entropy_side_channel(
                        scale_ind=scale_ind,
                        layer_idx=layer_idx,
                        source_scale_ind=source_scale_ind,
                        output_mode='entropy',
                    )
            oup = oup.view(B, L, C)
            
        else:
            # if self.cos_attn: q, k are in fp32; v is in bf16
            # else: q, k, v are in bf16
            if self.use_flex_attn and attn_fn is not None:
                oup = attn_fn(q, k, v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
            else:
                # flashattn
                q, k, v = q.transpose(1,2), k.transpose(1,2),v.transpose(1,2) # 2, 1, 28, 128
                oup = flash_attn_func(q.to(v.dtype), k.to(v.dtype), v, dropout_p=0, softmax_scale=self.scale).reshape(B, L, C)
                # slow attn
                #oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias_or_two_vector, dropout_p=0).transpose(1, 2).reshape(B, L, C) #b head l d
            # oup: bf16
        if return_q_k:
            return self.proj_drop(self.proj(oup)), q.transpose(1,2), k.transpose(1,2)
        else:
            return self.proj_drop(self.proj(oup))
    
    def extra_repr(self) -> str:
        tail = ''
        return f'using_flash={self.using_flash}, tau={self.tau}, cos_attn={self.cos_attn}{tail}'
    def get_attn_entropy(self):
        if self.attn_entropy is None:
            raise ValueError('Self-attn: no flash attn, no entropy')
        return self.attn_entropy

    def get_latest_attn_entropy(self, scale_ind=None):
        if self.latest_attn_entropy is None:
            raise ValueError('Self-attn: no latest flash attn entropy')
        if scale_ind is not None and self.latest_attn_entropy_scale_ind != int(scale_ind):
            raise ValueError(
                f'Self-attn latest entropy scale mismatch: expected {scale_ind}, '
                f'got {self.latest_attn_entropy_scale_ind}'
            )
        return self.latest_attn_entropy

    def get_latest_attn_entropy_per_head(self, scale_ind=None):
        if self.latest_attn_entropy_per_head is None:
            raise ValueError('Self-attn: no latest per-head flash attn entropy')
        if scale_ind is not None and self.latest_attn_entropy_per_head_scale_ind != int(scale_ind):
            raise ValueError(
                f'Self-attn latest per-head entropy scale mismatch: expected {scale_ind}, '
                f'got {self.latest_attn_entropy_per_head_scale_ind}'
            )
        return self.latest_attn_entropy_per_head
    



class CrossAttention(nn.Module):
    def __init__(
        self, for_attn_pool=False, embed_dim=768, kv_dim=4096, num_heads=12,
        proj_drop=0., cos_attn=False,
    ):
        """
        :param for_attn_pool: only used in VAR.text_proj_for_sos
        :param embed_dim: Q's dim
        :param kv_dim: K's and V's dim
        :param num_heads: num heads of multi-head attention
        :param proj_drop: proj drop out
        :param cos_attn: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        """
        cos_attn = False    # Cross attention with T5 keys/values uses scaled dot-product attention.
        super().__init__()
        self.for_attn_pool = for_attn_pool
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads  # =64
        self.cos_attn = cos_attn
        if self.cos_attn:
            self.scale = 1
            self.scale_mul_1H1 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim)
        
        if for_attn_pool:
            q = torch.empty(1, self.num_heads, self.head_dim)
            nn.init.trunc_normal_(q, mean=0, std=math.sqrt(1 / embed_dim / 3))
            self.mat_q = nn.Parameter(q)
        else:
            self.mat_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mat_kv = nn.Linear(kv_dim, embed_dim*2, bias=False)
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
    
    def forward(self, q, ca_kv):
        """
        :param q: shaped as (batch, seq_len, Q_dim)
        :param ca_kv: contains several vectors, each of which is shaped as (len_i, KV_dim). We have [len_1xKV_dim, len_2xKV_dim, len_3xKV_dim, ...] and lens == [len_1, len_2, len_3, ...]
            - kv_compact: shaped as (sum(lens), KV_dim)
            - cu_seqlens_k: cumulated sum of lens
            - max_seqlen_k: int, max(lens)
        NOTE: seq_len (num of Qs) can reach 10k;  but len_i (num of KVs) must <= 256
        
        :return: shaped as (batch, seq_len, Q_dim)
        """
        kv_compact, cu_seqlens_k, max_seqlen_k = ca_kv
        N = kv_compact.shape[0]
        
        kv_compact = F.linear(kv_compact, weight=self.mat_kv.weight, bias=torch.cat((self.zero_k_bias, self.v_bias))).view(N, 2, self.num_heads, self.head_dim) # NC => N2Hc
        # attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens
        
        if not self.for_attn_pool:
            B, Lq = q.shape[:2]
            q_compact = self.mat_q(q).view(-1, self.num_heads, self.head_dim)
        else:
            B = cu_seqlens_k.shape[0] - 1
            Lq = 1
            q_compact = self.mat_q.repeat(B, 1, 1).to(dtype=kv_compact.dtype)
        
        if self.cos_attn:   # always False
            scale_mul = self.scale_mul_1H1.clamp_max(self.max_scale_mul).exp()
            k, v = kv_compact.unbind(dim=1)
            q_compact = F.normalize(q_compact, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
            kv_compact = torch.stack((k, v), dim=1)
        
        q_compact = q_compact.contiguous()
        kv_compact = kv_compact.contiguous()
        
        cu_seqlens_q = torch.arange(0, Lq * (B+1), Lq, dtype=torch.int32, device=q_compact.device)
        if q_compact.dtype == torch.float32:    # run FlashAttention in bf16 for fp32 query tensors
            oup = flash_attn_varlen_kvpacked_func(q=q_compact.to(dtype=torch.bfloat16), kv=kv_compact.to(dtype=torch.bfloat16), cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=Lq, max_seqlen_k=max_seqlen_k, dropout_p=0, softmax_scale=self.scale).reshape(B, Lq, -1)
            oup = oup.float()
        else:
            oup = flash_attn_varlen_kvpacked_func(q=q_compact, kv=kv_compact, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=Lq, max_seqlen_k=max_seqlen_k, dropout_p=0, softmax_scale=self.scale).reshape(B, Lq, -1)
        
        return self.proj_drop(self.proj(oup))
    
    def extra_repr(self) -> str:
        return f'Cq={self.embed_dim}, Ckv={self.kv_dim}, cos_attn={self.cos_attn}'


def compute_SVD(entropy, scale=None, layer_index=None) :
    
    try:
        entropy = entropy.reshape(-1)
        L = entropy.shape[0]
        width = math.isqrt(L)
        if width * width != L:
            return None
        entropy = entropy.view((width, width)).to(entropy.device)
        with nvtx.range("Compute SVD function:"):
            S = torch.linalg.svdvals(entropy)
        if S.numel() < 2:
            return None
        ratio = S[0] / S[1].clamp_min(1e-12)
        return ratio
    
    except Exception as e:
        print(f"Failed to process attention entropy: {str(e)}")
        return None


class FastVARCrossAttnBlock(nn.Module):

    def __init__(
        self,
        attnController,
        embed_dim, kv_dim, cross_attn_layer_scale, cond_dim, act: bool, shared_aln: bool, norm_layer: partial,
        num_heads, mlp_ratio=4., drop=0., drop_path=0., tau=1, cos_attn=False,
        swiglu=False, customized_flash_attn=False, fused_mlp=False, fused_norm_func=None, checkpointing_sa_only=False,
        use_flex_attn=False, batch_size=2, pad_to_multiplier=1, apply_rope2d=False, rope2d_normalized_by_hw=False,
    ):
        super(FastVARCrossAttnBlock, self).__init__()
        self.C, self.D = embed_dim, cond_dim
        self.drop_path_rate = drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.sa = FastVARSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, proj_drop=drop, tau=tau, cos_attn=cos_attn, customized_flash_attn=customized_flash_attn,
            use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
        )
        self.sa.source_entropy_scale_ind = attnController.get_entropy_extract_scale_ind()
        self.sa.entropy_extract_scale_ind = attnController.get_entropy_extract_scale_inds()
        self.ca = CrossAttention(embed_dim=embed_dim, kv_dim=kv_dim, num_heads=num_heads, proj_drop=drop, cos_attn=cos_attn)
        self.using_swiglu = swiglu
        self.ffn = (FFNSwiGLU if swiglu else FFN)(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio / 256) * 256, drop=drop, fused_mlp=fused_mlp)
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.fused_norm_func = fused_norm_func
        self.norm_eps = norm_layer.keywords.get('eps', 1e-6)
        self.ca_norm = norm_layer(embed_dim, elementwise_affine=True)
        
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin) if act else nn.Sequential(lin)
        
        if cross_attn_layer_scale >= 0:
            self.ca_gamma = nn.Parameter(cross_attn_layer_scale * torch.ones(embed_dim), requires_grad=True)
        else:
            self.ca_gamma = 1
        
        self.attnController = attnController
        
        self.checkpointing_sa_only = checkpointing_sa_only

        self.previous_scale_cache_self_attn = None
        self.previous_scale_cache_cross_attn = None
        self.previous_scale_cache_ffn = None
        self.cached_size = [self.attnController.get_cache_scale_len_start(), self.attnController.get_cache_scale_len_start()] # cache scale features for subsequent feature restoration
        self.cnt = 0
        
      
    # NOTE: attn_bias_or_two_vector is None during inference
    
    
    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, scale_ind=0, layer_idx=-1, x_shape=None, file_path=None):
        
        width = x_shape[1] if x_shape is not None else 0
        def get_entropy_hook():
            return self.sa.get_attn_entropy()
        
        
        
        gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
        merge_fn, unmerge_fn, idx_fn = self.attnController.compute_merge_from_entropy(x=x, attn_entropy_fn=get_entropy_hook, x_shape=x_shape, layer_idx=layer_idx, scale_ind=scale_ind)

        shortcut = x
        timing_scope = self.attnController.timing_scope
        if merge_fn is skip_layer:
            return shortcut

        with nvtx.range(f'sa at scale={scale_ind}, width={width}'):
            with timing_scope('merge_sa'):
                merged_x_sa = merge_fn(x)
            with timing_scope('self_attn'):
                x_sa = self.fused_norm_func(C=self.C, eps=self.norm_eps, x=merged_x_sa, scale=scale1, shift=shift1)
                collect_attn_entropy = self.attnController.should_collect_attention_entropy(scale_ind=scale_ind)
                x_sa = self.sa(
                    x_sa,
                    attn_bias_or_two_vector,
                    attn_fn,
                    scale_schedule,
                    rope2d_freqs_grid,
                    scale_ind=scale_ind,
                    rope_idx=idx_fn,
                    ori_len=shortcut.shape[1],
                    collect_attn_entropy=collect_attn_entropy,
                    layer_idx=layer_idx,
                ).mul_(gamma1)
            if self.attnController.should_record_adaptive_entropy(scale_ind=scale_ind):
                self.attnController.record_adaptive_entropy(
                    scale_ind=scale_ind,
                    layer_idx=layer_idx,
                    attn_entropy=self.sa.get_latest_attn_entropy(scale_ind=scale_ind),
                )
            if self.attnController.should_record_svd_layer(scale_ind=scale_ind, layer_idx=layer_idx):
                with timing_scope('svd_compute'):
                    rho = compute_SVD(get_entropy_hook(), scale=scale_ind, layer_index=layer_idx)
                self.attnController.record_svd_layer(scale_ind=scale_ind, layer_idx=layer_idx, rho=rho)
            with timing_scope('unmerge_sa'):
                x_sa = unmerge_fn(x_sa,self.previous_scale_cache_self_attn,self.cached_size)
         
        with nvtx.range(f'update cache sa at  scale={scale_ind}, layer={layer_idx}'):
            if x.shape[1] in [self.cached_size[0]*self.cached_size[1]]:
                self.previous_scale_cache_self_attn = x_sa
        x = shortcut + self.drop_path(x_sa)
       
       
        with nvtx.range(f'ca at scale={scale_ind}, width={width}'):
            with timing_scope('merge_ca'):
                merged_x_ca = merge_fn(x)
            with timing_scope('cross_attn'):
                x_ca = self.ca(self.ca_norm(merged_x_ca), ca_kv).float().mul_(self.ca_gamma)
            with timing_scope('unmerge_ca'):
                x_ca = unmerge_fn(x_ca,self.previous_scale_cache_cross_attn, self.cached_size)
        with nvtx.range(f'update cache ca at  scale={scale_ind}, layer={layer_idx}'):
            if x.shape[1] in [self.cached_size[0]*self.cached_size[1]]:
                self.previous_scale_cache_cross_attn = x_ca
        x = x + x_ca

        with nvtx.range(f'ffn at scale={scale_ind}, width={width}'):
            with timing_scope('merge_ffn'):
                merged_x_ffn = merge_fn(x)
            with timing_scope('ffn'):
                x_ffn = self.ffn(self.fused_norm_func(C=self.C, eps=self.norm_eps, x=merged_x_ffn, scale=scale2, shift=shift2)).mul(gamma2)
            with timing_scope('unmerge_ffn'):
                x_ffn = unmerge_fn(x_ffn,self.previous_scale_cache_ffn,self.cached_size)
        with nvtx.range(f'update cache ffn at  scale={scale_ind}, layer={layer_idx}'): 
            if x.shape[1] in [self.cached_size[0] * self.cached_size[1]]:
                self.previous_scale_cache_ffn = x_ffn
            x = x + self.drop_path(x_ffn) # mul(gamma2) cannot be in-place because FusedMLP may reuse the tensor

        return x
        

    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}, fused_norm={self.fused_norm_func is not None}, ca_gamma={"<learnable>" if isinstance(self.ca_gamma, nn.Parameter) else self.ca_gamma}'
    def clear_cache(self):
        self.previous_scale_cache_self_attn = None
        self.previous_scale_cache_cross_attn = None
        self.previous_scale_cache_ffn = None


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, act: bool, norm_layer: partial, fused_norm_func=None):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.fused_norm_func = fused_norm_func
        self.norm_eps = norm_layer.keywords.get('eps', 1e-6)
        lin = nn.Linear(D, 2*C)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin) if act else nn.Sequential(lin)
    
    def forward(self, x_BLC: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        if self.fused_norm_func is None:
            return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)
        else:
            return self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x_BLC, scale=scale, shift=shift)
