import torch
from typing import Tuple, Callable
import torch
import math
from typing import Type, Dict, Any, Tuple, Callable
import torch.nn.functional as F
import gc
from typing import List, Tuple, Callable

import time


import gc
from typing import List, Tuple, Callable

def do_nothing(x: torch.Tensor, *args, **kwargs):
    return x

def skip_layer(x: torch.Tensor, *args, **kwargs):
    return 0




size_to_width_need_cache = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 7,
    8: 9,
    9: 12,
    10: 16,
    11: 21,
    12: 27,
    13: 36,
    14: 48,
    15: 64,
    100: 100000, # effectively disables cache-width pruning
}


class ToProVARControllerForExperiment:
    '''
    Controller for HART scale skipping, SVD layer grouping, and token pruning.

    `only_scale_skip` skips all selected scales directly. Otherwise, the
    controller decides whether to prune or skip selected layers based on the
    scale trigger, optional SVD grouping, and `use_token_pruning`.
    '''
    def __init__(self, only_scale_skip: bool, use_adaptive_scale: bool, scale_prop: float, entropy_boundary_line: float, 
                 start_scale_index: int, use_svd_depart_layers: bool, structure_layer_ration: float, use_token_pruning: bool) -> None:
        self.only_scale_skip = only_scale_skip
        self.use_adaptive_scale = use_adaptive_scale
        self.scale_prop = scale_prop 
        self.entropy_boundary_line = entropy_boundary_line  
        self.start_scale_index = start_scale_index 
        self.use_svd_depart_layers = use_svd_depart_layers 
        self.structure_layer_ration = structure_layer_ration 
        self.use_token_pruning = use_token_pruning 
        
        self.dict = {}
    
    def _sacle_start(self, scale_ind, attn_entropy=None):
        if self.use_adaptive_scale:
            return scale_ind >= 10 # adaptive mode currently uses a fixed start scale
        else:
            return scale_ind >= self.start_scale_index
    
    def _layer_start(self, layer_idx, attn_entropy=None):
        return layer_idx >= 2 and layer_idx <= 18

    def _get_tokens_ratios_for_tokens_by_entropy(self, attn_entropy, entropy_boundary_line, L, device,  scale_ind: int, layer_idx: int):
        # Ratios are ordered as low-entropy, high-entropy.
        entropy_w = int(math.sqrt(attn_entropy.size(0)))
        current_w = int(math.sqrt(L))
        attn_entropy = attn_entropy.view(entropy_w, entropy_w)
        full_entropy = F.interpolate(attn_entropy.unsqueeze(0).unsqueeze(0), size=(current_w, current_w), mode='bilinear').squeeze(0).squeeze(0)
        entropy = full_entropy.view(-1).contiguous()
        
        threshold = torch.mean(entropy) # current boundary for splitting low- and high-entropy tokens
        all_indices = torch.arange(L, device=device)
        mask_low_entropy = entropy <= threshold
        
        low_entropy_indices = all_indices[mask_low_entropy]
        high_entropy_indices = all_indices[~mask_low_entropy]
        token_sets = [low_entropy_indices, high_entropy_indices]
        ratios = [0.3, 0.9]
        kept_indices_list = []
        for ids, ratio in zip(token_sets, ratios):
            kgroup = ids.numel()
            sub_entropy = entropy[ids]
            num_keep = int(kgroup*(1.0 - ratio))
            if num_keep == 0:
                continue
            _, topk_rel = torch.topk(sub_entropy, k=num_keep, dim=0, largest=False)
            global_idx = ids[topk_rel]
            kept_indices_list.append(global_idx)
        
        final_keep_indices = torch.cat(kept_indices_list, dim=0)
        return final_keep_indices
    
    def compute_merge_from_entropy(self, x, attn_entropy_fn, x_shape, layer_idx, scale_ind):
        B, L, C = x.shape
        attn_entropy = None
        
        if self.use_adaptive_scale:
            attn_entropy = attn_entropy_fn()
        if self.only_scale_skip:
            if self._sacle_start(scale_ind=scale_ind, attn_entropy=attn_entropy):
                return skip_layer, skip_layer, skip_layer
            else:
                return do_nothing, do_nothing, do_nothing
        else:
            if not self._sacle_start(scale_ind=scale_ind, attn_entropy=attn_entropy):
                return do_nothing, do_nothing, do_nothing
                       
            if not self._layer_start(layer_idx=layer_idx, attn_entropy=attn_entropy):
                return do_nothing, do_nothing, do_nothing
            
            if not self.use_token_pruning:
                # Skip the selected layer when token pruning is disabled.
                return skip_layer, skip_layer, skip_layer
            
            # Apply token pruning for the selected scale and layer.
            if attn_entropy is None:
                attn_entropy = attn_entropy_fn()
            
            device = x.device
            current_h, current_w = x_shape, x_shape
            
            final_keep_indices = self._get_tokens_ratios_for_tokens_by_entropy(attn_entropy=attn_entropy, entropy_boundary_line=self.entropy_boundary_line, L=L, device=device, scale_ind=scale_ind, layer_idx=layer_idx)
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
        pass

    def get_cache_scale_len_start(self):
        if self.use_adaptive_scale:
            return 24
        else:
            return size_to_width_need_cache[self.start_scale_index]
        
