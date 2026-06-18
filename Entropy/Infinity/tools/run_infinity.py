import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
from typing import List
import time
import argparse
import json

import cv2
import numpy as np
import torch
torch._dynamo.config.cache_size_limit=64
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
import torch.nn.functional as F
from torch.cuda.amp import autocast
from safetensors.torch import load_file as load_safetensors_file


from infinity.models.infinity import Infinity
from infinity.models.basic import *
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
def parse_protected_global_layers(raw_value):
    if raw_value is None:
        return []
    if isinstance(raw_value, (list, tuple, set)):
        values = list(raw_value)
    else:
        text = str(raw_value).strip()
        if not text:
            return []
        if text.startswith('[') and text.endswith(']'):
            text = text[1:-1]
        values = [item.strip() for item in text.split(',') if item.strip()]
    return sorted({int(item) for item in values})


def configure_svd_controller(
    controller,
    *,
    model=None,
    svd_source_scale_index=7,
    svd_split_mode='ratio',
    svd_rho_threshold=-1.0,
    protected_global_layers=None,
):
    if controller is None:
        return None

    split_mode = str(svd_split_mode or 'ratio').strip().lower()
    if split_mode not in {'ratio', 'threshold', 'gap_raw', 'gap_log', 'gap_prior'}:
        raise ValueError(
            f'Unsupported svd_split_mode={svd_split_mode!r}, '
            'expected one of ratio, threshold, gap_raw, gap_log, gap_prior'
        )

    source_scale_index = int(svd_source_scale_index)
    if getattr(controller, 'use_adaptive_scale', False) and source_scale_index != 8:
        print(
            '[Adaptive Pruning] Normalizing svd_source_scale_index '
            f'from {source_scale_index} to 8 for adaptive SEGR.'
        )
        source_scale_index = 8
    start_scale_index = int(getattr(controller, 'effective_start_scale_index', None) or getattr(controller, 'start_scale_index', 0))
    if getattr(controller, 'use_adaptive_scale', False):
        start_scale_index = max(start_scale_index, 9)
    if getattr(controller, 'use_svd_depart_layers', False) and source_scale_index >= start_scale_index:
        raise ValueError(
            f'svd_source_scale_index={source_scale_index} must be smaller than '
            f'start_scale_index={start_scale_index}'
        )

    protected_layers = parse_protected_global_layers(protected_global_layers)
    normalized_threshold = None if svd_rho_threshold is None or float(svd_rho_threshold) <= 0.0 else float(svd_rho_threshold)

    controller.svd_source_scale_ind = source_scale_index
    controller.svd_source_scale_index = source_scale_index
    controller.svd_split_mode = split_mode
    controller.svd_rho_threshold = normalized_threshold
    controller.protected_global_layers = {int(layer_idx) for layer_idx in protected_layers}
    controller.svd_fallback_reason = None
    if model is not None:
        for block in getattr(model, "unregistered_blocks", []):
            sa = getattr(block, "sa", None)
            if sa is None:
                continue
            sa.source_entropy_scale_ind = source_scale_index
            if hasattr(controller, 'get_entropy_extract_scale_inds'):
                sa.entropy_extract_scale_ind = controller.get_entropy_extract_scale_inds()
            else:
                sa.entropy_extract_scale_ind = source_scale_index
            sa.attn_entropy = None
            if hasattr(sa, 'latest_attn_entropy'):
                sa.latest_attn_entropy = None
                sa.latest_attn_entropy_scale_ind = None
    if hasattr(controller, 'clear_cache'):
        controller.clear_cache()
    return controller


def encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt=False):
    if enable_positive_prompt:
        print(f'before positive_prompt aug: {prompt}')
        prompt = aug_with_positive_prompt(prompt)
        print(f'after positive_prompt aug: {prompt}')
    print(f'prompt={prompt}')
    captions = [prompt]
    tokens = text_tokenizer(text=captions, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = tokens.input_ids.cuda(non_blocking=True)
    mask = tokens.attention_mask.cuda(non_blocking=True)
    text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
    lens: List[int] = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    Ltext = max(lens)    
    kv_compact = []
    for len_i, feat_i in zip(lens, text_features.unbind(0)):
        kv_compact.append(feat_i[:len_i])
    kv_compact = torch.cat(kv_compact, dim=0)
    text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    return text_cond_tuple

def aug_with_positive_prompt(prompt):
    for key in ['man', 'woman', 'men', 'women', 'boy', 'girl', 'child', 'person', 'human', 'adult', 'teenager', 'employee',
                'employer', 'worker', 'mother', 'father', 'sister', 'brother', 'grandmother', 'grandfather', 'son', 'daughter']:
        if key in prompt:
            prompt = prompt + '. very smooth faces, good looking faces, face to the camera, perfect facial features'
            break
    return prompt


def gen_one_img(
    infinity_test, 
    vae, 
    text_tokenizer,
    text_encoder,
    prompt, 
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    file_path=None,
):
    sstt = time.time()
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)
    text_cond_tuple = encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt)
    if negative_prompt:
        negative_label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, negative_prompt)
    else:
        negative_label_B_or_BLT = None
    print(f'cfg: {cfg_list}, tau: {tau_list}')
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        _, intermidiate_list, img_list = infinity_test.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
            B=1, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
            cfg_sc=cfg_sc, cfg_list=cfg_list, tau_list=tau_list, top_k=top_k, top_p=top_p,
            returns_vemb=1, ratio_Bl1=None, gumbel=gumbel, norm_cfg=False,
            cfg_exp_k=cfg_exp_k, cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type, softmax_merge_topk=softmax_merge_topk,
            ret_img=True, trunk_scale=1000,
            gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
            sampling_per_bits=sampling_per_bits,
            file_path=file_path
        )

    print(f'time cost: {time.time() - sstt}')
    attn_controller = getattr(infinity_test, 'attnController', None)
    if attn_controller is not None and hasattr(attn_controller, 'print_timing_stats'):
        attn_controller.print_timing_stats()
    
    
    
    img = img_list[0]

    return img

def load_tokenizer(t5_path =''):
    print(f'[Loading tokenizer and text encoder]')
    text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(t5_path, revision=None, legacy=True)
    text_tokenizer.model_max_length = 512
    text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.float16)
    text_encoder.to('cuda')
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    return text_tokenizer, text_encoder

def load_infinity(
    rope2d_each_sa_layer, 
    rope2d_normalized_by_hw, 
    use_scale_schedule_embedding, 
    pn, 
    use_bit_label, 
    add_lvl_embeding_only_first_block, 
    model_path='', 
    scale_schedule=None, 
    vae=None, 
    device='cuda', 
    model_kwargs=None,
    text_channels=2048,
    apply_spatial_patchify=0,
    use_flex_attn=False,
    bf16=False,
    checkpoint_type='torch',
    only_scale_skip=False,
    use_adaptive_scale=False,
    scale_prop=1,
    entropy_boundary_line=1,
    start_scale_index=1,
    direct_skip_start_scale_index=100,
    use_svd_depart_layers=False,
    structure_layer_ration=1,
    use_token_pruning=False,
    svd_rho_threshold=-1.0,
    svd_source_scale_index=7,
    svd_split_mode='ratio',
    protected_global_layers=None,
    adaptive_entropy_s_min=None,
    adaptive_entropy_collect_start_scale_index=7,
    adaptive_entropy_trigger_start_scale_index=8,
    adaptive_entropy_gamma=1.4e-6,
    adaptive_entropy_epsilon=1e-5,
    adaptive_entropy_lambda=1.0,
    adaptive_entropy_score_mode='delta_h_width_power',
    adaptive_entropy_signal_mode='layer_mean_delta',
    adaptive_entropy_signal_layer_idx=5,
    adaptive_max_raw_start_scale_index=10,
    force_entropy_collection_scales='',
):
    def load_sharded_safetensors(model, model_dir, device):
        safe_index_path = osp.join(model_dir, 'model.safetensors.index.json')
        if not osp.exists(safe_index_path):
            raise ValueError(
                f"Can't find a safetensors checkpoint index (model.safetensors.index.json) in {model_dir}."
            )

        with open(safe_index_path, 'r', encoding='utf-8') as f:
            weight_map = json.load(f)['weight_map']

        shard_files = []
        seen = set()
        for shard_name in weight_map.values():
            if shard_name not in seen:
                seen.add(shard_name)
                shard_files.append(shard_name)

        for shard_name in shard_files:
            shard_path = osp.join(model_dir, shard_name)
            shard_state_dict = load_safetensors_file(shard_path, device='cpu')
            incompatible = model.load_state_dict(shard_state_dict, strict=False)
            if incompatible.unexpected_keys:
                raise ValueError(
                    f'Unexpected keys found while loading shard {shard_name}: {incompatible.unexpected_keys[:10]}'
                )
            del shard_state_dict

    print(f'[Loading Infinity]')
    text_maxlen = 512
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        infinity_test: Infinity = Infinity(
            vae_local=vae, text_channels=text_channels, text_maxlen=text_maxlen,
            shared_aln=True, raw_scale_schedule=scale_schedule,
            checkpointing='full-block',
            customized_flash_attn=True,
            fused_norm=True,
            pad_to_multiplier=128,
            use_flex_attn=use_flex_attn,
            add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
            use_bit_label=use_bit_label,
            rope2d_each_sa_layer=rope2d_each_sa_layer,
            rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            pn=pn,
            apply_spatial_patchify=apply_spatial_patchify,
            inference_mode=True,
            train_h_div_w_list=[1.0],
            only_scale_skip=only_scale_skip,
            use_adaptive_scale=use_adaptive_scale,
            scale_prop=scale_prop,
            entropy_boundary_line=entropy_boundary_line,
            start_scale_index=start_scale_index,
            direct_skip_start_scale_index=direct_skip_start_scale_index,
            use_svd_depart_layers=use_svd_depart_layers,
            structure_layer_ration=structure_layer_ration,
            use_token_pruning=use_token_pruning,
            svd_rho_threshold=svd_rho_threshold,
            svd_source_scale_index=svd_source_scale_index,
            svd_split_mode=svd_split_mode,
            protected_global_layers=protected_global_layers,
            adaptive_entropy_s_min=adaptive_entropy_s_min,
            adaptive_entropy_collect_start_scale_index=adaptive_entropy_collect_start_scale_index,
            adaptive_entropy_trigger_start_scale_index=adaptive_entropy_trigger_start_scale_index,
            adaptive_entropy_gamma=adaptive_entropy_gamma,
            adaptive_entropy_epsilon=adaptive_entropy_epsilon,
            adaptive_entropy_lambda=adaptive_entropy_lambda,
            adaptive_entropy_score_mode=adaptive_entropy_score_mode,
            adaptive_entropy_signal_mode=adaptive_entropy_signal_mode,
            adaptive_entropy_signal_layer_idx=adaptive_entropy_signal_layer_idx,
            adaptive_max_raw_start_scale_index=adaptive_max_raw_start_scale_index,
            force_entropy_collection_scales=force_entropy_collection_scales,
            **model_kwargs,
        ).to(device=device)
        print(f'[you selected Infinity with {model_kwargs=}] model size: {sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, bf16={bf16}')

        if bf16:
            for block in infinity_test.unregistered_blocks:
                block.bfloat16()

        infinity_test.eval()
        infinity_test.requires_grad_(False)

        infinity_test.cuda()
        torch.cuda.empty_cache()

        print(f'[Load Infinity weights]')
        if checkpoint_type == 'torch':
            state_dict = torch.load(model_path, map_location=device)
            print(infinity_test.load_state_dict(state_dict))
        elif checkpoint_type == 'torch_shard':
            safe_index_path = osp.join(model_path, 'model.safetensors.index.json')
            if osp.exists(safe_index_path):
                load_sharded_safetensors(infinity_test, model_path, device)
            else:
                from transformers.modeling_utils import load_sharded_checkpoint
                load_sharded_checkpoint(infinity_test, model_path, strict=False)
        infinity_test.rng = torch.Generator(device=device)
        configure_svd_controller(
            infinity_test.attnController,
            model=infinity_test,
            svd_source_scale_index=svd_source_scale_index,
            svd_split_mode=svd_split_mode,
            svd_rho_threshold=svd_rho_threshold,
            protected_global_layers=protected_global_layers,
        )
        return infinity_test

def load_visual_tokenizer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load vae
    if args.vae_type in [14, 16,18,20,24,32,64]:
        from infinity.models.bsq_vae.vae import vae_model
        schedule_mode = "dynamic"
        codebook_dim = args.vae_type
        codebook_size = 2**codebook_dim
        if args.apply_spatial_patchify:
            patch_size = 8
            encoder_ch_mult=[1, 2, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4]
        else:
            patch_size = 16
            encoder_ch_mult=[1, 2, 4, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4, 4]
        vae = vae_model(args.vae_path, schedule_mode, codebook_dim, codebook_size, patch_size=patch_size, 
                        encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult, test_mode=True).to(device)
    else:
        raise ValueError(f'vae_type={args.vae_type} not supported')
    return vae

def load_transformer(vae, args):
    if getattr(args, 'kernel_diff_probe', False):
        os.environ['TOPROVAR_KERNEL_DIFF_PROBE'] = '1'
        os.environ['TOPROVAR_KERNEL_DIFF_PROBE_SCALES'] = str(args.kernel_diff_probe_scales)
        os.environ['TOPROVAR_KERNEL_DIFF_PROBE_LAYERS'] = str(args.kernel_diff_probe_layers)
        os.environ['TOPROVAR_KERNEL_DIFF_PROBE_OUTPUT'] = str(args.kernel_diff_probe_output)
        os.environ['TOPROVAR_KERNEL_DIFF_PROBE_MAX_RECORDS'] = str(args.kernel_diff_probe_max_records)
        os.environ['TOPROVAR_KERNEL_DIFF_PROBE_USE_PLAIN_OUTPUT'] = str(args.kernel_diff_probe_use_plain_output)
    else:
        for key in (
            'TOPROVAR_KERNEL_DIFF_PROBE',
            'TOPROVAR_KERNEL_DIFF_PROBE_SCALES',
            'TOPROVAR_KERNEL_DIFF_PROBE_LAYERS',
            'TOPROVAR_KERNEL_DIFF_PROBE_OUTPUT',
            'TOPROVAR_KERNEL_DIFF_PROBE_MAX_RECORDS',
            'TOPROVAR_KERNEL_DIFF_PROBE_USE_PLAIN_OUTPUT',
        ):
            os.environ.pop(key, None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path
    if args.checkpoint_type == 'torch': 
        if osp.exists(args.cache_dir):
            local_model_path = osp.join(args.cache_dir, 'tmp', model_path.replace('/', '_'))
        else:
            local_model_path = model_path
        slim_model_path = model_path.replace('ar-', 'slim-')
        print(f'load checkpoint from {slim_model_path}')
    elif args.checkpoint_type == 'torch_shard':
        slim_model_path = model_path

    if args.model_type == 'infinity_2b':
        kwargs_model = dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8) # 2b model
    elif args.model_type == 'infinity_8b':
        kwargs_model = dict(depth=40, embed_dim=3584, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
    elif args.model_type == 'infinity_layer12':
        kwargs_model = dict(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer16':
        kwargs_model = dict(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer24':
        kwargs_model = dict(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer32':
        kwargs_model = dict(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer40':
        kwargs_model = dict(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer48':
        kwargs_model = dict(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    infinity = load_infinity(
        rope2d_each_sa_layer=args.rope2d_each_sa_layer, 
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        use_scale_schedule_embedding=args.use_scale_schedule_embedding,
        pn=args.pn,
        use_bit_label=args.use_bit_label, 
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block, 
        model_path=slim_model_path, 
        scale_schedule=None, 
        vae=vae, 
        device=device, 
        model_kwargs=kwargs_model,
        text_channels=args.text_channels,
        apply_spatial_patchify=args.apply_spatial_patchify,
        use_flex_attn=args.use_flex_attn,
        bf16=args.bf16,
        checkpoint_type=args.checkpoint_type,
        only_scale_skip=args.only_scale_skip,
        use_adaptive_scale=args.use_adaptive_scale,
        scale_prop=args.scale_prop,
        entropy_boundary_line=args.entropy_boundary_line,
        start_scale_index=args.start_scale_index,
        direct_skip_start_scale_index=args.direct_skip_start_scale_index,
        use_svd_depart_layers=args.use_svd_depart_layers,
        structure_layer_ration=args.structure_layer_ration,
        use_token_pruning=args.use_token_pruning,
        svd_rho_threshold=args.svd_rho_threshold,
        svd_source_scale_index=args.svd_source_scale_index,
        svd_split_mode=args.svd_split_mode,
        protected_global_layers=args.protected_global_layers,
        adaptive_entropy_s_min=args.adaptive_entropy_s_min,
        adaptive_entropy_collect_start_scale_index=args.adaptive_entropy_collect_start_scale_index,
        adaptive_entropy_trigger_start_scale_index=args.adaptive_entropy_trigger_start_scale_index,
        adaptive_entropy_gamma=args.adaptive_entropy_gamma,
        adaptive_entropy_epsilon=args.adaptive_entropy_epsilon,
        adaptive_entropy_lambda=args.adaptive_entropy_lambda,
        adaptive_entropy_score_mode=args.adaptive_entropy_score_mode,
        adaptive_entropy_signal_mode=args.adaptive_entropy_signal_mode,
        adaptive_entropy_signal_layer_idx=args.adaptive_entropy_signal_layer_idx,
        adaptive_max_raw_start_scale_index=args.adaptive_max_raw_start_scale_index,
        force_entropy_collection_scales=args.force_entropy_collection_scales,
    )
    return infinity

def add_common_arguments(parser):
    parser.add_argument('--cfg', type=str, default='3')
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--pn', type=str, required=True, choices=['0.06M', '0.25M', '1M'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cfg_insertion_layer', type=int, default=0)
    parser.add_argument('--vae_type', type=int, default=1)
    parser.add_argument('--vae_path', type=str, default='')
    parser.add_argument('--add_lvl_embeding_only_first_block', type=int, default=0, choices=[0,1])
    parser.add_argument('--use_bit_label', type=int, default=1, choices=[0,1])
    parser.add_argument('--model_type', type=str, default='infinity_2b')
    parser.add_argument('--rope2d_each_sa_layer', type=int, default=1, choices=[0,1])
    parser.add_argument('--rope2d_normalized_by_hw', type=int, default=2, choices=[0,1,2])
    parser.add_argument('--use_scale_schedule_embedding', type=int, default=0, choices=[0,1])
    parser.add_argument('--sampling_per_bits', type=int, default=1, choices=[1,2,4,8,16])
    parser.add_argument('--text_encoder_ckpt', type=str, default='')
    parser.add_argument('--text_channels', type=int, default=2048)
    parser.add_argument('--apply_spatial_patchify', type=int, default=0, choices=[0,1])
    parser.add_argument('--h_div_w_template', type=float, default=1.000)
    parser.add_argument('--use_flex_attn', type=int, default=0, choices=[0,1])
    parser.add_argument('--enable_positive_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--cache_dir', type=str, default='/dev/shm')
    parser.add_argument('--checkpoint_type', type=str, default='torch')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bf16', type=int, default=1, choices=[0,1])

    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--metadata_file', type=str, default='')
    parser.add_argument('--rewrite_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--load_rewrite_prompt_cache', type=int, default=1, choices=[0,1])
    
    parser.add_argument('--start_scale_index', type=int, default=0, choices=[1, 2, 3,4, 5, 6, 7, 8, 9, 10, 11, 12,100])
    parser.add_argument('--direct_skip_start_scale_index', type=int, default=100, choices=[1, 2, 3,4, 5, 6, 7, 8, 9, 10, 11, 12,100])
    parser.add_argument('--only_scale_skip', action='store_true')
    parser.add_argument('--use_adaptive_scale', action='store_true')
    parser.add_argument('--use_svd_depart_layers', action='store_true')
    parser.add_argument('--use_token_pruning', action='store_true')
    parser.add_argument('--scale_prop', type=float, default=1.000)
    parser.add_argument('--structure_layer_ration', type=float, default=1.000)
    parser.add_argument('--entropy_boundary_line', type=float, default=1.000)
    parser.add_argument('--svd_rho_threshold', type=float, default=-1.000)
    parser.add_argument('--svd_source_scale_index', type=int, default=7)
    parser.add_argument('--svd_split_mode', type=str.lower, default='ratio', choices=['ratio', 'threshold', 'gap_raw', 'gap_log', 'gap_prior'])
    parser.add_argument('--protected_global_layers', type=parse_protected_global_layers, default=[])
    parser.add_argument('--adaptive_entropy_s_min', type=int, default=None)
    parser.add_argument('--adaptive_entropy_collect_start_scale_index', type=int, default=7)
    parser.add_argument('--adaptive_entropy_trigger_start_scale_index', type=int, default=8)
    parser.add_argument('--adaptive_entropy_gamma', type=float, default=1.4e-6)
    parser.add_argument('--adaptive_entropy_epsilon', type=float, default=1e-5)
    parser.add_argument('--adaptive_entropy_lambda', type=float, default=1.0)
    parser.add_argument('--adaptive_entropy_score_mode', type=str.lower, default='delta_h_width_power', choices=['delta_h_width_power', 'legacy_delta_cost_ratio'])
    parser.add_argument('--adaptive_entropy_signal_mode', type=str.lower, default='layer_mean_delta', choices=['layer_mean_delta', 'global_mean_delta'])
    parser.add_argument('--adaptive_entropy_signal_layer_idx', type=int, default=5)
    parser.add_argument('--adaptive_max_raw_start_scale_index', type=int, default=10)
    parser.add_argument('--force_entropy_collection_scales', type=str, default='')
    parser.add_argument('--kernel_diff_probe', action='store_true')
    parser.add_argument('--kernel_diff_probe_scales', type=str, default='6,7,8,9')
    parser.add_argument('--kernel_diff_probe_layers', type=str, default='0,10,20,39')
    parser.add_argument('--kernel_diff_probe_output', type=str, default='')
    parser.add_argument('--kernel_diff_probe_max_records', type=int, default=256)
    parser.add_argument('--kernel_diff_probe_use_plain_output', type=int, default=1, choices=[0, 1])
    parser.add_argument('--oracle_start_map', type=str, default='')


    # Optional prompt directory used by evaluation wrappers.
    parser.add_argument('--prompt_dir', type=str, default='None')

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--prompt', type=str, default='a dog')
    parser.add_argument('--save_file', type=str, default='./tmp.jpg')
    args = parser.parse_args()

    # parse cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
    
    # load text encoder
    text_tokenizer, text_encoder = load_tokenizer(t5_path =args.text_encoder_ckpt)
    # load vae
    vae = load_visual_tokenizer(args)
    # load infinity
    infinity = load_transformer(vae, args)
    
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]['scales']
    scale_schedule = [ (1, h, w) for (_, h, w) in scale_schedule]

    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            generated_image = gen_one_img(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                args.prompt,
                g_seed=args.seed,
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=args.cfg,
                tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=args.enable_positive_prompt,
            )
    os.makedirs(osp.dirname(osp.abspath(args.save_file)), exist_ok=True)
    cv2.imwrite(args.save_file, generated_image.cpu().numpy())
    print(f'Save to {osp.abspath(args.save_file)}')
