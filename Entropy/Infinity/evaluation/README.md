# Evaluation

This directory provides lightweight paper-code entry points for the benchmarks reported in the paper. The main supported public workflow for this release is ToProVAR inference; full benchmark reproduction depends on the official benchmark projects, model weights, and local evaluation environments.

## Basic Configuration

Most evaluation scripts follow the same Infinity runtime assumptions as inference. Adjust these variables for your environment before running benchmark commands:

```shell
pn=1M
model_type=infinity_8b
infinity_model_path=/data/models/infinity-8b
out_dir_root=outputs/evaluation
vae_type=14
vae_path=/data/models/infinity-2b/infinity_vae_d56_f8_14_patchify.pth
cfg=4
tau=1
text_encoder_ckpt=/data/models/flan-t5-xl
text_channels=2048
sub_fix=cfg${cfg}_tau${tau}
```

Optional evaluation-only Python packages are listed in the repository root:

```shell
python -m pip install -r requirements-eval.txt
```

## GenEval

GenEval is an object-focused framework for evaluating text-to-image alignment. The repository includes the generation wrapper, evaluation metadata, object-name metadata, and scoring helpers used by the paper-code workflow.

```shell
bash Entropy/Infinity/scripts/run-geneval-8b.sh
bash Entropy/Infinity/scripts/score-geneval.sh
```

The `gen_eval/vendor_mmdet2` directory is intentionally not tracked. Prepare official GenEval/MMDetection dependencies in your environment, or place a local `vendor_mmdet2` checkout there for private paper-code reproduction.

## HPSv2.1

HPSv2.1 evaluates human preference alignment for generated images.

```shell
python Entropy/Infinity/evaluation/hpsv2/eval_hpsv2.py
```

Prepare the HPSv2.1 package and any required model files according to the official HPSv2 project.

## ImageReward

ImageReward evaluates human preference for text-to-image generations.

```shell
python Entropy/Infinity/evaluation/image_reward/infer4eval.py
python Entropy/Infinity/evaluation/image_reward/cal_imagereward.py
```

Prepare ImageReward and its dependencies according to the official ImageReward project.

## MJHQ30K / FID / CLIP

MJHQ30K FID / CLIP evaluation utilities are provided as paper-code references.

```shell
python Entropy/Infinity/evaluation/mjhq30k_fid_clip.py
```

Prepare the MJHQ30K data and any reference images outside this repository.

## Speed Benchmark

Speed benchmarking utilities are available for local analysis of ToProVAR inference behavior.

```shell
python Entropy/Infinity/tools/bench_toprovar_one_mode.py --help
python Entropy/Infinity/tools/bench_toprovar_speed.py --help
```

Adaptive-scale probes and plotting scripts under `Entropy/Infinity/tools/` are research utilities and are not part of the minimal public inference quickstart.
