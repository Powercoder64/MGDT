# MGDT: Multigrid Anisotropic Diffusion Transformer


MGDT replaces softmax self-attention with **edge-preserving anisotropic diffusion on a sparse token graph**,
solved with a **multigrid V-cycle**. Similar tokens exchange information through high-conductance edges while
dissimilar tokens are separated by low-conductance boundaries; coarse graphs propagate global context and fine
graphs refine local detail. The layer is a drop-in replacement for attention inside standard Transformer blocks.

This repository contains the full model, training and evaluation code and configurations for all benchmarks in
the paper (ImageNet-1K, Kinetics-400/700, ADE20K, Cityscapes, COCO Captions, Ego4D-NLQ and LRA-Text), the
ablation studies, the runtime/memory scaling benchmark, the toy diffusion visualisation and FLOP counting.

---

## Contents

- [Method in code](#method-in-code)
- [Repository structure](#repository-structure)
- [Prerequisites and installation](#prerequisites-and-installation)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Training recipes](#training-recipes)
- [Ablation studies](#ablation-studies)
- [Runtime and memory scaling](#runtime-and-memory-scaling)
- [Toy diffusion visualisation](#toy-diffusion-visualisation)
- [Parameters and FLOPs](#parameters-and-flops)
- [Repeated runs](#repeated-runs)
- [Reproducibility](#reproducibility)
- [Tests](#tests)
- [Citation](#citation)
- [License and acknowledgement](#license-and-acknowledgement)

---

## Method in code

Every MGDT block is `x + DropPath(MGDT(LN(x)))` followed by `x + DropPath(MLP(LN(x)))`. One MGDT token mixer
(`mgdt/layers/mgdt_mixer.py`) performs, per head:

| Step | Implementation | Code |
|---|---|---|
| Token graph | local spatial/temporal grid neighbours plus the top-`k` neighbours in projected key space (`k = 16` for image, segmentation, captioning and text; `k = 32` for video and temporal grounding); duplicates and self-loops removed | `mgdt/ops/graph.py`, `mgdt/ops/distance.py` |
| Conductance | `g(s) = exp(-(s/κ)²)`, `s_ij = ‖φ(x_i) − φ(x_j)‖₂`; φ is the key projection (a small shared projector for captioning and grounding); key norms clipped at 5.0; κ learned per head and linearly warmed up; symmetrised `(W + Wᵀ)/2`; conductances below `1e-4` dropped | `mgdt/ops/conductance.py` |
| Smoothing | degree-normalised explicit Euler diffusion `u ← u + τ D⁻¹(W u − D u)` with `τ = 0.5` | `mgdt/ops/multigrid.py` |
| Coarsening | heavy-edge matching (parallel handshake matching, 2× per level), restriction by (mass-weighted) averaging, coarse conductances averaged over cluster pairs, Galerkin-consistent coarse operator | `mgdt/ops/coarsening.py` |
| V-cycle | 1 pre-smoothing step → residual restriction → coarse correction (recursive, coarsest level solved by a few diffusion iterations) → prolongation → 1 post-smoothing step; `L = 1` level for ImageNet-1K, ADE20K, COCO and LRA, `L = 2` for Kinetics, Cityscapes and Ego4D-NLQ | `mgdt/ops/multigrid.py` |
| Precision | projections run under mixed precision; graph construction, conductances and all diffusion/residual reductions run in FP32 | `mgdt/layers/mgdt_mixer.py` |

Heads are concatenated and linearly projected, exactly as in multi-head attention. Self-interactions are carried
by the residual path. Variants used in the ablations (isotropic / rational / hard-threshold conductance, fixed or
per-layer κ, k-means coarsening, 4× coarsening, query / shared / `[q;k]` / asymmetric similarity spaces, cosine
distance, unnormalised updates, softmax attention) are configuration options of the same mixer.

---

## Repository structure

```
MGDT/
├── configs/
│   ├── imagenet1k.yaml        kinetics400.yaml   kinetics700.yaml
│   ├── ade20k.yaml            cityscapes.yaml    coco_captions.yaml
│   ├── ego4d_nlq.yaml         lra_text.yaml
│   └── ablations/             mechanism, multigrid, conductance, similarity, step_size, flop_matched
├── mgdt/
│   ├── ops/                   sparse graphs, kNN/edge distances, conductance, coarsening, V-cycle
│   ├── layers/                MGDT mixer, softmax attention, Transformer block
│   ├── models/                image backbone, video model, segmenter, captioner, grounding head, text model
│   ├── data/                  datasets and transforms for every benchmark
│   ├── metrics/               top-k, mIoU + boundary F-score, CIDEr-D, temporal-grounding recall
│   ├── engine/                trainer, tasks, optimiser/schedules, EMA, checkpoints, distributed helpers
│   └── utils/                 configs, seeding/logging, FLOP accounting
├── tools/
│   ├── train.py               evaluate.py             count_flops.py
│   ├── benchmark_scaling.py   visualize_diffusion.py  run_ablation.py
│   ├── prepare_kinetics.py    extract_video_features.py
│   └── run_seeds.sh           aggregate_results.py
├── tests/                     unit tests and end-to-end pipeline tests on synthetic data
├── requirements.txt  pyproject.toml  CITATION.cff  LICENSE
```

---

## Prerequisites and installation

**Hardware.** NVIDIA GPU(s) with CUDA for training (the paper uses Linux servers with NVIDIA RTX 3090 24 GB GPUs).
Everything also runs on CPU, which is used by the test suite.

**Software.**

| Requirement | Version |
|---|---|
| Python | ≥ 3.9 (paper environment: 3.10) |
| PyTorch | ≥ 2.1 (paper environment: 2.2, CUDA 12.1, cuDNN 8.9, NCCL 2.18) |
| torchvision | ≥ 0.16 (paper environment: 0.17) |
| timm | ≥ 0.9 (ImageNet RandAugment, random erasing, Mixup/CutMix) |
| PyAV | ≥ 10 (video decoding; bundles FFmpeg) |
| PyYAML, NumPy, Pillow, matplotlib | recent versions |

From the repository root:
```bash
conda create -n mgdt python=3.10 -y && conda activate mgdt
# Install the PyTorch build matching your CUDA driver first (see https://pytorch.org), for example:
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .            # optional: makes `import mgdt` available everywhere
```

All commands below are run from the repository root. Default dataset paths live under `data/` and every path can
be overridden on the command line (`--opts data.root=/path/to/dataset`).

---

## Data preparation

### ImageNet-1K
Download ILSVRC-2012 from [image-net.org](https://image-net.org) and arrange train and validation images in class
folders (the validation set must be sorted into the 1,000 class folders):
```
data/imagenet/
├── train/n01440764/*.JPEG ...
└── val/n01440764/*.JPEG ...
```

### Kinetics-400 / Kinetics-700
Download the clips and the official annotation CSVs (columns `label,youtube_id,time_start,time_end,split`), e.g. from
the [CVDF Kinetics release](https://github.com/cvdfoundation/kinetics-dataset). Clips named
`<youtube_id>_<start:06d>_<end:06d>.mp4` (or starting with the YouTube id) are matched automatically, in any folder
structure under `--video-root`. Create the training and validation lists (the class list is created from the
training annotations and reused for validation):
```bash
python tools/prepare_kinetics.py --annotations data/kinetics400/annotations/train.csv \
    --video-root data/kinetics400/videos --output data/kinetics400/train.csv --classes data/kinetics400/classes.txt
python tools/prepare_kinetics.py --annotations data/kinetics400/annotations/validate.csv \
    --video-root data/kinetics400/videos --output data/kinetics400/val.csv --classes data/kinetics400/classes.txt
```
Add `--check` to drop clips that cannot be decoded. Repeat with `kinetics700` paths for Kinetics-700. Decoding is
faster if clips are first re-encoded to a short side of 320 pixels, e.g.
`ffmpeg -i in.mp4 -vf "scale='if(gt(iw,ih),-2,320)':'if(gt(iw,ih),320,-2)'" -an out.mp4`.

### ADE20K
Download `ADEChallengeData2016.zip` from the [MIT Scene Parsing benchmark](http://sceneparsing.csail.mit.edu) and
extract it to `data/ADEChallengeData2016` (`images/{training,validation}`, `annotations/{training,validation}`).
Label 0 is ignored and classes 1–150 are trained.

### Cityscapes
Download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` from
[cityscapes-dataset.com](https://www.cityscapes-dataset.com) and extract both into `data/cityscapes`
(`leftImg8bit/{train,val}/<city>/`, `gtFine/{train,val}/<city>/`). The `*_gtFine_labelIds.png` files are mapped to
the 19 evaluation classes on the fly; no preprocessing script is needed.

### COCO Captions (Karpathy split)
Download `train2014.zip` and `val2014.zip` from [cocodataset.org](https://cocodataset.org) and the Karpathy split
file `dataset_coco.json` (in `caption_datasets.zip` from
[Karpathy's deep image–sentence page](https://cs.stanford.edu/people/karpathy/deepimagesent/)):
```
data/coco/
├── dataset_coco.json
└── images/{train2014,val2014}/*.jpg
```
Training uses the `train` + `restval` images (all five captions each); the vocabulary keeps words that occur at least
five times and is written to `<output_dir>/vocab.json`. Evaluation uses the 5,000 `test` images.

### Ego4D Natural Language Queries
1. Accept the [Ego4D license](https://ego4d-data.org) and download the canonical clips (`<clip_uid>.mp4`) and the
   annotations (`nlq_train.json`, `nlq_val.json`) with the Ego4D CLI into `data/ego4d` (see the Ego4D documentation
   for selecting the episodic-memory clips).
2. Train the Kinetics model (below) and extract clip features with its video encoder (one 768-d feature every
   16 frames at 30 fps, i.e. 1.875 features/s; each feature encodes 32 frames sampled with stride 2):
   ```bash
   python tools/extract_video_features.py --config configs/kinetics400.yaml \
       --checkpoint outputs/kinetics400/final.pth --video-dir data/ego4d/v2/clips \
       --annotations data/ego4d/v2/annotations/nlq_train.json data/ego4d/v2/annotations/nlq_val.json \
       --output-dir data/ego4d/mgdt_features --amp
   ```
   Use `--shard INDEX COUNT` to split the clips across several GPUs/jobs.
3. Point `data.train_json`, `data.val_json` and `data.feature_dir` of `configs/ego4d_nlq.yaml` to these files
   (defaults: `data/ego4d/annotations/nlq_{train,val}.json` and `data/ego4d/mgdt_features`).

### LRA Text
Download the IMDB reviews (`aclImdb_v1.tar.gz`) from [Stanford](https://ai.stanford.edu/~amaas/data/sentiment/)
and extract to `data/lra/aclImdb` (`train/{pos,neg}`, `test/{pos,neg}`). Reviews are encoded as UTF-8 bytes and
truncated/padded to 4,096 tokens.

---

## Training

Initialisation follows the paper: ImageNet-1K is trained from scratch; Kinetics, ADE20K, Cityscapes and COCO start
from the ImageNet-1K checkpoint (`model.pretrained`, EMA weights by default); Ego4D-NLQ uses features from the
Kinetics video encoder; LRA-Text is trained from scratch.

```bash
# ImageNet-1K (from scratch)
torchrun --nproc_per_node=8 tools/train.py --config configs/imagenet1k.yaml
# Kinetics-400 / 700, ADE20K, Cityscapes, COCO (initialised from outputs/imagenet1k/final.pth)
torchrun --nproc_per_node=8 tools/train.py --config configs/kinetics400.yaml
torchrun --nproc_per_node=8 tools/train.py --config configs/kinetics700.yaml
torchrun --nproc_per_node=8 tools/train.py --config configs/ade20k.yaml
torchrun --nproc_per_node=8 tools/train.py --config configs/cityscapes.yaml
torchrun --nproc_per_node=8 tools/train.py --config configs/coco_captions.yaml
# Ego4D-NLQ (after feature extraction) and LRA-Text
torchrun --nproc_per_node=8 tools/train.py --config configs/ego4d_nlq.yaml
python tools/train.py --config configs/lra_text.yaml
```

- **Batch sizes** in the configs are *global*. Each process uses `batch_size / (num_gpus × train.accum_steps)`;
  with fewer GPUs keep the recipe by adding gradient accumulation, e.g. `--opts train.accum_steps=2`.
  For Kinetics, `batch_size` counts clips (videos × `repeated_aug`).
- **Overrides**: any config entry can be changed with `--opts key=value` (values are parsed as YAML), e.g.
  `--opts model.mixer.levels=2 data.root=/datasets/imagenet`. `--output-dir` and `--seed` are shortcuts.
- **Outputs** (in `output_dir`): `config.yaml` (resolved config), `log.txt`, `metrics.jsonl` (every evaluation),
  `last.pth` (resumable state), `best.pth`, `final.pth` and `results.json` (final metrics, average step time and
  peak GPU memory). Interrupted runs resume automatically from `last.pth`.
- **Memory**: Kinetics and Cityscapes use activation checkpointing (`grad_checkpointing: true`); it can be enabled
  for any backbone if memory is tight.

---

## Evaluation

```bash
python tools/evaluate.py --config configs/imagenet1k.yaml --checkpoint outputs/imagenet1k/final.pth --ema
torchrun --nproc_per_node=8 tools/evaluate.py --config configs/kinetics400.yaml --checkpoint outputs/kinetics400/final.pth
```
Results are printed and written to `<output_dir>/eval_results.json` (`--output-dir` changes the location).
`--ema` evaluates the EMA weights stored in the checkpoint; `--track-energy` additionally reports the mean per-cycle
diffusion-energy reduction of the MGDT layers.

| Benchmark | Protocol | Metric |
|---|---|---|
| ImageNet-1K | 224×224 single centre crop (resize 256) | Top-1 / Top-5 |
| Kinetics-400 / 700 | 4 temporal clips × 3 spatial crops, 32 frames at 224², softmax averaging | Top-1 / Top-5 |
| ADE20K | single scale (short side 512), 512×512 sliding window, stride 341 | mIoU, aAcc, boundary F-score |
| Cityscapes | full-resolution 1024×2048 single forward pass, 19 classes | mIoU, aAcc, boundary F-score |
| COCO Captions | Karpathy test split, beam size 3, at most 20 words | CIDEr-D |
| Ego4D-NLQ | validation split, top-1 moment | R@1 at tIoU 0.3 / 0.5, mean tIoU |
| LRA-Text | 4,096 byte tokens, test split | Accuracy |

CIDEr-D is implemented in `mgdt/metrics/cider.py` and matches the `pycocoevalcap` scorer exactly on tokenised
captions; the Karpathy references are already tokenised and lower-cased. Generated captions are saved to
`<output_dir>/captions_pred.json`.

---

## Training recipes

All recipes use AdamW and mixed precision, the same update budget for matched baselines and ablations, and no extra
labelled data, pseudo-labels or distillation.

| Benchmark | Config | Recipe | Initialisation |
|---|---|---|---|
| ImageNet-1K | `imagenet1k.yaml` | 300 epochs; batch 1024; LR 1e-3, WD 0.05; cosine; 20-epoch warm-up; RandAugment, Mixup/CutMix, label smoothing 0.1, random erasing 0.25, stochastic depth, EMA | from scratch |
| Kinetics-400/700 | `kinetics400.yaml`, `kinetics700.yaml` | 120 epochs; batch 256; LR 5e-4, WD 0.05; cosine; 10-epoch warm-up; 32-frame 224² clips; spatial–temporal jitter, flip, repeated augmentation, dropout 0.1, stochastic depth | ImageNet-1K checkpoint |
| ADE20K | `ade20k.yaml` | 160k iterations; batch 16; LR 6e-5, WD 0.01; polynomial decay; 1.5k-iteration warm-up; random scale [0.5, 2.0], 512² crop, flip, dropout 0.1 | ImageNet-1K checkpoint |
| Cityscapes | `cityscapes.yaml` | 80k iterations; batch 8; LR 6e-5, WD 0.01; polynomial decay; 1.5k-iteration warm-up; random scale [0.5, 2.0], 1024² crop, flip | ImageNet-1K checkpoint |
| COCO Captions | `coco_captions.yaml` | 30 epochs; batch 512; LR 1e-4, WD 0.01; cosine; 5-epoch warm-up; random resized crop, flip, dropout 0.1; cross-entropy | ImageNet-1K encoder |
| Ego4D-NLQ | `ego4d_nlq.yaml` | 25 epochs; batch 64; LR 3e-4, WD 0.05; cosine; 3-epoch warm-up; temporal window sampling, temporal jitter, query dropout 0.1 | Kinetics video features |
| LRA-Text | `lra_text.yaml` | 50 epochs; batch 64; LR 3e-4, WD 0.01; cosine; 4k-step warm-up; dropout 0.1 | from scratch |

Shared MGDT settings: one V-cycle per layer, heavy-edge matching with 2× coarsening, one pre- and one post-smoothing
step, degree-normalised explicit Euler with τ = 0.5, κ learned per head and warmed up during the first 5 epochs
(first 5% of iterations for the iteration-based segmentation schedules), FP32 conductance and diffusion arithmetic.

---

## Ablation studies

Each file in `configs/ablations/` defines the benchmarks and variants of one ablation; `tools/run_ablation.py`
launches the runs (outputs in `outputs/ablations/<spec>/<benchmark>/<variant>/seed<k>`) and summarises them.

| Spec | Paper table | Benchmarks | Variants |
|---|---|---|---|
| `mechanism.yaml` | Table X | ImageNet-1K, ADE20K | anisotropic MGDT, isotropic diffusion (g ≡ 1), drop-in softmax attention |
| `multigrid.yaml` | Table XI | Kinetics-400, ADE20K | depth L ∈ {0,1,2,3}, heavy-edge matching vs k-means, 2× vs 4× coarsening (reports energy reduction) |
| `conductance.yaml` | Table XII | ADE20K, Kinetics-700 | rational / exponential g, fixed, per-layer or per-head κ, warm-up, hard threshold (reports Fb) |
| `similarity.yaml` | Table XIII | ImageNet-1K, COCO | key / query / shared φ / `[q;k]`→φ spaces, L2 vs cosine, symmetric vs asymmetric |
| `step_size.yaml` | Table XIV | ImageNet-1K, Kinetics-400 | τ ∈ {0.25, 0.5, 0.75}, smoothing 0/0, 1/1, 2/2, with/without degree normalisation (records divergence) |
| `flop_matched.yaml` | Table XV | ImageNet-1K, ADE20K | standard and FLOP-matched MGDT, FLOP-matched isotropic diffusion, softmax |

```bash
python tools/run_ablation.py --spec configs/ablations/mechanism.yaml --dry-run           # list the commands
python tools/run_ablation.py --spec configs/ablations/mechanism.yaml --benchmark imagenet1k --variant isotropic --nproc 8
python tools/run_ablation.py --spec configs/ablations/step_size.yaml --seeds 0 1 2 --nproc 8
python tools/run_ablation.py --spec configs/ablations/mechanism.yaml --summarize         # mean ± std per variant
```
The softmax variants reuse the key projection as query (`shared_qk: true`), so all mixers have identical parameter
counts. The unnormalised update scales the conductances by 1/(number of neighbour slots) instead of dividing by the
node degree. Summaries list every numeric entry of `results.json`, including the divergence rate over seeds, average
step time and peak memory used for relative time/memory and throughput comparisons.

---

## Runtime and memory scaling

Compares the complete MGDT layer (graph construction and every V-cycle operation included) with drop-in softmax
attention for growing sequence lengths, using the same width, heads, batch size and precision; each measurement is
averaged over five runs and out-of-memory cases are reported as OOM (paper Table XVI and Fig. 4):
```bash
python tools/benchmark_scaling.py --tokens 256 512 1024 2048 4096 8192 --batch-size 16 --amp \
    --output-dir outputs/scaling          # writes scaling.csv, scaling.md and scaling.png
```
Add `--backward` to time forward + backward passes.

---

## Toy diffusion visualisation

Reproduces the step-edge experiment of the paper (Fig. 3): a noisy 32×32 two-region image treated as scalar tokens
and processed by anisotropic diffusion, isotropic diffusion and softmax attention.
```bash
python tools/visualize_diffusion.py --output outputs/toy_step_edge.png
```

---

## Parameters and FLOPs

```bash
python tools/count_flops.py --config configs/kinetics400.yaml
```
FLOPs are counted as multiply–accumulates for the full forward pass under each evaluation protocol: token
projections, sparse graph construction, conductances, diffusion updates, restriction and prolongation, coarse
corrections, MLP blocks and task heads. Video FLOPs are reported per view and per video (all 12 views); captioning
FLOPs cover the encoder plus beam search for a full-length caption; ADE20K uses one 512×512 window and Cityscapes the
full 1024×2048 image. For Ego4D-NLQ the tool reports the grounding head on pre-extracted features.

---

## Repeated runs

```bash
bash tools/run_seeds.sh configs/ade20k.yaml 8      # seeds 0, 1, 2 on 8 GPUs, then mean ± std
python tools/aggregate_results.py outputs/seeds/ade20k/seed*/results.json
```

---

## Reproducibility

- The seed fixes model initialisation, data shuffling, augmentation randomness, dataloader workers and the
  tie-breaking of sparse-graph construction and heavy-edge matching. Evaluation uses deterministic data ordering
  and saved checkpoints.
- Distributed evaluation shards the data without padding, so every sample is counted exactly once.
- Training remains subject to standard GPU non-determinism from mixed precision, atomic scatter operations and
  distributed reductions; `deterministic: true` additionally selects deterministic cuDNN kernels.
- Runs whose loss stays non-finite for `train.max_nonfinite_steps` consecutive steps stop and are marked
  `"diverged": true` in `results.json`.

---

## Tests

```bash
pip install pytest pycocoevalcap      # pycocoevalcap is only needed for the CIDEr cross-check
python -m pytest tests
```
The unit tests check gradients of the sparse operators, the token graph against a dense reference, Galerkin
consistency of the coarse operators, energy reduction of the V-cycle, mass conservation, boundary preservation and
the metrics. The end-to-end tests build tiny synthetic datasets in each benchmark's on-disk layout and run training,
checkpointing, resuming and evaluation for every pipeline and tool on the CPU.

```

## License and acknowledgement

Released under the MIT License (see `LICENSE`). This material is partly supported by the National Science Foundation
under Grant No. 2322993.
