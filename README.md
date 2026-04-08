# SIGDM_my

> A diffusion-powered motion generation playground that unifies text-to-motion, action-to-motion, and speech-driven gesture pipelines in one research repo.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#quickstart)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C?logo=pytorch&logoColor=white)](#quickstart)
[![Status](https://img.shields.io/badge/Project-Active-0A7F2E)](#)

## Paper-Style Snapshot

### Abstract

Human motion generation requires modeling long-range temporal dynamics, semantic controllability, and physically plausible body behavior. This repository studies a diffusion-based framework for controllable motion synthesis under multiple conditions, including text prompts, action labels, and speech/audio signals. The project integrates unified training and sampling workflows, reusable preprocessing scripts, and standardized evaluation entry points across several motion benchmarks. In addition to text/action-conditioned generation, it provides an audio-driven gesture branch for speech-to-motion research. The codebase is designed for reproducible experimentation and rapid ablation in practical research pipelines.

### Contributions (Template)

- A unified diffusion framework that supports text-conditioned, action-conditioned, and audio-conditioned motion generation.
- A practical research codebase with end-to-end scripts for data preparation, training, sampling, and evaluation.
- A modular architecture that enables controlled ablations across conditioning pathways and datasets.
- An extensible branch for speech-driven gesture generation under `mydiffusion_zeggs/`.

### Experimental Setup (Template)

| Item | Recommended Description Template |
|---|---|
| Datasets | HumanML3D, KIT, HumanAct12, UESTC (specify split protocol) |
| Motion Representation | e.g., joint rotations / xyz positions / hml vectors |
| Backbone | e.g., Transformer encoder with diffusion denoiser |
| Diffusion Steps | e.g., 1000 |
| Noise Schedule | e.g., cosine |
| Training Steps | e.g., 600k |
| Batch Size | e.g., 64 |
| Guidance Scale | e.g., 2.5 for CFG sampling |
| Hardware | e.g., 1 to 8 GPUs, model-specific VRAM |
| Reproducibility | fixed seed, saved args.json, checkpoint interval |

### Main Results (Table Template)

Fill with your reported numbers. Add or remove metrics according to your benchmark protocol.

| Method | Dataset | R-Precision (Top-1) | FID | Diversity | MM-Dist | Notes |
|---|---|---:|---:|---:|---:|---|
| Baseline-A | HumanML3D | TBD | TBD | TBD | TBD | Replace with citation |
| Baseline-B | HumanML3D | TBD | TBD | TBD | TBD | Replace with citation |
| SIGDM_my | HumanML3D | TBD | TBD | TBD | TBD | Ours |
| SIGDM_my | KIT | TBD | TBD | TBD | TBD | Ours |

### Ablation Study (Table Template)

| Variant | Text Cond. | Action Cond. | Audio Cond. | CFG | Top-1 | FID | Diversity |
|---|---|---|---|---|---:|---:|---:|
| Full Model | Yes | Yes | Yes | Yes | TBD | TBD | TBD |
| w/o CFG | Yes | Yes | Yes | No | TBD | TBD | TBD |
| w/o Audio Branch | Yes | Yes | No | Yes | TBD | TBD | TBD |
| Lightweight Backbone | Yes | Yes | Yes | Yes | TBD | TBD | TBD |

### Qualitative Gallery (Template)

- Text-to-motion samples: add GIF/MP4 grids and prompt lists.
- Action-to-motion samples: add per-class visualizations.
- Audio-to-gesture samples: add synchronized audio + motion clips.

Recommended layout:

| Scenario | Prompt / Input | Visualization |
|---|---|---|
| T2M-01 | "A person turns and points forward." | Add GIF/MP4 link |
| A2M-01 | "Jump" | Add GIF/MP4 link |
| S2G-01 | Speech segment #12 | Add GIF/MP4 link |

---

## Why This Repository

`SIGDM_my` extends diffusion-based human motion generation with practical scripts for:

- Text-conditioned motion synthesis
- Action-conditioned motion synthesis
- Speech/audio-driven gesture generation
- Multi-dataset experimentation and evaluation

If you are building next-generation controllable motion generation systems, this repo gives you a strong, hackable base.

---

## Core Features

- Unified training and sampling flows for multiple conditioning modes
- Support for `HumanML3D`, `KIT`, `HumanAct12`, and `UESTC`-style workflows
- Built-in preparation scripts for key assets and evaluation dependencies
- Evaluation modules for constrained and unconstrained settings
- Extra audio2pose branch under `mydiffusion_zeggs/` for speech-to-gesture research

---

## Project Layout

```text
main/
|- assets/                 # Prompt/action examples
|- data_loaders/           # Dataset readers and collators
|- dataset/                # Dataset resources and statistics
|- diffusion/              # Diffusion process implementation
|- eval/                   # Evaluation scripts and metrics
|- model/                  # MDM-style model definitions
|- mydiffusion_zeggs/      # Audio/gesture generation branch
|- prepare/                # One-shot download helpers
|- process/                # BVH and dataset preprocessing utilities
|- sample/                 # Generation and editing entry points
|- train/                  # Training entry points and loops
|- utils/                  # Parsers, configuration, and helpers
```

---

## Quickstart

### 1) Clone

```bash
git clone https://github.com/bbd66/SIGDM_my.git
cd SIGDM_my/main
```

### 2) Create Environment

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
```

### 3) Install Dependencies

This repo does not currently expose a single root `requirements.txt`, so install your stack manually (or adapt from your current environment).

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install numpy scipy tqdm matplotlib gdown tensorboard
```

Optional but commonly needed:

- `ffmpeg` for video rendering in sampling scripts
- CUDA-enabled PyTorch build for training speed

---

## Data and Asset Preparation

From `main/`, run the helper scripts you need:

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_a2m_datasets.sh
bash prepare/download_unconstrained_datasets.sh
bash prepare/download_t2m_evaluators.sh
bash prepare/download_recognition_models.sh
bash prepare/download_recognition_unconstrained_models.sh
```

Notes:

- These scripts rely on `gdown` and downloadable archives.
- Dataset placement conventions are described in `dataset/README.md`.

---

## Training

### Standard MDM-style training

```bash
python train/train_mdm.py \
  --save_dir save/humanml_exp01 \
  --dataset humanml \
  --device 0 \
  --batch_size 64 \
  --num_steps 600000 \
  --overwrite
```

### Custom training entry

```bash
python train/mytrain.py \
  --save_dir save/my_custom_exp \
  --dataset kit \
  --device 0 \
  --overwrite
```

---

## Sampling

Generate motions from a checkpoint:

```bash
python sample/generate.py \
  --model_path save/humanml_exp01/model000500000.pt \
  --dataset humanml \
  --device 0 \
  --text_prompt "a person walks forward and waves"
```

Batch prompt file mode:

```bash
python sample/generate.py \
  --model_path save/humanml_exp01/model000500000.pt \
  --dataset humanml \
  --device 0 \
  --input_text assets/example_text_prompts.txt
```

Motion editing is available via:

```bash
python sample/edit.py --help
```

---

## Evaluation

```bash
python eval/eval_humanml.py \
  --model_path save/humanml_exp01/model000500000.pt \
  --device 0
```

For action datasets:

```bash
python eval/eval_humanact12_uestc.py \
  --model_path save/a2m_exp01/model000300000.pt \
  --dataset humanact12 \
  --device 0
```

---

## Audio-to-Gesture Branch

The `mydiffusion_zeggs/` directory contains an additional speech-driven gesture pipeline, configs, and data tools for beat-style and end-to-end experiments.

Start from:

- `mydiffusion_zeggs/end2end.py`
- `mydiffusion_zeggs/sample_DG2EST.py`
- `mydiffusion_zeggs/configs/`

---

## Repro Tips

- Fix random seeds with existing utilities in `utils/fixseed.py`.
- Save full argument snapshots (`args.json`) for each run.
- Keep one checkpoint directory per experiment variant.
- Record dataset version and preprocessing command history.

---

## Common Pitfalls

- Missing `args.json` next to checkpoints will break generation/evaluation argument loading.
- Large `num_samples` with small GPU memory can OOM during sampling.
- Missing `ffmpeg` prevents final merged video artifacts.
- Incomplete dataset downloads cause silent shape/index mismatches later.

---

## Roadmap

- Add a root-level lockfile/`requirements.txt` for one-command environment setup
- Add experiment cards with metrics and qualitative galleries
- Add CI checks for import and script sanity
- Add Docker support for reproducible onboarding

---

## Citation

If this repository contributes to your research, please cite the original upstream projects and this repo once a formal paper/report entry is available.

---

## Acknowledgements

This codebase builds on and adapts ideas/components from open-source diffusion and motion-generation communities, including guided diffusion, text-to-motion, and MDM-related implementations.
