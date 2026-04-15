# Two-Scale MTM for Discrete Diffusion LLMs

Official inference and evaluation code for the paper:

> **Training-Free Test-Time Scaling for Discrete Diffusion LLMs via Two-Scale Parallel Group Refinement**

This repository provides a focused, paper-oriented release of the decoder-side refinement pipeline used in our study. It is intentionally lightweight: only the components required to run the proposed text/code and multimodal inference pipelines are kept here.

## Highlights

- Training-free test-time scaling for discrete diffusion language models
- Two-scale refinement with local and global proposal updates
- Support for both text/code reasoning and multimodal LLaDA-V evaluation
- Minimal paper release without RL/SFT training clutter

## What Is Included

- Paper inference entrypoints for LLaDA and LLaDA-V
- Evaluation drivers for text/code and multimodal benchmarks
- Minimal reward and result-aggregation utilities
- Local model adapters required by the retained inference paths
- Ready-to-edit config files for the paper release

## What Is Not Included

- Reinforcement learning training
- Supervised fine-tuning
- Multi-node orchestration
- Unrelated diffusion backends from the larger internal research workspace
- Datasets, checkpoints, figures, and paper PDFs

## Repository Layout

```text
.
+-- configs/
|   +-- llada_twoscale_eval.yaml
|   +-- lladav_twoscale_eval.yaml
|   +-- the_new_llada_eval.yaml
|   `-- lladav_eval.yaml
+-- reward/
|   +-- execute.py
|   +-- math_utils.py
|   +-- math_utils_v.py
|   +-- reward.py
|   `-- reward_v.py
+-- sample/
|   +-- the_new_llada_sample.py
|   +-- lladav_sample.py
|   +-- llada/
|   `-- llava/
+-- eval.py
+-- eval_v.py
+-- generate.py
+-- lmms_eval_adapter.py
`-- requirements.txt
```

## Core Files

If you only want to understand the method implementation, start here:

- [`sample/the_new_llada_sample.py`](sample/the_new_llada_sample.py)
  Main implementation of the paper's LLaDA text/code inference pipeline. This is the most important file in the repository.
- [`sample/lladav_sample.py`](sample/lladav_sample.py)
  Multimodal decoding path for LLaDA-V.
- [`eval.py`](eval.py)
  Text/code evaluation entrypoint.
- [`eval_v.py`](eval_v.py)
  Multimodal evaluation entrypoint.
- [`reward/reward.py`](reward/reward.py)
  Text/math result aggregation and scoring.
- [`reward/reward_v.py`](reward/reward_v.py)
  Multimodal result aggregation and scoring.

## Installation

The original environment used Python 3.10.

```bash
pip install -r requirements.txt
```

You also need local checkpoints for the corresponding LLaDA or LLaDA-V model family.

## Quick Start

### Text / Code Evaluation

Edit [`configs/llada_twoscale_eval.yaml`](configs/llada_twoscale_eval.yaml):

- set `model` to your local LLaDA checkpoint
- set `dataset.eval_dataset` and `dataset.data_type`
- adjust rollout hyperparameters if needed

Then run:

```bash
python eval.py config=configs/llada_twoscale_eval.yaml
```

### Multimodal Evaluation

Edit [`configs/lladav_twoscale_eval.yaml`](configs/lladav_twoscale_eval.yaml):

- set `model` to your local LLaDA-V checkpoint
- set `image_root` to the benchmark image directory
- check the `lmms_task` and local `lmms-eval` path

Then run:

```bash
python eval_v.py config=configs/lladav_twoscale_eval.yaml
```

## Outputs

Outputs are written under the experiment directory defined in the config, typically including:

- `temp_data/outputs-*.json`
- `results/results-*.txt`

## Notes on `lmms-eval`

The multimodal path depends on a local `lmms-eval` checkout compatible with the LLaDA-V evaluation stack. By default, [`lmms_eval_adapter.py`](lmms_eval_adapter.py) expects:

```text
LLaDA-V-upstream/eval/lmms-eval
```

Adjust the path in the config or in the adapter if your environment differs.

## Reproducibility Scope

This repository is a paper release, not a full framework release. The goal is to keep the code path for the proposed decoder-side refinement mechanism clear, compact, and easy to inspect.

## Acknowledgment

This repository was curated from a larger internal research workspace and reduced to the method-relevant components needed for this paper release.
