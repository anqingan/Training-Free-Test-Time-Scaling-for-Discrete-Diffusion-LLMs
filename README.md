# Training-Free Test-Time Scaling for Discrete Diffusion LLMs

This repository contains a curated code subset for the paper **"Training-Free Test-Time Scaling for Discrete Diffusion LLMs via Two-Scale Parallel Group Refinement."**

The code was extracted and cleaned from a larger internal `dLLM-RL-main` workspace. Only the paper-related inference, evaluation, and model-adaptation components are kept here. Training code, RL/SFT pipelines, datasets, figures, PDFs, and unrelated model backends have been removed.

## Included Components

- `sample/the_new_llada_sample.py`: main text/code inference entry for the paper's LLaDA-based refinement pipeline
- `sample/lladav_sample.py`: multimodal sampling entry used for the LLaDA-V evaluation path
- `eval.py`: text/code evaluation driver
- `eval_v.py`: multimodal evaluation driver
- `reward/`: metric and post-processing utilities
- `configs/`: minimal config templates for text/code and multimodal evaluation
- `sample/llada/` and `sample/llava/`: local model code required by the kept inference paths

## Not Included

- RL training and value-model code
- SFT pipelines
- Multi-node orchestration
- Unrelated diffusion backends such as Dream, SDAR, and TraDo
- Datasets, checkpoints, benchmark images, and paper assets

## Environment

The original project used Python 3.10. Install dependencies with:

```bash
pip install -r requirements.txt
```

You will also need the corresponding LLaDA or LLaDA-V checkpoints available locally.

## Text / Code Evaluation

Edit `configs/the_new_llada_eval.yaml` first:

- set `model` to your local LLaDA checkpoint path
- set the dataset fields you want to evaluate
- adjust rollout hyperparameters as needed

Run:

```bash
python eval.py config=configs/the_new_llada_eval.yaml
```

This curated repo routes `eval.py` directly to `sample/the_new_llada_sample.py`, which is the retained paper-specific text/code inference entry.

## Multimodal Evaluation

Edit `configs/lladav_eval.yaml` first:

- set `model` to your local LLaDA-V checkpoint path
- set `image_root` to the benchmark image directory
- confirm the `lmms_task` and `lmms_task_dir` settings for your local setup

Run:

```bash
python eval_v.py config=configs/lladav_eval.yaml
```

## Note on `lmms-eval`

The multimodal evaluation path depends on a local `lmms-eval` checkout compatible with the original LLaDA-V evaluation stack. By default, `lmms_eval_adapter.py` expects it under:

```text
LLaDA-V-upstream/eval/lmms-eval
```

If your checkout lives elsewhere, update the path in `configs/lladav_eval.yaml` or adjust `lmms_eval_adapter.py` accordingly.

## Outputs

Evaluation outputs are written under the experiment directory defined in the config, typically including:

- `temp_data/outputs-*.json`
- `results/results-*.txt`

## Scope

This repository is intentionally narrow. It is meant to host the paper-related decoding and evaluation code in a cleaner form, not to preserve the full upstream research framework.
