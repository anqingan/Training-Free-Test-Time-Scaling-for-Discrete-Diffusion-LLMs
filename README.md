# Two-Scale MTM for Discrete Diffusion LLMs

Official method-focused code release for the paper:

> **Training-Free Test-Time Scaling for Discrete Diffusion LLMs via Two-Scale Parallel Group Refinement**

This repository provides a compact academic release centered on the decoder-side refinement mechanism introduced in the paper. The current version is intentionally streamlined: it retains the core implementation paths needed to inspect the proposed Two-Scale MTM decoding strategy, while excluding peripheral tooling from the larger internal research workspace.

## Overview

The repository is organized around the practical implementation of training-free test-time scaling for discrete diffusion language models. The released code emphasizes the main algorithmic path rather than a full end-to-end framework. In particular, it keeps:

- the text/code decoding path for the proposed Two-Scale MTM refinement;
- the multimodal decoding entrypoint used to mirror the LLaDA-V setting;
- the local LLaDA model wrapper required by the retained inference path; and
- minimal configuration files for understanding the intended setup.

## Highlights

- Training-free decoder-side refinement for discrete diffusion language models
- Two-scale proposal updates spanning local and broader revision scopes
- Compact, paper-oriented release with the method logic exposed directly
- Minimal repository surface for easier inspection and reuse in academic settings

## Repository Layout

```text
.
+-- configs/
|   +-- llada_twoscale_eval.yaml
|   `-- lladav_twoscale_eval.yaml
+-- sample/
|   +-- the_new_llada_sample.py
|   +-- lladav_sample.py
|   `-- llada/
|       +-- configuration_llada.py
|       `-- modeling_llada.py
+-- requirements.txt
`-- README.md
```

## Core Files

If you only want to inspect the method implementation, start from the following files:

- `sample/the_new_llada_sample.py`
  Main implementation of the text/code decoding path with the retained Two-Scale MTM logic.
- `sample/lladav_sample.py`
  Multimodal decoding entrypoint kept as a lightweight reference path.
- `sample/llada/modeling_llada.py`
  Local LLaDA model wrapper used by the retained decoding pipeline.
- `sample/llada/configuration_llada.py`
  LLaDA configuration definition required by the local model wrapper.

## Configuration

Two minimal configuration files are kept in `configs/`:

- `configs/llada_twoscale_eval.yaml` for the text/code setting
- `configs/lladav_twoscale_eval.yaml` for the multimodal setting

These files are preserved primarily as reference configurations for the released method path.

## Installation

The original project environment was based on Python 3.10.

```bash
pip install -r requirements.txt
```

Model checkpoints are not included in this repository and need to be prepared separately.

## Release Scope

This repository should be viewed as a paper code release rather than a full training or benchmarking framework. Non-core evaluation drivers, reward modules, visualization utilities, and unrelated support code were intentionally omitted to keep the method implementation focused and easier to audit.

## Notes

- The repository is intentionally lean and prioritizes code clarity over completeness of the original workspace.
- Some auxiliary components from the larger project were removed during curation.
- The remaining code is intended to present the core refinement logic in a concise form suitable for academic release.
