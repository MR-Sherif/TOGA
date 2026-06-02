# TOGA: Training-Only Heterogeneous Image-Patch-Text Graph Supervision for Advancing Few-Shot Learning Adapters

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue.svg)](https://cvpr.thecvf.com/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.18101-b31b1b.svg)](https://arxiv.org/abs/2603.18101)
[![PyTorch](https://img.shields.io/badge/PyTorch-Implementation-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#)

**Official PyTorch implementation of the CVPR 2026 paper: "Training-Only Heterogeneous Image-Patch-Text Graph Supervision for Advancing Few-Shot Learning Adapters"**

**Authors:** Mohammed Rahman Sherif Khan Mohammad, Ardhendu Behera, Sandip Pradhan, Swagat Kumar, Amr Ahmed  
**Institution:** Edge Hill University

---

## Abstract

Recent adapter-based CLIP tuning methods, such as Tip-Adapter, are strong few-shot learners that achieve efficiency by caching support features. However, these methods rely on global unimodal feature vectors, overlooking fine-grained patch relations and their structural alignment with class text.

**TOGA (Training-Only Graph Adapter)** bridges this gap through an asymmetric training-only framework. Instead of changing the lightweight inference adapter, TOGA introduces a high-capacity Modality-aware Graph Transformer (MGT) teacher that operates only during training.

Through a cache-aware dual-objective strategy, relational image-patch-text knowledge is supervised into the cache adapter. At test time, the graph teacher is discarded and inference follows the lightweight CLIP plus cache-adapter path.

## Key Contributions

* **Asymmetric Supervision:** A training-only graph supervision framework coupling a Tip-Adapter-style key-value cache with a high-capacity MGT teacher.
* **Modality-aware Graph Transformer (MGT):** Bi-modal visual-text reasoning over a unified patch-text graph with relation-aware message passing.
* **Discriminative Node Filtering:** High-fidelity class features are obtained by retaining informative foreground patches and suppressing background noise.
* **Cache-Aware Dual Objective:** A co-training strategy uses classification and teacher-forcing objectives to make the auxiliary graph teacher a robust training expert.
* **State-of-the-Art Results:** TOGA consistently improves over lightweight global-feature adapters and heavyweight patch-level adapters across 11 standard 1-16-shot benchmarks.

## Method Overview

![TOGA Architecture](Architecture_CR.png)

1. **Heterogeneous Graph Construction:** Multi-scale visual patches and class-text prompts are integrated into a unified graph topology.
2. **Cross-Modal Reasoning:** MGT performs relation-aware message passing across patch-patch and patch-text interactions.
3. **Discriminative Node Filtering:** Informative visual nodes are retained before graph-level readout.
4. **Low-Overhead Inference:** The graph teacher is used only during training; deployment uses the lightweight CLIP plus cache-adapter path.

## Installation

Create an environment with Python 3.10 or newer, then install the dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch and PyTorch Geometric builds that match your CUDA version if the generic packages do not match your machine.

## Datasets

TOGA follows the same 11-dataset benchmark protocol used by Tip-Adapter: ImageNet, Caltech101, Oxford Pets, Stanford Cars, Flowers102, Food101, FGVC-Aircraft, SUN397, DTD, EuroSAT, and UCF101.

Please prepare datasets following the official Tip-Adapter dataset instructions:

[Tip-Adapter DATASET.md](https://github.com/gaopengcuhk/Tip-Adapter/blob/main/DATASET.md)

Set `root_path` in the dataset YAML under `configs/`, or override it at launch:

```bash
python main.py --config ./configs/dtd.yaml --shots 1 --root_path /path/to/data
```

For ImageNet, the expected layout is:

```text
<root_path>/imagenet/images/train
<root_path>/imagenet/images/val
```

## Reproducing Results

Dataset- and shot-specific hyperparameters are stored in `configs/configs.yaml`. The main script automatically loads the matching preset from the dataset name and shot count.

```bash
python main.py --config ./configs/fgvc.yaml --shots 1
python main.py --config ./configs/eurosat.yaml --shots 16
python main.py --config ./configs/caltech101.yaml --shots 4
python main.py --config ./configs/imagenet.yaml --shots 16
```

Disable W&B logging for local checks:

```bash
python main.py --config ./configs/fgvc.yaml --shots 1 --wandb_mode disabled
```

Override preset values from the command line:

```bash
python main.py --config ./configs/ucf101.yaml --shots 8 --lr 0.0002 --pooling_ratio 0.5
```

On Windows, use a single-process loader for smoke tests:

```bash
python main.py --config ./configs/dtd.yaml --shots 1 --train_epoch 1 --wandb_mode disabled --num_workers 0
```

## Results Snapshot

TOGA establishes a new state of the art across 11 benchmark datasets: ImageNet, SUN397, FGVC-Aircraft, EuroSAT, Stanford Cars, Food101, Oxford Pets, Flowers102, Caltech101, DTD, and UCF101.

| Method | 1-Shot Avg | 2-Shot Avg | 4-Shot Avg | 8-Shot Avg | 16-Shot Avg | Test-Time Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Tip-Adapter-F | 64.3% | 66.1% | 69.1% | 73.3% | 75.8% | Low |
| GraphAdapter | 62.7% | 67.8% | 69.8% | 71.4% | 74.4% | High |
| **TOGA (Ours)** | **72.2%** | **75.0%** | **77.9%** | **80.0%** | **82.3%** | **Low** |

*(For full performance breakdowns and OOD generalization analysis, please refer to the main paper.)*

## Acknowledgements

This repository builds on the CLIP ecosystem and the Tip-Adapter codebase. We thank the Tip-Adapter authors for releasing their implementation and dataset preparation protocol; several adapter and dataset utilities in this repository are adapted from Tip-Adapter.

## Citation

If you find this research useful in your work, please consider citing our CVPR 2026 paper:

```bibtex
@misc{mohammad2026trainingonlyheterogeneousimagepatchtextgraph,
      title={Training-Only Heterogeneous Image-Patch-Text Graph Supervision for Advancing Few-Shot Learning Adapters},
      author={Mohammed Rahman Sherif Khan Mohammad and Ardhendu Behera and Sandip Pradhan and Swagat Kumar and Amr Ahmed},
      year={2026},
      eprint={2603.18101},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.18101},
}
```
