# TOGA: Training-Only Heterogeneous Image-Patch-Text Graph Supervision for Advancing Few-Shot Learning Adapters

Official PyTorch implementation of the CVPR 2026 paper:

**Training-Only Heterogeneous Image-Patch-Text Graph Supervision for Advancing Few-Shot Learning Adapters**

**Authors:** Mohammed Rahman Sherif Khan Mohammad, Ardhendu Behera, Sandip Pradhan, Swagat Kumar, Amr Ahmed  
**Institution:** Edge Hill University

## Overview

TOGA improves few-shot CLIP adaptation by training a lightweight cache adapter with a training-only Modality-aware Graph Transformer (MGT) teacher. The teacher reasons over multi-scale visual patch nodes and class-text nodes during training, then is discarded at test time. Inference keeps the Tip-Adapter-F path: CLIP logits plus cache logits, with no graph computation.

## Key Contributions

- **Asymmetric supervision:** a training-only graph teacher supervises the cache adapter while preserving zero test-time overhead.
- **MGT teacher:** relation-aware cross-modal reasoning over a unified patch-text graph.
- **Discriminative node filtering:** TopK visual node selection reduces feature dilution before the teacher readout.
- **Cache-aware dual objective:** the adapter is trained with the mixture logits, while a focal teacher-forcing loss keeps the graph teacher useful on hard examples.

## Repository Structure

- `main.py`: training and evaluation entry point.
- `mgt_layer.py`: MGT message-passing layer with modality-specific projections and relation-aware key/value/prior parameters.
- `mgt_model.py`: training-only patch-text graph teacher with unimodal encoders, MGT layers, and TopK patch filtering.
- `graph_builder.py`: dense patch-text graph construction.
- `datasets/`: dataset wrappers and graph data loaders, including ImageNet.
- `configs/*.yaml`: dataset paths and dataset-level settings.
- `configs/configs.yaml`: dataset/shot hyperparameter presets.

## Setup

Create an environment with Python 3.10 or newer, then install dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch and PyTorch Geometric builds that match your CUDA version if the generic packages do not match your machine.

## Data

Set `root_path` in each dataset file under `configs/` to the directory containing the datasets, or pass it at launch with `--root_path`. The expected layout follows each loader in `datasets/`.

For ImageNet, `configs/imagenet.yaml` expects:

```text
<root_path>/imagenet/images/train
<root_path>/imagenet/images/val
```

Cached visual features and adapter checkpoints are written under `caches/<dataset>/`. These files are ignored by Git because they are regenerated from the configured data and shot count.

## Reproducing Runs

Each dataset/shot preset is stored in `configs/configs.yaml`. Run with the dataset config and shot count; `main.py` automatically loads the matching preset.

```bash
python main.py --config ./configs/fgvc.yaml --shots 1
python main.py --config ./configs/eurosat.yaml --shots 16
python main.py --config ./configs/caltech101.yaml --shots 4
python main.py --config ./configs/imagenet.yaml --shots 16
```

For a local data directory:

```bash
python main.py --config ./configs/dtd.yaml --shots 1 --root_path C:/Users/Mohamm/Documents/data
```

To run without W&B logging:

```bash
python main.py --config ./configs/fgvc.yaml --shots 1 --wandb_mode disabled
```

You can override any preset value from the command line, for example:

```bash
python main.py --config ./configs/ucf101.yaml --shots 8 --lr 0.0002 --pooling_ratio 0.5
```

On Windows, add `--num_workers 0` for a single-process smoke test.

## Hyperparameter Presets

The MGT preset keys in `configs/configs.yaml` are:

- `mgt_num_layers`, `mgt_num_heads`
- `init_delta` for the graph teacher logit weight
- `lambda_teacher` for the focal teacher-forcing loss weight
- `focal_loss_gamma` for the focal loss focusing parameter
- `patience` for optional early stopping, used by the ImageNet presets

The remaining adapter and transformer keys keep their conventional names, such as `init_beta`, `init_alpha`, `transformer_num_layers`, and `transformer_nhead`.

## Results Snapshot

TOGA reports state-of-the-art performance across 11 benchmark datasets: ImageNet, SUN397, FGVC-Aircraft, EuroSAT, Stanford Cars, Food101, Oxford Pets, Flowers102, Caltech101, DTD, and UCF101.

| Method | 1-Shot Avg | 2-Shot Avg | 4-Shot Avg | 8-Shot Avg | 16-Shot Avg | Test-Time Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Tip-Adapter-F | 64.3% | 66.1% | 69.1% | 73.3% | 75.8% | Zero |
| GraphAdapter | 62.7% | 67.8% | 69.8% | 71.4% | 74.4% | High |
| TOGA | 72.2% | 75.0% | 77.9% | 80.0% | 82.3% | Zero |

## Reproducibility Notes

The default seed is `1`, matching the experiment code path. The graph teacher is trained jointly with the cache adapter, but it is discarded at evaluation time. Final reported inference uses CLIP logits plus cache logits, preserving the lightweight test-time behavior.

## Citation

```bibtex
@misc{mohammad2026trainingonlyheterogeneousimagepatchtextgraph,
  title={Training-Only Heterogeneous Image-Patch-Text Graph Supervision for Advancing Few-Shot Learning Adapters},
  author={Mohammed Rahman Sherif Khan Mohammad and Ardhendu Behera and Sandip Pradhan and Swagat Kumar and Amr Ahmed},
  year={2026},
  eprint={2603.18101},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.18101}
}
```
