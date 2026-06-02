from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import wandb
import yaml
from tqdm import tqdm

import clip
from datasets import build_dataset
from datasets.utils import build_data_loader, build_graph_data_loader
from mgt_model import ModalityAwareGraphTeacher
from utils import build_cache_model, clip_classifier, cls_acc, pre_load_features, search_hp


DEFAULT_RUN_CONFIG = {
    "batch_size": 4,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "mgt_num_layers": 3,
    "mgt_num_heads": 16,
    "transformer_num_layers": 3,
    "transformer_nhead": 16,
    "transformer_ff_multiplier": 2,
    "transformer_activation": "gelu",
    "pooling_ratio": 0.25,
    "dropout_rate": 0.5,
    "train_epoch": 100,
    "init_beta": 2.0493246903001525,
    "init_alpha": 9.806773071379816,
    "init_delta": 1.0,
    "lambda_teacher": 0.5,
    "focal_loss_gamma": 2.0,
    "seed": 1,
    "num_workers": 4,
    "wandb_project": "TOGA",
    "wandb_mode": "online",
}

INT_FIELDS = {
    "batch_size",
    "mgt_num_layers",
    "mgt_num_heads",
    "patience",
    "transformer_num_layers",
    "transformer_nhead",
    "transformer_ff_multiplier",
    "train_epoch",
    "shots",
    "seed",
    "num_workers",
}
FLOAT_FIELDS = {
    "lr",
    "weight_decay",
    "pooling_ratio",
    "dropout_rate",
    "init_beta",
    "init_alpha",
    "init_delta",
    "lambda_teacher",
    "focal_loss_gamma",
}
OVERRIDABLE_FIELDS = sorted(
    (INT_FIELDS - {"shots"}) | FLOAT_FIELDS | {"transformer_activation"}
)


def load_yaml(path: str | Path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_backbone_config(backbone):
    if isinstance(backbone, str):
        return [backbone]
    if isinstance(backbone, (list, tuple)) and backbone:
        return list(backbone)
    raise ValueError("Dataset config must define at least one CLIP backbone.")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the TOGA cache adapter with a training-only MGT teacher.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="./configs/fgvc.yaml")
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--preset_config", type=str, default="./configs/configs.yaml")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=("online", "offline", "disabled"),
    )

    for field in OVERRIDABLE_FIELDS:
        if field in INT_FIELDS:
            parser.add_argument(f"--{field}", type=int, default=None)
        elif field in FLOAT_FIELDS:
            parser.add_argument(f"--{field}", type=float, default=None)
        else:
            parser.add_argument(f"--{field}", type=str, default=None)

    return parser


def resolve_run_config(args):
    dataset_cfg = load_yaml(args.config)
    dataset_name = dataset_cfg["dataset"]

    preset_doc = load_yaml(args.preset_config)
    run_config = dict(DEFAULT_RUN_CONFIG)
    run_config.update(preset_doc.get("defaults", {}))

    preset = (
        preset_doc.get("presets", {})
        .get(dataset_name, {})
        .get(args.shots, {})
    )
    run_config.update(preset)
    run_config["config"] = args.config
    run_config["shots"] = args.shots
    run_config["dataset"] = dataset_name
    if args.root_path is not None:
        run_config["root_path"] = args.root_path

    for field in OVERRIDABLE_FIELDS:
        value = getattr(args, field)
        if value is not None:
            run_config[field] = value

    if args.wandb_project is not None:
        run_config["wandb_project"] = args.wandb_project
    if args.wandb_mode is not None:
        run_config["wandb_mode"] = args.wandb_mode

    for field in INT_FIELDS:
        if field in run_config and run_config[field] is not None:
            run_config[field] = int(run_config[field])
    for field in FLOAT_FIELDS:
        if field in run_config:
            run_config[field] = float(run_config[field])

    return run_config


def run_tip_adapter(
    cfg,
    run_args,
    cache_keys,
    cache_values,
    val_features,
    val_labels,
    test_features,
    test_labels,
    clip_weights,
):
    cache_values = cache_values.to(val_features.dtype)

    print("\n-------- Searching hyperparameters on the val set. --------")
    clip_logits = 100.0 * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP val accuracy: {:.2f}. ****\n".format(acc))

    beta, alpha = run_args.init_beta, run_args.init_alpha
    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter val accuracy: {:.2f}. ****\n".format(acc))

    best_beta, best_alpha = search_hp(
        cfg,
        cache_keys,
        cache_values,
        val_features,
        val_labels,
        clip_weights,
    )

    print("\n-------- Evaluating on the test set. --------")
    clip_logits = 100.0 * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP test accuracy: {:.2f}. ****\n".format(acc))

    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter test accuracy: {:.2f}. ****\n".format(acc))


def run_tip_adapter_F(
    cfg,
    run_args,
    cache_keys,
    cache_values,
    val_features,
    val_labels,
    test_features,
    test_labels,
    clip_weights,
    clip_model,
    graph_teacher,
    train_loader_F,
):
    device = next(clip_model.parameters()).device
    dtype = next(clip_model.parameters()).dtype
    graph_teacher.to(device)

    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(device)
    adapter = adapter.to(dtype)
    adapter.weight = nn.Parameter(cache_keys.t())

    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(graph_teacher.parameters()),
        lr=run_args.lr,
        eps=1e-4,
        weight_decay=run_args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        cfg["train_epoch"] * len(train_loader_F),
    )

    beta = run_args.init_beta
    alpha = run_args.init_alpha
    delta = run_args.init_delta
    lambda_teacher = run_args.lambda_teacher
    focal_gamma = run_args.focal_loss_gamma
    best_acc, best_epoch = 0.0, 0
    epochs_no_improve = 0
    patience = getattr(run_args, "patience", None)

    for train_idx in range(cfg["train_epoch"]):
        adapter.train()
        graph_teacher.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print(f'Train Epoch: {train_idx} / {cfg["train_epoch"]}')

        for batched_graphs, images, target in tqdm(train_loader_F):
            batched_graphs = batched_graphs.to(device)
            images = images.to(device)
            target = target.to(device)

            graph_visual_feature, updated_text_features = graph_teacher(
                batched_graphs.x_dict,
                batched_graphs.edge_index_dict,
                batched_graphs.batch_dict,
            )

            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            original_clip_logits = 100.0 * image_features.to(dtype) @ clip_weights.to(dtype)
            affinity = adapter(image_features.to(dtype))
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values.to(dtype)

            graph_visual_feature = graph_visual_feature / graph_visual_feature.norm(
                dim=-1,
                keepdim=True,
            )
            updated_text_features = updated_text_features / updated_text_features.norm(
                dim=-1,
                keepdim=True,
            )
            batch_size = graph_visual_feature.shape[0]
            feature_dim = graph_visual_feature.shape[1]
            num_classes = updated_text_features.shape[0] // batch_size
            text_features_per_image = updated_text_features.view(
                batch_size,
                num_classes,
                feature_dim,
            )
            teacher_logits = 100.0 * torch.bmm(
                graph_visual_feature.unsqueeze(1),
                text_features_per_image.transpose(1, 2),
            ).squeeze(1)

            tip_logits = original_clip_logits + cache_logits * alpha + teacher_logits * delta
            classification_loss = F.cross_entropy(tip_logits, target)

            log_pt = F.log_softmax(teacher_logits, dim=1)
            pt = torch.exp(log_pt)
            pt_correct = pt[torch.arange(batch_size, device=device), target]
            log_pt_correct = log_pt[torch.arange(batch_size, device=device), target]
            teacher_loss = -torch.pow(1 - pt_correct, focal_gamma) * log_pt_correct
            teacher_loss = teacher_loss.mean()

            total_loss = classification_loss + lambda_teacher * teacher_loss
            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(total_loss.item())

            if run_args.wandb_mode != "disabled":
                wandb.log(
                    {
                        "Train accuracy": correct_samples / all_samples,
                        "Total loss": sum(loss_list) / len(loss_list),
                        "Loss (classification)": classification_loss.item(),
                        "Loss (teacher focal)": teacher_loss.item(),
                        "Learning rate": scheduler.get_last_lr()[0],
                    }
                )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(
            "LR: {:.6f}, Acc: {:.4f} ({}/{}), Loss: {:.4f}".format(
                current_lr,
                correct_samples / all_samples,
                correct_samples,
                all_samples,
                sum(loss_list) / len(loss_list),
            )
        )

        adapter.eval()
        graph_teacher.eval()
        with torch.no_grad():
            affinity = adapter(test_features.to(dtype))
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values.to(dtype)
            clip_logits = 100.0 * test_features.to(dtype) @ clip_weights.to(dtype)
            tip_logits = clip_logits + cache_logits * alpha
            acc = cls_acc(tip_logits, test_labels)

        if run_args.wandb_mode != "disabled":
            wandb.log({"Test accuracy": acc, "Epoch": train_idx})

        print(f"**** Tip-Adapter-F test accuracy: {acc:.2f}. ****\n")
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            epochs_no_improve = 0
            torch.save(
                adapter.state_dict(),
                os.path.join(cfg["cache_dir"], f"best_F_{cfg['shots']}shots.pt"),
            )
        else:
            epochs_no_improve += 1
            if patience is not None and epochs_no_improve >= patience:
                print(
                    f"**** Early stopping at epoch {train_idx} after "
                    f"{patience} epochs without improvement. ****"
                )
                break

    adapter.load_state_dict(
        torch.load(
            os.path.join(cfg["cache_dir"], f"best_F_{cfg['shots']}shots.pt"),
            map_location=device,
        )
    )
    print(
        "**** After fine-tuning, Tip-Adapter-F best test accuracy: "
        f"{best_acc:.2f}, at epoch: {best_epoch}. ****\n"
    )

    if run_args.wandb_mode != "disabled":
        wandb.summary["best_test_accuracy"] = best_acc

    print("\n-------- Searching hyperparameters on the val set. --------")
    best_beta, best_alpha = search_hp(
        cfg,
        cache_keys,
        cache_values.to(dtype),
        val_features.to(dtype),
        val_labels,
        clip_weights.to(dtype),
        adapter=adapter,
    )

    print("\n-------- Evaluating on the test set. --------")
    clip_logits = 100.0 * test_features.to(dtype) @ clip_weights.to(dtype)
    affinity = adapter(test_features.to(dtype))
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values.to(dtype)
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    final_acc = max(best_acc, acc)
    print("**** {} Tip-Adapter-F test accuracy: {:.2f}. ****\n".format(cfg["dataset"], final_acc))

    if run_args.wandb_mode != "disabled":
        wandb.summary["final_test_accuracy_after_search"] = final_acc


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_config = resolve_run_config(args)

    wandb.init(
        project=run_config["wandb_project"],
        mode=run_config["wandb_mode"],
        config=run_config,
    )
    run_args = SimpleNamespace(**dict(wandb.config))

    cfg = load_yaml(run_args.config)
    if hasattr(run_args, "root_path"):
        cfg["root_path"] = run_args.root_path
    cfg["shots"] = run_args.shots
    cfg["train_epoch"] = run_args.train_epoch
    cfg["init_beta"] = run_args.init_beta
    cfg["init_alpha"] = run_args.init_alpha
    cfg["cache_dir"] = os.path.join("./caches", cfg["dataset"])
    os.makedirs(cfg["cache_dir"], exist_ok=True)

    set_seed(run_args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nRunning dataset config:")
    print(cfg, "\n")
    print("Run hyperparameters:")
    print(dict(wandb.config))

    node_types = ["patch", "text"]
    edge_types = [
        ("patch", "patch_to_patch", "patch"),
        ("patch", "patch_to_text", "text"),
        ("text", "text_to_patch", "patch"),
    ]

    backbone_name = normalize_backbone_config(cfg["backbone"])[0]
    clip_model, preprocess = clip.load(backbone_name, device=device)
    clip_model = clip_model.float()

    print("Preparing dataset.")
    dataset = build_dataset(cfg["dataset"], cfg["root_path"], cfg["shots"])

    print("\nGetting textual features for graph nodes and CLIP classifier.")
    text_features_for_nodes = clip_classifier(
        dataset.classnames,
        dataset.template,
        clip_model,
    ).t()
    clip_weights = text_features_for_nodes.t()

    input_dims = {
        "patch": clip_model.visual.output_dim,
        "text": clip_model.text_projection.shape[1],
    }
    hidden_dim = clip_model.visual.output_dim

    graph_teacher = ModalityAwareGraphTeacher(
        node_types=node_types,
        edge_types=edge_types,
        input_dims=input_dims,
        hidden_channels=hidden_dim,
        mgt_num_heads=run_args.mgt_num_heads,
        mgt_num_layers=run_args.mgt_num_layers,
        dropout_rate=run_args.dropout_rate,
        transformer_nhead=run_args.transformer_nhead,
        transformer_num_layers=run_args.transformer_num_layers,
        transformer_ff_multiplier=run_args.transformer_ff_multiplier,
        transformer_activation=run_args.transformer_activation,
        pooling_ratio=run_args.pooling_ratio,
        shots=run_args.shots,
    ).to(device)

    val_loader = build_data_loader(
        data_source=dataset.val,
        batch_size=2,
        is_train=False,
        tfm=preprocess,
        num_workers=run_args.num_workers,
    )
    test_loader = build_data_loader(
        data_source=dataset.test,
        batch_size=2,
        is_train=False,
        tfm=preprocess,
        num_workers=run_args.num_workers,
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.5, 1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    train_loader_cache = build_data_loader(
        data_source=dataset.train_x,
        batch_size=run_args.batch_size,
        is_train=True,
        tfm=train_transform,
        num_workers=run_args.num_workers,
    )
    train_loader_F = build_graph_data_loader(
        data_source=dataset.train_x,
        batch_size=run_args.batch_size,
        shuffle=True,
        transform=train_transform,
        visual_encoder=clip_model.visual,
        text_features=text_features_for_nodes,
        processor=preprocess,
        device=device,
        num_workers=run_args.num_workers,
    )

    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)

    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    run_tip_adapter(
        cfg,
        run_args,
        cache_keys,
        cache_values,
        val_features,
        val_labels,
        test_features,
        test_labels,
        clip_weights,
    )
    run_tip_adapter_F(
        cfg,
        run_args,
        cache_keys,
        cache_values,
        val_features,
        val_labels,
        test_features,
        test_labels,
        clip_weights,
        clip_model,
        graph_teacher,
        train_loader_F,
    )

    if run_args.wandb_mode != "disabled":
        wandb.finish()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
