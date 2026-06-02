from __future__ import annotations

import torch
from torch_geometric.data import HeteroData


def build_fully_connected_hetero_graph(
    patch_f: torch.Tensor,
    text_f: torch.Tensor,
    image_id: str | int,
) -> HeteroData:
    """
    Build a fully connected patch-text graph for one image.

    Node types are ``patch`` for multi-scale visual crops and ``text`` for
    class-name prompt embeddings.
    """
    data = HeteroData()
    num_patches = patch_f.shape[0]
    num_classes = text_f.shape[0]

    data["patch"].x = patch_f
    data["text"].x = text_f

    def create_dense_edges(num_src, num_dst, with_self_loops=False):
        src, dst = [], []
        for i in range(num_src):
            for j in range(num_dst):
                if not with_self_loops and i == j and num_src == num_dst:
                    continue
                src.append(i)
                dst.append(j)
        return torch.tensor([src, dst], dtype=torch.long)

    if num_patches > 1:
        data["patch", "patch_to_patch", "patch"].edge_index = create_dense_edges(
            num_patches,
            num_patches,
        )
    else:
        data["patch", "patch_to_patch", "patch"].edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

    edge_index_patch_text = create_dense_edges(
        num_patches,
        num_classes,
        with_self_loops=True,
    )
    data["patch", "patch_to_text", "text"].edge_index = edge_index_patch_text
    data["text", "text_to_patch", "patch"].edge_index = edge_index_patch_text.flip([0])

    data.image_id = image_id
    return data
