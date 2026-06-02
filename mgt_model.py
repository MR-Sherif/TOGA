from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import TopKPooling, global_add_pool

from mgt_layer import MGT


class ModalityAwareGraphTeacher(nn.Module):
    """Training-only patch-text graph teacher used for asymmetric supervision."""

    def __init__(
        self,
        node_types,
        edge_types,
        input_dims,
        hidden_channels,
        mgt_num_heads,
        mgt_num_layers,
        dropout_rate,
        transformer_nhead,
        transformer_num_layers,
        transformer_ff_multiplier,
        transformer_activation,
        shots,
        pooling_ratio: float,
    ):
        super().__init__()

        self.node_types = node_types
        self.visual_node_types = [node_type for node_type in node_types if node_type != "text"]
        self.edge_types = edge_types
        self.metadata = (self.node_types, self.edge_types)
        self.hidden_channels = hidden_channels
        self.num_mgt_layers = mgt_num_layers
        self.shots = shots

        if hidden_channels % mgt_num_heads != 0:
            raise ValueError(
                f"MGT hidden_channels ({hidden_channels}) must be divisible by "
                f"mgt_num_heads ({mgt_num_heads})."
            )
        if hidden_channels % transformer_nhead != 0:
            raise ValueError(
                f"Transformer hidden_channels ({hidden_channels}) must be divisible by "
                f"transformer_nhead ({transformer_nhead})."
            )

        self.input_projection = nn.ModuleDict(
            {
                node_type: nn.Linear(input_dims[node_type], hidden_channels)
                for node_type in self.node_types
            }
        )

        self.unimodal_encoders = nn.ModuleDict()
        for node_type in self.node_types:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_channels,
                nhead=transformer_nhead,
                dim_feedforward=hidden_channels * transformer_ff_multiplier,
                dropout=dropout_rate,
                activation=transformer_activation,
                batch_first=True,
            )
            self.unimodal_encoders[node_type] = nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_num_layers,
            )

        self.mgt_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.num_mgt_layers):
            self.mgt_layers.append(
                MGT(hidden_channels, hidden_channels, self.metadata, mgt_num_heads)
            )
            self.norms.append(
                nn.ModuleDict(
                    {
                        node_type: nn.LayerNorm(hidden_channels)
                        for node_type in self.node_types
                    }
                )
            )

        self.dropout = nn.Dropout(dropout_rate)
        self.topk_pools = nn.ModuleDict(
            {
                node_type: TopKPooling(hidden_channels, ratio=pooling_ratio)
                for node_type in self.visual_node_types
            }
        )

    def forward(self, x_dict, edge_index_dict, batch_dict):
        projected_x_dict = {
            node_type: self.input_projection[node_type](features)
            for node_type, features in x_dict.items()
        }

        transformed_x_dict = {}
        for node_type, features in projected_x_dict.items():
            batch = batch_dict[node_type]
            per_image_outputs = []
            for image_idx in torch.unique(batch):
                image_mask = batch == image_idx
                sequence = features[image_mask].unsqueeze(0)
                encoded = self.unimodal_encoders[node_type](sequence)
                per_image_outputs.append(encoded.squeeze(0))
            transformed_x_dict[node_type] = torch.cat(per_image_outputs, dim=0)

        current_x_dict = transformed_x_dict
        for mgt_layer, norm_dict in zip(self.mgt_layers, self.norms):
            residual_x_dict = current_x_dict
            updated_x_dict = mgt_layer(current_x_dict, edge_index_dict)

            for node_type in residual_x_dict.keys():
                normalized = norm_dict[node_type](updated_x_dict[node_type]).relu()
                current_x_dict[node_type] = residual_x_dict[node_type] + self.dropout(
                    normalized
                )

        pooled_visual_features = []
        for node_type in self.visual_node_types:
            features = current_x_dict[node_type]
            edge_type = (node_type, "patch_to_patch", node_type)
            edge_index = edge_index_dict.get(
                edge_type,
                torch.empty(2, 0, device=features.device, dtype=torch.long),
            )

            if edge_index.numel() == 0:
                selected_features = features
                selected_batch = batch_dict[node_type]
            else:
                selected_features, _, _, selected_batch, _, _ = self.topk_pools[
                    node_type
                ](
                    x=features,
                    edge_index=edge_index,
                    batch=batch_dict[node_type],
                )

            pooled_visual_features.append(
                global_add_pool(selected_features, selected_batch)
            )

        if len(pooled_visual_features) != 1:
            raise RuntimeError(
                f"Expected one visual modality, found {len(pooled_visual_features)}."
            )

        graph_visual_feature = pooled_visual_features[0]
        updated_text_features = current_x_dict["text"]
        return graph_visual_feature, updated_text_features
