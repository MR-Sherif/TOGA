import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import HeteroDictLinear, HeteroLinear
from torch_geometric.nn.inits import ones
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax
from torch_geometric.utils.hetero import construct_bipartite_edge_index


class MGT(MessagePassing):
    """Modality-aware graph transformer layer for typed patch-text graphs."""

    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        **kwargs,
    ):
        super().__init__(aggr="add", node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by heads ({heads})"
            )

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.relation_type_to_id = {
            edge_type: i for i, edge_type in enumerate(metadata[1])
        }
        self.target_node_types = {edge_type[-1] for edge_type in self.edge_types}

        self.modality_qkv_projection = HeteroDictLinear(
            self.in_channels,
            self.out_channels * 3,
        )
        self.modality_output_projection = HeteroDictLinear(
            self.out_channels,
            self.out_channels,
            types=self.node_types,
        )

        head_dim = out_channels // heads
        num_relation_heads = heads * len(self.edge_types)
        self.relation_key_projection = HeteroLinear(
            head_dim,
            head_dim,
            num_relation_heads,
            bias=False,
            is_sorted=True,
        )
        self.relation_value_projection = HeteroLinear(
            head_dim,
            head_dim,
            num_relation_heads,
            bias=False,
            is_sorted=True,
        )

        self.modality_residual_gate = ParameterDict(
            {
                node_type: Parameter(torch.empty(1))
                for node_type in self.node_types
            }
        )
        self.relation_attention_prior = ParameterDict()
        for edge_type in self.edge_types:
            relation_name = "__".join(edge_type)
            self.relation_attention_prior[relation_name] = Parameter(
                torch.empty(1, heads)
            )

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.modality_qkv_projection.reset_parameters()
        self.modality_output_projection.reset_parameters()
        self.relation_key_projection.reset_parameters()
        self.relation_value_projection.reset_parameters()
        ones(self.modality_residual_gate)
        ones(self.relation_attention_prior)

    def _pack_by_type(
        self,
        x_dict: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, int]]:
        offset = {}
        packed = []
        cursor = 0
        for node_type, features in x_dict.items():
            packed.append(features)
            offset[node_type] = cursor
            cursor += features.size(0)
        return torch.cat(packed, dim=0), offset

    def _relation_project_sources(
        self,
        key_by_type: Dict[str, Tensor],
        value_by_type: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
    ) -> Tuple[Tensor, Tensor, Dict[EdgeType, int]]:
        cursor = 0
        num_relations = len(self.edge_types)
        heads = self.heads
        head_dim = self.out_channels // self.heads

        keys = []
        values = []
        relation_ids = []
        source_offset = {}
        for edge_type in edge_index_dict.keys():
            source_type = edge_type[0]
            num_source_nodes = key_by_type[source_type].size(0)
            source_offset[edge_type] = cursor
            cursor += num_source_nodes

            relation_id = self.relation_type_to_id[edge_type]
            ids = (
                torch.arange(
                    heads,
                    dtype=torch.long,
                    device=key_by_type[source_type].device,
                )
                .view(-1, 1)
                .repeat(1, num_source_nodes)
                * num_relations
                + relation_id
            )

            relation_ids.append(ids)
            keys.append(key_by_type[source_type])
            values.append(value_by_type[source_type])

        keys = torch.cat(keys, dim=0).transpose(0, 1).reshape(-1, head_dim)
        values = torch.cat(values, dim=0).transpose(0, 1).reshape(-1, head_dim)
        relation_ids = torch.cat(relation_ids, dim=1).flatten()

        relation_keys = self.relation_key_projection(keys, relation_ids)
        relation_values = self.relation_value_projection(values, relation_ids)

        relation_keys = relation_keys.view(heads, -1, head_dim).transpose(0, 1)
        relation_values = relation_values.view(heads, -1, head_dim).transpose(0, 1)
        return relation_keys, relation_values, source_offset

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
    ) -> Dict[NodeType, Optional[Tensor]]:
        feature_dim = self.out_channels
        heads = self.heads
        head_dim = feature_dim // heads

        key_by_type = {}
        query_by_type = {}
        value_by_type = {}
        out_dict = {}

        projected_by_type = self.modality_qkv_projection(x_dict)
        for node_type, projection in projected_by_type.items():
            key, query, value = torch.tensor_split(projection, 3, dim=1)
            key_by_type[node_type] = key.view(-1, heads, head_dim)
            query_by_type[node_type] = query.view(-1, heads, head_dim)
            value_by_type[node_type] = value.view(-1, heads, head_dim)

        query, target_offset = self._pack_by_type(query_by_type)
        key, value, source_offset = self._relation_project_sources(
            key_by_type,
            value_by_type,
            edge_index_dict,
        )

        edge_index, relation_prior = construct_bipartite_edge_index(
            edge_index_dict,
            source_offset,
            target_offset,
            edge_attr_dict=self.relation_attention_prior,
            num_nodes=key.size(0),
        )

        out = self.propagate(
            edge_index,
            key=key,
            query=query,
            value=value,
            relation_prior=relation_prior,
        )

        for node_type, start_offset in target_offset.items():
            end_offset = start_offset + query_by_type[node_type].size(0)
            if node_type in self.target_node_types:
                out_dict[node_type] = out[start_offset:end_offset]

        projected_out = self.modality_output_projection(
            {
                node_type: torch.nn.functional.gelu(features)
                if features is not None
                else features
                for node_type, features in out_dict.items()
            }
        )

        for node_type, features in out_dict.items():
            features = projected_out[node_type]
            if features.size(-1) == x_dict[node_type].size(-1):
                gate = self.modality_residual_gate[node_type].sigmoid()
                features = gate * features + (1 - gate) * x_dict[node_type]
            out_dict[node_type] = features

        return out_dict

    def message(
        self,
        key_j: Tensor,
        query_i: Tensor,
        value_j: Tensor,
        relation_prior: Tensor,
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int],
    ) -> Tensor:
        attention = (query_i * key_j).sum(dim=-1) * relation_prior
        attention = attention / math.sqrt(query_i.size(-1))
        attention = softmax(attention, index, ptr, size_i)
        out = value_j * attention.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(-1, {self.out_channels}, heads={self.heads})"
