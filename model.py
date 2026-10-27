from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import HeteroLinear, Linear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax

from config import *


# Implements drainage-specific HEATConv with node- and edge-type-aware attention and message passing.
class DrainageHEATConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, num_node_types: int, num_edge_types: int,
                 edge_type_emb_dim: int, edge_dim: int, edge_attr_emb_dim: int, heads: int = 1,
                 concat: bool = True, negative_slope: float = 0.2, dropout: float = 0.0,
                 root_weight: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.edge_type_emb_dim = edge_type_emb_dim
        self.edge_dim = edge_dim
        self.edge_attr_emb_dim = edge_attr_emb_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.root_weight = root_weight

        self.hetero_lin = HeteroLinear(in_channels, out_channels, num_node_types, bias=bias)
        self.edge_type_emb = Embedding(num_edge_types, edge_type_emb_dim)
        self.edge_attr_emb = HeteroLinear(edge_dim, edge_attr_emb_dim, num_edge_types, bias=False)
        self.att = Linear(2 * out_channels + edge_type_emb_dim + edge_attr_emb_dim, heads, bias=False)
        self.lin = HeteroLinear(out_channels + edge_attr_emb_dim, out_channels, num_edge_types, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.hetero_lin.reset_parameters()
        self.edge_type_emb.reset_parameters()
        self.edge_attr_emb.reset_parameters()
        self.att.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, node_type: Tensor, edge_type: Tensor,
                edge_attr: OptTensor = None) -> Tensor:
        if edge_attr is None:
            num_edges = edge_index.size(1)
            edge_attr = x.new_zeros((num_edges, self.edge_dim))

        x = self.hetero_lin(x, node_type)
        edge_type_emb = F.leaky_relu(self.edge_type_emb(edge_type), self.negative_slope)
        out = self.propagate(edge_index, x=x, edge_type=edge_type,
                             edge_type_emb=edge_type_emb, edge_attr=edge_attr)

        if self.concat:
            if self.root_weight: out = out + x.view(-1, 1, self.out_channels)
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            if self.root_weight: out = out + x
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_type: Tensor, edge_type_emb: Tensor,
                edge_attr: Tensor, index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        edge_attr = F.leaky_relu(self.edge_attr_emb(edge_attr, edge_type), self.negative_slope)
        alpha_input = torch.cat([x_i, x_j, edge_type_emb, edge_attr], dim=-1)
        alpha = F.leaky_relu(self.att(alpha_input), self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        out = self.lin(msg_input, edge_type).unsqueeze(-2)
        return out * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})"


# Transformer-context + HEATConv + GRU digital twin
class TransformerHEATGRUDigitalTwin(nn.Module):

    # Initializes the Transformer-HEATConv-GRU digital-twin architecture and prediction heads.
    def __init__(self, in_channels: int, hidden_channels: int, num_node_types: int, num_edge_types: int, edge_type_emb_dim: int, edge_dim: int, edge_attr_emb_dim: int, heads: int, dropout: float, transformer_num_heads: int=TRANSFORMER_NUM_HEADS, transformer_encoder_layers: int=TRANSFORMER_ENCODER_LAYERS, transformer_feedforward_dim: int=TRANSFORMER_FEEDFORWARD_DIM, transformer_dropout: float=TRANSFORMER_DROPOUT, context_token_count: int=CONTEXT_TOKEN_COUNT, context_update_interval: int=CONTEXT_UPDATE_INTERVAL):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.context_token_count = context_token_count
        self.context_update_interval = context_update_interval
        self.register_buffer('junction_max_depth_std', torch.zeros(NUM_JUNCTIONS, dtype=torch.float32), persistent=True)
        self.conv1 = DrainageHEATConv(in_channels=in_channels, out_channels=hidden_channels, num_node_types=num_node_types, num_edge_types=num_edge_types, edge_type_emb_dim=edge_type_emb_dim, edge_dim=edge_dim, edge_attr_emb_dim=edge_attr_emb_dim, heads=heads, concat=True, dropout=dropout, root_weight=True, bias=True)
        conv1_out_dim = hidden_channels * heads
        self.conv2 = DrainageHEATConv(in_channels=conv1_out_dim, out_channels=hidden_channels, num_node_types=num_node_types, num_edge_types=num_edge_types, edge_type_emb_dim=edge_type_emb_dim, edge_dim=edge_dim, edge_attr_emb_dim=edge_attr_emb_dim, heads=1, concat=False, dropout=dropout, root_weight=True, bias=True)
        self.gnn_norm1 = nn.LayerNorm(conv1_out_dim)
        self.gnn_norm2 = nn.LayerNorm(hidden_channels)
        self.latent_conv = DrainageHEATConv(in_channels=hidden_channels, out_channels=hidden_channels, num_node_types=num_node_types, num_edge_types=num_edge_types, edge_type_emb_dim=edge_type_emb_dim, edge_dim=edge_dim, edge_attr_emb_dim=edge_attr_emb_dim, heads=1, concat=False, dropout=dropout, root_weight=True, bias=True)
        self.latent_norm = nn.LayerNorm(hidden_channels)
        self.context_daily_encoder = nn.Sequential(nn.Linear(CONTEXT_DAILY_FEATURE_DIM, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.context_hourly_encoder = nn.Sequential(nn.Linear(CONTEXT_HOURLY_FEATURE_DIM, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.context_position_embedding = nn.Embedding(context_token_count, hidden_channels)
        self.context_token_type_embedding = nn.Embedding(2, hidden_channels)
        context_layer = nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=transformer_num_heads, dim_feedforward=transformer_feedforward_dim, dropout=transformer_dropout, batch_first=True, norm_first=False)
        self.context_transformer = nn.TransformerEncoder(encoder_layer=context_layer, num_layers=transformer_encoder_layers)
        self.context_norm = nn.LayerNorm(hidden_channels)
        self.forcing_encoder = nn.Sequential(nn.Linear(FORECAST_HORIZON * WEATHER_FEATURE_DIM, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels), nn.ReLU())
        self.node_gru = nn.GRU(input_size=4 * hidden_channels, hidden_size=hidden_channels, num_layers=1, batch_first=True)
        channel_gru_input_dim = 4 * hidden_channels + edge_dim
        self.channel_gru = nn.GRU(input_size=channel_gru_input_dim, hidden_size=hidden_channels, num_layers=1, batch_first=True)

        # Builds a direct multi-horizon prediction head.
        def horizon_head(out_dim):
            return nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, FORECAST_HORIZON * out_dim))
        self.subcatchment_head = horizon_head(SUBCATCHMENT_OUT_DIM)
        self.junction_depth_head = horizon_head(1)
        self.junction_flood_head = nn.Sequential(nn.Linear(hidden_channels + 4, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, 1))
        self.outfall_head = horizon_head(OUTFALL_OUT_DIM)
        self.channel_head = horizon_head(CHANNEL_OUT_DIM)

    # Decodes a latent representation into direct multi-horizon predictions.
    def _decode_horizon(self, head_module: nn.Module, latent: Tensor, out_dim: int) -> Tensor:
        raw = head_module(latent)
        return raw.reshape(*latent.shape[:-1], FORECAST_HORIZON, out_dim)

    # Decodes junction depth and physics-conditioned flood-occurrence predictions.
    def _decode_junction_horizon(self, junction_latent: Tensor) -> Tensor:

        pred_depth_std = self._decode_horizon(self.junction_depth_head, junction_latent, 1)
        num_junctions = junction_latent.size(-2)
        if num_junctions != self.junction_max_depth_std.numel():
            raise ValueError('Junction count does not match junction_max_depth_std.')
        latent_detached = junction_latent.detach().unsqueeze(-2).expand(*junction_latent.shape[:-1], FORECAST_HORIZON, self.hidden_channels)
        max_depth_std = self.junction_max_depth_std.to(device=pred_depth_std.device, dtype=pred_depth_std.dtype)
        max_depth_view_shape = [1] * (junction_latent.dim() - 2) + [num_junctions, 1, 1]
        max_depth_std = max_depth_std.view(*max_depth_view_shape).expand(*junction_latent.shape[:-2], num_junctions, FORECAST_HORIZON, 1)
        pred_depth_detached = pred_depth_std.detach()
        depth_margin_std = pred_depth_detached - max_depth_std
        soft_capacity_gate = torch.sigmoid(depth_margin_std)
        flood_features = torch.cat([latent_detached, pred_depth_detached, max_depth_std, depth_margin_std, soft_capacity_gate], dim=-1)
        flood_logit = self.junction_flood_head(flood_features)
        return torch.cat([pred_depth_std, flood_logit], dim=-1)

    # Encodes the static drainage graph for one timestep using FP32 HEATConv.
    def encode_nodes_one_timestep(self, x_t: Tensor, edge_index: Tensor, node_type: Tensor, edge_type: Tensor, edge_attr_t: Tensor) -> Tensor:

        with torch.autocast(device_type=x_t.device.type, enabled=False):
            x_t = x_t.float()
            edge_attr_t = edge_attr_t.float()
            h = self.conv1(x=x_t, edge_index=edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_attr_t)
            h = F.relu(self.gnn_norm1(h))
            h = self.conv2(x=h, edge_index=edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_attr_t)
            return F.relu(self.gnn_norm2(h))

    # Propagates the current latent node state through the static drainage graph.
    def encode_latent_one_timestep(self, latent_t: Tensor, edge_index: Tensor, node_type: Tensor, edge_type: Tensor, edge_static_std: Tensor) -> Tensor:

        with torch.autocast(device_type=latent_t.device.type, enabled=False):
            latent_t = latent_t.float()
            edge_static_std = edge_static_std.float()
            h = self.latent_conv(x=latent_t, edge_index=edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_static_std)
            return F.relu(self.latent_norm(h))

    # Applies latent HEATConv message passing to a batch of scenarios.
    def encode_latent_batch_one_timestep(self, latent_batch: Tensor, edge_index: Tensor, node_type: Tensor, edge_type: Tensor, edge_static_std: Tensor) -> Tensor:

        batch_size, num_nodes, hidden_dim = latent_batch.shape
        num_edges = edge_type.numel()
        edge_index_b, node_type_b, edge_type_b = self.repeat_static_graph_for_timesteps(edge_index=edge_index, node_type=node_type, edge_type=edge_type, num_repeats=batch_size)
        latent_flat = latent_batch.reshape(batch_size * num_nodes, hidden_dim)
        edge_static_flat = edge_static_std.unsqueeze(0).expand(batch_size, num_edges, -1).reshape(batch_size * num_edges, -1)
        latent_message_flat = self.encode_latent_one_timestep(latent_t=latent_flat, edge_index=edge_index_b, node_type=node_type_b, edge_type=edge_type_b, edge_static_std=edge_static_flat)
        return latent_message_flat.reshape(batch_size, num_nodes, hidden_dim)

    # Repeats static graph indices and types for batched spatial processing.
    def repeat_static_graph_for_timesteps(self, edge_index: Tensor, node_type: Tensor, edge_type: Tensor, num_repeats: int):
        num_nodes = node_type.size(0)
        num_edges = edge_type.size(0)
        offsets = torch.arange(num_repeats, device=edge_index.device).repeat_interleave(num_edges) * num_nodes
        repeated_edge_index = edge_index.repeat(1, num_repeats) + offsets.view(1, -1)
        repeated_node_type = node_type.repeat(num_repeats)
        repeated_edge_type = edge_type.repeat(num_repeats)
        return (repeated_edge_index, repeated_node_type, repeated_edge_type)

    # Vectorizes static HEATConv encoding over multiple timesteps.
    def encode_nodes_many_timesteps(self, x_steps: Tensor, edge_index: Tensor, node_type: Tensor, edge_type: Tensor, edge_attr_steps: Tensor, spatial_time_batch_size: int=SPATIAL_TIME_BATCH_SIZE) -> Tensor:

        outputs = []
        num_times = x_steps.size(0)
        num_nodes = x_steps.size(1)
        num_edges = edge_type.size(0)
        for start in range(0, num_times, spatial_time_batch_size):
            end = min(start + spatial_time_batch_size, num_times)
            local_steps = end - start
            edge_index_b, node_type_b, edge_type_b = self.repeat_static_graph_for_timesteps(edge_index=edge_index, node_type=node_type, edge_type=edge_type, num_repeats=local_steps)
            x_flat = x_steps[start:end].reshape(local_steps * num_nodes, -1)
            edge_attr_flat = edge_attr_steps[start:end].reshape(local_steps * num_edges, -1)
            h_flat = self.encode_nodes_one_timestep(x_t=x_flat, edge_index=edge_index_b, node_type=node_type_b, edge_type=edge_type_b, edge_attr_t=edge_attr_flat)
            outputs.append(h_flat.view(local_steps, num_nodes, self.hidden_channels))
        return torch.cat(outputs, dim=0)

    # Builds rolling 30-day Transformer context from daily and hourly weather tokens.
    def build_context_sequence_batch(self, weather_full_std: Tensor, origin_steps: Tensor) -> Tensor:

        if weather_full_std.dim() != 3:
            raise ValueError('weather_full_std must have shape [B,T,F].')
        if origin_steps.dim() != 1:
            raise ValueError('origin_steps must be 1-D.')
        batch_size = weather_full_std.size(0)
        total_steps = weather_full_std.size(1)
        num_origins = origin_steps.numel()
        if num_origins == 0:
            return weather_full_std.new_empty((batch_size, 0, self.hidden_channels))
        origin_steps = origin_steps.long()
        if int(origin_steps.min().item()) < FIRST_FORECAST_ORIGIN:
            raise ValueError(f'Full 30-day history is required. Earliest valid origin = {FIRST_FORECAST_ORIGIN}.')
        if int(origin_steps.max().item()) >= total_steps:
            raise ValueError('Forecast origin exceeds weather data.')
        history_offsets = torch.arange(CONTEXT_RAW_STEPS, device=weather_full_std.device, dtype=torch.long)
        history_start = origin_steps - CONTEXT_RAW_STEPS + 1
        history_indices = history_start.unsqueeze(1) + history_offsets.unsqueeze(0)
        history = weather_full_std[:, history_indices, :]
        older = history[:, :, :CONTEXT_OLDER_STEPS, :].reshape(batch_size, num_origins, CONTEXT_DAILY_TOKENS, STEPS_PER_DAY, WEATHER_FEATURE_DIM)
        older_mean = older.mean(dim=3)
        older_min = older.amin(dim=3)
        older_max = older.amax(dim=3)
        older_std = older.std(dim=3, unbiased=False)
        daily_features = torch.cat([older_mean, older_min, older_max, older_std], dim=-1)
        daily_tokens = self.context_daily_encoder(daily_features)
        recent = history[:, :, CONTEXT_OLDER_STEPS:, :].reshape(batch_size, num_origins, CONTEXT_HOURLY_TOKENS, CONTEXT_HOUR_RAW_STEPS, WEATHER_FEATURE_DIM)
        recent_features = recent.reshape(batch_size, num_origins, CONTEXT_HOURLY_TOKENS, CONTEXT_HOURLY_FEATURE_DIM)
        hourly_tokens = self.context_hourly_encoder(recent_features)
        tokens = torch.cat([daily_tokens, hourly_tokens], dim=2)
        positions = torch.arange(CONTEXT_TOKEN_COUNT, device=weather_full_std.device, dtype=torch.long)
        token_types = torch.cat([torch.zeros(CONTEXT_DAILY_TOKENS, dtype=torch.long, device=weather_full_std.device), torch.ones(CONTEXT_HOURLY_TOKENS, dtype=torch.long, device=weather_full_std.device)], dim=0)
        tokens = tokens + self.context_position_embedding(positions).view(1, 1, CONTEXT_TOKEN_COUNT, self.hidden_channels) + self.context_token_type_embedding(token_types).view(1, 1, CONTEXT_TOKEN_COUNT, self.hidden_channels)
        tokens = tokens.reshape(batch_size * num_origins, CONTEXT_TOKEN_COUNT, self.hidden_channels)
        encoded = self.context_transformer(tokens)
        context = self.context_norm(encoded[:, -1, :])
        return context.reshape(batch_size, num_origins, self.hidden_channels)

    # Keeps the deprecated physical-state teacher-forced path disabled.
    def forward_teacher_forced_chunk(self, *args, **kwargs):

        raise RuntimeError('forward_teacher_forced_chunk() is deprecated. Use forward_latent_batch_chunk().')

    # Runs batched hourly latent-state evolution and direct six-step forecasts.
    def forward_latent_batch_chunk(self, x_static_std: Tensor, edge_static_std: Tensor, weather_full_std: Tensor, forcing_blocks_std: Tensor, origin_steps: Tensor, edge_index: Tensor, node_type: Tensor, edge_type: Tensor, node_hidden: Optional[Tensor]=None, channel_hidden: Optional[Tensor]=None):

        if forcing_blocks_std.dim() != 4:
            raise ValueError('forcing_blocks_std must have shape [B,K,6,F].')
        batch_size = weather_full_std.size(0)
        num_blocks = forcing_blocks_std.size(1)
        num_nodes = node_type.numel()
        if forcing_blocks_std.size(2) != FORECAST_HORIZON:
            raise ValueError(f'Each forcing block must contain {FORECAST_HORIZON} future steps.')
        if forcing_blocks_std.size(3) != WEATHER_FEATURE_DIM:
            raise ValueError('Wrong weather feature dimension.')
        if origin_steps.numel() != num_blocks:
            raise ValueError('origin_steps and forcing blocks have different lengths.')
        spatial_static = self.encode_nodes_one_timestep(x_t=x_static_std, edge_index=edge_index, node_type=node_type, edge_type=edge_type, edge_attr_t=edge_static_std)
        spatial_static_batch = spatial_static.unsqueeze(0).expand(batch_size, -1, -1)
        context = self.build_context_sequence_batch(weather_full_std=weather_full_std, origin_steps=origin_steps)
        forcing_flat = forcing_blocks_std.reshape(batch_size, num_blocks, FORECAST_HORIZON * WEATHER_FEATURE_DIM)
        forcing = self.forcing_encoder(forcing_flat)
        sub_nodes = torch.where(node_type == SUBCATCHMENT_TYPE)[0]
        junc_nodes = torch.where(node_type == JUNCTION_TYPE)[0]
        outfall_nodes = torch.where(node_type == OUTFALL_TYPE)[0]
        downstream_mask = edge_type == DOWNSTREAM_CHANNEL_EDGE_TYPE
        channel_src = edge_index[0, downstream_mask]
        channel_dst = edge_index[1, downstream_mask]
        num_channels = channel_src.numel()
        channel_static = edge_static_std[downstream_mask, :]
        num_edges = edge_type.numel()
        latent_edge_index_b, latent_node_type_b, latent_edge_type_b = self.repeat_static_graph_for_timesteps(edge_index=edge_index, node_type=node_type, edge_type=edge_type, num_repeats=batch_size)
        latent_edge_static_b = edge_static_std.unsqueeze(0).expand(batch_size, num_edges, -1).reshape(batch_size * num_edges, -1)
        channel_static_batch = channel_static.unsqueeze(0).expand(batch_size, -1, -1)
        pred_sub_blocks = []
        pred_junc_blocks = []
        pred_outfall_blocks = []
        pred_channel_blocks = []
        for local_block in range(num_blocks):
            if node_hidden is None:
                latent_prev = torch.zeros(batch_size, num_nodes, self.hidden_channels, dtype=spatial_static.dtype, device=spatial_static.device)
            else:
                latent_prev = node_hidden[0].reshape(batch_size, num_nodes, self.hidden_channels)
            latent_flat = latent_prev.reshape(batch_size * num_nodes, self.hidden_channels)
            latent_message_flat = self.encode_latent_one_timestep(latent_t=latent_flat, edge_index=latent_edge_index_b, node_type=latent_node_type_b, edge_type=latent_edge_type_b, edge_static_std=latent_edge_static_b)
            latent_message = latent_message_flat.reshape(batch_size, num_nodes, self.hidden_channels)
            context_t = context[:, local_block, :].unsqueeze(1).expand(-1, num_nodes, -1)
            forcing_t = forcing[:, local_block, :].unsqueeze(1).expand(-1, num_nodes, -1)
            node_input = torch.cat([spatial_static_batch, latent_message, context_t, forcing_t], dim=-1)
            node_input = node_input.reshape(batch_size * num_nodes, 1, 4 * self.hidden_channels)
            node_output_flat, node_hidden = self.node_gru(node_input, node_hidden)
            node_z = node_output_flat[:, 0, :].reshape(batch_size, num_nodes, self.hidden_channels)
            pred_sub_blocks.append(self._decode_horizon(self.subcatchment_head, node_z[:, sub_nodes, :], SUBCATCHMENT_OUT_DIM))
            pred_junc_blocks.append(self._decode_junction_horizon(node_z[:, junc_nodes, :]))
            pred_outfall_blocks.append(self._decode_horizon(self.outfall_head, node_z[:, outfall_nodes, :], OUTFALL_OUT_DIM))
            channel_context_t = context[:, local_block, :].unsqueeze(1).expand(-1, num_channels, -1)
            channel_forcing_t = forcing[:, local_block, :].unsqueeze(1).expand(-1, num_channels, -1)
            channel_input = torch.cat([node_z[:, channel_src, :], node_z[:, channel_dst, :], channel_context_t, channel_forcing_t, channel_static_batch], dim=-1)
            channel_input = channel_input.reshape(batch_size * num_channels, 1, channel_input.size(-1))
            channel_output_flat, channel_hidden = self.channel_gru(channel_input, channel_hidden)
            channel_z = channel_output_flat[:, 0, :].reshape(batch_size, num_channels, self.hidden_channels)
            pred_channel_blocks.append(self._decode_horizon(self.channel_head, channel_z, CHANNEL_OUT_DIM))

        def merge_blocks(blocks, out_dim):
            stacked = torch.stack(blocks, dim=2)
            return stacked.reshape(stacked.size(0), stacked.size(1), num_blocks * FORECAST_HORIZON, out_dim)
        predictions = {'subcatchment': merge_blocks(pred_sub_blocks, SUBCATCHMENT_OUT_DIM), 'junction': merge_blocks(pred_junc_blocks, JUNCTION_OUT_DIM), 'channel': merge_blocks(pred_channel_blocks, CHANNEL_OUT_DIM), 'outfall': merge_blocks(pred_outfall_blocks, OUTFALL_OUT_DIM)}
        return (predictions, node_hidden, channel_hidden)

    # Runs one hourly latent-state update for single-scenario inference.
    def step_block(self, x_t: Tensor, edge_attr_t: Tensor, context_t: Tensor, forcing_block_std: Tensor, edge_index: Tensor, node_type: Tensor, edge_type: Tensor, node_hidden: Optional[Tensor]=None, channel_hidden: Optional[Tensor]=None, spatial_static: Optional[Tensor]=None):

        if forcing_block_std.shape != (FORECAST_HORIZON, WEATHER_FEATURE_DIM):
            raise ValueError(f'forcing_block_std must have shape [{FORECAST_HORIZON}, {WEATHER_FEATURE_DIM}].')
        if spatial_static is None:
            spatial_static = self.encode_nodes_one_timestep(x_t=x_t, edge_index=edge_index, node_type=node_type, edge_type=edge_type, edge_attr_t=edge_attr_t)
        num_nodes = spatial_static.size(0)
        if node_hidden is None:
            latent_prev = torch.zeros(num_nodes, self.hidden_channels, dtype=spatial_static.dtype, device=spatial_static.device)
        else:
            latent_prev = node_hidden[0]
        latent_message = self.encode_latent_one_timestep(latent_t=latent_prev, edge_index=edge_index, node_type=node_type, edge_type=edge_type, edge_static_std=edge_attr_t)
        forcing_t = self.forcing_encoder(forcing_block_std.reshape(1, FORECAST_HORIZON * WEATHER_FEATURE_DIM))[0]
        node_input = torch.cat([spatial_static, latent_message, context_t.view(1, -1).expand(num_nodes, -1), forcing_t.view(1, -1).expand(num_nodes, -1)], dim=-1).unsqueeze(1)
        node_output, node_hidden_out = self.node_gru(node_input, node_hidden)
        node_z = node_output[:, 0, :]
        sub_nodes = torch.where(node_type == SUBCATCHMENT_TYPE)[0]
        junc_nodes = torch.where(node_type == JUNCTION_TYPE)[0]
        outfall_nodes = torch.where(node_type == OUTFALL_TYPE)[0]
        downstream_mask = edge_type == DOWNSTREAM_CHANNEL_EDGE_TYPE
        channel_src = edge_index[0, downstream_mask]
        channel_dst = edge_index[1, downstream_mask]
        channel_input = torch.cat([node_z[channel_src], node_z[channel_dst], context_t.view(1, -1).expand(channel_src.numel(), -1), forcing_t.view(1, -1).expand(channel_src.numel(), -1), edge_attr_t[downstream_mask]], dim=-1).unsqueeze(1)
        channel_output, channel_hidden_out = self.channel_gru(channel_input, channel_hidden)
        channel_z = channel_output[:, 0, :]
        predictions = {'subcatchment': self._decode_horizon(self.subcatchment_head, node_z[sub_nodes], SUBCATCHMENT_OUT_DIM), 'junction': self._decode_junction_horizon(node_z[junc_nodes]), 'channel': self._decode_horizon(self.channel_head, channel_z, CHANNEL_OUT_DIM), 'outfall': self._decode_horizon(self.outfall_head, node_z[outfall_nodes], OUTFALL_OUT_DIM)}
        return (predictions, node_hidden_out, channel_hidden_out)
