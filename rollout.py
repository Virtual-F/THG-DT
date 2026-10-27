from typing import Optional
import torch
from torch import Tensor

from config import *
from preprocessing import build_standardized_static_graph
from losses_metrics import inverse_target, inverse_target_component


    # Prepares edge-to-subcatchment and edge-to-channel mappings for closed-loop physical-state reconstruction.
def prepare_rollout_edge_maps(topology: dict, device: torch.device):
    edge_type, edge_index = topology["edge_type"], topology["edge_index"]

    runoff_sub_idx = torch.full((edge_type.numel(),), -1, dtype=torch.long)
    runoff_edges = torch.where(edge_type == RUNOFF_EDGE_TYPE)[0]
    if runoff_edges.numel() > 0:
        src = edge_index[0, runoff_edges]
        runoff_sub_idx[runoff_edges] = topology["global_to_subcatchment"][src]

    channel_name_to_idx = {name: i for i, name in enumerate(topology["downstream_conduit_names"])}
    conduit_channel_idx = torch.full((edge_type.numel(),), -1, dtype=torch.long)

    for e, (etype, link_name) in enumerate(zip(edge_type.tolist(), topology["edge_original_link_names"])):
        if etype in (DOWNSTREAM_CHANNEL_EDGE_TYPE, REVERSE_BACKWATER_EDGE_TYPE):
            if link_name not in channel_name_to_idx: raise KeyError(f"No channel output row found for link {link_name!r}")
            conduit_channel_idx[e] = channel_name_to_idx[link_name]

    return {"runoff_sub_idx": runoff_sub_idx.to(device), "conduit_channel_idx": conduit_channel_idx.to(device)}




# Builds one standardized node-feature matrix from raw recurrent dynamics and static node attributes.
def build_single_standardized_node_features_from_raw_dynamic(raw_dyn: Tensor, topology: dict, scalers: dict,
                                                             device: torch.device):
    dyn = (raw_dyn.float() - scalers["recurrent_node_dynamic_mean"].to(raw_dyn.device)) / scalers["recurrent_node_dynamic_std"].to(raw_dyn.device)
    static = topology["x_static"].float()
    static = (static - scalers["x_static_mean"].to(static.device)) / scalers["x_static_std"].to(static.device)
    return torch.cat([static, dyn], dim=-1).to(device)




# Builds one standardized edge-feature matrix from raw dynamic and static edge attributes.
def build_single_standardized_edge_features_from_raw_dynamic(raw_edge_dyn: Tensor, topology: dict, scalers: dict,
                                                             device: torch.device):
    edge_dyn = (raw_edge_dyn.float() - scalers["edge_dynamic_mean"].to(raw_edge_dyn.device)) / scalers["edge_dynamic_std"].to(raw_edge_dyn.device)
    edge_static = topology["edge_static"].float()
    edge_static = (edge_static - scalers["edge_static_mean"].to(edge_static.device)) / scalers["edge_static_std"].to(edge_static.device)
    return torch.cat([edge_static, edge_dyn], dim=-1).to(device)




# Converts standardized model predictions to physical outputs and thresholds junction flood logits into occurrences.
def inverse_prediction_dict(predictions_model: dict, scalers: dict) -> dict:
    result = {}
    result["subcatchment"] = inverse_target(predictions_model["subcatchment"], scalers, "subcatchment").clamp_min(0.0)

    junction_depth = inverse_target_component(predictions_model["junction"][..., 0], scalers, "junction", 0).clamp_min(0.0)

    if "flood_probability_threshold" not in scalers:
        raise RuntimeError("Deployment flood threshold has not been stored.")

    flood_probability = torch.sigmoid(predictions_model["junction"][..., 1])
    flood_occurrence = (flood_probability >= float(scalers["flood_probability_threshold"])).float()
    result["junction"] = torch.stack([junction_depth, flood_occurrence], dim=-1)

    for group in ["channel", "outfall"]:
        result[group] = inverse_target(predictions_model[group], scalers, group).clamp_min(0.0)

    return result



# Rolls out one scenario hour-by-hour using latent state propagation and converts predictions back to physical units.
@torch.no_grad()
def rollout_scenario_latent(model, scenario, topology, scalers, device, start_step: int = FIRST_FORECAST_ORIGIN,
                            num_steps: Optional[int] = None, forecast_horizon: int = FORECAST_HORIZON,
                            forecast_stride: int = FORECAST_STRIDE, node_hidden: Optional[Tensor] = None,
                            channel_hidden: Optional[Tensor] = None):
    model.eval()

    if forecast_horizon != FORECAST_HORIZON: raise ValueError(f"forecast_horizon must be {FORECAST_HORIZON}.")
    if forecast_stride != FORECAST_STRIDE: raise ValueError(f"forecast_stride must be {FORECAST_STRIDE}.")
    if start_step < FIRST_FORECAST_ORIGIN:
        raise ValueError("A complete 30-day history is required. " f"start_step must be >= {FIRST_FORECAST_ORIGIN}.")

    T = int(scenario.weather_full.size(1))
    max_available_steps = T - 1 - start_step

    if max_available_steps < FORECAST_HORIZON:
        raise ValueError("Not enough future forcing for one forecast block.")

    if num_steps is None: num_steps = max_available_steps
    num_steps = min(int(num_steps), max_available_steps)

    num_complete_blocks = num_steps // FORECAST_STRIDE
    if num_complete_blocks < 1: raise ValueError("No complete one-hour forecast block.")
    num_steps = num_complete_blocks * FORECAST_STRIDE

    edge_index = topology["edge_index"].to(device)
    node_type = topology["node_type"].to(device)
    edge_type = topology["edge_type"].to(device)

    x_static_std, edge_static_std = build_standardized_static_graph(topology=topology, scalers=scalers, device=device)
    spatial_static = model.encode_nodes_one_timestep(
        x_t=x_static_std, edge_index=edge_index, node_type=node_type,
        edge_type=edge_type, edge_attr_t=edge_static_std
    )

    weather_raw = scenario.weather_full[0].float().to(device)
    weather_std = (weather_raw - scalers["weather_mean"].to(device)) / scalers["weather_std"].to(device)

    outputs = {"subcatchment": [], "junction": [], "channel": [], "outfall": []}
    forecast_blocks = []

    for block_start_offset in range(0, num_steps, FORECAST_STRIDE):
        t = start_step + block_start_offset
        first_prediction_step = t + 1
        last_prediction_step = t + FORECAST_HORIZON
        if last_prediction_step >= T: break

        origin_tensor = torch.tensor([t], dtype=torch.long, device=device)
        context_t = model.build_context_sequence_batch(
            weather_full_std=weather_std.unsqueeze(0), origin_steps=origin_tensor
        )[0, 0, :]

        forcing_block_std = weather_std[t + 1:t + FORECAST_HORIZON + 1, :]

        pred_std, node_hidden, channel_hidden = model.step_block(
            x_t=x_static_std, edge_attr_t=edge_static_std, context_t=context_t,
            forcing_block_std=forcing_block_std, edge_index=edge_index, node_type=node_type,
            edge_type=edge_type, node_hidden=node_hidden, channel_hidden=channel_hidden,
            spatial_static=spatial_static
        )

        pred_raw = inverse_prediction_dict(pred_std, scalers)

        for key in outputs:
            outputs[key].append(pred_raw[key].detach().cpu())

        forecast_blocks.append({
            "block_index": len(forecast_blocks),
            "forecast_origin_step": t,
            "first_prediction_step": first_prediction_step,
            "last_prediction_step": last_prediction_step,
            "num_prediction_steps": FORECAST_HORIZON,
        })

    for key in outputs:
        if outputs[key]: outputs[key] = torch.cat(outputs[key], dim=1)

    return {
        "predictions": outputs,
        "forecast_blocks": forecast_blocks,
        "forecast_horizon_steps": FORECAST_HORIZON,
        "forecast_stride_steps": FORECAST_STRIDE,
        "time_step_minutes": MODEL_TIME_STEP_MINUTES,
        "state_update_minutes": FORECAST_STRIDE * MODEL_TIME_STEP_MINUTES,
        "node_hidden": node_hidden,
        "channel_hidden": channel_hidden,
    }
