import gc
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from tqdm.auto import tqdm

from config import *
from data_loading import load_compact_scenarios_parallel




# Builds recurrent physical node-state slices containing snow depth, hydraulic depth, and flooding while excluding meteorological forcing.
def build_recurrent_node_dynamic_slice(scenario: Data, topology: dict, start: int, end: int) -> Tensor:
    if not (0 <= start < end <= scenario.x_dynamic_full.size(1)):
        raise ValueError(f"Invalid recurrent node slice: start={start}, end={end}")

    base = scenario.x_dynamic_full[:, start:end, :].permute(1, 0, 2).contiguous()
    length, num_nodes, _ = base.shape

    dyn = torch.zeros(length, num_nodes, RECURRENT_NODE_DYNAMIC_DIM, dtype=base.dtype, device=base.device)
    dyn[..., REC_NODE_WATER_DEPTH] = base[..., NODE_WATER_DEPTH_COL - NODE_STATIC_DIM]
    dyn[..., REC_NODE_FLOODING] = base[..., NODE_FLOODING_VOLUME_COL - NODE_STATIC_DIM]

    sub_nodes = topology["subcatchment_nodes"].to(base.device)
    snow = scenario.y_subcatchment_full[:, start:end, 0].transpose(0, 1).contiguous()
    dyn[:, sub_nodes, REC_NODE_SNOW_DEPTH] = snow

    return dyn







# Builds standardized recurrent node features by combining static graph attributes with dynamic physical states.
def build_recurrent_node_features_chunk(scenario: Data, topology: dict, start: int, end: int,
                                          scalers: dict, device: torch.device) -> Tensor:

    dyn = build_recurrent_node_dynamic_slice(scenario, topology, start, end).float()
    dyn_mean = scalers["recurrent_node_dynamic_mean"].to(dyn.device)
    dyn_std = scalers["recurrent_node_dynamic_std"].to(dyn.device)
    dyn = (dyn - dyn_mean) / dyn_std

    static = topology["x_static"].float()
    static = (static - scalers["x_static_mean"].to(static.device)) / scalers["x_static_std"].to(static.device)

    static = static.unsqueeze(0).expand(dyn.size(0), -1, -1)
    return torch.cat([static, dyn], dim=-1).contiguous().to(device)





# Builds standardized edge features by combining static and dynamic edge attributes.
def build_edge_features_chunk(scenario: Data, topology: dict, start: int, end: int, scalers: dict, device: torch.device) -> Tensor:

    edge_dyn = scenario.edge_dynamic_full[:, start:end, :].permute(1, 0, 2).contiguous().float()
    edge_dyn = (edge_dyn - scalers["edge_dynamic_mean"].to(edge_dyn.device)) / scalers["edge_dynamic_std"].to(edge_dyn.device)

    edge_static = topology["edge_static"].float()
    edge_static = (edge_static - scalers["edge_static_mean"].to(edge_static.device)) / scalers["edge_static_std"].to(edge_static.device)

    edge_static = edge_static.unsqueeze(0).expand(edge_dyn.size(0), -1, -1)
    return torch.cat([edge_static, edge_dyn], dim=-1).contiguous().to(device)




# Standardizes the complete meteorological forcing sequence.
def standardize_weather_full(scenario: Data, scalers: dict, device: torch.device) -> Tensor:
    weather = scenario.weather_full[0].float()
    weather = (weather - scalers["weather_mean"].to(weather.device)) / scalers["weather_std"].to(weather.device)
    return weather.contiguous().to(device)




# Builds standardized static node and edge features for the drainage-network graph.
def build_standardized_static_graph(topology: dict, scalers: dict, device: torch.device):
    x_static = topology["x_static"].float()
    x_static_std = (x_static - scalers["x_static_mean"].to(x_static.device)) / scalers["x_static_std"].to(x_static.device)

    edge_static = topology["edge_static"].float()
    edge_static_std = (edge_static - scalers["edge_static_mean"].to(edge_static.device)) / scalers["edge_static_std"].to(edge_static.device)

    return x_static_std.contiguous().to(device), edge_static_std.contiguous().to(device)




# Builds standardized next-step targets while keeping junction flooding as a binary classification target.
def build_standardized_transition_targets(scenario: Data, start: int, end: int, scalers: dict, device: torch.device) -> dict:

    target_start = start + 1
    target_end = end + 1

    def z(value, mean_key, std_key):
        value = value.float()
        return (value - scalers[mean_key].to(value.device)) / scalers[std_key].to(value.device)

    junction_raw = scenario.y_junction_full[:, target_start:target_end, :].float()

    junction_depth = junction_raw[..., 0:1]
    junction_depth_std = (
        junction_depth - scalers["y_junction_mean"][0:1].to(junction_depth.device)
    ) / scalers["y_junction_std"][0:1].to(junction_depth.device)

    flood_occurrence = (junction_raw[..., 1:2] > FLOOD_OCCURRENCE_VOLUME_THRESHOLD).float()

    return {
        "subcatchment": z(
            scenario.y_subcatchment_full[:, target_start:target_end, :],
            "y_subcatchment_mean", "y_subcatchment_std",
        ).to(device),

        "junction": torch.cat([junction_depth_std, flood_occurrence], dim=-1).to(device),

        "channel": z(
            scenario.y_channel_full[:, target_start:target_end, :],
            "y_channel_mean", "y_channel_std",
        ).to(device),

        "outfall": z(
            scenario.y_outfall_full[:, target_start:target_end, :],
            "y_outfall_mean", "y_outfall_std",
        ).to(device),
    }



# Builds batched standardized recurrent node features for multiple scenarios.
def build_recurrent_node_features_batch(scenarios, topology, start: int, end: int,
                                        scalers: dict, device: torch.device) -> Tensor:
    return torch.stack(
        [build_recurrent_node_features_chunk(scenario=scenario, topology=topology, start=start, end=end,
                                             scalers=scalers, device=device) for scenario in scenarios],
        dim=0,
    )




# Builds batched standardized edge features for multiple scenarios.
def build_edge_features_batch(scenarios, topology, start: int, end: int,
                              scalers: dict, device: torch.device) -> Tensor:

    return torch.stack(
        [build_edge_features_chunk(scenario=scenario, topology=topology, start=start, end=end,
                                   scalers=scalers, device=device) for scenario in scenarios],
        dim=0,
    )




# Standardizes and batches full weather sequences for multiple scenarios.
def standardize_weather_batch(scenarios, scalers: dict, device: torch.device) -> Tensor:

    return torch.stack(
        [standardize_weather_full(scenario=scenario, scalers=scalers, device=device) for scenario in scenarios],
        dim=0,
    )




# Builds batched standardized transition targets for all prediction groups.
def build_standardized_transition_targets_batch(scenarios, start: int, end: int,
                                                  scalers: dict, device: torch.device) -> dict:

    per_scenario_targets = [
        build_standardized_transition_targets(
            scenario=scenario, start=start, end=end, scalers=scalers, device=device
        )
        for scenario in scenarios
    ]

    return {
        group: torch.stack([target[group] for target in per_scenario_targets], dim=0)
        for group in ["subcatchment", "junction", "channel", "outfall"]
    }




# Yields consecutive TBPTT input ranges used to predict the following time steps.
def iter_tbptt_ranges(sequence_length: int, tbptt_length: int = TBPTT_LENGTH):

    last_input_exclusive = sequence_length - 1

    for start in range(0, last_input_exclusive, tbptt_length):
        end = min(start + tbptt_length, last_input_exclusive)
        if end > start: yield start, end




# Yields hourly non-overlapping forecast-origin chunks for stateful TBPTT processing.
def iter_forecast_origin_chunks(sequence_length: int, blocks_per_chunk: int = TBPTT_BLOCKS_PER_CHUNK):

    last_valid_origin = sequence_length - 1 - FORECAST_HORIZON

    if last_valid_origin < FIRST_FORECAST_ORIGIN:
        return

    origins = list(range(FIRST_FORECAST_ORIGIN, last_valid_origin + 1, FORECAST_STRIDE))

    for start in range(0, len(origins), blocks_per_chunk):
        chunk = origins[start:start + blocks_per_chunk]
        if chunk: yield chunk



  # Tracks streaming feature-wise means and standard deviations without storing all observations.
class RunningMoments:
    def __init__(self, feature_dim: int):
        self.n = 0
        self.sum = torch.zeros(feature_dim, dtype=torch.float64)
        self.sum_sq = torch.zeros(feature_dim, dtype=torch.float64)

    def update(self, value: Tensor):
        value = value.detach().cpu().reshape(-1, value.size(-1)).double()
        self.n += int(value.size(0))
        self.sum += value.sum(dim=0)
        self.sum_sq += (value * value).sum(dim=0)

    def finalize(self, eps: float = 1e-8):
        if self.n == 0: raise ValueError("RunningMoments received no observations.")

        mean = self.sum / self.n
        if self.n > 1: var = (self.sum_sq - self.n * mean * mean) / (self.n - 1)
        else: var = torch.zeros_like(mean)

        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var).clamp_min(eps)
        return mean.float(), std.float()




# Fits TRAIN-only streaming standardizers for static graph features, weather forcing, and SWMM prediction targets.
def fit_standardizers_streaming(records, topology, eps: float = 1e-8) -> dict:
    records = list(records)
    if not records: raise ValueError("No training records supplied for scaler fitting.")

    x_static = topology["x_static"].float()
    edge_static = topology["edge_static"].float()

    weather_stats = RunningMoments(WEATHER_FEATURE_DIM)
    y_sub_stats = RunningMoments(SUBCATCHMENT_OUT_DIM)
    y_junc_stats = RunningMoments(JUNCTION_OUT_DIM)
    y_chan_stats = RunningMoments(CHANNEL_OUT_DIM)
    y_outfall_stats = RunningMoments(OUTFALL_OUT_DIM)

    print(f"Fitting streaming standardizers from {len(records)} TRAIN scenarios (weather + hydraulic targets only)...")
    progress = tqdm(total=len(records), desc="Scaler pass (train only)")

    for batch_start in range(0, len(records), PARALLEL_LOAD_BATCH_SIZE):
        batch_end = min(batch_start + PARALLEL_LOAD_BATCH_SIZE, len(records))
        batch_records = records[batch_start:batch_end]

        scenarios = load_compact_scenarios_parallel(
            records=batch_records, topology=topology,
            start_scenario_idx=batch_start, max_workers=NUM_IO_WORKERS,
        )

        for scenario in scenarios:
            T = int(scenario.weather_full.size(1))

            for start in range(0, T, STATS_TIME_CHUNK):
                end = min(start + STATS_TIME_CHUNK, T)

                weather_stats.update(scenario.weather_full[0, start:end, :])
                y_sub_stats.update(scenario.y_subcatchment_full[:, start:end, :].permute(1, 0, 2))
                y_junc_stats.update(scenario.y_junction_full[:, start:end, :].permute(1, 0, 2))
                y_chan_stats.update(scenario.y_channel_full[:, start:end, :].permute(1, 0, 2))
                y_outfall_stats.update(scenario.y_outfall_full[:, start:end, :].permute(1, 0, 2))

            progress.update(1)

        del scenarios
        gc.collect()

    progress.close()

    weather_mean, weather_std = weather_stats.finalize(eps)
    y_sub_mean, y_sub_std = y_sub_stats.finalize(eps)
    y_junc_mean, y_junc_std = y_junc_stats.finalize(eps)
    y_chan_mean, y_chan_std = y_chan_stats.finalize(eps)
    y_outfall_mean, y_outfall_std = y_outfall_stats.finalize(eps)

    return {
        "x_static_mean": x_static.mean(dim=0), "x_static_std": x_static.std(dim=0).clamp_min(eps),
        "edge_static_mean": edge_static.mean(dim=0), "edge_static_std": edge_static.std(dim=0).clamp_min(eps),
        "weather_mean": weather_mean, "weather_std": weather_std,
        "y_subcatchment_mean": y_sub_mean, "y_subcatchment_std": y_sub_std,
        "y_junction_mean": y_junc_mean, "y_junction_std": y_junc_std,
        "y_channel_mean": y_chan_mean, "y_channel_std": y_chan_std,
        "y_outfall_mean": y_outfall_mean, "y_outfall_std": y_outfall_std,
    }




# Randomly samples flattened tensor values for lightweight statistical estimation.
def _sample_flat_tensor(value: Tensor, sample_size: int, generator: torch.Generator) -> Tensor:
    flat = value.detach().cpu().reshape(-1).float()

    if flat.numel() <= sample_size: return flat.clone()

    index = torch.randint(low=0, high=flat.numel(), size=(sample_size,), generator=generator)
    return flat[index]



# Fits TRAIN-only flood prevalence, class weight, and extreme-tail statistics from supervised targets.
def fit_training_aux_statistics_streaming(records, topology) -> dict:
    records = list(records)
    supervised_start = FIRST_FORECAST_ORIGIN + 1

    groups = ["snow_depth", "runoff_volume", "junction_depth", "channel", "outfall"]
    samples = {group: [] for group in groups}

    generator = torch.Generator()
    generator.manual_seed(TAIL_SAMPLING_SEED)

    flood_positive_count = 0
    flood_total_count = 0

    print("Fitting TRAIN-only flood/tail auxiliary statistics...")

    for batch_start in tqdm(range(0, len(records), PARALLEL_LOAD_BATCH_SIZE), desc="Aux statistics"):
        batch_records = records[batch_start:batch_start + PARALLEL_LOAD_BATCH_SIZE]

        scenarios = load_compact_scenarios_parallel(
            records=batch_records, topology=topology,
            start_scenario_idx=batch_start, max_workers=NUM_IO_WORKERS,
        )

        for scenario in scenarios:
            flood_volume = scenario.y_junction_full[:, supervised_start:, 1]

            flood_positive_count += int(
                (flood_volume > FLOOD_OCCURRENCE_VOLUME_THRESHOLD).sum().item()
            )
            flood_total_count += int(flood_volume.numel())

            target_map = {
                "snow_depth": scenario.y_subcatchment_full[:, supervised_start:, 0],
                "runoff_volume": scenario.y_subcatchment_full[:, supervised_start:, 1],
                "junction_depth": scenario.y_junction_full[:, supervised_start:, 0],
                "channel": scenario.y_channel_full[:, supervised_start:, 0],
                "outfall": scenario.y_outfall_full[:, supervised_start:, 0],
            }

            for group, values in target_map.items():
                samples[group].append(
                    _sample_flat_tensor(
                        value=values, sample_size=TAIL_SAMPLE_PER_SCENARIO, generator=generator
                    )
                )

        del scenarios
        gc.collect()

    flood_negative_count = flood_total_count - flood_positive_count

    if flood_positive_count == 0:
        raise RuntimeError("No flood-positive TRAIN samples.")

    flood_positive_weight = float(np.sqrt(flood_negative_count / flood_positive_count))

    tail_thresholds = {}

    for group in groups:
        values = torch.cat(samples[group], dim=0).numpy()
        q95, q99 = np.quantile(values, [0.95, 0.99])

        tail_thresholds[group] = {
            "q95": float(q95),
            "q99": float(q99),
            "sample_size": int(len(values)),
        }

    flood_prevalence = flood_positive_count / flood_total_count

    print()
    print("TRAIN flood prevalence:", f"{100 * flood_prevalence:.6f}%")
    print("Flood positive weight:", flood_positive_weight)
    print("TRAIN runoff Q95:", tail_thresholds["runoff_volume"]["q95"])
    print("TRAIN runoff Q99:", tail_thresholds["runoff_volume"]["q99"])
    print()

    return {
        "flood_positive_count": flood_positive_count,
        "flood_negative_count": flood_negative_count,
        "flood_prevalence": flood_prevalence,
        "flood_positive_weight": flood_positive_weight,
        "tail_thresholds": tail_thresholds,
        "tail_statistics_supervised_start": supervised_start,
        "tail_sample_per_scenario": TAIL_SAMPLE_PER_SCENARIO,
        "tail_sampling_seed": TAIL_SAMPLING_SEED,
    }
