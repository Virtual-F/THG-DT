from typing import Sequence, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from config import *
from utils import format_bytes
from data_loading import (
    path_name,
    load_real_swmm_static_topology,
    build_full_swmm_scenario_data,
)
from losses_metrics import update_metric_state, finalize_metric_state






# Defines scenario cache keys, estimates tensor memory usage, and optionally rounds floating-point tensors in place.
def scenario_cache_key(record, round_decimals=SCENARIO_CACHE_ROUND_DECIMALS) -> tuple:
    return (
        str(record["scenario_output_dir"]).rstrip("/"), str(record["scenario_input_path"]), round_decimals,
        "compact_v3_depth_flood_snow_runoff_outfall_no_storage", NUM_NODES, NUM_GRAPH_EDGES,
        NODE_FEATURE_DIM, EDGE_FEATURE_DIM, NODE_DYNAMIC_DIM, EDGE_DYNAMIC_DIM,
        SUBCATCHMENT_OUT_DIM, JUNCTION_OUT_DIM, CHANNEL_OUT_DIM, OUTFALL_OUT_DIM,
    )


def data_tensor_bytes(data: Data) -> int:
    total = 0
    for key in data.keys():
        value = data[key]
        if torch.is_tensor(value): total += value.numel() * value.element_size()
    return total


def round_float_tensors_in_data_(data: Data, decimals):
    if decimals is None: return data
    scale = float(10 ** decimals)
    for key in data.keys():
        value = data[key]
        if torch.is_tensor(value) and value.is_floating_point(): value.mul_(scale).round_().div_(scale)
    return data






# Caches preloaded SWMM scenarios in RAM, enforces cache limits, tracks usage, and retrieves cached scenarios.
class ScenarioRamCache:
    def __init__(self, max_bytes=SCENARIO_RAM_CACHE_MAX_BYTES, max_items=SCENARIO_RAM_CACHE_MAX_ITEMS,
                 round_decimals=SCENARIO_CACHE_ROUND_DECIMALS):
        self.max_bytes = max_bytes
        self.max_items = max_items
        self.round_decimals = round_decimals
        self.data = {}
        self.item_bytes = {}
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def key(self, record) -> tuple:
        return scenario_cache_key(record, round_decimals=self.round_decimals)

    def contains(self, record) -> bool:
        return self.key(record) in self.data

    def _assert_can_store(self, record, scenario_bytes: int) -> None:
        scenario_name = path_name(record["scenario_output_dir"])

        if self.max_items is not None and len(self.data) >= self.max_items:
            raise MemoryError(
                "Scenario RAM preload exceeded item limit"
                f"Cannot store scenario={scenario_name}. cached_items={len(self.data)}, "
                f"max_items={self.max_items}, cached_bytes={format_bytes(self.bytes)}."
            )

        if self.max_bytes is not None and self.bytes + scenario_bytes > self.max_bytes:
            raise MemoryError(
                "Scenario RAM preload exceeded byte limit before training. "
                f"Cannot store scenario={scenario_name}. scenario_size={format_bytes(scenario_bytes)}, "
                f"cached_before={format_bytes(self.bytes)}, would_be={format_bytes(self.bytes + scenario_bytes)}, "
                f"max_bytes={format_bytes(self.max_bytes)}. "
                "Reduce MAX_TRAIN_RECORDS / MAX_VAL_RECORDS or increase SCENARIO_RAM_CACHE_MAX_BYTES."
            )

    def build_and_add(self, record, topology, scenario_id):
        key = self.key(record)

        if key in self.data:
            return self.get(record=record, scenario_id=scenario_id)

        scenario = build_full_swmm_scenario_data(
            topology=topology, scenario_output_dir=record["scenario_output_dir"],
            scenario_input_path=record["scenario_input_path"], scenario_id=scenario_id, compact=True,
        )

        round_float_tensors_in_data_(scenario, self.round_decimals)
        scenario_bytes = data_tensor_bytes(scenario)
        self._assert_can_store(record=record, scenario_bytes=scenario_bytes)

        self.data[key] = scenario
        self.item_bytes[key] = scenario_bytes
        self.bytes += scenario_bytes
        return scenario

    def get(self, record, scenario_id=None):
        key = self.key(record)

        if key not in self.data:
            self.misses += 1
            scenario_name = path_name(record["scenario_output_dir"])
            raise KeyError(
                "Scenario RAM cache MISS"
                "training tried to read a scenario that was not preloaded"
                f"Missing scenario={scenario_name}."
            )

        self.hits += 1
        scenario = self.data[key]
        if scenario_id is not None:
            scenario.scenario_id = torch.tensor([scenario_id], dtype=torch.long)
        return scenario
    def summary(self) -> str:
        max_bytes_text = "unlimited" if self.max_bytes is None else format_bytes(self.max_bytes)
        return (
            f"{len(self.data)} cached scenarios, {format_bytes(self.bytes)} / {max_bytes_text}, "
            f"hits={self.hits}, misses={self.misses}"
        )





# Preloads selected scenarios into RAM and retrieves cached scenarios for training or evaluation.
def preload_selected_scenarios_into_ram(records, topology, scenario_cache: ScenarioRamCache, split_name: str, max_records=None):
    records = list(records if max_records is None else records[:max_records])

    print(f"Preloading {len(records)} selected {split_name} scenarios from Allas/S3 into RAM before training...")

    for scenario_idx, record in enumerate(tqdm(records, desc=f"Preloading {split_name}")):
        scenario_cache.build_and_add(record=record, topology=topology, scenario_id=scenario_idx)

    print(f"Finished preloading {split_name}. Scenario RAM cache: {scenario_cache.summary()}")
    return records


def load_scenario_for_record(record, topology, scenario_idx, scenario_cache=None):
    if scenario_cache is None:
        raise RuntimeError(
            "scenario_cache is required during training/evaluation. "
            "Preload scenarios first with preload_selected_scenarios_into_ram(...)."
        )

    return scenario_cache.get(record=record, scenario_id=scenario_idx)






# Generates full-event PyG scenario objects from SWMM outputs using the real static topology.
def generate_swmm_scenario_dataset(scenario_records: Sequence[Dict[str, Any]], graph_path: str = REAL_STATIC_GRAPH_PATH) -> list:

    topology = load_real_swmm_static_topology(
        graph_path=graph_path, node_table_path=NODE_TABLE_PATH, edge_table_path=EDGE_TABLE_PATH
    )

    dataset = []

    for i, record in enumerate(scenario_records):
        data = build_full_swmm_scenario_data(
            topology=topology,
            scenario_output_dir=record["scenario_output_dir"],
            scenario_input_path=record["scenario_input_path"],
            scenario_id=i,
        )
        dataset.append(data)

    return dataset





# Deprecated helper that would convert full scenarios into rolling forecast samples while hiding future hydraulic states.
def create_rolling_forecast_samples(full_scenarios: list) -> list:

    raise RuntimeError(
        "create_rolling_forecast_samples is deprecated. "
        "Use iter_forecast_start_chunks_from_scenario + build_rolling_forecast_chunk instead."
    )

    samples = []

    for scenario in full_scenarios:
        for forecast_start in range(LOOKBACK_LENGTH, FULL_SEQUENCE_LENGTH - FORECAST_HORIZON + 1, WINDOW_STRIDE):
            hist_start = forecast_start - LOOKBACK_LENGTH
            hist_end = forecast_start
            future_start = forecast_start
            future_end = forecast_start + FORECAST_HORIZON
            input_start = hist_start
            input_end = future_end

            x_hist = scenario.x_full[:, input_start:input_end, :].clone()
            edge_attr_hist = scenario.edge_attr_full[:, input_start:input_end, :].clone()
            weather_future = scenario.weather_full[:, input_start:input_end, :].clone()

            y_subcatchment_seq = scenario.y_subcatchment_full[:, future_start:future_end, :]
            y_junction_seq = scenario.y_junction_full[:, future_start:future_end, :]
            y_channel_seq = scenario.y_channel_full[:, future_start:future_end, :]

            # Compatibility snapshot: current observed state is t13, not t17.
            x_current = x_hist[:, LOOKBACK_LENGTH - 1, :]
            edge_attr_current = edge_attr_hist[:, LOOKBACK_LENGTH - 1, :]

            sample = Data(
                x=x_current.float(), edge_index=scenario.edge_index, edge_attr=edge_attr_current.float(),
                node_type=scenario.node_type, edge_type=scenario.edge_type,
                x_hist=x_hist.float(), edge_attr_hist=edge_attr_hist.float(), weather_future=weather_future.float(),
                y_subcatchment_seq=y_subcatchment_seq.float(), y_junction_seq=y_junction_seq.float(),
                y_channel_seq=y_channel_seq.float(),
            )

            sample.scenario_id = scenario.scenario_id
            sample.forecast_start = torch.tensor([forecast_start], dtype=torch.long)
            samples.append(sample)

    return samples



# Updates epoch-level hydrological metric states, optionally converting standardized values back to physical units.
def update_epoch_metric_states(metric_states: dict, predictions: dict, batch: Data, scalers=None) -> None:
    def _inverse(value, mean_key, std_key):
        if scalers is None: return value
        mean = scalers[mean_key].to(value.device)
        std = scalers[std_key].to(value.device)
        return value * std + mean

    pred_sub = _inverse(predictions["subcatchment"], "y_subcatchment_mean", "y_subcatchment_std")
    true_sub = _inverse(batch.y_subcatchment_seq, "y_subcatchment_mean", "y_subcatchment_std")

    pred_junc = _inverse(predictions["junction"], "y_junction_mean", "y_junction_std")
    true_junc = _inverse(batch.y_junction_seq, "y_junction_mean", "y_junction_std")

    pred_chan = _inverse(predictions["channel"], "y_channel_mean", "y_channel_std")
    true_chan = _inverse(batch.y_channel_seq, "y_channel_mean", "y_channel_std")

    update_metric_state(metric_states["subcatchment"], pred_sub, true_sub)
    update_metric_state(metric_states["junction"], pred_junc, true_junc)
    update_metric_state(metric_states["channel"], pred_chan, true_chan)

    update_metric_state(metric_states["all"], pred_sub, true_sub)
    update_metric_state(metric_states["all"], pred_junc, true_junc)
    update_metric_state(metric_states["all"], pred_chan, true_chan)




# Finalizes all epoch-level metric groups into their computed hydrological metrics.
def finalize_epoch_metric_states(metric_states: dict) -> dict:
    return {key: finalize_metric_state(value) for key, value in metric_states.items()}



# Computes the weighted multi-task MSE loss for subcatchment, junction, and channel predictions.
def compute_multi_task_loss(predictions: dict, batch: Data) -> tuple:
    loss_subcatchment = F.mse_loss(predictions["subcatchment"], batch.y_subcatchment_seq)
    loss_junction = F.mse_loss(predictions["junction"], batch.y_junction_seq)
    loss_channel = F.mse_loss(predictions["channel"], batch.y_channel_seq)

    total_loss = (
        LOSS_WEIGHT_SUBCATCHMENT * loss_subcatchment
        + LOSS_WEIGHT_JUNCTION * loss_junction
        + LOSS_WEIGHT_CHANNEL * loss_channel)

    loss_dict = {
        "total": total_loss,"subcatchment": loss_subcatchment,"junction": loss_junction,"channel": loss_channel}
    return total_loss, loss_dict



    

# Keeps the deprecated DataLoader evaluator disabled and defines the active stateful digital-twin configuration.
@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    raise RuntimeError(
        "evaluate_model is deprecated because it uses rolling-sample DataLoader "
        "evaluation and cannot remove overlapping-window spatial redundancy. "
        "Use evaluate_streaming(...) instead."
    )




# Deprecated physical-state reconstruction path retained for compatibility with the former closed-loop rollout design.
def build_next_raw_physical_state(predictions_raw: dict, topology: dict, edge_maps: dict, device: torch.device):
    raise RuntimeError(
        "build_next_raw_physical_state() is deprecated. "
        "The current digital twin uses latent-state propagation "
        "and junction flooding is now binary occurrence, not volume."
    )

    num_nodes, num_edges = topology["num_nodes"], topology["num_edges"]
    node_dyn = torch.zeros(num_nodes, RECURRENT_NODE_DYNAMIC_DIM, dtype=torch.float32, device=device)

    sub_nodes = topology["subcatchment_nodes"].to(device)
    junc_nodes = topology["junction_nodes"].to(device)
    storage_nodes = topology["storage_nodes"].to(device)
    outfall_nodes = topology["outfall_nodes"].to(device)

    node_dyn[sub_nodes, REC_NODE_SNOW_DEPTH] = predictions_raw["subcatchment"][:, 0]
    node_dyn[junc_nodes, REC_NODE_WATER_DEPTH] = predictions_raw["junction"][:, 0]
    node_dyn[junc_nodes, REC_NODE_FLOODING] = predictions_raw["junction"][:, 1]
    node_dyn[storage_nodes, REC_NODE_WATER_DEPTH] = predictions_raw["storage"][:, 0]
    node_dyn[outfall_nodes, REC_NODE_WATER_DEPTH] = predictions_raw["outfall"][:, 0]

    edge_dyn = torch.zeros(num_edges, EDGE_DYNAMIC_DIM, dtype=torch.float32, device=device)
    edge_type = topology["edge_type"].to(device)

    runoff_edges = torch.where(edge_type == RUNOFF_EDGE_TYPE)[0]
    if runoff_edges.numel() > 0:
        sub_idx = edge_maps["runoff_sub_idx"][runoff_edges]
        edge_dyn[runoff_edges, 0] = predictions_raw["subcatchment"][sub_idx, 1]

    conduit_edges = torch.where(
        (edge_type == DOWNSTREAM_CHANNEL_EDGE_TYPE) | (edge_type == REVERSE_BACKWATER_EDGE_TYPE)
    )[0]
    if conduit_edges.numel() > 0:
        chan_idx = edge_maps["conduit_channel_idx"][conduit_edges]
        edge_dyn[conduit_edges, 0] = predictions_raw["channel"][chan_idx, 0]

    return node_dyn, edge_dyn
