from pathlib import Path
import gc
import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from config import *
from data_loading import iter_record_batches, load_compact_scenarios_parallel
from preprocessing import (
    build_standardized_static_graph,
    build_standardized_transition_targets_batch,
    standardize_weather_batch,
    iter_forecast_origin_chunks,
)
from losses_metrics import (
    compute_state_transition_loss,
    init_stateful_metric_states,
    init_stateful_sparse_diagnostic_states,
    new_flood_classification_state,
    init_stateful_extreme_metric_states,
    update_stateful_metric_states,
    update_stateful_sparse_diagnostic_states,
    update_flood_classification_state,
    update_stateful_extreme_metric_states,
    finalize_stateful_metric_states,
    finalize_stateful_sparse_diagnostic_states,
    finalize_flood_classification_state,
    finalize_extreme_metric_state,
    inverse_target,
    inverse_target_component,
)



# Writes TEST predictions, observations, metadata, and flood probabilities directly to a local HDF5 file.
class LocalTestValueWriter:

    # Initializes the local HDF5 TEST file, metadata, and output datasets.
    def __init__(self, path, records, topology, flood_probability_threshold):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.first_target_step = FIRST_FORECAST_ORIGIN + 1
        self.num_steps = FULL_SEQUENCE_LENGTH - self.first_target_step
        if self.num_steps <= 0 or self.num_steps % FORECAST_HORIZON != 0:
            raise ValueError("Unexpected TEST prediction length.")

        self.file = h5py.File(self.path, "w")
        self.file.attrs["first_target_step"] = int(self.first_target_step)
        self.file.attrs["num_prediction_steps"] = int(self.num_steps)
        self.file.attrs["time_step_minutes"] = int(MODEL_TIME_STEP_MINUTES)
        self.file.attrs["forecast_horizon_steps"] = int(FORECAST_HORIZON)
        self.file.attrs["flood_probability_threshold"] = float(flood_probability_threshold)

        metadata = self.file.create_group("metadata")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        metadata.create_dataset("scenario_name", data=np.asarray([str(record["scenario_name"]) for record in records], dtype=object), dtype=string_dtype)
        metadata.create_dataset("subcatchment_name", data=np.asarray(topology["subcatchment_names"], dtype=object), dtype=string_dtype)
        metadata.create_dataset("junction_name", data=np.asarray(topology["junction_names"], dtype=object), dtype=string_dtype)
        metadata.create_dataset("channel_name", data=np.asarray(topology["downstream_conduit_names"], dtype=object), dtype=string_dtype)
        metadata.create_dataset("outfall_name", data=np.asarray(topology["outfall_names"], dtype=object), dtype=string_dtype)
        metadata.create_dataset("target_step", data=np.arange(self.first_target_step, FULL_SEQUENCE_LENGTH, dtype=np.int32))
        metadata.create_dataset(
            "lead_minutes_within_hour",
            data=np.tile(np.arange(1, FORECAST_HORIZON + 1, dtype=np.int16) * MODEL_TIME_STEP_MINUTES,
                         self.num_steps // FORECAST_HORIZON),
        )

        self.n_scenarios = len(records)
        self._create_continuous_group("snow_depth_mm", topology["num_subcatchments"])
        self._create_continuous_group("runoff_volume_m3_per_10min", topology["num_subcatchments"])
        self._create_continuous_group("junction_depth_m", topology["num_junctions"])
        self._create_continuous_group("channel_depth_m", topology["num_physical_channels"])
        self._create_continuous_group("outfall_depth_m", topology["num_outfall_nodes"])

        flood_group = self.file.create_group("flood_occurrence")
        flood_shape = (self.n_scenarios, self.num_steps, topology["num_junctions"])
        flood_chunks = (1, min(STEPS_PER_DAY, self.num_steps), topology["num_junctions"])
        flood_group.create_dataset("observed", shape=flood_shape, dtype=np.uint8, chunks=flood_chunks,
                                   compression="gzip", compression_opts=4, shuffle=True)
        flood_group.create_dataset("probability", shape=flood_shape, dtype=np.float32, chunks=flood_chunks,
                                   compression="gzip", compression_opts=4, shuffle=True)
        flood_group.create_dataset("predicted", shape=flood_shape, dtype=np.uint8, chunks=flood_chunks,
                                   compression="gzip", compression_opts=4, shuffle=True)

    # Creates compressed observed/predicted datasets for one continuous TEST variable.
    def _create_continuous_group(self, name, num_objects):
        group = self.file.create_group(name)
        shape = (self.n_scenarios, self.num_steps, num_objects)
        chunks = (1, min(STEPS_PER_DAY, self.num_steps), num_objects)
        for dataset_name in ["observed", "predicted"]:
            group.create_dataset(dataset_name, shape=shape, dtype=np.float32, chunks=chunks,
                                 compression="gzip", compression_opts=4, shuffle=True)

    # Converts model [B, objects, time] tensors to saved [B, time, objects] NumPy arrays.
    @staticmethod
    def _to_time_major(value):
        return value.detach().float().permute(0, 2, 1).contiguous().cpu().numpy()

    # Converts one TEST chunk to physical units and writes continuous and flood outputs to disk.
    def write_chunk(self, batch_start, target_start, target_end_exclusive, predictions, targets, scalers,
                    flood_probability_threshold):
        batch_size = predictions["subcatchment"].size(0)
        scenario_slice = slice(batch_start, batch_start + batch_size)
        local_start = target_start - self.first_target_step
        local_end = target_end_exclusive - self.first_target_step
        time_slice = slice(local_start, local_end)
        expected_steps = local_end - local_start
        if expected_steps != predictions["subcatchment"].size(2):
            raise ValueError("Mismatch between TEST writer time slice and model output length.")

        pred_sub = inverse_target(predictions["subcatchment"], scalers, "subcatchment").clamp_min(0.0)
        obs_sub = inverse_target(targets["subcatchment"], scalers, "subcatchment")
        pred_junction_depth = inverse_target_component(predictions["junction"][..., 0], scalers, "junction", 0).clamp_min(0.0)
        obs_junction_depth = inverse_target_component(targets["junction"][..., 0], scalers, "junction", 0)
        pred_channel = inverse_target(predictions["channel"], scalers, "channel").clamp_min(0.0)[..., 0]
        obs_channel = inverse_target(targets["channel"], scalers, "channel")[..., 0]
        pred_outfall = inverse_target(predictions["outfall"], scalers, "outfall").clamp_min(0.0)[..., 0]
        obs_outfall = inverse_target(targets["outfall"], scalers, "outfall")[..., 0]

        flood_probability = torch.sigmoid(predictions["junction"][..., 1])
        flood_observed = targets["junction"][..., 1]
        flood_predicted = (flood_probability >= float(flood_probability_threshold)).to(torch.uint8)

        self.file["snow_depth_mm/predicted"][scenario_slice, time_slice, :] = self._to_time_major(pred_sub[..., 0])
        self.file["snow_depth_mm/observed"][scenario_slice, time_slice, :] = self._to_time_major(obs_sub[..., 0])
        self.file["runoff_volume_m3_per_10min/predicted"][scenario_slice, time_slice, :] = self._to_time_major(pred_sub[..., 1])
        self.file["runoff_volume_m3_per_10min/observed"][scenario_slice, time_slice, :] = self._to_time_major(obs_sub[..., 1])
        self.file["junction_depth_m/predicted"][scenario_slice, time_slice, :] = self._to_time_major(pred_junction_depth)
        self.file["junction_depth_m/observed"][scenario_slice, time_slice, :] = self._to_time_major(obs_junction_depth)
        self.file["channel_depth_m/predicted"][scenario_slice, time_slice, :] = self._to_time_major(pred_channel)
        self.file["channel_depth_m/observed"][scenario_slice, time_slice, :] = self._to_time_major(obs_channel)
        self.file["outfall_depth_m/predicted"][scenario_slice, time_slice, :] = self._to_time_major(pred_outfall)
        self.file["outfall_depth_m/observed"][scenario_slice, time_slice, :] = self._to_time_major(obs_outfall)
        self.file["flood_occurrence/observed"][scenario_slice, time_slice, :] = self._to_time_major(flood_observed).astype(np.uint8)
        self.file["flood_occurrence/probability"][scenario_slice, time_slice, :] = self._to_time_major(flood_probability)
        self.file["flood_occurrence/predicted"][scenario_slice, time_slice, :] = self._to_time_major(flood_predicted).astype(np.uint8)
        self.file.flush()

    # Flushes and closes the local TEST HDF5 file.
    def close(self):
        if self.file is not None:
            self.file.flush()
            self.file.close()
            self.file = None





   # Evaluates the stateful latent digital twin over scenario batches and finalizes losses, regression, flood, sparse-event, extreme, and optional TEST-output diagnostics.
@torch.no_grad()
def evaluate_stateful_latent(model, records, topology, device, scalers, tbptt_blocks: int = TBPTT_BLOCKS_PER_CHUNK,
                             scenario_batch_size: int = BATCH_SIZE, flood_probability_threshold=None, test_value_writer=None):
    model.eval()
    records = list(records)

    loss_sums = {"total": 0.0, "subcatchment": 0.0, "junction": 0.0, "channel": 0.0, "outfall": 0.0}
    metric_states = init_stateful_metric_states()
    sparse_diagnostic_states = init_stateful_sparse_diagnostic_states()
    flood_classification_state = new_flood_classification_state()
    extreme_metric_states = init_stateful_extreme_metric_states(scalers)
    total_transitions = 0

    edge_index = topology["edge_index"].to(device)
    node_type = topology["node_type"].to(device)
    edge_type = topology["edge_type"].to(device)
    x_static_std, edge_static_std = build_standardized_static_graph(topology=topology, scalers=scalers, device=device)

    num_batches = (len(records) + scenario_batch_size - 1) // scenario_batch_size
    batch_iterator = iter_record_batches(records, scenario_batch_size)

    for batch_start, batch_records in tqdm(batch_iterator, total=num_batches, desc="Evaluation scenario batches"):
        scenarios = load_compact_scenarios_parallel(
            records=batch_records, topology=topology, start_scenario_idx=batch_start, max_workers=NUM_IO_WORKERS
        )

        actual_batch_size = len(scenarios)
        if actual_batch_size == 0: continue

        lengths = [int(scenario.weather_full.size(1)) for scenario in scenarios]
        if len(set(lengths)) != 1: raise ValueError(f"Scenario lengths differ: {lengths}")

        T = lengths[0]
        weather_full_std = standardize_weather_batch(scenarios=scenarios, scalers=scalers, device=device)
        node_hidden = channel_hidden = None

        for origin_chunk in iter_forecast_origin_chunks(sequence_length=T, blocks_per_chunk=tbptt_blocks):
            first_origin, last_origin, num_blocks = origin_chunk[0], origin_chunk[-1], len(origin_chunk)

            targets = build_standardized_transition_targets_batch(
                scenarios=scenarios, start=first_origin, end=last_origin + FORECAST_HORIZON,
                scalers=scalers, device=device
            )

            origin_steps = torch.tensor(origin_chunk, dtype=torch.long, device=device)
            target_start = first_origin + 1
            target_end_exclusive = last_origin + FORECAST_HORIZON + 1
            future_weather = weather_full_std[:, target_start:target_end_exclusive, :]
            forcing_blocks_std = future_weather.reshape(
                actual_batch_size, num_blocks, FORECAST_HORIZON, WEATHER_FEATURE_DIM
            )

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=USE_BF16 and device.type == "cuda"):
                predictions, node_hidden, channel_hidden = model.forward_latent_batch_chunk(
                    x_static_std=x_static_std, edge_static_std=edge_static_std, weather_full_std=weather_full_std,
                    forcing_blocks_std=forcing_blocks_std, origin_steps=origin_steps, edge_index=edge_index,
                    node_type=node_type, edge_type=edge_type, node_hidden=node_hidden, channel_hidden=channel_hidden
                )
                loss, loss_dict = compute_state_transition_loss(predictions, targets, scalers)

            n_target_steps = num_blocks * FORECAST_HORIZON
            batch_transitions = actual_batch_size * n_target_steps
            total_transitions += batch_transitions

            for key in loss_sums:
                loss_sums[key] += float(loss_dict[key].item()) * batch_transitions

            update_stateful_metric_states(metric_states, predictions, targets, scalers)
            update_stateful_sparse_diagnostic_states(sparse_diagnostic_states, predictions, targets, scalers)
            update_flood_classification_state(
                flood_classification_state, predictions["junction"][..., 1], targets["junction"][..., 1]
            )

            if test_value_writer is not None:
                if flood_probability_threshold is None:
                    raise ValueError("flood_probability_threshold is required when saving TEST values.")
                test_value_writer.write_chunk(
                    batch_start=batch_start, target_start=target_start, target_end_exclusive=target_end_exclusive,
                    predictions=predictions, targets=targets, scalers=scalers,
                    flood_probability_threshold=flood_probability_threshold
                )

            update_stateful_extreme_metric_states(extreme_metric_states, predictions, targets, scalers)

            del targets, predictions, loss, loss_dict, forcing_blocks_std, future_weather, origin_steps

        del scenarios, weather_full_std
        gc.collect()

    avg_losses = {key: value / max(total_transitions, 1) for key, value in loss_sums.items()}

    return {
        "loss": avg_losses,
        "metrics": finalize_stateful_metric_states(metric_states),
        "sparse_diagnostics": finalize_stateful_sparse_diagnostic_states(sparse_diagnostic_states),
        "flood_classification": finalize_flood_classification_state(
            flood_classification_state, fixed_threshold=flood_probability_threshold
        ),
        "extreme_diagnostics": {key: finalize_extreme_metric_state(value) for key, value in extreme_metric_states.items()},
        "num_transitions": total_transitions,
        "num_scenarios": len(records),
    }
