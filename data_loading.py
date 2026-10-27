from typing import Sequence, Dict, Any, List
from pathlib import Path, PurePosixPath
import re
import random
import numpy as np
import pandas as pd
import torch
import s3fs
from torch import Tensor
from torch_geometric.data import Data
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

from config import *

S3 = s3fs.S3FileSystem(client_kwargs={"endpoint_url": ALLAS_ENDPOINT})

def is_s3_path(path) -> bool:
    path = str(path)
    return path.startswith("s3://") or path.startswith(f"{ALLAS_BUCKET}/")


def s3_key(path: str) -> str:
    path = str(path)
    if path.startswith("s3://"): path = path[5:]
    return path


def join_uri(base: str, *parts: str) -> str:
    return str(base).rstrip("/") + "/" + "/".join(str(p).strip("/") for p in parts)


def path_name(path: str) -> str:
    return PurePosixPath(str(path).rstrip("/")).name


def read_parquet_any(path: str) -> pd.DataFrame:
    path = str(path)
    if is_s3_path(path):
        with S3.open(s3_key(path), "rb") as f: return pd.read_parquet(f)
    return pd.read_parquet(path)


def read_csv_any(path: str) -> pd.DataFrame:
    path = str(path)
    if is_s3_path(path):
        with S3.open(s3_key(path), "rb") as f: return pd.read_csv(f)
    return pd.read_csv(path)





# Adds Allas-aware pandas readers and activates them globally.
def read_csv_allas_aware(filepath_or_buffer, *args, **kwargs):
    if isinstance(filepath_or_buffer, str) and is_s3_path(filepath_or_buffer):
        with S3.open(s3_key(filepath_or_buffer), "rb") as f: return pd._allas_original_read_csv(f, *args, **kwargs)
    return pd._allas_original_read_csv(filepath_or_buffer, *args, **kwargs)


def read_parquet_allas_aware(filepath_or_buffer, *args, **kwargs):
    if isinstance(filepath_or_buffer, str) and is_s3_path(filepath_or_buffer):
        with S3.open(s3_key(filepath_or_buffer), "rb") as f: return pd._allas_original_read_parquet(f, *args, **kwargs)
    return pd._allas_original_read_parquet(filepath_or_buffer, *args, **kwargs)

if not hasattr(pd, "_allas_original_read_csv"): pd._allas_original_read_csv = pd.read_csv
if not hasattr(pd, "_allas_original_read_parquet"): pd._allas_original_read_parquet = pd.read_parquet

pd.read_csv = read_csv_allas_aware
pd.read_parquet = read_parquet_allas_aware
print("Allas/S3 helpers loaded.")





# Loads PyTorch files safely from local storage or Allas/S3 across PyTorch versions.
def torch_load_safe(path: str):
    path = str(path)
    if is_s3_path(path):
        with S3.open(s3_key(path), "rb") as f:
            try: return torch.load(f, map_location="cpu", weights_only=False)
            except TypeError:
                f.seek(0)
                return torch.load(f, map_location="cpu")
    try: return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(path, map_location="cpu")



# Loads and validates the SWMM static topology and prepares graph masks, mappings, names, and attributes.
def load_real_swmm_static_topology(graph_path: str = REAL_STATIC_GRAPH_PATH, node_table_path: str = NODE_TABLE_PATH, edge_table_path: str = EDGE_TABLE_PATH) -> dict:
    graph = torch_load_safe(graph_path)
    x_static_full = graph["x_static"].float()
    edge_index_full = graph["edge_index"].long()
    edge_type_full = graph["edge_type"].long()
    edge_static_full = graph["edge_static"].float()
    node_feature_names = graph["node_feature_names"]
    edge_feature_names = graph["edge_feature_names"]
    node_feature_index = {name: i for i, name in enumerate(node_feature_names)}
    edge_feature_index = {name: i for i, name in enumerate(edge_feature_names)}

    node_table = pd.read_csv(node_table_path).sort_values("node_id").reset_index(drop=True)
    edge_table = pd.read_csv(edge_table_path).sort_values("edge_id").reset_index(drop=True)
    if len(node_table) != x_static_full.size(0): raise ValueError(f"node_table has {len(node_table)} rows, but x_static has {x_static_full.size(0)} nodes.")
    if len(edge_table) != edge_static_full.size(0): raise ValueError(f"edge_table has {len(edge_table)} rows, but edge_static has {edge_static_full.size(0)} edges.")

    x_static = x_static_full
    edge_index = edge_index_full.contiguous()
    edge_type = edge_type_full.contiguous()
    edge_static = edge_static_full.contiguous()
    reduced_node_table = node_table.copy().reset_index(drop=True)
    reduced_node_table["new_node_id"] = reduced_node_table["node_id"]
    reduced_edge_table = edge_table.copy().reset_index(drop=True)
    reduced_edge_table["new_edge_id"] = reduced_edge_table["edge_id"]
    node_type = torch.tensor(reduced_node_table["node_type"].to_numpy(), dtype=torch.long)

    subcatchment_nodes = torch.where(node_type == SUBCATCHMENT_TYPE)[0]
    junction_nodes = torch.where(node_type == JUNCTION_TYPE)[0]
    storage_nodes = torch.where(node_type == STORAGE_TYPE)[0]
    outfall_nodes = torch.where(node_type == OUTFALL_TYPE)[0]
    hydraulic_nodes = torch.where((node_type == JUNCTION_TYPE) | (node_type == STORAGE_TYPE) | (node_type == OUTFALL_TYPE))[0]
    pump_mask = edge_type == PUMP_EDGE_TYPE
    pump_reverse_mask = edge_type == PUMP_REVERSE_EDGE_TYPE
    runoff_mask = edge_type == RUNOFF_EDGE_TYPE
    downstream_channel_mask = edge_type == DOWNSTREAM_CHANNEL_EDGE_TYPE
    reverse_channel_mask = edge_type == REVERSE_BACKWATER_EDGE_TYPE

    num_nodes = x_static.size(0)
    num_edges = edge_index.size(1)
    num_subcatchments = int(subcatchment_nodes.numel())
    num_junctions = int(junction_nodes.numel())
    num_storage_nodes = int(storage_nodes.numel())
    num_outfall_nodes = int(outfall_nodes.numel())
    num_hydraulic_nodes = int(hydraulic_nodes.numel())
    num_physical_channels = int(downstream_channel_mask.sum())
    num_pump_links = int(pump_mask.sum())

    assert num_nodes == NUM_NODES, f"Expected {NUM_NODES} nodes, got {num_nodes}"
    assert num_edges == NUM_GRAPH_EDGES, f"Expected {NUM_GRAPH_EDGES} edges, got {num_edges}"
    assert num_subcatchments == NUM_SUBCATCHMENTS
    assert num_junctions == NUM_JUNCTIONS
    assert num_storage_nodes == NUM_STORAGE_NODES
    assert num_outfall_nodes == NUM_OUTFALL_NODES
    assert num_hydraulic_nodes == NUM_HYDRAULIC_NODES
    assert num_physical_channels == NUM_PHYSICAL_CHANNELS
    assert num_pump_links == NUM_PUMP_LINKS

    global_to_subcatchment = torch.full((num_nodes,), -1, dtype=torch.long)
    global_to_subcatchment[subcatchment_nodes] = torch.arange(num_subcatchments, dtype=torch.long)
    global_to_hydraulic = torch.full((num_nodes,), -1, dtype=torch.long)
    global_to_hydraulic[hydraulic_nodes] = torch.arange(num_hydraulic_nodes, dtype=torch.long)

    node_names = reduced_node_table["swmm_name"].astype(str).tolist()
    subcatchment_names = reduced_node_table.loc[reduced_node_table["node_type"] == 0, "swmm_name"].astype(str).tolist()
    junction_names = reduced_node_table.loc[reduced_node_table["node_type"] == 1, "swmm_name"].astype(str).tolist()
    storage_names = reduced_node_table.loc[reduced_node_table["node_type"] == STORAGE_TYPE, "swmm_name"].astype(str).tolist()
    outfall_names = reduced_node_table.loc[reduced_node_table["node_type"] == OUTFALL_TYPE, "swmm_name"].astype(str).tolist()
    hydraulic_names = reduced_node_table.loc[reduced_node_table["node_type"].isin([JUNCTION_TYPE, STORAGE_TYPE, OUTFALL_TYPE]), "swmm_name"].astype(str).tolist()
    pump_names = reduced_edge_table.loc[reduced_edge_table["edge_type"] == PUMP_EDGE_TYPE, "original_link_name"].astype(str).tolist()
    edge_original_link_names = reduced_edge_table["original_link_name"].fillna("").astype(str).tolist()
    downstream_conduit_names = reduced_edge_table.loc[reduced_edge_table["edge_type"] == DOWNSTREAM_CHANNEL_EDGE_TYPE, "original_link_name"].astype(str).tolist()

    sub_area = x_static[subcatchment_nodes, node_feature_index["sub_area"]]
    sub_imperviousness = x_static[subcatchment_nodes, node_feature_index["sub_impervious_frac"]]
    sub_surface_slope = x_static[subcatchment_nodes, node_feature_index["sub_slope_frac"]]
    invert_elevation = x_static[hydraulic_nodes, node_feature_index["node_invert_elevation"]]
    storage_proxy = x_static[hydraulic_nodes, node_feature_index["node_max_depth"]].clamp_min(0.10)

    return {
        "num_subcatchments": num_subcatchments, "num_junctions": num_junctions, "num_nodes": num_nodes, "num_edges": num_edges, "num_physical_channels": num_physical_channels,
        "node_type": node_type, "edge_index": edge_index, "edge_type": edge_type, "num_storage_nodes": num_storage_nodes, "num_outfall_nodes": num_outfall_nodes, "num_hydraulic_nodes": num_hydraulic_nodes,
        "storage_nodes": storage_nodes, "outfall_nodes": outfall_nodes, "junction_nodes": junction_nodes,
        "storage_names": storage_names, "outfall_names": outfall_names, "hydraulic_names": hydraulic_names, "pump_names": pump_names,
        "pump_mask": pump_mask, "pump_reverse_mask": pump_reverse_mask, "x_static": x_static, "edge_static": edge_static,
        "node_feature_names": node_feature_names, "edge_feature_names": edge_feature_names, "node_feature_index": node_feature_index, "edge_feature_index": edge_feature_index,
        "node_table": reduced_node_table, "edge_table": reduced_edge_table,
        "node_names": node_names, "subcatchment_names": subcatchment_names, "junction_names": junction_names, "downstream_conduit_names": downstream_conduit_names, "edge_original_link_names": edge_original_link_names,
        "subcatchment_nodes": subcatchment_nodes, "hydraulic_nodes": hydraulic_nodes, "global_to_subcatchment": global_to_subcatchment, "global_to_hydraulic": global_to_hydraulic,
        "runoff_mask": runoff_mask, "downstream_channel_mask": downstream_channel_mask, "reverse_channel_mask": reverse_channel_mask,
        "sub_area": sub_area, "sub_imperviousness": sub_imperviousness, "sub_surface_slope": sub_surface_slope, "invert_elevation": invert_elevation, "storage": storage_proxy,
    }





# Reads selected SWMM wide-format variables and returns them as a [num_objects, T] tensor.
def read_wide_swmm_matrix(df: pd.DataFrame, object_ids: Sequence[str], variable_suffix: str, file_label: str) -> Tensor:
    columns = [f"{object_id}__{variable_suffix}" for object_id in object_ids]
    missing = [column for column in columns if column not in df.columns]
    if missing: raise ValueError(f"{file_label} is missing {len(missing)} columns. First missing columns: {missing[:10]}")
    return torch.from_numpy(df[columns].to_numpy(dtype=np.float32).T.copy())


    

# Finds valid forcing/output scenario pairs in Allas/S3 and builds scenario records.
def build_scenario_records(output_root: str = SWMM_SCENARIO_OUTPUT_ROOT, input_root: str = SCENARIO_INPUT_ROOT) -> List[Dict[str, Any]]:
    output_root, input_root = str(output_root).rstrip("/"), str(input_root).rstrip("/")
    excluded_names = {"junctions.parquet", "subcatchments.parquet", "channels.parquet", "storages.parquet",
                      "outfalls.parquet", "pumps.parquet", "network_metrics.parquet"}
    scenario_pattern = re.compile(r"_\d{4}_W")

    print("Searching forcing parquet files directly in Allas/S3...")
    input_files = []
    for key in S3.find(s3_key(input_root)):
        p = PurePosixPath(key)
        if key.endswith(".parquet") and p.name.lower() not in excluded_names and scenario_pattern.search(p.stem):
            input_files.append(key)

    input_index = {path_name(p).replace(".parquet", ""): p for p in input_files}
    if not input_index: raise FileNotFoundError(f"No forcing scenario parquet files found under:\n{input_root}")

    print("Searching SWMM output files directly in Allas/S3...")
    required_files = {"junctions.parquet", "subcatchments.parquet", "channels.parquet", "outfalls.parquet"}
    by_scenario = {}

    for key in S3.find(s3_key(output_root)):
        p, filename = PurePosixPath(key), PurePosixPath(key).name
        if filename not in required_files: continue
        folder_key, scenario_name = str(p.parent), PurePosixPath(p.parent).name
        if scenario_name not in by_scenario: by_scenario[scenario_name] = {"folder": folder_key, "files": set()}
        by_scenario[scenario_name]["files"].add(filename)

    output_dirs, incomplete_outputs = [], []
    for scenario_name, info in by_scenario.items():
        if required_files.issubset(info["files"]): output_dirs.append((scenario_name, info["folder"]))
        else: incomplete_outputs.append(scenario_name)
    output_dirs = sorted(output_dirs, key=lambda x: x[0])

    records, missing_inputs = [], []
    for scenario_name, output_dir in output_dirs:
        scenario_input_path = input_index.get(scenario_name)
        if scenario_input_path is None:
            missing_inputs.append(scenario_name)
            continue
        records.append({"scenario_name": scenario_name, "scenario_output_dir": output_dir, "scenario_input_path": scenario_input_path})

    if not records:
        raise FileNotFoundError(
            "No matching scenario records found.\n"
            f"Output root: {output_root}\nInput root: {input_root}\n"
            f"Incomplete output folders: {len(incomplete_outputs)}\nMissing input files: {len(missing_inputs)}"
        )

    print("\n========== Scenario record summary ==========")
    print(f"Valid scenarios:{len(records)}")
    print(f"Incomplete output folders:{len(incomplete_outputs)}")
    print(f"Missing forcing input files: {len(missing_inputs)}")
    print("============================================\n")
    return records





# Builds weather forcing features from precipitation and temperature for each time step.
def build_weather_from_forcing_df(forcing_df: pd.DataFrame) -> Tensor:
    required_columns = ["precipitation_mm_10min", "temperature_C"]
    missing = [col for col in required_columns if col not in forcing_df.columns]
    if missing: raise ValueError(f"Forcing file is missing required columns: {missing}")
    precip = torch.from_numpy(forcing_df["precipitation_mm_10min"].to_numpy(dtype=np.float32).copy()).clamp_min(0.0)
    temp = torch.from_numpy(forcing_df["temperature_C"].to_numpy(dtype=np.float32).copy())
    snowfall_proxy = torch.where(temp < 0.0, precip, torch.zeros_like(precip))
    melt_factor_proxy = torch.clamp(temp, min=0.0) * 0.05
    weather_full = torch.stack([precip, temp, snowfall_proxy, melt_factor_proxy], dim=1).unsqueeze(0)
    return weather_full.float()



# Builds one full SWMM scenario by combining forcing, graph topology, dynamic states, and prediction targets.
def build_full_swmm_scenario_data(topology: dict, scenario_output_dir: Path, scenario_input_path: Path,
                                  scenario_id: int, compact: bool = False) -> Data:

    scenario_output_dir = str(scenario_output_dir).rstrip("/")
    scenario_input_path = str(scenario_input_path)
    junction_df = read_parquet_any(join_uri(scenario_output_dir, "junctions.parquet"))
    subcatchment_df = read_parquet_any(join_uri(scenario_output_dir, "subcatchments.parquet"))
    channel_df = read_parquet_any(join_uri(scenario_output_dir, "channels.parquet"))
    outfall_df = read_parquet_any(join_uri(scenario_output_dir, "outfalls.parquet"))
    forcing_df = read_parquet_any(scenario_input_path)

    if "datetime" in junction_df.columns: junction_df = junction_df.sort_values("datetime").reset_index(drop=True)
    if "datetime" in subcatchment_df.columns: subcatchment_df = subcatchment_df.sort_values("datetime").reset_index(drop=True)
    if "datetime" in channel_df.columns: channel_df = channel_df.sort_values("datetime").reset_index(drop=True)
    if "datetime" in outfall_df.columns: outfall_df = outfall_df.sort_values("datetime").reset_index(drop=True)
    if "datetime_relative" in forcing_df.columns: forcing_df = forcing_df.sort_values("datetime_relative").reset_index(drop=True)

    T = len(junction_df)

    if T != FULL_SEQUENCE_LENGTH:
        raise ValueError(f"Expected FULL_SEQUENCE_LENGTH={FULL_SEQUENCE_LENGTH}, got T={T} for {scenario_output_dir}")

    if (len(subcatchment_df) != T or len(channel_df) != T or len(outfall_df) != T or len(forcing_df) != T):
        raise ValueError(
            "Input/output time lengths are not consistent for "
            f"{path_name(scenario_output_dir)}: "
            f"junction={len(junction_df)}, subcatchment={len(subcatchment_df)}, "
            f"channel={len(channel_df)}, outfall={len(outfall_df)}, forcing={len(forcing_df)}"
        )

    num_nodes = topology["num_nodes"]
    num_edges = topology["num_edges"]
    node_type = topology["node_type"].clone()
    edge_index = topology["edge_index"].clone()
    edge_type = topology["edge_type"].clone()
    x_static = topology["x_static"].clone()
    edge_static = topology["edge_static"].clone()
    subcatchment_nodes = topology["subcatchment_nodes"]
    junction_nodes = topology["junction_nodes"]
    outfall_nodes = topology["outfall_nodes"]
    subcatchment_names = topology["subcatchment_names"]
    junction_names = topology["junction_names"]
    storage_names = topology["storage_names"]
    outfall_names = topology["outfall_names"]
    downstream_conduit_names = topology["downstream_conduit_names"]
    edge_original_link_names = topology["edge_original_link_names"]
    global_to_subcatchment = topology["global_to_subcatchment"]

    weather_full = build_weather_from_forcing_df(forcing_df)
    precip = weather_full[0, :, 0]
    temp = weather_full[0, :, 1]

    sub_snow_depth = read_wide_swmm_matrix(subcatchment_df, subcatchment_names, "snow_depth_mm", "subcatchments.parquet")
    sub_runoff_volume = read_wide_swmm_matrix(subcatchment_df, subcatchment_names, "runoff_volume_m3_per_10min", "subcatchments.parquet")
    junc_depth = read_wide_swmm_matrix(junction_df, junction_names, "water_depth_m", "junctions.parquet")
    junc_flood_volume = read_wide_swmm_matrix(junction_df, junction_names, "flooding_volume_m3_per_10min", "junctions.parquet")
    channel_depth = read_wide_swmm_matrix(channel_df, downstream_conduit_names, "water_depth_m", "channels.parquet")
    outfall_depth = read_wide_swmm_matrix(outfall_df, outfall_names, "water_depth_m", "outfalls.parquet")

    dynamic_node = torch.zeros(num_nodes, T, NODE_DYNAMIC_DIM)
    dynamic_node[subcatchment_nodes, :, NODE_PRECIP_COL - NODE_STATIC_DIM] = precip.view(1, T)
    dynamic_node[:, :, NODE_TEMPERATURE_COL - NODE_STATIC_DIM] = temp.view(1, T)
    dynamic_node[junction_nodes, :, NODE_WATER_DEPTH_COL - NODE_STATIC_DIM] = junc_depth
    dynamic_node[outfall_nodes, :, NODE_WATER_DEPTH_COL - NODE_STATIC_DIM] = outfall_depth
    dynamic_node[junction_nodes, :, NODE_FLOODING_VOLUME_COL - NODE_STATIC_DIM] = junc_flood_volume
    x_dynamic_full = dynamic_node.float().contiguous()

    if compact:
        x_last = torch.cat([x_static.float(), x_dynamic_full[:, -1, :]], dim=1)
    else:
        x_full = torch.cat([x_static.unsqueeze(1).expand(-1, T, -1), x_dynamic_full], dim=2).contiguous()
        x_last = x_full[:, -1, :]

    dynamic_edge = torch.zeros(num_edges, T, EDGE_DYNAMIC_DIM)
    downstream_conduit_to_row = {name: i for i, name in enumerate(downstream_conduit_names)}

    for e in range(num_edges):
        e_type = int(edge_type[e].item())

        if e_type == RUNOFF_EDGE_TYPE:
            src = int(edge_index[0, e].item())
            sub_idx = int(global_to_subcatchment[src].item())
            if sub_idx >= 0: dynamic_edge[e, :, 0] = sub_runoff_volume[sub_idx, :]

        elif e_type == DOWNSTREAM_CHANNEL_EDGE_TYPE:
            link_name = edge_original_link_names[e]
            link_idx = downstream_conduit_to_row[link_name]
            dynamic_edge[e, :, 0] = channel_depth[link_idx, :]

        elif e_type == REVERSE_BACKWATER_EDGE_TYPE:
            link_name = edge_original_link_names[e]
            link_idx = downstream_conduit_to_row[link_name]
            dynamic_edge[e, :, 0] = channel_depth[link_idx, :]

        elif e_type in (PUMP_EDGE_TYPE, PUMP_REVERSE_EDGE_TYPE):
            dynamic_edge[e, :, 0] = 0.0

        else:
            raise ValueError(f"Unexpected edge type in full graph: {e_type}")

    edge_dynamic_full = dynamic_edge.float().contiguous()

    if compact:
        edge_attr_last = torch.cat([edge_static.float(), edge_dynamic_full[:, -1, :]], dim=1)
    else:
        edge_attr_full = torch.cat([edge_static.unsqueeze(1).expand(-1, T, -1), edge_dynamic_full], dim=2).contiguous()
        edge_attr_last = edge_attr_full[:, -1, :]

    y_subcatchment_full = torch.stack([sub_snow_depth, sub_runoff_volume], dim=2)
    y_junction_full = torch.stack([junc_depth, junc_flood_volume], dim=2)
    y_channel_full = channel_depth.unsqueeze(2)
    y_outfall_full = outfall_depth.unsqueeze(2)

    data_kwargs = dict(
        x=x_last.float(), edge_index=edge_index, edge_attr=edge_attr_last.float(),
        node_type=node_type, edge_type=edge_type, weather_full=weather_full.float(),
        y_subcatchment_full=y_subcatchment_full.float(), y_junction_full=y_junction_full.float(),
        y_channel_full=y_channel_full.float(), y_outfall_full=y_outfall_full.float(),
    )

    if compact:
        data_kwargs.update(
            x_static_cache=x_static.float().contiguous(), x_dynamic_full=x_dynamic_full,
            edge_static_cache=edge_static.float().contiguous(), edge_dynamic_full=edge_dynamic_full,
        )
    else:
        data_kwargs.update(x_full=x_full.float(), edge_attr_full=edge_attr_full.float())

    data = Data(**data_kwargs)
    data.is_compact_scenario = compact
    data.scenario_id = torch.tensor([scenario_id], dtype=torch.long)
    data.scenario_name = path_name(scenario_output_dir)

    return data




# Splits scenarios randomly into train, validation, and test sets at the scenario level.
def split_scenario_records(records, train_frac=0.70, val_frac=0.20, test_frac=0.10, seed=123):
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-8:
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0.")
    records = list(records)
    random.Random(seed).shuffle(records)
    n = len(records)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    n_test = n - n_train - n_val

    train_records = records[:n_train]
    val_records = records[n_train:n_train + n_val]
    test_records = records[n_train + n_val:]

    print()
    print("========== Scenario split")
    print(f"Total scenarios:      {n}")
    print(f"Train scenarios:      {len(train_records)} ({train_frac:.0%})")
    print(f"Validation scenarios: {len(val_records)} ({val_frac:.0%})")
    print(f"Test scenarios:       {len(test_records)} ({test_frac:.0%})")
    print("====================================")
    print()

    return train_records, val_records, test_records





# Yields consecutive batches of complete scenario records for processing.
def iter_record_batches(records, batch_size: int = BATCH_SIZE):
    records = list(records)

    if batch_size <= 0: raise ValueError("batch_size must be positive.")

    for start in range(0, len(records), batch_size):
        yield start, records[start:start + batch_size]



# Loads one compact SWMM scenario directly without storing it in the global RAM cache.
def load_compact_scenario_direct(record, topology, scenario_idx):
    return build_full_swmm_scenario_data(
        topology=topology, scenario_output_dir=record["scenario_output_dir"],
        scenario_input_path=record["scenario_input_path"], scenario_id=scenario_idx, compact=True,
    )



    
# Loads multiple compact SWMM scenarios in parallel using a thread pool.
def load_compact_scenarios_parallel(records, topology, start_scenario_idx: int = 0, max_workers: int = NUM_IO_WORKERS):
    records = list(records)
    if not records: return []
    num_workers = min(max_workers, len(records))

    def _load_one(item):
        local_idx, record = item
        return load_compact_scenario_direct(
            record=record, topology=topology, scenario_idx=start_scenario_idx + local_idx)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        scenarios = list(executor.map(_load_one, enumerate(records)))
    return scenarios
