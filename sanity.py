import torch
from torch_geometric.data import Data

from config import *




# Validates the real SWMM graph topology, node/edge types, edge counts, and conduit reverse-pair structure.
def validate_static_graph_structure(data: Data) -> None:
    assert data.edge_index.dim() == 2
    assert data.edge_index.size(0) == 2
    num_edges = data.edge_index.size(1)
    assert num_edges == NUM_GRAPH_EDGES
    assert data.node_type.shape == (NUM_NODES,)
    assert data.edge_type.shape == (num_edges,)
    assert int((data.node_type == SUBCATCHMENT_TYPE).sum()) == NUM_SUBCATCHMENTS
    assert int((data.node_type == JUNCTION_TYPE).sum()) == NUM_JUNCTIONS
    assert int((data.node_type == STORAGE_TYPE).sum()) == NUM_STORAGE_NODES
    assert int((data.node_type == OUTFALL_TYPE).sum()) == NUM_OUTFALL_NODES
    assert int(data.node_type.min()) >= 0
    assert int(data.node_type.max()) < NUM_NODE_TYPES
    assert int(data.edge_type.min()) >= 0
    assert int(data.edge_type.max()) < NUM_EDGE_TYPES
    source_nodes = data.edge_index[0]
    target_nodes = data.edge_index[1]
    source_types = data.node_type[source_nodes]
    target_types = data.node_type[target_nodes]
    runoff_mask = data.edge_type == RUNOFF_EDGE_TYPE
    downstream_mask = data.edge_type == DOWNSTREAM_CHANNEL_EDGE_TYPE
    reverse_mask = data.edge_type == REVERSE_BACKWATER_EDGE_TYPE
    assert int(runoff_mask.sum()) == 82
    assert int(downstream_mask.sum()) == NUM_PHYSICAL_CHANNELS
    assert int(reverse_mask.sum()) == NUM_PHYSICAL_CHANNELS
    # runoff_to: subcatchment -> hydraulic node
    assert torch.all(source_types[runoff_mask] == SUBCATCHMENT_TYPE)
    assert torch.all(target_types[runoff_mask] == JUNCTION_TYPE)
    pump_mask = data.edge_type == PUMP_EDGE_TYPE
    pump_reverse_mask = data.edge_type == PUMP_REVERSE_EDGE_TYPE
    assert int(pump_mask.sum()) == NUM_PUMP_LINKS
    assert int(pump_reverse_mask.sum()) == NUM_PUMP_LINKS
    hydraulic_types = torch.tensor([JUNCTION_TYPE, STORAGE_TYPE, OUTFALL_TYPE], device=data.node_type.device)
    assert torch.isin(source_types[downstream_mask], hydraulic_types).all()
    assert torch.isin(target_types[downstream_mask], hydraulic_types).all()
    assert torch.isin(source_types[reverse_mask], hydraulic_types).all()
    assert torch.isin(target_types[reverse_mask], hydraulic_types).all()
    downstream_indices = torch.where(downstream_mask)[0]
    reverse_indices = torch.where(reverse_mask)[0]
    reverse_lookup = {}
    for idx in reverse_indices:
        idx_int = int(idx.item())
        src = int(source_nodes[idx_int].item())
        dst = int(target_nodes[idx_int].item())
        reverse_lookup[(src, dst)] = idx_int

    for downstream_idx in downstream_indices:
        downstream_idx = int(downstream_idx.item())
        u = int(source_nodes[downstream_idx].item())
        v = int(target_nodes[downstream_idx].item())
        assert (v, u) in reverse_lookup





# Validates one full SWMM scenario, including tensor shapes, direction flags, and physical conduit attributes.
def validate_full_scenario_graph(data: Data) -> None:
    validate_static_graph_structure(data)

    num_edges = data.edge_index.size(1)

    assert data.x.shape == (NUM_NODES, NODE_FEATURE_DIM)
    assert data.edge_attr.shape == (num_edges, EDGE_FEATURE_DIM)
    assert data.x_full.shape == (NUM_NODES, FULL_SEQUENCE_LENGTH, NODE_FEATURE_DIM)
    assert data.edge_attr_full.shape == (num_edges, FULL_SEQUENCE_LENGTH, EDGE_FEATURE_DIM)
    assert data.weather_full.shape == (1, FULL_SEQUENCE_LENGTH, WEATHER_FEATURE_DIM)
    assert data.y_subcatchment_full.shape == (NUM_SUBCATCHMENTS, FULL_SEQUENCE_LENGTH, SUBCATCHMENT_OUT_DIM)
    assert data.y_junction_full.shape == (NUM_JUNCTIONS, FULL_SEQUENCE_LENGTH, JUNCTION_OUT_DIM)
    assert data.y_channel_full.shape == (NUM_PHYSICAL_CHANNELS, FULL_SEQUENCE_LENGTH, CHANNEL_OUT_DIM)

    runoff_mask = data.edge_type == RUNOFF_EDGE_TYPE
    downstream_mask = data.edge_type == DOWNSTREAM_CHANNEL_EDGE_TYPE
    reverse_mask = data.edge_type == REVERSE_BACKWATER_EDGE_TYPE
    assert torch.all(data.edge_attr_full[runoff_mask, :, EDGE_DIRECTION_FLAG_COL] == 0.0)
    assert torch.all(data.edge_attr_full[downstream_mask, :, EDGE_DIRECTION_FLAG_COL] == 1.0)
    assert torch.all(data.edge_attr_full[reverse_mask, :, EDGE_DIRECTION_FLAG_COL] == -1.0)

    # Physical conduit attributes are checked only on real downstream conduit edges.
    assert torch.all(data.edge_attr_full[downstream_mask, :, EDGE_LENGTH_COL] > 0.0)
    assert torch.all(data.edge_attr_full[downstream_mask, :, EDGE_ROUGHNESS_COL] > 0.0)
    assert torch.all(data.edge_attr_full[downstream_mask, :, EDGE_CAPACITY_PROXY_COL] > 0.0)





# Validates full-scenario datasets, rolling forecast samples, and post-standardization masking consistency.
def validate_full_scenario_dataset(dataset: list) -> None:

    assert len(dataset) > 0

    for data in dataset:
        validate_full_scenario_graph(data)

    print("Full scenario dataset validation passed.")


def validate_rolling_forecast_sample(data: Data) -> None:

    validate_static_graph_structure(data)

    num_edges = data.edge_index.size(1)

    assert data.x.shape == (NUM_NODES, NODE_FEATURE_DIM)
    assert data.edge_attr.shape == (num_edges, EDGE_FEATURE_DIM)
    assert data.x_hist.shape == (NUM_NODES, INPUT_SEQUENCE_LENGTH, NODE_FEATURE_DIM)
    assert data.edge_attr_hist.shape == (num_edges, INPUT_SEQUENCE_LENGTH, EDGE_FEATURE_DIM)
    assert data.weather_future.shape == (1, INPUT_SEQUENCE_LENGTH, WEATHER_FEATURE_DIM)
    assert data.y_subcatchment_seq.shape == (NUM_SUBCATCHMENTS, FORECAST_HORIZON, SUBCATCHMENT_OUT_DIM)
    assert data.y_junction_seq.shape == (NUM_JUNCTIONS, FORECAST_HORIZON, JUNCTION_OUT_DIM)
    assert data.y_channel_seq.shape == (NUM_PHYSICAL_CHANNELS, FORECAST_HORIZON, CHANNEL_OUT_DIM)

    runoff_mask = data.edge_type == RUNOFF_EDGE_TYPE
    downstream_mask = data.edge_type == DOWNSTREAM_CHANNEL_EDGE_TYPE
    reverse_mask = data.edge_type == REVERSE_BACKWATER_EDGE_TYPE

    # Direction flags over the input sequence.
    assert torch.all(data.edge_attr_hist[runoff_mask, :, EDGE_DIRECTION_FLAG_COL] == 0.0)
    assert torch.all(data.edge_attr_hist[downstream_mask, :, EDGE_DIRECTION_FLAG_COL] == 1.0)
    assert torch.all(data.edge_attr_hist[reverse_mask, :, EDGE_DIRECTION_FLAG_COL] == -1.0)


def validate_rolling_forecast_dataset(dataset: list, name: str = "dataset", validate_all: bool = False) -> None:

    assert len(dataset) > 0

    if validate_all:
        for data in dataset:
            validate_rolling_forecast_sample(data)
    else:
        validate_rolling_forecast_sample(dataset[0])

    print(f"Rolling forecast {name} validation passed.")


def validate_standardized_future_masking_sample(data: Data) -> None:
    future_slice = slice(LOOKBACK_LENGTH, INPUT_SEQUENCE_LENGTH)
    assert torch.allclose(
        data.x_hist[:, future_slice, NODE_WATER_DEPTH_COL],
        torch.zeros_like(data.x_hist[:, future_slice, NODE_WATER_DEPTH_COL]),
    )
    assert torch.allclose(
        data.x_hist[:, future_slice, NODE_FLOODING_VOLUME_COL],
        torch.zeros_like(data.x_hist[:, future_slice, NODE_FLOODING_VOLUME_COL]),
    )
    assert torch.allclose(
        data.edge_attr_hist[:, future_slice, EDGE_RESPONSE_COL],
        torch.zeros_like(data.edge_attr_hist[:, future_slice, EDGE_RESPONSE_COL]),
    )

    assert torch.allclose(data.x, data.x_hist[:, LOOKBACK_LENGTH - 1, :])
    assert torch.allclose(data.edge_attr, data.edge_attr_hist[:, LOOKBACK_LENGTH - 1, :])


def validate_standardized_future_masking_dataset(dataset: list, name: str = "dataset", validate_all: bool = False) -> None:
    assert len(dataset) > 0

    if validate_all:
        for data in dataset:
            validate_standardized_future_masking_sample(data)
    else:
        validate_standardized_future_masking_sample(dataset[0])

    print(f"Standardized future masking for {name} validation passed.")




# Prints concise summaries of full-scenario graph structure and rolling forecast sample tensor shapes.
def print_full_scenario_graph_summary(data: Data) -> None:

    num_edges = data.edge_index.size(1)
    num_runoff_edges = int((data.edge_type == RUNOFF_EDGE_TYPE).sum())
    num_downstream_edges = int((data.edge_type == DOWNSTREAM_CHANNEL_EDGE_TYPE).sum())
    num_reverse_edges = int((data.edge_type == REVERSE_BACKWATER_EDGE_TYPE).sum())

    print(f"Number of nodes: {NUM_NODES}")
    print(f"Number of subcatchments: {NUM_SUBCATCHMENTS}")
    print(f"Number of junctions: {NUM_JUNCTIONS}")
    print(f"Number of edges: {num_edges}")
    print(f"Number of node types: {NUM_NODE_TYPES}")
    print("  Node type 0 = Subcatchment")
    print("  Node type 1 = Junction")
    print(f"Number of edge types: {NUM_EDGE_TYPES}")
    print("  Edge type 0 = runoff_to, Subcatchment -> junction")
    print("  Edge type 1 = conduit_downstream, junction -> junction")
    print("  Edge type 2 = conduit_reverse_backwater, junction -> junction")
    print(f"Number of runoff_to edges: {num_runoff_edges}")
    print(f"Number of downstream_channel edges: {num_downstream_edges}")
    print(f"Number of reverse_backwater edges: {num_reverse_edges}")


def print_rolling_forecast_sample_summary(data: Data) -> None:

    print()
    print("========== Rolling forecast sample summary ==========")
    print(f"x shape: {tuple(data.x.shape)}")
    print(f"edge_attr shape: {tuple(data.edge_attr.shape)}")
    print(f"x_hist shape: {tuple(data.x_hist.shape)}")
    print(f"edge_attr_hist shape: {tuple(data.edge_attr_hist.shape)}")
    print(f"weather_future shape: {tuple(data.weather_future.shape)}")
    print(f"Subcatchment target sequence shape: {tuple(data.y_subcatchment_seq.shape)}")
    print(f"Junction target sequence shape: {tuple(data.y_junction_seq.shape)}")
    print(f"Channel target sequence shape: {tuple(data.y_channel_seq.shape)}")
    print("====================================================")
    print()
