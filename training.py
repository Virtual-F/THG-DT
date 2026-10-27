import copy
import pandas as pd
import torch
from tqdm.auto import tqdm

from config import *
from data_loading import (
    build_scenario_records,
    split_scenario_records,
    load_real_swmm_static_topology,
    iter_record_batches,
    load_compact_scenarios_parallel,
)
from preprocessing import (
    build_standardized_static_graph,
    build_standardized_transition_targets_batch,
    standardize_weather_batch,
    iter_forecast_origin_chunks,
    fit_standardizers_streaming,
    fit_training_aux_statistics_streaming,
)
from model import TransformerHEATGRUDigitalTwin
from losses_metrics import compute_state_transition_loss, empty_stateful_metric_report
from evaluation import evaluate_stateful_latent, LocalTestValueWriter
from reporting import print_stateful_epoch_report, print_stateful_split_report



# Trains the latent digital twin for one stateful TBPTT epoch with separate flood-head gradient clipping.
def train_one_epoch_stateful(model, records, topology, optimizer, device, scalers,
                             tbptt_blocks: int = TBPTT_BLOCKS_PER_CHUNK, scenario_batch_size: int = BATCH_SIZE):
    model.train()
    flood_parameters = list(model.junction_flood_head.parameters())
    flood_parameter_ids = {id(parameter) for parameter in flood_parameters}
    main_parameters = [parameter for parameter in model.parameters() if id(parameter) not in flood_parameter_ids]
    records = list(records)

    loss_sums = {key: torch.zeros((), dtype=torch.float64, device=device)
                 for key in ["total", "subcatchment", "junction", "channel", "outfall"]}
    total_transitions = 0

    edge_index = topology["edge_index"].to(device)
    node_type = topology["node_type"].to(device)
    edge_type = topology["edge_type"].to(device)
    x_static_std, edge_static_std = build_standardized_static_graph(topology=topology, scalers=scalers, device=device)

    num_batches = (len(records) + scenario_batch_size - 1) // scenario_batch_size
    batch_iterator = iter_record_batches(records, scenario_batch_size)

    for batch_start, batch_records in tqdm(batch_iterator, total=num_batches, desc="Training scenario batches"):
        scenarios = load_compact_scenarios_parallel(records=batch_records, topology=topology,
                                                    start_scenario_idx=batch_start, max_workers=NUM_IO_WORKERS)
        actual_batch_size = len(scenarios)
        if actual_batch_size == 0: continue

        lengths = [int(scenario.weather_full.size(1)) for scenario in scenarios]
        if len(set(lengths)) != 1: raise ValueError(f"Scenario lengths differ: {lengths}")
        T = lengths[0]
        weather_full_std = standardize_weather_batch(scenarios=scenarios, scalers=scalers, device=device)
        node_hidden = None
        channel_hidden = None

        for origin_chunk in iter_forecast_origin_chunks(sequence_length=T, blocks_per_chunk=tbptt_blocks):
            first_origin, last_origin, num_blocks = origin_chunk[0], origin_chunk[-1], len(origin_chunk)
            targets = build_standardized_transition_targets_batch(
                scenarios=scenarios, start=first_origin, end=last_origin + FORECAST_HORIZON,
                scalers=scalers, device=device,
            )
            origin_steps = torch.tensor(origin_chunk, dtype=torch.long, device=device)
            target_start = first_origin + 1
            target_end_exclusive = last_origin + FORECAST_HORIZON + 1
            future_weather = weather_full_std[:, target_start:target_end_exclusive, :]
            expected_future_steps = num_blocks * FORECAST_HORIZON
            if future_weather.size(1) != expected_future_steps: raise RuntimeError("Future weather geometry is wrong.")

            forcing_blocks_std = future_weather.reshape(actual_batch_size, num_blocks, FORECAST_HORIZON, WEATHER_FEATURE_DIM)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(USE_BF16 and device.type == "cuda")):
                predictions, node_hidden, channel_hidden = model.forward_latent_batch_chunk(
                    x_static_std=x_static_std, edge_static_std=edge_static_std, weather_full_std=weather_full_std,
                    forcing_blocks_std=forcing_blocks_std, origin_steps=origin_steps, edge_index=edge_index,
                    node_type=node_type, edge_type=edge_type, node_hidden=node_hidden, channel_hidden=channel_hidden,
                )
                loss, loss_dict = compute_state_transition_loss(predictions, targets, scalers)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(main_parameters, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(flood_parameters, max_norm=1.0)
            optimizer.step()

            node_hidden = node_hidden.detach()
            channel_hidden = channel_hidden.detach()
            n_target_steps = num_blocks * FORECAST_HORIZON
            batch_transitions = actual_batch_size * n_target_steps
            total_transitions += batch_transitions

            for key in loss_sums:
                loss_sums[key] += loss_dict[key].detach().double() * batch_transitions

            del targets, predictions, loss, loss_dict, forcing_blocks_std, future_weather, origin_steps

        del scenarios, weather_full_std

    avg_losses = {key: (value / max(total_transitions, 1)).item() for key, value in loss_sums.items()}
    return {"loss": avg_losses, "metrics": empty_stateful_metric_report(),
            "num_transitions": total_transitions, "num_scenarios": len(records)}


# Runs the complete stateful training, validation, held-out testing, checkpointing, and deployment packaging workflow.
def train_example_stateful():
    NUM_EPOCHS = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
    print('Device:', device)
    print('Training architecture: Static HEATConv + Latent HEATConv + Transformer + GRU')
    print('Transformer lookback:', CONTEXT_LOOKBACK_DAYS, 'rolling days |', CONTEXT_TOKEN_COUNT, 'tokens =', CONTEXT_DAILY_TOKENS, 'daily +', CONTEXT_HOURLY_TOKENS, 'hourly')
    print('Forecast state-update interval:', FORECAST_STRIDE * MODEL_TIME_STEP_MINUTES, 'minutes')
    print('TBPTT gradient chunk:', TBPTT_BLOCKS_PER_CHUNK, 'hourly forecast blocks =', TBPTT_DAYS, 'day')
    print('GRU latent state: carried across TBPTT chunks')
    print('Context update interval:', CONTEXT_UPDATE_INTERVAL, 'steps =', CONTEXT_UPDATE_INTERVAL * MODEL_TIME_STEP_MINUTES, 'minutes')
    print('Native model timestep:', MODEL_TIME_STEP_MINUTES, 'minutes')
    print('Deployment forecast horizon:', FORECAST_HORIZON, 'steps =', FORECAST_HORIZON * MODEL_TIME_STEP_MINUTES, 'minutes')
    print('Deployment forecast stride:', FORECAST_STRIDE, 'steps =', FORECAST_STRIDE * MODEL_TIME_STEP_MINUTES, 'minutes')
    print('Deployment forecast windows: NON-OVERLAPPING')
    print('Scaler RAM policy: one scenario at a time; training RAM policy: one scenario batch at a time')
    all_scenario_records = build_scenario_records(output_root=SWMM_SCENARIO_OUTPUT_ROOT, input_root=SCENARIO_INPUT_ROOT)
    print('Total matched scenarios:', len(all_scenario_records))
    train_records, val_records, test_records = split_scenario_records(all_scenario_records, train_frac=0.7, val_frac=0.2, test_frac=0.1, seed=123)
    topology = load_real_swmm_static_topology(graph_path=REAL_STATIC_GRAPH_PATH, node_table_path=NODE_TABLE_PATH, edge_table_path=EDGE_TABLE_PATH)
    print('Topology loaded.')
    print('Nodes:', topology['num_nodes'])
    print('Edges:', topology['num_edges'])
    print('Subcatchments:', topology['num_subcatchments'])
    print('Junctions:', topology['num_junctions'])
    SCALER_CACHE_PATH = TRAINING_RESULTS_DIR / 'train_scalers_10500_no_storage.pt'
    if SCALER_CACHE_PATH.exists():
        scalers = torch.load(SCALER_CACHE_PATH, map_location='cpu', weights_only=False)
        print('Loaded cached TRAIN-only standardizers:', SCALER_CACHE_PATH)
    else:
        scalers = fit_standardizers_streaming(records=train_records, topology=topology)
        torch.save(scalers, SCALER_CACHE_PATH)
        print('Streaming TRAIN-only standardizers fitted and cached:', SCALER_CACHE_PATH)
    AUX_STATS_CACHE_PATH = TRAINING_RESULTS_DIR / f'train_aux_stats_{len(train_records)}_binary_flood_tail_no_storage_v1.pt'
    if AUX_STATS_CACHE_PATH.exists():
        aux_stats = torch.load(AUX_STATS_CACHE_PATH, map_location='cpu', weights_only=False)
        print('Loaded cached TRAIN-only auxiliary statistics:', AUX_STATS_CACHE_PATH)
    else:
        aux_stats = fit_training_aux_statistics_streaming(records=train_records, topology=topology)
        torch.save(aux_stats, AUX_STATS_CACHE_PATH)
        print('Saved TRAIN-only auxiliary statistics:', AUX_STATS_CACHE_PATH)
    scalers = dict(scalers)
    scalers.update(aux_stats)
    junction_max_depth_raw = topology['x_static'][topology['junction_nodes'], topology['node_feature_index']['node_max_depth']].float()
    junction_depth_mean = scalers['y_junction_mean'][0].float()
    junction_depth_std = scalers['y_junction_std'][0].float().clamp_min(1e-08)
    junction_max_depth_target_std = ((junction_max_depth_raw - junction_depth_mean) / junction_depth_std).contiguous()
    if not torch.isfinite(junction_max_depth_target_std).all():
        raise RuntimeError('Non-finite standardized junction max depths.')
    model = TransformerHEATGRUDigitalTwin(in_channels=MODEL_NODE_INPUT_DIM, hidden_channels=32, num_node_types=NUM_NODE_TYPES, num_edge_types=NUM_EDGE_TYPES, edge_type_emb_dim=8, edge_dim=MODEL_EDGE_INPUT_DIM, edge_attr_emb_dim=16, heads=2, dropout=0.1).to(device)
    with torch.no_grad():
        model.junction_max_depth_std.copy_(junction_max_depth_target_std.to(device=device, dtype=model.junction_max_depth_std.dtype))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-05)
    print('Model created.')
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = None
    best_train_report = None
    best_val_report = None
    history_rows = []
    training_summary_rows = []
    validation_summary_rows = []
    for epoch in range(1, NUM_EPOCHS + 1):
        train_report = train_one_epoch_stateful(model=model, records=train_records, topology=topology, optimizer=optimizer, device=device, scalers=scalers)
        val_report = evaluate_stateful_latent(model=model, records=val_records, topology=topology, device=device, scalers=scalers)
        print_stateful_epoch_report(epoch, train_report, val_report)
        training_summary_rows.append({'epoch': epoch, 'loss_total': train_report['loss']['total'], 'loss_subcatchment': train_report['loss']['subcatchment'], 'loss_junction': train_report['loss']['junction'], 'loss_channel': train_report['loss']['channel'], 'loss_outfall': train_report['loss']['outfall'], 'num_transitions': train_report['num_transitions'], 'num_scenarios': train_report['num_scenarios']})
        pd.DataFrame(training_summary_rows).to_csv(TRAINING_RESULTS_DIR / 'training_epoch_summary.csv', index=False)
        validation_summary_rows.append({'epoch': epoch, 'loss_total': val_report['loss']['total'], 'loss_subcatchment': val_report['loss']['subcatchment'], 'loss_junction': val_report['loss']['junction'], 'loss_channel': val_report['loss']['channel'], 'loss_outfall': val_report['loss']['outfall'], 'rmse_all': val_report['metrics']['all']['rmse'], 'nse_all': val_report['metrics']['all']['nse'], 'kge_all': val_report['metrics']['all']['kge'], 'rmse_snow_depth': val_report['metrics']['snow_depth']['rmse'], 'nse_snow_depth': val_report['metrics']['snow_depth']['nse'], 'kge_snow_depth': val_report['metrics']['snow_depth']['kge'], 'rmse_runoff_volume': val_report['metrics']['runoff_volume']['rmse'], 'nse_runoff_volume': val_report['metrics']['runoff_volume']['nse'], 'kge_runoff_volume': val_report['metrics']['runoff_volume']['kge'], 'rmse_junction_depth': val_report['metrics']['junction_depth']['rmse'], 'nse_junction_depth': val_report['metrics']['junction_depth']['nse'], 'kge_junction_depth': val_report['metrics']['junction_depth']['kge'], 'rmse_channel': val_report['metrics']['channel']['rmse'], 'nse_channel': val_report['metrics']['channel']['nse'], 'kge_channel': val_report['metrics']['channel']['kge'], 'rmse_outfall': val_report['metrics']['outfall']['rmse'], 'nse_outfall': val_report['metrics']['outfall']['nse'], 'kge_outfall': val_report['metrics']['outfall']['kge'], 'flood_threshold': val_report['flood_classification']['threshold'], 'flood_precision': val_report['flood_classification']['precision'], 'flood_recall': val_report['flood_classification']['recall'], 'flood_f1': val_report['flood_classification']['f1'], 'flood_csi': val_report['flood_classification']['csi'], 'flood_pr_auc_approx': val_report['flood_classification']['pr_auc_approx'], 'num_transitions': val_report['num_transitions'], 'num_scenarios': val_report['num_scenarios']})
        pd.DataFrame(validation_summary_rows).to_csv(VALIDATION_RESULTS_DIR / 'validation_epoch_summary.csv', index=False)
        history_rows.append({'epoch': epoch, 'train_loss': train_report['loss']['total'], 'train_rmse_all': train_report['metrics']['all']['rmse'], 'train_nse_all': train_report['metrics']['all']['nse'], 'train_kge_all': train_report['metrics']['all']['kge'], 'val_loss': val_report['loss']['total'], 'val_rmse_all': val_report['metrics']['all']['rmse'], 'val_nse_all': val_report['metrics']['all']['nse'], 'val_kge_all': val_report['metrics']['all']['kge'], 'val_rmse_snow_depth': val_report['metrics']['snow_depth']['rmse'], 'val_nse_snow_depth': val_report['metrics']['snow_depth']['nse'], 'val_kge_snow_depth': val_report['metrics']['snow_depth']['kge'], 'val_r_snow_depth': val_report['metrics']['snow_depth']['r'], 'val_rmse_runoff_volume': val_report['metrics']['runoff_volume']['rmse'], 'val_nse_runoff_volume': val_report['metrics']['runoff_volume']['nse'], 'val_kge_runoff_volume': val_report['metrics']['runoff_volume']['kge'], 'val_r_runoff_volume': val_report['metrics']['runoff_volume']['r'], 'val_rmse_junction_depth': val_report['metrics']['junction_depth']['rmse'], 'val_nse_junction_depth': val_report['metrics']['junction_depth']['nse'], 'val_kge_junction_depth': val_report['metrics']['junction_depth']['kge'], 'val_r_junction_depth': val_report['metrics']['junction_depth']['r'], 'val_rmse_channel': val_report['metrics']['channel']['rmse'], 'val_nse_channel': val_report['metrics']['channel']['nse'], 'val_kge_channel': val_report['metrics']['channel']['kge'], 'val_r_channel': val_report['metrics']['channel']['r'], 'val_flood_threshold': val_report['flood_classification']['threshold'], 'val_flood_precision': val_report['flood_classification']['precision'], 'val_flood_recall': val_report['flood_classification']['recall'], 'val_flood_f1': val_report['flood_classification']['f1'], 'val_flood_csi': val_report['flood_classification']['csi'], 'val_flood_pr_auc_approx': val_report['flood_classification']['pr_auc_approx'], 'val_rmse_outfall': val_report['metrics']['outfall']['rmse'], 'val_nse_outfall': val_report['metrics']['outfall']['nse'], 'val_kge_outfall': val_report['metrics']['outfall']['kge'], 'val_r_outfall': val_report['metrics']['outfall']['r'], **{f'val_extreme_{group}_{metric}': value for group, extreme_values in val_report['extreme_diagnostics'].items() for metric, value in extreme_values.items()}})
        val_loss = val_report['loss']['total']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_train_report = copy.deepcopy(train_report)
            best_val_report = copy.deepcopy(val_report)
        checkpoint = {'epoch': epoch, 'architecture': 'Static-HEATConv + Latent-HEATConv + weather-Transformer + continuous-latent-GRU + decoupled physics-conditioned flood head', 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'train_report': train_report, 'val_report': val_report, 'scalers': scalers, 'tbptt_blocks_per_chunk': TBPTT_BLOCKS_PER_CHUNK, 'context_lookback_days': CONTEXT_LOOKBACK_DAYS, 'context_raw_steps': CONTEXT_RAW_STEPS, 'context_token_count': CONTEXT_TOKEN_COUNT, 'context_daily_tokens': CONTEXT_DAILY_TOKENS, 'context_hourly_tokens': CONTEXT_HOURLY_TOKENS, 'context_update_interval_raw_steps': CONTEXT_UPDATE_INTERVAL, 'first_forecast_origin': FIRST_FORECAST_ORIGIN, 'model_node_input_dim': MODEL_NODE_INPUT_DIM, 'model_edge_input_dim': MODEL_EDGE_INPUT_DIM, 'tbptt_days': TBPTT_DAYS, 'latent_state_carried_across_tbptt_chunks': True, 'dynamic_latent_spatial_message_passing': True, 'scenario_batch_size': BATCH_SIZE, 'model_time_step_minutes': MODEL_TIME_STEP_MINUTES, 'forecast_horizon_steps': FORECAST_HORIZON, 'forecast_stride_steps': FORECAST_STRIDE, 'forecast_horizon_minutes': FORECAST_HORIZON * MODEL_TIME_STEP_MINUTES, 'forecast_stride_minutes': FORECAST_STRIDE * MODEL_TIME_STEP_MINUTES}
        ckpt_path = MODELS_RESULTS_DIR / f'transformer_heatconv_gru_stateful_epoch_{epoch:03d}.pt'
        torch.save(checkpoint, ckpt_path)
        print('Saved:', ckpt_path)
    if best_model_state is None:
        raise RuntimeError('No best model was selected during training.')
    model.load_state_dict(best_model_state)
    best_flood_probability_threshold = float(best_val_report['flood_classification']['threshold'])
    print('Best validation flood threshold:', best_flood_probability_threshold)
    scalers['flood_probability_threshold'] = best_flood_probability_threshold
    model.eval()
    print()
    print('=' * 88)
    print('FINAL HELD-OUT TEST EVALUATION')
    print('=' * 88)
    test_value_writer = LocalTestValueWriter(path=TEST_VALUES_H5_PATH, records=test_records, topology=topology, flood_probability_threshold=best_flood_probability_threshold)
    try:
        test_report = evaluate_stateful_latent(model=model, records=test_records, topology=topology, device=device, scalers=scalers, flood_probability_threshold=best_flood_probability_threshold, test_value_writer=test_value_writer)
    finally:
        test_value_writer.close()
    print('Saved final TEST values:', TEST_VALUES_H5_PATH)
    print_stateful_split_report(f'TEST ({len(test_records)} unseen scenarios)', test_report)
    print('=' * 88)
    print()
    test_metric_rows = []
    for variable_name, metric_values in test_report['metrics'].items():
        row = {'variable': variable_name}
        row.update(metric_values)
        test_metric_rows.append(row)
    FINAL_TEST_METRICS_PATH = TEST_RESULTS_DIR / 'final_test_metrics_transformer_heatconv_gru.csv'
    pd.DataFrame(test_metric_rows).to_csv(FINAL_TEST_METRICS_PATH, index=False)
    test_loss_row = {**test_report['loss'], 'num_transitions': test_report['num_transitions'], 'num_scenarios': test_report['num_scenarios']}
    FINAL_TEST_LOSS_PATH = TEST_RESULTS_DIR / 'final_test_loss_transformer_heatconv_gru.csv'
    pd.DataFrame([test_loss_row]).to_csv(FINAL_TEST_LOSS_PATH, index=False)
    pd.DataFrame([test_report['flood_classification']]).to_csv(TEST_RESULTS_DIR / 'final_test_flood_classification.csv', index=False)
    print('Saved final flood classification:', 'final_test_flood_classification.csv')
    print('Saved final test metrics:', 'final_test_metrics_transformer_heatconv_gru.csv')
    print('Saved final test loss:', 'final_test_loss_transformer_heatconv_gru.csv')
    extreme_rows = []
    for variable_name, extreme_values in test_report['extreme_diagnostics'].items():
        row = {'variable': variable_name}
        row.update(extreme_values)
        extreme_rows.append(row)
    pd.DataFrame(extreme_rows).to_csv(TEST_RESULTS_DIR / 'final_test_extreme_diagnostics.csv', index=False)
    print('Saved final extreme diagnostics:', 'final_test_extreme_diagnostics.csv')
    deployment_model_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    deployment_scalers = {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in scalers.items()}
    deployment_bundle = {'artifact_type': 'TransformerHEATGRUDigitalTwin', 'architecture': 'Static HEATConv + Latent HEATConv + 30-day weather Transformer + latent GRU + decoupled physics-conditioned flood head', 'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'best_train_report': best_train_report, 'best_validation_report': best_val_report, 'test_report': test_report, 'model_state_dict': deployment_model_state, 'model_kwargs': {'in_channels': MODEL_NODE_INPUT_DIM, 'hidden_channels': 32, 'num_node_types': NUM_NODE_TYPES, 'num_edge_types': NUM_EDGE_TYPES, 'edge_type_emb_dim': 8, 'edge_dim': MODEL_EDGE_INPUT_DIM, 'edge_attr_emb_dim': 16, 'heads': 2, 'dropout': 0.1, 'transformer_num_heads': TRANSFORMER_NUM_HEADS, 'transformer_encoder_layers': TRANSFORMER_ENCODER_LAYERS, 'transformer_feedforward_dim': TRANSFORMER_FEEDFORWARD_DIM, 'transformer_dropout': TRANSFORMER_DROPOUT, 'context_token_count': CONTEXT_TOKEN_COUNT, 'context_update_interval': CONTEXT_UPDATE_INTERVAL}, 'scalers': deployment_scalers, 'flood_configuration': {'target': 'binary_occurrence', 'volume_threshold_m3_per_10min': FLOOD_OCCURRENCE_VOLUME_THRESHOLD, 'focal_gamma': FLOOD_FOCAL_GAMMA, 'head_type': 'physics_conditioned_stop_gradient', 'backbone_gradient_from_flood': False, 'capacity_reference': 'node_max_depth', 'capacity_features': ['predicted_junction_depth_std', 'junction_max_depth_std', 'depth_minus_max_depth_std', 'soft_capacity_gate'], 'manual_depth_flood_share_weights': False, 'positive_weight': float(scalers['flood_positive_weight']), 'probability_threshold': float(best_flood_probability_threshold), 'threshold_selection': 'maximum_validation_CSI', 'threshold_num_bins': FLOOD_THRESHOLD_NUM_BINS}, 'runoff_loss_configuration': {'base': 'standardized_tail_weighted_MSE', 'train_q95': float(scalers['tail_thresholds']['runoff_volume']['q95']), 'train_q99': float(scalers['tail_thresholds']['runoff_volume']['q99']), 'q95_weight': RUNOFF_Q95_WEIGHT, 'q99_weight': RUNOFF_Q99_WEIGHT, 'peak_loss_weight': RUNOFF_PEAK_LOSS_WEIGHT, 'peak_block_steps': FORECAST_HORIZON}, 'topology': topology, 'raw_meteorological_inputs': ['precipitation_mm_10min', 'temperature_C'], 'weather_feature_order': ['precipitation_mm_10min', 'temperature_C', 'snowfall_proxy', 'melt_factor_proxy'], 'context_lookback_days': CONTEXT_LOOKBACK_DAYS, 'context_raw_steps': CONTEXT_RAW_STEPS, 'model_time_step_minutes': MODEL_TIME_STEP_MINUTES, 'forecast_horizon_steps': FORECAST_HORIZON, 'forecast_horizon_minutes': FORECAST_HORIZON * MODEL_TIME_STEP_MINUTES, 'forecast_stride_steps': FORECAST_STRIDE, 'output_order': {'subcatchment': ['snow_depth_mm', 'runoff_volume_m3_per_10min'], 'junction': ['water_depth_m', 'flood_occurrence_0_1'], 'channel': ['water_depth_m'], 'outfall': ['water_depth_m']}}
    DEPLOYMENT_MODEL_PATH = MODELS_RESULTS_DIR / 'digital_twin_heatconv_transformer_gru_best.pt'
    torch.save(deployment_bundle, DEPLOYMENT_MODEL_PATH)
    print('Saved final deployment model:', DEPLOYMENT_MODEL_PATH)
    results_bundle = {'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'best_train_report': best_train_report, 'best_validation_report': best_val_report, 'test_report': test_report, 'epoch_history': history_rows, 'scenario_splits': {'train': [dict(record) for record in train_records], 'validation': [dict(record) for record in val_records], 'test': [dict(record) for record in test_records]}, 'network_summary': {'num_nodes': topology['num_nodes'], 'num_edges': topology['num_edges'], 'num_subcatchments': topology['num_subcatchments'], 'num_junctions': topology['num_junctions'], 'num_storage_nodes': topology['num_storage_nodes'], 'num_outfall_nodes': topology['num_outfall_nodes'], 'num_physical_channels': topology['num_physical_channels']}, 'node_table': topology['node_table'], 'edge_table': topology['edge_table'], 'flood_configuration': deployment_bundle['flood_configuration'], 'runoff_loss_configuration': deployment_bundle['runoff_loss_configuration'], 'tail_thresholds': scalers['tail_thresholds'], 'scalers': deployment_scalers, 'model_configuration': {'artifact_type': deployment_bundle['artifact_type'], 'architecture': deployment_bundle['architecture'], 'model_kwargs': deployment_bundle['model_kwargs'], 'raw_meteorological_inputs': deployment_bundle['raw_meteorological_inputs'], 'weather_feature_order': deployment_bundle['weather_feature_order'], 'context_lookback_days': deployment_bundle['context_lookback_days'], 'context_raw_steps': deployment_bundle['context_raw_steps'], 'model_time_step_minutes': deployment_bundle['model_time_step_minutes'], 'forecast_horizon_steps': deployment_bundle['forecast_horizon_steps'], 'forecast_horizon_minutes': deployment_bundle['forecast_horizon_minutes'], 'forecast_stride_steps': deployment_bundle['forecast_stride_steps']}, 'output_order': deployment_bundle['output_order']}
    RESULTS_BUNDLE_PATH = MODELS_RESULTS_DIR / 'digital_twin_training_results.pt'
    torch.save(results_bundle, RESULTS_BUNDLE_PATH)
    print('Saved self-contained results bundle:', RESULTS_BUNDLE_PATH)
    print()
    print()
    print('========== Stateful training finished ==========')
    print('Best epoch:', best_epoch)
    print(f'Best 1-hour-block validation loss: {best_val_loss:.6f}')
    print('================================================')
    print(f'Important: final digital-twin deployment should be evaluated with rollout_scenario_latent() using non-overlapping {FORECAST_HORIZON * MODEL_TIME_STEP_MINUTES}-minute forecast blocks.')
    return {'model': model, 'scalers': scalers, 'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'best_train_report': best_train_report, 'best_validation_report': best_val_report, 'test_report': test_report, 'test_records': test_records, 'results_bundle_path': RESULTS_BUNDLE_PATH}
