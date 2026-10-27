import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from config import *




# Initializes streaming accumulators for RMSE, NSE, correlation, and KGE calculations.
def new_metric_state() -> dict:
    return {"n": 0, "sum_obs": 0.0, "sum_sim": 0.0, "sum_obs2": 0.0,
            "sum_sim2": 0.0, "sum_obs_sim": 0.0, "sse": 0.0}





# Updates streaming hydrological metric statistics from one prediction-target pair.
def update_metric_state(state: dict, pred: Tensor, target: Tensor) -> None:
    with torch.no_grad():
        sim = pred.detach().float().reshape(-1)
        obs = target.detach().float().reshape(-1)

        if sim.numel() != obs.numel():
            raise ValueError(f"Prediction and target sizes differ: {sim.numel()} vs {obs.numel()}")

        error = sim - obs

        state["n"] += int(obs.numel())
        state["sum_obs"] += float(obs.sum().item())
        state["sum_sim"] += float(sim.sum().item())
        state["sum_obs2"] += float((obs ** 2).sum().item())
        state["sum_sim2"] += float((sim ** 2).sum().item())
        state["sum_obs_sim"] += float((obs * sim).sum().item())
        state["sse"] += float((error ** 2).sum().item())





# Finalizes accumulated statistics into RMSE, NSE, KGE, correlation, variability, and bias metrics.
def finalize_metric_state(state: dict, eps: float = 1e-8) -> dict:
    n = max(int(state["n"]), 1)
    mean_obs = state["sum_obs"] / n
    mean_sim = state["sum_sim"] / n
    mse = state["sse"] / n
    rmse = float(np.sqrt(max(mse, 0.0)))
    obs_var_sum = state["sum_obs2"] - n * mean_obs ** 2
    sim_var_sum = state["sum_sim2"] - n * mean_sim ** 2

    obs_var_sum = max(obs_var_sum, eps)
    sim_var_sum = max(sim_var_sum, eps)

    nse = 1.0 - state["sse"] / obs_var_sum

    cov_sum = state["sum_obs_sim"] - n * mean_obs * mean_sim
    r = cov_sum / np.sqrt(obs_var_sum * sim_var_sum)
    r = float(np.clip(r, -1.0, 1.0))

    obs_std = np.sqrt(obs_var_sum / n)
    sim_std = np.sqrt(sim_var_sum / n)
    alpha = sim_std / max(obs_std, eps)

    if abs(mean_obs) < eps:
        beta = np.nan
        kge = np.nan
    else:
        beta = mean_sim / mean_obs
        kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)

    return {
        "rmse": float(rmse), "nse": float(nse), "kge": float(kge) if not np.isnan(kge) else np.nan,
        "r": float(r), "alpha": float(alpha), "beta": float(beta) if not np.isnan(beta) else np.nan,
    }




# Computes regression metrics for one prediction-target pair.
def regression_metrics(pred: Tensor, target: Tensor) -> dict:
    state = new_metric_state()
    update_metric_state(state, pred, target)
    return finalize_metric_state(state)



# Computes class-weighted focal binary cross-entropy for flood-occurrence logits.
def weighted_binary_focal_loss_with_logits(logits: Tensor, targets: Tensor, positive_weight: float,
                                           gamma: float = FLOOD_FOCAL_GAMMA) -> Tensor:
    logits = logits.float()
    targets = targets.float()

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    focal_factor = (1.0 - p_t).pow(gamma)

    positive_weight_tensor = logits.new_tensor(float(positive_weight))
    class_weight = torch.where(targets > 0.5, positive_weight_tensor, torch.ones_like(targets))

    weighted_loss = class_weight * focal_factor * bce
    return weighted_loss.sum() / class_weight.sum().clamp_min(1.0)



# Computes runoff loss with Q95/Q99 tail weighting and an additional six-step peak-error penalty.
def compute_runoff_extreme_aware_loss(pred_runoff_std: Tensor, true_runoff_std: Tensor, scalers: dict):
    base_mse = F.mse_loss(pred_runoff_std, true_runoff_std)

    runoff_mean = scalers["y_subcatchment_mean"][1].to(true_runoff_std.device)
    runoff_std = scalers["y_subcatchment_std"][1].to(true_runoff_std.device).clamp_min(1e-8)

    q95_raw = float(scalers["tail_thresholds"]["runoff_volume"]["q95"])
    q99_raw = float(scalers["tail_thresholds"]["runoff_volume"]["q99"])
    q95_std = (true_runoff_std.new_tensor(q95_raw) - runoff_mean) / runoff_std
    q99_std = (true_runoff_std.new_tensor(q99_raw) - runoff_mean) / runoff_std

    squared_error = (pred_runoff_std - true_runoff_std).pow(2)
    weights = torch.ones_like(squared_error)
    weights = torch.where(true_runoff_std > q95_std, weights.new_tensor(RUNOFF_Q95_WEIGHT), weights)
    weights = torch.where(true_runoff_std > q99_std, weights.new_tensor(RUNOFF_Q99_WEIGHT), weights)
    weighted_mse = (weights * squared_error).sum() / weights.sum().clamp_min(1.0)

    num_steps = pred_runoff_std.size(-1)
    if num_steps % FORECAST_HORIZON != 0:
        raise RuntimeError("Runoff time dimension must be divisible by FORECAST_HORIZON.")

    num_blocks = num_steps // FORECAST_HORIZON
    pred_blocks = pred_runoff_std.reshape(*pred_runoff_std.shape[:-1], num_blocks, FORECAST_HORIZON)
    true_blocks = true_runoff_std.reshape(*true_runoff_std.shape[:-1], num_blocks, FORECAST_HORIZON)

    pred_peak = pred_blocks.max(dim=-1).values
    true_peak = true_blocks.max(dim=-1).values
    peak_mask = true_peak > q95_std

    if peak_mask.any():
        peak_loss = F.mse_loss(pred_peak[peak_mask], true_peak[peak_mask])
    else:
        peak_loss = pred_runoff_std.new_zeros(())

    total = weighted_mse + RUNOFF_PEAK_LOSS_WEIGHT * peak_loss
    return total, base_mse, weighted_mse, peak_loss



# Computes the full multi-task state-transition loss for subcatchments, junctions, channels, and outfalls.
def compute_state_transition_loss(predictions: dict, targets: dict, scalers: dict):
    loss_snow = F.mse_loss(predictions["subcatchment"][..., 0], targets["subcatchment"][..., 0])

    loss_runoff, loss_runoff_base, loss_runoff_weighted, loss_runoff_peak = compute_runoff_extreme_aware_loss(
        pred_runoff_std=predictions["subcatchment"][..., 1],
        true_runoff_std=targets["subcatchment"][..., 1],
        scalers=scalers,
    )

    loss_subcatchment = 0.5 * loss_snow + 0.5 * loss_runoff

    loss_junction_depth = F.mse_loss(predictions["junction"][..., 0], targets["junction"][..., 0])

    loss_flood_occurrence = weighted_binary_focal_loss_with_logits(
        logits=predictions["junction"][..., 1],
        targets=targets["junction"][..., 1],
        positive_weight=float(scalers["flood_positive_weight"]),
        gamma=FLOOD_FOCAL_GAMMA,
    )

    # Flood loss updates only junction_flood_head because flood-head inputs are detached.
    loss_junction = loss_junction_depth + loss_flood_occurrence

    loss_channel = F.mse_loss(predictions["channel"], targets["channel"])
    loss_outfall = F.mse_loss(predictions["outfall"], targets["outfall"])

    total = (
        LOSS_WEIGHT_SUBCATCHMENT * loss_subcatchment
        + LOSS_WEIGHT_JUNCTION * loss_junction
        + LOSS_WEIGHT_CHANNEL * loss_channel
        + LOSS_WEIGHT_OUTFALL * loss_outfall
    )

    losses = {
        "total": total, "subcatchment": loss_subcatchment, "junction": loss_junction,
        "channel": loss_channel, "outfall": loss_outfall, "snow_depth": loss_snow,
        "runoff_volume": loss_runoff, "runoff_base": loss_runoff_base,
        "runoff_weighted": loss_runoff_weighted, "runoff_peak": loss_runoff_peak,
        "junction_depth": loss_junction_depth, "flood_occurrence": loss_flood_occurrence,
    }

    return total, losses




# Initializes streaming metric states for all continuous prediction targets.
def init_stateful_metric_states():
    return {"snow_depth": new_metric_state(), "runoff_volume": new_metric_state(), "junction_depth": new_metric_state(),
            "channel": new_metric_state(), "outfall": new_metric_state(), "all": new_metric_state()}




# Converts an entire standardized target group back to physical units.
def inverse_target(value: Tensor, scalers: dict, group: str) -> Tensor:
    return value * scalers[f"y_{group}_std"].to(value.device) + scalers[f"y_{group}_mean"].to(value.device)




# Converts one standardized target component back to physical units.
def inverse_target_component(value: Tensor, scalers: dict, group: str, component_index: int) -> Tensor:
    mean = scalers[f"y_{group}_mean"][component_index].to(value.device)
    std = scalers[f"y_{group}_std"][component_index].to(value.device)
    return value * std + mean




# Initializes sparse-event classification and zero-baseline diagnostic statistics.
def new_sparse_event_state(threshold: float) -> dict:
    return {
        "threshold": float(threshold), "n": 0, "obs_positive": 0, "pred_positive": 0, "pred_negative": 0,
        "tp": 0, "fp": 0, "fn": 0, "tn": 0, "sse_model": 0.0, "sse_zero": 0.0,
        "positive_n": 0, "positive_sse_model": 0.0, "positive_sse_zero": 0.0,
        "obs_max": float("-inf"), "pred_max": float("-inf"),
    }




# Updates sparse-event occurrence, zero-baseline, positive-event, and peak diagnostics.
def update_sparse_event_state(state: dict, pred: Tensor, target: Tensor) -> None:
    with torch.no_grad():
        sim = pred.detach().float().reshape(-1)
        obs = target.detach().float().reshape(-1)

        if sim.numel() != obs.numel():
            raise ValueError(f"Sparse diagnostic prediction/target sizes differ: {sim.numel()} vs {obs.numel()}")
        if obs.numel() == 0: return

        threshold = state["threshold"]
        obs_positive = obs > threshold
        pred_positive = sim > threshold
        pred_negative = sim < -threshold

        tp = obs_positive & pred_positive
        fp = (~obs_positive) & pred_positive
        fn = obs_positive & (~pred_positive)
        tn = (~obs_positive) & (~pred_positive)

        state["n"] += int(obs.numel())
        state["obs_positive"] += int(obs_positive.sum().item())
        state["pred_positive"] += int(pred_positive.sum().item())
        state["pred_negative"] += int(pred_negative.sum().item())
        state["tp"] += int(tp.sum().item())
        state["fp"] += int(fp.sum().item())
        state["fn"] += int(fn.sum().item())
        state["tn"] += int(tn.sum().item())

        error = sim - obs
        state["sse_model"] += float((error ** 2).sum().item())
        state["sse_zero"] += float((obs ** 2).sum().item())

        if bool(obs_positive.any().item()):
            sim_positive = sim[obs_positive]
            obs_positive_values = obs[obs_positive]
            positive_error = sim_positive - obs_positive_values

            state["positive_n"] += int(obs_positive_values.numel())
            state["positive_sse_model"] += float((positive_error ** 2).sum().item())
            state["positive_sse_zero"] += float((obs_positive_values ** 2).sum().item())

        state["obs_max"] = max(state["obs_max"], float(obs.max().item()))
        state["pred_max"] = max(state["pred_max"], float(sim.max().item()))




# Finalizes sparse-event diagnostics into occurrence, skill, positive-event, and peak metrics.
def finalize_sparse_event_state(state: dict, eps: float = 1e-12) -> dict:
    n = max(int(state["n"]), 1)
    tp, fp, fn = state["tp"], state["fp"], state["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan

    model_rmse_all = np.sqrt(state["sse_model"] / n)
    zero_rmse_all = np.sqrt(state["sse_zero"] / n)

    if state["sse_zero"] > eps: skill_vs_zero_all = 1.0 - state["sse_model"] / state["sse_zero"]
    else: skill_vs_zero_all = np.nan

    positive_n = state["positive_n"]

    if positive_n > 0:
        positive_rmse = np.sqrt(state["positive_sse_model"] / positive_n)
        positive_zero_rmse = np.sqrt(state["positive_sse_zero"] / positive_n)

        if state["positive_sse_zero"] > eps:
            positive_skill_vs_zero = 1.0 - state["positive_sse_model"] / state["positive_sse_zero"]
        else:
            positive_skill_vs_zero = np.nan
    else:
        positive_rmse = np.nan
        positive_zero_rmse = np.nan
        positive_skill_vs_zero = np.nan

    obs_max = state["obs_max"]
    pred_max = state["pred_max"]

    if np.isfinite(obs_max) and obs_max > eps:
        peak_ratio = pred_max / obs_max
        peak_bias_pct = 100.0 * (pred_max - obs_max) / obs_max
    else:
        peak_ratio = np.nan
        peak_bias_pct = np.nan

    return {
        "threshold": state["threshold"],
        "observed_positive_pct": 100.0 * state["obs_positive"] / n,
        "predicted_positive_pct": 100.0 * state["pred_positive"] / n,
        "predicted_negative_pct": 100.0 * state["pred_negative"] / n,
        "precision": float(precision) if not np.isnan(precision) else np.nan,
        "recall": float(recall) if not np.isnan(recall) else np.nan,
        "f1": float(f1) if not np.isnan(f1) else np.nan,
        "csi": float(csi) if not np.isnan(csi) else np.nan,
        "model_rmse_all": float(model_rmse_all),
        "zero_baseline_rmse_all": float(zero_rmse_all),
        "skill_vs_zero_all": float(skill_vs_zero_all) if not np.isnan(skill_vs_zero_all) else np.nan,
        "positive_only_rmse": float(positive_rmse) if not np.isnan(positive_rmse) else np.nan,
        "positive_only_zero_rmse": float(positive_zero_rmse) if not np.isnan(positive_zero_rmse) else np.nan,
        "positive_only_skill_vs_zero": float(positive_skill_vs_zero) if not np.isnan(positive_skill_vs_zero) else np.nan,
        "observed_max": float(obs_max), "predicted_max": float(pred_max),
        "peak_ratio": float(peak_ratio) if not np.isnan(peak_ratio) else np.nan,
        "peak_bias_pct": float(peak_bias_pct) if not np.isnan(peak_bias_pct) else np.nan,
        "tp": int(state["tp"]), "fp": int(state["fp"]), "fn": int(state["fn"]), "tn": int(state["tn"]),
    }




# Initializes sparse-event diagnostic states for all configured event-driven targets.
def init_stateful_sparse_diagnostic_states():
    return {group: new_sparse_event_state(threshold) for group, threshold in SPARSE_EVENT_THRESHOLDS.items()}




# Updates sparse-event diagnostics using inverse-standardized physical runoff predictions and targets.
def update_stateful_sparse_diagnostic_states(states: dict, predictions: dict, targets: dict, scalers: dict):
    pred_sub = inverse_target(predictions["subcatchment"], scalers, "subcatchment")
    true_sub = inverse_target(targets["subcatchment"], scalers, "subcatchment")
    update_sparse_event_state(states["runoff_volume"], pred_sub[..., 1], true_sub[..., 1])




# Initializes histogram-based flood-classification statistics for threshold evaluation.
def new_flood_classification_state():
    return {
        "num_bins": FLOOD_THRESHOLD_NUM_BINS,
        "positive_hist": np.zeros(FLOOD_THRESHOLD_NUM_BINS, dtype=np.int64),
        "negative_hist": np.zeros(FLOOD_THRESHOLD_NUM_BINS, dtype=np.int64),
    }




# Updates positive and negative flood-probability histograms from logits and binary targets.
def update_flood_classification_state(state: dict, logits: Tensor, target_binary: Tensor):
    probability = torch.sigmoid(logits.detach().float()).reshape(-1)
    target = target_binary.detach().reshape(-1) > 0.5
    num_bins = state["num_bins"]

    bin_index = torch.clamp((probability * num_bins).long(), min=0, max=num_bins - 1)
    positive_counts = torch.bincount(bin_index[target], minlength=num_bins)
    negative_counts = torch.bincount(bin_index[~target], minlength=num_bins)

    state["positive_hist"] += positive_counts.cpu().numpy()
    state["negative_hist"] += negative_counts.cpu().numpy()




# Finalizes all sparse-event diagnostic states.
def finalize_stateful_sparse_diagnostic_states(states: dict):
    return {key: finalize_sparse_event_state(value) for key, value in states.items()}





# Finalizes histogram-based flood classification metrics using either validation-optimized or fixed thresholds.
def finalize_flood_classification_state(state: dict, fixed_threshold=None):
    pos_hist = state["positive_hist"].astype(np.float64)
    neg_hist = state["negative_hist"].astype(np.float64)
    num_bins = state["num_bins"]

    total_pos, total_neg = pos_hist.sum(), neg_hist.sum()
    tp_all = np.cumsum(pos_hist[::-1])[::-1]
    fp_all = np.cumsum(neg_hist[::-1])[::-1]
    fn_all, tn_all = total_pos - tp_all, total_neg - fp_all

    denominator = tp_all + fp_all + fn_all
    csi_all = np.divide(tp_all, denominator, out=np.zeros_like(tp_all), where=denominator > 0)

    if fixed_threshold is None:
        idx = int(np.argmax(csi_all))
        threshold_source = "validation_csi_optimized"
    else:
        idx = int(np.clip(round(fixed_threshold * num_bins), 0, num_bins - 1))
        threshold_source = "fixed_from_validation"

    threshold = idx / num_bins
    tp, fp, fn, tn = tp_all[idx], fp_all[idx], fn_all[idx], tn_all[idx]

    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0 else np.nan
    csi = tp / (tp + fp + fn) if tp + fp + fn > 0 else np.nan
    false_alarm_ratio = fp / (tp + fp) if tp + fp > 0 else np.nan
    specificity = tn / (tn + fp) if tn + fp > 0 else np.nan
    total = total_pos + total_neg

    tp_desc = np.cumsum(pos_hist[::-1])
    fp_desc = np.cumsum(neg_hist[::-1])
    precision_desc = np.divide(tp_desc, tp_desc + fp_desc, out=np.ones_like(tp_desc), where=(tp_desc + fp_desc) > 0)

    if total_pos > 0:
        recall_desc = tp_desc / total_pos
        previous_recall = np.concatenate([[0.0], recall_desc[:-1]])
        pr_auc_approx = ((recall_desc - previous_recall) * precision_desc).sum()
    else:
        pr_auc_approx = np.nan

    return {
        "threshold": float(threshold), "threshold_source": threshold_source,
        "observed_positive_pct": float(100 * total_pos / total),
        "predicted_positive_pct": float(100 * (tp + fp) / total),
        "precision": float(precision), "recall": float(recall), "f1": float(f1), "csi": float(csi),
        "false_alarm_ratio": float(false_alarm_ratio), "specificity": float(specificity),
        "pr_auc_approx": float(pr_auc_approx),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }




# Initializes extreme-event metrics using fixed TRAIN-derived Q95 and Q99 thresholds.
def new_extreme_metric_state(q95, q99):
    return {
        "q95": float(q95), "q99": float(q99), "n95": 0, "n99": 0,
        "sse95": 0.0, "sse99": 0.0,
        "sum_obs95": 0.0, "sum_pred95": 0.0,
        "sum_obs99": 0.0, "sum_pred99": 0.0,
        "obs_max": float("-inf"), "pred_max": float("-inf"),
    }




# Updates Q95/Q99 extreme-event errors, mean ratios, and peak values.
def update_extreme_metric_state(state, pred, target):
    sim = pred.detach().float().reshape(-1)
    obs = target.detach().float().reshape(-1)
    error2 = (sim - obs).pow(2)

    mask95 = obs > state["q95"]
    mask99 = obs > state["q99"]

    if mask95.any():
        state["n95"] += int(mask95.sum().item())
        state["sse95"] += float(error2[mask95].sum().item())
        state["sum_obs95"] += float(obs[mask95].sum().item())
        state["sum_pred95"] += float(sim[mask95].sum().item())

    if mask99.any():
        state["n99"] += int(mask99.sum().item())
        state["sse99"] += float(error2[mask99].sum().item())
        state["sum_obs99"] += float(obs[mask99].sum().item())
        state["sum_pred99"] += float(sim[mask99].sum().item())

    state["obs_max"] = max(state["obs_max"], float(obs.max().item()))
    state["pred_max"] = max(state["pred_max"], float(sim.max().item()))




# Finalizes Q95/Q99 extreme-event RMSE, mean-ratio, and peak diagnostics.
def finalize_extreme_metric_state(state):
    n95, n99 = state["n95"], state["n99"]

    rmse95 = np.sqrt(state["sse95"] / n95) if n95 > 0 else np.nan
    rmse99 = np.sqrt(state["sse99"] / n99) if n99 > 0 else np.nan
    ratio95 = state["sum_pred95"] / state["sum_obs95"] if state["sum_obs95"] != 0 else np.nan
    ratio99 = state["sum_pred99"] / state["sum_obs99"] if state["sum_obs99"] != 0 else np.nan

    obs_max, pred_max = state["obs_max"], state["pred_max"]
    peak_ratio = pred_max / obs_max if obs_max > 0 else np.nan
    peak_bias_pct = 100 * (pred_max - obs_max) / obs_max if obs_max > 0 else np.nan

    return {
        "train_q95_threshold": state["q95"], "train_q99_threshold": state["q99"],
        "rmse_p95plus": float(rmse95), "rmse_p99plus": float(rmse99),
        "mean_ratio_p95plus": float(ratio95), "mean_ratio_p99plus": float(ratio99),
        "observed_max": float(obs_max), "predicted_max": float(pred_max),
        "peak_ratio": float(peak_ratio), "peak_bias_pct": float(peak_bias_pct),
    }




# Initializes extreme-event metric states for all prediction groups from TRAIN tail thresholds.
def init_stateful_extreme_metric_states(scalers):
    groups = ["snow_depth", "runoff_volume", "junction_depth", "channel", "outfall"]
    return {
        group: new_extreme_metric_state(
            q95=scalers["tail_thresholds"][group]["q95"],
            q99=scalers["tail_thresholds"][group]["q99"],
        )
        for group in groups
    }





# Updates TRAIN-threshold extreme metrics in physical units for all continuous prediction groups.
def update_stateful_extreme_metric_states(states: dict, predictions: dict, targets: dict, scalers: dict):
    pred_sub = inverse_target(predictions["subcatchment"], scalers, "subcatchment").clamp_min(0.0)
    true_sub = inverse_target(targets["subcatchment"], scalers, "subcatchment")

    update_extreme_metric_state(states["snow_depth"], pred_sub[..., 0], true_sub[..., 0])
    update_extreme_metric_state(states["runoff_volume"], pred_sub[..., 1], true_sub[..., 1])

    pred_junction_depth = inverse_target_component(predictions["junction"][..., 0], scalers, "junction", 0).clamp_min(0.0)
    true_junction_depth = inverse_target_component(targets["junction"][..., 0], scalers, "junction", 0)
    update_extreme_metric_state(states["junction_depth"], pred_junction_depth, true_junction_depth)

    for group in ["channel", "outfall"]:
        pred = inverse_target(predictions[group], scalers, group).clamp_min(0.0)
        true = inverse_target(targets[group], scalers, group)
        update_extreme_metric_state(states[group], pred, true)




# Updates physical-unit streaming regression metrics for all continuous prediction targets.
def update_stateful_metric_states(states: dict, predictions: dict, targets: dict, scalers: dict):
    pred_sub = inverse_target(predictions["subcatchment"], scalers, "subcatchment").clamp_min(0.0)
    true_sub = inverse_target(targets["subcatchment"], scalers, "subcatchment")

    update_metric_state(states["snow_depth"], pred_sub[..., 0], true_sub[..., 0])
    update_metric_state(states["runoff_volume"], pred_sub[..., 1], true_sub[..., 1])
    update_metric_state(states["all"], pred_sub, true_sub)

    pred_junction_depth = inverse_target_component(predictions["junction"][..., 0], scalers, "junction", 0).clamp_min(0.0)
    true_junction_depth = inverse_target_component(targets["junction"][..., 0], scalers, "junction", 0)

    update_metric_state(states["junction_depth"], pred_junction_depth, true_junction_depth)
    update_metric_state(states["all"], pred_junction_depth, true_junction_depth)

    for group in ["channel", "outfall"]:
        pred = inverse_target(predictions[group], scalers, group).clamp_min(0.0)
        true = inverse_target(targets[group], scalers, group)
        update_metric_state(states[group], pred, true)
        update_metric_state(states["all"], pred, true)




# Finalizes all streaming regression metric states.
def finalize_stateful_metric_states(states: dict):
    return {key: finalize_metric_state(value) for key, value in states.items()}




# Returns empty TRAIN metric placeholders when full regression metrics are intentionally skipped.
def empty_stateful_metric_report():

    groups = ["all", "snow_depth", "runoff_volume", "junction_depth", "channel", "outfall"]
    return {group: {"rmse": None, "nse": None, "kge": None, "r": None} for group in groups}
