import numpy as np

# Formats metric values safely for reporting.
def safe_metric_value(value) -> str:
    if value is None: return "nan"
    try:
        if np.isnan(value): return "nan"
    except TypeError: pass
    return f"{value:.4f}"




# Prints losses, regression metrics, sparse-event diagnostics, flood classification, and extreme diagnostics for one split.
def print_stateful_split_report(split_name: str, report: dict):
    loss, metrics = report["loss"], report["metrics"]

    print(f"{split_name}:")
    print(f"  transitions={report['num_transitions']} | scenarios={report['num_scenarios']}")
    print(
        f"  loss_total={loss['total']:.6f} | sub={loss['subcatchment']:.6f} | "
        f"junc={loss['junction']:.6f} | chan={loss['channel']:.6f} | outfall={loss['outfall']:.6f}"
    )

    for group in ["all", "snow_depth", "runoff_volume", "junction_depth", "channel", "outfall"]:
        m = metrics[group]
        print(
            f"  {group:22s} | RMSE={safe_metric_value(m['rmse'])} | NSE={safe_metric_value(m['nse'])} | "
            f"KGE={safe_metric_value(m['kge'])} | r={safe_metric_value(m['r'])} | "
            f"alpha={safe_metric_value(m.get('alpha'))} | beta={safe_metric_value(m.get('beta'))}"
        )

    sparse_diagnostics = report.get("sparse_diagnostics")
    if sparse_diagnostics:
        print()
        print("  Sparse-event diagnostics (zero-collapse check):")

        for group in ["runoff_volume"]:
            d = sparse_diagnostics[group]
            print()
            print(f"  {group} | positive_threshold={d['threshold']:.2e}:")
            print(
                "    occurrence | "
                f"observed_positive={safe_metric_value(d['observed_positive_pct'])}% | "
                f"predicted_positive={safe_metric_value(d['predicted_positive_pct'])}% | "
                f"predicted_negative={safe_metric_value(d['predicted_negative_pct'])}%"
            )
            print(
                "    detection  | "
                f"Precision={safe_metric_value(d['precision'])} | Recall={safe_metric_value(d['recall'])} | "
                f"F1={safe_metric_value(d['f1'])} | CSI={safe_metric_value(d['csi'])}"
            )
            print(
                "    zero-test  | "
                f"Model_RMSE={safe_metric_value(d['model_rmse_all'])} | "
                f"Zero_RMSE={safe_metric_value(d['zero_baseline_rmse_all'])} | "
                f"Skill_vs_zero={safe_metric_value(d['skill_vs_zero_all'])}"
            )
            print(
                "    positive   | "
                f"Model_RMSE={safe_metric_value(d['positive_only_rmse'])} | "
                f"Zero_RMSE={safe_metric_value(d['positive_only_zero_rmse'])} | "
                f"Skill_vs_zero={safe_metric_value(d['positive_only_skill_vs_zero'])}"
            )
            print(
                "    extremes   | "
                f"Observed_max={safe_metric_value(d['observed_max'])} | "
                f"Predicted_max={safe_metric_value(d['predicted_max'])} | "
                f"Peak_ratio={safe_metric_value(d['peak_ratio'])} | "
                f"Peak_bias={safe_metric_value(d['peak_bias_pct'])}%"
            )

    flood = report.get("flood_classification")
    if flood:
        print()
        print("  Flood-occurrence classification:")
        print(
            "    threshold | "
            f"value={safe_metric_value(flood['threshold'])} | source={flood['threshold_source']}"
        )
        print(
            "    occurrence | "
            f"observed_positive={safe_metric_value(flood['observed_positive_pct'])}% | "
            f"predicted_positive={safe_metric_value(flood['predicted_positive_pct'])}%"
        )
        print(
            "    detection | "
            f"Precision={safe_metric_value(flood['precision'])} | Recall={safe_metric_value(flood['recall'])} | "
            f"F1={safe_metric_value(flood['f1'])} | CSI={safe_metric_value(flood['csi'])}"
        )
        print(
            "    additional | "
            f"FAR={safe_metric_value(flood['false_alarm_ratio'])} | "
            f"Specificity={safe_metric_value(flood['specificity'])} | "
            f"PR_AUC_approx={safe_metric_value(flood['pr_auc_approx'])}"
        )

    extremes = report.get("extreme_diagnostics")
    if extremes:
        print()
        print("  Regression extreme diagnostics:")

        for group in ["snow_depth", "runoff_volume", "junction_depth", "channel", "outfall"]:
            e = extremes[group]
            print()
            print(f"  {group}:")
            print(
                "    tail | "
                f"Q95={safe_metric_value(e['train_q95_threshold'])} | "
                f"RMSE_Q95+={safe_metric_value(e['rmse_p95plus'])} | "
                f"MeanRatio_Q95+={safe_metric_value(e['mean_ratio_p95plus'])}"
            )
            print(
                "    extreme-tail | "
                f"Q99={safe_metric_value(e['train_q99_threshold'])} | "
                f"RMSE_Q99+={safe_metric_value(e['rmse_p99plus'])} | "
                f"MeanRatio_Q99+={safe_metric_value(e['mean_ratio_p99plus'])}"
            )
            print(
                "    maximum | "
                f"Observed={safe_metric_value(e['observed_max'])} | Predicted={safe_metric_value(e['predicted_max'])} | "
                f"PeakRatio={safe_metric_value(e['peak_ratio'])} | PeakBias={safe_metric_value(e['peak_bias_pct'])}%"
            )




# Prints the complete TRAIN and VALIDATION report for one epoch.
def print_stateful_epoch_report(epoch: int, train_report: dict, val_report: dict):
    print()
    print("=" * 88)
    print(f"Epoch {epoch:03d} | Transformer-context + HEATConv + GRU state-transition")
    print("=" * 88)
    print_stateful_split_report("TRAIN", train_report)
    print()
    print_stateful_split_report("VALIDATION (weather + static graph + latent memory)", val_report)
    print("=" * 88)
    print()
