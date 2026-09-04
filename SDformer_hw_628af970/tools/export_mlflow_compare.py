#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
import yaml


def maybe_parse_dict(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = ast.literal_eval(raw)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def load_eval_metrics(repo_root: Path, runid: str) -> dict[str, Any]:
    metrics_dir = repo_root / "third_party" / "SDformerFlow" / "results_inference" / runid
    if not metrics_dir.exists():
        return {}
    metric_files = sorted(metrics_dir.glob("metrics_*.yml"))
    if not metric_files:
        return {}
    with metric_files[-1].open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return {f"eval_{k}": v for k, v in data.items()}


def history_rows(client: MlflowClient, runid: str, metric: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in client.get_metric_history(runid, metric):
        rows.append(
            {
                "run_id": runid,
                "metric": metric,
                "step": item.step,
                "timestamp_ms": item.timestamp,
                "value": item.value,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def try_plot(metric_rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    outputs: list[Path] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in metric_rows:
        grouped.setdefault(row["metric"], []).append(row)

    for metric, rows in grouped.items():
        plt.figure(figsize=(8, 5))
        run_ids = sorted({row["run_id"] for row in rows})
        for runid in run_ids:
            series = sorted((r for r in rows if r["run_id"] == runid), key=lambda x: x["step"])
            if not series:
                continue
            plt.plot([r["step"] for r in series], [r["value"] for r in series], marker="o", label=runid[:8])
        plt.title(metric)
        plt.xlabel("step")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        out = output_dir / f"{metric}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        outputs.append(out)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Export comparable MLflow run summaries and plots")
    parser.add_argument("--tracking-uri", default="file:///root/private_data/sdformer_mlflow")
    parser.add_argument("--repo-root", default="/root/private_data/work/SDformer")
    parser.add_argument("--runids", nargs="+", required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "train_loss",
            "valid_loss",
            "lr",
            "epoch_time_sec",
            "train_step_time_sec",
            "valid_time_sec",
            "max_gpu_mem_gib",
        ],
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/mlflow_compare",
        help="relative to repo-root unless absolute",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    summary_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for runid in args.runids:
        run = client.get_run(runid)
        params = run.data.params
        loader_cfg = maybe_parse_dict(params.get("loader", ""))
        optimizer_cfg = maybe_parse_dict(params.get("optimizer", ""))
        model_cfg = maybe_parse_dict(params.get("model", ""))
        eval_metrics = load_eval_metrics(repo_root, runid)

        start_time = (
            datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc).isoformat()
            if run.info.start_time
            else ""
        )
        end_time = (
            datetime.fromtimestamp(run.info.end_time / 1000, tz=timezone.utc).isoformat()
            if run.info.end_time
            else ""
        )
        duration_min = ""
        if run.info.start_time and run.info.end_time:
            duration_min = round((run.info.end_time - run.info.start_time) / 60000, 3)

        row = {
            "run_id": runid,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time_utc": start_time,
            "end_time_utc": end_time,
            "duration_min": duration_min,
            "artifact_uri": run.info.artifact_uri,
            "model_name": model_cfg.get("name", ""),
            "num_params": params.get("number of params", ""),
            "batch_size": loader_cfg.get("batch_size", ""),
            "n_epochs": loader_cfg.get("n_epochs", ""),
            "crop": loader_cfg.get("crop", ""),
            "lr_init": optimizer_cfg.get("lr", ""),
        }
        for metric_name, metric_value in run.data.metrics.items():
            row[f"metric_{metric_name}"] = metric_value
        row.update(eval_metrics)
        summary_rows.append(row)

        for metric in args.metrics:
            metric_rows.extend(history_rows(client, runid, metric))

    summary_fieldnames = sorted({key for row in summary_rows for key in row.keys()})
    metrics_fieldnames = ["run_id", "metric", "step", "timestamp_ms", "value"]
    write_csv(output_dir / "run_summary.csv", summary_rows, summary_fieldnames)
    write_csv(output_dir / "metric_history.csv", metric_rows, metrics_fieldnames)
    plot_paths = try_plot(metric_rows, output_dir)

    report_path = output_dir / "README.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# MLflow Run Comparison\n\n")
        handle.write(f"Generated at: {datetime.now(timezone.utc).isoformat()}\n\n")
        handle.write("## Runs\n\n")
        for row in summary_rows:
            handle.write(
                f"- `{row['run_id']}` status={row['status']} model={row.get('model_name','')} "
                f"train_loss={row.get('metric_train_loss','')} valid_loss={row.get('metric_valid_loss','')}\n"
            )
        handle.write("\n## Files\n\n")
        handle.write(f"- `run_summary.csv`\n")
        handle.write(f"- `metric_history.csv`\n")
        for plot in plot_paths:
            handle.write(f"- `{plot.name}`\n")

    print(f"summary_csv: {output_dir / 'run_summary.csv'}")
    print(f"metric_history_csv: {output_dir / 'metric_history.csv'}")
    print(f"report_md: {report_path}")
    for plot in plot_paths:
        print(f"plot: {plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
