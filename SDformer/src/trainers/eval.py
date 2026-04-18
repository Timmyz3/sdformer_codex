"""Local evaluator and result-table generator."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import build_dataset
from src.datasets.transforms import move_batch_to_device
from src.models.registry import build_model
from src.trainers.losses import build_loss
from src.trainers.metrics import compute_metrics
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.logging import write_csv, write_json, write_markdown_table
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg["dataset"]["name"] = args.dataset
        if args.dataset == "mvsec":
            cfg["dataset"]["resolution"] = [260, 346]
            cfg["dataset"]["crop"] = [256, 256]

    set_seed(cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["runtime"]["device"] != "cpu" else "cpu")
    dataset = build_dataset(cfg, "eval")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg["runtime"]["num_workers"])

    model = build_model(cfg).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=str(device))
    if hasattr(model, "configure_backend"):
        model.configure_backend()

    loss_fn = build_loss(cfg, device)
    model.eval()
    total_loss = 0.0
    total_metrics: Dict[str, float] = {}
    count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            pred = outputs["flow_pred"]
            flow_list = outputs["aux"]["flow_list"]
            loss = loss_fn(flow_list, batch["gt_flow"], batch["valid_mask"], gamma=cfg["loss"]["gamma"])
            metrics = compute_metrics(cfg, pred, batch["gt_flow"], batch["valid_mask"])
            total_loss += float(loss.item())
            count += 1
            for key, value in metrics.items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value

    summary = {
        "config": args.config,
        "variant": cfg["model"]["name"],
        "dataset": cfg["dataset"]["name"],
        "loss": total_loss / max(count, 1),
    }
    for key, value in total_metrics.items():
        summary[key] = value / max(count, 1)

    out_dir = Path("experiments/results")
    write_json(out_dir / "tables" / f"{cfg['model']['name']}_{cfg['dataset']['name']}.json", summary)

    if args.write_summary:
        summary_path = out_dir / "tables" / "ablation_summary.csv"
        existing = []
        if summary_path.exists():
            import csv

            with summary_path.open("r", encoding="utf-8") as handle:
                existing = list(csv.DictReader(handle))
        existing.append(summary)
        write_csv(summary_path, existing)
        write_markdown_table(out_dir / "tables" / "ablation_summary.md", existing)

        dataset_table = "dsec_main" if cfg["dataset"]["name"] == "dsec" else "mvsec_generalization"
        dataset_summary_path = out_dir / "tables" / f"{dataset_table}.csv"
        existing_dataset = []
        if dataset_summary_path.exists():
            import csv

            with dataset_summary_path.open("r", encoding="utf-8") as handle:
                existing_dataset = list(csv.DictReader(handle))
        existing_dataset.append(summary)
        write_csv(dataset_summary_path, existing_dataset)
        write_markdown_table(out_dir / "tables" / f"{dataset_table}.md", existing_dataset)


if __name__ == "__main__":
    main()
