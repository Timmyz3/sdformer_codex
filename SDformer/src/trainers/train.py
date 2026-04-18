"""Local trainer for baseline and early variants."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import build_dataset
from src.datasets.transforms import move_batch_to_device
from src.models.registry import build_model
from src.trainers.losses import build_loss
from src.trainers.metrics import compute_metrics
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config import load_config
from src.utils.logging import write_csv, write_json
from src.utils.seed import set_seed


def evaluate_once(cfg: Dict, model, loader, loss_fn, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = {}
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            flow_list = outputs["aux"]["flow_list"]
            pred = outputs["flow_pred"]
            loss = loss_fn(flow_list, batch["gt_flow"], batch["valid_mask"], gamma=cfg["loss"]["gamma"])
            metrics = compute_metrics(cfg, pred, batch["gt_flow"], batch["valid_mask"])
            total_loss += float(loss.item())
            count += 1
            for key, value in metrics.items():
                total[key] = total.get(key, 0.0) + value
    results = {"loss": total_loss / max(count, 1)}
    for key, value in total.items():
        results[key] = value / max(count, 1)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--output-dir", default="experiments/logs/train")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["runtime"]["device"] != "cpu" else "cpu")
    train_dataset = build_dataset(cfg, "train")
    eval_dataset = build_dataset(cfg, "eval")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["runtime"].get("batch_size", 1),
        shuffle=True,
        num_workers=cfg["runtime"]["num_workers"],
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["runtime"]["num_workers"],
    )

    model = build_model(cfg).to(device)
    if hasattr(model, "configure_backend"):
        model.configure_backend()

    optimizer = AdamW(model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["optimizer"]["milestones"],
        gamma=0.5,
    )
    loss_fn = build_loss(cfg, device)

    start_epoch = 0
    if args.checkpoint:
        state = load_checkpoint(args.checkpoint, model, optimizer, scheduler, map_location=str(device))
        start_epoch = int(state.get("epoch", 0)) + 1

    best_aee = float("inf")
    epochs = args.epochs or cfg["runtime"].get("epochs", len(cfg["optimizer"]["milestones"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"train-{epoch}", leave=False):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            outputs = model(batch)
            flow_list = outputs["aux"]["flow_list"]
            loss = loss_fn(flow_list, batch["gt_flow"], batch["valid_mask"], gamma=cfg["loss"]["gamma"])
            loss.backward()
            if cfg["loss"]["clip_grad"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["loss"]["clip_grad"])
            optimizer.step()
            running_loss += float(loss.item())

        scheduler.step()
        eval_results = evaluate_once(cfg, model, eval_loader, loss_fn, device)
        eval_results["train_loss"] = running_loss / max(len(train_loader), 1)
        eval_results["epoch"] = epoch

        checkpoint_path = output_dir / f"{cfg['model']['name']}_epoch{epoch}.pth"
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, eval_results)
        write_json(output_dir / f"{cfg['model']['name']}_epoch{epoch}.json", eval_results)
        history.append(eval_results)
        write_csv(output_dir / f"{cfg['model']['name']}_history.csv", history)

        if eval_results.get("AEE", float("inf")) < best_aee:
            best_aee = eval_results["AEE"]
            save_checkpoint(output_dir / f"{cfg['model']['name']}_best.pth", model, optimizer, scheduler, epoch, eval_results)


if __name__ == "__main__":
    main()
