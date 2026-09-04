#!/usr/bin/env python3
"""Register H67/NB0 seed1/2 configs. Do not launch training."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
GEN = EXP / "configs/generated"
OUTPUT = REPO / "neuron_autoresearch/DSEC_SEED12_REGISTRY_20260813.json"
OUTPUT_MD = OUTPUT.with_suffix(".md")
SEEDS = (1, 2)
SOURCES = (
    {
        "method": "NB0",
        "config": GEN / "dsec_fullres_w15_NB0_equal_plus10_ep40.yml",
        "prefix": "dsec_fullres_w15_NB0_equal_plus10_ep40",
    },
    {
        "method": "H67",
        "config": GEN / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml",
        "prefix": "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    rows = []
    for source in SOURCES:
        base_path = source["config"]
        if not base_path.is_file():
            raise FileNotFoundError(base_path)
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        for seed in SEEDS:
            config = deepcopy(base)
            name = f"{source['prefix']}_seed{seed}"
            config["experiment"] = name
            config.setdefault("runtime", {})["seed"] = seed
            config["runtime"]["seed_registry"] = "date_table_f_optional_after_qf"
            config["runtime"]["auto_launch"] = False
            config["note"] = (
                f"Registered seed{seed} robustness config copied from {base_path.name}. "
                "This is a same-parent fullres continuation seed, not an independent "
                "from-scratch lineage. Do not auto-start; DATE default claim is seed0."
            )
            path = GEN / f"{name}.yml"
            rendered = yaml.safe_dump(config, sort_keys=False, width=100)
            if path.exists() and path.read_text(encoding="utf-8") != rendered:
                raise RuntimeError(f"generated config drift: {path}")
            path.write_text(rendered, encoding="utf-8")
            rows.append(
                {
                    "method": source["method"],
                    "seed": seed,
                    "status": "registered_not_launched",
                    "config": str(path.resolve()),
                    "config_sha256": sha256(path),
                    "source_config": str(base_path.resolve()),
                    "source_config_sha256": sha256(base_path),
                }
            )
    payload = {
        "schema": "dsec_fullres_seed12_registry_v1",
        "status": "REGISTERED_NOT_LAUNCHED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "paper_default": "seed0_only",
        "launch_gate": (
            "optional after H81, Local5 40-50, final mainline audit, and QF5-QF8; "
            "do not steal the live GPU queue"
        ),
        "claim_boundary": (
            "same-parent fullres continuation seeds; not independent crop-to-fullres lineages"
        ),
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# DSEC seed1/2 registry",
        "",
        "Status: `REGISTERED_NOT_LAUNCHED`. DATE default claim remains seed0.",
        "",
        "| method | seed | status | config |",
        "|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['seed']} | {row['status']} | `{Path(row['config']).name}` |"
        )
    lines.extend(
        [
            "",
            "These configs reuse the frozen fullres continuation parents. They are not queued.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
