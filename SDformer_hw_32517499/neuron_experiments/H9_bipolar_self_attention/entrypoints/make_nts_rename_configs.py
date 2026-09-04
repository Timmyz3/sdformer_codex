"""Create NTS-named aliases for existing h60 full30 configs (experiment field only)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"

ALIASES = [
    ("ntx_h60_full30.yml", "nts01_full30.yml", "nts01_full30", "NTS-01: TX+SC score fusion, mu=0.1, alpha_k=0.02"),
    (
        "ntx_h60_v2_mu005_a003_full30.yml",
        "nts03_mu005_a003_full30.yml",
        "nts03_mu005_a003_full30",
        "NTS-03: TX+SC + K_mag, mu=0.05, alpha_k=0.03 (pending full30)",
    ),
]


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> int:
    for src_name, dst_name, experiment, note in ALIASES:
        src = GENERATED / src_name
        if not src.exists():
            print(f"skip missing {src}")
            continue
        cfg = read_yaml(src)
        cfg = deepcopy(cfg)
        cfg["experiment"] = experiment
        cfg["note"] = note
        dst = GENERATED / dst_name
        write_yaml(dst, cfg)
        print(dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())