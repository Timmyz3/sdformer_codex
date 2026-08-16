#!/usr/bin/env python3
"""Ranked-checkpoint receipt adapter for the frozen descriptor analyzer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import analyze_ds_flm_descriptor_manifest as analyzer
from local5_release_receipt import validate_release_receipt


def validate_ranked_identity_receipt(identity: dict[str, Any]) -> None:
    receipt_path = Path(str(identity.get("release_receipt", ""))).resolve()
    receipt = validate_release_receipt(
        receipt_path,
        str(identity.get("release_receipt_sha256", "")),
    )
    checks = {
        "watcher_session_uuid": (
            identity.get("watcher_session_uuid")
            == receipt.get("watcher_session_uuid")
        ),
        "ranking_path": receipt.get("ranking_path") == identity.get("ranking"),
        "ranking_sha256": (
            receipt.get("ranking_sha256") == identity.get("ranking_sha256")
        ),
        "checkpoint_path": (
            receipt.get("checkpoint_path") == identity.get("checkpoint")
        ),
        "checkpoint_sha256": (
            receipt.get("checkpoint_sha256")
            == identity.get("checkpoint_sha256")
        ),
        "config_path": receipt.get("config_path") == identity.get("config"),
        "config_sha256": (
            receipt.get("config_sha256") == identity.get("config_sha256")
        ),
        "best_epoch": receipt.get("best_epoch") == identity.get("best_epoch"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "ranked run identity release receipt mismatch: "
            + ",".join(failed)
        )


analyzer.validate_release_receipt = validate_ranked_identity_receipt


if __name__ == "__main__":
    raise SystemExit(analyzer.main())
