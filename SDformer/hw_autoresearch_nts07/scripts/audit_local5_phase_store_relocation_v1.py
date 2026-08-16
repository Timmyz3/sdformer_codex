#!/usr/bin/env python3
"""为已密封 Phase Array Store 建立当前位置可解析的外部绑定 receipt。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-package", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--identity-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = args.phase_package.resolve()
    complete_path = package / "complete.json"
    complete = read_json(complete_path)
    if (
        complete.get("schema") != "local5_phase_array_store_canary_complete_v3"
        or complete.get("status") != "PASS_SEALED_STREAMING_MMAP_CANARY_NOT_G0"
        or complete.get("formal_g0") != "DENY"
    ):
        raise ValueError("phase package is not the accepted v3 non-G0 package")
    paths = {
        "source_trace": args.source_trace.resolve(),
        "expected_identity_manifest": args.identity_manifest.resolve(),
        "expected_identity_receipt": args.identity_receipt.resolve(),
    }
    original = complete.get("external_bindings")
    if not isinstance(original, dict):
        raise ValueError("phase complete lacks external bindings")
    resolved: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        expected = original.get(name, {}).get("sha256")
        observed = sha256(path)
        if expected != observed:
            raise ValueError(f"{name} SHA differs after relocation")
        resolved[name] = {
            "path": os.path.relpath(path, start=package),
            "path_base": "phase_package_dir",
            "sha256": observed,
        }
    report = {
        "schema": "local5_phase_store_relocation_audit_v1",
        "status": "PASS_RELOCATED_EXTERNAL_BINDINGS_NOT_G0",
        "evidence": "[归档完整性审计]",
        "formal_g0": "DENY",
        "phase_package": {
            "path": str(package),
            "complete_sha256": sha256(complete_path),
            "identity": complete.get("identity"),
        },
        "resolved_external_bindings": resolved,
        "original_path_issue": (
            "原 v3 complete 的 absolute path 含父级 staging 名；SHA 正确但路径已陈旧"
        ),
        "repair_scope": (
            "不修改已密封原包；本 receipt 提供当前位置相对路径并绑定原 complete SHA"
        ),
        "source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "boundary": [
            "只修复归档 locator 可解析性，不改变 RTL、数值、trace 或 formal G0 结论",
            "同目录 SHA 仍不是外部不可篡改信任根",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
