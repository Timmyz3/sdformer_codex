#!/usr/bin/env python3
"""Exact-graph successor to the source-only M1321 decoder adapter.

M1322 found that M1321 validated the projected decoder rows but did not prove
that the complete ordered stream was a duplicate-free 0..9879 graph.  This
additive wrapper closes that boundary before delegating payload semantics to
the frozen M1321 implementation.  It remains read-only and source-only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
M1321_PATH = HERE / "build_m1321_ep34_decoder_capture_adapter_source.py"
_SPEC = importlib.util.spec_from_file_location("m1321_frozen", M1321_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load frozen M1321 adapter")
M1321 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(M1321)

AdapterError = M1321.AdapterError
DEFAULT_CAPTURE_ROOT = M1321.DEFAULT_CAPTURE_ROOT
EXPECTED_ORDERED_ROWS = M1321.EXPECTED_ORDERED_ROWS


def require(value: bool, message: str) -> None:
    if not value:
        raise AdapterError(message)


def validate_complete_ordered_graph(capture_root: Path) -> dict[str, int]:
    """Prove file ordinal == exact-int global_order for the complete stream."""
    root = Path(capture_root)
    ordered_path = root / "unified_ordered_records.jsonl"
    M1321.regular(ordered_path, "ordered records")
    count = 0
    with ordered_path.open("r", encoding="utf-8") as stream:
        for file_ordinal, line in enumerate(stream):
            require(line.endswith("\n"), "ordered JSONL line lacks terminal newline")
            row = M1321.strict_json_text(line)
            require(type(row) is dict, "ordered JSONL row is not an object")
            global_order = row.get("global_order")
            require(type(global_order) is int,
                    "global_order is not an exact integer")
            require(global_order == file_ordinal,
                    "global_order differs from file ordinal")
            count += 1
    require(count == EXPECTED_ORDERED_ROWS,
            "ordered population is not 9880")
    return {"rows": count, "first_global_order": 0,
            "last_global_order": count - 1}


def validate_weight_identities(rows: Any, checkpoint_sha256: str):
    """Reject bool ordinals before delegating all frozen M1321 checks."""
    require(type(rows) is list and len(rows) == 4,
            "weight identity population must be four")
    for ordinal, row in enumerate(rows):
        require(type(row) is dict, "weight identity row is not an object")
        module_ordinal = row.get("module_ordinal")
        require(type(module_ordinal) is int,
                "module ordinal is not an exact integer")
        require(module_ordinal == ordinal, "module ordinal drift")
    return M1321.validate_weight_identities(rows, checkpoint_sha256)


def audit_capture(capture_root: Path = DEFAULT_CAPTURE_ROOT,
                  weight_identities: Any | None = None,
                  checkpoint_sha256: str | None = None) -> dict[str, Any]:
    graph = validate_complete_ordered_graph(capture_root)
    if weight_identities is not None or checkpoint_sha256 is not None:
        require(weight_identities is not None and checkpoint_sha256 is not None,
                "weights and checkpoint SHA must be supplied together")
        weights = validate_weight_identities(weight_identities, checkpoint_sha256)
        result = M1321.audit_capture(capture_root)
        result["weight_identities"] = weights
    else:
        result = M1321.audit_capture(capture_root)
    result["schema"] = "m1436_ep34_decoder_capture_adapter_exact_graph_audit_r1"
    result["status"] = (
        "PASS_EXACT_GRAPH_SOURCE_AUDIT__RESULT_HAMMER_AND_WRITER_REQUIRED")
    result["ordered_graph"] = graph
    result["claim_boundary"] = {
        "source_only": True,
        "read_only": True,
        "capture_result_hammered": False,
        "normalized_payload_written": False,
        "production_replay": False,
        "cycles": False,
        "traffic": False,
        "speedup": False,
        "system_speedup": False,
        "energy": False,
        "ppa": False,
        "table_a": False,
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", action="store_true")
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    require(args.source_audit, "only --source-audit is available")
    result = audit_capture(args.capture_root)
    print(json.dumps({key: value for key, value in result.items()
                      if key != "calls"}, indent=2, sort_keys=True,
                     allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AdapterError as error:
        print("M1436_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
