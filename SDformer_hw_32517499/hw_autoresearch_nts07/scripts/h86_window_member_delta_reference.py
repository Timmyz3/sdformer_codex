#!/usr/bin/env python3
"""H86 / Motion-directory successor: window Class File + member insert/delete.

This is the hardware object, not production RTL.

Motion 389 remaining door: keep the RQTB/Q7 quotient to a new storage and
schedule boundary, and keep destination identity. The live H67 directory
already streams class hist + a 450-entry scored list and then recomputes
token exp (C7). H86 changes that boundary:

    ST_CLASSIFY window Class File
    ST_SHIFTMAX  (C7: * multiplicity; H82/H86: one vote)
    ST_EXPAND    dest-owned K[i] * gate_c, no length-450 gate operand
    ST_PATCH     adjacent hardware-scan row applies member insert/delete

Algorithm H86 (not yet frozen, no GPU) said the same sentence: window-level
class-major, member insert/delete is the execution object. This TLM is how
the Motion line receives that SHA later. It is not an H67 rename and must
not enter docs/359.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.h82_class_file_reference import (  # noqa: E402
    TOKENS,
    build_class_file,
    c7_multiplicity_weighted_gates,
    jaccard,
    q7_codes,
)
from scripts.h82_class_file_reference import window_codes  # noqa: E402
from scripts.h82_hwscan_delta_directory import scan_delta  # noqa: E402


@dataclass(frozen=True)
class PatchOp:
    kind: str  # insert | delete | stay
    class_id: int
    token: int


def member_patch(prev: np.ndarray, curr: np.ndarray) -> tuple[PatchOp, ...]:
    prev_sets: dict[int, set[int]] = {}
    curr_sets: dict[int, set[int]] = {}
    for index, class_id in enumerate(np.asarray(prev).tolist()):
        prev_sets.setdefault(int(class_id), set()).add(int(index))
    for index, class_id in enumerate(np.asarray(curr).tolist()):
        curr_sets.setdefault(int(class_id), set()).add(int(index))
    ops: list[PatchOp] = []
    for class_id in sorted(set(prev_sets) | set(curr_sets)):
        was = prev_sets.get(class_id, set())
        now = curr_sets.get(class_id, set())
        for token in sorted(now - was):
            ops.append(PatchOp("insert", class_id, token))
        for token in sorted(was - now):
            ops.append(PatchOp("delete", class_id, token))
        for token in sorted(was & now):
            ops.append(PatchOp("stay", class_id, token))
    return tuple(ops)


def apply_patch(prev: np.ndarray, ops: tuple[PatchOp, ...]) -> np.ndarray:
    """Rebuild membership from prev + insert/delete. stay is informational."""

    roster: dict[int, set[int]] = {}
    for index, class_id in enumerate(np.asarray(prev).tolist()):
        roster.setdefault(int(class_id), set()).add(int(index))
    for op in ops:
        if op.kind == "insert":
            roster.setdefault(op.class_id, set()).add(op.token)
            for class_id, tokens in list(roster.items()):
                if class_id != op.class_id and op.token in tokens:
                    tokens.remove(op.token)
        elif op.kind == "delete":
            roster.setdefault(op.class_id, set()).discard(op.token)
    out = np.zeros_like(prev)
    for class_id, tokens in roster.items():
        for token in tokens:
            out[token] = class_id
    return out


def expand_dest_owned(
    scores: np.ndarray,
    k: np.ndarray,
    *,
    one_vote: bool,
) -> np.ndarray:
    """Destination keeps its own K row. No 450-gate tensor is an input."""

    if one_vote:
        gates = build_class_file(scores, preserve_mean=False).gate_tokens()
    else:
        gates = c7_multiplicity_weighted_gates(scores, preserve_mean=False)
    k = np.asarray(k, dtype=np.float64)
    if k.ndim == 1:
        k = k[:, None]
    return k * gates[:, None]


def motion_boundary_contract() -> dict:
    return {
        "keeps_rqtb_quotient": True,
        "destination_identity": "K[i] stays dest i; only the scalar is gate_c(class(i))",
        "old_h67_schedule": "ST_BUILD scored 450 + ST_CLASS *multiplicity + ST_ACTIVE recompute exp",
        "h86_schedule": "ST_CLASSIFY Class File + ST_SHIFTMAX + ST_EXPAND dest-owned + ST_PATCH members",
        "c7_vs_h82": "same directory, different partition (multiplicity vote vs one vote)",
        "not_motion_xor": True,
        "not_local5": True,
        "rtl_gate": "H82/H86 rank-1 + Class File stats; no production RTL before that",
    }


def demo(seed: int = 86) -> dict:
    rng = np.random.default_rng(seed)
    field = rng.normal(0.0, 0.35, size=(17, 17, 2)).clip(-2, 2)
    prev = window_codes(field, 0, 0)
    curr = window_codes(field, 0, 1)
    ops = member_patch(prev, curr)
    rebuilt = apply_patch(prev, tuple(op for op in ops if op.kind != "stay"))
    k = rng.normal(size=(TOKENS, 2))
    scores = (prev.astype(np.float64) - 256.0) / 128.0
    attn_h82 = expand_dest_owned(scores, k, one_vote=True)
    attn_c7 = expand_dest_owned(scores, k, one_vote=False)
    delta = scan_delta(prev, curr)
    return {
        "n_ops": len(ops),
        "n_insert": sum(op.kind == "insert" for op in ops),
        "n_delete": sum(op.kind == "delete" for op in ops),
        "n_stay": sum(op.kind == "stay" for op in ops),
        "patch_roundtrip": bool(np.array_equal(rebuilt, curr)),
        "h82_vs_c7_maxabs": float(np.max(np.abs(attn_h82 - attn_c7))),
        "scan_delta": delta,
        "contract": motion_boundary_contract(),
    }


def main() -> None:
    out = _ROOT / "results" / "h82_class_file_isa_20260817"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "h86_window_member_delta_v1", "demo": demo()}
    (out / "h86_window_member_delta.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
