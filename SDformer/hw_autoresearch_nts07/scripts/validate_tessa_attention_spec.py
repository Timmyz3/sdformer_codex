#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "spec/tessa_attention_subsystem_spec.json"


def main() -> int:
    data = json.loads(SPEC.read_text(encoding="utf-8"))
    params = data["parameters"]
    interfaces = data["interfaces"]

    assert params["tokens_per_row"] == 2 * params["pairs_per_row"]
    assert params["active_entry_width"] == 16 + 32 + 8
    assert params["context_options"] == [1, 2, 4]
    assert params["first_rtl_contexts"] in params["context_options"]
    assert sum(interfaces["descriptor"]["fields"].values()) == interfaces["descriptor"]["width"]
    assert sum(interfaces["pair_input"]["payload_fields"].values()) == interfaces["pair_input"]["payload_width"]
    assert sum(interfaces["pair_result"]["slot_fields"].values()) == interfaces["pair_result"]["slot_width"]
    assert data["compile_modes"]["H67"]["score_class_depth"] == params["head_dim"] + 3
    assert data["compile_modes"]["H68"]["score_class_depth"] == 3
    assert data["context_memory"]["active_bank"]["depth"] >= params["tokens_per_row"]
    assert data["row_completion_invariants"]["semantic_tokens_committed"] == params["tokens_per_row"]
    assert data["commit_contract"]["pccc_bypass_required"] is True
    print(f"TESSA接口规格校验通过：{SPEC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
