#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "scripts/build_m1228_motion_cross_run_final_checkpoint_rebind_binder_source.py"
TEST = HW / "tests/test_build_m1228_motion_cross_run_final_checkpoint_rebind_binder_source.py"
CONTRACT = HW / "contracts/m1228_motion_cross_run_final_checkpoint_rebind_binder_source_contract_r1_20260830.json"
CONTRACT_SUM = CONTRACT.with_name(CONTRACT.name + ".sha256")
CONTRACT_SEAL = CONTRACT.with_name(CONTRACT.name + ".sha256.seal.sha256")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    SOURCE: "9b2b43b4d36ed64741cbb39db0d9f5d75eb7bec09b00f4e496f3d52ce3ae5efe",
    TEST: "972bd46f3d5b4046f4c639fa234c31fd0615c95ffd6512dc729e1645ff30538e",
    CONTRACT: "ea94c4832dfe235a0fe3ab5c6a034ac9c98dff0611f2b753ec97bcae682389df",
    CONTRACT_SUM: "1d76ebcc17414a2f4fd45348c650057deb45e964c9ed55b70ca82a9fb6749e3a",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


for path, digest in EXPECTED.items():
    assert path.is_file(), path
    assert sha(path) == digest, (path, sha(path))

assert CONTRACT_SUM.read_text().strip() == EXPECTED[CONTRACT] + "  " + CONTRACT.name
assert CONTRACT_SEAL.read_text().strip() == EXPECTED[CONTRACT_SUM] + "  " + CONTRACT_SUM.name

contract = json.loads(CONTRACT.read_text())
assert contract["scope"]["source_only"] is True
assert contract["scope"]["production_builder_executed"] is False
assert contract["scope"]["remote_read_or_write"] is False
assert contract["scope"]["selection_made_now"] is False
assert [row["epoch"] for row in contract["candidate_policy"]] == [29, 30, 32, 34]
assert contract["new_run_manifest"]["required_evaluation_epochs"] == [30, 32, 34]

spec = importlib.util.spec_from_file_location("m1228_author_static_import", SOURCE)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
policy = module.PRODUCTION_POLICY
assert [candidate.epoch for candidate in policy.candidates] == [29, 30, 32, 34]
assert len({candidate.run_dir for candidate in policy.candidates}) == 2
assert len({candidate.config for candidate in policy.candidates}) == 2
assert policy.candidates[0].expected_checkpoint_sha256 == (
    "2144dfd628cd928bfb768b92d4fa097b720db112c32d930b9f3cd85c6217286a"
)
assert policy.candidates[0].config_sha256.startswith("c7b5b994")
assert all(candidate.config_sha256.startswith("630e735c") for candidate in policy.candidates[1:])
assert module.NEW_EVALUATION_EPOCHS == (30, 32, 34)
assert [row["id"] for row in module.activation_rebind_targets()] == [f"E{i}" for i in range(9)]

print(
    "M1228_AUTHOR_MECHANICAL_PASS source_only=true production_run=false "
    "remote=false checkpoint=false valid825=false gpu=false eda=false candidates=4 e0e8=9"
)
