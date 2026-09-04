#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1111D author-stage static/self-test; never calls runner main or production."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HW = Path(__file__).resolve().parents[2]
RUNNER = HW / "system_simulator/scripts/run_m1111d_m1105dr2_decoder_only_production_zero_arg.py"
CONTRACT = HW / "contracts/m1111d_m1105dr2_decoder_only_production_runner_source_contract_r1_20260830.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUT = Path(__file__).with_name("mechanical_checks.json")
EXPECTED_RUNNER = "52407204479fa320f28f43bf7425abcf45acc7f126dfe83d076e7d9a8fe15f7a"
EXPECTED_CONTRACT = "82bba9ed495f8b1d316ea02647e9f28868c4845da69f95723132f7921a8535f6"
EXPECTED_CONTRACT_SIDECAR = "71f636ba43fc27321dda569e1133770cd77c9065974a2e974cd81b9b950df90e"
EXPECTED_CONTRACT_OUTER = "7a94c5ce60291f76e8a460886486271868d628e523d96c8c63080dea6409b4dc"
EXPECTED_DOCS359 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(module) -> dict:
    return {
        "attempt": module.ATTEMPT.exists() or module.ATTEMPT.is_symlink(),
        "result": module.RESULT.exists() or module.RESULT.is_symlink(),
        "lock": module.LOCK.exists() or module.LOCK.is_symlink(),
        "work": sorted(path.name for path in module.RESULT.parent.glob(module.WORK_PREFIX + "*")),
        "quarantine": sorted(path.name for path in module.RESULT.parent.glob(module.FAILURE_PREFIX + "*")),
    }


def main() -> None:
    require(sha(RUNNER) == EXPECTED_RUNNER and sha(CONTRACT) == EXPECTED_CONTRACT and
            sha(Path(str(CONTRACT) + ".sha256")) == EXPECTED_CONTRACT_SIDECAR and
            sha(Path(str(CONTRACT) + ".sha256.seal.sha256")) == EXPECTED_CONTRACT_OUTER and
            sha(DOCS359) == EXPECTED_DOCS359, "author identity drift")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    require(contract["status"] ==
            "SOURCE_ONLY__DIFFERENT_AUTHOR_FINAL_RUNNER_HAMMER_REQUIRED__NO_PRODUCTION" and
            contract["production_scope"]["calls"] == 120 and
            contract["production_scope"]["m700_external_input_allowed"] is False and
            contract["production_scope"]["final_checkpoint_rebind_required"] is True and
            contract["production_scope"]["d1"]["theta_word_uint32"] == 1065353139 and
            contract["production_scope"]["d1"]["weight_folding_allowed"] is False and
            contract["claim_boundary"]["system_speedup_admitted"] is False and
            contract["claim_boundary"]["paper_ppa_ready"] is False,
            "contract semantic drift")
    text = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(text)
    main_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and
                     node.name == "main")
    calls = [(node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id,
              node.lineno) for node in ast.walk(main_node) if isinstance(node, ast.Call) and
             isinstance(node.func, (ast.Attribute, ast.Name))]
    lines = {name: sorted(line for called, line in calls if called == name) for name in
             ("validate_authorities", "sanitize_environment", "resource_gate",
              "acquire_lock", "consume_attempt", "execute_production",
              "publish_result", "quarantine_work", "release_lock")}
    require(all(len(lines[name]) == 1 for name in (
        "validate_authorities", "sanitize_environment", "resource_gate",
        "acquire_lock", "consume_attempt", "execute_production", "publish_result",
        "quarantine_work", "release_lock")), "main call multiplicity drift")
    require(lines["validate_authorities"][0] < lines["sanitize_environment"][0] <
            lines["resource_gate"][0] < lines["acquire_lock"][0] <
            lines["consume_attempt"][0] < lines["execute_production"][0] <
            lines["publish_result"][0], "main order drift")
    require("M700" not in text and "m700_h67" not in text and
            '"system_speedup_admitted": False' in text and
            '"paper_ppa_ready": False' in text and
            '"final_checkpoint_rebind_required": True' in text and
            'scope["d1"]["weight_folding_allowed"] is False' in text,
            "runner forbidden-input/boundary literal drift")
    # The lower-case m700 references are boolean rejection fields inherited
    # from M1105Dr2, never a path or loaded module.
    require("m700_external_input_allowed" in text and
            "canonical[\"external_baseline_rejection\"][\"m700_admitted\"] is False" in text,
            "M700 rejection gate absent")
    spec = importlib.util.spec_from_file_location("m1111d_author_runner", RUNNER)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    before = snapshot(module)
    require(before == {"attempt": False, "result": False, "lock": False,
                       "work": [], "quarantine": []}, "namespace not fresh")
    self_test = module.source_static_self_test()
    after = snapshot(module)
    require(after == before and self_test["status"] ==
            "PASS_M1111D_RUNNER_SOURCE_STATIC_SELF_TEST__NO_PRODUCTION" and
            self_test["synthetic_transactions"] == 6 and
            self_test["d1_theta_exact"] is True and
            self_test["m700_external_input"] is False and
            self_test["attempt_created"] is False and
            self_test["canonical_payload_opened"] is False and
            self_test["production_replay_executed"] is False,
            "runner static self-test drift")
    result = {
        "schema": "m1111d_decoder_runner_author_static_checks_v1",
        "status": "PASS_M1111D_AUTHOR_STATIC_SELF_TEST__NO_PRODUCTION",
        "identity": {"runner_sha256": EXPECTED_RUNNER,
            "contract_sha256": EXPECTED_CONTRACT,
            "contract_sidecar_sha256": EXPECTED_CONTRACT_SIDECAR,
            "contract_outer_seal_file_sha256": EXPECTED_CONTRACT_OUTER,
            "m1110d_outer_seal_file_sha256":
                "9caf64e422b4cb696a600b69415bd8265dc4694066fae7ec67a5f34019f39e23",
            "docs359_sha256": EXPECTED_DOCS359},
        "static_main_call_lines": lines,
        "self_test": self_test,
        "output_contract": {"calls": 120, "six_kinds": True,
            "address_timed_per_call_kind_compressed": True,
            "diagnostic_cycles_traffic_only": True,
            "system_speedup_admitted": False, "paper_ppa_ready": False,
            "m700_external_input": False,
            "final_checkpoint_rebind_required": True,
            "d1_theta_exact_no_weight_fold": True},
        "execution": {"runner_main_called": False, "attempt_created": False,
            "result_created": False, "canonical_payload_opened": False,
            "production_replay_executed": False}
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")


if __name__ == "__main__":
    main()
