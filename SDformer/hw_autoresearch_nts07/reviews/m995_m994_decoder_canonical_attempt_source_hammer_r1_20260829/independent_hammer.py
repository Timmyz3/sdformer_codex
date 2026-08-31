#!/usr/bin/env python3
"""Independent static/temp-only M995 hammer for M994; never runs D2/D3."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
DRIVER = HW / "system_simulator/scripts/execute_m994_m982_decoder_canonical_attempt_source_r1.py"
RUNNER = HW / "system_simulator/scripts/run_m998_m994_decoder_canonical_attempt_one_shot.sh"
CONTRACT = HW / "contracts/m994_m982_decoder_canonical_attempt_source_contract_r1_20260829.json"
SOURCE_RECEIPT = HW / "reviews/m994_m982_decoder_canonical_attempt_source_receipt_r1_20260829"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path):
    def pairs(values):
        out = {}
        for key, value in values:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs)


def verify_flat_seal(directory):
    directory = Path(directory)
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and outer.is_file(), "flat seal absent")
    require(outer.read_text(encoding="utf-8") ==
            sha(manifest) + "  SHA256SUMS\n", "outer seal mismatch")
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        require(name not in listed, "duplicate manifest member")
        member = directory / name
        require(member.is_file() and not member.is_symlink() and
                sha(member) == digest, "member drift: " + name)
        listed[name] = digest
    actual = {item.name for item in directory.iterdir()
              if item.is_file() and item.name not in
              ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(set(listed) == actual, "flat manifest coverage drift")
    return {"manifest_sha256": sha(manifest),
            "outer_seal_file_sha256": sha(outer),
            "member_count": len(actual)}


def load_driver():
    spec = importlib.util.spec_from_file_location("m995_independent_m994", DRIVER)
    require(spec is not None and spec.loader is not None, "cannot load M994")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fault_attack(module, fault):
    authority = {"release_sha256": "a" * 64,
                 "release_hammer_review_sha256": "b" * 64}
    with tempfile.TemporaryDirectory(prefix="m995_attack_") as tmp:
        parent = Path(tmp)
        attempt = parent / module.ATTEMPT.name
        result = parent / module.RESULT.name
        try:
            module.consume_attempt(authority, attempt, result, parent, fault)
        except RuntimeError as error:
            require("injected" in str(error), fault + " wrong exception")
        else:
            raise RuntimeError(fault + " did not interrupt")
        require(attempt.is_dir() and not attempt.is_symlink(),
                fault + " lost canonical attempt")
        require(not result.exists() and not result.is_symlink(),
                fault + " created result")
        if fault == "after_canonical_mkdir":
            require(not (attempt / "attempt.json").exists() and
                    not (attempt / module.B.SEAL_DIR).exists(),
                    "mkdir fault unexpectedly advanced")
        elif fault == "after_attempt_receipt":
            require((attempt / "attempt.json").is_file() and
                    not (attempt / module.B.SEAL_DIR).exists(),
                    "receipt fault state drift")
        else:
            module.validate_attempt(authority, attempt)
        try:
            module.consume_attempt(authority, attempt, result, parent)
        except RuntimeError as error:
            require("already consumed" in str(error), fault + " retry reason drift")
        else:
            raise RuntimeError(fault + " allowed retry")
        require(not list(parent.glob(module.ATTEMPT.name + ".stage.*")),
                fault + " created random attempt stage")
        return {"fault": fault, "canonical_attempt_present": True,
                "retry_blocked": True, "result_absent": True,
                "random_attempt_stage_absent": True}


def main():
    module = load_driver()
    contract = strict_json(CONTRACT)
    source_seal = verify_flat_seal(SOURCE_RECEIPT)
    source_validation = module.validate_source_contract(CONTRACT, RUNNER)
    expected_chain = {
        "source": "M994", "independent_source_hammer": "M995",
        "one_attempt_release": "M996",
        "independent_release_hammer": "M997", "sole_execution": "M998",
        "automatic_retry": False,
    }
    require(contract["future_chain"] == expected_chain, "M994-M998 chain drift")
    require(contract["frozen_execution"]["order"] == ["D2", "D3"],
            "row order drift")
    rows = contract["frozen_execution"]["rows"]
    require(rows == [
        {"layer": "D2", "sample_id": 0, "config": "A1_OSG",
         "timestep": 0, "expanded_prefix": 10000},
        {"layer": "D3", "sample_id": 0, "config": "A1_OSG",
         "timestep": 0, "expanded_prefix": 10000}], "row identity drift")

    runner = RUNNER.read_text(encoding="utf-8")
    consume_pos = runner.index("m998_auth --consume-attempt")
    work_pos = runner.index('/usr/bin/mkdir -m 700 "${m998_work}"')
    loop_pos = runner.index("for m998_layer in D2 D3;do")
    assemble_pos = runner.index("m998_auth --assemble")
    require(consume_pos < work_pos < loop_pos < assemble_pos,
            "runner phase order drift")
    base = module.BASE_PATH.read_text(encoding="utf-8")
    run_row_start = base.index("def run_row(")
    quarantine_start = base.index("def quarantine_work(")
    run_row_body = base[run_row_start:quarantine_start]
    require('return {"payload":payload,"seal":atomic_seal(stage)}' in run_row_body,
            "row can return before atomic seal")
    require("set -euo pipefail" in runner and
            'for m998_layer in D2 D3;do' in runner,
            "D2-before-D3 fail-closed loop drift")

    identities = contract["source_identity"]
    for key in ("frozen_m946", "frozen_m896"):
        item = identities[key]
        path = HW / item["path"]
        require(sha(path) == item["sha256"], key + " drift")
    driver = DRIVER.read_text(encoding="utf-8")
    require(all(token in driver for token in
                ('("retry", "d2_or_d3_100k", "full_row", "production",',
                 '"eda_gpu_remote"')),
            "authorization prohibition drift")
    require(contract["claim_boundary"]["real_10k_executed"] is False and
            contract["claim_boundary"]["eda_gpu_remote_used"] is False and
            contract["claim_boundary"]["m998_execution_authorized"] is False,
            "source claim boundary expanded")

    result_parent = module.RESULT.parent
    before = sorted(item.name for item in result_parent.iterdir()
                    if item.name == module.ATTEMPT.name or
                    item.name == module.RESULT.name or
                    item.name.startswith(module.FAILURE_PREFIX) or
                    item.name.startswith(module.RESULT.name + ".work."))
    require(not before, "M998 result namespace not fresh")
    attacks = [fault_attack(module, fault) for fault in
               ("after_canonical_mkdir", "after_attempt_receipt",
                "after_attempt_seal")]
    after = sorted(item.name for item in result_parent.iterdir()
                   if item.name == module.ATTEMPT.name or
                   item.name == module.RESULT.name or
                   item.name.startswith(module.FAILURE_PREFIX) or
                   item.name.startswith(module.RESULT.name + ".work."))
    require(after == before, "temporary attacks mutated production results")

    return {
        "schema": "m995_m994_independent_source_hammer_v1",
        "status": "PASS_M995_M994_CANONICAL_ATTEMPT_SOURCE_HAMMER",
        "verdict": "GO_AUTHOR_M996_RELEASE_ONLY",
        "contract_sha256": sha(CONTRACT),
        "driver_sha256": sha(DRIVER),
        "runner_sha256": sha(RUNNER),
        "source_receipt_seal": source_seal,
        "source_validation": source_validation,
        "chain": expected_chain,
        "fault_attacks": attacks,
        "d2_atomic_seal_before_d3": True,
        "m946_sha256": identities["frozen_m946"]["sha256"],
        "m896_sha256": identities["frozen_m896"]["sha256"],
        "results_fresh_before_after": True,
        "real_10k_executed": False,
        "full_row_or_100k_executed": False,
        "eda_gpu_remote_used": False,
        "docs359_sha256": sha(DOCS359),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
