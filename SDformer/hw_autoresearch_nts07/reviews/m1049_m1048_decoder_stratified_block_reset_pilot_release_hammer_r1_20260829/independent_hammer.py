#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Receipt-blind M1049 source/release/runner hammer.

This hammer is deliberately synthetic and read-only with respect to M699
payload members.  It does not execute M1050.  Payload access is replaced by a
tripwire so pre-attempt ordering can be tested without opening a bitpack.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
DRIVER = HW / "system_simulator/scripts/execute_m1048_decoder_stratified_block_reset_pilot_release.py"
RUNNER = HW / "system_simulator/scripts/run_m1050_m1048_decoder_stratified_block_reset_pilot_one_shot.sh"
CONTRACT = HW / "contracts/m1048_decoder_stratified_block_reset_pilot_release_contract_r1_20260829.json"
M699 = HW / "system_handoff/outgoing/m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828"
M705 = HW / "reviews/m705_m699_multisequence_decoder_payload_fresh_result_hammer_r1_20260828"
M1042 = HW / "reviews/m1042_m1041_decoder_stratified_block_reset_windows_source_r4_hammer_r1_20260829"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
ATTEMPT = HW / "results/.m1050_m1048_decoder_stratified_block_reset_pilot_attempt_consumed"
RESULT = HW / "results/m1050_m1048_decoder_stratified_block_reset_pilot_r1_20260829"

EXPECTED = {
    "driver": "3e2fa596e7cb0406feecc4124280643eaa093df80e9dcc7915fa9dcc7074267a",
    "runner": "681fdd6c1d51c37f3a7cb837bd49a0dd7af97f8b18730f314397cdabb29347a5",
    "contract": "ad21250e6789753b6408372a6cb4a3812d63844298d67ae9b05f98125e14bc9b",
    "m699_manifest": "e2d7c92a038c213b590603ff534a33f3579bf1224cc3f56c11629e1d4c813dc0",
    "m699_outer": "eaf975a9a1a4829b2c0a2251e7ef297abd53b83b30e23630e5ce51db5c5de18c",
    "m705_review": "6af48fb271254ef20f6baa1e435acfe51fdf38b457fe9782d6cac0b0e2883bd3",
    "m705_outer": "26781f5de30c6b6283c955144bbdac9c2b094aac3c19962b37016a57a6d24ff7",
    "m1042_review": "d0b26d9fa8cf4e272657835ac48be8b20ffbf577ddaf7bbd900a17d506138e88",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(value, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict(path: Path):
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


def verify_outer(directory: Path, expected_outer: str) -> None:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(sha(outer) == expected_outer, "outer identity drift: " + directory.name)
    require(outer.read_text(encoding="utf-8").split() ==
            [sha(manifest), "SHA256SUMS"], "outer binding drift")


def load_driver():
    spec = importlib.util.spec_from_file_location("m1049_m1048_under_hammer", DRIVER)
    require(spec is not None and spec.loader is not None, "cannot load M1048")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def runner_call(env):
    return subprocess.run(["/bin/bash", "-p", str(RUNNER)], cwd=HW.parent, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=30, check=False)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def main():
    before_doc = sha(DOC359)
    require(not ATTEMPT.exists() and not ATTEMPT.is_symlink(),
            "canonical M1050 attempt already occupied")
    require(not RESULT.exists() and not RESULT.is_symlink(),
            "canonical M1050 result already occupied")
    for path, key in ((DRIVER, "driver"), (RUNNER, "runner"),
                      (CONTRACT, "contract"), (DOC359, "docs359")):
        require(path.is_file() and not path.is_symlink(), "bad source: " + key)
        require(sha(path) == EXPECTED[key], "identity drift: " + key)
    require(sha(M699 / "manifest.json") == EXPECTED["m699_manifest"],
            "M699 manifest drift")
    verify_outer(M699, EXPECTED["m699_outer"])
    require(sha(M705 / "review.json") == EXPECTED["m705_review"],
            "M705 review drift")
    verify_outer(M705, EXPECTED["m705_outer"])
    require(sha(M1042 / "review.json") == EXPECTED["m1042_review"],
            "M1042 review drift")

    contract = strict(CONTRACT)
    m705 = strict(M705 / "review.json")
    manifest = strict(M699 / "manifest.json")
    require(m705["status"] ==
            "GO_M699_PAYLOAD_DENSITY_AND_OBSERVED_S3_STABILITY_ONLY__P0_0_P1_0",
            "M705 authority status drift")
    rows = [row for row in manifest["records"]
            if row["sequence"] == "interlaken_01_a" and
            row["sequence_sample_id"] == 0]
    routes = {int(row["module_index"]): row["route"] for row in rows}
    require(routes == {0: "EXACT_BINARY_BITPACK",
                       1: "EXACT_SCALED_BINARY_BITPACK",
                       2: "EXACT_BINARY_BITPACK",
                       3: "EXACT_BINARY_BITPACK"},
            "M699 selected route identity drift")
    require(contract["workload"]["layers"] == ["D0", "D2", "D3"] and
            contract["D1"]["status"] ==
            "DIAGNOSTIC_ONLY_NO_GENERATOR_NO_SCHEDULER_CALL" and
            contract["claim_boundary"]["d1_scheduled"] is False,
            "D0/D2/D3 exact or D1 diagnostic boundary drift")

    module = load_driver()
    require(module.self_test()["status"] ==
            "PASS_M1048_RELEASE_SMALL_SYNTHETIC_SELFTEST__NO_REAL_PAYLOAD",
            "synthetic self-test failed")

    # Independent transaction census: every synthetic compressed transaction
    # must appear in exactly one semantic block.
    pop, cfg = "M1049_SYNTHETIC", module.CONFIG
    def tx(identifier, kind, deps=(), produces=True):
        return module.CompressedTransaction(
            transaction_id=identifier, population_id=pop, config=cfg,
            kind=kind, base_address=1 << 60, address_stride_bytes=1,
            count=1, bank_pattern=(0,), width_bytes=1,
            dependency_tokens=tuple(deps),
            produces_token_prefix=(identifier + ":done" if produces else ""))
    source = tx(pop + ":A1_OSG:m0:t0:source_fetch", "external_read")
    rows_tx = [source]
    source_done = module.M890.terminal_token(source)
    for group in range(12):
        desc = tx(pop + f":A1_OSG:m0:t0:g{group}:osg_header", "external_read", (source_done,))
        lane = tx(pop + f":A1_OSG:m0:t0:g{group}:k1_descriptor0", "external_read", (source_done,))
        weight = tx(pop + f":A1_OSG:m0:t0:g{group}:k1_weight0", "weight_read",
                    (module.M890.terminal_token(desc), module.M890.terminal_token(lane), "external-ready"))
        read = tx(pop + f":A1_OSG:m0:t0:g{group}:psum_read", "psum_read", ("external-psum",))
        compute = tx(pop + f":A1_OSG:m0:t0:g{group}:compute", "compute",
                     (module.M890.terminal_token(weight), module.M890.terminal_token(read)))
        write = tx(pop + f":A1_OSG:m0:t0:g{group}:psum_write", "psum_write",
                   (module.M890.terminal_token(compute),))
        rows_tx.extend((desc, lane, weight, read, compute, write))
    for commit in range(12):
        rows_tx.append(tx(pop + f":A1_OSG:m0:t0:commit{commit}", "commit", ("external-final",)))
    blocks = list(module.iter_semantic_blocks(iter(rows_tx), "D0"))
    assigned = [item.transaction_id for _meta, body in blocks for item in body]
    original = [item.transaction_id for item in rows_tx]
    require(len(original) == len(set(original)) == len(assigned) and
            sorted(original) == sorted(assigned),
            "synthetic compressed transaction was dropped or multiply assigned")
    selected = module.select_streaming(iter(blocks), "D0")
    require(selected["selection_frozen_before_cycle_replay"] is True and
            selected["generated_compressed_transactions"] ==
            selected["assigned_compressed_transactions"] == len(rows_tx),
            "selection/conservation drift")
    replay_text = inspect.getsource(module.replay_layer)
    require(replay_text.index("selection = select_streaming") <
            replay_text.index("M1041.paired_replay"),
            "cycle replay occurs before selection freeze")

    # Runner must be inert for absent pins and a wrong contract pin.  Both are
    # guaranteed to fail before source validation can reach M699.
    base_env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    missing = runner_call(base_env)
    require(missing.returncode != 0, "runner accepted absent caller pins")
    wrong_env = dict(base_env)
    wrong_env.update({
        "M1050_EXPECTED_CONTRACT_SHA": "0" * 64,
        "M1050_EXPECTED_M1049_REVIEW_SHA": "1" * 64,
        "M1050_EXPECTED_M1049_MANIFEST_SHA": "2" * 64,
        "M1050_EXPECTED_M1049_OUTER_SHA": "3" * 64,
    })
    wrong = runner_call(wrong_env)
    require(wrong.returncode != 0, "runner accepted wrong caller pins")
    require(not ATTEMPT.exists() and not RESULT.exists(),
            "inert/wrong-pin attack consumed real namespace")

    direct = subprocess.run([
        "/opt/anaconda3/envs/pytorch310/bin/python3.10", str(DRIVER),
        "--run-pilot", "--work", str(HW / "results/.m1050_m1048_decoder_stratified_block_reset_pilot_r1_20260829.work.attack"),
        "--attempt", str(ATTEMPT)], cwd=HW.parent, env=base_env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=30, check=False)
    require(direct.returncode != 0 and not ATTEMPT.exists() and not RESULT.exists(),
            "direct run-pilot bypass was accepted")

    for path, role in ((Path("relative"), "work"),
                       (HW / "results/wrong", "result"),
                       (HW / "results/wrong", "attempt")):
        try:
            module.safe_runtime_path(path, role)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("runtime namespace attack accepted: " + role)

    # P0-1: prove source validation reaches the full M699 sealed-directory
    # verifier before an attempt can exist.  The tripwire prevents any member
    # from actually being opened by this audit.
    payload_tripwire = {"calls": 0}
    original_verify = module.M785.verify_sealed_directory
    original_validate_m785 = module.M785.validate_source_contract
    def tripwire(path):
        payload_tripwire["calls"] += 1
        require(Path(path).resolve() == M699.resolve(), "unexpected payload path")
        raise RuntimeError("M1049_PAYLOAD_OPEN_TRIPWIRE_BEFORE_ATTEMPT")
    module.M785.verify_sealed_directory = tripwire
    module.M785.validate_source_contract = lambda _repo, _contract: {
        "status": "M1049_SYNTHETIC_PREDECESSOR_BYPASS_ONLY"}
    try:
        module.validate_source(CONTRACT, RUNNER)
    except RuntimeError as error:
        require(str(error) == "M1049_PAYLOAD_OPEN_TRIPWIRE_BEFORE_ATTEMPT",
                "payload tripwire rejected for wrong reason")
    else:
        raise RuntimeError("payload tripwire not reached")
    finally:
        module.M785.verify_sealed_directory = original_verify
        module.M785.validate_source_contract = original_validate_m785
    require(payload_tripwire["calls"] == 1 and not ATTEMPT.exists(),
            "pre-attempt payload-open finding not reproduced")

    # P0-2: same-UID post-run tampering can inject candidate means/speedup into
    # raw_windows.json, update its SHA in result.json, and pass assemble().
    # This uses a synthetic temporary RESULTS root and no cycle replay.
    with tempfile.TemporaryDirectory(prefix="m1049_assemble_attack.") as td:
        root = Path(td)
        prior_results = module.RESULTS
        module.RESULTS = root
        try:
            work = root / ("." + module.RESULT_NAME + ".work.attack")
            work.mkdir()
            raw = {
                "schema": module.RAW_SCHEMA,
                "exact_mismatch_count": 0,
                "layers": [{"coverage": {"candidate_mean_cycles": 1.0},
                            "ci_publication_envelope": {"point_speedup": 999.0}}],
                "claim_boundary": {"paper_citable": False},
            }
            write_json(work / "raw_windows.json", raw)
            result_value = {
                "schema": module.RESULT_SCHEMA,
                "status": "ATTACK_FAKE_STATUS",
                "raw_windows_sha256": sha(work / "raw_windows.json"),
                "exact_mismatch_count": 0,
                "d1_scheduled": False,
                "paper_citable": False,
                "decoder_complete": False,
                "table_a_row": False,
                "system_speedup": False,
                "local_speedup": False,
                "continuous_row_cycles": False,
                "eda_gpu_remote_used": False,
            }
            write_json(work / "result.json", result_value)
            (work / "RUN_COMPLETE.txt").write_text("ATTACK\n", encoding="utf-8")
            assembled = module.assemble(work)
            require(assembled["status"] == "PASS_M1050_WORK_SEALED",
                    "semantic injection was unexpectedly rejected")
        finally:
            module.RESULTS = prior_results

    # P1: contract_value binds selected workload but accepts an unbound D1
    # semantic rewrite and arbitrary nested prefills when a caller follows the
    # modified contract SHA.  No payload is opened for this direct parser test.
    mutated = copy.deepcopy(contract)
    mutated["D1"]["status"] = "ATTACK_FULLY_SCHEDULED"
    mutated["D1"]["candidate_mean_cycles"] = 1.0
    mutated["semantic_attack"] = {"point_speedup": 999.0}
    with tempfile.TemporaryDirectory(prefix="m1049_contract_attack.") as td:
        altered = Path(td) / "contract.json"
        write_json(altered, mutated)
        accepted_contract = module.contract_value(altered)
        require(accepted_contract["D1"]["status"] == "ATTACK_FULLY_SCHEDULED",
                "unbound contract semantic attack unexpectedly rejected")

    runner_text = RUNNER.read_text(encoding="utf-8")
    ordering = [
        runner_text.index("m1050_py --validate-source"),
        runner_text.index("m1050_auth --validate-authority"),
        runner_text.index("m1050_flock}\" -n 9"),
        runner_text.index("for m1050_process in dc_shell vcs simv fm_shell pt_shell"),
        runner_text.index("m1050_mem"),
        runner_text.index("m1050_auth --consume-attempt"),
        runner_text.index("/usr/bin/mkdir -m 700"),
        runner_text.index("--run-pilot"),
    ]
    require(ordering == sorted(ordering), "runner gate ordering drift")
    require("16777216" in runner_text and
            "/tmp/m1050_decoder_stratified_block_reset_pilot.lock" in runner_text,
            "resource/lock gate source drift")

    require(not ATTEMPT.exists() and not RESULT.exists(),
            "audit caused canonical payload/cycle execution")
    require(sha(DOC359) == before_doc == EXPECTED["docs359"], "docs359 drift")
    output = {
        "schema": "m1049_m1048_decoder_stratified_pilot_release_hammer_mechanical_v1",
        "status": "FAIL_M1049_M1048_PREATTEMPT_PAYLOAD_OPEN_AND_ASSEMBLE_SEMANTIC_INJECTION__STOP_M1050",
        "score_out_of_100": 41,
        "severity": {"p0": 2, "p1": 1, "p2": 0},
        "identity": {key: EXPECTED[key] for key in EXPECTED},
        "positive": {
            "m699_m705_m1042_identities": True,
            "d0_d2_d3_exact_d1_diagnostic_current_contract": True,
            "synthetic_transaction_conservation": f"{len(rows_tx)}/{len(rows_tx)}",
            "selection_before_replay": True,
            "missing_pin_rejected": True,
            "wrong_pin_rejected": True,
            "direct_run_bypass_rejected": True,
            "namespace_attacks_rejected": 3,
            "runner_lock_eda_resource_attempt_work_order_static": True,
        },
        "blocking_attacks": {
            "pre_attempt_payload_open_tripwire_calls": payload_tripwire["calls"],
            "assemble_semantic_injection_accepted": True,
            "unbound_contract_d1_and_semantic_fields_accepted": True,
        },
        "scope": {
            "real_payload_members_opened_by_hammer": 0,
            "real_cycle_replays": 0,
            "canonical_attempts_consumed": 0,
            "m1050_executed": False,
            "eda_gpu_remote_used": False,
        },
        "docs359_sha256": before_doc,
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    raise SystemExit(main())
