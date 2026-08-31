#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1112r3 author static/self-consistent chain test; no launch, attempt or EDA."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import tempfile


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
ENGINE = HW / "dc_handoff/scripts/m1112r3_c2_async_observation_authorized_engine_source_r1.py"
ENGINE_SHA = "48616ebde16e07b132bbb2e686bd34a9f18270d0bc0693ab0ee956beb60f02be"
CONTRACT = HW / "contracts/m1112r3_c2_async_observation_shadow_source_contract_r1_20260830.json"
CONTRACT_ID = ("92117a56e50a946d674c82ce9fc084548b480df139e0a4e5a9b4aed391292bef",
    "cfe40a1d11bcdf77cd4ac33e149381b202c57cc8edc22cb1131559fba8e412fd",
    "ddda54a99c1638f39c828faf75775a7f5c0dae975ee26f7b251cbafa926906cf")
M1116 = HW / "reviews/m1116_m1112r2_c2_launch_chain_circularity_audit_r1_20260830"
M1116_OUTER = "3aacd467ac2a7d3fcd58d82e0977a53541b317abc1d88cf71e34ee0b95651f94"
M1114R2 = HW / "reviews/m1114r2_m1112r2_c2_async_observation_engine_hammer_r1_20260830"
M1114R2_OUTER = "15e1f136aa4d892a965a005f97a5845d81a144634f86c9db432bfdf4bec884a9"
M1109 = HW / "reviews/m1109_m1091r3_c2_observation_mapped_x_failure_audit_r1_20260830"
M1109_OUTER = "5c7a1f667c6c800f84a0e8219ddf58574412090812cda5d8bdaf36265f43d52d"
M1113 = HW / "reviews/m1113_m1112_c2_async_observation_engine_hammer_r1_20260830"
M1113_OUTER = "ee665be8def8c598669566467a6d1e59dc021a3b0743e2faf43122ed0991da64"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
OUT = HERE / "mechanical_checks.json"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str) -> None:
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and
            sha(path) == expected, "regular identity drift " + str(path))


def seal_flat(directory: Path) -> str:
    for name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
        path = directory / name
        if path.exists() or path.is_symlink():
            path.unlink()
    members = sorted(path for path in directory.rglob("*") if path.is_file())
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join(f"{sha(path)}  {path.relative_to(directory).as_posix()}\n"
        for path in members), encoding="utf-8")
    outer = directory / "SHA256SUMS.seal.sha256"
    outer.write_text(f"{sha(manifest)}  SHA256SUMS\n", encoding="utf-8")
    return sha(outer)


def seal_double(path: Path) -> str:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    side.write_text(f"{sha(path)}  {path.relative_to(HW).as_posix()}\n", encoding="utf-8")
    outer.write_text(f"{sha(side)}  {side.relative_to(HW).as_posix()}\n", encoding="utf-8")
    return sha(outer)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def rejected(function) -> bool:
    try:
        function()
    except Exception:
        return True
    return False


regular(ENGINE, ENGINE_SHA)
regular(CONTRACT, CONTRACT_ID[0])
regular(Path(str(CONTRACT) + ".sha256"), CONTRACT_ID[1])
regular(Path(str(CONTRACT) + ".sha256.seal.sha256"), CONTRACT_ID[2])
regular(DOCS359, DOCS359_SHA)

spec = importlib.util.spec_from_file_location("m1112r3_author_subject", ENGINE)
require(spec is not None and spec.loader is not None, "engine import spec")
engine = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = engine
spec.loader.exec_module(engine)

engine.verify_double(CONTRACT, CONTRACT_ID[0], CONTRACT_ID[2])
engine.verify_exact_flat(M1116, M1116_OUTER)
engine.verify_exact_flat(M1114R2, M1114R2_OUTER)
engine.verify_exact_flat(M1109, M1109_OUTER)
engine.verify_exact_flat(M1113, M1113_OUTER)
require(engine.load(M1116 / "review.json")["status"] ==
        "STOP_M1116_M1112R2_FUTURE_LAUNCH_HASH_CYCLE__ADDITIVE_R3_REQUIRED",
        "M1116 status")

source = ENGINE.read_text(encoding="utf-8")
ast.parse(source)
require('launch_outer = verify_flat_self_consistent(M1118R3)' in source and
        'result["m1118r3_outer_seal_file_sha256"] = launch_outer' in source and
        '"m1118r3_outer_seal_file_sha256" in receipt' in source and
        'verify_exact_flat(M1118R3, receipt["m1118r3_outer_seal_file_sha256"])' not in source,
        "acyclic future authority source shape")

good = "module mapped(input rst_core);\nINVD1BWP35P140 reset_inv (.I(rst_core), .ZN(rst_core_n));\n" + "".join(
    f"DFCNQD1BWP35P140 shadow_service_group_count_q_reg_{index}_ (.D(d{index}), .CP(clk_core), .CDN(rst_core_n), .Q(q{index}));\n"
    for index in range(337)) + "endmodule\n"
require(engine.structural_reset_gate_text(good)["shadow_register_bits"] == 337,
        "preserved reset provenance legal shape")
fake_reset_rejected = rejected(lambda: engine.structural_reset_gate_text(
    good.replace(".I(rst_core)", ".I(fake_reset)")))
require(fake_reset_rejected, "preserved reset provenance attack escaped")

with tempfile.TemporaryDirectory(prefix="m1112r3_acyclic_", dir=HERE) as raw:
    root = Path(raw)
    launcher = root / "launcher.py"
    launcher.write_text("# fixed launcher\n", encoding="utf-8")
    author_outer = "a" * 64
    engine_hammer = root / "engine_hammer"; engine_hammer.mkdir()
    write_json(engine_hammer / "review.json", {
        "status": "PASS_M1117R3_M1112R3_ENGINE_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY__NO_EDA",
        "identity": {"engine_sha256": ENGINE_SHA, "contract_sha256": CONTRACT_ID[0],
            "m1116_outer_seal_file_sha256": M1116_OUTER,
            "author_receipt_outer_seal_file_sha256": author_outer}})
    engine_hammer_outer = seal_flat(engine_hammer)

    launch_receipt = root / "launch_receipt.json"
    receipt_value = {
        "schema": "m1112r3_c2_authorized_launch_receipt_r1_v1",
        "status": "M1112R3_LAUNCH_SOURCE_FROZEN__M1118R3_REQUIRED__NO_EDA",
        "launcher_sha256": sha(launcher), "engine_sha256": ENGINE_SHA,
        "engine_contract_sha256": CONTRACT_ID[0],
        "engine_contract_outer_seal_file_sha256": CONTRACT_ID[2],
        "engine_author_receipt_outer_seal_file_sha256": author_outer,
        "m1116_outer_seal_file_sha256": M1116_OUTER,
        "m1117r3_outer_seal_file_sha256": engine_hammer_outer,
        "arguments": 0, "caller_selected_authority_allowed": False,
        "caller_environment_forwarded": False, "m1118r3_required": True,
        "launch_now": False, "attempt_now": False, "dc_now": False,
        "mapped_vcs_now": False, "maximum_attempts": 1,
        "automatic_retry": False, "paper_citable": False}
    write_json(launch_receipt, receipt_value)
    launch_receipt_outer = seal_double(launch_receipt)

    launch_hammer = root / "launch_hammer"; launch_hammer.mkdir()
    good_launch_review = {"status": "PASS_M1118R3_M1112R3_LAUNCH_HAMMER__GO_ONE_ATTEMPT",
        "identity": {"launch_receipt_outer_seal_file_sha256": launch_receipt_outer,
            "launcher_sha256": sha(launcher), "engine_sha256": ENGINE_SHA,
            "engine_contract_outer_seal_file_sha256": CONTRACT_ID[2],
            "engine_author_receipt_outer_seal_file_sha256": author_outer,
            "m1116_outer_seal_file_sha256": M1116_OUTER,
            "m1117r3_outer_seal_file_sha256": engine_hammer_outer}}
    write_json(launch_hammer / "review.json", good_launch_review)
    launch_hammer_outer = seal_flat(launch_hammer)

    old_paths = (engine.LAUNCHER, engine.LAUNCH_RECEIPT, engine.M1117R3, engine.M1118R3,
                 engine.verify_parent_launcher)
    engine.LAUNCHER, engine.LAUNCH_RECEIPT = launcher, launch_receipt
    engine.M1117R3, engine.M1118R3 = engine_hammer, launch_hammer
    engine.verify_parent_launcher = lambda receipt: engine.verify_regular(
        launcher, receipt["launcher_sha256"])
    try:
        accepted = engine.verify_future_authority()
        require(accepted["m1118r3_outer_seal_file_sha256"] == launch_hammer_outer,
                "self-consistent future outer discovery")

        forged_receipt = dict(receipt_value)
        forged_receipt["m1118r3_outer_seal_file_sha256"] = launch_hammer_outer
        write_json(launch_receipt, forged_receipt); seal_double(launch_receipt)
        future_outer_in_receipt_rejected = rejected(engine.verify_future_authority)
        write_json(launch_receipt, receipt_value); launch_receipt_outer = seal_double(launch_receipt)

        bad_review = json.loads(json.dumps(good_launch_review))
        bad_review["identity"]["launch_receipt_outer_seal_file_sha256"] = "0" * 64
        write_json(launch_hammer / "review.json", bad_review); seal_flat(launch_hammer)
        wrong_receipt_binding_rejected = rejected(engine.verify_future_authority)
        write_json(launch_hammer / "review.json", good_launch_review); seal_flat(launch_hammer)

        (launch_hammer / "EXTRA").write_text("forged\n", encoding="utf-8")
        unlisted_future_extra_rejected = rejected(engine.verify_future_authority)
        (launch_hammer / "EXTRA").unlink()
        seal_flat(launch_hammer)

        manifest = launch_hammer / "SHA256SUMS"
        real_manifest = root / "real_manifest"
        real_manifest.write_bytes(manifest.read_bytes()); manifest.unlink(); manifest.symlink_to(real_manifest)
        future_manifest_symlink_rejected = rejected(engine.verify_future_authority)
        manifest.unlink(); real_manifest.rename(manifest)
    finally:
        (engine.LAUNCHER, engine.LAUNCH_RECEIPT, engine.M1117R3, engine.M1118R3,
         engine.verify_parent_launcher) = old_paths

    duplicate = root / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    duplicate_json_rejected = rejected(lambda: engine.load(duplicate))
    nonfinite = root / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}\n', encoding="utf-8")
    nonfinite_json_rejected = rejected(lambda: engine.load(nonfinite))

mutations = {"future_outer_in_receipt": future_outer_in_receipt_rejected,
    "wrong_receipt_binding": wrong_receipt_binding_rejected,
    "unlisted_future_extra": unlisted_future_extra_rejected,
    "future_manifest_symlink": future_manifest_symlink_rejected,
    "duplicate_json": duplicate_json_rejected, "nonfinite_json": nonfinite_json_rejected,
    "fake_reset": fake_reset_rejected}
require(all(mutations.values()), "bounded mutation escaped")

r3_attempt = HW / "results/.m1112r3_c2_async_observation_dc_mapped_vcs_attempt_consumed"
r3_result = HW / "results/m1112r3_c2_async_observation_dc_mapped_vcs_r1_20260830"
r3_launcher = HW / "dc_handoff/scripts/run_m1112r3_c2_async_observation_authorized_launch_r1.py"
r3_launch_receipt = HW / "contracts/m1112r3_c2_async_observation_authorized_launch_receipt_r1_20260830.json"
require(not any(path.exists() or path.is_symlink() for path in
    (r3_attempt, r3_result, r3_launcher, r3_launch_receipt)), "r3 future/canonical namespace drift")
require(sha(DOCS359) == DOCS359_SHA, "docs359 drift")

output = {"schema": "m1112r3_c2_launch_chain_author_static_mutation_checks_v1",
    "status": "PASS_M1112R3_ACYCLIC_LAUNCH_CHAIN_SOURCE_TEST__M1117R3_REQUIRED__NO_EDA",
    "score": 100,
    "identity": {"engine_sha256": ENGINE_SHA, "contract_sha256": CONTRACT_ID[0],
        "contract_sidecar_sha256": CONTRACT_ID[1],
        "contract_outer_seal_file_sha256": CONTRACT_ID[2],
        "m1116_outer_seal_file_sha256": M1116_OUTER,
        "m1114r2_outer_seal_file_sha256": M1114R2_OUTER,
        "m1109_outer_seal_file_sha256": M1109_OUTER,
        "m1113_outer_seal_file_sha256": M1113_OUTER,
        "docs359_sha256": DOCS359_SHA},
    "acyclic_self_test": {"valid_chain_accepted": True,
        "launch_receipt_contains_future_hammer_outer": False,
        "future_hammer_outer_discovered": launch_hammer_outer,
        "future_hammer_review_binds_launch_receipt_outer": True,
        "mutations": mutations, "mutations_rejected": sum(mutations.values()),
        "mutations_total": len(mutations)},
    "preserved": {"reset_provenance_337": True, "fake_reset_rejected": True,
        "same_r2_wrapper_tb": True, "maximum_attempts": 1,
        "automatic_retry": False, "post_attempt_quarantine": True},
    "execution": {"engine_main": False, "static_gate": False, "launcher": False,
        "attempt": False, "result": False, "dc": False, "vcs": False,
        "temporary_chain_only": True},
    "authorization": {"different_author_m1117r3_engine_hammer": True,
        "launcher_authoring_now": False, "production_now": False}}
OUT.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
               encoding="utf-8")
print(output["status"])
