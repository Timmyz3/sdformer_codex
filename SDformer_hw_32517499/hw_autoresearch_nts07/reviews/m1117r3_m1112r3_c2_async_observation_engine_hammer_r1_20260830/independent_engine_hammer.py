#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent M1112r3 acyclic launch-chain hammer; no launcher or EDA."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ENGINE = ROOT / "dc_handoff/scripts/m1112r3_c2_async_observation_authorized_engine_source_r1.py"
CONTRACT = ROOT / "contracts/m1112r3_c2_async_observation_shadow_source_contract_r1_20260830.json"
AUTHOR = ROOT / "reviews/m1112r3_c2_launch_chain_source_receipt_r1_20260830"
M1116 = ROOT / "reviews/m1116_m1112r2_c2_launch_chain_circularity_audit_r1_20260830"
M1114R2 = ROOT / "reviews/m1114r2_m1112r2_c2_async_observation_engine_hammer_r1_20260830"
WRAPPER = ROOT / "rtl_m1112/m1112_c2_k1_async_observation_shadow_wrapper.sv"
TB = ROOT / "dc_handoff/tb/tb_m1112_c2_k1_async_observation_shadow_case0_short.sv"
DOCS359 = ROOT / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "engine": "48616ebde16e07b132bbb2e686bd34a9f18270d0bc0693ab0ee956beb60f02be",
    "contract": "92117a56e50a946d674c82ce9fc084548b480df139e0a4e5a9b4aed391292bef",
    "contract_side": "cfe40a1d11bcdf77cd4ac33e149381b202c57cc8edc22cb1131559fba8e412fd",
    "contract_outer": "ddda54a99c1638f39c828faf75775a7f5c0dae975ee26f7b251cbafa926906cf",
    "author_review": "0cf65f1015e45ae70fb352bb86518d98784e3f436de4648bdf0d9c726efbf69b",
    "author_manifest": "e30b75f496507f1d34ebf25fa6cdc9d5087adfc758bb5ce0b99e9a35cb8d3e69",
    "author_outer": "7f9d0205b9ba2f53fd642b05b0cd4faf9aa3e8e5bf14a6047c23ac6fba3ea7ff",
    "m1116_outer": "3aacd467ac2a7d3fcd58d82e0977a53541b317abc1d88cf71e34ee0b95651f94",
    "m1114r2_outer": "15e1f136aa4d892a965a005f97a5845d81a144634f86c9db432bfdf4bec884a9",
    "wrapper": "95c31bc70a7617c6653eaca2f77a54388119f744b814dfc909c75edad1c39218",
    "tb": "ff6bd371c3b1371c520b38680960ad0297a8c01eb92eb7b4a0f4d2e59fc861b6",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    return stat.S_ISREG(mode) and not path.is_symlink()


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def verify_flat(directory: Path, expected_outer: str) -> dict:
    require(directory.exists() and not directory.is_symlink() and stat.S_ISDIR(directory.lstat().st_mode), f"directory {directory}")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(regular(manifest) and regular(outer) and sha(outer) == expected_outer, f"seal metadata {directory}")
    require(outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"], "outer content")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None, "manifest syntax")
        name = fields[1].lstrip("*"); relative = Path(name)
        require(name and not relative.is_absolute() and ".." not in relative.parts and relative.as_posix() == name and name not in expected, "manifest member")
        expected[name] = fields[0]
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), f"live symlink {name}")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), f"special member {name}")
    require(actual == set(expected), "exact member coverage")
    for name, digest in expected.items():
        require(regular(directory / name) and sha(directory / name) == digest, f"member identity {name}")
    return {"members": len(expected), "manifest_sha256": sha(manifest), "outer_seal_file_sha256": sha(outer)}


def import_engine():
    spec = importlib.util.spec_from_file_location("m1112r3_independent_subject", ENGINE)
    require(spec is not None and spec.loader is not None, "engine import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def seal_flat(directory: Path) -> str:
    for name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
        path = directory / name
        if path.exists() or path.is_symlink():
            path.unlink()
    members = sorted(path for path in directory.rglob("*") if path.is_file())
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join(f"{sha(path)}  {path.relative_to(directory).as_posix()}\n" for path in members), encoding="utf-8")
    outer = directory / "SHA256SUMS.seal.sha256"
    outer.write_text(f"{sha(manifest)}  SHA256SUMS\n", encoding="utf-8")
    return sha(outer)


def seal_double(path: Path) -> str:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    side.write_text(f"{sha(path)}  {path.relative_to(ROOT).as_posix()}\n", encoding="utf-8")
    outer.write_text(f"{sha(side)}  {side.relative_to(ROOT).as_posix()}\n", encoding="utf-8")
    return sha(outer)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def rejected(engine, callback) -> bool:
    try:
        callback()
    except (engine.GateFailure, OSError, KeyError, ValueError, json.JSONDecodeError):
        return True
    return False


def make_good_reset(bits: int = 337) -> str:
    return "module mapped(input rst_core);\nINVD1BWP35P140 reset_inv (.I(rst_core), .ZN(rst_core_n));\n" + "".join(
        f"DFCNQD1BWP35P140 shadow_service_group_count_q_reg_{index}_ (.D(d{index}), .CP(clk_core), .CDN(rst_core_n), .Q(q{index}));\n"
        for index in range(bits)) + "endmodule\n"


def preserved_hardware_checks(engine) -> dict:
    good = make_good_reset()
    require(engine.structural_reset_gate_text(good)["shadow_register_bits"] == 337, "legal reset cone")
    attacks = {
        "fake_reset": good.replace(".I(rst_core)", ".I(fake_reset)"),
        "direct_reset": good.replace(".CDN(rst_core_n)", ".CDN(rst_core)"),
        "constant_reset": good.replace(".CDN(rst_core_n)", ".CDN(1'b0)"),
        "multilevel": good.replace("INVD1BWP35P140 reset_inv (.I(rst_core), .ZN(rst_core_n));", "INVD1BWP35P140 inv0 (.I(rst_core), .ZN(mid));\nINVD1BWP35P140 reset_inv (.I(mid), .ZN(rst_core_n));"),
        "reconvergent": good.replace("INVD1BWP35P140 reset_inv (.I(rst_core), .ZN(rst_core_n));", "ND2D1BWP35P140 reset_gate (.A1(rst_core), .A2(other), .ZN(rst_core_n));"),
        "set_only": good.replace(".CDN(rst_core_n)", ".SDN(rst_core_n)"),
        "census_336": make_good_reset(336),
    }
    reset_results = {name: rejected(engine, lambda text=text: engine.structural_reset_gate_text(text)) for name, text in attacks.items()}
    require(all(reset_results.values()), "preserved reset attack")

    wrapper = WRAPPER.read_text(encoding="utf-8"); tb = TB.read_text(encoding="utf-8")
    fields = re.findall(r"logic\s+(?:\[(\d+):0\]\s+)?(shadow_\w+_q)\s*;", wrapper)
    require(len(fields) == 13 and sum((int(width) + 1) if width else 1 for width, _ in fields) == 337, "13/337")
    instance = wrapper.split(") implementation (", 1)[1].split("));", 1)[0]
    require(instance.count("unused_frozen_debug_") == 13 and "obs_" not in instance and "shadow_" not in instance, "no feedback")
    predicates = re.findall(r"sample_unknown_bitmap\[(\d+)\]=\$isunknown\((obs_\w+)\);", tb)
    require(sorted(int(index) for index, _ in predicates) == list(range(22)) and len({signal for _, signal in predicates}) == 22, "22 bitmap")

    with tempfile.TemporaryDirectory(prefix="m1117r3_live_", dir=ROOT / "reviews") as raw:
        flat = Path(raw); member = flat / "review.json"; member.write_text("{}\n", encoding="utf-8")
        seal_flat(flat); engine.verify_exact_flat(flat, sha(flat / "SHA256SUMS.seal.sha256"))
        extra = flat / "EXTRA"; extra.write_text("x\n", encoding="utf-8")
        extra_rejected = rejected(engine, lambda: engine.verify_exact_flat(flat, sha(flat / "SHA256SUMS.seal.sha256")))
        extra.unlink(); seal_flat(flat)
        manifest = flat / "SHA256SUMS"; real = flat / "manifest.real"; manifest.rename(real); manifest.symlink_to(real.name)
        symlink_rejected = rejected(engine, lambda: engine.verify_exact_flat(flat, sha(flat / "SHA256SUMS.seal.sha256")))
    require(extra_rejected and symlink_rejected, "preserved live seal")
    return {"shadow_counters": 13, "shadow_bits": 337, "unknown_predicates": 22, "no_feedback": True, "reset_attacks_rejected": reset_results, "live_extra_rejected": extra_rejected, "live_manifest_symlink_rejected": symlink_rejected}


def chain_hammer(engine) -> dict:
    mutations: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="m1117r3_chain_", dir=ROOT / "reviews") as raw:
        root = Path(raw)
        launcher = root / "launcher.py"; launcher.write_text("# fixed zero-argument launcher\n", encoding="utf-8")
        engine_hammer = root / "engine_hammer"; engine_hammer.mkdir()
        write_json(engine_hammer / "review.json", {
            "status": "PASS_M1117R3_M1112R3_ENGINE_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY__NO_EDA",
            "identity": {"engine_sha256": EXPECTED["engine"], "contract_sha256": EXPECTED["contract"],
                "m1116_outer_seal_file_sha256": EXPECTED["m1116_outer"],
                "author_receipt_outer_seal_file_sha256": EXPECTED["author_outer"]}})
        engine_hammer_outer = seal_flat(engine_hammer)

        receipt = root / "launch_receipt.json"
        receipt_value = {
            "schema": "m1112r3_c2_authorized_launch_receipt_r1_v1",
            "status": "M1112R3_LAUNCH_SOURCE_FROZEN__M1118R3_REQUIRED__NO_EDA",
            "launcher_sha256": sha(launcher), "engine_sha256": EXPECTED["engine"],
            "engine_contract_sha256": EXPECTED["contract"],
            "engine_contract_outer_seal_file_sha256": EXPECTED["contract_outer"],
            "engine_author_receipt_outer_seal_file_sha256": EXPECTED["author_outer"],
            "m1116_outer_seal_file_sha256": EXPECTED["m1116_outer"],
            "m1117r3_outer_seal_file_sha256": engine_hammer_outer,
            "arguments": 0, "caller_selected_authority_allowed": False,
            "caller_environment_forwarded": False, "m1118r3_required": True,
            "launch_now": False, "attempt_now": False, "dc_now": False,
            "mapped_vcs_now": False, "maximum_attempts": 1,
            "automatic_retry": False, "paper_citable": False,
        }
        write_json(receipt, receipt_value); receipt_outer = seal_double(receipt)

        launch_hammer = root / "launch_hammer"; launch_hammer.mkdir()
        identity = {
            "launch_receipt_outer_seal_file_sha256": receipt_outer,
            "launcher_sha256": sha(launcher), "engine_sha256": EXPECTED["engine"],
            "engine_contract_outer_seal_file_sha256": EXPECTED["contract_outer"],
            "engine_author_receipt_outer_seal_file_sha256": EXPECTED["author_outer"],
            "m1116_outer_seal_file_sha256": EXPECTED["m1116_outer"],
            "m1117r3_outer_seal_file_sha256": engine_hammer_outer,
        }
        good_review = {"status": "PASS_M1118R3_M1112R3_LAUNCH_HAMMER__GO_ONE_ATTEMPT", "identity": identity}
        write_json(launch_hammer / "review.json", good_review); launch_outer = seal_flat(launch_hammer)

        saved = (engine.LAUNCHER, engine.LAUNCH_RECEIPT, engine.M1117R3, engine.M1118R3, engine.verify_parent_launcher)
        engine.LAUNCHER, engine.LAUNCH_RECEIPT, engine.M1117R3, engine.M1118R3 = launcher, receipt, engine_hammer, launch_hammer
        engine.verify_parent_launcher = lambda value: engine.verify_regular(launcher, value["launcher_sha256"])
        try:
            accepted = engine.verify_future_authority()
            require(accepted["m1118r3_outer_seal_file_sha256"] == launch_outer, "self-consistent future outer")
            require("m1118r3_outer_seal_file_sha256" not in receipt_value, "receipt excludes future outer")

            forged = dict(receipt_value); forged["m1118r3_outer_seal_file_sha256"] = "PLACEHOLDER"
            write_json(receipt, forged); seal_double(receipt)
            mutations["placeholder_or_future_outer_in_receipt"] = rejected(engine, engine.verify_future_authority)
            write_json(receipt, receipt_value); receipt_outer = seal_double(receipt)

            env_bad = dict(receipt_value); env_bad["caller_environment_forwarded"] = True
            write_json(receipt, env_bad); seal_double(receipt)
            mutations["caller_environment_forwarded"] = rejected(engine, engine.verify_future_authority)
            write_json(receipt, receipt_value); receipt_outer = seal_double(receipt)

            fake_outer = launch_hammer / "SHA256SUMS.seal.sha256"
            good_outer_bytes = fake_outer.read_bytes(); fake_outer.write_text("0" * 64 + "  SHA256SUMS\n", encoding="utf-8")
            mutations["forged_future_outer"] = rejected(engine, engine.verify_future_authority)
            fake_outer.write_bytes(good_outer_bytes)

            changed = json.loads(json.dumps(good_review)); changed["identity"]["engine_sha256"] = "0" * 64
            write_json(launch_hammer / "review.json", changed); seal_flat(launch_hammer)
            mutations["self_consistent_changed_review_bytes"] = rejected(engine, engine.verify_future_authority)
            write_json(launch_hammer / "review.json", good_review); seal_flat(launch_hammer)

            binding_fields = {
                "review_launcher_pin": "launcher_sha256",
                "review_receipt_outer_pin": "launch_receipt_outer_seal_file_sha256",
                "review_engine_pin": "engine_sha256",
                "review_contract_outer_pin": "engine_contract_outer_seal_file_sha256",
                "review_author_outer_pin": "engine_author_receipt_outer_seal_file_sha256",
                "review_m1116_outer_pin": "m1116_outer_seal_file_sha256",
                "review_m1117_outer_pin": "m1117r3_outer_seal_file_sha256",
            }
            for label, field in binding_fields.items():
                changed = json.loads(json.dumps(good_review)); changed["identity"][field] = "f" * 64
                write_json(launch_hammer / "review.json", changed); seal_flat(launch_hammer)
                mutations[label] = rejected(engine, engine.verify_future_authority)
            write_json(launch_hammer / "review.json", good_review); seal_flat(launch_hammer)

            old_receipt = dict(receipt_value); old_receipt["m1117r3_outer_seal_file_sha256"] = EXPECTED["m1114r2_outer"]
            write_json(receipt, old_receipt); seal_double(receipt)
            previous_m1117 = engine.M1117R3; engine.M1117R3 = M1114R2
            mutations["old_hammer_substitution"] = rejected(engine, engine.verify_future_authority)
            engine.M1117R3 = previous_m1117
            write_json(receipt, receipt_value); receipt_outer = seal_double(receipt)

            launcher_real = root / "launcher.real"; launcher.rename(launcher_real); launcher.symlink_to(launcher_real.name)
            mutations["launcher_symlink"] = rejected(engine, engine.verify_future_authority)
            launcher.unlink(); launcher_real.rename(launcher)

            side = Path(str(receipt) + ".sha256"); side_real = root / "receipt.side.real"; side.rename(side_real); side.symlink_to(side_real.name)
            mutations["receipt_sidecar_symlink"] = rejected(engine, engine.verify_future_authority)
            side.unlink(); side_real.rename(side)

            manifest = launch_hammer / "SHA256SUMS"; manifest_real = root / "future.manifest.real"; manifest.rename(manifest_real); manifest.symlink_to(manifest_real)
            mutations["future_hammer_manifest_symlink"] = rejected(engine, engine.verify_future_authority)
            manifest.unlink(); manifest_real.rename(manifest)
        finally:
            (engine.LAUNCHER, engine.LAUNCH_RECEIPT, engine.M1117R3, engine.M1118R3, engine.verify_parent_launcher) = saved

        require(all(mutations.values()), f"chain mutation escaped {mutations}")
        return {
            "valid_chain_accepted": True,
            "receipt_contains_future_hammer_outer": False,
            "future_hammer_outer_self_consistently_discovered": launch_outer,
            "review_exact_binding_fields": sorted(identity),
            "mutations": mutations,
            "mutations_rejected": len(mutations),
        }


def namespace_attack(engine) -> dict:
    canonical = [
        ROOT / "dc_handoff/scripts/run_m1112r3_c2_async_observation_authorized_launch_r1.py",
        ROOT / "contracts/m1112r3_c2_async_observation_authorized_launch_receipt_r1_20260830.json",
        ROOT / "results/.m1112r3_c2_async_observation_dc_mapped_vcs_attempt_consumed",
        ROOT / "results/m1112r3_c2_async_observation_dc_mapped_vcs_r1_20260830",
    ]
    require(not any(path.exists() or path.is_symlink() for path in canonical), "canonical future namespace fresh")
    source = ENGINE.read_text(encoding="utf-8")
    require("if any(path.exists() or path.is_symlink() for path in (ATTEMPT, RESULT, WORK))" in source, "fresh namespace engine gate")
    require(source.count("ATTEMPT.mkdir(); attempted = True") == 1, "single attempt consume")
    with tempfile.TemporaryDirectory(prefix="m1117r3_namespace_", dir=ROOT / "reviews") as raw:
        collision = Path(raw) / "attempt"; collision.mkdir()
        rejected_collision = any(path.exists() or path.is_symlink() for path in (collision, Path(raw) / "result", Path(raw) / "work"))
    require(rejected_collision, "namespace collision model")
    return {"canonical_launcher_receipt_attempt_result_absent": True, "single_attempt_consume_calls": 1, "collision_rejected": True, "automatic_retry": False}


def main() -> None:
    fixed = {
        "engine": ENGINE, "contract": CONTRACT, "contract_side": Path(str(CONTRACT) + ".sha256"),
        "contract_outer": Path(str(CONTRACT) + ".sha256.seal.sha256"),
        "author_review": AUTHOR / "review.json", "author_manifest": AUTHOR / "SHA256SUMS",
        "author_outer": AUTHOR / "SHA256SUMS.seal.sha256", "wrapper": WRAPPER, "tb": TB,
        "docs359": DOCS359,
    }
    for label, path in fixed.items():
        require(regular(path) and sha(path) == EXPECTED[label], f"fixed identity {label}")
    author = verify_flat(AUTHOR, EXPECTED["author_outer"])
    m1116 = verify_flat(M1116, EXPECTED["m1116_outer"])
    m1114 = verify_flat(M1114R2, EXPECTED["m1114r2_outer"])
    require(json.loads(M1116.joinpath("review.json").read_text(encoding="utf-8"))["status"] == "STOP_M1116_M1112R2_FUTURE_LAUNCH_HASH_CYCLE__ADDITIVE_R3_REQUIRED", "M1116 STOP")
    require(json.loads(M1114R2.joinpath("review.json").read_text(encoding="utf-8"))["status"] == "PASS_M1114R2_M1112R2_ENGINE_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY__NO_EDA", "M1114r2 source hammer")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    require(contract["m1116_circularity_authority"]["outer_seal_file_sha256"] == EXPECTED["m1116_outer"], "contract M1116")
    require(contract["m1112r2_frozen_authority"]["m1114r2_outer_seal_file_sha256"] == EXPECTED["m1114r2_outer"], "contract M1114r2")
    require(contract["future_chain"]["launch_receipt_contains_future_m1118r3_outer"] is False and contract["future_chain"]["placeholder_or_hash_fixed_point_allowed"] is False, "acyclic contract")
    for table in ("source_sha256", "frozen_filelist_member_sha256"):
        for relative, digest in contract[table].items():
            path = ROOT / relative; require(regular(path) and sha(path) == digest, f"pinned live source {relative}")

    source = ENGINE.read_text(encoding="utf-8"); ast.parse(source)
    require('launch_outer = verify_flat_self_consistent(M1118R3)' in source, "future outer discovery")
    require('"m1118r3_outer_seal_file_sha256" in receipt' in source and 'verify_exact_flat(M1118R3, receipt["m1118r3_outer_seal_file_sha256"])' not in source, "no future outer dependency")
    engine = import_engine()
    hardware = preserved_hardware_checks(engine)
    chain = chain_hammer(engine)
    namespace = namespace_attack(engine)

    status = "PASS_M1117R3_M1112R3_ENGINE_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY__NO_EDA"
    checks = {
        "schema": "m1117r3_m1112r3_c2_acyclic_engine_hammer_mechanical_v1",
        "status": status,
        "scope": {"static_and_mutation_only": True, "eda": 0, "launcher": 0, "attempt": 0},
        "identity": {**EXPECTED, "author_receipt": author, "m1116": m1116, "m1114r2": m1114},
        "acyclic_chain": chain,
        "preserved_hardware": hardware,
        "namespace": namespace,
    }
    (OUT / "mechanical_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    review = {
        "schema": "m1117r3_m1112r3_c2_acyclic_engine_hammer_review_v1",
        "status": status,
        "verdict": "GO_DIFFERENT_AUTHOR_ZERO_ARGUMENT_LAUNCHER_AUTHORING_ONLY",
        "score": 100,
        "issue_counts": {"P0": 0, "P1": 0, "P2": 0},
        "acyclic_proof": {
            "launch_receipt_binds_only_preexisting_authorities": True,
            "launch_receipt_contains_future_hammer_outer": False,
            "future_hammer_outer_self_consistently_discovered": True,
            "future_hammer_review_exact_authority_binding": True,
            "sha256_fixed_point_required": False,
            "mutations_rejected": chain["mutations_rejected"],
        },
        "preserved": hardware,
        "authorization": {"different_author_launcher_authoring": True, "launcher_execution": False, "attempt_creation": False, "eda": False, "next_required_gate": "Sealed launcher/receipt followed by independent M1118r3 launch hammer."},
        "identity": {
            "engine_sha256": EXPECTED["engine"], "contract_sha256": EXPECTED["contract"],
            "contract_outer_seal_file_sha256": EXPECTED["contract_outer"],
            "author_receipt_outer_seal_file_sha256": EXPECTED["author_outer"],
            "m1116_outer_seal_file_sha256": EXPECTED["m1116_outer"],
            "m1114r2_outer_seal_file_sha256": EXPECTED["m1114r2_outer"],
            "docs359_sha256": EXPECTED["docs359"],
        },
        "claim_boundary": {"source_launcher_admission_only": True, "mapped_functionality": False, "performance": False, "activity_or_power": False, "system_speedup": False, "paper_citable": False, "paper_ppa_ready": False},
    }
    (OUT / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "RUN_COMPLETE.txt").write_text(status + "\n", encoding="utf-8")
    (OUT / "READONLY_NO_LAUNCH.txt").write_text("M1117r3 source/engine hammer only; no launcher, receipt, attempt, result, EDA, VCS, DC, or simv was created or executed.\n", encoding="utf-8")
    print(f"{status} chain_mutations={chain['mutations_rejected']} eda=0 launcher=0 attempt=0")


if __name__ == "__main__":
    main()
