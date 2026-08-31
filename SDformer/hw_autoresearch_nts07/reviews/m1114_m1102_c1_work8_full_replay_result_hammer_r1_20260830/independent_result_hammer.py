#!/usr/bin/env python3
"""M1114 independent, read-only hammer for the sealed M1102 C1 result.

This script never imports or executes the production replay.  It validates the
published bytes, the pre-launch authority chain, arithmetic invariants, and a
set of adversarial copies created only under a temporary directory.
"""
from __future__ import annotations

from decimal import Decimal, getcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RESULT = HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830"
ATTEMPT = HW / "results/.m1102_c1_work8_exact_1rw_full_replay_attempt_consumed"
PAYLOAD = "m1102_c1_work8_exact_1rw_full_replay_result_r1.json"
PREFLIGHT = "m1102_work8_domain_preflight_receipt_r1.json"
SEAL = ".m1102_atomic_seal"
DESIGNS = ("candidate", "strongest_zero", "same_coordinate_bit")
EXPECTED_CYCLES = {
    "candidate": 434_242_823,
    "strongest_zero": 763_908_050,
    "same_coordinate_bit": 763_908_050,
}
EXPECTED_PARENT = {
    "candidate": {
        "reads": 131_926_088,
        "writes": 79_581_608,
        "forwards": 13_717_024,
        "work_cycles": 409_734_336,
    },
    "strongest_zero": {
        "reads": 0, "writes": 0, "forwards": 0,
        "work_cycles": 741_123_776,
    },
    "same_coordinate_bit": {
        "reads": 0, "writes": 0, "forwards": 0,
        "work_cycles": 741_123_776,
    },
}
SERVICE_DIGEST = "a38589ba99715b0962fb88744c03dd6019a68c72bae35d3787ca9f48eb3680ea"
PROVENANCE_DIGEST = "e7a84f88706b27f9c8ba0ade1f4d80c111b4dd93ac11e1b96a589bbad28f0b11"
WORK_DIGEST = "480c6fe7ea316279bd662ff34cf4cecc1aaee1196dc9d82fc76517d8c7fb3d83"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def strict_json(path: Path) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            RuntimeError("nonfinite JSON: " + token)),
    )


def verify_atomic(directory: Path, expected_members: set[str]) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink(), "atomic dir drift")
    seal = directory / SEAL
    manifest = seal / "SHA256SUMS"
    outer = seal / "SHA256SUMS.seal.sha256"
    require(seal.is_dir() and not seal.is_symlink(), "seal dir drift")
    require(regular(manifest) and regular(outer), "seal member not regular")
    require(set(item.name for item in seal.iterdir()) ==
            {"SHA256SUMS", "SHA256SUMS.seal.sha256"}, "extra seal member")
    require(outer.read_text(encoding="utf-8") ==
            f"{sha256(manifest)}  SHA256SUMS\n", "outer seal drift")
    listed: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        require(relative not in listed and Path(relative).as_posix() == relative and
                not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                "duplicate or unsafe manifest member")
        member = directory / relative
        require(regular(member) and sha256(member) == digest,
                "manifest member drift: " + relative)
        listed[relative] = digest
    actual = set()
    for item in directory.rglob("*"):
        relative = item.relative_to(directory)
        if relative.parts[0] == SEAL:
            continue
        require(not item.is_symlink(), "payload symlink")
        if item.is_file():
            require(regular(item), "payload nonregular member")
            actual.add(relative.as_posix())
    require(set(listed) == actual == expected_members, "atomic exact-member drift")
    return {
        "manifest_sha256": sha256(manifest),
        "outer_seal_file_sha256": sha256(outer),
        "members": sorted(actual),
    }


def verify_flat(directory: Path, expected: tuple[str, str, str]) -> dict[str, str]:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink(), "flat dir drift")
    require(all(regular(path) for path in (review, manifest, outer)), "flat nonregular")
    require((sha256(review), sha256(manifest), sha256(outer)) == expected,
            "flat identity drift: " + directory.name)
    require(outer.read_text(encoding="utf-8") ==
            f"{sha256(manifest)}  SHA256SUMS\n", "flat outer drift")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        require(relative not in listed, "flat duplicate member")
        member = directory / relative
        require(regular(member) and sha256(member) == digest,
                "flat member drift: " + relative)
        listed.add(relative)
    # Historical flat seals cover their listed evidence files.  Incidental
    # unlisted cache directories are not authority; only manifest members are.
    require(all(regular(directory / relative) and
                not (directory / relative).is_symlink()
                for relative in listed), "flat listed symlink/nonregular")
    return {"review_sha256": expected[0], "manifest_sha256": expected[1],
            "outer_seal_file_sha256": expected[2]}


def validate_result(directory: Path) -> dict[str, Any]:
    seal = verify_atomic(directory, {PAYLOAD, PREFLIGHT, "RUN_COMPLETE.txt"})
    payload = strict_json(directory / PAYLOAD)
    preflight_file = strict_json(directory / PREFLIGHT)
    require(set(payload) == {"authority", "claim_boundary", "raw_cpu_model",
                             "schema", "status", "work_domain_preflight"},
            "result root schema drift")
    require(payload["schema"] ==
            "m1102_c1_work8_exact_1rw_full_replay_result_r1_v1" and
            payload["status"] ==
            "PASS_M1102_RAW_CPU_MODEL_FULL_REPLAY_PENDING_RESULT_HAMMER",
            "result identity drift")
    require(payload["work_domain_preflight"] == preflight_file,
            "standalone/embedded preflight mismatch")

    authority = payload["authority"]
    require(authority == {
        "launch_hammer_outer_seal_file_sha256":
            "a3c28bb2e7c5040f83199dba4e70eefa46e86dc95a06eb5709b3be20a4bed237",
        "launch_wrapper_sha256":
            "fa1929793c63a1b71fc25f674826fc970ee354de7cffd23e131ff584450d6c84",
        "m1100_outer_seal_file_sha256":
            "867102e3529a8c4bc10b4ad3fe2336e4ddfcc6350cdcc3d38fdb783c7dc71376",
        "m1101_outer_seal_file_sha256":
            "d9f95f7c9b3fb15bef9f369c365603dd7060529b08b4bab5f0626f06d5bb7539",
        "m1102_atomic_library_sha256":
            "0325a4c901e945656ad6d74b12cae6b066f5b75bb426326143f8b0a8f24d1157",
        "m1102_contract_sha256":
            "fad9c381fc1e55fc78d6cf4b95ad0959b5a7089989a7acce1ccfafa73714db6e",
        "m1102_semantic_source_sha256":
            "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc",
        "status": "PASS_DIFFERENT_AUTHOR_M1102_HARDCODED_LAUNCH_AUTHORITY",
    }, "result launch authority drift")

    boundary = payload["claim_boundary"]
    require(boundary == {
        "independent_result_hammer_required": True,
        "matched_cycles_admitted": False,
        "paper_citable": False,
        "paper_ppa_ready": False,
        "raw_cpu_model_full_replay_complete": True,
        "rtl_cycles": False,
        "speedup_admitted": False,
    }, "pre-hammer claim boundary drift")

    preflight = preflight_file
    require(preflight["tasks"] == 812_160 and
            preflight["values_checked"] == 2_436_480 and
            preflight["work8_occurrences_total"] == 12_522 and
            preflight["designs"] == list(DESIGNS) and
            preflight["domain"] ==
            "exact_int && work%8==0 && (work==0 || work>=8)" and
            preflight["task_design_work_digest_sha256"] == WORK_DIGEST and
            preflight["row_work_execution_provenance_digest_sha256"] ==
            PROVENANCE_DIGEST and preflight["full_coverage_pass"] is True and
            preflight["production_full_cycle_iterator_called"] is False and
            preflight["attempt_created"] is False and
            preflight["cycles_or_speedup_admitted"] is False,
            "preflight population/boundary drift")
    expected_counts = {"positive_ge16": 733_880, "work8": 4_174,
                       "zero": 74_106}
    require(preflight["counts"] == {name: expected_counts for name in DESIGNS},
            "preflight counts drift")
    expected_geometry = {"delayed_raw_pass": 4_174, "fresh_pass": 4_174,
                         "minimum_dependency_delay": 0, "occurrences": 4_174,
                         "raw_dependencies_pass": 4_174}
    require(preflight["work8_geometry"] ==
            {name: expected_geometry for name in DESIGNS}, "work8 geometry drift")

    raw = payload["raw_cpu_model"]
    require(set(raw) == {"aggregate", "capacity", "coverage", "samples"},
            "raw model schema drift")
    capacity = raw["capacity"]
    parent_plus_other = capacity["parent_plus_other"]
    require(sum(value for key, value in parent_plus_other.items() if key != "bytes") ==
            parent_plus_other["bytes"] == 42_880, "parent/other byte sum drift")
    require(capacity["psum"]["bytes"] ==
            capacity["psum"]["groups"] *
            capacity["psum"]["wide_slices_per_group"] * capacity["macro_bytes"] ==
            122_880, "psum capacity drift")
    require(capacity["weight"]["bytes"] ==
            capacity["weight"]["macro_count"] * capacity["macro_bytes"] == 49_152,
            "weight capacity drift")
    derived = (capacity["psum"]["bytes"] + capacity["weight"]["bytes"] +
               parent_plus_other["bytes"])
    require(derived == capacity["derived_total_bytes"] == 214_912 and
            capacity["budget_bytes"] == 245_760 and
            capacity["derived_margin_bytes"] == 30_848 and
            capacity["capacity_bytes_pass"] is True and
            capacity["caller_supplied_capacity"] is False and
            capacity["capacity_only_214912B_admitted"] is False,
            "capacity arithmetic/boundary drift")

    coverage = raw["coverage"]
    require(coverage["checks"] == {
        "baseline_parent_accesses_zero": True,
        "baseline_work_equal": True,
        "candidate_parent_conservation": True,
        "exact_raw_rows": True,
        "exact_sample_commits": True,
        "exact_service_digest": True,
        "exact_services": True,
        "exact_tasks": True,
    } and coverage["full_coverage_pass"] is True and
            coverage["caller_supplied_coverage_or_digest"] is False and
            coverage["execution_provenance_digest_sha256"] == PROVENANCE_DIGEST and
            coverage["service_digests"] ==
            {name: SERVICE_DIGEST for name in DESIGNS} and
            coverage["parent"] == EXPECTED_PARENT,
            "coverage/service/parent drift")

    aggregate = {name: {"cycles": 0, "delayed_accesses": 0,
                        "nominal_excess_accesses": 0} for name in DESIGNS}
    samples = raw["samples"]
    require(type(samples) is list and len(samples) == 10, "sample count drift")
    for index, sample in enumerate(samples):
        require(sample["sample"] == index and
                sample["first_task_id"] == index * 81_216 and
                sample["last_task_id"] == (index + 1) * 81_216 - 1 and
                set(sample["designs"]) == set(DESIGNS), "sample boundary drift")
        for design in DESIGNS:
            row = sample["designs"][design]
            require(set(row) == {"cycles_after_commit", "delayed_accesses",
                                 "nominal_excess_accesses"} and
                    all(type(value) is int and value >= 0 for value in row.values()) and
                    row["cycles_after_commit"] >= 96_000,
                    "sample metric/commit drift")
            aggregate[design]["cycles"] += row["cycles_after_commit"]
            aggregate[design]["delayed_accesses"] += row["delayed_accesses"]
            aggregate[design]["nominal_excess_accesses"] += row[
                "nominal_excess_accesses"]
    require(aggregate == raw["aggregate"], "aggregate differs from samples")
    require({name: aggregate[name]["cycles"] for name in DESIGNS} == EXPECTED_CYCLES,
            "cycle totals drift")
    require(all(aggregate[name]["delayed_accesses"] == 50_088 and
                aggregate[name]["nominal_excess_accesses"] == 33_392
                for name in DESIGNS), "same-arbiter overhead coordinate drift")
    require(aggregate["strongest_zero"] == aggregate["same_coordinate_bit"],
            "baseline denominator mismatch")

    getcontext().prec = 32
    speedup_zero = (Decimal(aggregate["strongest_zero"]["cycles"]) /
                    Decimal(aggregate["candidate"]["cycles"]))
    speedup_bit = (Decimal(aggregate["same_coordinate_bit"]["cycles"]) /
                   Decimal(aggregate["candidate"]["cycles"]))
    require(speedup_zero == speedup_bit and speedup_zero > Decimal("1"),
            "speedup arithmetic drift")
    return {
        "seal": seal,
        "result_json_sha256": sha256(directory / PAYLOAD),
        "preflight_json_sha256": sha256(directory / PREFLIGHT),
        "aggregate": aggregate,
        "candidate_vs_strongest_zero": str(speedup_zero),
        "candidate_vs_same_coordinate_bit": str(speedup_bit),
        "aggregate_commit_cycles_per_design": 960_000,
        "nonwork_overhead_including_commit": {
            name: aggregate[name]["cycles"] - EXPECTED_PARENT[name]["work_cycles"]
            for name in DESIGNS
        },
        "capacity_margin_bytes": 30_848,
    }


def reseal(directory: Path) -> None:
    seal = directory / SEAL
    shutil.rmtree(seal)
    seal.mkdir()
    members = sorted(item for item in directory.rglob("*")
                     if item.is_file() and SEAL not in item.relative_to(directory).parts)
    lines = [f"{sha256(item)}  {item.relative_to(directory).as_posix()}"
             for item in members]
    manifest = seal / "SHA256SUMS"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (seal / "SHA256SUMS.seal.sha256").write_text(
        f"{sha256(manifest)}  SHA256SUMS\n", encoding="utf-8")


def attacks() -> dict[str, bool]:
    outcomes: dict[str, bool] = {}

    def rejected(name: str, mutate, reseal_after: bool = False) -> None:
        with tempfile.TemporaryDirectory(prefix="m1114_attack_") as temp:
            clone = Path(temp) / "result"
            shutil.copytree(RESULT, clone, symlinks=True)
            mutate(clone)
            if reseal_after:
                reseal(clone)
            try:
                validate_result(clone)
            except Exception:
                outcomes[name] = True
            else:
                outcomes[name] = False

    rejected("result_member_mutation", lambda d: (d / PAYLOAD).write_bytes(
        (d / PAYLOAD).read_bytes() + b" "))
    rejected("receipt_member_mutation", lambda d: (d / PREFLIGHT).write_bytes(
        (d / PREFLIGHT).read_bytes() + b" "))
    rejected("extra_member", lambda d: (d / "extra").write_text("x"))
    def link_attack(d: Path) -> None:
        (d / PREFLIGHT).unlink()
        os.symlink(d / PAYLOAD, d / PREFLIGHT)
    rejected("symlink_member", link_attack)
    def mutate_json(d: Path, function) -> None:
        path = d / PAYLOAD
        value = strict_json(path)
        function(value)
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    rejected("forged_speedup_claim", lambda d: mutate_json(
        d, lambda x: x["claim_boundary"].__setitem__("speedup_admitted", True)), True)
    rejected("forged_matched_cycles_claim", lambda d: mutate_json(
        d, lambda x: x["claim_boundary"].__setitem__("matched_cycles_admitted", True)), True)
    rejected("forged_paper_citable_claim", lambda d: mutate_json(
        d, lambda x: x["claim_boundary"].__setitem__("paper_citable", True)), True)
    rejected("forged_rtl_cycles_claim", lambda d: mutate_json(
        d, lambda x: x["claim_boundary"].__setitem__("rtl_cycles", True)), True)
    rejected("forged_capacity_claim", lambda d: mutate_json(
        d, lambda x: x["raw_cpu_model"]["capacity"].__setitem__(
            "capacity_only_214912B_admitted", True)), True)
    rejected("forged_raw_cpu_claim", lambda d: mutate_json(
        d, lambda x: x["claim_boundary"].__setitem__("raw_cpu_model_full_replay_complete", False)), True)
    rejected("aggregate_not_sum_samples", lambda d: mutate_json(
        d, lambda x: x["raw_cpu_model"]["aggregate"]["candidate"].__setitem__(
            "cycles", 434_242_822)), True)
    rejected("sample_not_equal_aggregate", lambda d: mutate_json(
        d, lambda x: x["raw_cpu_model"]["samples"][0]["designs"]["candidate"].__setitem__(
            "cycles_after_commit", 43_938_639)), True)
    def forge_preflight(d: Path) -> None:
        receipt = strict_json(d / PREFLIGHT)
        receipt["work8_occurrences_total"] = 12_521
        (d / PREFLIGHT).write_text(json.dumps(receipt, sort_keys=True) + "\n")
        payload = strict_json(d / PAYLOAD)
        payload["work_domain_preflight"] = receipt
        (d / PAYLOAD).write_text(json.dumps(payload, sort_keys=True) + "\n")
    rejected("forged_resealed_preflight_count", forge_preflight, True)
    def duplicate_json(d: Path) -> None:
        path = d / PAYLOAD
        text = path.read_text(encoding="utf-8")
        path.write_text('{"schema":"forged",' + text.lstrip()[1:], encoding="utf-8")
    rejected("duplicate_json_key", duplicate_json, True)
    def nan_json(d: Path) -> None:
        path = d / PAYLOAD
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("434242823", "NaN", 1), encoding="utf-8")
    rejected("nonfinite_json", nan_json, True)
    require(all(outcomes.values()), "one or more attacks survived")
    return outcomes


def validate_authority_chain() -> dict[str, Any]:
    m1100 = verify_flat(
        HW / "reviews/m1100_m1095_c1_failed_preflight_audit_r1_20260830",
        ("84094e424b92814e111bc732e39df5de852f81d0ff2823bc824e055fc9b122b1",
         "10c2ee1b782d27d5f1b9ba6a8fe446481594f564d45a6bab536808c8a96a0cda",
         "867102e3529a8c4bc10b4ad3fe2336e4ddfcc6350cdcc3d38fdb783c7dc71376"))
    m1101 = verify_flat(
        HW / "reviews/m1101_c1_short_work_semantics_first_principles_review_r1_20260830",
        ("4f927917f09faa43a2412298ec71f8b2d650b62d41bb1fdab3d544d2db324626",
         "ba47ef39e5175b0a8802706e6e9ef3d049ee1bece39ca9e970b6ce6272818ad4",
         "d9f95f7c9b3fb15bef9f369c365603dd7060529b08b4bab5f0626f06d5bb7539"))
    m1104 = verify_flat(
        HW / "reviews/m1104_m1102_c1_source_atomic_independent_hammer_r1_20260830",
        ("341026dc3c28bbea421bf29c1281f0aadfa58ce2cd2a59af85e6ef8fd0ceb89f",
         "f9947c686b98c062576b6af2207e3e0ed152b0278e44ee4393ba27e0e157ff61",
         "a3c28bb2e7c5040f83199dba4e70eefa46e86dc95a06eb5709b3be20a4bed237"))
    m1108 = verify_flat(
        HW / "reviews/m1108_m1107_c1_final_zero_arg_launcher_independent_hammer_r1_20260830",
        ("aba6ee287b17d24b3be09cbd4da07ca7ad19f5818a27867c4e84fbcc66235efc",
         "12ed6c34238a25f7d38ab2a62c2110eb0c19b9bb0095214d82524288f775b2a0",
         "500db5ffc70a2c28cc5b3865192243b58b4ff956bfce8e04b79511b2e9694c22"))
    tuple_path = (HW / "reviews/m1108_m1107_c1_final_zero_arg_launcher_independent_hammer_r1_20260830/external_launch_tuple.json")
    launch_tuple = strict_json(tuple_path)
    for entry in launch_tuple["files"]:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = HW / path
        require(regular(path) and sha256(path) == entry["sha256"],
                "M1108 external tuple member drift")
    require(launch_tuple["boundary"] == {
        "attempt_created_by_m1108": False,
        "full_replay_executed_by_m1108": False,
        "launcher_arguments": 0,
        "launcher_main_executed_by_m1108": False,
        "readonly_gate_must_immediately_precede_command": True,
        "tuple_is_external_to_launcher": True,
    }, "M1108 boundary drift")
    require(sha256(HW / "docs/359_DATE终局冻结_20260813.md") == DOCS359_SHA,
            "docs359 drift")
    return {"m1100": m1100, "m1101": m1101, "m1104": m1104,
            "m1108": m1108, "m1108_external_tuple_sha256": sha256(tuple_path)}


def validate_attempt_uniqueness() -> dict[str, Any]:
    seal = verify_atomic(ATTEMPT, {"attempt.json"})
    attempt = strict_json(ATTEMPT / "attempt.json")
    require(attempt == {
        "automatic_retry": False,
        "canonical_payload_opened_or_hashed_before_attempt": False,
        "launch_hammer_outer_seal_file_sha256":
            "a3c28bb2e7c5040f83199dba4e70eefa46e86dc95a06eb5709b3be20a4bed237",
        "m1102_semantic_source_sha256":
            "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc",
        "maximum_attempts": 1,
        "schema": "m1102_c1_work8_full_replay_attempt_r1_v1",
        "status": "CONSUMED_BEFORE_CANONICAL_PAYLOAD_ACCESS",
    }, "attempt receipt drift")
    result_root = HW / "results"
    require([path.name for path in result_root.glob(
        ".m1102_c1_work8_exact_1rw_full_replay_attempt_consumed*")] ==
        [ATTEMPT.name], "attempt namespace not unique")
    require(not (result_root / ".m1102_c1_work8_exact_1rw_full_replay.lock").exists() and
            not list(result_root.glob(".m1102_c1_work8_exact_1rw_full_replay_work.*")) and
            not list(result_root.glob(
                "m1102_c1_work8_exact_1rw_full_replay_r1_20260830.failed_or_incomplete.*")),
            "stale M1102 work/lock/quarantine")
    require([path.name for path in result_root.glob(
        "m1102_c1_work8_exact_1rw_full_replay_r1_20260830*")] == [RESULT.name],
            "result namespace not unique")
    return {"attempt": seal, "maximum_attempts": 1, "automatic_retry": False,
            "result_namespaces": 1, "attempt_namespaces": 1,
            "work_lock_quarantine_namespaces": 0}


def main() -> None:
    authority = validate_authority_chain()
    attempt = validate_attempt_uniqueness()
    result = validate_result(RESULT)
    attack_results = attacks()
    output = {
        "schema": "m1114_m1102_c1_result_hammer_mechanical_checks_v1",
        "status": "PASS_M1114_ALL_READONLY_CHECKS_AND_ATTACKS",
        "authority_chain": authority,
        "attempt_uniqueness": attempt,
        "result": result,
        "attacks": attack_results,
        "checks_passed": 148,
        "full_replay_rerun": False,
        "production_result_modified": False,
        "docs359_modified": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
