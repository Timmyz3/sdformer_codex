#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1122C Path-C source-only author check; no RTL or EDA execution."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import stat
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CONTRACT = HW / "contracts/m1122c_m1121c_c1_path_c_identical_external_common_charge_source_contract_r1_20260830.json"
CONTRACT_ID = (
    "cc263438c7fe98250258440fcfe9bf3dbb7942d7ee5cb09dde22cedd58b4a014",
    "474af7d4a847a3bb20d631a88738e9a873c507fbbb130b58fafedc479b581dd2",
    "e796e4b7e1ecf1c1ea9cf429fb7dcce18762ffa7d02d93f399957c31b65ab930",
)
M1121C = HW / "reviews/m1121c_c1_214912_capacity_vs_physical_obligation_first_principles_audit_r1_20260830"
M1121C_ID = (
    "263fb4346c935fa1dd37dacbc6124f59efb874c4ce75c3f03c28715533594d0c",
    "7876f32ed5b9327dc7425fd738063dd27e7b66dea9e24c4631aadb72e7dadfa8",
    "a7cf1eb1dfda536e7d0c8e1e597bebba4f320468357a09dae78f94ef395d33d3",
)
M1114 = HW / "reviews/m1114_m1102_c1_work8_full_replay_result_hammer_r1_20260830"
M1114_ID = (
    "8ced2392215b7bd70b8afcc90efab3f6078c9b3cc9b1a9d7b0c1d5e33d36b8bc",
    "3f48f2c91e1feba599fca3eab9f3c8348ed5ca5af1d317de14dd01a548b1c1b7",
    "f423e3317825cdb02e637e70d12a9b625df2c4519a4041c3ad9b4440a65c9ef4",
)
M1000 = HW / "reviews/m1000_c1_same_ledger_storage_physical_closure_first_principles_r1_20260829"
M1000_ID = (
    "475dace8e8b8d7e3c40e6c252c2eea5e4f1ae228d7789bac26ea482fb58c6944",
    "5424a5a5c60d7040327cfcfca40e16f3eb28aa6de9504fed8b98c12304d05eac",
    "fd700b7f9e1497fb4ed7fda5f1c725c5408233a84238da6787a871e69892f4d5",
)
M1102_ROOT = HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830"
M1102_RESULT = M1102_ROOT / "m1102_c1_work8_exact_1rw_full_replay_result_r1.json"
M1102_ID = (
    "a229c21b1469f2482ade412a8965e66018db1e4aaa5d434329994a0572587d91",
    "6af45f4091ab4a88b6a60a70f4caf89ceccccee7857a7debe6d8433f9843ee12",
    "f6c9d12b105991ec4ed046e709a2b4d8d983636882cfdcebaae194bd852be96f",
)
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


checks: list[str] = []
attacks: list[str] = []


def require(value: bool, label: str) -> None:
    if not value:
        raise RuntimeError(label)
    checks.append(label)


def reject(label: str, function, *args) -> None:
    try:
        function(*args)
    except Exception:
        attacks.append(label)
        return
    raise RuntimeError("mutation accepted: " + label)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RuntimeError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


def regular(path: Path, expected: str, label: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), label + " regular")
    require(sha(path) == expected, label + " identity")


def double(path: Path, identity: tuple[str, str, str], label: str) -> dict:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    regular(path, identity[0], label + " file")
    regular(side, identity[1], label + " side")
    regular(outer, identity[2], label + " outer")
    require(side.read_text(encoding="utf-8").split() ==
            [identity[0], path.relative_to(HW).as_posix()], label + " side content")
    require(outer.read_text(encoding="utf-8").split() ==
            [identity[1], side.relative_to(HW).as_posix()], label + " outer content")
    return strict_json(path)


def flat(directory: Path, identity: tuple[str, str, str], status: str, label: str) -> dict:
    mode = directory.lstat().st_mode
    require(stat.S_ISDIR(mode) and not directory.is_symlink(), label + " directory")
    review, manifest = directory / "review.json", directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular(review, identity[0], label + " review")
    regular(manifest, identity[1], label + " manifest")
    regular(outer, identity[2], label + " outer")
    require(outer.read_text(encoding="utf-8").split() ==
            [identity[1], "SHA256SUMS"], label + " outer content")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None,
                label + " manifest grammar")
        relative = fields[1].lstrip("*"); relpath = Path(relative)
        require(relative not in expected and relative == relpath.as_posix() and
                not relpath.is_absolute() and ".." not in relpath.parts,
                label + " safe member")
        expected[relative] = fields[0]
    actual: set[str] = set()
    for member in directory.rglob("*"):
        relative = member.relative_to(directory).as_posix()
        if relative in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        member_mode = member.lstat().st_mode
        require(not stat.S_ISLNK(member_mode), label + " rejects symlink")
        if stat.S_ISREG(member_mode): actual.add(relative)
        else: require(stat.S_ISDIR(member_mode), label + " rejects special member")
    require(actual == set(expected), label + " exact coverage")
    for relative, digest in expected.items():
        regular(directory / relative, digest, label + " member " + relative)
    value = strict_json(review)
    require(value.get("status") == status, label + " status")
    return value


def m1102_atomic() -> dict:
    seal = M1102_ROOT / ".m1102_atomic_seal"
    manifest, outer = seal / "SHA256SUMS", seal / "SHA256SUMS.seal.sha256"
    regular(M1102_RESULT, M1102_ID[0], "M1102 result")
    regular(manifest, M1102_ID[1], "M1102 manifest")
    regular(outer, M1102_ID[2], "M1102 outer")
    require(outer.read_text(encoding="utf-8").split() ==
            [M1102_ID[1], "SHA256SUMS"], "M1102 outer content")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(None, 1); relative = relative.lstrip("*")
        regular(M1102_ROOT / relative, digest, "M1102 member " + relative)
    return strict_json(M1102_RESULT)


def snapshot(value: dict[str, Any]) -> dict[str, Any]:
    ext = value["identical_external_capacity_charge"]
    known = ext["known_geometry"]
    residual = ext["conservative_residual_common_charge"]
    identity = ext["three_axis_identity"]
    boundary = value["measurement_boundaries"]
    no_double = value["no_double_count"]
    raw = value["frozen_raw_cpu_opportunity"]
    return {
        "axes": value["comparison_axes"],
        "candidate_bytes": identity["candidate_capacity_bytes"],
        "zero_bytes": identity["strongest_zero_capacity_bytes"],
        "bit_bytes": identity["same_coordinate_bit_capacity_bytes"],
        "known_macro_equivalents": known["known_macro_equivalents"],
        "known_bytes": known["known_capacity_bytes"],
        "residual_bytes": residual["bytes_each_axis"],
        "residual_live": residual["live_storage"],
        "residual_instantiated": residual["instantiated_storage"],
        "residual_macro_count": residual["physical_macro_count"],
        "technology_identical": identity["technology_parameters_identical"],
        "ports_identical": identity["port_parameters_identical_when_frozen"],
        "actual_access_counts_forced_identical": identity["actual_dynamic_access_counts_forced_identical"],
        "logic_top_parent_macros": no_double["logic_only_top_parent_macro_instances_allowed"],
        "external_parent_macros": no_double["external_parent_macro_equivalents"],
        "parent_exactly_once": no_double["parent_macro_area_may_appear_in_exactly_one_of_logic_or_external"],
        "logic_only_may_be_total": boundary["logic_only_dc"]["may_be_called_total"],
        "external_numeric_model_frozen": boundary["external_memory_model"]["numeric_model_frozen_now"],
        "raw_ratio": raw["candidate_vs_strongest_zero"],
        "raw_admitted": raw["admitted"],
        "rtl_speedup": raw["rtl_speedup"],
        "system_speedup": raw["system_speedup"],
        "ppa_or_energy": raw["ppa_or_energy"],
        "eda_authorized": value["authorization"]["dc_pt_saif_ptpx_now"],
    }


EXPECTED = {
    "axes": ["candidate", "strongest_zero", "same_coordinate_bit"],
    "candidate_bytes": 214912,
    "zero_bytes": 214912,
    "bit_bytes": 214912,
    "known_macro_equivalents": 93,
    "known_bytes": 190464,
    "residual_bytes": 24448,
    "residual_live": False,
    "residual_instantiated": False,
    "residual_macro_count": None,
    "technology_identical": True,
    "ports_identical": True,
    "actual_access_counts_forced_identical": False,
    "logic_top_parent_macros": 0,
    "external_parent_macros": 9,
    "parent_exactly_once": True,
    "logic_only_may_be_total": False,
    "external_numeric_model_frozen": False,
    "raw_ratio": 1.7591725401987818,
    "raw_admitted": True,
    "rtl_speedup": False,
    "system_speedup": False,
    "ppa_or_energy": False,
    "eda_authorized": False,
}


def validate(value: dict[str, Any]) -> None:
    observed = snapshot(value)
    if observed != EXPECTED:
        raise RuntimeError("Path-C snapshot drift")
    if observed["known_macro_equivalents"] * 2048 != observed["known_bytes"]:
        raise RuntimeError("known geometry arithmetic")
    if observed["known_bytes"] + observed["residual_bytes"] != observed["candidate_bytes"]:
        raise RuntimeError("total charge arithmetic")
    if not (observed["candidate_bytes"] == observed["zero_bytes"] == observed["bit_bytes"]):
        raise RuntimeError("three-axis byte identity")


def main() -> None:
    contract = double(CONTRACT, CONTRACT_ID, "Path-C contract")
    m1121c = flat(M1121C, M1121C_ID,
                  "PASS_SCOPE_CORRECTION__GO_PATH_C_OR_B__PATH_A_STOP_REMAINS",
                  "M1121C")
    m1114 = flat(M1114, M1114_ID,
                 "PASS_M1114_M1102_C1_RAW_CPU_SAME_LEDGER_RESULT_HAMMER", "M1114")
    m1000 = flat(M1000, M1000_ID,
                 "PASS_M1000_STORAGE_RECONCILIATION__147246UM2_COMPONENT_ONLY_AFTER_PROMOTION__MAIN_TABLE_BLOCKED",
                 "M1000")
    m1102 = m1102_atomic()
    regular(DOCS359, DOCS359_SHA, "docs359")

    require(contract["authority"]["m1121c_outer_seal_file_sha256"] == M1121C_ID[2] and
            contract["authority"]["m1114_outer_seal_file_sha256"] == M1114_ID[2] and
            contract["authority"]["m1000_outer_seal_file_sha256"] == M1000_ID[2] and
            contract["authority"]["m1102_result_outer_seal_file_sha256"] == M1102_ID[2],
            "exact authority binding")
    validate(contract); require(True, "canonical Path-C snapshot")
    capacity = m1102["raw_cpu_model"]["capacity"]
    require(capacity["derived_total_bytes"] == 214912 and
            capacity["psum"]["macro_count"] + capacity["weight"]["macro_count"] + 9 == 93 and
            capacity["psum"]["bytes"] + capacity["weight"]["bytes"] +
                capacity["parent_plus_other"]["parent_scratch_bytes"] == 190464,
            "M1102 93-equivalent/190464 evidence")
    require(m1114["cycle_rederivation"]["candidate_vs_strongest_zero"] ==
            1.7591725401987818 and
            m1114["admission"]["raw_cpu_same_ledger_speedup_admitted"] is True and
            m1114["admission"]["rtl_speedup_admitted"] is False and
            m1114["admission"]["ppa_or_energy_admitted"] is False,
            "M1114 raw-only boundary")
    require(m1121c["paths"]["C_identical_external_common_charge"]["legal"] is True and
            m1121c["paths"]["C_identical_external_common_charge"]["recommended_for_fastest_date_closure"] is True,
            "M1121C Path-C authority")
    require(m1000["reconciliation"]["fifo_control_reserve_status"].startswith(
                "16384-byte analytical reserve is not an instantiated memory") and
            m1000["claim_boundary"]["paper_ppa_ready"] is False,
            "M1000 reserve/model boundary")

    formulas = contract["future_matched_aggregation_formulas"]
    require(formulas["area_total_axis"] == "A_total_axis = A_logic_axis + A_ext_common",
            "area formula")
    require("joint_replay" in formulas["execution_time_axis"] and
            "do not add independently overlapped cycle totals" in formulas["execution_time_axis"],
            "joint timing formula")
    require("Nread_axis_k*Eread_common_k" in formulas["external_dynamic_energy_axis"] and
            "Nwrite_axis_k*Ewrite_common_k" in formulas["external_dynamic_energy_axis"],
            "actual access/common coefficient formula")
    require("E_logic_axis + E_ext_dyn_axis + E_ext_leak_axis" in
            formulas["energy_total_axis"] and
            formulas["speedup_candidate_vs_baseline"] == "Speedup = T_baseline / T_candidate",
            "energy/speedup formulas")
    require(len(contract["forbidden_claims"]) == 11 and
            "93 macros implement the complete 214912-byte coordinate" in contract["forbidden_claims"] and
            "1.7591725402x RTL or mapped-gate speedup" in contract["forbidden_claims"],
            "forbidden claim list")

    mutation_paths = (
        ("axis removal", ("comparison_axes",), ["candidate", "strongest_zero"]),
        ("candidate bytes", ("identical_external_capacity_charge", "three_axis_identity", "candidate_capacity_bytes"), 214911),
        ("zero bytes", ("identical_external_capacity_charge", "three_axis_identity", "strongest_zero_capacity_bytes"), 190464),
        ("bit bytes", ("identical_external_capacity_charge", "three_axis_identity", "same_coordinate_bit_capacity_bytes"), 245760),
        ("claim 105 known macros", ("identical_external_capacity_charge", "known_geometry", "known_macro_equivalents"), 105),
        ("known covers total", ("identical_external_capacity_charge", "known_geometry", "known_capacity_bytes"), 214912),
        ("erase residual", ("identical_external_capacity_charge", "conservative_residual_common_charge", "bytes_each_axis"), 0),
        ("claim residual live", ("identical_external_capacity_charge", "conservative_residual_common_charge", "live_storage"), True),
        ("claim residual instantiated", ("identical_external_capacity_charge", "conservative_residual_common_charge", "instantiated_storage"), True),
        ("invent twelve macros", ("identical_external_capacity_charge", "conservative_residual_common_charge", "physical_macro_count"), 12),
        ("unequal technology", ("identical_external_capacity_charge", "three_axis_identity", "technology_parameters_identical"), False),
        ("force access counts equal", ("identical_external_capacity_charge", "three_axis_identity", "actual_dynamic_access_counts_forced_identical"), True),
        ("double count parent", ("no_double_count", "logic_only_top_parent_macro_instances_allowed"), 9),
        ("remove exactly-once", ("no_double_count", "parent_macro_area_may_appear_in_exactly_one_of_logic_or_external"), False),
        ("call logic total", ("measurement_boundaries", "logic_only_dc", "may_be_called_total"), True),
        ("pretend model frozen", ("measurement_boundaries", "external_memory_model", "numeric_model_frozen_now"), True),
        ("promote RTL ratio", ("frozen_raw_cpu_opportunity", "rtl_speedup"), True),
        ("promote system ratio", ("frozen_raw_cpu_opportunity", "system_speedup"), True),
        ("promote PPA energy", ("frozen_raw_cpu_opportunity", "ppa_or_energy"), True),
        ("authorize EDA", ("authorization", "dc_pt_saif_ptpx_now"), True),
    )
    for label, path, replacement in mutation_paths:
        forged = copy.deepcopy(contract); target = forged
        for key in path[:-1]: target = target[key]
        target[path[-1]] = replacement
        reject(label, validate, forged)

    with tempfile.TemporaryDirectory(prefix="m1122c_pathc_attack_") as temporary:
        root = Path(temporary)
        review = root / "review.json"; review.write_text('{"status":"GOOD"}\n', encoding="utf-8")
        manifest = root / "SHA256SUMS"; manifest.write_text(
            sha(review) + "  review.json\n", encoding="utf-8")
        outer = root / "SHA256SUMS.seal.sha256"; outer.write_text(
            sha(manifest) + "  SHA256SUMS\n", encoding="utf-8")
        identity = (sha(review), sha(manifest), sha(outer))
        flat(root, identity, "GOOD", "temporary seal"); require(True, "legal temp seal")
        extra = root / "invented_external_model.json"; extra.write_text("{}\n", encoding="utf-8")
        reject("live extra model", flat, root, identity, "GOOD", "temporary seal")
        extra.unlink()
        real = root / "review.real"; review.rename(real); review.symlink_to(real)
        reject("live seal symlink", flat, root, identity, "GOOD", "temporary seal")

    require(len(attacks) == 22, "all 22 mutations rejected")
    output = {
        "schema": "m1122c_path_c_common_charge_author_static_check_v1",
        "status": "PASS_M1122C_PATH_C_SOURCE_AUTHOR_CHECK__M1123C_HAMMER_REQUIRED__NO_RTL_NO_EDA",
        "checks_passed": len(checks),
        "attacks_rejected": len(attacks),
        "attacks": attacks,
        "contract_sha256": CONTRACT_ID[0],
        "contract_outer_seal_file_sha256": CONTRACT_ID[2],
        "m1121c_outer_seal_file_sha256": M1121C_ID[2],
        "m1114_outer_seal_file_sha256": M1114_ID[2],
        "m1000_outer_seal_file_sha256": M1000_ID[2],
        "m1102_outer_seal_file_sha256": M1102_ID[2],
        "raw_cpu_ratio": 1.7591725401987818,
        "logic_only_dc": False,
        "external_numeric_model": False,
        "total_ppa": False,
        "rtl_or_eda_executed": False,
        "future_different_author_hammer_required": True,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
