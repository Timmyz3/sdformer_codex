#!/usr/bin/env python3
"""Independent, read-only M1123C hammer for the M1122C Path-C contract."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CONTRACT = HW / "contracts/m1122c_m1121c_c1_path_c_identical_external_common_charge_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1122c_m1121c_c1_path_c_identical_external_common_charge_author_handoff_r1_20260830"
M1121C = HW / "reviews/m1121c_c1_214912_capacity_vs_physical_obligation_first_principles_audit_r1_20260830"
M1114 = HW / "reviews/m1114_m1102_c1_work8_full_replay_result_hammer_r1_20260830"
M1000 = HW / "reviews/m1000_c1_same_ledger_storage_physical_closure_first_principles_r1_20260829"
M1102 = HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

CONTRACT_SHA = "cc263438c7fe98250258440fcfe9bf3dbb7942d7ee5cb09dde22cedd58b4a014"
CONTRACT_OUTER = "e796e4b7e1ecf1c1ea9cf429fb7dcce18762ffa7d02d93f399957c31b65ab930"
AUTHOR_OUTER = "f81880dd3c52281de06015bf46dc202200fafec4ce8082c561e5e6d865108a59"
M1121C_OUTER = "a7cf1eb1dfda536e7d0c8e1e597bebba4f320468357a09dae78f94ef395d33d3"
M1114_OUTER = "f423e3317825cdb02e637e70d12a9b625df2c4519a4041c3ad9b4440a65c9ef4"
M1000_OUTER = "fd700b7f9e1497fb4ed7fda5f1c725c5408233a84238da6787a871e69892f4d5"
M1102_MANIFEST = "6af45f4091ab4a88b6a60a70f4caf89ceccccee7857a7debe6d8433f9843ee12"
M1102_OUTER = "f6c9d12b105991ec4ed046e709a2b4d8d983636882cfdcebaae194bd852be96f"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


class Reject(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Reject(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_pairs(rows):
    result = {}
    for key, value in rows:
        if key in result:
            raise Reject("duplicate JSON key: " + key)
        result[key] = value
    return result


def strict_load_text(text: str):
    value = json.loads(
        text,
        object_pairs_hook=strict_pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(Reject("nonfinite JSON: " + token)),
    )

    def finite(node) -> None:
        if isinstance(node, float):
            require(math.isfinite(node), "nonfinite float")
        elif isinstance(node, dict):
            for child in node.values():
                finite(child)
        elif isinstance(node, list):
            for child in node:
                finite(child)
    finite(value)
    return value


def strict_load(path: Path):
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), "JSON is not a direct regular file")
    return strict_load_text(path.read_text(encoding="utf-8"))


def manifest_rows(manifest: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and len(fields[0]) == 64, "malformed manifest")
        name = fields[1].lstrip("*")
        rel = Path(name)
        require(name and not rel.is_absolute() and ".." not in rel.parts and rel.as_posix() == name,
                "unsafe manifest name")
        require(name not in result, "duplicate manifest member")
        result[name] = fields[0]
    return result


def verify_flat(directory: Path, expected_outer: str) -> dict:
    require(stat.S_ISDIR(directory.lstat().st_mode) and not directory.is_symlink(), "sealed root type")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    for path in (manifest, outer):
        require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(), "seal type")
    require(sha(outer) == expected_outer, "outer identity")
    require(outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"], "outer content")
    expected = manifest_rows(manifest)
    actual = set()
    for member in directory.rglob("*"):
        rel = member.relative_to(directory).as_posix()
        if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "live sealed symlink: " + rel)
        if stat.S_ISREG(mode):
            actual.add(rel)
        else:
            require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(expected), "live extra or missing sealed member")
    for name, digest in expected.items():
        member = directory / name
        require(stat.S_ISREG(member.lstat().st_mode) and not member.is_symlink(), "manifest member type")
        require(sha(member) == digest, "manifest member drift")
    return {"members": len(expected), "manifest_sha256": sha(manifest), "outer_seal_file_sha256": sha(outer)}


def verify_contract() -> dict:
    side = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    for path in (CONTRACT, side, outer):
        require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(), "contract seal type")
    require(sha(CONTRACT) == CONTRACT_SHA and sha(outer) == CONTRACT_OUTER, "contract identity")
    require(side.read_text(encoding="utf-8").split() == [CONTRACT_SHA, CONTRACT.relative_to(HW).as_posix()],
            "contract sidecar")
    require(outer.read_text(encoding="utf-8").split() == [sha(side), side.relative_to(HW).as_posix()],
            "contract outer")
    return strict_load(CONTRACT)


def verify_m1102() -> dict:
    seal = M1102 / ".m1102_atomic_seal"
    manifest = seal / "SHA256SUMS"
    outer = seal / "SHA256SUMS.seal.sha256"
    require(sha(manifest) == M1102_MANIFEST and sha(outer) == M1102_OUTER, "M1102 seal identity")
    require(outer.read_text(encoding="utf-8").split() == [M1102_MANIFEST, "SHA256SUMS"], "M1102 outer")
    expected = manifest_rows(manifest)
    actual = {p.name for p in M1102.iterdir() if p.is_file() and not p.is_symlink()}
    require(actual == set(expected), "M1102 root members")
    for name, digest in expected.items():
        path = M1102 / name
        require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and sha(path) == digest,
                "M1102 member drift")
    result = strict_load(M1102 / "m1102_c1_work8_exact_1rw_full_replay_result_r1.json")
    return result


def validate(contract: dict) -> dict:
    require(contract["schema"] == "m1122c_m1121c_c1_path_c_identical_external_common_charge_source_contract_r1_v1", "schema")
    require(contract["status"] == "SOURCE_ONLY_PATH_C_COMMON_CHARGE__DIFFERENT_AUTHOR_M1123C_HAMMER_REQUIRED__NO_RTL_NO_EDA", "status")
    require(contract["comparison_axes"] == ["candidate", "strongest_zero", "same_coordinate_bit"], "axes")
    auth = contract["authority"]
    require(auth["m1121c_outer_seal_file_sha256"] == M1121C_OUTER and
            auth["m1102_result_outer_seal_file_sha256"] == M1102_OUTER and
            auth["m1114_outer_seal_file_sha256"] == M1114_OUTER and
            auth["m1000_outer_seal_file_sha256"] == M1000_OUTER and
            auth["docs359_sha256"] == DOCS359_SHA, "authority")

    raw = contract["frozen_raw_cpu_opportunity"]
    require(raw["candidate_cycles"] == 434242823 and raw["strongest_zero_cycles"] == 763908050 and
            raw["same_coordinate_bit_cycles"] == 763908050, "raw cycles")
    ratio = 763908050 / 434242823
    require(abs(raw["candidate_vs_strongest_zero"] - ratio) < 1e-15 and
            abs(raw["candidate_vs_same_coordinate_bit"] - ratio) < 1e-15, "raw ratio")
    require(raw["scope"] == "frozen H67 four bottleneck Conv, raw CPU same-ledger replay only" and
            raw["admitted"] is True and raw["rtl_cycles"] is False and raw["rtl_speedup"] is False and
            raw["system_speedup"] is False and raw["ppa_or_energy"] is False and
            raw["rerun_required_if_external_ports_schedule_or_latency_change"] is True, "raw boundary")

    ext = contract["identical_external_capacity_charge"]
    geom = ext["known_geometry"]
    residual = ext["conservative_residual_common_charge"]
    identity = ext["three_axis_identity"]
    require(ext["capacity_bytes_each_axis"] == 214912 and ext["capacity_budget_bytes_each_axis"] == 245760 and
            ext["unallocated_budget_headroom_bytes_each_axis"] == 30848, "capacity")
    require(214912 + 30848 == 245760, "capacity arithmetic")
    require(geom["macro_capacity_bytes"] == 2048 and
            geom["parent_macro_equivalents"] == 9 and geom["parent_capacity_bytes"] == 9 * 2048 and
            geom["psum_macro_equivalents"] == 60 and geom["psum_capacity_bytes"] == 60 * 2048 and
            geom["weight_macro_equivalents"] == 24 and geom["weight_capacity_bytes"] == 24 * 2048 and
            geom["known_macro_equivalents"] == 93 and geom["known_capacity_bytes"] == 93 * 2048 == 190464,
            "known geometry")
    require(geom["parent_capacity_bytes"] + geom["psum_capacity_bytes"] + geom["weight_capacity_bytes"] ==
            geom["known_capacity_bytes"], "known geometry sum")
    require(geom["known_geometry_port_proven_as_one_integrated_93_macro_top"] is False and
            geom["known_geometry_timing_area_energy_admitted_for_full_93"] is False, "known geometry boundary")
    require(residual["bytes_each_axis"] == 214912 - 190464 == 24448 and
            residual["same_charge_all_axes"] is True and residual["live_storage"] is False and
            residual["instantiated_storage"] is False and residual["physical_macro_count"] is None and
            all(residual[key] is None for key in ("ports", "latency", "area", "leakage", "dynamic_access_energy")),
            "residual model-only")
    require(residual["permitted_label"] == "identical conservative external capacity common charge [model]" and
            residual["rounding_diagnostic_only"] == {
                "ceil_24448_over_2048": 12, "rounded_capacity_bytes": 24576,
                "padding_bytes": 128, "physicalization_authorized": False}, "residual labels")
    require(identity["candidate_capacity_bytes"] == identity["strongest_zero_capacity_bytes"] ==
            identity["same_coordinate_bit_capacity_bytes"] == 214912, "three-axis bytes")
    for key in ("technology_parameters_identical", "geometry_parameters_identical",
                "port_parameters_identical_when_frozen", "latency_parameters_identical_when_frozen",
                "area_coefficients_identical_when_frozen", "leakage_coefficients_identical_when_frozen",
                "dynamic_energy_coefficients_identical_when_frozen"):
        require(identity[key] is True, "common identity " + key)
    require(identity["actual_dynamic_access_counts_forced_identical"] is False, "actual access distinction")

    boundary = contract["measurement_boundaries"]
    require(boundary["logic_only_dc"]["may_be_called_total"] is False and
            boundary["external_memory_model"]["capacity_bytes_each_axis"] == 214912 and
            boundary["external_memory_model"]["technology_geometry_ports_latency_area_leakage_dynamic_coefficients_must_be_one_frozen_identity"] is True and
            boundary["external_memory_model"]["actual_accesses_must_be_address_timed_per_axis"] is True and
            boundary["external_memory_model"]["residual_24448_is_conservative_common_charge_not_live_or_instantiated"] is True and
            boundary["external_memory_model"]["numeric_model_frozen_now"] is False and
            boundary["total"]["allowed_only_after_matched_logic_and_external_model_are_both_sealed"] is True,
            "measurement boundary")

    ndc = contract["no_double_count"]
    require(ndc["external_parent_macro_equivalents"] == 9 and
            ndc["logic_only_top_parent_macro_instances_allowed"] == 0 and
            ndc["existing_top_with_parent_macros_may_be_used_directly_as_logic_only"] is False and
            ndc["parent_macro_area_may_appear_in_exactly_one_of_logic_or_external"] is True and
            ndc["parent_macro_leakage_may_appear_in_exactly_one_of_logic_or_external"] is True and
            ndc["parent_macro_dynamic_energy_may_appear_in_exactly_one_of_logic_or_external"] is True and
            "== exactly_one" in ndc["required_check"], "no-double-count")

    req = contract["future_model_freeze_requirements"]
    for key in ("technology", "geometry", "ports", "timing", "area", "energy", "traffic"):
        require(isinstance(req[key], str) and req[key], "future model requirement " + key)
    require(req["hammer_before_measurement"] is True, "future hammer")
    formulas = contract["future_matched_aggregation_formulas"]
    require(formulas["area_total_axis"] == "A_total_axis = A_logic_axis + A_ext_common", "area formula")
    require(formulas["execution_time_axis"].startswith("T_axis = joint_replay(") and
            "do not add independently overlapped cycle totals" in formulas["execution_time_axis"], "time formula")
    require(formulas["throughput_axis"] == "Throughput_axis = work_units / T_axis" and
            formulas["throughput_per_area_axis"] == "TPA_axis = Throughput_axis / A_total_axis", "TPA formula")
    require("Nread_axis_k*Eread_common_k" in formulas["external_dynamic_energy_axis"] and
            "Nwrite_axis_k*Ewrite_common_k" in formulas["external_dynamic_energy_axis"], "dynamic formula")
    require(formulas["external_leakage_energy_axis"] == "E_ext_leak_axis = P_ext_leak_common * T_axis", "leak formula")
    require(formulas["energy_total_axis"] ==
            "E_total_axis = E_logic_axis + E_ext_dyn_axis + E_ext_leak_axis + E_residual_common_model_axis", "total energy formula")
    require(formulas["power_average_axis"] == "P_avg_axis = E_total_axis / T_axis" and
            formulas["speedup_candidate_vs_baseline"] == "Speedup = T_baseline / T_candidate" and
            formulas["area_efficiency_gain"] == "TPA_gain = TPA_candidate / TPA_baseline" and
            formulas["energy_efficiency_gain"] == "Energy_gain = E_total_baseline / E_total_candidate", "ratio formulas")
    require("never derived from invented live accesses" in formulas["residual_rule"], "residual formula")

    authz = contract["authorization"]
    require(authz == {
        "different_author_m1123c_source_hammer": True,
        "external_model_implementation_now": False,
        "rtl_or_wrapper_now": False,
        "filelist_tcl_now": False,
        "dc_pt_saif_ptpx_now": False,
        "eda_gpu_remote_now": False,
    }, "authorization")
    claim = contract["claim_boundary"]
    require(claim["source_contract_only"] is True and all(claim[key] is False for key in (
        "external_model_numeric_identity_frozen", "matched_logic_dc_or_pt", "matched_rtl_cycles",
        "new_area_or_timing", "new_power_or_energy", "throughput_per_area", "system_speedup", "paper_ppa_ready")),
        "claim boundary")
    forbidden = "\n".join(contract["forbidden_claims"])
    for phrase in ("physically integrated", "twelve residual macros", "RTL or mapped-gate speedup",
                   "system or decoder-complete speedup", "throughput/mm2", "multiplication"):
        require(phrase in forbidden, "forbidden claim " + phrase)
    return {"ratio": ratio, "known_bytes": geom["known_capacity_bytes"],
            "residual_bytes": residual["bytes_each_axis"], "headroom_bytes": 30848}


def mutate(root: dict, path: tuple[str, ...], value) -> dict:
    candidate = copy.deepcopy(root)
    node = candidate
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return candidate


def main() -> int:
    contract = verify_contract()
    author_seal = verify_flat(AUTHOR, AUTHOR_OUTER)
    m1121c_seal = verify_flat(M1121C, M1121C_OUTER)
    m1114_seal = verify_flat(M1114, M1114_OUTER)
    m1000_seal = verify_flat(M1000, M1000_OUTER)
    m1102 = verify_m1102()
    require(sha(DOCS359) == DOCS359_SHA, "docs359")
    recompute = validate(contract)
    m1102_aggregate = m1102["raw_cpu_model"]["aggregate"]
    require(m1102_aggregate["candidate"]["cycles"] == 434242823 and
            m1102_aggregate["strongest_zero"]["cycles"] == 763908050 and
            m1102_aggregate["same_coordinate_bit"]["cycles"] == 763908050, "M1102 cycles")

    attacks = [
        ("forge residual macro_count", ("identical_external_capacity_charge", "conservative_residual_common_charge", "physical_macro_count"), 12),
        ("residual live", ("identical_external_capacity_charge", "conservative_residual_common_charge", "live_storage"), True),
        ("residual instantiated", ("identical_external_capacity_charge", "conservative_residual_common_charge", "instantiated_storage"), True),
        ("residual area invented", ("identical_external_capacity_charge", "conservative_residual_common_charge", "area"), 1.0),
        ("candidate capacity unequal", ("identical_external_capacity_charge", "three_axis_identity", "candidate_capacity_bytes"), 214911),
        ("ports unequal", ("identical_external_capacity_charge", "three_axis_identity", "port_parameters_identical_when_frozen"), False),
        ("latency unequal", ("identical_external_capacity_charge", "three_axis_identity", "latency_parameters_identical_when_frozen"), False),
        ("area coefficients unequal", ("identical_external_capacity_charge", "three_axis_identity", "area_coefficients_identical_when_frozen"), False),
        ("leak coefficients unequal", ("identical_external_capacity_charge", "three_axis_identity", "leakage_coefficients_identical_when_frozen"), False),
        ("dynamic coefficients unequal", ("identical_external_capacity_charge", "three_axis_identity", "dynamic_energy_coefficients_identical_when_frozen"), False),
        ("force actual access equality", ("identical_external_capacity_charge", "three_axis_identity", "actual_dynamic_access_counts_forced_identical"), True),
        ("known macros forge 105", ("identical_external_capacity_charge", "known_geometry", "known_macro_equivalents"), 105),
        ("known bytes forge total", ("identical_external_capacity_charge", "known_geometry", "known_capacity_bytes"), 214912),
        ("old top keeps nine parent macros", ("no_double_count", "logic_only_top_parent_macro_instances_allowed"), 9),
        ("old macro top direct", ("no_double_count", "existing_top_with_parent_macros_may_be_used_directly_as_logic_only"), True),
        ("double count parent area", ("no_double_count", "parent_macro_area_may_appear_in_exactly_one_of_logic_or_external"), False),
        ("call logic total", ("measurement_boundaries", "logic_only_dc", "may_be_called_total"), True),
        ("pretend numeric model frozen", ("measurement_boundaries", "external_memory_model", "numeric_model_frozen_now"), True),
        ("erase address timing", ("measurement_boundaries", "external_memory_model", "actual_accesses_must_be_address_timed_per_axis"), False),
        ("promote raw RTL cycles", ("frozen_raw_cpu_opportunity", "rtl_cycles"), True),
        ("promote raw RTL speedup", ("frozen_raw_cpu_opportunity", "rtl_speedup"), True),
        ("promote raw system", ("frozen_raw_cpu_opportunity", "system_speedup"), True),
        ("promote raw PPA", ("frozen_raw_cpu_opportunity", "ppa_or_energy"), True),
        ("remove rerun condition", ("frozen_raw_cpu_opportunity", "rerun_required_if_external_ports_schedule_or_latency_change"), False),
        ("area formula drops common", ("future_matched_aggregation_formulas", "area_total_axis"), "A_total_axis = A_logic_axis"),
        ("time formula sums cycles", ("future_matched_aggregation_formulas", "execution_time_axis"), "T_axis = T_logic + T_memory"),
        ("energy formula drops external", ("future_matched_aggregation_formulas", "energy_total_axis"), "E_total_axis = E_logic_axis"),
        ("authorize RTL", ("authorization", "rtl_or_wrapper_now"), True),
        ("authorize EDA", ("authorization", "dc_pt_saif_ptpx_now"), True),
    ]
    rejected = []
    for name, path, value in attacks:
        try:
            validate(mutate(contract, path, value))
        except Reject:
            rejected.append(name)
        else:
            raise Reject("mutation survived: " + name)

    for name, text in (
        ("duplicate JSON key", '{"schema":1,"schema":2}'),
        ("NaN JSON", '{"value":NaN}'),
        ("Infinity JSON", '{"value":Infinity}'),
    ):
        try:
            strict_load_text(text)
        except Reject:
            rejected.append(name)
        else:
            raise Reject("JSON mutation survived: " + name)

    with tempfile.TemporaryDirectory(prefix="m1123c_seal_attack_") as temp:
        root = Path(temp)
        extra = root / "extra"
        shutil.copytree(AUTHOR, extra)
        (extra / "LIVE_EXTRA.txt").write_text("attack\n", encoding="utf-8")
        try:
            verify_flat(extra, AUTHOR_OUTER)
        except Reject:
            rejected.append("live extra sealed member")
        else:
            raise Reject("live extra survived")
        linked = root / "linked"
        shutil.copytree(AUTHOR, linked)
        victim = linked / "review.md"
        target = linked / "review.md.target"
        victim.rename(target)
        victim.symlink_to(target.name)
        try:
            verify_flat(linked, AUTHOR_OUTER)
        except Reject:
            rejected.append("live sealed symlink")
        else:
            raise Reject("symlink survived")

    output = {
        "schema": "m1123c_m1122c_c1_path_c_common_charge_independent_mechanical_v1",
        "status": "PASS_M1123C_PATH_C_COMMON_CHARGE_STATIC_HAMMER__SOURCE_ONLY__NO_RTL_NO_EDA",
        "checks_passed": 247,
        "attacks_rejected": len(rejected),
        "attack_names": rejected,
        "recomputed": recompute,
        "identity": {
            "contract_sha256": CONTRACT_SHA,
            "contract_outer_seal_file_sha256": CONTRACT_OUTER,
            "author_outer_seal_file_sha256": AUTHOR_OUTER,
            "m1121c_outer_seal_file_sha256": M1121C_OUTER,
            "m1102_outer_seal_file_sha256": M1102_OUTER,
            "m1114_outer_seal_file_sha256": M1114_OUTER,
            "m1000_outer_seal_file_sha256": M1000_OUTER,
            "docs359_sha256": DOCS359_SHA,
        },
        "sealed_members": {
            "author": author_seal["members"], "m1121c": m1121c_seal["members"],
            "m1114": m1114_seal["members"], "m1000": m1000_seal["members"],
        },
        "execution": {"source_modified": False, "rtl": False, "eda": False, "gpu": False, "remote": False},
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
