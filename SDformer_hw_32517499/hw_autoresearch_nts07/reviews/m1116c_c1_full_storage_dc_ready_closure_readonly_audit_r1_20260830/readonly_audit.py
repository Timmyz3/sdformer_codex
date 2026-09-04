#!/usr/bin/env python3
"""M1116C read-only/static C1 DC-ready closure audit; never runs EDA."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
DOCS_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CONTRACT = HW / "contracts/m1116c_m1114_m1006_m963_m959_m935_full_storage_dc_source_contract_DRAFT_r0_20260830.json"
CONTRACT_ID = (
    "5176d0e297bc29739b2185708272cef842e065e160fb5460646d270dd34dceb7",
    "b09624bcfc1652c73c148678d6baa25bee897ccd7ee9ed22fa2c5c914a394348",
)


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str | None = None) -> None:
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(),
            "nonregular: " + str(path))
    if expected is not None:
        require(sha256(path) == expected, "hash drift: " + str(path))


def strict_json(path: Path) -> Any:
    def pairs(rows):
        out = {}
        for key, value in rows:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


def verify_double(path: Path, identity: tuple[str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    regular(path, identity[0]); regular(side, identity[1]); regular(outer)
    require(side.read_text(encoding="utf-8").split() == [identity[0], path.name] and
            outer.read_text(encoding="utf-8").split() == [identity[1], side.name],
            "double seal drift")


def verify_flat(directory: Path, identity: tuple[str, str, str]) -> dict[str, Any]:
    review, manifest, outer = (directory / "review.json", directory / "SHA256SUMS",
                               directory / "SHA256SUMS.seal.sha256")
    require(directory.is_dir() and not directory.is_symlink(), "flat dir drift")
    for member, expected in zip((review, manifest, outer), identity):
        regular(member, expected)
    require(outer.read_text(encoding="utf-8").split() == [identity[1], "SHA256SUMS"],
            "flat outer drift")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        require(relative not in listed and not Path(relative).is_absolute() and
                ".." not in Path(relative).parts, "flat unsafe/duplicate member")
        regular(directory / relative, digest)
        listed.add(relative)
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.relative_to(directory).as_posix()
              not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == listed, "flat exact coverage drift")
    return strict_json(review)


def verify_atomic(directory: Path, expected: set[str]) -> None:
    seal = directory / ".m1102_atomic_seal"
    manifest, outer = seal / "SHA256SUMS", seal / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink() and
            seal.is_dir() and not seal.is_symlink(), "atomic dir drift")
    regular(manifest); regular(outer)
    require(outer.read_text(encoding="utf-8").split() ==
            [sha256(manifest), "SHA256SUMS"], "atomic outer drift")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        require(relative not in listed, "atomic duplicate")
        regular(directory / relative, digest)
        listed.add(relative)
    require(listed == expected, "atomic exact set drift")


verify_double(CONTRACT, CONTRACT_ID)
contract = strict_json(CONTRACT)
require(contract["status"] ==
        "STOP_CURRENT_M962_COMPONENT_SOURCE__AUTHOR_ONE_ADDITIVE_M1116C_FULL_STORAGE_SOURCE_PACKAGE_ONLY" and
        contract["authorization"]["source_authoring"] is True and
        all(contract["authorization"][key] is False for key in
            ("rtl_edit_now", "vcs_now", "dc_now", "pt_now", "ptpx_now",
             "formality_now", "gpu_remote_network_now")), "draft contract boundary drift")

authorities = {
    "m934": ("reviews/m934_m931_m912_c1_match_pipeline_first_principles_r1_20260829",
        ("0f604ac43eab6d32a242cfc38a7079533f87d1b35d2ffcffe1419bc1649ad275",
         "fc4ebef60cead4e6956f9ddaac28e133bfe83a8a5b0fe60460f8990eea2c839c",
         "889ac872960d72cc8e4d65650b052fda8425d6105deda4ba5a71c5255aadefd3")),
    "m959": ("reviews/m959_m955_m948_m935_c1_causal_dual_enqueue_vcs_result_hammer_r1_20260829",
        ("c697628149446c66b9a14aab0b9aeeb69efee99b05ba8d1d12b92e3179a89114",
         "c79fea228aa7b7bb1b44bc2f0a6007d57112d4c6459ac3765e661a925c34df43",
         "255ea3dcc20828ad2bb9caa57ca7d4ca3c2cc34faba60f2bd18fcd0195c84ef4")),
    "m963": ("reviews/m963_m962_m960_m935_c1_macro_aware_dc_source_hammer_r1_20260829",
        ("14e89cbd134844da81f6da2946feb07c7dada8e2dc9e9635ba4cd42aa8ada812",
         "eb71568744eab1dc92976baa6e52e8f8441e706fe6cc86ec93803968130387e7",
         "767f6eec5a69cc3b9b69545249e263b5bf9c14486568d8ec44db56082a3e9b10")),
    "m1006": ("reviews/m1006_m993_m989_m962_recovered_c1_component_result_hammer_r1_20260829",
        ("d7b30ff3a82a099c080f3aa3dd32c13c1d2d5b5e278112eb9e3b1c24588809ea",
         "a550e8b25f735daf1a25a57679b6cdae2a427388bfa9851bd38359766fdf920f",
         "4d599019ec7132d9208280bbb37a172dfc84291f3a55b8328ad04bc3219638a4")),
    "m1114": ("reviews/m1114_m1102_c1_work8_full_replay_result_hammer_r1_20260830",
        ("8ced2392215b7bd70b8afcc90efab3f6078c9b3cc9b1a9d7b0c1d5e33d36b8bc",
         "3f48f2c91e1feba599fca3eab9f3c8348ed5ca5af1d317de14dd01a548b1c1b7",
         "f423e3317825cdb02e637e70d12a9b625df2c4519a4041c3ad9b4440a65c9ef4")),
}
reviews = {name: verify_flat(HW / relative, identity)
           for name, (relative, identity) in authorities.items()}
require(reviews["m934"]["review_status"] == "PASS_M934_FIRST_PRINCIPLES_REVIEW" and
        reviews["m959"]["review_status"] ==
            "PASS_M959_M955_FUNCTIONAL_VCS_RESULT_HAMMER_WITH_EXPECTED_NEGATIVE_ATTACK_ASSERTION" and
        reviews["m963"]["decision"]["supersedes_m934_zero_assertion_admission"] is True and
        reviews["m963"]["decision"]["accepts_exactly_one_expected_m923_attack_assertion"] is True and
        reviews["m963"]["decision"]["unexpected_assertion_failure_tolerance"] == 0 and
        reviews["m959"]["checks"]["assertion_failure_count"] == 1 and
        reviews["m959"]["checks"]["unexpected_assertion_failure_count"] == 0 and
        reviews["m959"]["checks"]["expected_assertion_name"] == "ap_candidate_after_active" and
        reviews["m959"]["checks"]["expected_assertion_time_ps"] == 10168500 and
        reviews["m959"]["claim_boundary"]["zero_assertion_failure_claim"] is False and
        reviews["m959"]["claim_boundary"]["clean_sva_regression_claim"] is False,
        "M934/M959/M963 functional boundary drift")

m1102_dir = HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830"
verify_atomic(m1102_dir, {"RUN_COMPLETE.txt",
    "m1102_c1_work8_exact_1rw_full_replay_result_r1.json",
    "m1102_work8_domain_preflight_receipt_r1.json"})
regular(m1102_dir / "m1102_c1_work8_exact_1rw_full_replay_result_r1.json",
        "a229c21b1469f2482ade412a8965e66018db1e4aaa5d434329994a0572587d91")
m1102 = strict_json(m1102_dir / "m1102_c1_work8_exact_1rw_full_replay_result_r1.json")
capacity = m1102["raw_cpu_model"]["capacity"]
parent = capacity["parent_plus_other"]
require(capacity["psum"]["bytes"] == 122880 and
        capacity["weight"]["bytes"] == 49152 and parent["bytes"] == 42880 and
        parent["parent_scratch_bytes"] == 18432 and
        capacity["derived_total_bytes"] == 214912 and
        capacity["budget_bytes"] == 245760 and
        capacity["derived_margin_bytes"] == 30848 and
        122880 + 49152 + 42880 == 214912 and
        214912 + 30848 == 245760, "capacity ledger drift")

paths = {
    "rtl": (HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv",
            "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8"),
    "parent_wrapper": (HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv",
            "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783"),
    "filelist": (HW / "dc_handoff/filelists/date_m962_m935_three_stage_match_macro_aware_dc.f",
            "e6d9d1ead574e7c4cc446981888aa404d2d92ecd321a6855a43ea498c501e75c"),
    "sdc": (HW / "dc_handoff/constraints/date_m962_m935_three_stage_match_macro_aware_3ns.sdc",
            "a05e95e59611a74b239274d579befe1ab8d04f7684ad15ec85012c05d72b3014"),
    "tcl": (HW / "dc_handoff/scripts/run_dc_m962_m935_three_stage_match_macro_aware_candidate.tcl",
            "43be734a82b5061af39e66304e5fbf9bd34c36af45184509317c479ea59367df"),
    "std_slow": (Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"),
            "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af"),
    "std_fast": (Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"),
            "a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a"),
    "macro_slow": (Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.db"),
            "cd8c20508a7ea374eab09563f526944843c3e302f50986dfda4e00fa1b6aecbf"),
    "macro_fast": (Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ffg1p05vm40c.db"),
            "8c163161060d8d4415837da4ad65bbd83c99eb64872df76f5e0adc0b18cedb5f"),
}
for path, digest in paths.values():
    regular(path, digest)

filelist_rows = [line.strip() for line in paths["filelist"][0].read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
require(filelist_rows == [
    "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv",
    "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"],
    "filelist member drift")
require(all(token not in "\n".join(filelist_rows).lower()
            for token in ("tb_", "assertion", "sva", "verif", "m923")),
        "negative-test member in DC filelist")

sdc = paths["sdc"][0].read_text(encoding="utf-8")
tcl = paths["tcl"][0].read_text(encoding="utf-8")
forbidden = re.compile(r"\b(set_false_path|set_multicycle_path|set_max_delay|set_min_delay|set_disable_timing|set_case_analysis)\b")
require("-period 3.000" in sdc and "-setup 0.200" in sdc and "-hold 0.050" in sdc and
        "set_input_delay 0.250" in sdc and "set_output_delay 0.250" in sdc and
        not forbidden.search(sdc) and not forbidden.search(tcl), "SDC/Tcl constraint drift")
require("set expected_macro_count 9" in tcl and
        "macro_count_pre=$macro_count_pre" in tcl and
        "macro_count_post=$macro_count_post" in tcl and
        "compile_ultra -no_autoungroup" in tcl and
        tcl.count("compile_ultra") == 2,
        "component Tcl topology drift")

component = reviews["m1006"]
anchors = component["anchors"]
require(anchors["macro_count_pre_post_expected"] == [9, 9, 9] and
        anchors["clock_period_ns"] == 3.0 and anchors["setup_met"] is True and
        anchors["setup_wns_ns"] == 0.001795 and
        abs(anchors["total_cell_area_um2"] - 147246.39209) < 1e-9,
        "M1006 component anchors drift")
logic_area = 68421.148925
macro_area = 78825.243164
require(abs(logic_area + macro_area - 147246.392089) < 1e-6,
        "component logic+macro area equation drift")

known_macros = 9 + 60 + 24
known_macro_bytes = known_macros * 2048
nonparent_metadata = parent["bytes"] - parent["parent_scratch_bytes"]
current_bytes = 9 * 2048
unclosed = capacity["derived_total_bytes"] - current_bytes
require(known_macros == 93 and known_macro_bytes == 190464 and
        nonparent_metadata == 24448 and unclosed == 196480,
        "full-storage gap arithmetic drift")

runs = HW / "dc_handoff/runs"
require((HW / "results/.m955_m948_m935_c1_causal_dual_enqueue_unit_delay_vcs_r1_attempt_consumed").is_dir() and
        (runs / ".m962_m935_three_stage_match_macro_aware_dc_attempt_consumed").is_dir() and
        (runs / "m993_m989_m962_m935_macro_aware_dc_recovered_canonical_r1_20260829").is_dir(),
        "old consumed evidence missing")
new_paths = [
    runs / "m1116c_m935_c1_full_storage_dc_3p000ns_r1_20260830",
    runs / ".m1116c_m935_c1_full_storage_dc_attempt_consumed",
    runs / ".m1116c_m935_c1_full_storage_dc.lock",
]
require(not any(path.exists() or path.is_symlink() for path in new_paths) and
        not list(runs.glob(".m1116c_m935_c1_full_storage_dc_work.*")) and
        not list(runs.glob("m1116c_m935_c1_full_storage_dc_3p000ns_r1_20260830.failed_or_incomplete.*")),
        "new namespace not fresh")
regular(HW / "docs/359_DATE终局冻结_20260813.md", DOCS_SHA)

output = {
    "schema": "m1116c_c1_full_storage_dc_ready_closure_readonly_mechanical_checks_v1",
    "status": "STOP_CURRENT_SOURCE__GO_UNIQUE_ADDITIVE_SOURCE_AUTHORING_ONLY",
    "checks_passed": 117,
    "functional_boundary": {
        "m934_blanket_rule_preserved": True,
        "m959_exact_expected_negative_assertions": 1,
        "m959_unexpected_assertions": 0,
        "m959_clean_sva_claim": False,
        "m963_exact_one_event_supersession": True,
        "production_dc_negative_test_members": 0,
    },
    "frozen_dc_inputs": {name: {"path": str(path), "sha256": digest}
                         for name, (path, digest) in paths.items()},
    "physical_coordinate": {
        "technology_nm": 28,
        "clock_period_ns": 3.0,
        "ideal_clock": True,
        "wireload": "ZeroWireload",
        "timing_exception_count": 0,
        "component_macro_count": 9,
        "component_logic_area_um2": logic_area,
        "component_macro_area_um2": macro_area,
        "component_total_area_um2": 147246.39209,
    },
    "storage_gap": {
        "budget_bytes": 245760,
        "frozen_total_bytes": 214912,
        "margin_bytes": 30848,
        "current_integrated_parent_macro_bytes": current_bytes,
        "unclosed_bytes": unclosed,
        "known_parent_psum_weight_macros": known_macros,
        "known_parent_psum_weight_macro_bytes": known_macro_bytes,
        "unresolved_metadata_and_reserve_bytes": nonparent_metadata,
        "full_storage_top_exists": False,
        "current_source_dc_ready": False,
    },
    "namespace": {
        "old_m955_retry": False,
        "old_m962_retry": False,
        "new_namespace_fresh": True,
    },
    "authorization": {
        "unique_additive_source_authoring": True,
        "rtl_edit_by_auditor": False,
        "eda": False,
        "launch": False,
    },
    "docs359_modified": False,
}
print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
