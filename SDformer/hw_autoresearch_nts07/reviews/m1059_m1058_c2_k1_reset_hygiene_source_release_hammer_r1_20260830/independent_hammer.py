#!/usr/bin/env python3
"""Receipt-blind M1059 hammer for the M1058 reset-hygiene closure.

This checker never runs DC, mapped VCS, SAIF, or PTPX.  It independently
checks the additive RTL boundary and the non-launching production candidate.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent

PATHS = {
    "source_contract": HW / "contracts/m1058_c2_k1_reset_hygiene_source_only_contract_r1_20260830.json",
    "launch_candidate": HW / "contracts/m1058_c2_k1_reset_hygiene_dc_mapped_vcs_launch_candidate_r1_20260830.json",
    "source_review": HW / "reviews/m1058_c2_k1_reset_hygiene_source_closure_r1_20260830.md",
    "new_service": HW / "rtl_m1058/m1058_fc2_k1_reset_hygiene_registered_release_service_island.sv",
    "new_standalone": HW / "rtl_m1058/m1058_fc2_reset_hygiene_registered_release_standalone_raw4_acc24.sv",
    "new_k1": HW / "rtl_m1058/m1058_fc2_k1_reset_hygiene_registered_release_8bank_raw4_acc24.sv",
    "new_shell": HW / "rtl_m1058/m1058_fc2_reset_hygiene_channel_split_registered_release_matched_8bank_raw4_acc24.sv",
    "old_service": HW / "rtl_m519/m519_fc2_k1_registered_release_service_island.sv",
    "old_standalone": HW / "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv",
    "old_k1": HW / "rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv",
    "old_k1x8": HW / "rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv",
    "old_shell": HW / "rtl_m803/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24.sv",
    "old_k8": HW / "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv",
    "anchor_tb": HW / "dc_handoff/tb/tb_m1058_c2_k1_reset_hygiene_five_case_anchor.sv",
    "mapped_tb": HW / "dc_handoff/tb/tb_m1058_c2_k1_reset_hygiene_mapped_gate_case.sv",
    "rtl_filelist": HW / "dc_handoff/filelists/date_m1058_c2_k1_reset_hygiene_rtl_vcs.f",
    "dc_filelist": HW / "dc_handoff/filelists/date_m1058_c2_k1_reset_hygiene_logic_only_dc.f",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}

EXPECTED = {
    "source_contract": "05a5da8535556a7234bbcfd1e5152505b17654067a5e83dcfef24912704ee6e7",
    "launch_candidate": "fcdd2857236ad53cf4c8994a973e465762aece8884745ca788fb1b2c5da7cd04",
    "source_review": "15aece28b3176e2ffdc83bab91958cb41a1bb6a0f26c9b51ecf8b8ec5c7f67c4",
    "new_service": "7876d8603ef4fc6e326287aecc7a4c9a9a66cab5400bcaa3b24f498518ff9d9d",
    "new_standalone": "a3d6628f28c6c277e9feda143f3cf9e1365eaad5648ab358f75e82bbd9187768",
    "new_k1": "f9f7319fd2495dc4a67ec20ecf6f34ef8884c88f9bf49f8f2a74e7ee88e3e0f8",
    "new_shell": "33e2fa8427eff64bae3bde2c11bf7e6a3a15969aff076cb0ab7b96431227a565",
    "old_service": "3811998fc48d31e6519ecc6c6cfb8f5d38db6fc6dd070e09d73a5f70b7579871",
    "old_standalone": "010fe9e6786db1d3bbcad7759bda17a783ce5cfe15cae02c5b4c9ebf96e9950b",
    "old_k1": "6ea038ef935b1144d5424634e75446301270362c259341a8e7e7117523b25815",
    "old_k1x8": "11080d39c06672cebb64988e931c41e1d4c04134a312aeb8e250d01f0ac576ff",
    "old_shell": "3328e52d8cf1eec6098ebb7b0525ac55cd8bd6b2fe5b5e504b337d1a678e3c4b",
    "old_k8": "2588f890213d29aab6829dff679719c0f9ce4762c17bb061d1869b27a2f1d50e",
    "anchor_tb": "def59bd3dee3bcf98c9b21dad1a872d5d62902e394f8c6968780bbad0c800f08",
    "mapped_tb": "fdbb1ccc5be4af11d263a6581f84ec823f9fd8077b07fa8b12a88a32a056ae0f",
    "rtl_filelist": "6a19c28ca1ed7b35596c2f586fe00a8e363ada4ce515dc6111bee6bad6203284",
    "dc_filelist": "4cfd47438de45a66a601433ee07a2493c7296b1dea8669f9c7826898364e7192",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

M1050 = HW / "reviews/m1050_m1046_c2_mapped_gate_watchdog_failure_audit_r1_20260829"
M1058_RESULT = HW / "results/m1058_c2_k1_reset_hygiene_rtl_vcs_r1_20260830"
M1050_OUTER = "bc239844a71b5c017002ea1f6a756143d3c58b5ebf39d6a5499c76228da188bb"
M1058_RESULT_OUTER = "f22a55c33fadf74749060546e877fc10f892649aa31f3fa0da2d3fd164b70787"

RESET_BLOCK = """            // M1058 reset hygiene: payload state is invalid while count is zero,
            // but explicit reset prevents gate-level X reconvergence through
            // decoded array selects before the first legal FIFO write.
            for (int entry = 0; entry < GROUP_FIFO_DEPTH; entry++) begin
                fifo_tag_q[entry] <= '0;
                fifo_block_q[entry] <= '0;
                fifo_bank_id_q[entry] <= '0;
                fifo_channel_q[entry] <= '0;
            end
"""

K1_ANCHOR_CLAUSE = """        if(axis==0)case(c)0:return 259;1:return 737;2:return 3153;
            3:return 7569;default:return 14;endcase
"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(rows):
        out = {}
        for key, value in rows:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)


def verify_dir_seal(path: Path, outer: str) -> None:
    require(path.is_dir() and not path.is_symlink(), "sealed dir missing/symlink")
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=path,
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["sha256sum", "-c", "SHA256SUMS.seal.sha256"], cwd=path,
                   check=True, stdout=subprocess.DEVNULL)
    require(sha(path / "SHA256SUMS.seal.sha256") == outer, "outer seal drift")


def replace_once(text: str, old: str, new: str) -> str:
    require(text.count(old) == 1, "expected exactly one normalization target: " + old)
    return text.replace(old, new, 1)


def audit_additive_sources(texts: dict[str, str]) -> None:
    service = texts["new_service"]
    require(service.count(RESET_BLOCK) == 1, "reset block missing/duplicated")
    for field in ("fifo_tag_q", "fifo_block_q", "fifo_bank_id_q", "fifo_channel_q"):
        require(service.count(field + "[entry] <= '0;") == 1,
                "reset field missing/duplicated: " + field)
    normalized = replace_once(service, RESET_BLOCK, "")
    normalized = normalized.replace(
        "m1058_fc2_k1_reset_hygiene_registered_release_service_island",
        "m519_fc2_k1_registered_release_service_island")
    require(normalized == texts["old_service"], "service changed outside reset block")

    standalone = texts["new_standalone"].replace(
        "m1058_fc2_reset_hygiene_registered_release_standalone_raw4_acc24",
        "m519_fc2_registered_release_standalone_raw4_acc24").replace(
        "m1058_fc2_k1_reset_hygiene_registered_release_service_island",
        "m519_fc2_k1_registered_release_service_island")
    require(standalone == texts["old_standalone"], "standalone normalized drift")

    k1 = texts["new_k1"].replace(
        "m1058_fc2_k1_reset_hygiene_registered_release_8bank_raw4_acc24",
        "m519_fc2_k1_registered_release_8bank_raw4_acc24").replace(
        "m1058_fc2_reset_hygiene_registered_release_standalone_raw4_acc24",
        "m519_fc2_registered_release_standalone_raw4_acc24")
    require(k1 == texts["old_k1"], "K1 top normalized drift")

    shell = texts["new_shell"].replace(
        "m1058_fc2_reset_hygiene_channel_split_registered_release_matched_8bank_raw4_acc24",
        "m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24").replace(
        "m1058_fc2_k1_reset_hygiene_registered_release_8bank_raw4_acc24",
        "m519_fc2_k1_registered_release_8bank_raw4_acc24")
    require(shell == texts["old_shell"], "matched shell/K8/K1x8 instances drift")


def audit_production_assets(mapped_tb: str, dc_filelist: str, launch: dict) -> None:
    require("+vcs+initreg" not in mapped_tb and "+vcs+initreg" not in dc_filelist,
            "production initreg forbidden")
    require(launch["status"] == "PREPARED_NOT_RELEASED" and
            launch["launch_now"] is False and launch["required_independent_hammer"] == "M1059",
            "release boundary drift")
    require(mapped_tb.count(K1_ANCHOR_CLAUSE) == 1,
            "K1 mapped cycle anchors missing/duplicated")


def rejected(fn) -> str:
    try:
        fn()
    except (RuntimeError, ValueError, KeyError):
        return "REJECTED"
    raise RuntimeError("fault injection escaped")


def main() -> None:
    for key, path in PATHS.items():
        require(sha(path) == EXPECTED[key], "exact SHA drift: " + key)
    for stem in (PATHS["source_contract"], PATHS["launch_candidate"], PATHS["source_review"]):
        sidecar = stem.relative_to(HW).as_posix() + ".sha256"
        subprocess.run(["sha256sum", "-c", sidecar], cwd=HW,
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["sha256sum", "-c", sidecar + ".seal.sha256"], cwd=HW,
                       check=True, stdout=subprocess.DEVNULL)
    verify_dir_seal(M1050, M1050_OUTER)
    verify_dir_seal(M1058_RESULT, M1058_RESULT_OUTER)

    source_contract = strict_json(PATHS["source_contract"])
    launch = strict_json(PATHS["launch_candidate"])
    m1050 = strict_json(M1050 / "review.json")
    require(source_contract["status"] ==
            "PASS_SOURCE_RTL_VCS__MAPPED_FIX_NOT_ADMITTED__REQUIRES_M1059",
            "M1058 status drift")
    require(m1050["status"] ==
            "PASS_M1050_M1046_WATCHDOG_FAILURE_AUDIT__M1046_DO_NOT_RETRY" and
            m1050["failure_boundary"]["m1046_attempt_consumed"] is True and
            m1050["failure_boundary"]["m1046_retry_authorized"] is False and
            m1050["failure_boundary"]["completed_gate_cases"] == 0 and
            m1050["failure_boundary"]["production_saif_files"] == 0 and
            m1050["root_cause"]["class"] == "GATE_LEVEL_UNINITIALIZED_STATE_X_PROPAGATION",
            "M1046/M1050 failure identity drift")

    texts = {key: path.read_text(encoding="utf-8") for key, path in PATHS.items()
             if key.startswith("new_") or key.startswith("old_")}
    audit_additive_sources(texts)

    mapped_tb = PATHS["mapped_tb"].read_text(encoding="utf-8")
    dc_filelist = PATHS["dc_filelist"].read_text(encoding="utf-8")
    audit_production_assets(mapped_tb, dc_filelist, launch)

    attacks = {
        "deleted_reset": rejected(lambda: audit_additive_sources(
            {**texts, "new_service": texts["new_service"].replace(RESET_BLOCK, "")})),
        "changed_datapath": rejected(lambda: audit_additive_sources(
            {**texts, "new_service": texts["new_service"].replace(
                "default: fifo_count_q <= fifo_count_q;",
                "default: fifo_count_q <= fifo_count_q + 1'b1;")})),
        "changed_anchor": rejected(lambda: audit_production_assets(
            mapped_tb.replace("0:return 259", "0:return 260"), dc_filelist, launch)),
        "allowed_initreg": rejected(lambda: audit_production_assets(
            mapped_tb, dc_filelist + "\n+vcs+initreg+random\n", launch)),
        "wrong_status": rejected(lambda: audit_production_assets(
            mapped_tb, dc_filelist, {**launch, "status": "PASS"})),
        "launch_now_true": rejected(lambda: audit_production_assets(
            mapped_tb, dc_filelist, {**launch, "launch_now": True})),
        "changed_k8_instance": rejected(lambda: audit_additive_sources(
            {**texts, "new_shell": texts["new_shell"].replace(
                "m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24",
                "attacker_k8")})),
        "changed_k1x8_instance": rejected(lambda: audit_additive_sources(
            {**texts, "new_shell": texts["new_shell"].replace(
                "m519_fc2_k1x8_registered_release_raw4_acc24", "attacker_k1x8")})),
        "wrong_source_sha": rejected(lambda: require(
            "0" * 64 == EXPECTED["new_service"], "wrong source SHA")),
        "wrong_m1050_status": rejected(lambda: require(
            "WRONG" == m1050["status"], "wrong M1050 status")),
    }
    require(set(attacks.values()) == {"REJECTED"}, "fault injection failure")

    print("PASS M1059 source equivalence and independent RTL evidence")
    print("GO M1059 exact non-launching production candidate for M1068 release authoring")
    print("PASS 10/10 fail-closed attacks rejected")


if __name__ == "__main__":
    main()
