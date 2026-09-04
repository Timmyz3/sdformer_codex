#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Exact-byte, source-only gate for the M1265 R12 one-shot release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


HW = Path(__file__).resolve().parents[1]
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1265_m1258r12_m1162_c1_common_charge_protocol_exact_byte_r12.sh"
FILELIST = HW / "dc_handoff/filelists/date_m1265_m1258r12_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
TB = HW / "verif_m1258r12_c1_common_charge_protocol/tb_m1258r12_m1162_common_charge_protocol_unit_delay_r12.sv"
PARENT = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
FOUNDRY = Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v")
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
CONTRACT = HW / "contracts/m1265_c1_r12_exact_byte_vcs_release_source_contract_r1_20260830.json"
RELEASE = HW / "contracts/m1265_c1_r12_exact_byte_vcs_launch_release_r1_20260830.json"
AUTHOR = HW / "reviews/m1265_c1_r12_exact_byte_vcs_release_source_author_r1_20260830"

EXPECTED = {
    RUNNER: "320e8f692557f8111c708f245987d2f831710204a23199030e7a90c3ba6bea28",
    FILELIST: "eb579191d78eee1870fb98866a3436db732db52fdb638e742151b0f10f849de0",
    TB: "e13d630f4cf2e2f7e0264dc2325218aee4cc580497be3b37deb1ff7a641ad302",
    PARENT: "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    WRAPPER: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    SVA: "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    FOUNDRY: "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
    VCS: "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    PYTHON: "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sidecar(path: Path) -> None:
    sums = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    assert sums.read_text().split() == [sha(path), path.name]
    assert outer.read_text().split() == [sha(sums), sums.name]


def verify_dir(root: Path) -> None:
    sums, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    assert root.is_dir() and not root.is_symlink()
    assert outer.read_text().split() == [sha(sums), "SHA256SUMS"]
    listed = {}
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        assert name not in listed and not Path(name).is_absolute() and ".." not in Path(name).parts
        listed[name] = digest
    actual = set()
    for base_text, dirs, files in os.walk(root, followlinks=False):
        base = Path(base_text); dirs[:] = [n for n in dirs if not (base/n).is_symlink()]
        for name in files:
            path = base/name; rel = path.relative_to(root).as_posix()
            if rel not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"} and not path.is_symlink():
                actual.add(rel)
    assert actual == set(listed)
    for name, digest in listed.items(): assert sha(root/name) == digest


def env_gate(env: dict[str, str]) -> bool:
    names = ("M1265_EXPECTED_RELEASE_SHA256",
             "M1265_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256",
             "M1265_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256",
             "M1265_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256")
    return all(re.fullmatch(r"[0-9a-f]{64}", env.get(name, "")) is not None for name in names)


def exact_byte_gate(path: Path, text: str) -> bool:
    return path in EXPECTED and sha_text(text) == EXPECTED[path]


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--skip-author", action="store_true")
    args = parser.parse_args(); checks = 0
    for path, digest in EXPECTED.items():
        assert path.is_file() and not path.is_symlink() and sha(path) == digest, path
        checks += 1
    syntax = subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, check=False)
    assert syntax.returncode == 0; checks += 1
    expected_lines = [str(x) for x in (FOUNDRY, PARENT, M935, WRAPPER, SVA, TB)]
    assert FILELIST.read_text().splitlines() == expected_lines; checks += 1
    runner = RUNNER.read_text()
    assert runner.count('"${VCS_BIN}" -full64') == 1; checks += 1
    assert runner.count('./simv -no_save') == 1; checks += 1
    assert runner.count('/usr/bin/timeout --signal=TERM --kill-after=30s') == 2; checks += 1
    assert "automatic_retry=true" not in runner and "rm -" not in runner; checks += 1
    assert "Exact-byte technical corpus" in runner and "failed_or_incomplete.$$.quarantine" in runner; checks += 1
    for path in (CONTRACT, RELEASE): sidecar(path); checks += 1
    contract, release = json.loads(CONTRACT.read_text()), json.loads(RELEASE.read_text())
    assert contract["status"] == "M1265_R12_EXACT_BYTE_ONE_SHOT_RELEASE_SOURCE_READY__FRESH_M1266_REQUIRED"
    assert release["status"] == "AUTHORIZE_ONE_M1265_R12_UNIT_DELAY_VCS_ATTEMPT_AFTER_FRESH_M1266_HAMMER"
    assert contract["identity"]["runner_sha256"] == sha(RUNNER) == release["identity"]["runner_sha256"]
    assert release["identity"]["source_contract_sha256"] == sha(CONTRACT); checks += 1
    if not args.skip_author:
        verify_dir(AUTHOR); author = json.loads((AUTHOR/"review.json").read_text())
        assert author["status"] == "PASS_M1265_R12_EXACT_BYTE_RELEASE_SOURCE__FRESH_M1266_HAMMER_REQUIRED"
        assert author["bindings"]["runner_sha256"] == sha(RUNNER)
        assert author["bindings"]["source_contract_sha256"] == sha(CONTRACT)
        assert author["bindings"]["release_sha256"] == sha(RELEASE); checks += 1
    good = {name: "a"*64 for name in ("M1265_EXPECTED_RELEASE_SHA256",
        "M1265_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256",
        "M1265_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256",
        "M1265_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256")}
    assert env_gate(good); checks += 1
    for name in tuple(good):
        bad = dict(good); bad.pop(name); assert not env_gate(bad); checks += 1
    attempt = HW/"results/.m1265_m1258r12_m1162_c1_common_charge_protocol_vcs_r12_attempt_consumed"
    result = HW/"results/m1265_m1258r12_m1162_c1_common_charge_protocol_unit_delay_vcs_r12_20260830"
    assert not os.path.lexists(attempt) and not os.path.lexists(result); checks += 1
    for data in (contract, release):
        for key in ("functional_vcs_verified","timing_verified","cycles_measured","speedup","ppa","power","energy","system_speedup","paper_citable"):
            assert data["claim_boundary"][key] is False; checks += 1
    print(json.dumps({"schema":"m1265_c1_r12_exact_byte_release_static_r1_v1",
        "status":"PASS_M1265_SOURCE_ONLY__FRESH_M1266_REQUIRED__NO_VCS_NO_EDA",
        "checks_passed":checks,"exact_byte_corpus":True,"one_compile":True,
        "one_sim":True,"automatic_retry":False,"failure_double_seal":True,
        "vcs_runs":0,"simv_runs":0,"all_eda_runs":0,
        "docs359_sha256":sha(DOCS359)},indent=2,sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
