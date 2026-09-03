#!/usr/bin/env python3
"""Read-only M1997 source/runner hammer.  This script never invokes EDA."""

import difflib
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
BASE = HW / "rtl_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend.sv"
RTL = HW / "rtl_m1995/m1995_m1880_c2_tsbg_b4_dc_keyword_legal_frontend.sv"
FILELIST = HW / "dc_handoff/filelists/iscas_m1996_m1995_c2_tsbg_b4_keyword_legal_directed_vcs.f"
RUNNER = HW / "dc_handoff/scripts/run_m1998_m1997_m1995_c2_tsbg_b4_keyword_legal_vcs_one_shot.sh"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M1990 = HW / "reviews/m1990_m1986_c2_tsbg_b4_parseable_vcs_result_hammer_r1_20260902"
M1995 = HW / "reviews/m1995_m1992_tsbg_dc_keyword_failure_hammer_r1_20260902"

EXPECTED = {
    RTL: "2c1a8a7644b359a153decdc3106a8718992d37d54809007b61e184121fcc14fd",
    BASE: "8524f6a7a6d09e1aaab55ee91515bd1fce9ea57fa2a478a9817f637685299a05",
    FILELIST: "a89c09074abbde86fc3f5b2a748418bef601f5bf9ec6f53736992e155e29414c",
    RUNNER: "de872c1b7f323b483a008108fc48d793c8176667eb19f4edfc543e3035b22e96",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_seal(directory: Path) -> None:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    assert manifest.is_file() and not manifest.is_symlink()
    assert outer.is_file() and not outer.is_symlink()
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        rel = rel.lstrip(" *")
        assert sha(directory / rel) == digest
    outer_digest, outer_rel = outer.read_text().split(maxsplit=1)
    assert outer_rel.strip().lstrip("*") == "SHA256SUMS"
    assert sha(manifest) == outer_digest


def verify_runner(text: str) -> None:
    required_once = [
        'sha_exact 2c1a8a7644b359a153decdc3106a8718992d37d54809007b61e184121fcc14fd "${RTL}"',
        'verify_dir_seal "${M1990_DIR}"',
        'verify_dir_seal "${M1995_DIR}"',
        'verify_dir_seal "${SOURCE_REVIEW_DIR}"',
        '"${LMUTIL}" lmstat -a -c "${LICENSE_SERVER}"',
        '"${VCS}" -full64 -sverilog -assert svaext -top "${TOP}" -f "${FILELIST}"',
        '/usr/bin/timeout --signal=TERM --kill-after=10s 180s "${WORK}/simv"',
        '-assert global_finish_maxfail=1 >"${WORK}/simv.log" 2>&1',
        'grep -Fxc "${EXPECTED_PASS}" "${WORK}/simv.log"',
        "grep -Fc 'M1970_LOAD_BEGIN' \"${WORK}/simv.log\")\" -eq 52",
        "grep -Fc 'M1970_LOAD_COMPLETE' \"${WORK}/simv.log\")\" -eq 52",
        "grep -Fc 'M1970_LOAD_TIMEOUT' \"${WORK}/simv.log\")\" -eq 0",
        'mv -T -- "${WORK}" "${RESULT}.failed_or_incomplete.$$.quarantine"',
        'mv -T -- "${WORK}" "${RESULT}"',
    ]
    for token in required_once:
        assert text.count(token) == 1, token
    assert text.count("M1998_EXPECTED_RUNNER_SHA256") == 2
    assert text.count("M1998_EXPECTED_REVIEW_SHA256") == 2
    assert text.count("retry=false") == 3
    assert "retry=true" not in text
    assert "automatic_retry': False" in text
    assert text.count("lmstat -a") == 1
    assert len(re.findall(r'\"\$\{VCS\}\"\s+-full64', text)) == 1
    assert len(re.findall(r'180s\s+\"\$\{WORK\}/simv\"', text)) == 1
    assert "[[ ! -e \"${RESULT}\" && ! -e \"${ATTEMPT}\" && ! -e \"${WORK}\" && ! -e \"${LOCK}\" ]]" in text
    assert "mkdir -- \"${LOCK}\" \"${ATTEMPT}\" \"${WORK}\"" in text
    assert "seal_dir \"${ATTEMPT}\"" in text
    assert "seal_dir \"${WORK}\"" in text
    assert "WORK_ACTIVE=1" in text and "WORK_ACTIVE=0" in text
    phase_header = "for phase in reset full_load full_execute retired_replay replay_reset_recovery stale_attack stale_reset_recovery recovery_load recovery_execute final_checks; do"
    assert text.count(phase_header) == 1
    assert text.count('grep -Fc "M1970_PHASE ${phase}_begin"') == 1
    assert text.count('grep -Fc "M1970_PHASE ${phase}_complete"') == 1
    assert "Warning-\\[SVAA-RNF\\]" in text
    assert "Assertion[^[:cntrl:]]*failed" in text
    assert "whole-test watchdog expired" in text
    assert "directed timeout" in text
    assert "post-reset legal-service timeout" in text
    assert "same_area=false" in text
    assert "exact_cycle_speedup=false" in text
    assert "system_speedup=false" in text


def main() -> None:
    for path, digest in EXPECTED.items():
        assert path.is_file() and not path.is_symlink(), path
        assert sha(path) == digest, path
    verify_seal(M1990)
    verify_seal(M1995)

    base = BASE.read_text()
    rtl = RTL.read_text()
    assert len(re.findall(r"\bcontext\b", base)) == 16
    assert len(re.findall(r"\bctx\b", rtl)) == 16
    assert len(re.findall(r"\bcontext\b", rtl)) == 0
    assert re.sub(r"\bctx\b", "context", rtl) == base
    changed = [(a, b) for a, b in zip(base.splitlines(), rtl.splitlines()) if a != b]
    assert len(changed) == 12
    assert sum(len(re.findall(r"\bcontext\b", a)) for a, _ in changed) == 16
    assert all(re.sub(r"\bctx\b", "context", b) == a for a, b in changed)
    assert len(base.splitlines()) == len(rtl.splitlines())
    assert re.findall(r"\bmodule\s+(\w+)", base) == re.findall(r"\bmodule\s+(\w+)", rtl)

    entries = [line for line in FILELIST.read_text().splitlines() if line.strip()]
    assert entries == [
        "+incdir+hw_autoresearch_nts07/rtl_m803",
        "hw_autoresearch_nts07/rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
        "hw_autoresearch_nts07/rtl_m1995/m1995_m1880_c2_tsbg_b4_dc_keyword_legal_frontend.sv",
        "hw_autoresearch_nts07/verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv",
        "hw_autoresearch_nts07/tb_m1984/tb_m1984_c2_tsbg_b4_parseable_pass.sv",
    ]
    verify_runner(RUNNER.read_text())

    # Eight independently malformed runner variants must fail the same checker.
    original = RUNNER.read_text()
    mutations = [
        original.replace(EXPECTED[RTL], "0" * 64, 1),
        original.replace('verify_dir_seal "${M1995_DIR}"', ":", 1),
        original.replace("--kill-after=10s 180s", "--kill-after=10s 181s", 1),
        original.replace("-assert global_finish_maxfail=1", "-assert global_finish_maxfail=2", 1),
        original.replace("grep -Fxc \"${EXPECTED_PASS}\"", "grep -Fc \"${EXPECTED_PASS}\"", 1),
        original.replace("retry=false", "retry=true", 1),
        original.replace('mv -T -- "${WORK}" "${RESULT}.failed_or_incomplete.$$.quarantine"', ":", 1),
        original.replace('"${LMUTIL}" lmstat -a -c "${LICENSE_SERVER}"', '"${LMUTIL}" lmstat -a -c "${LICENSE_SERVER}"\n"${LMUTIL}" lmstat -a -c "${LICENSE_SERVER}"', 1),
    ]
    rejected = 0
    for mutant in mutations:
        try:
            verify_runner(mutant)
        except AssertionError:
            rejected += 1
    assert rejected == len(mutations)
    print("PASS_M1997_READONLY_PROBE changed_lines=12 renamed_tokens=16 mutations=8/8")


if __name__ == "__main__":
    main()
