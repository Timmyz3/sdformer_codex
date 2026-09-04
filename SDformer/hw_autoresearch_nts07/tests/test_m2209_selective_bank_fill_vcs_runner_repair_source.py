#!/opt/anaconda3/bin/python3
"""No-EDA mutation tests for the M2209 fixed-Python one-shot runner."""
from __future__ import annotations

from pathlib import Path
import re
import hashlib
import subprocess


HW = Path(__file__).resolve().parents[1]
RUNNER = HW / "dc_handoff/scripts/run_m2211_m2210_m2209_selective_bank_fill_directed_vcs_one_shot.sh"
BASE_TEST = HW / "tests/test_m2197_c2_tsbg_selective_bank_fill_validation_repair_source.py"
BASE_TEST_SHA = "81d4cb93e7534e5ebb6cf68c02ded17db862479ab646deccc9ef9eb60e50dd5d"
PYTHON_SHA = "873a1168d6d2a7d1b406b85c2a1ea986a6f086041069ab1ee3f70b9217f10161"
PARSER_SHA = "fde65c8372c9eab82ae49caea03137cdd93d0bd996fe65e9549220869a743571"


class Rejected(RuntimeError):
    pass


def need(ok: bool, message: str) -> None:
    if not ok:
        raise Rejected(message)


def audit_runner(text: str) -> dict[str, object]:
    need(text.count("PYTHON=/opt/anaconda3/bin/python3.12") == 1,
         "fixed interpreter path")
    need(text.count(f"sha_mode_exact {PYTHON_SHA} 755 yes \"${{PYTHON}}\"") == 1,
         "interpreter SHA/mode pin")
    need(text.count(f"sha_mode_exact {PARSER_SHA} 664 no \"${{PARSER}}\"") == 1,
         "parser SHA/mode pin")
    need(text.count('"${PYTHON}" -B "${PARSER}" --sim-log "${WORK}/simv.log"') == 1,
         "fixed interpreter parser invocation")
    need(not re.search(r'(?m)^\s*"\$\{PARSER\}"\s+--sim-log', text),
         "direct parser execution")
    need(not re.search(r'(?m)^\s*(?:chmod|cp|install)\b.*\$\{PARSER\}', text),
         "parser mutation command")
    need(text.count('rm -f -- "${WORK}/simv" "${WORK}/vc_hdrs.h"') == 1,
         "build file cleanup")
    need(text.count('rm -rf -- "${WORK}/csrc" "${WORK}/simv.daidir" "${WORK}/simv.vdb"') == 1,
         "all build directory cleanup")
    need(text.count("for build_only in simv vc_hdrs.h csrc simv.daidir simv.vdb; do") == 1,
         "post-cleanup absence gate")
    need(text.count('[[ -z "$(find -P "${WORK}" -type l -print -quit)" ]] || exit 5') == 1,
         "success symlink rejection")
    need('RESULT="${HW_ROOT}/results/m2211_m2209_c2_tsbg_selective_bank_fill_directed_vcs_r1_20260904"' in text,
         "fresh result identity")
    need('ATTEMPT="${HW_ROOT}/results/.m2211_m2209_selective_bank_fill_vcs_attempt_consumed"' in text and
         'LOCK="${HW_ROOT}/results/.m2211_m2209_selective_bank_fill_vcs_launch_lock"' in text,
         "fresh attempt/lock identity")
    need("m2199_m2197_c2_tsbg_selective_bank_fill_directed_vcs_r1_20260904.failed" not in text,
         "old raw reuse")
    need("retry=false" in text and "automatic_retry':False" in text and
         "reuse_old_artifacts=false" in text and "'reuse_old_artifacts':False" in text,
         "no retry/reuse budget")
    need(text.count("M2211_EXPECTED_RUNNER_SHA256") >= 2 and
         text.count("M2211_EXPECTED_M2210_REVIEW_SHA256") >= 2,
         "runner/review launch pins")
    immutable = {
        "rtl": "f651ea3a3b4dfab04d021a1e44797e7ab72c244cb7edf7496e18ac1ac033339e",
        "m803": "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
        "sva": "8003115edb919e9c5c6c9c36ce4ba75dfb37d9ec9f23e7c4cf59e2aed3b461b4",
        "tb": "a8a954826324aa20443e7b2acbbc6a0b1b2a92f83ebdd84bfdbb0879920526e3",
        "filelist": "5beddf477b6938b599cfab962eba60f6d79dceeb825380f2e5cdc6f22b49dc13",
    }
    need(all(digest in text for digest in immutable.values()), "immutable source pin")
    need(text.count('"${VCS}" -full64 -sverilog -assert svaext -timescale=1ns/1ps') == 1,
         "frozen compile command")
    need(text.count('"${WORK}/simv" \\') == 1 and
         text.count('-assert global_finish_maxfail=1') == 1, "frozen sim command")
    return {"status": "PASS_M2209_STATIC_RUNNER_SOURCE", "mutations_required": 10,
            "fixed_python": True, "build_only_cleanup": 5,
            "functional_sources_immutable": True}


def rejected(text: str) -> bool:
    try:
        audit_runner(text)
    except Rejected:
        return True
    return False


def main() -> None:
    source = RUNNER.read_text()
    assert audit_runner(source)["status"] == "PASS_M2209_STATIC_RUNNER_SOURCE"
    # The M2197 test ends with a deliberate virgin-attempt assertion.  M2199 has
    # already consumed that one-shot identity, so rerunning main() now would be a
    # false failure.  Pin the previously sealed test bytes instead; the immutable
    # RTL/TB/SVA/filelist/parser identities are independently pinned by the new
    # runner audit above.
    assert hashlib.sha256(BASE_TEST.read_bytes()).hexdigest() == BASE_TEST_SHA
    subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)], check=True,
                   capture_output=True, text=True, timeout=30)

    mutations = {
        "direct_parser": source.replace('"${PYTHON}" -B "${PARSER}"', '"${PARSER}"', 1),
        "wrong_interpreter_path": source.replace(
            "PYTHON=/opt/anaconda3/bin/python3.12", "PYTHON=/usr/bin/python3.12", 1),
        "wrong_interpreter_sha": source.replace(PYTHON_SHA, "0" * 64, 1),
        "wrong_interpreter_mode": source.replace(
            f"sha_mode_exact {PYTHON_SHA} 755 yes", f"sha_mode_exact {PYTHON_SHA} 775 yes", 1),
        "parser_sha_drift": source.replace(PARSER_SHA, "1" * 64, 1),
        "parser_mode_drift": source.replace(
            f"sha_mode_exact {PARSER_SHA} 664 no", f"sha_mode_exact {PARSER_SHA} 755 yes", 1),
        "missing_simv_vdb_cleanup": source.replace(
            'rm -rf -- "${WORK}/csrc" "${WORK}/simv.daidir" "${WORK}/simv.vdb"',
            'rm -rf -- "${WORK}/csrc" "${WORK}/simv.daidir"', 1),
        "old_result_identity": source.replace(
            "m2211_m2209_c2_tsbg_selective_bank_fill_directed_vcs_r1_20260904",
            "m2199_m2197_c2_tsbg_selective_bank_fill_directed_vcs_r1_20260904", 1),
        "retry_enabled": source.replace("retry=false", "retry=true"),
        "old_artifact_reuse": source.replace("reuse_old_artifacts=false",
                                              "reuse_old_artifacts=true", 1),
    }
    assert all(rejected(text) for text in mutations.values())
    assert len(mutations) == 10
    print("PASS_M2209_SOURCE_TESTS inherited_m2197_bytes_pinned=1 bash_syntax=1 "
          "runner_control=1 runner_mutations=10 parser_runs=0 vcs_runs=0 "
          "license_queries=0 eda_runs=0 gpu_runs=0")


if __name__ == "__main__":
    main()
