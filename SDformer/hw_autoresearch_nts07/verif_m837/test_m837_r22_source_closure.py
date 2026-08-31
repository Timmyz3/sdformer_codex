#!/usr/bin/env python3
"""Static source closure for M837 R22; no EDA/tool probe."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function_block(text, name):
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.startswith(name + "() {"))
    depth = 0
    result = []
    for line in lines[start:]:
        result.append(line)
        depth += line.count("{") - line.count("}")
        if depth == 0:
            return "\n".join(result) + "\n"
    raise RuntimeError("unterminated function " + name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hw_root", type=Path)
    args = parser.parse_args()
    root = args.hw_root.resolve(strict=True)
    runner = root / "dc_handoff/scripts/run_vcs_m837_c2_r22_identity_compat_exact_sha.sh"
    predecessor = root / "dc_handoff/scripts/run_vcs_m833_c2_r21_unicode_exact_sha.sh"
    guard = root / "verif_m837/m837_c2_r22_identity_compat_guard.py"
    base_guard = root / "verif_m826/m826_c2_r20_atomic_guard.py"
    contract = root / "contracts/m837_c2_r22_identity_compat_source_only_contract_r1_20260829.json"
    candidate = root / "contracts/m837_c2_r22_identity_compat_vcs_launch_candidate_source_only_r1_20260829.json"
    for path in (runner, predecessor, guard, base_guard, contract, candidate):
        require(path.is_file() and not path.is_symlink(),
                "missing/nonregular: " + str(path))
    syntax = subprocess.run(["/usr/bin/bash", "-n", str(runner)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    require(syntax.returncode == 0, "bash -n failed")
    text = runner.read_text(encoding="utf-8")
    old = predecessor.read_text(encoding="utf-8")
    require(text.count('python36_utf8 "${guard}"') == 12 and
            text.count('python36_utf8 - "${work}"') == 1,
            "local UTF-8 call population drift")
    require("export LANG" not in text and "export LC_ALL" not in text,
            "global locale export")
    require("PYTHONUTF8" not in text and "PYTHONIOENCODING" not in text,
            "ineffective Unicode knob present")
    for name in ("license_gate", "compile_and_run"):
        require(function_block(text, name) == function_block(old, name),
                name + " drifted from M833")
    require(text.count('"${vcs}" -full64') == 1 and
            text.count('"${phase_dir}/simv" "+ntb_random_seed=${seed}"') == 1,
            "VCS/simv outer-locale invocation drift")
    source = subprocess.run([
        "/usr/libexec/platform-python3.6", str(guard), "validate-source",
        "--hw-root", str(root), "--contract", str(contract),
        "--candidate", str(candidate), "--runner", str(runner),
    ], env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8"}, stdout=subprocess.PIPE,
       stderr=subprocess.PIPE, universal_newlines=True)
    require(source.returncode == 0, "validate-source failed: " + source.stderr)
    value = json.loads(source.stdout)
    require(value["status"] ==
            "PASS_M837_R22_SOURCE_IDENTITY__NO_VCS_OR_EDA",
            "source status drift")
    require(digest(root / "docs/359_DATE终局冻结_20260813.md") ==
            "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
            "docs359 drift")
    print(json.dumps({
        "schema": "m837_r22_source_closure_v1",
        "status": "PASS_M837_R22_SOURCE_CLOSURE__NO_EDA",
        "python": __import__("sys").version.split()[0],
        "runner_sha256": digest(runner), "guard_sha256": digest(guard),
        "contract_sha256": digest(contract),
        "candidate_sha256": digest(candidate),
        "local_utf8_guard_calls": 12, "local_utf8_inline_calls": 1,
        "license_and_compile_functions_frozen": True,
        "vcs_simv_locale_leak": False, "source": value,
        "vcs_executed": False, "license_queried": False,
        "eda_executed": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
