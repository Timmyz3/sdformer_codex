#!/usr/bin/env python3
"""Static/source closure for the additive C2 R21 Unicode-only repair."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess


DEF_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{")
KNOWN = {"sha", "fail", "expect_file_sha", "log_phase", "trace_event",
         "source_dry_run", "collision_gate", "resource_gate", "license_gate",
         "compile_and_run", "publish_failure_receipt", "failure_cleanup",
         "signal_exit", "python36_utf8"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


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


def function_closure(text):
    definitions = Counter(match.group(1) for match in
                          map(DEF_RE.match, text.splitlines()) if match)
    require(all(count == 1 for count in definitions.values()),
            "duplicate function definition")
    require(set(definitions) == KNOWN, "function definition set drift")
    return dict(sorted(definitions.items()))


def load_guard(path):
    spec = importlib.util.spec_from_file_location("m833_guard", path)
    require(spec is not None and spec.loader is not None, "cannot load guard")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hw_root", type=Path)
    args = parser.parse_args()
    root = args.hw_root.resolve(strict=True)
    runner = root / "dc_handoff/scripts/run_vcs_m833_c2_r21_unicode_exact_sha.sh"
    old_runner = root / "dc_handoff/scripts/run_vcs_m826_c2_r20_atomic_exact_sha.sh"
    guard_path = root / "verif_m826/m826_c2_r20_atomic_guard.py"
    contract_path = root / "contracts/m833_c2_r21_unicode_source_only_contract_r1_20260829.json"
    candidate_path = root / "contracts/m833_c2_r21_unicode_vcs_launch_candidate_source_only_r1_20260829.json"
    for path in (runner, old_runner, guard_path, contract_path, candidate_path):
        require(path.is_file() and not path.is_symlink(),
                "missing/nonregular source: " + str(path))
    syntax = subprocess.run(["/usr/bin/bash", "-n", str(runner)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    require(syntax.returncode == 0, "bash -n failed: " + syntax.stderr)
    text = runner.read_text(encoding="utf-8")
    definitions = function_closure(text)
    require('env LANG=C.UTF-8 LC_ALL=C.UTF-8 "${python36}" "$@"' in text,
            "runner-local UTF-8 wrapper absent")
    require("PYTHONUTF8" not in text and "PYTHONIOENCODING" not in text,
            "known ineffective Unicode knob present")
    direct = [line.strip() for line in text.splitlines()
              if '"${python36}"' in line]
    require(direct == [
        'env LANG=C.UTF-8 LC_ALL=C.UTF-8 "${python36}" "$@"',
        'expect_file_sha "${python36}" 9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f',
    ], "unwrapped Python execution remains")
    require(text.count('python36_utf8 "${guard}"') == 12 and
            text.count('python36_utf8 - "${work}"') == 1,
            "Python wrapper call population drift")
    require("export LANG" not in text and "export LC_ALL" not in text,
            "locale leaked into runner global environment")
    old_text = old_runner.read_text(encoding="utf-8")
    for name in ("license_gate", "compile_and_run"):
        require(function_block(text, name) == function_block(old_text, name),
                name + " changed; VCS/license locale boundary is not frozen")
    guard = load_guard(guard_path)
    source = guard.validate_source(root, contract_path, candidate_path, runner)
    contract = guard.strict_json(contract_path)
    require(contract["runner_sha256"] == digest(runner), "runner SHA drift")
    audit = root / "reviews/m832_m826_c2_r20_unicode_preattempt_failure_hammer_r1_20260829"
    audit_identity = guard.verify_sealed_directory(audit)
    require(digest(audit / "review.json") ==
            "a0099bbd4ec42679a31c7cdaf44964427c16bf3d971f4c5673955c0db9f06de2",
            "M832 review SHA drift")
    require(audit_identity["manifest_sha256"] ==
            "93372a7ab3a14e9932f5d37af6464274d4b67825a35b8b6dd355be8e122cd5bc" and
            audit_identity["outer_seal_file_sha256"] ==
            "1426d81e47c027e19d0ed9b38f60a0a7339127de205c8b98a18a457a7c06f6cd",
            "M832 double seal drift")
    audit_json = guard.strict_json(audit / "review.json")
    require(audit_json["claim_boundary"]["m826_release_reusable"] is False and
            audit_json["claim_boundary"]["m826_attempt_consumed"] is False,
            "M832 release/attempt boundary drift")
    for path in (
        root / "results/.m826_c2_r20_atomic_channel_split_vcs_attempt_consumed",
        root / "results/m826_c2_r20_atomic_channel_split_vcs_r1_20260829",
        root / "results/.m833_c2_r21_unicode_channel_split_vcs_attempt_consumed",
        root / "results/m833_c2_r21_unicode_channel_split_vcs_r1_20260829",
        root / "contracts/m835_m833_c2_r21_unicode_vcs_launch_admission_r1_20260829.json",
        root / "reviews/m834_m833_c2_r21_unicode_source_fresh_hammer_r1_20260829",
        root / "reviews/m836_m835_m833_c2_r21_unicode_final_launch_hammer_r1_20260829",
    ):
        require(not path.exists() and not path.is_symlink(),
                "formal/future identity unexpectedly exists: " + str(path))
    print(json.dumps({
        "schema": "m833_r21_source_closure_v1",
        "status": "PASS_M833_R21_UNICODE_SOURCE_CLOSURE__NO_EDA_EXECUTED",
        "python": __import__("sys").version.split()[0],
        "bash_n": "PASS", "function_definitions": definitions,
        "python_wrapper_guard_calls": 12, "python_wrapper_inline_calls": 1,
        "vcs_simv_locale_leak": False, "source_validation": source,
        "m832_failure_authority": "PASS_DOUBLE_SEALED",
        "m826_release_reusable": False, "m826_attempt_consumed": False,
        "future_identities_absent": True, "vcs_executed": False,
        "license_queried": False, "eda_executed": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
