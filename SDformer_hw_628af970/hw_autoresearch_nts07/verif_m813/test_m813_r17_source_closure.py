#!/usr/bin/env python3
"""Python-3.6 source closure for the M813/C2 R17 launch package."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess


DEF_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{")
HEREDOC_RE = re.compile(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")
KNOWN_FUNCTIONS = {
    "sha", "fail", "expect_file_sha", "log_phase", "trace_event",
    "source_dry_run", "collision_gate", "resource_gate", "license_gate",
    "compile_and_run", "publish_failure_receipt", "failure_cleanup",
    "signal_exit",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def strip_heredocs(text):
    output = []
    terminator = None
    for line in text.splitlines():
        if terminator is not None:
            if line == terminator or line.lstrip("\t") == terminator:
                terminator = None
            continue
        output.append(line)
        match = HEREDOC_RE.search(line)
        if match:
            terminator = match.group(1)
    require(terminator is None, "unterminated heredoc")
    return "\n".join(output) + "\n"


def function_audit(text):
    code = strip_heredocs(text)
    definitions = Counter()
    for line in code.splitlines():
        match = DEF_RE.match(line)
        if match:
            definitions[match.group(1)] += 1
    pattern = re.compile(
        r"(?:^\s*|[;&|]\s*|\(\s*|\$\(\s*|\b(?:if|elif|while|until|then|do|else)\s+)"
        r"(?:!\s*)?(?:[A-Za-z_][A-Za-z0-9_]*=[^\s;|&]+\s+)*"
        r"([A-Za-z_][A-Za-z0-9_]*)")
    calls = []
    for number, raw in enumerate(code.splitlines(), 1):
        line = raw.split("#", 1)[0]
        if DEF_RE.match(line):
            line = DEF_RE.sub("{", line, count=1)
        for match in pattern.finditer(line):
            word = match.group(1)
            if word in definitions or word in KNOWN_FUNCTIONS:
                calls.append((number, word))
        trap_match = re.search(r"\btrap\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if trap_match:
            calls.append((number, trap_match.group(1)))
    undefined = sorted(set(word for _, word in calls
                           if definitions.get(word, 0) == 0))
    duplicate = sorted(name for name, count in definitions.items() if count != 1)
    return {"definitions": dict(sorted(definitions.items())),
            "calls": calls, "undefined": undefined, "duplicate": duplicate,
            "pass": not undefined and not duplicate}


def load_guard(path):
    spec = importlib.util.spec_from_file_location("m813_closure_guard", path)
    require(spec is not None and spec.loader is not None, "cannot load guard")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hw_root", type=Path)
    args = parser.parse_args()
    root = args.hw_root.resolve(strict=True)
    runner = root / "dc_handoff/scripts/run_vcs_m813_c2_r17_atomic_exact_sha.sh"
    guard_path = root / "verif_m813/m813_c2_r17_atomic_guard.py"
    contract_path = root / "contracts/m813_c2_r17_atomic_source_only_contract_r1_20260829.json"
    candidate_path = root / "contracts/m813_c2_r17_vcs_launch_candidate_source_only_r1_20260829.json"
    for path in (runner, guard_path, contract_path, candidate_path):
        require(path.is_file() and not path.is_symlink(),
                "missing/nonregular source: " + str(path))
    bash_n = subprocess.run(["/usr/bin/bash", "-n", str(runner)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    require(bash_n.returncode == 0, "bash -n failed: " + bash_n.stderr)
    text = runner.read_text(encoding="utf-8")
    closure = function_audit(text)
    require(closure["pass"], "runner function closure failed: " +
            repr(closure))
    mutated = text.replace("publish_failure_receipt() {",
                           "publish_failure_receipt_deleted() {", 1)
    negative = function_audit(mutated)
    require(not negative["pass"] and
            "publish_failure_receipt" in negative["undefined"],
            "undefined-function negative escaped")
    require('mv "${work}" "${result}"' not in text and
            "publish-no-replace --source \"${work}\"" in text and
            "create-attempt-stage --stage \"${attempt_stage}\"" in text and
            "verify-attempt --path \"${attempt}\"" in text and
            "write-failure-quarantine" in text,
            "atomic runner boundary drift")
    require(text.index("trap failure_cleanup EXIT") <
            text.index("create-attempt-stage"),
            "failure trap is not armed before attempt staging")

    guard = load_guard(guard_path)
    source = guard.validate_source(root, contract_path, candidate_path, runner)
    contract = guard.strict_json(contract_path)
    require(contract["runner_sha256"] == digest(runner), "runner SHA binding")
    require(contract["candidate_sha256"] == digest(candidate_path),
            "candidate SHA binding")
    require(digest(root / "docs/359_DATE终局冻结_20260813.md") ==
            guard.DOCS359_SHA256, "docs359 drift")
    frozen = {
        "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv":
            "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
        "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv":
            "2588f890213d29aab6829dff679719c0f9ce4762c17bb061d1869b27a2f1d50e",
        "rtl_m803/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24.sv":
            "3328e52d8cf1eec6098ebb7b0525ac55cd8bd6b2fe5b5e504b337d1a678e3c4b",
        "verif_m803/m803_fc2_channel_split_cutthrough_adapter_assertions.sv":
            "6d7803e587eb28d45d296c2795a1c7ab2f43a96aa66b5fe9528e7caa01ff5c30",
        "tb_m803/tb_m803_c2_r16_channel_split_adapter_attacks.sv":
            "b89948e7e8329da5527c326e1f0e5808c3795ec4da98a6f249b067bfca226872",
        "tb_m803/tb_m803_c2_r16_channel_split_k8_vs_k1x8_raw4_acc24.sv":
            "6d1c16127fe863c24235b55cb7de3eaaa4edd8117a21aa7a1772a8708bfdee40",
    }
    for rel, expected in frozen.items():
        require(digest(root / rel) == expected, "frozen M803 drift: " + rel)
    for future in (
        root / "reviews/m814_m813_c2_r17_atomic_source_fresh_hammer_r1_20260829",
        root / "contracts/m813_c2_r17_atomic_vcs_launch_admission_r1_20260829.json",
        root / "reviews/m816_m813_c2_r17_atomic_final_launch_hammer_r1_20260829",
        root / "results/.m813_c2_r17_atomic_channel_split_vcs_attempt_consumed",
        root / "results/m813_c2_r17_atomic_channel_split_vcs_r1_20260829",
    ):
        require(not future.exists() and not future.is_symlink(),
                "future/formal identity unexpectedly exists: " + str(future))
    print(json.dumps({
        "schema": "m813_r17_source_closure_v1",
        "status": "PASS_M813_R17_SOURCE_CLOSURE__NO_EDA_EXECUTED",
        "python": __import__("sys").version.split()[0],
        "bash_n": "PASS", "function_closure": closure,
        "undefined_function_negative": negative["undefined"],
        "source_validation": source,
        "frozen_m803": "PASS", "future_identities_absent": True,
        "vcs_executed": False, "license_queried": False,
        "eda_executed": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
