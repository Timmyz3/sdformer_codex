#!/usr/bin/env python3
"""Python-3.6-compatible fail-closed source audit for M803/C2 R16.

This is not an HDL compiler and never invokes VCS, simv, lmutil, DC, or any
other EDA tool.  It closes custom runner functions, exact source identities,
filelists, frozen ancestors, additive namespace boundaries, and negative
undefined-function detection.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


DEF_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{")
HEREDOC_RE = re.compile(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")
CUSTOM_PREFIX = (
    "expect_", "verify_", "json_", "seal_", "failure_", "trace_",
    "collision_", "resource_", "compile_",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    if terminator is not None:
        raise RuntimeError("unterminated heredoc")
    return "\n".join(output) + "\n"


def command_words(text):
    pattern = re.compile(
        r"(?:^\s*|[;&|]\s*|\(\s*|\$\(\s*|\b(?:if|elif|while|until|then|do|else)\s+)"
        r"(?:!\s*)?(?:[A-Za-z_][A-Za-z0-9_]*=[^\s;|&]+\s+)*"
        r"([A-Za-z_][A-Za-z0-9_]*)"
    )
    words = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0]
        if DEF_RE.match(line):
            line = DEF_RE.sub("{", line, count=1)
        for match in pattern.finditer(line):
            words.append((number, match.group(1)))
        trap_match = re.search(r"\btrap\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if trap_match:
            words.append((number, trap_match.group(1)))
    return words


def function_audit(text):
    code = strip_heredocs(text)
    definitions = Counter()
    for line in code.splitlines():
        match = DEF_RE.match(line)
        if match:
            definitions[match.group(1)] += 1
    calls = [(line, word) for line, word in command_words(code)
             if word in definitions or word.startswith(CUSTOM_PREFIX)]
    undefined = sorted(set(word for _, word in calls
                           if definitions.get(word, 0) == 0))
    duplicate = sorted(name for name, count in definitions.items() if count != 1)
    return {
        "definitions": dict(sorted(definitions.items())),
        "custom_calls": [{"line": line, "name": word} for line, word in calls],
        "undefined_custom_calls": undefined,
        "duplicate_definitions": duplicate,
        "pass": not undefined and not duplicate,
    }


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hw_root", type=Path)
    args = parser.parse_args()
    root = args.hw_root.resolve(strict=True)
    runner = root / "dc_handoff/scripts/run_vcs_m803_c2_r16_channel_split_exact_sha.sh"
    contract_path = root / "contracts/m803_c2_r16_channel_split_source_only_contract_r1_20260828.json"
    require(runner.is_file() and not runner.is_symlink(), "runner is not regular")
    require(contract_path.is_file() and not contract_path.is_symlink(), "contract missing")

    bash_n = subprocess.run(["/usr/bin/bash", "-n", str(runner)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    require(bash_n.returncode == 0, "bash -n failed: " + bash_n.stderr)

    runner_text = runner.read_text(encoding="utf-8")
    closure = function_audit(runner_text)
    require(closure["pass"], "runner function closure failed")
    mutated = runner_text.replace("verify_source_contract() {",
                                  "verify_source_contract_deleted() {", 1)
    negative = function_audit(mutated)
    require(not negative["pass"] and
            "verify_source_contract" in negative["undefined_custom_calls"],
            "undefined-function negative mutation escaped")

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require(contract.get("status") == "SOURCE_ONLY__NO_VCS_AUTHORIZATION",
            "contract status")
    require(contract.get("authorization", {}).get("launch_now") is False,
            "contract launch_now")
    require(contract.get("runner_sha256") == digest(runner), "runner SHA binding")
    for rel, expected in sorted(contract.get("source_sha256", {}).items()):
        path = (root / rel).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise RuntimeError("source path escapes root: " + rel)
        require(path.is_file() and not path.is_symlink(), "source non-regular: " + rel)
        require(digest(path) == expected, "source SHA mismatch: " + rel)

    filelists = [
        "dc_handoff/filelists/date_m803_c2_r16_channel_split_adapter_attacks_vcs.f",
        "dc_handoff/filelists/date_m803_c2_r16_channel_split_k8_vs_k1x8_vcs.f",
        "dc_handoff/filelists/date_m803_c2_r16_channel_split_three_axis_logic_only_dc.f",
    ]
    filelist_entries = {}
    for rel in filelists:
        lines = [line.strip() for line in (root / rel).read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        require(len(lines) == len(set(lines)), "duplicate filelist entry: " + rel)
        for entry in lines:
            path = (root / entry).resolve()
            require(path.is_file() and not path.is_symlink(), "filelist source missing: " + entry)
        filelist_entries[rel] = lines

    require(digest(root / "rtl_m490/m490_fc2_bundle_to_8bank_cutthrough_adapter.sv") ==
            "597e4d9e9a606afa58111d01be8e8304e4fb5d4656cabdd4da9fca4b8393f43b",
            "frozen M490 drift")
    require(digest(root / "rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv") ==
            "44f7df331af66ba62fadf5e336b9c0c00d00f809e215aa8e091e9de011c5627e",
            "frozen M499 drift")
    require(digest(root / "docs/359_DATE终局冻结_20260813.md") ==
            "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
            "docs/359 drift")

    new_files = list((root / "rtl_m803").glob("*.sv")) + \
        list((root / "verif_m803").glob("*.sv")) + list((root / "tb_m803").glob("*.sv"))
    require(all("\\n//" not in path.read_text(encoding="utf-8") for path in new_files),
            "literal escaped newline remains")
    adapter = (root / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv").read_text(encoding="utf-8")
    require("req_slot_open = !slot_valid_q[core_req_slot]" in adapter and
            "core_rsp_accept && complete_slot == core_req_slot" in adapter,
            "M490 same-cycle slot reuse was not preserved")
    require(adapter.index("if (response_channel_open)") <
            adapter.index("if (request_channel_open)", adapter.index("always_ff")),
            "response/request state ordering not split")
    k8 = (root / "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv").read_text(encoding="utf-8")
    matched = (root / "rtl_m803/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24.sv").read_text(encoding="utf-8")
    require("m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter" in k8,
            "new K8 adapter not bound")
    require("m519_fc2_registered_release_standalone_raw4_acc24" in k8,
            "frozen M519 K8 arithmetic core not retained")
    require("m519_fc2_k1_registered_release_8bank_raw4_acc24" in matched and
            "m519_fc2_k1x8_registered_release_raw4_acc24" in matched,
            "frozen K1/K1x8 wrapper bindings missing")

    result = {
        "schema": "m803_r16_source_closure_v1",
        "status": "PASS_SOURCE_ONLY__NO_EDA_EXECUTED",
        "python": sys.version.split()[0],
        "runner_sha256": digest(runner),
        "contract_sha256": digest(contract_path),
        "bash_n": "PASS",
        "function_closure": closure,
        "undefined_function_negative": negative["undefined_custom_calls"],
        "filelists": filelist_entries,
        "frozen_ancestors": "PASS",
        "additive_namespace": "PASS",
        "vcs_executed": False,
        "dc_executed": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
