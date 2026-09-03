#!/usr/bin/env python3
"""Independent, source-only M1884 hammer; never queries a license or runs EDA."""
from __future__ import print_function

import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
ROOT = HW.parent
CHECKER = HW / "system_simulator/scripts/check_m1882_m1880_c2_tsbg_b4_campaign_source.py"
SPEC = importlib.util.spec_from_file_location("m1884_independent_checker", str(CHECKER))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)

EXPECTED = {
    "runner": "29cdc882cc419b7dca525751d8e243a77316cb8f1ac79088004c4a8ab142bea1",
    "checker": "70b8c6d8882b469b27ab2abb17a908f13ea7a4e7c59a501f5d502320ae14ef84",
    "tests": "ea6a2e72f438c6bcd950d8b8ca9589c816f1aeecc5b91f69bf407b6863dece1f",
    "contract": "06c74e771b5dc51780386fa619527e7e4a9fb25db270f58c4aa02dcbc61bcb3c",
    "contract_sidecar": "90b8528b228b2dc4145c6433a64517dbf530da0ae6c1c5f556c2b4dfc2a23ac4",
    "contract_outer": "4b523254917539d1ef7e894a9cc561c5a4e4abc8a2c2b373ba995a9ec75baf40",
    "author_receipt": "3e9119d5ec843c4ab6e37659c5c330e6aa6bab7c6220d3ffe4be2f5c6d465fa4",
    "author_manifest": "84c07c2af66a345a7395a72a3f05a83ccab689d826e142ffe234c4676cbe0de3",
    "author_outer": "bda165759b2b10c2b19b4c8a6d815ffea81faf48926f13fc53ad08a06fea6058",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

PATHS = {
    "runner": CHECK.RUNNER,
    "checker": CHECKER,
    "tests": CHECK.TEST,
    "contract": CHECK.CONTRACT,
    "contract_sidecar": Path(str(CHECK.CONTRACT) + ".sha256"),
    "contract_outer": Path(str(CHECK.CONTRACT) + ".sha256.seal.sha256"),
    "author_receipt": HW / "reviews/m1882_m1881_m1880_c2_tsbg_b4_campaign_source_author_receipt_r1_20260902/author_receipt.json",
    "author_manifest": HW / "reviews/m1882_m1881_m1880_c2_tsbg_b4_campaign_source_author_receipt_r1_20260902/SHA256SUMS",
    "author_outer": HW / "reviews/m1882_m1881_m1880_c2_tsbg_b4_campaign_source_author_receipt_r1_20260902/SHA256SUMS.seal.sha256",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}

MUTATIONS = (
    ("helper_verify_authority_early_return",
     "def verify_authority():\n    exact(RUNNER,",
     "def verify_authority():\n    return\n    exact(RUNNER,"),
    ("helper_namespaces_fresh_early_return",
     "def namespaces_fresh():\n    fixed =",
     "def namespaces_fresh():\n    return\n    fixed ="),
    ("helper_collision_gate_early_return",
     "def collision_gate():\n    blocked =",
     "def collision_gate():\n    return\n    blocked ="),
    ("helper_resource_gate_early_return",
     "def resource_gate():\n    values =",
     "def resource_gate():\n    return\n    values ="),
    ("helper_run_tool_early_return",
     "def run_tool(command, cwd, timeout, output):\n    CHECK.validate_sources()",
     "def run_tool(command, cwd, timeout, output):\n    return\n    CHECK.validate_sources()"),
    ("helper_seal_dir_early_return",
     "def seal_dir(root):\n    rows =",
     "def seal_dir(root):\n    return\n    rows ="),
    ("helper_publish_early_return",
     "def publish_no_replace(source, destination):\n    libc =",
     "def publish_no_replace(source, destination):\n    return\n    libc ="),
    ("extra_unaccounted_license_query",
     "        state[\"phase\"] = \"ATTEMPT_CONSUMED\"",
     "        subprocess.run([str(LMUTIL), \"lmstat\", \"-a\", \"-c\", LICENSE_SERVER])\n"
     "        state[\"phase\"] = \"ATTEMPT_CONSUMED\""),
    ("license_query_faked_success",
     "license_check = subprocess.run(",
     "license_check = subprocess.CompletedProcess([], 0, stdout=b\"\") if True else subprocess.run("),
    ("extra_unaccounted_simv",
     "        if state[\"simv_runs\"] != COUNTS[\"simv_runs\"]:",
     "        subprocess.run([str(simv)])\n"
     "        if state[\"simv_runs\"] != COUNTS[\"simv_runs\"]:"),
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    current = dict((name, sha(path)) for name, path in PATHS.items())
    if current != EXPECTED:
        raise RuntimeError("M1884 exact identity drift")
    positive = CHECK.validate_sources()
    base = CHECK.source_texts()
    runner = base[CHECK.RUNNER]
    escaped = []
    rejected = []
    for name, old, new in MUTATIONS:
        if runner.count(old) != 1:
            raise RuntimeError("mutation anchor cardinality " + name)
        mutated = dict(base)
        mutated[CHECK.RUNNER] = runner.replace(old, new, 1)
        try:
            CHECK.validate_semantics(mutated)
        except CHECK.CheckFailure:
            rejected.append(name)
        else:
            escaped.append(name)
    license_pos = runner.index('license_check = subprocess.run(')
    attempt_pos = runner.index('ATTEMPT.mkdir()')
    value = {
        "status": "FAIL_CLOSED_M1884_INDEPENDENT_PROBE",
        "positive_checker_status": positive["status"],
        "exact_identity": current,
        "attempt_before_first_license_use": attempt_pos < license_pos,
        "license_precedes_attempt_in_current_runner": license_pos < attempt_pos,
        "independent_mutations": len(MUTATIONS),
        "escaped_mutations": escaped,
        "rejected_mutations": rejected,
        "authorization": {
            "query_license": False,
            "create_attempt": False,
            "run_vcs": False,
            "run_simv": False,
            "run_eda": False,
            "create_m1885": False,
            "create_m1886": False,
        },
    }
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
