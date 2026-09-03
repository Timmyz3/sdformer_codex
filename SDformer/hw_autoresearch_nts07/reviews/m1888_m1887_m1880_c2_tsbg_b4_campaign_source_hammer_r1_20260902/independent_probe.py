#!/usr/bin/env python3
"""Independent source-only M1888 governance hammer; never runs tools or EDA."""
from __future__ import print_function

import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
CHECKER = HW / "system_simulator/scripts/check_m1887_m1880_c2_tsbg_b4_campaign_source.py"
SPEC = importlib.util.spec_from_file_location("m1888_independent_checker", str(CHECKER))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)

EXPECTED = {
    "runner": "ef7196c10c3d780a4abb168550761dfe3d409717f28a0ee5b8c20917401016d5",
    "checker": "f9a63ecfa51ef89b81959826c86e6380ca222c311aec2ffab9a1ac6a5e22f8ce",
    "tests": "362345e4be1597af1664531d023eea3d78f8f15d98abe90849d030cc073c2767",
    "contract": "d92d118174279c2c73971d8257eb197de1c550af339ebce6c3a029ce770997b6",
    "contract_sidecar": "e096f52c56096ee28cdedd6c62d8c681261bbd9ebbc1d558d6a4fd995ee85f90",
    "contract_outer": "bbfc13660923d672d27d216a9924546ee7ea7d7747b5a73d98253e6e4260f04d",
    "author_receipt": "734af941416846637821193caf87f0fcaff3d7628972eceb8ebf5b93e2f7ac75",
    "author_manifest": "aae65c732d48ee03ab772080bc0a0a0d9773040e7caca172363cd9393a095c2d",
    "author_outer": "649cf7fe6966446fe0c2b3a70cc6400a31e4ea4213c29181cc616f5c033c14ea",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

PATHS = {
    "runner": CHECK.RUNNER,
    "checker": CHECKER,
    "tests": CHECK.TEST,
    "contract": CHECK.CONTRACT,
    "contract_sidecar": Path(str(CHECK.CONTRACT) + ".sha256"),
    "contract_outer": Path(str(CHECK.CONTRACT) + ".sha256.seal.sha256"),
    "author_receipt": HW / "reviews/m1887_m1884_m1882_m1880_c2_tsbg_b4_campaign_successor_source_author_receipt_r1_20260902/author_receipt.json",
    "author_manifest": HW / "reviews/m1887_m1884_m1882_m1880_c2_tsbg_b4_campaign_successor_source_author_receipt_r1_20260902/SHA256SUMS",
    "author_outer": HW / "reviews/m1887_m1884_m1882_m1880_c2_tsbg_b4_campaign_successor_source_author_receipt_r1_20260902/SHA256SUMS.seal.sha256",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}

ATTACKS = (
    ("pre_attempt_alias_subprocess_run",
     "        state[\"phase\"] = \"ATTEMPT_CONSUMED\"",
     "        hidden = subprocess.run\n"
     "        hidden([str(LMUTIL), \"lmstat\", \"-a\", \"-c\", LICENSE_SERVER])\n"
     "        state[\"phase\"] = \"ATTEMPT_CONSUMED\""),
    ("pre_attempt_subprocess_getoutput",
     "        state[\"phase\"] = \"ATTEMPT_CONSUMED\"",
     "        subprocess.getoutput(str(LMUTIL) + \" lmstat -a -c \" + LICENSE_SERVER)\n"
     "        state[\"phase\"] = \"ATTEMPT_CONSUMED\""),
    ("extra_alias_license_after_accounted_query",
     "        if state[\"license_queries\"] != COUNTS[\"license_queries\"]:",
     "        hidden = subprocess.run\n"
     "        hidden([str(LMUTIL), \"lmstat\", \"-a\", \"-c\", LICENSE_SERVER])\n"
     "        if state[\"license_queries\"] != COUNTS[\"license_queries\"]:"),
    ("extra_getoutput_license_after_accounted_query",
     "        if state[\"license_queries\"] != COUNTS[\"license_queries\"]:",
     "        subprocess.getoutput(str(LMUTIL) + \" lmstat -a -c \" + LICENSE_SERVER)\n"
     "        if state[\"license_queries\"] != COUNTS[\"license_queries\"]:"),
    ("extra_os_spawnv_simv",
     "        if state[\"simv_runs\"] != COUNTS[\"simv_runs\"]:",
     "        os.spawnv(os.P_WAIT, str(simv), [str(simv)])\n"
     "        if state[\"simv_runs\"] != COUNTS[\"simv_runs\"]:"),
    ("fake_all_subprocess_results_by_monkeypatch",
     "        state[\"phase\"] = \"LICENSE_PREFLIGHT\"",
     "        subprocess.run = lambda *args, **kwargs: type(\"R\", (), "
     "{\"stdout\": b\"fake\", \"returncode\": 0})()\n"
     "        state[\"phase\"] = \"LICENSE_PREFLIGHT\""),
    ("shadow_verify_authority",
     "        verify_authority()\n        CHECK.validate_sources()",
     "        verify_authority = lambda: None\n"
     "        verify_authority()\n        CHECK.validate_sources()"),
    ("shadow_namespaces_fresh",
     "        verify_authority()\n        CHECK.validate_sources()",
     "        namespaces_fresh = lambda: None\n"
     "        verify_authority()\n        CHECK.validate_sources()"),
    ("shadow_collision_gate",
     "        verify_authority()\n        CHECK.validate_sources()",
     "        collision_gate = lambda: None\n"
     "        verify_authority()\n        CHECK.validate_sources()"),
    ("shadow_resource_gate",
     "        verify_authority()\n        CHECK.validate_sources()",
     "        resource_gate = lambda: None\n"
     "        verify_authority()\n        CHECK.validate_sources()"),
    ("shadow_seal_dir",
     "        verify_authority()\n        CHECK.validate_sources()",
     "        seal_dir = lambda root: None\n"
     "        verify_authority()\n        CHECK.validate_sources()"),
    ("shadow_publish_no_replace",
     "        seal_dir(STAGE)\n        publish_no_replace(STAGE, RESULT)",
     "        publish_no_replace = lambda source, destination: None\n"
     "        seal_dir(STAGE)\n        publish_no_replace(STAGE, RESULT)"),
    ("shadow_attempt_terminal_gate",
     "        attempt_terminal_gate(state)\n        return 0",
     "        attempt_terminal_gate = lambda state: None\n"
     "        attempt_terminal_gate(state)\n        return 0"),
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
        raise RuntimeError("M1888 exact identity drift")
    positive = CHECK.validate_sources()
    base = CHECK.source_texts()
    runner = base[CHECK.RUNNER]
    escaped = []
    rejected = []
    for name, old, new in ATTACKS:
        if runner.count(old) != 1:
            raise RuntimeError("attack anchor cardinality " + name)
        mutated = dict(base)
        mutated[CHECK.RUNNER] = runner.replace(old, new, 1)
        try:
            CHECK.validate_semantics(mutated)
        except CHECK.CheckFailure:
            rejected.append(name)
        else:
            escaped.append(name)
    value = {
        "status": "FAIL_CLOSED_M1888_INDEPENDENT_PROBE",
        "positive_checker_status": positive["status"],
        "exact_identity": current,
        "independent_attacks": len(ATTACKS),
        "escaped_attacks": escaped,
        "rejected_attacks": rejected,
        "authorization": {
            "query_license": False,
            "create_attempt": False,
            "run_vcs": False,
            "run_simv": False,
            "run_eda": False,
            "create_m1889": False,
            "create_m1890": False,
        },
    }
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    if len(escaped) != len(ATTACKS) or rejected:
        raise RuntimeError("unexpected independent-hammer disposition")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
