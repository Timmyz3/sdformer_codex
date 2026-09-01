#!/usr/bin/env python3
"""Finalize and double-seal the M1600 different-author hammer."""

from __future__ import print_function

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                       sort_keys=True) + "\n", encoding="utf-8")


qa36 = json.loads((HERE / "cpython36_hammer.json").read_text(encoding="utf-8"))
qa310 = json.loads((HERE / "cpython310_hammer.json").read_text(encoding="utf-8"))
if qa36 != qa310:
    raise RuntimeError("dual-runtime hammer results differ")
expected_status = ("PASS_M1600_M1595_DIFFERENT_AUTHOR_HAMMER__"
                   "ONE_D0_CALL0_THREE_CONFIG_PILOT_ACTUAL_AUTHORIZED__NOT_EXECUTED")
if qa36.get("status") != expected_status:
    raise RuntimeError("independent hammer did not authorize the narrow pilot")

mechanical = {
    "schema": "m1600_m1595_decoder_runner_independent_mechanical_checks_r1_v1",
    "status": "PASS",
    "runtimes": [
        {"implementation": "CPython", "version": "3.6.8",
         "author_unit_tests_replayed": "6/6", "independent_mutations": "62/62"},
        {"implementation": "CPython", "version": "3.10.18",
         "author_unit_tests_replayed": "6/6", "independent_mutations": "62/62"},
    ],
    "dual_runtime_hammer_byte_identical": True,
    "synthetic_success_configurations": 3,
    "synthetic_failure_child_ordinal": 1,
    "failed_attempt_retry_reached_launcher": False,
    "production_preflight_read_only": True,
    "execution_by_m1600": qa36["execution_by_m1600"],
}
write_json("mechanical_checks.json", mechanical)

review = {
    "schema": "m1600_m1595_decoder_one_process_per_config_runner_independent_hammer_r1_v1",
    "status": expected_status,
    "score": 99,
    "identity": qa36["identity"],
    "verified": {
        "m1592_tree_and_outer_seal": True,
        "m1583_exact_source": True,
        "m1595_contract_inner_and_outer_seal": True,
        "m1595_author_tree_and_outer_seal": True,
        "ordered_three_nonproduct_configurations": True,
        "fixed_fresh_python_command_per_configuration": True,
        "exactly_one_m1583_entry_per_child": True,
        "attempt_consumed_before_first_child": True,
        "failure_permanently_consumes_attempt": True,
        "success_and_failure_no_replace_publish": True,
        "child_envelope_ticket_and_digest_bound": True,
        "configuration_and_resource_exact": True,
        "request_kind_conservation": True,
        "common_commit_sequence": True,
        "rss_gate_positive_strict_lt_8gib_and_monotonic": True,
    },
    "hammer": qa36["synthetic_hammer"],
    "execution_by_review": qa36["execution_by_m1600"],
    "authorization": qa36["authorization"],
    "next_gate": ("Consume exactly one M1595 global attempt for D0/call0 in the "
                  "three frozen non-product configurations. Independently hammer "
                  "the sealed result before any performance or paper claim."),
    "claim_boundary": qa36["claim_boundary"],
}
write_json("review.json", review)

review_md = """# M1600 — M1595 decoder one-process-per-config runner independent hammer

Verdict: **PASS different-author source hammer. Authorize exactly one global
actual pilot attempt for D0/call0 across the three frozen non-product
configurations, one fresh fixed Python child per configuration. Do not execute
from this review. This is not the 120-call population.**

## Identity and scope

M1595 pins the exact M1583 engine (`f92c91f0...`), the fully sealed M1592
engineering review, the fixed CPython 3.10.18 executable, docs/359, and resource
manifest `64661d82...`. Both M1592 and the M1595 author receipt pass complete
member-manifest and outer-seal verification. The M1595 contract passes its
inner and outer seals.

The only admitted order is:

1. `DENSE_TYPED_K8`
2. `BIT_EQUAL_SERVICE_K1X8`
3. `BIT_TYPED_K8`

`PRODUCT_CAPTURE_TYPED_K8` is rejected before the M1583 entry. Each child
command starts the pinned Python binary on the pinned M1595 source with a clean
environment, a target-bound private ticket, and exactly one
`M1583.one_shot_worker_entry(config)` call. The parent requires three distinct
child PIDs and tickets. This is structural source admission; the hammer itself
started no real child and opened no payload.

## Attempt and result conservation

The global attempt marker is created with exclusive-create before the first
child. Success and failure publish through `renameat2(RENAME_NOREPLACE)`. A
synthetic second-child failure left the attempt consumed, published a sealed
failure tree, and a second invocation was rejected before reaching the
launcher. A successful synthetic run likewise rejected reuse.

Every child envelope binds parent PID, child PID, configuration, target ticket,
M1583 source, and a canonical result digest. M1583's unchanged result gate
checks exact configuration/resource identity, positive cycles/requests,
request-count equals the sum of kind counts, nonnegative byte counts,
address/commit/payload digests, exact D0/call0/T10 scope, nonmaterialized
streaming, positive RSS gate calls, strict RSS `< 8,388,608 KiB`, and monotonic
RSS maxima. The parent additionally requires one common resource manifest and
one common commit sequence across the three configurations.

The independent hammer rejected 62/62 mutations under CPython 3.6.8 and
62/62 under CPython 3.10.18; the JSON reports are byte-identical. The existing
author suite was independently replayed at 6/6 on both runtimes. All success
and failure exercises used an injected synthetic launcher. Actual worker call,
payload open, GPU, and EDA counts are all zero.

## Narrow release

The sealed authorization is one invocation of M1595 `--run`, consuming one
global attempt and exactly these three fresh child processes for the same
`decoder_stage=D0`, `module_ordinal=0`, `call_ordinal=0`, `timesteps=10` pilot.
It does not authorize product capture, retry after any failure, production,
the full 120-call population, GPU, RTL, or EDA work.

The future result is still diagnostic-only and must receive a separate
independent result hammer. Until then there is no citable cycle, traffic,
speedup, energy, or paper result. M1600 preserves docs/359 at SHA-256
`dedde7ce...` and changes no author file.
"""
(HERE / "review.md").write_text(review_md, encoding="utf-8")

(HERE / "RUN_COMPLETE.txt").write_text(
    expected_status + "\n", encoding="ascii")

members = [
    "RUN_COMPLETE.txt", "cpython310_hammer.json", "cpython36_hammer.json",
    "finalize_review.py", "independent_hammer.py", "mechanical_checks.json",
    "review.json", "review.md",
]
inner = "".join(sha256(HERE / name) + "  " + name + "\n" for name in members)
(HERE / "SHA256SUMS").write_text(inner, encoding="ascii")
(HERE / "SHA256SUMS.seal.sha256").write_text(
    sha256(HERE / "SHA256SUMS") + "  SHA256SUMS\n", encoding="ascii")

