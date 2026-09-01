#!/usr/bin/env python3
"""Finalize and double-seal the read-only M1606 result/semantics review."""

from __future__ import print_function

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
R36 = HERE / "cpython36_forensic.json"
R310 = HERE / "cpython310_forensic.json"
STATUS = (
    "PASS_M1606_RESULT_AUDIT__M1604_LEGAL_ACCEPT_POSTEDGE_"
    "COMBINATIONAL_FALSE_ERROR__NO_TOOL_AUTHORITY"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    require(type(value) is dict, "JSON root drift")
    return value


def write_json(name, value):
    (HERE / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main():
    r36 = load(R36)
    r310 = load(R310)
    require(R36.read_bytes() == R310.read_bytes(),
            "dual-runtime forensic outputs are not byte-identical")
    require(r36 == r310 and r36.get("status") == STATUS,
            "independent forensic verdict drift")

    execution = r36["m1604_execution"]
    simulation = execution["simulation"]
    semantics = r36["synchronous_semantics"]
    classification = r36["classification"]
    repair = r36["repair_comparison"]
    next_step = r36["unique_minimum_next_step"]

    require(execution["attempt_consumed"] is True and
            execution["compile"]["executed_commands"] == 1 and
            simulation["executed_commands"] == 1,
            "one-shot execution population drift")
    require(simulation["first_stop_cycle"] == 4 and
            simulation["first_difference_cycle"] == -1 and
            simulation["rtl_top_pns"] == "100" and
            simulation["mapped_top_pns"] == "100" and
            simulation["endpoint_faults"] == "all_zero" and
            simulation["registered_fault_taps"] == "all_zero",
            "M1604 diagnostic signature drift")
    require(classification == {
        "actual_protocol_violation_at_accepting_edge": False,
        "m1594_settle_repair_effective": True,
        "m1604_result": "FAILED_DIAGNOSTIC__SEMANTIC_FALSE_POSITIVE__NOT_CITABLE",
        "postaccept_combinational_protocol_error_pulse": True,
        "root": "LEGAL_READY_VALID_ACCEPT_REINTERPRETED_BY_POSTEDGE_STATE_WHILE_VALID_HELD",
        "rtl_mapped_difference": False,
        "stable_x": False,
    }, "classification drift")
    require(semantics["pre_edge"]["raw_accept"] == 1 and
            semantics["pre_edge"]["illegal_request"] == 0 and
            semantics["post_edge_settled"]["raw_accept"] == 0 and
            semantics["post_edge_settled"]["illegal_request"] == 1 and
            semantics["post_edge_settled"]["compactor_fault_q"] == 0,
            "edge-semantics proof drift")
    require(repair["pre_edge_sampling"]["verdict"] == "DO_NOT_USE_AS_FIX" and
            repair["post_edge_settled_sampling"]["verdict"] == "KEEP_AS_CHECKER" and
            repair["registered_fault_only_public_protocol_error"]["verdict"] ==
            "PREFERRED_SEMANTIC_REPAIR",
            "repair comparison drift")
    require(next_step["source_review_before_tools"] is True and
            next_step["resynthesis_eventually_required"] is True and
            next_step["vcs_authorized_now"] is False and
            next_step["dc_authorized_now"] is False and
            next_step["ptpx_authorized_now"] is False and
            next_step["file_edits_by_m1606"] is False,
            "authority boundary drift")

    mechanical = {
        "schema": "m1606_m1604_c2_result_semantics_mechanical_checks_r1_v1",
        "status": STATUS,
        "dual_runtime": {
            "cpython36_forensic_sha256": sha256(R36),
            "cpython310_forensic_sha256": sha256(R310),
            "byte_identical": True,
        },
        "execution": execution,
        "settled_sample": {
            "sample_time_ps": simulation["sample_time_ps"],
            "top_pns_rtl_mapped": "100/100",
            "decoded_top_pns": {
                "protocol_error": 1,
                "numeric_overflow": 0,
                "stale_response_seen": 0,
            },
            "endpoint_faults": "all_zero",
            "registered_fault_taps": "all_zero",
            "first_difference_cycle": -1,
        },
        "edge_semantics": semantics,
        "tool_execution_by_m1606": {
            "vcs": 0, "simv": 0, "dc": 0, "ptpx": 0,
        },
    }
    write_json("mechanical_checks.json", mechanical)

    review = {
        "schema": "m1606_m1604_c2_settled_result_semantics_independent_review_r1_v1",
        "status": STATUS,
        "score": 100,
        "identity": r36["identity"],
        "m1604_execution": execution,
        "verdict": {
            "m1604_citable": False,
            "rtl_mapped_pass": False,
            "rtl_mapped_difference": False,
            "stable_x": False,
            "actual_protocol_violation_at_accepting_edge": False,
            "classification": classification["root"],
            "plain_language": (
                "The raw beat was legal and accepted at the clock edge. The producer "
                "correctly kept raw_valid asserted until the following negedge, while "
                "raw_done_q advanced immediately after acceptance. Re-evaluating the "
                "same held beat against that postaccept state made combinational "
                "illegal_request pulse without setting any registered fault."
            ),
        },
        "synchronous_ready_valid_proof": semantics,
        "repair_comparison": repair,
        "unique_minimum_next_step": next_step,
        "authorization": {
            "source_only_successor_candidate": True,
            "independent_static_source_review_after_candidate": True,
            "vcs": 0,
            "simv": 0,
            "dc": 0,
            "ptpx": 0,
            "saif": 0,
            "mapped_retry": False,
            "note": (
                "A later RTL-vs-mapped run requires a new mapped netlist because the "
                "recommended repair changes RTL. M1606 does not authorize that work."
            ),
        },
        "claim_boundary": r36["claim_boundary"],
    }
    write_json("review.json", review)

    review_md = """# M1606 — M1604 C2 settled result and handshake-semantics review

Verdict: **M1604 is a consumed failed diagnostic, not an RTL/mapped PASS and
not citable. The cycle-4 stop is a post-accept combinational false protocol
error, not a protocol violation at the accepting edge and not an RTL/netlist
difference.** M1606 ran no VCS, `simv`, DC, SAIF, or PTPX and authorizes none.

## Execution and exact observation

The frozen M1604 identity consumed its only attempt and contains exactly one
VCS compile and one `k8_case0` simulation. Compilation has zero errors. The
simulation is clean for cycles 1--3, then stops at the 22.501 ns settled sample
in cycle 4 with `top_pns=100/100`. This field means protocol error is one while
numeric overflow and stale response are zero in both RTL and mapped DUTs.
`first_difference_cycle=-1`; both eight-bit endpoint-fault vectors and all six
registered fault/stale taps are zero. Thus M1594's 1 ps settle repair did its
job: the prior active-region mapped X is gone and both implementations agree.

## Ready/valid proof

The relevant timeline is:

| Time | Producer / state | Accept and legality | Public result |
|---:|---|---|---|
| 21.000 ns negedge | producer asserts `raw_valid=1`, four legal lanes and `raw_last=1` | token is active and `raw_done_q=0` | clean |
| 22.500 ns just before posedge | `raw_valid=1`, `raw_packet_legal=1` | `raw_accept=1`, `illegal_request=0` | `fault_q=0`, `protocol_error=0` |
| 22.501 ns settled | producer legally still holds `raw_valid=1`; accepted terminal beat has advanced `raw_done_q=1` | the same held beat is now reinterpreted as `!raw_packet_legal`, so `raw_accept=0`, `illegal_request=1` | `fault_q=0`, combinational `protocol_error=1` |
| 24.000 ns scheduled negedge | producer withdraws `raw_valid` | `illegal_request` returns to zero | no registered fault was ever set |

This is normal synchronous ready/valid behavior: the producer is permitted to
withdraw `valid` only after observing the accepting edge. The compactor instead
computes `protocol_error = fault_q || illegal_request` combinationally while its
post-edge state already declares the terminal packet done. The false pulse then
propagates through the frontend and core to the K8 top. All registered fault
taps remain zero because `illegal_request` was zero at the accepting edge.

## Three possible treatments

Pre-edge sampling would observe the legal transaction (`raw_accept=1`, no
error), but it is not a repair: it hides a real public-output pulse and a naive
posedge checker would recreate M1593's active-region gate-level race. Keep the
post-edge 1 ps checker; it is race-free and faithfully reports the implemented
interface.

The minimum semantic repair is to expose only the sticky registered fault:
change the compactor's public assignment from
`protocol_error = fault_q || illegal_request` to
`protocol_error = fault_q`. Preserve the `illegal_request` expression, its
`fault_q` latch, `raw_ready/raw_packet_legal` acceptance gate, M1601 settled
checker, stimulus, and all other behavior. A genuinely illegal request present
at a sampling edge still sets `fault_q` and becomes visible after that edge;
malformed input remains unable to handshake through `raw_ready`.

## Unique next step and boundary

Author one source-only RTL successor containing only that public-output change,
then submit it to an independent static source review. Do not change the
checker to pre-edge sampling. Because the RTL identity changes, any eventual
RTL/mapped comparison requires resynthesis and a new mapped netlist, but M1606
authorizes no VCS, simulation, DC, PTPX, SAIF, or retry. M1604 remains
`FAILED_DIAGNOSTIC__SEMANTIC_FALSE_POSITIVE__NOT_CITABLE`; it creates no timing,
power, PPA, speedup, system, or paper claim. docs/359 remains `dedde7ce...`.
"""
    (HERE / "review.md").write_text(review_md, encoding="utf-8")
    (HERE / "RUN_COMPLETE.txt").write_text(STATUS + "\n", encoding="ascii")

    members = [
        "RUN_COMPLETE.txt", "cpython310_forensic.json",
        "cpython36_forensic.json", "finalize_review.py",
        "independent_static_forensic.py", "mechanical_checks.json",
        "review.json", "review.md",
    ]
    inner = "".join(sha256(HERE / name) + "  " + name + "\n"
                    for name in members)
    (HERE / "SHA256SUMS").write_text(inner, encoding="ascii")
    (HERE / "SHA256SUMS.seal.sha256").write_text(
        sha256(HERE / "SHA256SUMS") + "  SHA256SUMS\n", encoding="ascii")
    print(STATUS)


if __name__ == "__main__":
    main()
