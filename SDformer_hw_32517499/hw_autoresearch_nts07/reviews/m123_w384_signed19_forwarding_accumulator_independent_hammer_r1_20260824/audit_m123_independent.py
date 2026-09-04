#!/usr/bin/env python3
"""Machine audit for M123 production and independent commercial-VCS evidence."""

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REVIEW = Path(__file__).resolve().parent
SEALED = ROOT / "dc_handoff/runs/m123_w384_signed19_forwarding_accumulator_vcs_r1_sealed_20260824"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_manifest(path: Path) -> dict:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(None, 1)
        target = Path(name.strip())
        if not target.is_absolute():
            target = ROOT / target
        actual = sha256(target)
        rows.append({"path": str(target), "expected": expected, "actual": actual, "match": expected == actual})
    require(rows and all(row["match"] for row in rows), f"manifest mismatch: {path}")
    return {"path": str(path), "entries": len(rows), "all_match": True, "rows": rows}


def parse_pass(log: Path, prefix: str) -> dict:
    lines = [line for line in log.read_text().splitlines() if line.startswith(prefix)]
    require(len(lines) == 1, f"expected one PASS line in {log}")
    tokens = {}
    for token in lines[0].split():
        if "=" in token:
            key, value = token.split("=", 1)
            if re.fullmatch(r"-?[0-9]+", value):
                tokens[key] = int(value)
            elif value in ("true", "false"):
                tokens[key] = value == "true"
            else:
                tokens[key] = value
    return {"line": lines[0], "tokens": tokens}


def covers(report: Path) -> dict:
    values = {
        name: int(matches)
        for name, matches in re.findall(r"\.([A-Za-z0-9_]+),\s+\d+ attempts,\s+(\d+) match", report.read_text())
    }
    require(values, f"no covers in {report}")
    return values


def no_failures(*paths: Path) -> bool:
    pattern = re.compile(r"Warning-\[|Error-\[|^Error|^Fatal|failed at|Offending", re.I | re.M)
    return all(pattern.search(path.read_text(encoding="utf-8", errors="replace")) is None for path in paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "contract": ROOT / "contracts/m123_w384_signed19_forwarding_accumulator_vcs_contract_r1_20260824.json",
        "core_rtl": ROOT / "rtl_m123/m123_w384_signed19_forwarding_accumulator_frontend.sv",
        "adapter_rtl": ROOT / "rtl_m123/m123_w384_signed19_forwarding_lane_sliced_accumulator_adapter.sv",
        "sva": ROOT / "verif_m123/m123_w384_signed19_forwarding_lane_sliced_accumulator_assertions.sv",
        "production_tb": ROOT / "tb_m123/tb_m123_w384_signed19_forwarding_lane_sliced_accumulator.sv",
        "production_filelist": ROOT / "dc_handoff/filelists/date_m123_w384_signed19_forwarding_lane_accumulator_directed_vcs.f",
        "production_runner": ROOT / "dc_handoff/scripts/run_vcs_m123_w384_signed19_forwarding_accumulator.sh",
        "m120_review_manifest": ROOT / "reviews/m120_integrated_pwp_tail_mapper_signed19_accumulator_independent_hammer_r1_20260824/manifest.sha256",
        "docs_359": ROOT / "docs/359_DATE终局冻结_20260813.md",
        "independent_tb": REVIEW / "tb_m123_independent_hammer.sv",
        "independent_filelist": REVIEW / "m123_independent.f",
        "integrated_shim": REVIEW / "m118_name_m123_forwarding_shim.sv",
        "integrated_tb": REVIEW / "tb_m120_with_m123_integrated_hammer.sv",
        "integrated_filelist": REVIEW / "m123_m120_integrated_review.f",
    }
    expected = {
        "contract": "63432933d974b277453545118ac02f5d8a803987f8102982e56ee70177eb3f87",
        "core_rtl": "7729848c8172b9f3f768cac1b6ce3bf310b9f9b1a1e8def8ea3725c4b7356adc",
        "adapter_rtl": "a040675cb03f69edeb24e321ea3e163f49c9c9eadebb08f7c0c94ce1dbd963e7",
        "sva": "2e4333d7a19f1adfa11f28d0a5ee1ee49efccd32711ea83b845c76032b45137f",
        "production_tb": "7a198caed3e0cb90eb9a07db2fe5168826681795d4fd5717f071a506917a4a58",
        "production_filelist": "7072f0a32a2efe78d9690adef462fdd70f7c3e07c1aaa55253f0d2e8e2eaaacb",
        "production_runner": "9aeb9d0b06457bf86ab1d24e1934f67c255e4a028fee51cde4f6d52e73c2f79c",
        "m120_review_manifest": "51ad53084fd73b64c3e7bf902ea72313bf0f4df660adaf4124c08cb2cb8116f1",
        "docs_359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
        "independent_tb": "b3a723e52714de99c2d7bd35a12941cb9fb24715bc7de39a9973bf5f3b9c90c3",
        "independent_filelist": "f96f8a3439f238a3a8ad5b986e457cd37091ef4ab8494e011c031724cfbebbd2",
        "integrated_shim": "fadbb8068ede7d673bb7a936bca2c9a2f3268d3849658b5810824c2c40284cb6",
        "integrated_tb": "fda4b5dcf9ac810dc6498c2f50241f4b2ef4b54170aac5e05e01187dce53ce79",
        "integrated_filelist": "2fc53b2c17cff8520b2b6824a3d391f6e4bd4dcdb8bde003121e09cb4fdc107f",
    }
    observed = {name: sha256(path) for name, path in paths.items()}
    hash_checks = {name: observed[name] == value for name, value in expected.items()}
    require(all(hash_checks.values()), f"input SHA mismatch: {hash_checks}")

    contract = json.loads(paths["contract"].read_text())
    require(contract["admission"]["reset_recovery"] is False, "reset boundary drift")
    require(contract["admission"]["physical_speedup"] is False, "physical claim drift")
    require(contract["admission"]["system_speedup"] is False, "system claim drift")

    sealed_pass = parse_pass(SEALED / "sim.raw.log", "PASS M123 W384 forwarding")
    sealed_covers = covers(SEALED / "assert.report")
    require(sealed_pass["tokens"]["updates"] == 1072, "sealed update count drift")
    require(sealed_pass["tokens"]["positive_memory_writes"] == 1072, "sealed write count drift")
    require(sealed_pass["tokens"]["same_address_accept_pairs"] == 15, "sealed same-address drift")
    require(sealed_pass["tokens"]["same_address_forward_read_suppressed"] == 15, "sealed forwarding drift")
    require(sealed_covers["cp_same_address_forward_chain"] == 14, "sealed forward cover drift")
    require(no_failures(SEALED / "compile.raw.log", SEALED / "sim.raw.log", SEALED / "assert.report"), "sealed failure text")

    sealed_rerun = REVIEW / "sealed_vcs_rerun"
    rerun_pass = parse_pass(sealed_rerun / "sim.raw.log", "PASS M123 W384 forwarding")
    require(rerun_pass["line"] == sealed_pass["line"], "sealed rerun PASS drift")
    require(covers(sealed_rerun / "assert.report") == sealed_covers, "sealed rerun covers drift")
    require(no_failures(sealed_rerun / "compile.raw.log", sealed_rerun / "sim.raw.log", sealed_rerun / "assert.report"), "sealed rerun failure")

    independent = REVIEW / "independent_vcs"
    independent_pass = parse_pass(independent / "sim.raw.log", "PASS M123 independent hammer")
    independent_covers = covers(independent / "assert.report")
    it = independent_pass["tokens"]
    for key, value in {
        "positive_updates": 16,
        "positive_writes": 16,
        "positive_write_lane_checks": 1536,
        "commits": 6144,
        "commit_lane_checks": 589824,
        "same_address_pairs": 6,
        "same_address_reads_suppressed": 6,
        "forwarded_overflow_attacks": 2,
        "invalid_row_attacks": 1,
        "reset_edge_write_enable": 1,
        "reset_edge_accept": 1,
        "reset_physical_writes": 1,
    }.items():
        require(it[key] == value, f"independent counter drift: {key}")
    require(it["original_m120_two_event_closed"] is True, "two-event closure drift")
    require(it["pending_sum_data_exact"] is True, "pending sum check drift")
    require(it["reset_quiescence"] is False, "reset finding drift")
    require(independent_covers["cp_same_address_forward_chain"] == 1, "independent forward cover drift")
    require(no_failures(independent / "compile.raw.log", independent / "sim.raw.log", independent / "assert.report"), "independent failure")

    latency_log = (independent / "macro_latency2_negative.raw.log").read_text()
    require("Fatal:" in latency_log and "M123 hammer forwarded/pending sum mismatch" in latency_log, "two-cycle boundary not detected")
    require("PASS M123 independent hammer" not in latency_log, "two-cycle negative falsely passed")

    integrated = REVIEW / "m120_integrated_vcs"
    integrated_pass = parse_pass(integrated / "sim.raw.log", "PASS M123 integrated M120 counterexample")
    integrated_covers = covers(integrated / "assert.report")
    jt = integrated_pass["tokens"]
    require(jt["same_address_events_accepted"] == 2, "integrated service count drift")
    require(jt["same_address_mapped_updates"] == 2, "integrated update count drift")
    require(jt["same_address_updates_written"] == 2, "integrated write count drift")
    require(jt["same_address_lane_checks"] == 96, "integrated numeric check drift")
    require(jt["same_address_accept_then_loss_closed"] is True, "integrated P0 closure drift")
    require(no_failures(integrated / "compile.raw.log", integrated / "sim.raw.log", integrated / "assert.report"), "integrated failure")

    rtl = paths["core_rtl"].read_text()
    sva = paths["sva"].read_text()
    static = {
        "forward_condition_checks_pending_block_and_row": all(fragment in rtl for fragment in (
            "same_address_rdw_forward = update_pipe_valid_q && update_valid",
            "update_block == update_pipe_block_q",
            "update_row == update_pipe_row_q",
        )),
        "forward_base_is_prior_computed_write_vector": "update_pipe_base_forward_data_q\n                        <= update_write_vector;" in rtl,
        "overflow_suppresses_entire_pending_write": "if (update_pipe_valid_q && !request_fault_q\n                && !update_pipe_overflow) begin" in rtl,
        "memory_write_enable_explicitly_gated_by_reset": "if (!rst_core && update_pipe_valid_q" in rtl,
        "ready_accept_explicitly_gated_by_reset": "assign update_ready = !rst_core" in rtl,
        "production_sva_disable_iff_reset_count": sva.count("disable iff (rst_core)"),
        "production_sva_has_reset_quiescence_property": "reset_quies" in sva.lower(),
        "production_sva_every_accept_property_allows_protocol_error_escape": "update_accept |=> lane_mem_wr_en || protocol_error" in sva,
    }
    require(static["forward_condition_checks_pending_block_and_row"], "forward condition drift")
    require(static["forward_base_is_prior_computed_write_vector"], "forward data drift")
    require(static["overflow_suppresses_entire_pending_write"], "overflow guard drift")

    output = {
        "schema": "m123_w384_signed19_forwarding_accumulator_independent_hammer_audit_v1",
        "status": "PASS_RESET_FREE_FORWARDING_AND_REVIEW_INTEGRATED_COUNTEREXAMPLE_WITH_P1_RESET_QUIESCENCE",
        "score": 86,
        "severity_counts": {"P0": 0, "P1": 1, "P2": 3},
        "input_hashes": observed,
        "input_hash_checks": hash_checks,
        "manifests": {
            "sealed_output": verify_manifest(SEALED / "output_sha256.txt"),
            "independent_vcs_output": verify_manifest(REVIEW / "vcs_output.sha256"),
            "integrated_vcs_output": verify_manifest(integrated / "output.sha256"),
        },
        "production_sealed": {"pass": sealed_pass, "covers": sealed_covers},
        "independent_standalone": {
            "pass": independent_pass,
            "covers": independent_covers,
            "macro_latency2_negative_detected": True,
            "positive_accept_write_conservation": "16/16",
            "positive_pending_sum_lane_checks": 1536,
            "full_commit_lane_checks": 589824,
        },
        "review_only_m120_integration": {
            "pass": integrated_pass,
            "covers": integrated_covers,
            "same_address_events_mapped_written": "2/2/2",
            "same_address_lane_checks": 96,
            "production_m120_rtl_modified": False,
            "review_only_name_shim": True,
        },
        "static_checks": static,
        "findings": [
            {
                "id": "P1-1",
                "severity": "P1",
                "title": "Reset is not externally quiescent and can write the lane memory",
                "evidence": "After one legal update accept, asserting synchronous reset before its write produced lane_mem_wr_en=1 on the reset edge and one physical memory write. Holding window_start_valid while reset remained high produced window_start_accept=1 even though reset prevented state capture. All production assertions are disabled during reset.",
                "impact": "A macro can be mutated during reset and an upstream block can observe a handshake that the accumulator does not retain. This is outside the frozen reset-free admission, but blocks a robust integration contract.",
                "required_fix": "Gate all ready/accept/commit_valid and macro read/write enables with !rst_core, add reset-quiescence SVA, and define whether an update accepted immediately before reset is aborted or drained before reset acknowledgement.",
            },
            {
                "id": "P2-1",
                "severity": "P2",
                "title": "M120 closure is demonstrated only by a review-only substitution",
                "evidence": "The frozen M120 wrapper still instantiates M118. A review-only same-name shim substituted M123 without editing production and replayed the exact M120 hammer: 2 services, 2 mapped updates, 2 writes, 96 exact doubled-lane checks, no fault.",
                "impact": "The architecture fix is credible, but no production M120-r2 exact-SHA seal yet binds the new accumulator into the mapper island.",
                "required_fix": "Create a production integrated wrapper using M123 and rerun the unchanged M120 positive and counterexample campaigns under a new exact-SHA contract.",
            },
            {
                "id": "P2-2",
                "severity": "P2",
                "title": "The macro interface is fixed to one-cycle synchronous read latency",
                "evidence": "The independent one-cycle macro with poisoned no-read data passes. The same binary with a two-cycle read model fails at A-B-A with a pending-sum mismatch and never prints PASS. There is no response-valid/tag channel.",
                "impact": "M123 is RDW-mode independent for consecutive same-address traffic, but not latency-elastic and not yet bound to a real 3072x19 macro timing contract.",
                "required_fix": "State one-cycle latency as a hard interface requirement and bind an exact foundry macro/wrapper, or add tagged response-valid buffering before claiming macro portability.",
            },
            {
                "id": "P2-3",
                "severity": "P2",
                "title": "Production SVA is non-vacuous but weaker than the paper-safe conservation statement",
                "evidence": "All six production covers match, including 14 three-deep forward chains. However the next-cycle property accepts either a write or any protocol_error, has no exact write address/data or write-to-prior-accept assertion, and reset disables every property. Directed scoreboards close these paths only for tested traces.",
                "impact": "The directed evidence is strong, but the assertion set alone cannot prove general exact conservation or reset isolation.",
                "required_fix": "Add an accepted-update transaction scoreboard in SVA/formal: exact next-cycle address/data, no write without a pending accept, onehot read/write ports, forward-base equality, and reset quiescence.",
            },
        ],
        "claim_boundary": {
            "admit": [
                "Reset-free directed M123 same-address forwarding closes the standalone accepted-update loss with exact positive accept/write conservation and full numeric commits.",
                "A review-only M119/M120 integration replay closes the original two-event same-address counterexample at 2 services / 2 updates / 2 writes with 96 exact lane checks.",
                "Consecutive same-address updates suppress the undefined macro read and use the prior computed signed19 vector as the next base under a one-cycle synchronous 1R1W macro model.",
            ],
            "withhold": [
                "reset quiescence, persistence, or exact-once recovery",
                "production-integrated M120-r2 seal",
                "retry deduplication",
                "arbitrary macro latency or foundry macro qualification",
                "DC/STA/Formality, physical or system speedup, energy, PPA, or headline",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print("PASS M123 independent audit score=86 P0=0 P1=1 P2=3 reset_free_forwarding=true reset_quiescence=false")


if __name__ == "__main__":
    main()
