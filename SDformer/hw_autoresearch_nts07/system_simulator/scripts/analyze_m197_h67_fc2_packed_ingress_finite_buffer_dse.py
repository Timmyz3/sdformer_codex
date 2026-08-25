#!/usr/bin/env python3
"""Exact-payload packed-ingress DSE for token-flush FC2 pair fusion.

M196 proved that one accepted nonzero96 descriptor per cycle starves the
two-window fusion datapath.  M197 preserves the exact M196 scheduling rules
and varies only the number of descriptors accepted into one window per fill
cycle.  It reports both legacy-width and iso-width baselines so descriptor
packing cannot be mistaken for fusion gain.

Packing never crosses a window or token boundary.  Weight SRAM response
latency, downstream result backpressure, RTL timing and physical cost remain
excluded.
"""

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path


EXPECTED_M196_ANALYZER_SHA256 = (
    "76552ac1a83131eab9eb3674302e250d205666b1f115a5793cd5dda03e8a70fd"
)
EXPECTED_MANIFEST_SHA256 = (
    "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e"
)
EXPECTED_M195_RESULT_SHA256 = (
    "58732122f31635b3f958972b3f3b42252a10627d5407ef76f3b2076c2bc84d60"
)
EXPECTED_DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
EXPECTED_W1_F1_B2_WALL = 97607807
EXPECTED_PAIR_F1 = {2: 97909442, 3: 97628132, 4: 97389935}
FILL_WIDTHS = (1, 2, 4, 8)
BUFFER_POINTS = (2, 3, 4)
DESCRIPTOR_BITS = 96


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pinned(path, expected_sha, name):
    require(sha256(path) == expected_sha, name + " identity drift")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fraction(numerator, denominator):
    require(denominator > 0, "zero denominator")
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "float": float(numerator) / float(denominator),
    }


def packed_finite_wall(fill_width):
    """Return the M196 scheduler with ceil(entries/fill_width) fill."""
    require(fill_width >= 1, "invalid fill width")

    def schedule(entries, groups, windows_per_job, output_blocks, buffers):
        require(buffers >= max(windows_per_job or [1]),
                "insufficient buffers")
        fill_end = 0
        drain_end = 0
        buffer_free = [0 for _ in range(buffers)]
        window_index = 0
        entry_index = 0
        for job, job_windows in enumerate(windows_per_job):
            slots = []
            for _unused in range(job_windows):
                slot = window_index % buffers
                entry_cycles = (
                    int(entries[entry_index]) + fill_width - 1
                ) // fill_width
                fill_end = max(fill_end, buffer_free[slot]) + entry_cycles
                slots.append(slot)
                window_index += 1
                entry_index += 1
            drain_end = max(fill_end, drain_end) \
                + int(groups[job]) * int(output_blocks)
            for slot in slots:
                buffer_free[slot] = drain_end
        require(entry_index == len(entries), "window/job extent drift")
        return drain_end + 1 if entries else 2

    return schedule


def run_width(fill_width, records, args, m196, m172, m192):
    m196.finite_wall = packed_finite_wall(fill_width)
    aggregate = m196.empty_ledger()
    per_stage = defaultdict(m196.empty_ledger)
    for ordinal, record in enumerate(records):
        stage, ledger = m196.audit_record(
            record, args.payload_root, m172, m192, args.chunk_tokens
        )
        m196.merge(aggregate, ledger)
        m196.merge(per_stage[stage], ledger)
        print("[M197 F{}] {}/120".format(fill_width, ordinal + 1),
              flush=True)
    return aggregate, per_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--m172-analyzer", required=True, type=Path)
    parser.add_argument("--m192-analyzer", required=True, type=Path)
    parser.add_argument("--m195-result", required=True, type=Path)
    parser.add_argument("--m196-analyzer", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-tokens", type=int, default=32768)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    script_path = Path(__file__).resolve()
    script_start = sha256(script_path)
    require(sha256(args.manifest) == EXPECTED_MANIFEST_SHA256,
            "manifest identity drift")
    require(sha256(args.m195_result) == EXPECTED_M195_RESULT_SHA256,
            "M195 result identity drift")
    require(sha256(args.docs359) == EXPECTED_DOCS359_SHA256,
            "docs359 identity drift")
    m196 = load_pinned(
        args.m196_analyzer, EXPECTED_M196_ANALYZER_SHA256,
        "m196_pinned_m197"
    )
    m172 = m196.load_pinned(
        args.m172_analyzer, m196.EXPECTED_M172_ANALYZER_SHA256,
        "m172_pinned_m197"
    )
    m192 = m196.load_pinned(
        args.m192_analyzer, m196.EXPECTED_M192_ANALYZER_SHA256,
        "m192_pinned_m197"
    )
    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    records = [
        record for record in manifest["records"]
        if record["operator"] == "Linear" and ".mlp.fc2" in record["name"]
    ]
    require(len(records) == 120, "expected 120 FC2 records")

    width_ledgers = {}
    stage_ledgers = {}
    for fill_width in FILL_WIDTHS:
        aggregate, per_stage = run_width(
            fill_width, records, args, m196, m172, m192
        )
        width_ledgers[fill_width] = aggregate
        stage_ledgers[fill_width] = per_stage

    f1 = width_ledgers[1]
    require(f1["w1_b2_wall_cycles"] == EXPECTED_W1_F1_B2_WALL,
            "M196 W1/F1/B2 cross-check drift")
    for buffers, expected in EXPECTED_PAIR_F1.items():
        require(f1["pair_b{}_wall_cycles".format(buffers)] == expected,
                "M196 pair/F1/B{} cross-check drift".format(buffers))
    for fill_width in FILL_WIDTHS[1:]:
        require(width_ledgers[fill_width]["events"] == f1["events"],
                "event identity changed with fill width")
        require(width_ledgers[fill_width]["nonzero96_descriptors"] ==
                f1["nonzero96_descriptors"],
                "descriptor identity changed with fill width")
        require(width_ledgers[fill_width]["pair_replay_cycles"] ==
                f1["pair_replay_cycles"],
                "replay changed with fill width")

    points = {}
    for fill_width in FILL_WIDTHS:
        ledger = width_ledgers[fill_width]
        iso_w1 = ledger["w1_b2_wall_cycles"]
        for buffers in BUFFER_POINTS:
            wall = ledger["pair_b{}_wall_cycles".format(buffers)]
            key = "f{}_b{}".format(fill_width, buffers)
            points[key] = {
                "fill_descriptors_per_cycle": fill_width,
                "ingress_bits_per_cycle": DESCRIPTOR_BITS * fill_width,
                "window_buffers": buffers,
                "w1_same_width_b2_wall_cycles": iso_w1,
                "pair_wall_cycles": wall,
                "speed_vs_legacy_w1_f1_b2": fraction(
                    EXPECTED_W1_F1_B2_WALL, wall
                ),
                "fusion_increment_vs_iso_width_w1_b2": fraction(
                    iso_w1, wall
                ),
                "w1_packing_speed_vs_legacy": fraction(
                    EXPECTED_W1_F1_B2_WALL, iso_w1
                ),
            }

    result = {
        "schema": "m197_h67_fc2_packed_ingress_finite_buffer_dse_v1",
        "status": "PASS_EXACT_PACKED_INGRESS_DSE_RTL_OPEN",
        "identity": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "m195_result_sha256": EXPECTED_M195_RESULT_SHA256,
            "m196_analyzer_sha256": EXPECTED_M196_ANALYZER_SHA256,
            "analyzer_start_sha256": script_start,
            "m196_w1_f1_b2_crosscheck": EXPECTED_W1_F1_B2_WALL,
            "m196_pair_f1_crosschecks": {
                str(key): value for key, value in EXPECTED_PAIR_F1.items()
            },
        },
        "architecture": {
            "fill_widths": list(FILL_WIDTHS),
            "buffer_points": list(BUFFER_POINTS),
            "descriptor_bits": DESCRIPTOR_BITS,
            "packing_crosses_window": False,
            "packing_crosses_token": False,
            "token_boundary_flush": True,
            "drain_group_results_per_cycle": 1,
        },
        "points": points,
        "aggregate_by_fill_width": {
            str(fill_width): width_ledgers[fill_width]
            for fill_width in FILL_WIDTHS
        },
        "per_stage_points": {
            str(stage): {
                "f{}_b{}".format(fill_width, buffers): {
                    "w1_same_width_b2_wall_cycles":
                        stage_ledgers[fill_width][stage]["w1_b2_wall_cycles"],
                    "pair_wall_cycles": stage_ledgers[fill_width][stage][
                        "pair_b{}_wall_cycles".format(buffers)
                    ],
                    "fusion_increment_vs_iso_width_w1_b2": fraction(
                        stage_ledgers[fill_width][stage]["w1_b2_wall_cycles"],
                        stage_ledgers[fill_width][stage][
                            "pair_b{}_wall_cycles".format(buffers)
                        ],
                    ),
                }
                for fill_width in FILL_WIDTHS
                for buffers in BUFFER_POINTS
            }
            for stage in m192.STAGE_GEOMETRY
        },
        "claim_boundary": {
            "exact_payload_finite_buffer_cycles": True,
            "iso_width_ablation": True,
            "weight_sram_response_latency": False,
            "result_backpressure": False,
            "integrated_rtl": False,
            "logic_only_dc": False,
            "physical_speedup": False,
            "complete_fc2": False,
            "ffn_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    require(sha256(script_path) == script_start,
            "analyzer changed during run")


if __name__ == "__main__":
    main()
