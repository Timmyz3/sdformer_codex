#!/usr/bin/env python3
"""Read-only M102 bit-sparse physical-baseline preflight.

The script does not import or execute the M78/M88 producers.  It SHA-pins and
strict-parses their frozen sources/results, reconstructs the shared32 service
ledger, and inventories production RTL candidates.  Semantic suitability of
the short candidate list is documented in the companion review.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
OUTPUT = HERE / "m102_bit_sparse_baseline_preflight_audit.json"

PATHS = {
    "m78_analyzer": HW / "system_simulator/scripts/analyze_m78_precision_elastic_pwp.py",
    "m78_result": HW / "results/m78_precision_elastic_pwp_valid825_internal_dev_r1_20260823/m78_precision_elastic_pwp.json",
    "m88_analyzer": HW / "system_simulator/scripts/analyze_m88_bounded_sync_bank_double_buffer.py",
    "m88_result": HW / "results/m88_bounded_sync_bank_double_buffer_valid825_internal_r1_20260823/m88_bounded_sync_bank_double_buffer.json",
    "m79": HW / "rtl_m79/precision_elastic_pwp_beat_assembler.sv",
    "m82": HW / "rtl_m82/zero_bubble_elastic_pwp_stream.sv",
    "m85": HW / "rtl_m85/guarded_wordpacked_pwp_stream.sv",
    "m99": HW / "rtl_m99/phase_slack_guarded_wordpacked_pwp_stream.sv",
    "sparse_mac": HW / "rtl/sparse_mac_pe.v",
    "hatf96": HW / "rtl_hitflow/gatestack_hatf96_weight_coalescer.sv",
    "m101_contract": HW / "contracts/m101_pwp_metadata_fmax_sweep_synopsys_contract_r1_20260824.json",
}
EXPECTED_SHA256 = {
    "m78_analyzer": "9215c2eeff8ccbfa0ef7d27f48ed6100a56813c1881873013e9e23a2e149df6b",
    "m78_result": "00d2802eb8e4085fdf740f0183b23488ef2def5ca38f027c57ccba04f30064cc",
    "m88_analyzer": "5b62d1f23555fba4bc00f1e1b427ae5861089e0a8ea5f8ae98c062acb071dfae",
    "m88_result": "36e9b0603422ccff7afd23e6e5e2309bc5d53b3c7e9898538095d6baa23da483",
    "m79": "00bf98d682759906a932c5518561393c5fc74104407e9df35ec3af42835fcad7",
    "m82": "2e8842234917355ee082968487229e83789e1a2f212296168d3a972f83631e1f",
    "m85": "ec2680f2fc97500133f3333e063fc268602ad793324a2cf6b8dbc1eb4b5207b0",
    "m99": "adb2dfd95ee3dd179cb373eb5ead937d9beb4db25648325634ebba755243b082",
    "sparse_mac": "6d8c30b31e2c87113e7a38ef5258d77f5e05e09114cb7d3be09e4d30f17a2219",
    "hatf96": "c911236dc0b496427cffca9583f8aaee5b1e4b8bb643c07e591b2a6b38e3237a",
    "m101_contract": "dad2b791d505b9532f7924b80e28cd899983e2b097f993f5b1df1c1a97a16c50",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, "duplicate JSON key " + key)
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          ValueError("nonstandard JSON " + token)))


def rtl_inventory():
    files = []
    for directory in sorted(HW.glob("rtl*")):
        if directory.is_dir():
            files.extend(path for path in directory.rglob("*")
                         if path.suffix.lower() in (".sv", ".v"))
    modules = []
    literal_bit_sparse = []
    same_io_shape = []
    for path in sorted(set(files)):
        text = path.read_text(encoding="utf-8", errors="ignore")
        names = re.findall(
            r"(?m)^\s*module\s+([A-Za-z_][A-Za-z0-9_]*)", text)
        modules.extend({"path": str(path.relative_to(HW)), "module": name}
                       for name in names)
        lower = text.lower()
        if "bit_sparse" in lower or "bit-sparse" in lower:
            literal_bit_sparse.append(str(path.relative_to(HW)))
        if ("[255:0]" in text and
                ("96*12" in text or "LANES*OUT_W" in text)):
            same_io_shape.append(str(path.relative_to(HW)))
    return {
        "rtl_file_count": len(set(files)),
        "module_count": len(modules),
        "bit_sparse_literal_files": literal_bit_sparse,
        "files_with_256bit_input_and_96x12_style_output": same_io_shape,
    }


def main():
    observed = {}
    for name, path in PATHS.items():
        require(path.is_file(), "missing " + name)
        observed[name] = sha256(path)
        require(observed[name] == EXPECTED_SHA256[name], name + " SHA drift")

    m78_source = PATHS["m78_analyzer"].read_text(encoding="utf-8")
    m88_source = PATHS["m88_analyzer"].read_text(encoding="utf-8")
    for token in (
            'baseline_compute = (base["baseline_ops_per_block"] * OUTPUT_BLOCKS *',
            'port["weight_cycles"])',
            'baseline_cycles += max(baseline_compute, next_weight) + COMPUTE_TAIL_CYCLES',
            '"name": "SHARED_32B", "weight_cycles": 3, "pwp_port_bytes": 32',
            'WEIGHT_PHASE_BYTES = PARTITION_BITS * OUTPUT_BLOCKS * WEIGHT_VECTOR_BYTES',
            'DRAM_BYTES_PER_CYCLE = 32'):
        require(token in m78_source, "M78 charging source token drift: " + token)
    for token in (
            'baseline_compute = (base["baseline_ops_per_block"] *',
            'm78.OUTPUT_BLOCKS *',
            'm78.PORTS[2]["weight_cycles"])',
            'baseline_duration = baseline_compute + m78.COMPUTE_TAIL_CYCLES',
            'baseline_preparations = [',
            'WEIGHT_PHASE_BYTES / float(DRAM_BYTES_PER_CYCLE)',
            'baseline = bounded_double_buffer(baseline_durations,'):
        require(token in m88_source, "M88 charging source token drift: " + token)

    m78 = strict_json(PATHS["m78_result"])
    m88 = strict_json(PATHS["m88_result"])
    require(m78["status"] ==
            "PASS_M78_EXACT_INT8_PWP_WIDTH_AND_BLOCK_ESCAPE_DSE_INTERNAL_ONLY",
            "M78 status")
    require(m88["status"] ==
            "PASS_M88_BOUNDED_MODULE_CYCLE_SIM_VALID825_INTERNAL_ONLY",
            "M88 status")
    cap11 = next(row for row in m78["configurations"]
                 if row["signed_width_cap"] == 11)
    shared32 = next(row for row in cap11["cycle_simulations"]
                    if row["port"] == "SHARED_32B")
    held = cap11["heldout"]

    baseline_ops = held["baseline_bit_sparse_vector_ops_all_blocks"]
    require(baseline_ops == 371461096, "baseline vector-op count")
    baseline_service_cycles = baseline_ops * 3
    baseline_sram_read_bytes = baseline_ops * 96
    phases = 5 * 1728
    baseline_initial_prefetch = 5 * (12288 // 32)
    baseline_tail = phases * 2
    baseline_bounded_cycles = baseline_service_cycles + \
        baseline_initial_prefetch + baseline_tail
    require(baseline_bounded_cycles == shared32["bit_sparse_cycles"] ==
            m88["aggregate"]["bounded_bit_sparse_cycles"],
            "bit-sparse bounded cycle reconstruction")
    require(baseline_sram_read_bytes ==
            held["baseline_weight_sram_read_bytes"],
            "bit-sparse SRAM byte reconstruction")

    pwp_uses = dict((int(width), count) for width, count in
                    held["pwp_uses_by_width"].items())
    service_by_width = {8: 3, 9: 4, 10: 4, 11: 5}
    pwp_service_cycles = sum(pwp_uses[width] * service_by_width[width]
                             for width in service_by_width)
    correction_ops = held["correction_ops_all_blocks"]
    candidate_service_cycles = correction_ops * 3 + pwp_service_cycles
    candidate_physical_service_bytes = candidate_service_cycles * 32
    candidate_logical_read_bytes = (
        held["candidate_correction_sram_read_bytes"] +
        held["candidate_pwp_sram_read_bytes"])
    candidate_bounded_cycles = m88["aggregate"]["bounded_candidate_cycles"]
    require(candidate_service_cycles == 790667725 and
            candidate_bounded_cycles == 790706475,
            "candidate service/bounded cycles")
    require(sum(pwp_uses.values()) == held["pwp_ops_all_blocks"] == 58969374,
            "PWP use population")
    require(correction_ops == 188148490, "correction population")
    require(candidate_logical_read_bytes == 24500425188,
            "candidate logical SRAM reads")

    aggregate = m88["aggregate"]
    require(abs(aggregate["speedup_vs_bit_sparse"] -
                baseline_bounded_cycles / float(candidate_bounded_cycles)) < 1e-15,
            "M88 speedup reconstruction")
    resource = m88["resource_model"]
    require(resource["dram_bytes_per_cycle"] == 32 and
            resource["local_storage"]["two_weight_phase_buffers_bytes"] == 24576 and
            resource["local_storage"]["total_bytes"] == 116525,
            "M88 resource ledger")

    m101 = strict_json(PATHS["m101_contract"])
    require("bit-sparse physical baseline frequency area or energy" in
            m101["claim_boundary"]["not_admitted"] and
            "multiplication by the M88 1.409375695x cycle estimate" in
            m101["claim_boundary"]["not_admitted"],
            "M101 anti-multiplication boundary")

    inventory = rtl_inventory()
    known_candidates = {
        "rtl_m79/precision_elastic_pwp_beat_assembler.sv":
            "96x8 fixed-width mode can assemble three 256-bit beats, but has no source/block weight address mapper",
        "rtl_m82/zero_bubble_elastic_pwp_stream.sv":
            "shared zero-bubble assembler building block, but no 16-source by 8-block weight service top",
        "rtl/sparse_mac_pe.v":
            "default 8 lane spike-per-lane accumulator; no 96-output vector or 32-byte three-beat service",
        "rtl_hitflow/gatestack_hatf96_weight_coalescer.sv":
            "three parallel 32-lane banks return 96 bytes, not the frozen serial 32-byte-per-cycle denominator",
    }

    output = {
        "schema": "m102_bit_sparse_physical_baseline_preflight_audit_v1",
        "status": "NO_EXISTING_MATCHED_PHYSICAL_DENOMINATOR_GO_MINIMAL_BASELINE_SPEC",
        "producer_or_synopsys_executed": False,
        "sha256": observed,
        "m78_m88_exact_shared32_denominator": {
            "samples": 5,
            "phases": phases,
            "output_lanes_per_vector_op": 96,
            "output_blocks": 8,
            "active_source_vector_ops_all_blocks": baseline_ops,
            "bytes_per_weight_vector": 96,
            "service_port_bytes_per_cycle": 32,
            "cycles_per_weight_vector_op": 3,
            "service_cycles": baseline_service_cycles,
            "initial_weight_prefetch_cycles": baseline_initial_prefetch,
            "compute_tail_cycles": baseline_tail,
            "bounded_bit_sparse_cycles": baseline_bounded_cycles,
            "onchip_weight_read_bytes": baseline_sram_read_bytes,
            "shared_dram_weight_bytes": 106168320,
            "weight_bytes_per_phase": 12288,
            "single_weight_phase_buffer_bytes": 12288,
            "double_weight_phase_buffer_bytes": 24576,
        },
        "candidate_scope_reconciliation": {
            "correction_weight_vector_ops": correction_ops,
            "correction_service_cycles": correction_ops * 3,
            "pwp_vector_ops": sum(pwp_uses.values()),
            "pwp_uses_by_width": dict((str(key), value)
                                      for key, value in sorted(pwp_uses.items())),
            "pwp_service_cycles": pwp_service_cycles,
            "combined_candidate_service_cycles": candidate_service_cycles,
            "bounded_candidate_cycles": candidate_bounded_cycles,
            "candidate_logical_sram_read_bytes": candidate_logical_read_bytes,
            "candidate_physical_32byte_service_bytes": candidate_physical_service_bytes,
            "candidate_alignment_fetch_overhead_bytes":
                candidate_physical_service_bytes - candidate_logical_read_bytes,
            "service_only_cycle_ratio":
                baseline_service_cycles / float(candidate_service_cycles),
            "m88_bounded_cycle_ratio":
                baseline_bounded_cycles / float(candidate_bounded_cycles),
        },
        "rtl_inventory": inventory,
        "exact_matching_standalone_bit_sparse_baseline_found": False,
        "known_near_candidates": known_candidates,
        "physical_throughput_admitted_now": False,
        "m85_m99_frequency_ratio_may_multiply_m88_cycles": False,
        "required_formula": {
            "service_island_speedup":
                "(1114383288 / f_bit_sparse_weight_service) / (790667725 / f_combined_candidate_weight_plus_pwp_service)",
            "bounded_module_speedup":
                "(1114402488 / f_bit_sparse_complete_top) / (790706475 / f_candidate_complete_top)",
            "common_clock_case":
                "frequency cancels; use matched cycle ratio only",
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M102 read-only bit-sparse physical-baseline preflight")
    print("existing_matched_rtl=false physical_throughput_admitted=false")
    print("bit_sparse vector_ops=371461096 service_cycles=1114383288 bounded_cycles=1114402488")
    print("candidate service_cycles=790667725 bounded_cycles=790706475")
    print("service_ratio={:.9f} bounded_ratio={:.9f}".format(
        baseline_service_cycles / float(candidate_service_cycles),
        baseline_bounded_cycles / float(candidate_bounded_cycles)))
    print(str(OUTPUT))


if __name__ == "__main__":
    main()
