#!/usr/bin/env python3
"""Independent, read-only hammer for the M1152 LB-FUSE prefix fast-kill.

This checker reads only the contracted 21-row prefix and sealed/configuration
authorities.  It never opens the live producer path for writing and does not
run a simulator, RTL, or EDA tool.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
CONTRACT_PATH = HW / "contracts/m1152_decoder_lbfuse_live_prefix_fastkill_contract_r1_20260830.json"
RESULT = HW / "results/m1152_decoder_lbfuse_live_prefix_fastkill_r1_20260830"
RUNNER_CONTRACT_PATH = HW / "contracts/m1111dr2_m1105dr2_decoder_only_production_runner_source_contract_r2_20260830.json"
M722 = HW / "results/m722r2_lb_fuse_decoder_cpu_fastkill_r1_20260828"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_PREFIX = {
    "rows": 21,
    "bytes": 67751,
    "sha256": "584b47f2b74dc877dac22084283ea9f028387c2d1eb86e045dac573ad11d98c0",
}
EXPECTED_RESOURCE = {
    "lanes": 96,
    "accumulator_bits": 24,
    "onchip_sram_bytes_macro_rounded": 245760,
    "weight_bytes": 13824,
    "psum_bytes": 221184,
    "descriptor_control_bytes": 8192,
    "reserved_unallocated_bytes": 2560,
    "psum_banks": 6,
    "psum_mode": "1RW",
    "psum_row_bytes": 48,
    "psum_read_latency_cycles": 2,
    "psum_write_latency_cycles": 1,
}
MODULES = {
    0: ("D0", 40),
    1: ("D1", 80),
    2: ("D2", 160),
    3: ("D3", 320),
}
EXPECTED_CAPACITY = {
    "D0": (34560, True, True, 23040, 17280),
    "D1": (69120, True, True, 46080, 34560),
    "D2": (138240, True, True, 92160, 69120),
    "D3": (276480, False, False, 184320, 138240),
}


class Reject(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Reject(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json_text(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        text,
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(Reject("non-finite JSON")),
    )


def strict_json(path: Path):
    return strict_json_text(path.read_text(encoding="utf-8"))


def verify_flat_seal(directory: Path) -> dict[str, str]:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and outer.is_file(), "seal files missing")
    require(
        outer.read_text(encoding="utf-8") == f"{sha256(manifest)}  SHA256SUMS\n",
        "outer seal mismatch",
    )
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        member = directory / name
        require(name not in listed and member.is_file() and sha256(member) == digest,
                "sealed member mismatch")
        listed[name] = digest
    actual = {
        p.name for p in directory.iterdir()
        if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    }
    require(set(listed) == actual, "flat seal coverage mismatch")
    return {"manifest_sha256": sha256(manifest), "outer_file_sha256": sha256(outer)}


def validate_claim(candidate: dict) -> None:
    require(candidate["fair_a1_onchip"] is True, "weak/off-chip A1 baseline")
    require(candidate["fair_a1_offchip_psum_spill_bytes"] == 0, "A1 spill mislabel")
    require(candidate["ordinary_onchip_rmw_is_dram"] is False, "on-chip RMW mislabeled DRAM")
    require(candidate["psum_rmw_bytes"] == 90452061504, "RMW repeated/omitted counting")
    require(candidate["partial_prefix"] is True and candidate["full_decoder"] is False,
            "partial prefix relabeled full")
    require(candidate["final_checkpoint_rebind_required"] is True,
            "final checkpoint rebind suppressed")
    require(candidate["d3_acc16_numerically_admitted"] is False,
            "unsupported Acc16 admission")
    require(candidate["live_producer_modification_authorized"] is False,
            "live producer modification authorized")


def main() -> None:
    checks = 0
    contract = strict_json(CONTRACT_PATH)
    report = strict_json(RESULT / "report.json")
    runner = strict_json(RUNNER_CONTRACT_PATH)
    m722 = strict_json(M722 / "report.json")
    result_seal = verify_flat_seal(RESULT)
    checks += 8

    require(sha256(CONTRACT_PATH) == report["identity"]["contract_sha256"], "contract pin")
    require(sha256(RUNNER_CONTRACT_PATH) == "821819b00503b91a8fb8dfca8fe000208e10746e751a3815131dc8ff1cbed515", "runner contract pin")
    require(sha256(DOCS359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4", "docs359 pin")
    require(sha256(M722 / "SHA256SUMS.seal.sha256") == "bf767f0698f33874d8793e8b5478cc54b1cede73627c67157d9d404f093e3357", "M722 outer pin")
    require(m722["decision"]["fair_a1_zero_offchip_psum_spill"] is True, "fair A1 spill authority")
    checks += 5

    frozen = contract["frozen_prefix"]
    require({key: frozen[key] for key in EXPECTED_PREFIX} == EXPECTED_PREFIX, "prefix contract drift")
    live = HW / frozen["path"]
    require(live.is_file() and not live.is_symlink(), "live prefix path")
    rows_raw = []
    with live.open("rb") as stream:
        for _ in range(EXPECTED_PREFIX["rows"]):
            raw = stream.readline()
            require(raw.endswith(b"\n"), "short/torn prefix")
            rows_raw.append(raw)
    prefix = b"".join(rows_raw)
    require(len(prefix) == EXPECTED_PREFIX["bytes"], "prefix byte count")
    require(sha256_bytes(prefix) == EXPECTED_PREFIX["sha256"], "prefix SHA")
    checks += 5

    common = runner["common_resource"]
    observed_resource = {
        "lanes": common["lanes"],
        "accumulator_bits": common["accumulator_bits"],
        "onchip_sram_bytes_macro_rounded": common["onchip_sram_bytes_macro_rounded"],
        "weight_bytes": common["partitions"]["weight_bytes"],
        "psum_bytes": common["partitions"]["psum_bytes"],
        "descriptor_control_bytes": common["partitions"]["descriptor_control_bytes"],
        "reserved_unallocated_bytes": common["partitions"]["reserved_unallocated_bytes"],
        "psum_banks": common["ports"]["psum"]["banks"],
        "psum_mode": common["ports"]["psum"]["mode"],
        "psum_row_bytes": common["ports"]["psum"]["row_bytes"],
        "psum_read_latency_cycles": common["ports"]["psum"]["read_latency_cycles"],
        "psum_write_latency_cycles": common["ports"]["psum"]["write_latency_cycles"],
    }
    require(observed_resource == EXPECTED_RESOURCE, "common resource mismatch")
    require(sum(common["partitions"].values()) == 245760, "240-KiB partition sum")
    checks += len(EXPECTED_RESOURCE) + 1

    totals = Counter()
    modules = defaultdict(Counter)
    expected_cycle = 0
    expected_transaction = 0
    for ordinal, raw in enumerate(rows_raw):
        text = raw.decode("utf-8")
        row = strict_json_text(text)
        require(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" == text,
                "noncanonical row")
        require(row["global_call_ordinal"] == ordinal, "row ordinal")
        require(row["cycle_start"] == expected_cycle, "cycle continuity")
        require(row["transaction_ordinal_first"] == expected_transaction, "transaction continuity")
        require(row["cycle_end"] - row["cycle_start"] == row["diagnostic_cycles"], "cycle projection")
        require(row["transaction_ordinal_last"] + 1 == row["transaction_ordinal_first"] + row["transaction_count"], "transaction projection")
        expected_cycle = row["cycle_end"]
        expected_transaction = row["transaction_ordinal_last"] + 1
        module = int(row["module_ordinal"])
        require(module in MODULES and row["configuration"] == "M1105DR2_EXACT_TYPED_K8", "module/configuration")
        kind = row["kind_summaries"]
        count = int(kind["compute"]["count"])
        require(int(kind["psum_read"]["count"]) == count == int(kind["psum_write"]["count"]), "exact RMW count")
        require(int(kind["psum_read"]["traffic_bytes"]) == count * 288, "read bytes")
        require(int(kind["psum_write"]["traffic_bytes"]) == count * 288, "write bytes")
        require(int(kind["output_commit"]["traffic_bytes"]) == int(kind["output_commit"]["count"]) * 288, "dense commit bytes")
        require(row["claim_boundary"] == {
            "diagnostic_only": True,
            "final_checkpoint_rebind_required": True,
            "paper_ppa_ready": False,
            "speedup_admitted": False,
            "system_speedup_admitted": False,
        }, "row claim boundary")
        totals["cycles"] += int(row["diagnostic_cycles"])
        totals["psum_read"] += int(row["diagnostic_traffic_bytes"]["psum_read"])
        totals["psum_write"] += int(row["diagnostic_traffic_bytes"]["psum_write"])
        totals["commits"] += int(kind["output_commit"]["count"])
        totals["calls"] += 1
        modules[module]["calls"] += 1
        modules[module]["input_descriptor_read"] += int(row["diagnostic_traffic_bytes"]["input_descriptor_read"])
        modules[module]["commits"] += int(kind["output_commit"]["count"])
        checks += 14

    require(totals == Counter(calls=21, cycles=628231055, psum_read=45226030752,
                              psum_write=45226030752, commits=5568000), "prefix totals")
    require({index: modules[index]["calls"] for index in MODULES} == {0: 6, 1: 5, 2: 5, 3: 5}, "module calls")
    require([modules[index]["commits"] // modules[index]["calls"] for index in MODULES] == [48000, 96000, 192000, 768000], "dense commits per call")
    require(modules[3]["input_descriptor_read"] == 1332936320, "D3 descriptor read")
    checks += 4

    capacity = {}
    for index, (name, width) in MODULES.items():
        full24 = 3 * width * 96 * 3
        full16 = 3 * width * 96 * 2
        half24 = 3 * width * 48 * 3
        observed = (full24, full24 <= 245760, full24 <= 221184, full16, half24)
        require(observed == EXPECTED_CAPACITY[name], f"{name} capacity")
        capacity[name] = {
            "acc24_full96_bytes": full24,
            "acc24_full96_fits_240kib": full24 <= 245760,
            "acc24_full96_fits_psum_partition": full24 <= 221184,
            "acc16_full96_bytes_but_not_numerically_admitted": full16,
            "acc24_cout48_bytes": half24,
        }
        checks += 5

    psum_rmw = totals["psum_read"] + totals["psum_write"]
    require(psum_rmw == 90452061504, "single read+write RMW count")
    require(report["fair_same_port_candidate"]["baseline_cycles"] == totals["cycles"], "reported baseline cycles")
    require(report["fair_same_port_candidate"]["candidate_executable_cycle_lower_bound"] == totals["cycles"], "candidate lower bound")
    require(report["fair_same_port_candidate"]["baseline_onchip_psum_rmw_bytes"] == psum_rmw, "reported baseline RMW")
    require(report["fair_same_port_candidate"]["candidate_onchip_psum_rmw_bytes"] == psum_rmw, "reported candidate RMW")
    require(report["fair_same_port_candidate"]["baseline_over_candidate_speedup_upper_bound"] == "1.000000000000", "speedup bound")
    require(report["fair_same_port_candidate"]["onchip_psum_byte_reduction_fraction"] == "0.000000000000", "onchip reduction")
    require(report["fair_same_port_candidate"]["offchip_psum_byte_reduction_fraction"] == "0.000000000000", "offchip reduction")
    split = report["fair_same_port_candidate"]["d3_acc24_cout48_two_pass_cost"]
    require(split["extra_input_descriptor_read_bytes_if_not_retained"] == modules[3]["input_descriptor_read"], "D3 split extra read")
    require(split["compute_output_lane_passes"] == 2 and split["same_96_lane_throughput_claim_allowed"] is False, "D3 split claim")
    checks += 10

    base_claim = {
        "fair_a1_onchip": True,
        "fair_a1_offchip_psum_spill_bytes": 0,
        "ordinary_onchip_rmw_is_dram": False,
        "psum_rmw_bytes": psum_rmw,
        "partial_prefix": True,
        "full_decoder": False,
        "final_checkpoint_rebind_required": True,
        "d3_acc16_numerically_admitted": False,
        "live_producer_modification_authorized": False,
    }
    validate_claim(base_claim)
    attacks = {
        "weak_baseline": {"fair_a1_onchip": False},
        "offchip_mislabel": {"ordinary_onchip_rmw_is_dram": True},
        "repeated_rmw_count": {"psum_rmw_bytes": psum_rmw + totals["psum_read"]},
        "partial_as_full": {"partial_prefix": False, "full_decoder": True},
        "acc16_without_proof": {"d3_acc16_numerically_admitted": True},
        "live_producer_modification": {"live_producer_modification_authorized": True},
    }
    rejected = []
    for name, mutation in attacks.items():
        attacked = dict(base_claim)
        attacked.update(mutation)
        try:
            validate_claim(attacked)
        except Reject:
            rejected.append(name)
        else:
            raise Reject(f"attack survived: {name}")
    require(len(rejected) == len(attacks), "attack count")
    checks += len(attacks) + 2

    evidence = {
        "status": "PASS_M1153_INDEPENDENT_HAMMER__KILL_M1152_LBFUSE_NO_RTL",
        "checks": checks,
        "controlled_attacks_rejected": rejected,
        "result_seal": result_seal,
        "prefix": EXPECTED_PREFIX,
        "resource": EXPECTED_RESOURCE,
        "totals": dict(totals),
        "psum_rmw_bytes_read_plus_write_once": psum_rmw,
        "capacity": capacity,
        "d3_acc24_cout48_extra_input_descriptor_read_bytes_if_not_retained": modules[3]["input_descriptor_read"],
        "speedup_upper_bound": "1.000000000000",
        "onchip_psum_byte_reduction_fraction": "0.000000000000",
        "offchip_psum_spill_bytes_a1_and_candidate": 0,
        "claim_boundary": {
            "partial_prefix": True,
            "checkpoint": "H67_ep35",
            "final_checkpoint_rebind_required": True,
            "performance_headline": False,
            "rtl_authorized": False,
        },
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
