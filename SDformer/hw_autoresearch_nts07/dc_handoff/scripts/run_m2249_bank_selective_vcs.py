#!/usr/bin/env python3
"""Actual C2 frontend, mask-aware ordinary and B4, same six fixed pilots.

CPU supplies bank-read counts only. RTL supplies measured cycles and every
Acc24 result is checked against the independent signed scalar TB reference.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import run_m2233_ep34_tsbg_matched_power_repair_one_shot as cfg

sys.path.insert(0, str(cfg.HW / "system_simulator/scripts"))
from m2244_consumer_union_bank_reads import masked_reads


def directed_round(index):
    masks = (1, 256, 265, 8) if index == 0 else (257, 512, 520, 777)
    return [[(masks[c] | (masks[c] << 16 if c % 2 == 0 else 0))
             if g < (5 if index == 3 else 4) else 0 for g in range(48)] for c in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after-m2248", action="store_true")
    ap.add_argument("--reuse-builds", type=Path)
    ap.add_argument("--demand-only", action="store_true",
                    help="Causal third axis: group-major order without union prefetch")
    ap.add_argument("--memory-latency", type=int, choices=range(1,17))
    ap.add_argument("--always-ready", action="store_true")
    args = ap.parse_args()
    if args.after_m2248:
        print("Waiting for M2248; no second EDA process launched.", flush=True)
        progress = cfg.HW / "results/m2248_matched_power/progress.json"
        while True:
            stage = json.loads(progress.read_text())["stage"]
            if stage.startswith(("STOPPED", "TOOL_RUNS_COMPLETE")):
                break
            time.sleep(15)
    cfg.no_same_uid_eda()
    out = Path(tempfile.mkdtemp(prefix="m2249_bank_selective_", dir=cfg.HW / "results"))
    print("Output:", out, flush=True)
    fixture = cfg.HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.memh"
    packed = [int(v, 16) for v in fixture.read_text().split()]
    sources = [cfg.HW / p for p in (
        "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
        "rtl_m2249/m2249_c2_consumer_scoped_bank_fill_frontend.sv",
        "verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv",
        "tb_m2018/tb_m2160_m2018_ordinary_native_saif_report_reset_preflight.sv",
        "tb_m2249/tb_m2249_consumer_scoped_bank_fill.sv")]
    env = {**os.environ, "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
        "SNPSLMD_LICENSE_FILE": cfg.LICENSE_SERVER, "LM_LICENSE_FILE": cfg.LICENSE_FILE,
        "VCS_HOME": str(cfg.VCS.parent.parent), "VCS_ARCH_OVERRIDE": "linux"}
    def command(argv, cwd, log):
        with log.open("w") as stream:
            subprocess.run(argv, cwd=cwd, env=env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=1200, check=True)
        text = log.read_text()
        if re.search(r"Error-\[|Fatal:|Error:|Assertion.*fail", text, re.I):
            raise RuntimeError("VCS error: " + str(log))
        return text
    rows, directed = [], []
    service_args = ([f"+M2258_MEMORY_LATENCY={args.memory_latency}"] if args.memory_latency else [])
    if args.always_ready:
        service_args.append("+M2258_ALWAYS_READY")
    axes = ((1, "tsbg_demand"),) if args.demand_only else ((0, "ordinary"), (1, "tsbg"))
    for mode, axis in axes:
        point = out / axis
        point.mkdir()
        previous = args.reuse_builds.resolve() / axis if args.reuse_builds else point
        build = previous if (previous / "simv").is_file() else point
        if not (build / "simv").is_file():
            command([str(cfg.VCS), "-full64", "-sverilog", "-timescale=1ns/1ps",
                "+vcs+initreg+random", f"+define+M2217_SCHEDULE_MODE={mode}",
                f"+define+M2254_UNION_PREFETCH={0 if args.demand_only else 1}",
                "-assert", "svaext", "-lca", *map(str, sources), "-top",
                "tb_m2249_consumer_scoped_bank_fill", "-o", str(build / "simv")],
                build, point / "compile.log")
        for window, slot in (("low", 1606), ("median", 526), ("high", 1071)):
            words = [packed[slot*192+c*48:slot*192+(c+1)*48] for c in range(4)]
            # Demand/union have equal scalar reads for a complete B4 group;
            # they differ in refill transaction count and waiting cycles.
            predicted = masked_reads(words, "tsbg" if mode else "ordinary", 4)[0]["bank_reads"]
            text = command([str(build / "simv"), "-no_save", *service_args, f"+M2217_STRATUM={window}",
                f"+EXPECTED_MASKED_READS={predicted}"], build, point / f"{window}.log")
            match = re.search(r"PASS_M2249_BANK_SELECTIVE mode=(\d+) stratum=(\w+) "
                r"slot=(\d+) cycles=(\d+) bank_reads=(\d+) products=(\d+) "
                r"commits=(\d+) duration_ns=([\d.]+)", text)
            if not match:
                raise RuntimeError("No completed numeric/reference result: " + str(point / f"{window}.log"))
            values = match.groups()
            rows.append(dict(axis=axis, window=window, slot=slot, cycles=int(values[3]),
                bank_reads=int(values[4]), products=int(values[5]), commits=int(values[6]),
                duration_ns=float(values[7]), expected_bank_reads=predicted))
            print(match.group(), flush=True)
        cache, expected = None, []
        for index in range(4):
            counts, cache = masked_reads(directed_round(index), "tsbg" if mode else "ordinary", 4, cache)
            expected.append(counts["bank_reads"])
        text = command([str(build / "simv"), "-no_save", *service_args, "+M2249_PARTIAL_WARM",
            *[f"+EXPECTED_MASKED_READS{i}={v}" for i,v in enumerate(expected)]],
            build, point / "partial_warm.log")
        passes = re.findall(r"PASS_M2249_BANK_SELECTIVE[^\n]+", text)
        if len(passes) != 4:
            raise RuntimeError("Partial/warm/eviction test incomplete")
        directed.append(dict(axis=axis, expected_bank_reads_per_round=expected, pass_lines=passes))
        print(axis, "partial/warm/eviction:", expected, flush=True)
    result = dict(status="PASS", rows=rows, directed=directed,
        memory_latency_cycles=args.memory_latency, bank_ready_unstalled=args.always_ready,
        scope="Three preselected G48 windows and warm directed rounds; fixed memory/backpressure; not full population",
        consumer_union_enabled=not args.demand_only,
        arithmetic="Every committed Acc24 lane compared against independent signed INT8 reference",
        comparison="Both modes use per-bank valid LRU4; only context-vs-group scheduling and consumer union differ",
        timing_area_power_measured=False, system_speedup_measured=False)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS M2249 campaign; result:", out / "result.json", flush=True)


if __name__ == "__main__":
    main()
