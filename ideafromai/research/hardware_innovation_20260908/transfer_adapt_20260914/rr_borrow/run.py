import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

H = Path(__file__).resolve().parent
P = H.parents[1] / "fusion_ten_trials_20260914/phase_borrow"
subprocess.run(["/opt/anaconda3/bin/python3.12", str(H / "build_from_sources.py")], check=True)
with (H / "build.log").open("w") as log:
    subprocess.run(["verilator", "-Wall", "--cc", "--exe", "--top-module", "interleave_stream", "--Mdir", "obj",
                    "interleave_stream.sv", "rr_context.sv", "i24_consumer.sv", "wide_phase_alu.sv", "tb.cpp", "-CFLAGS", "-O3 -std=c++14"],
                   cwd=H, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(["make", "-C", "obj", "-f", "Vinterleave_stream.mk", "-j2"], cwd=H, stdout=log, stderr=subprocess.STDOUT, check=True)
cases = json.loads((P / "fixtures.json").read_text())
(H / "logs").mkdir(exist_ok=True)
jobs = [(c, m, s, 1) for c in cases for m in [1, 2, 3] for s in [0, 1]]
# Homogeneous native sources give identical gold for two real adjacent tiles;
# these cases force actual cross-context repair/normalization contention.
jobs += [(c, m, s, 2) for c in cases if c["name"] in ["one", "extreme_1", "extreme_-1"] for m in [1, 2, 3] for s in [0, 1]]

def run(job):
    case, mode, stall, count = job
    name = f"{case['name']}_n{count}_m{mode}_s{stall}"
    cmd = [str(H / "obj/Vinterleave_stream"), str(P / "fixtures" / case["name"]), str(mode), str(stall), str(case["tile_id"]), str(count)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    (H / "logs" / f"{name}.jsonl").write_text(result.stdout)
    (H / "logs" / f"{name}.err").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError((name, result.returncode, result.stderr))
    return [dict(json.loads(line), fixture=case["name"], first_tile=case["tile_id"], tiles=count) for line in result.stdout.splitlines()]

rows = []
with ThreadPoolExecutor(max_workers=4) as pool:
    for result in pool.map(run, jobs):
        rows.extend(result)
(H / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(dict(commands=len(rows), values_each_stage=sum(r["outputs"] for r in rows), passed=True,
                      dual_context_repair_waits=sum(r["core_repair_arbitration_stalls"] for r in rows if r["tiles"] == 2),
                      dual_context_normalization_waits=sum(r["core_normalization_arbitration_stalls"] for r in rows if r["tiles"] == 2))), flush=True)
