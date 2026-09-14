import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

H = Path(__file__).resolve().parent
OLD = H.parents[1] / "fusion_ten_trials_20260914/phase_borrow"
subprocess.run(["python3.12", str(H / "build_glue.py")], check=True)
with (H / "build.log").open("w") as log:
    subprocess.run(["verilator", "-Wall", "--cc", "--exe", "--top-module", "consumer_stream", "--Mdir", "obj",
                    "consumer_stream.sv", "modular_core.sv", "i24_consumer.sv", "tb.cpp", "-CFLAGS", "-O3 -std=c++14"],
                   cwd=H, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(["make", "-C", "obj", "-f", "Vconsumer_stream.mk", "-j2"], cwd=H, stdout=log, stderr=subprocess.STDOUT, check=True)
cases = json.loads((OLD / "fixtures.json").read_text())
(H / "logs").mkdir(exist_ok=True)

def run(job):
    case, mode, stall = job
    cmd = [str(H / "obj/Vconsumer_stream"), str(OLD / "fixtures" / case["name"]), str(mode), str(stall), str(case["tile_id"])]
    result = subprocess.run(cmd, capture_output=True, text=True)
    name = f"{case['name']}_m{mode}_s{stall}"
    (H / "logs" / f"{name}.jsonl").write_text(result.stdout)
    (H / "logs" / f"{name}.err").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError((name, result.returncode, result.stderr))
    return [dict(json.loads(line), fixture=case["name"]) for line in result.stdout.splitlines()]

rows = []
with ThreadPoolExecutor(max_workers=4) as pool:
    for result in pool.map(run, [(c, m, s) for c in cases for m in [0, 1, 2] for s in [0, 1]]):
        rows.extend(result)
(H / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
totals = {m: sum(r["total_cycles"] for r in rows if r["fixture"].startswith("real_") and r["mode"] == m and r["stall"] == 0 and r["command"] == 0) for m in [0, 1, 2]}
print(json.dumps(dict(commands=len(rows), values_each_stage=sum(r["outputs"] for r in rows), real8_cold_noBP_cycles=totals,
                      candidate_saving_vs_strong=totals[1] - totals[2], all_passed=True)), flush=True)
