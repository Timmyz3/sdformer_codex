import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

H = Path(__file__).resolve().parent
B = H.parents[1]
D = B / "r8_consumer_fusion_20260914/data"
P = B / "fusion_ten_trials_20260914/phase_borrow"
parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["small", "64"], default="small")
args = parser.parse_args()
if args.stage == "64":
    checks = json.loads((H / "checks.json").read_text())
    assert checks["passed"] and checks["gate_64"]
with (H / "build_stream.log").open("w") as log:
    subprocess.run(["verilator", "-Wall", "--cc", "--exe", "--top-module", "interleave_stream", "--Mdir", "obj_stream",
                    "interleave_stream.sv", "rr_context.sv", "i24_consumer.sv", "wide_phase_alu.sv", "stream_tb.cpp", "-CFLAGS", "-O3 -std=c++14"],
                   cwd=H, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(["make", "-C", "obj_stream", "-f", "Vinterleave_stream.mk", "-j2"], cwd=H, stdout=log, stderr=subprocess.STDOUT, check=True)

def run(job):
    first, count, mode, stall = job
    name = f"first{first}_n{count}_m{mode}_s{stall}"
    cmd = [str(H / "obj_stream/Vinterleave_stream"),
           *[str(D / f) for f in ["first_source_words.npy", "identity_fp32_full.npy", "raw_p_full.npy", "i24_new_full.npy", "identity_q20_full.npy"]],
           str(P / "fixtures/real_0"), str(mode), str(first), str(count), str(stall), "2", "10000000"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    (H / "logs" / f"{name}.jsonl").write_text(result.stdout)
    (H / "logs" / f"{name}.err").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError((name, result.returncode, result.stderr))
    out = [json.loads(line) for line in result.stdout.splitlines()]
    print(json.dumps({"job": job, "cycles": [r["total_cycles"] for r in out]}), flush=True)
    return out

ranges = [(159, 3), (19197, 3)] if args.stage == "small" else [(128, 64)]
rows = []
with ThreadPoolExecutor(max_workers=3) as pool:
    for result in pool.map(run, [(first, count, mode, stall) for first, count in ranges for mode in [1, 2, 3] for stall in [0, 1]]):
        rows.extend(result)
name = "results_small.json" if args.stage == "small" else "results_64.json"
(H / name).write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps({"commands": len(rows), "values_each_stage": sum(r["outputs"] for r in rows), "passed": True}))
