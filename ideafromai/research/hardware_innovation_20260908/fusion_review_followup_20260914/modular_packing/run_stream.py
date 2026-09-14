import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

H = Path(__file__).resolve().parent
B = H.parents[1]
D = B / "r8_consumer_fusion_20260914/data"
P = B / "fusion_ten_trials_20260914/phase_borrow"
checks = json.loads((H / "checks.json").read_text())
assert checks["passed"] and checks["gate_64"]
with (H / "build_stream.log").open("w") as log:
    subprocess.run(["verilator", "-Wall", "--cc", "--exe", "--top-module", "consumer_stream", "--Mdir", "obj_stream",
                    "consumer_stream.sv", "modular_core.sv", "i24_consumer.sv", "stream_tb.cpp", "-CFLAGS", "-O3 -std=c++14"],
                   cwd=H, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(["make", "-C", "obj_stream", "-f", "Vconsumer_stream.mk", "-j2"], cwd=H, stdout=log, stderr=subprocess.STDOUT, check=True)

def run(job):
    mode, stall = job
    cmd = [str(H / "obj_stream/Vconsumer_stream"),
           *[str(D / f) for f in ["first_source_words.npy", "identity_fp32_full.npy", "raw_p_full.npy", "i24_new_full.npy", "identity_q20_full.npy"]],
           str(P / "fixtures/real_0"), str(mode), "128", "64", str(stall), "2", "10000000"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    name = f"64_m{mode}_s{stall}"
    (H / "logs" / f"{name}.jsonl").write_text(result.stdout)
    (H / "logs" / f"{name}.err").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError((name, result.returncode, result.stderr))
    out = [json.loads(line) for line in result.stdout.splitlines()]
    print(json.dumps({"mode": mode, "stall": stall, "cycles": [r["total_cycles"] for r in out]}), flush=True)
    return out

rows = []
with ThreadPoolExecutor(max_workers=3) as pool:
    for result in pool.map(run, [(m, s) for m in [0, 1, 2] for s in [0, 1]]):
        rows.extend(result)
(H / "results_64.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps({"commands": len(rows), "values_each_stage": sum(r["outputs"] for r in rows), "passed": True}))
