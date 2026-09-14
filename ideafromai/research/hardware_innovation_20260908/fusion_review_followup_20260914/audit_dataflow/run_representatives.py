"""Read old RTL/fixtures; build and save every new receipt in this audit directory."""
import json
import shutil
import subprocess
from pathlib import Path

H = Path(__file__).resolve().parent
B = H.parents[1]
OLD = B / "fusion_ten_trials_20260914"
D = B / "r8_consumer_fusion_20260914/data"
FAMILIES = {
    "d1_forward": ("dataflow/d1_forward", "consumer_stream", [1, 2]),
    "d2_halo": ("dataflow/d2_halo", "consumer_stream", [0, 1]),
    "d3_interleave": ("dataflow/d3_interleave", "interleave_stream", [0, 1, 2]),
    "phase_borrow": ("phase_borrow", "consumer_stream", [2, 3]),
    "joint_selected": ("joint_selected", "consumer_stream", [2, 3]),
}
INPUTS = ["first_source_words.npy", "identity_fp32_full.npy", "raw_p_full.npy", "i24_new_full.npy", "identity_q20_full.npy"]
rows = []
for family, (relative, top, modes) in FAMILIES.items():
    source = OLD / relative
    build = H / family
    build.mkdir(exist_ok=True)
    sv = sorted(source.glob("*.sv"))
    for p in sv + [source / "stream_tb.cpp"]:
        shutil.copy2(p, build / p.name)
    with (build / "build.log").open("w") as log:
        subprocess.run(["verilator", "-Wall", "--cc", "--exe", "--top-module", top,
                        "--Mdir", "obj", *[p.name for p in sv], "stream_tb.cpp", "-CFLAGS", "-O3"],
                       cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(["make", "-C", "obj", "-f", f"V{top}.mk", "-j2"],
                       cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True)
    # Cross-row odd batch and the frame's final odd batch, cold + warm, BP + no BP.
    for first in [159, 19197]:
        for mode in modes:
            for stall in [0, 1]:
                name = f"first{first}_m{mode}_s{stall}"
                cmd = [str(build / "obj" / f"V{top}"), *[str(D / p) for p in INPUTS],
                       str(source / "fixtures/real_0"), str(mode), str(first), "3", str(stall), "2", "10000000"]
                with (build / f"{name}.err").open("w") as err:
                    run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=err, text=True, check=True)
                (build / f"{name}.jsonl").write_text(run.stdout)
                result = [dict(json.loads(line), family=family) for line in run.stdout.splitlines()]
                rows.extend(result)
    # Recompute the old 64-tile no-BP receipt from these newly built source copies.
    old64 = json.loads((source / "results_64.json").read_text())
    for mode in modes:
        cmd = [str(build / "obj" / f"V{top}"), *[str(D / p) for p in INPUTS],
               str(source / "fixtures/real_0"), str(mode), "128", "64", "0", "2", "10000000"]
        run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        (build / f"match64_m{mode}.jsonl").write_text(run.stdout)
        for line in run.stdout.splitlines():
            record = json.loads(line)
            old = next(r for r in old64 if r["mode"] == mode and r["stall"] == 0 and r["command"] == record["command"])
            for key, value in record.items():
                if key != "wall_seconds_so_far":
                    assert old[key] == value, (family, key, old[key], value)
            rows.append(dict(record, family=family, old64_match=True))
    (H / "representative_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"family": family, "new_commands": sum(r["family"] == family for r in rows), "old64": "all counters match"}), flush=True)
print(json.dumps({"commands": len(rows), "scalar_outputs_each_stage": sum(r["outputs"] for r in rows), "passed": True}), flush=True)
