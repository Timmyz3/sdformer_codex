"""Run the frozen E0 recovery on the dedicated A100 after actual smoke audit."""
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parent
SSH = [
    "ssh", "-F", "/dev/null",
    "-o", "ControlPath=/tmp/flow_a100_transfer_%C",
    "-o", "BatchMode=yes", "-p", "30449",
    "root@36.213.75.115",
]
audit_path = ROOT / "independent_review/E0_SMOKE_ACTUAL_AUDIT_V1.json"
audit = json.loads(audit_path.read_text())
if audit["status"] != "PASS" or not audit["engineering_smoke"]:
    raise RuntimeError("Actual smoke execution has not passed independent audit")
args = json.loads((ROOT / "e0_ordinary/LAUNCH_ARGS_V2.json").read_text())["recovery"]
log = "/data/flow_a/runs/round32_20261021/e0_recovery_v1.log"
output = "/data/flow_a/runs/round32_20261021/e0_recovery_v1"
receipt = ROOT / "e0_ordinary/FORMAL_LAUNCH_V1.json"
if receipt.exists():
    raise RuntimeError("Fresh launch receipt required; do not repeat a launched run")
gpu_processes = subprocess.check_output(
    SSH + ["nvidia-smi --query-compute-apps=pid --format=csv,noheader"],
    text=True,
)
if gpu_processes.strip():
    raise RuntimeError("Dedicated GPU is occupied; no launch")
command = (
    "test ! -e " + shlex.quote(log)
    + " && test ! -e " + shlex.quote(output)
    + " && OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 "
    + shlex.join(["timeout", "--signal=TERM", "--kill-after=30s", "3700"] + args + ["--execute"])
    + " > " + shlex.quote(log) + " 2>&1"
)
record = dict(
    schema="flow_e0_formal_launch_v1",
    status="LAUNCHED",
    smoke_audit=str(audit_path),
    remote_argv=args + ["--execute"],
    output=output,
    log=log,
    initial_GPU_compute_processes=gpu_processes.strip(),
    smoke_weights_reused=False,
    stdout_log_only=True,
)
receipt.write_text(json.dumps(record, indent=2) + "\n")
start = time.monotonic()
ret = subprocess.run(SSH + [command])
record.update(
    status="REMOTE_PROCESS_EXITED",
    returncode=ret.returncode,
    elapsed_seconds=time.monotonic() - start,
)
receipt.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
raise SystemExit(ret.returncode)
