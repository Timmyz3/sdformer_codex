"""Execute one frozen 118-forward probe after the complete E0 review decision."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parent
audit_file = ROOT/"independent_review/E0_FORMAL_ACTUAL_AUDIT_V1.json"
decision_file = ROOT/"e3_rne_probe/EXECUTION_DECISION_V1.json"
audit = json.loads(audit_file.read_text())
decision = json.loads(decision_file.read_text())
assert audit["status"] == "PASS"
assert decision["execute"] is True and decision["model_forwards"] == 118
assert decision["train_predictor"] is False
assert decision["E0_audit_sha256"] == hashlib.sha256(audit_file.read_bytes()).hexdigest()
args = json.loads((ROOT/"e3_rne_probe/LAUNCH_ARGS_V1.json").read_text())
ssh = ["ssh", "-F", "/dev/null", "-o", "ControlPath=/tmp/flow_a100_transfer_%C",
       "-o", "BatchMode=yes", "-p", "30449", "root@36.213.75.115"]
occupied = subprocess.check_output(
    ssh+["nvidia-smi --query-compute-apps=pid --format=csv,noheader"], text=True
)
if occupied.strip():
    raise RuntimeError("GPU occupied; do not launch")
log = "/data/flow_a/runs/round32_20261021/e3_rne_probe_v1.log"
output = args[args.index("--output")+1]
command = (
    "test ! -e "+shlex.quote(log)+" && test ! -e "+shlex.quote(output)
    + " && OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 "
    + shlex.join(["timeout", "--signal=TERM", "--kill-after=30s", "3700"]+args+["--execute"])
    + " > "+shlex.quote(log)+" 2>&1"
)
path = ROOT/"e3_rne_probe/ACTUAL_LAUNCH_V1.json"
receipt = dict(
    schema="flow_e3_actual_launch_v1", status="LAUNCHED",
    decision_sha256=hashlib.sha256(decision_file.read_bytes()).hexdigest(),
    remote_argv=args+["--execute"], log=log, output=output,
    GPU_initial_processes=occupied.strip(), formal_forwards=112, engineering_forwards=6,
    training=False,
)
with path.open("x") as f:
    json.dump(receipt, f, indent=2)
    f.write("\n")
start = time.monotonic()
result = subprocess.run(ssh+[command])
receipt.update(status="REMOTE_PROCESS_EXITED", returncode=result.returncode,
               elapsed_seconds=time.monotonic()-start)
path.write_text(json.dumps(receipt, indent=2)+"\n")
print(json.dumps(receipt, indent=2))
raise SystemExit(result.returncode)
