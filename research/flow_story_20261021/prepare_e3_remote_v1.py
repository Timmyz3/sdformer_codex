"""Bind the completed E0 endpoint and perform E3 CPU preflight; never launch GPU."""
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
SSH = ["ssh", "-F", "/dev/null", "-o", "ControlPath=/tmp/flow_a100_transfer_%C",
       "-o", "BatchMode=yes", "-p", "30449", "root@36.213.75.115"]
PYTHON = "/data/flow_a/envs/v2/bin/python"
BASE = "/data/flow_a/code/research_20261021"
E3 = BASE + "/e3_rne_probe"
RUN = "/data/flow_a/runs/round32_20261021/e0_recovery_v1"
CONTRACT = E3 + "/ENDPOINT_CONTRACT_V1.json"
PREFLIGHT = E3 + "/REMOTE_PREFLIGHT_V1.json"
core = BASE + "/e0_ordinary/mixed_integer_core.py"
selection = BASE + "/selections_v1.json"
binding = [
    PYTHON, "-B", E3+"/probe_contract.py",
    "--core", core, "--inventory", E3+"/INVENTORY_V1.json",
    "--selections", selection, "--endpoint-result", RUN+"/results.json",
    "--weights", RUN+"/recovered_step512.pt", "--output", CONTRACT,
]
receipt = json.loads(subprocess.check_output(SSH+[shlex.join(binding)], text=True))
assert receipt["status"] == "FROZEN_ENDPOINT_STEP512_PENDING_ROOT_EXECUTION_DECISION"
frozen = json.loads((ROOT/"e3_rne_probe/SOURCE_FREEZE_V1.json").read_text())
args = [
    PYTHON, "-B", E3+"/collect_probe.py", "--e0-core", core,
    "--r28-root", "/data/flow_a/code/research_20261019/round28",
    "--r26-root", "/data/flow_a/code/research_20261017/window_precision_round26",
    "--members", "/data/flow_a/code/research_20261019/round27/independent_review/membership_v1.json",
    "--weights", RUN+"/recovered_step512.pt",
    "--endpoint-result", RUN+"/results.json", "--selections", selection,
    "--contract", CONTRACT, "--contract-sha256", receipt["sha256"],
    "--source-freeze", E3+"/SOURCE_FREEZE_V1.json",
    "--source-freeze-sha256", "bc486bf63585c905a464e4393dabbc5cd16c622185ec293423ee70799468847a",
    "--runner-sha256", frozen["files"]["collect_probe.py"],
    "--output", "/data/flow_a/runs/round32_20261021/e3_rne_probe_v1",
    "--wall-minutes", "60",
]
outfile = ROOT/"e3_rne_probe/LAUNCH_ARGS_V1.json"
with outfile.open("x") as f:
    json.dump(args, f, indent=2)
    f.write("\n")
subprocess.run(SSH+[shlex.join(args+["--preflight", PREFLIGHT])], check=True)
for remote, local in [(CONTRACT, "ENDPOINT_CONTRACT_V1.json"),
                      (PREFLIGHT, "REMOTE_PREFLIGHT_V1.json")]:
    subprocess.run(["scp", "-o", "ControlPath=/tmp/flow_a100_transfer_%C", "-P", "30449",
                    "root@36.213.75.115:"+remote, str(ROOT/"e3_rne_probe"/local)], check=True)
print(json.dumps(dict(status="ENDPOINT_BOUND_CPU_PREFLIGHT_COMPLETE_GPU_NOT_RUN",
                      contract=receipt, launch_args=str(outfile)), indent=2))
