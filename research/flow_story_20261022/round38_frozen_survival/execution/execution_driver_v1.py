"""One bounded R38 launch; process supervision outside CUDA Python."""
from pathlib import Path
import subprocess,json,time,datetime,hashlib,sys
root=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
assert read(root/'REMOTE_CPU_PREFLIGHT_V1.json')['status']=='CPU_PREFLIGHT_PASS'
command=read(root/'GPU_COMMAND_V1.json');receipt=root/'GPU_PROCESS_RECEIPT_V1.json';assert not receipt.exists()
check=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
assert not check.stdout.strip(),'Dedicated GPU occupied; no launch'
v=dict(schema='flow_r38_bounded_process_receipt_v1',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),gpu_processes_before=check.stdout,external_timeout_seconds=900,kill_grace_seconds=5,status='STARTING')
with (root/'GPU_STDOUT_V1.log').open('x') as log:
    p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);v.update(pid=p.pid,status='RUNNING');write(receipt,v);started=time.monotonic();exitcode=p.wait()
v.update(exit_code=exitcode,wall_seconds=time.monotonic()-started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PROCESS_COMPLETE')
r=Path('/data/flow_a/runs/round38_20261022/survival_v1/results.json')
if r.exists():v['result_sha256']=hashlib.sha256(r.read_bytes()).hexdigest();v['collector_status']=read(r).get('status')
write(receipt,v)
