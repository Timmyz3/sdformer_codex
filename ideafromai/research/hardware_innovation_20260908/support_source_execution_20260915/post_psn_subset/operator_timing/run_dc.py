#!/opt/anaconda3/bin/python3.12
from pathlib import Path
import os,subprocess,fcntl,time,json
P=Path(__file__).resolve().parent
assert json.loads((P/'FUNCTIONAL.json').read_text())['status']=='PASS'
lib='/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db'
assert Path(lib).is_file()
fd=open('/tmp/date_dual_synopsys_same_uid_eda_queue.lock','a')
while True:
 try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);break
 except BlockingIOError:print('Waiting for shared EDA queue lock',flush=True);time.sleep(15)
print('Shared EDA queue lock acquired; two sequential DC points only',flush=True)
env=os.environ.copy()
env.update(SNPSLMD_LICENSE_FILE='27030@ic.ismd-nemo',LM_LICENSE_FILE='27030@ic.ismd-nemo',OP_TIMING_DIR=str(P),OP_TIMING_LIB=lib)
receipts=[]
try:
 for arm in [0,1]:
  out=P/f'dc_{arm}';out.mkdir(exist_ok=True)
  marker=out/'launch.json'
  if marker.exists():raise RuntimeError('This synthesis point has already launched; no automatic rerun')
  marker.write_text(json.dumps({'arm':arm,'launch_time':time.strftime('%Y-%m-%d %H:%M:%S'),'clock_ns':3})+'\n')
  env['OP_TIMING_ARM']=str(arm)
  print(f'Launching DC arm {arm}',flush=True)
  with (out/'dc.log').open('w') as log:
   cp=subprocess.run(['dc_shell','-f',str(P/'map.tcl')],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
  row={'arm':arm,'returncode':cp.returncode,'area_report':(out/'reports/area.rpt').exists(),'timing_report':(out/'reports/timing_all.rpt').exists()}
  receipts.append(row);(P/'DC_RUN.json').write_text(json.dumps(receipts,indent=2)+'\n')
  print(json.dumps(row),flush=True)
  if cp.returncode or not row['area_report'] or not row['timing_report']:break
finally:
 fcntl.flock(fd,fcntl.LOCK_UN);fd.close()
 print('Shared EDA queue lock released',flush=True)

