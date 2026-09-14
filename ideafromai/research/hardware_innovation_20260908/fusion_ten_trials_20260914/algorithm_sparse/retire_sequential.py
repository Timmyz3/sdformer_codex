"""Retire only the original sequence after its mode5 complete receipt; retain any unused mode6 prefix."""
import json,time,os,signal
from pathlib import Path
H=Path(__file__).resolve().parent;pid=1379522
while True:
 try:d=json.loads((H/'quality_valid_5.json').read_text())
 except (FileNotFoundError,json.JSONDecodeError):d={}
 if d.get('complete') and d.get('result',{}).get('frames')==825:
  command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
  assert 'evaluate_sparse.py --split valid --modes 1,2,5,6' in command,command
  os.kill(pid,signal.SIGTERM)
  time.sleep(1)
  prefix=H/'quality_valid_6.json'
  if prefix.exists():
   original=prefix.read_text();(H/'unused_sequential_mode6_prefix.json').write_text(original)
  rows=H/'quality_work/temporal_rank_deadband_frames.json'
  if rows.exists():(H/'unused_sequential_mode6_frames.json').write_text(rows.read_text())
  result=dict(complete=True,retired_pid=pid,reason='mode5 official825 complete; mode6 is evaluated by independent parallel process only',mode5_AEE=d['result']['AEE_frame_mean'],terminated_command=command,unused_mode6_prefix_exists=prefix.exists())
  (H/'sequential_retirement.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);break
 time.sleep(1)
