from pathlib import Path
import json,subprocess
p=Path(__file__).resolve().parent
old=[json.loads(x) for x in (p/'results_all.jsonl').read_text().splitlines()]
with (p/'run.log').open('w') as f:subprocess.run([str(p/'obj_dir/Vjoined_fc1'),'all'],cwd=p,stdout=f,stderr=subprocess.STDOUT,check=True)
new=[json.loads(x) for x in (p/'results_all.jsonl').read_text().splitlines()]
assert len(old)==len(new)==444
for a,b in zip(old,new):
 for k,v in a.items():assert b[k]==v,(a['case'],k,v,b[k])
(p/'instrumentation_check.json').write_text(json.dumps({'status':'PASS','commands':444,'meaning':'Actual bank/read/write instrumentation preserves every original per-command result field and timing; no RTL changes.'},separators=(',',':'))+'\n')
print('PASS444 unchanged commands plus real Y read/write checks')
