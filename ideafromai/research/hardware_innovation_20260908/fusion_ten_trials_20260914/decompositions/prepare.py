"""Copy fixed existing fixtures only; never regenerate or modify RTL/TB."""
from pathlib import Path
import json,shutil,subprocess,sys
H=Path(__file__).resolve().parent
OLD=H.parents[1]/'r8_consumer_fusion_20260914/packed_rtl'
meta=json.loads((OLD/'definition.json').read_text())
for name in ['q1_bitplanes','q2_da','q1_dictionary']:
 h=H/name
 for c in meta['fixtures']:
  d=h/'fixtures'/c['name'];d.mkdir(parents=True,exist_ok=True)
  for f in ['source.hex','origin.hex','q1.hex','q2.hex','k_live.hex','gold.hex']:shutil.copyfile(OLD/'fixtures'/c['name']/f,d/f)
 m=dict(meta);m.update(modes={'14':'latest dual-P native-window packed cachedOS with shared extra resources','15':name},
  source_definition=str(OLD/'definition.json'),configured_cycles=4258 if name=='q1_dictionary' else 3361)
 (h/'definition.json').write_text(json.dumps(m,indent=2)+'\n')
subprocess.run([sys.executable,str(H/'q1_dictionary/prepare_dictionary.py')],check=True)
