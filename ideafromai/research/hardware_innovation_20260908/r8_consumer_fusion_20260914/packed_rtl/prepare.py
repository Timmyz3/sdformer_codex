from pathlib import Path
import json,shutil
H=Path(__file__).resolve().parent
OLD=H.parents[1]/'r0_stream_fusion_20260914/integer_factor'
meta=json.loads((OLD/'definition.json').read_text())
for case in meta['fixtures']:
    d=H/'fixtures'/case['name'];d.mkdir(parents=True,exist_ok=True)
    for f in ['source.hex','origin.hex','q1.hex','q2.hex','k_live.hex','gold.hex']:
        shutil.copyfile(OLD/'fixtures'/case['name']/f,d/f)
meta.update(source_definition=str(OLD/'definition.json'),modes={'14':'native-window cachedOS scalar packed26 control','15':'same resources dual-P packed26'},
    function='p=Q2@(Q1@g), no intermediate RNE, full C96/N96/K864/T10/R8',
    z_layout='8 banks x20rows x26bits: row=(p//2)*10+t; half=p%2',
    multiplier='8 shared signed19x13->signed32',configured_cycles=3361)
(H/'definition.json').write_text(json.dumps(meta,indent=2)+'\n')
print('Prepared',len(meta['fixtures']),'fixtures')
