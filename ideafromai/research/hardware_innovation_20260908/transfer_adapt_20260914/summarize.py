"""One table from completed RTL results; resource points remain explicit."""
from pathlib import Path
import csv
import json

H=Path(__file__).resolve().parent
out=[]
for subset,stage in [('128–191','held'),('4000–4063','disjoint')]:
    source=H/'pair_psum_overlay/temporal_pairing'/f'results_{stage}.json'
    rows=json.loads(source.read_text())
    for mode,name in [(14,'native_dual_208'),(20,'count8_psum'),(21,'count8_psum_fixed_time_pairs')]:
        a=[r for r in rows if r['mode']==mode and not r['stall'] and r['command']<64]
        assert len(a)==64
        core=sum(r['cycles'] for r in a);cfg=sum(r['configuration_cycles'] for r in a)
        out.append(dict(tiles=subset,mechanism=name,z_port_bits=208,z_bytes=520,core_cycles=core,configuration_cycles=cfg,start_cycles=64,service_cycles=core+cfg+64,source=str(source.relative_to(H))))
    source=H/'temporal_endpoints'/('SUMMARY_stream.json' if stage=='held' else 'SUMMARY_disjoint.json')
    a=json.loads(source.read_text())
    for mode,name in [(4,'fixed_order_mixed_endpoint'),(5,'fixed_order_all_endpoint')]:
        r=a[f'm{mode}_s0_repeat0']
        # Both modes are compiled in the 1040B two-domain module. Mode5 only
        # initializes/uses 520B; do not claim the unused array was synthesized away.
        out.append(dict(tiles=subset,mechanism=name,z_port_bits=208,z_bytes=1040,core_cycles=r['cycles'],configuration_cycles=r['configuration_cycles'],start_cycles=64,service_cycles=r['service_cycles'],source=str(source.relative_to(H))))
    source=H/'strong_raw_control'/('SUMMARY_64.json' if stage=='held' else 'SUMMARY_disjoint.json')
    a=json.loads(source.read_text())
    for mode,name in [(0,'native_dual_416'),(1,'native_triple_proven'),(2,'native_four_repaired')]:
        r=a[f'm{mode}_s0_repeat0']
        out.append(dict(tiles=subset,mechanism=name,z_port_bits=416,z_bytes=520,core_cycles=r['cycles'],configuration_cycles=r['configuration_cycles'],start_cycles=64,service_cycles=r['service_cycles'],source=str(source.relative_to(H))))
for r in out:
    assert r['service_cycles']==r['core_cycles']+r['configuration_cycles']+r['start_cycles']
    native=next(a for a in out if a['tiles']==r['tiles'] and a['mechanism']=='native_dual_208')
    r['fraction_fewer_cycles_vs_dual208']=1-r['service_cycles']/native['service_cycles']
with (H/'comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]),lineterminator='\n');w.writeheader();w.writerows(out)
(H/'comparison.json').write_text(json.dumps(dict(rows=out,scope='Verilator raw service only; differing z ports/state, not equal-area or I24/SoC comparison; same frame with two output regions'),indent=2)+'\n')
print('Wrote 16 measured same-input rows with explicit port/state distinctions.')
