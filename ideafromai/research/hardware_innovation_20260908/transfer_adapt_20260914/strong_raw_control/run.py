"""Reuse the completed modular-packing RTL unchanged, at the raw boundary.

The 416-bit z port is wider than pair/endpoint's 208-bit port. This is a
stronger resource point, not an equal-area comparison. No consumer ALU is
borrowed; all three modes use the original eight 32-bit producer chains.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import subprocess

H = Path(__file__).resolve().parent
B = H.parents[1]
P = H.parent / 'pair_sparse'
OLD = B / 'fusion_review_followup_20260914/modular_packing'
parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=['small', '64', 'disjoint'], default='small')
args = parser.parse_args()

if args.stage == 'small':
    # Existing datapath copied verbatim; only its external raw harness changes.
    (H / 'modular_core.sv').write_text((OLD / 'modular_core.sv').read_text())
    s = (P / 'stream_tb.cpp').read_text().replace('Vdecomp_core', 'Vmodular_core')
    start = s.index(' if(mode>=15){')
    end = s.index(' d.cfg_valid=0;unsigned checked=0;', start)
    s = s[:start] + s[end:]
    start = s.index('<<",\\"dual_updates\\":')
    end = s.index('<<",\\"state_cycles\\":[";', start)
    s = s[:start] + '''<<",\\"merged_updates\\":"<<d.merged_updates
      <<",\\"repair_issues\\":"<<d.repair_issues<<",\\"repair_fields\\":"<<d.repair_fields
      <<",\\"normalization_issues\\":"<<d.normalization_issues
      <<",\\"range_ok\\":"<<int(d.range_ok)<<",\\"fallback_used\\":"<<int(d.fallback_used)
      ''' + s[end:]
    proof = '''
 bool expected_range=true;
 for(int lane=0;lane<8;++lane){
  int pos=0,neg=0;
  for(int k=0;k<864;++k){int q=int32_t(q1[k*8+lane]);if(q>=0)pos+=q;else neg+=q;}
  auto field=[&](const auto& bus){unsigned bit=lane*13,word=bit/32;uint64_t v=bus[word];if(word<3)v|=uint64_t(bus[word+1])<<32;int z=(v>>(bit%32))&8191;return z&4096?z-8192:z;};
  if(field(d.positive_bounds)!=pos||field(d.negative_bounds)!=neg)return 30;
  expected_range &= pos<=511&&neg>=-512;
 }
 if(bool(d.range_ok)!=expected_range)return 31;
 '''
    s = s.replace(' d.cfg_valid=0;unsigned checked=0;', ' d.cfg_valid=0;d.eval();' + proof + '\n unsigned checked=0;')
    s = s.replace('if(outputs!=480)return 6;', '''if(outputs!=480)return 6;
    if(bool(d.fallback_used)!=(mode==1&&!expected_range))return 32;
    if(d.normalization_issues!=(mode==2?10:0))return 33;
    if(mode!=2&&(d.repair_issues||d.repair_fields))return 34;''')
    (H / 'tb.cpp').write_text(s)
    with (H / 'build.log').open('w') as log:
        for cmd in [
            ['verilator', '-Wall', '--cc', '--exe', '--top-module', 'modular_core', '--Mdir', 'obj', 'modular_core.sv', 'tb.cpp', '-CFLAGS', '-O2 -std=c++14'],
            ['make', '-C', 'obj', '-f', 'Vmodular_core.mk', '-j2'],
        ]:
            subprocess.run(cmd, cwd=H, stdout=log, stderr=subprocess.STDOUT, check=True)
    cases = []
    for rec in json.loads((P / 'fixtures.json').read_text()):
        p = P / 'fixtures' / rec['name']
        if not p.exists():
            p = B / 'fusion_review_followup_20260914/pair_dictionary/fixtures' / rec['name']
        manifest = H / 'fixtures' / (rec['name'] + '.txt')
        manifest.parent.mkdir(exist_ok=True)
        manifest.write_text(str(p) + '\n')
        cases.append((rec['name'], p, manifest))
else:
    assert (H / 'checks_small.json').exists()
    manifest = (P / 'fixtures/stream/manifest.txt' if args.stage == '64'
                else H.parent / 'audit/fixtures/disjoint_4000/manifest.txt')
    cases = [(args.stage, B / 'fusion_review_followup_20260914/pair_dictionary/fixtures/real_0', manifest)]

def run(job):
    case, mode, stall = job
    label, params, manifest = case
    r = subprocess.run([str(H / 'obj/Vmodular_core'), str(params), str(mode), str(stall), str(manifest)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError((label, mode, stall, r.returncode, r.stdout[-3000:], r.stderr))
    paths = manifest.read_text().split()
    rows = [dict(json.loads(line), fixture=paths[i % len(paths)], case=label) for i, line in enumerate(r.stdout.splitlines())]
    assert len(rows) == 2 * len(paths)
    print('PASS', label, mode, stall, len(rows), flush=True)
    return rows

with ThreadPoolExecutor(max_workers=3) as pool:
    rows = [r for group in pool.map(run, [(c, m, s) for c in cases for m in [0, 1, 2] for s in [0, 1]]) for r in group]
(H / f'results_{args.stage}.json').write_text(json.dumps(rows, indent=2) + '\n')
checks = dict(passed=True, commands=len(rows), raw_values=sum(r['outputs'] for r in rows), original_datapath_unchanged=(H/'modular_core.sv').read_text()==(OLD/'modular_core.sv').read_text())
(H / f'checks_{args.stage}.json').write_text(json.dumps(checks, indent=2) + '\n')
if args.stage != 'small':
    summary = {}
    for mode in [0, 1, 2]:
        for stall in [0, 1]:
            for repeat in [0, 1]:
                a = [r for r in rows if r['mode']==mode and r['stall']==stall and r['command']//64==repeat]
                v = {k:sum(r[k] for r in a) for k in ['cycles','configuration_cycles','first_issues','repair_issues','repair_fields','normalization_issues','source_stalls','weight_stalls','output_stalls','z_vector_reads','z_scalar_reads','z_writes','mac_issues']}
                v['service_cycles'] = v['cycles'] + v['configuration_cycles'] + len(a)
                summary[f'm{mode}_s{stall}_repeat{repeat}'] = v
    (H / f'SUMMARY_{args.stage}.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
