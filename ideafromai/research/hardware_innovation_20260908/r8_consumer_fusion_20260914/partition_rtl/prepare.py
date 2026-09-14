from pathlib import Path
import json,shutil

HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'r0_stream_fusion_20260914/integer_factor'
meta=json.loads((OLD/'definition.json').read_text())
files=['source.hex','origin.hex','q1.hex','q2.hex','k_live.hex','gold.hex']
for case in meta['fixtures']:
    target=HERE/'fixtures'/case['name'];target.mkdir(parents=True,exist_ok=True)
    for name in files:shutil.copyfile(OLD/'fixtures'/case['name']/name,target/name)
(HERE/'definition.json').write_text(json.dumps({
    'source_definition':str(OLD/'definition.json'),'fixtures':meta['fixtures'],
    'function':'p=Q2@(Q1@g), no intermediate RNE; fullC96/N96/T10/K864,4x4to2x2',
    'modes':{'12':'cached-full-R8 Q2 output-stationary strong control','13':'signed-absolute partition, shared descriptors, finite8 coefficient cache'},
    'input':'native1536 T10 words; no latent input; all14oldfixtures copied without modification',
    'canonical_group':'leader lowest original rank; v=z_leader; sign_r=sign(z_r)*sign(v)',
    'coefficient_bound':[-262144,262143],'signed_coefficient_bits':19,
    'multiplier':'8 shared signed19x13->signed32','accumulator_bits':32,
    'cache_entries':8,'descriptor_capacity':320,'descriptor_bits':29,
    'configured_cycles':3361},indent=2)+'\n')
print('Prepared',len(meta['fixtures']),'fixtures')
