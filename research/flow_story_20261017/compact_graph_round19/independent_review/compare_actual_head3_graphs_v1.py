#!/usr/bin/env python3
"""Read all matched raw head3 arrays across separately audited R17/R19 runs."""
from pathlib import Path
import hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent
DATE=HERE.parent.parent
OLD=DATE/'dense_reference_round17/actual_v1/run_v1'
NEW=HERE.parent/'actual_v1/run_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def run():
    assert sha(OLD/'results.json')=='0b4ef2744c9da8fd3513b65141b4133f381cf267db68083e4bb854165ae07a2e'
    assert sha(NEW/'results.json')=='a7bf3bfbb2dea82eb0a02830e48575ed58cbddbfbb4cb914a27f99e76fdbe214'
    a=json.loads((OLD/'results.json').read_text());b=json.loads((NEW/'results.json').read_text())
    old={(r['target_id'],r['arm']):r for r in a['forward_records']}
    new={(r['target_id'],r['arm']):r for r in b['forward_records'] if r['quality_endpoint']}
    assert len(old)==len(new)==120 and old.keys()==new.keys()
    rows=[]
    for key,x in old.items():
        y=new[key];arrays=[]
        for base,r in ((OLD,x),(NEW,y)):
            p=base/r['prediction_file'];assert sha(p)==r['prediction_file_sha256']
            with np.load(p,allow_pickle=False) as z:arr=z['head3'].copy()
            assert arr.dtype==np.float32 and arr.shape==(1,2,480,640) and np.isfinite(arr).all()
            assert payload(arr)==r['head_records'][3]['sha256'];arrays.append(arr)
        xarr,yarr=arrays;xh,yh=payload(xarr),payload(yarr)
        rows.append({'target_id':key[0],'arm':key[1],'dense_head3_sha256':xh,'compact_head3_sha256':yh,
          'actual_bytes_equal':xarr.tobytes()==yarr.tobytes(),
          'maximum_absolute_difference':float(np.max(np.abs(xarr.astype(np.float64)-yarr)))})
    single=[r for r in rows if r['arm'].startswith('fine_')];assert len(single)==96
    return {'schema':'flow_r19_R17_actual_head3_pairing_v1','status':'COMPLETE','source_sha256':sha(Path(__file__)),
      'raw_archive_pairs_checked':120,'raw_head3_arrays_read':240,'all96_singleton_head3_bytes_equal':all(r['actual_bytes_equal'] for r in single),
      'differing_pairs':[{'target_id':r['target_id'],'arm':r['arm']} for r in rows if not r['actual_bytes_equal']],
      'rows':rows,'scope':'Post-primary descriptive complete matched raw head3 population; not arbitrary-mask/device equivalence, no GT or label reuse.'}

if __name__=='__main__':
    p=HERE/'ACTUAL_HEAD3_GRAPH_PAIRING_V1.json';assert not p.exists()
    out=run();p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:out[k] for k in ('status','all96_singleton_head3_bytes_equal','differing_pairs')}))
