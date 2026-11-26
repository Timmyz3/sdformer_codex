#!/usr/bin/env python3
"""Post-primary descriptive all-action graph sensitivity, no method selection."""
from pathlib import Path
import hashlib,json

HERE=Path(__file__).resolve().parent
OLD=HERE.parent.parent/'dense_reference_round17/independent_review/ACTUAL_RAW_QUALITY_V1.json'
NEW=HERE/'ACTUAL_RAW_QUALITY_V1.json'
SINGLETONS=tuple(f'fine_{r}_{c}' for r in range(3) for c in range(4))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def run():
    assert sha(OLD)=='d3a0901a24dd6f15407dcb9425ecda9be2daa33c947ba279bd2b30e19f6e7990'
    assert sha(NEW)=='8e9cb8296f2b2b0060204339be08e0717a7707d7160ec22396b103b7c725fc86'
    a,b=json.loads(OLD.read_text()),json.loads(NEW.read_text())
    assert [f['target_id'] for f in a['frames']]==[f['target_id'] for f in b['frames']]
    out=[]
    for x,y in zip(a['frames'],b['frames']):
        rows={}
        for arm in ('native','FF','CC')+SINGLETONS:
            regs={}
            for reg,m in x['arms'][arm].items():
                n=y['arms'][arm][reg];assert m['valid_pixels']==n['valid_pixels']
                regs[reg]={'valid_pixels':m['valid_pixels'],'dense_AEE':m['AEE'],'compact_AEE':n['AEE'],
                  'compact_minus_dense_AEE':None if m['AEE'] is None else n['AEE']-m['AEE'],
                  'all_metric_fields_equal':m==n}
            rows[arm]=regs
        gains=[]
        for arm in SINGLETONS:
            gx=x['arms']['CC']['all_valid']['AEE']-x['arms'][arm]['all_valid']['AEE']
            gy=y['arms']['CC']['all_valid']['AEE']-y['arms'][arm]['all_valid']['AEE']
            gains.append({'arm':arm,'dense_gain':gx,'compact_gain':gy,'difference':gy-gx,'sign_changed':(gx>0)-(gx<0)!=(gy>0)-(gy<0)})
        ranking=lambda key:sorted(range(12),key=lambda i:(-gains[i][key],i))
        out.append({'target_id':x['target_id'],'all15_endpoints_all16_regions':rows,'signed_allGT_gains':gains,
          'complete_singleton_order_unchanged':ranking('dense_gain')==ranking('compact_gain')})
    return {'schema':'flow_r19_graph_dependence_description_v1','status':'COMPLETE','source_sha256':sha(Path(__file__)),
      'dense_quality_sha256':sha(OLD),'compact_quality_sha256':sha(NEW),'frames':out,
      'all96_singleton_all16_metric_cells_equal':all(v['all_metric_fields_equal'] for f in out for arm in SINGLETONS for v in f['all15_endpoints_all16_regions'][arm].values()),
      'allGT_gain_sign_changes':sum(g['sign_changed'] for f in out for g in f['signed_allGT_gains']),
      'all8_complete_singleton_order_unchanged':all(f['complete_singleton_order_unchanged'] for f in out),
      'scope':'Authorized post-primary full-population diagnostic. Metric equality is not raw-array identity or general equivalence; no label transfer/GT method choice.'}

if __name__=='__main__':
    p=HERE/'GRAPH_DEPENDENCE_DESCRIPTION_V1.json';assert not p.exists()
    o=run();p.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:o[k] for k in ('status','all96_singleton_all16_metric_cells_equal','allGT_gain_sign_changes','all8_complete_singleton_order_unchanged')}))
