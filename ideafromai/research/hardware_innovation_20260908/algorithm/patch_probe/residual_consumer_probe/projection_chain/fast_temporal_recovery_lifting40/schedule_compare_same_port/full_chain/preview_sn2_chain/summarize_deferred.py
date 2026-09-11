"""Summarize measured service and compare every saved output bit across arms."""
import json
import numpy as np
from run_chain import HERE,difference


def compare(a,b):
    out={}
    for k in a.files:
        aa,bb=a[k],b[k]
        r=difference(aa,bb)
        r['bitwise_differences']=int(np.count_nonzero(aa.view('u'+str(aa.itemsize))!=bb.view('u'+str(bb.itemsize))))
        assert r['bitwise_differences']==0,(k,r)
        out[k]=r
    return out


def main():
    result=dict(scope='Two local source-to-final-PED windows; global BN producer wait unresolved. CPU service, not RTL.',
        table=[],ready_stress_equality={},direct_deferred_equality={},complete_layer=False)
    for stress in (False,True):
        suffix='_stress' if stress else '';key='stress' if stress else 'ready'
        axes={a:json.loads((HERE/f'deferred_compare_{a}{suffix}.json').read_text())['windows'] for a in ('ordinary','lifting_raw')}
        for label in ('corner','interior'):
            o,l=axes['ordinary'][label],axes['lifting_raw'][label]
            row=dict(condition=key,window=label,
                ordinary_direct=o['direct']['local_active_service_slots'],lifting_direct=l['direct']['local_active_service_slots'],
                ordinary_deferred=o['deferred']['local_active_service_slots'],lifting_deferred=l['deferred']['local_active_service_slots'])
            row['lifting_reduction_with_deferred']=1-row['lifting_deferred']/row['ordinary_deferred']
            row['ordinary_defer_reduction']=1-row['ordinary_deferred']/row['ordinary_direct']
            row['lifting_defer_reduction']=1-row['lifting_deferred']/row['lifting_direct']
            result['table'].append(row)
            for axis in axes:
                paths=[HERE/f'{axis}_{label}_{m}{suffix}_completion.npz' for m in ('direct','deferred')]
                with np.load(paths[0]) as a,np.load(paths[1]) as b:
                    result['direct_deferred_equality'][f'{axis}/{label}/{key}']=compare(a,b)
    for axis in ('ordinary','lifting_raw'):
        for label in ('corner','interior'):
            for mode in ('direct','deferred'):
                with np.load(HERE/f'{axis}_{label}_{mode}_completion.npz') as a,np.load(HERE/f'{axis}_{label}_{mode}_stress_completion.npz') as b:
                    result['ready_stress_equality'][f'{axis}/{label}/{mode}']=compare(a,b)
    (HERE/'deferred_compare_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['table'],indent=2))


if __name__=='__main__':main()
