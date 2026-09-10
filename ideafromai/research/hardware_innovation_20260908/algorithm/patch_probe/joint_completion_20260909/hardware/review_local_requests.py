"""Independent address/column accounting for saved FP32 training decisions."""
from pathlib import Path
import json
import numpy as np

from training_request_costs import group_union_cost

HERE = Path(__file__).resolve().parent


def main():
    z = np.load(HERE/'training_request_costs.npz')
    ids = np.flatnonzero(z['splits']=='valid')
    words = z['source_words'][ids].reshape(-1,864)
    w_any = z['W_any_H8_k']
    column = z['column_W64_uses'][ids]
    full = z['full_batch_W64_uses'][ids]
    checks = {}
    jobs=[]
    for folder in ('local_train16','word_train16','optimized_prefix_train16'):
        root=HERE.parent/folder
        if not (root/'result.json').exists():continue
        original=json.loads((root/'result.json').read_text())
        jobs += [(root,axis,record) for axis,record in original['axes'].items()]
    for root,axis,record in jobs:
        decisions=np.load(root/(axis+'_valid_decisions.npz'))
        params=np.load(root/(axis+'.npz'))
        prefix_ids=params['prefix'].tolist()
        need=decisions['need_column'].reshape(len(ids),64,12,10)
        tail=need.copy();tail[...,prefix_ids]=False
        prefix_need=np.zeros_like(need);prefix_need[...,prefix_ids]=True
        prefix=group_union_cost(words,prefix_need.reshape(-1,12,10),w_any)
        # Full captured K order is retained. A vector address includes H8 and k;
        # only equal addresses within the SAME prefix/tail epoch are merged.
        cost=group_union_cost(words,tail.reshape(-1,12,10),w_any).reshape(len(ids),64,12)
        counts=dict(column_visits_full=int(column.sum()),
                    column_visits_early=int((column*need).sum()),
                    shared_w_full=int(full.sum()),
                    shared_w_prefix_tail=int(prefix.sum()+cost.sum()))
        checks[root.name+'/'+axis]=dict(counts=counts,
            prefix=prefix_ids,
            differences={k:float(v-record['valid'][k]) for k,v in counts.items()},
            full_10Y_one_epoch_ratio=counts['shared_w_prefix_tail']/counts['shared_w_full'],
            time_major_column_visit_ratio=counts['column_visits_early']/counts['column_visits_full'],
            all_3_prefix_columns_requested=bool(need[...,prefix_ids].all()),
            compiled_Q14_nonzeros=int(np.count_nonzero(params['temporal_q14'])))
    result=dict(
        scope='four FP32-parent validation captures, 64 P4/frame, all H96/T10; no integer gate or AEE verification',
        source_order='source k=((c*3)+kh)*3+kw; groups flatten as frame, spatial group, H8; h=8*H8+lane',
        dependency='train_local.demands reduces over output row t before max over H8/P4; correct E[t,s] direction',
        numeric_boundary='Q14 temporal A is evaluated with captured FP32 norm1 Y and floating radius/bias. This is a different student from compiled_integer.',
        whole_column_exact='every original FP32 and compiledW8 H8 vector has a nonzero (10368/10368 checked); P4-OR is exact for the saved all-or-none need_column schedule',
        per_lane_limit='gate-level cancellation needs separate p masks; this review does not equate whole-column counters with every-lane dynamic execution',
        physical_boundary='counts are H8 logical coefficient-vector uses within declared reuse epochs. Local trained Conv1 is FP32: one vector256bit/four64bit beats and331776B fullW; later W8 deployment is64bit/one beat and82944B. Vector ratios are not actualW64 transactions or same-capacity services.',
        checks=checks,
    )
    (HERE/'local_training_request_review.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(checks,ensure_ascii=False))


if __name__=='__main__':main()
