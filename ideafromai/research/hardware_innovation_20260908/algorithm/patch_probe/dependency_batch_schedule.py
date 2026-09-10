"""Train-selected temporal conv batches under finite retained-Y capacity.

All new B columns occupy convolution partial-sum slots simultaneously. After
the batch completes, ready PSN rows consume their Y, then last-use Y is freed.
Costs count logical P4-shared W vectors, never physical memory cycles.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path

from dependency_state_schedule import graph,retained,members

HERE=Path(__file__).resolve().parent
FULL=1023
CELLS=16588800


def batch_costs(hist):
    """cost[B]=sum(hist[pattern] for pattern intersecting B), via subset zeta."""
    assert len(hist)==1024 and all(isinstance(v,int) and v>=0 for v in hist)
    subset=hist.copy()
    for bit in range(10):
        for mask in range(1024):
            if mask>>bit&1:subset[mask]+=subset[mask^(1<<bit)]
    total=sum(hist)
    return [total-subset[FULL^batch] for batch in range(1024)]


def choose_schedule(rows,capacity,cost):
    """Exact min(reads,batches,lexicographic batch masks), train costs only."""
    @lru_cache(None)
    def solve(done):
        if done==FULL:return (0,0,())
        room=capacity-retained(rows,done).bit_count()
        if room<=0:return None
        remaining=FULL^done;batch=remaining;best=None
        while batch:
            if batch.bit_count()<=room:
                suffix=solve(done|batch)
                if suffix is not None:
                    candidate=(cost[batch]+suffix[0],1+suffix[1],(batch,*suffix[2]))
                    if best is None or candidate<best:best=candidate
            batch=(batch-1)&remaining
        return best
    result=solve(0)
    return None if result is None else dict(
        train_total_W_vector_reads=result[0],batch_count=result[1],batch_masks=list(result[2]),
        batches=[members(b) for b in result[2]],visited_prefix_sets=solve.cache_info().currsize)


def replay(rows,capacity,batches):
    """Separate legality/last-use replay. Completed outputs keep original t."""
    done=0;outputs=set();Y=set();peak=0;transient_stores=0;trace=[]
    for batch in batches:
        assert batch and not batch&done
        before=sorted(Y);new=members(batch)
        # B includes every newly produced conv partial, even a time column that
        # will be consumed immediately at the end of this batch.
        during=Y|set(new);peak=max(peak,len(during))
        assert len(during)<=capacity
        after=done|batch
        ready=[r for r,deps in enumerate(rows) if r not in outputs and not deps&~after]
        assert all(set(members(rows[r]))<=during for r in ready)
        outputs.update(ready)
        keep={c for c in during if any(r not in outputs and deps>>c&1 for r,deps in enumerate(rows))}
        assert keep==set(members(retained(rows,after)))
        trace.append(dict(new_time_columns=new,retained_before=before,
            live_conv_partial_and_Y_columns=sorted(during),peak_slots=len(during),
            completed_PSN_rows=ready,released_columns=sorted(during-keep),retained_after=sorted(keep)))
        transient_stores+=len(new);done=after;Y=keep
    assert done==FULL and len(outputs)==10 and not Y and transient_stores==10
    return dict(peak_Y_slots_including_all_new_conv_partials=peak,working_U_slots_additional=1,
                total_time_columns_produced=10,PSN_outputs_completed=10,trace=trace)


def load_variants(directory):
    fit=json.loads((directory/'fit.json').read_text())
    variants=dict(fit['variants'])
    reordered=json.loads((directory/'reordered_masked_k4.json').read_text())
    variants[reordered['name']]=reordered
    variants.update(json.loads((directory/'reassigned_groups.json').read_text())['variants'])
    return variants


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=HERE/'dependency')
    args=parser.parse_args();directory=args.directory
    raw=json.loads((directory/'source_batch_histogram.json').read_text())
    frames=raw['frames'];train=[f for f in frames if f['split']=='train'];valid=[f for f in frames if f['split']=='valid']
    assert len(train)==32 and len(valid)==10
    assert len({f['file'] for f in train})==32 and len({f['file'] for f in valid})==10
    assert not {f['file'] for f in train}&{f['file'] for f in valid}
    for frame in frames:
        assert frame['cell_count']==CELLS==sum(frame['hist'])
        assert frame['shape']==[10,1,96,240,320]
    train_hist=[sum(f['hist'][i] for f in train) for i in range(1024)]
    if 'train_hist' in raw:assert train_hist==raw['train_hist']
    cost=batch_costs(train_hist)
    # A short independent direct check verifies the bin bit order and zeta
    # transform. It does not fit or alter the schedule from validation data.
    for b in [1,2,3,7,32,63,341,682,1023]:
        assert cost[b]==sum(v for pat,v in enumerate(train_hist) if pat&b)
    variants=load_variants(directory);plans={}
    for name,v in variants.items():
        rows=graph(v['weight'])
        assert sum(r.bit_count() for r in rows)==v['actual_connections']
        plans[name]={}
        for capacity in (5,10):
            selected=choose_schedule(rows,capacity,cost)
            if selected is None:
                plans[name][str(capacity)]=dict(status='UNREACHABLE_IN_RETAIN_Y_FAMILY',Y_capacity=capacity,
                    reason='No complete10-column schedule satisfies retained_Y(done)+new_batch<=capacity; not an execution failure.')
                continue
            witness=replay(rows,capacity,selected['batch_masks'])
            if capacity==10:
                assert selected['batch_masks']==[FULL] and selected['train_total_W_vector_reads']==cost[FULL]
            plans[name][str(capacity)]=dict(status='REACHABLE',Y_capacity=capacity,**selected,**witness)
    # Every schedule is now frozen from train32. Validation supplies fees only.
    valid_costs=[(f,batch_costs(f['hist'])) for f in valid]
    for name,by_cap in plans.items():
        for result in by_cap.values():
            if result['status']!='REACHABLE':continue
            masks=result['batch_masks'];payments=[]
            for frame,vc in valid_costs:
                payments.append(dict(file=frame['file'],W_vector_reads=sum(vc[b] for b in masks),
                    per_batch_W_vector_reads=[vc[b] for b in masks],
                    singleton_time_W_vector_reads=sum(vc[1<<c] for c in range(10)),
                    complete_T10_batch_W_vector_reads=vc[FULL]))
            total=sum(x['W_vector_reads'] for x in payments)
            singleton=sum(x['singleton_time_W_vector_reads'] for x in payments)
            full=sum(x['complete_T10_batch_W_vector_reads'] for x in payments)
            result.update(train_mean_W_vector_reads=result['train_total_W_vector_reads']/len(train),
                valid_mean_W_vector_reads=total/len(valid),valid_total_W_vector_reads=total,
                valid_vs_singleton_read_reduction=1-total/singleton,
                valid_excess_over_complete_T10_batch_reads=total/full-1,
                valid_frames=payments)
    # Conv1 produces96 channels over240x320. Counts remain logical scalar
    # consumer operations; multiplier format and actual issue rate are unset.
    coordinates=96*240*320
    operations={name:dict(PSN_nonzero_coefficient_uses_per_coordinate=v['actual_connections'],
        PSN_nonzero_coefficient_uses_per_frame=v['actual_connections']*coordinates,
        PSN_bias_applications_per_frame=10*coordinates,threshold_decisions_per_frame=10*coordinates,
        theta='Complete10 theta*g outputs retained; theta is independent of the decision threshold.')
        for name,v in variants.items()}
    row34_fee=plans['row34']['5']['valid_mean_W_vector_reads']
    capacity5_comparison={name:dict(
        valid_mean_logical_W_vector_uses=v['5']['valid_mean_W_vector_reads'],
        fractional_reduction_from_row34=1-v['5']['valid_mean_W_vector_reads']/row34_fee,
        fixed_batch_sizes=list(map(len,v['5']['batches'])))
        for name,v in plans.items() if v['5']['status']=='REACHABLE'}
    report=dict(kind='TRAIN_SELECTED_FINITE_Y_CONV_BATCH_AND_PSN_DEPENDENCY_MODEL',
        input='source_batch_histogram.json: actual common fixed4BN parent r1.conv1 inputs',
        train_frames=[f['file'] for f in train],valid_frames=[f['file'] for f in valid],
        histogram=dict(bin_bits='pattern bit=t',cells_per_frame=CELLS,
            cell='one input-channel/kernel-offset/output-P4-position group; OR over four actual source positions including padding',
            vector='one logical W-vector use in a fixed P4 context and one common output-channel group, when this cell is active in any time of B',
            output_group_scaling='Shown counts use one common group. Covering outputC96 with H8 groups multiplies every axis by12; H96 uses one group. This is not an observed SRAM transaction.',
            source_amplitudes=[{k:f[k] for k in ('file','split','theta','nonzero_amplitude_min','nonzero_amplitude_max') if k in f} for f in frames]),
        model=dict(Y_capacities=[5,10],Y_slot='one temporal conv-result plane for P4 x one common output-channel group, width unspecified',
            working_U_slots_additional=1,
            production='All B new columns occupy simultaneous Conv partial-sum slots until the whole batch finishes.',
            consumption='Immediately compute every now-complete PSN output with one working U, then release each Y at last use. Completed gate bits may be kept with original time labels.',
            legality='popcount(retainedY(done))+popcount(B)<=Ycap; nonempty B disjoint from done',
            objective='minimize train32 total logical W vector reads; ties choose fewer batches then lexicographic masks',
            recurrence='F(done)=min_B[cost_train(B)+F(done union B)] over legal B; cost(B)=sum(hist[pat] if pat&B !=0)',
            exactness='All input subset states and admissible next batches; fixed schedule per variant/capacity for all spatial cells and validation frames.',
            histogram_transform='Subset-zeta: cost(B)=total-zeta[full XOR B]. No validation-dependent schedule choices.'),
        common_baselines=dict(train_mean_singleton_W_vector_reads=sum(cost[1<<c] for c in range(10))/len(train),
            train_mean_complete_T10_batch_W_vector_reads=cost[FULL]/len(train),
            valid_mean_singleton_W_vector_reads=sum(sum(vc[1<<c] for c in range(10)) for _,vc in valid_costs)/len(valid),
            valid_mean_complete_T10_batch_W_vector_reads=sum(vc[FULL] for _,vc in valid_costs)/len(valid),
            complete_T10_note='Ycap10 admits all ten columns in one batch for every structure, including dense100. This is a legal strong control, not an unattainable lower bound.'),
        capacity5_comparison_to_row34=capacity5_comparison,
        interpretation=[
            'A smaller dependency frontier can admit larger convolution batches, but batch count is not proportional to actual source-supported W use.',
            'The read-optimal row34 schedule chooses6 batches although a5-batch legal schedule exists. The objective is train W use, not fewest batches.',
            'Ordinary continuous334 has only about2.03% fewer validation W uses than optimized row34 at commonYcap5; fit334 has about1.27% fewer. These are model-use counts, not SRAM savings or speedup.',
            'Closure334 has about9.34% fewer validation uses than row34, but its algorithm accuracy and ordinary stronger reuse/latent baselines remain separate requirements.',
            'All structures have identical one-batch logical W use atYcap10. Native dense100 is unreachable only in theYcap5 retain-Y family, not an invalid algorithm.',
            'Original source amplitudes in this captured set are theta=1; the support-based cost does not replace the actual output theta*g semantics with a unit-amplitude assumption.'],
        PSN_consumer_operations=operations,variants=plans,
        limits=['Only the retain-Y family is optimized; push-U, latent-factor and intra-conv/PSN fusion routes are outside.',
            'Logical W-vector uses are neither SRAM transactions, external DMA nor cycles. Multi-PE same-address merging, broadcast across P4 groups and hot-W register caching are unmodeled ordinary stronger controls.',
            'Ycap charges every new Conv partial and retained result, but not producer line buffers, source reload/reorder traffic, coefficient caches, ports or bit widths.',
            'Each time column is produced exactly once; dense biases, all PSN coefficient edges and all ten threshold/theta*g decisions remain.',
            'PSN multiply/add time, Conv2 and halo traffic are not modeled. A read-use reduction alone does not imply a complete-layer advantage.',
            'No numerical equivalence, latency, energy, SRAM macro or PPA claim; no new RTL or network forward is performed by this script.',
            'Rank2/3 continuous latent controls and other mixed schedules are outside this retain-Y family; its optimum is not an optimum over all possible state strategies.',
            'Scheduling and ordinary time batching are strong compiler controls, not a new hardware mechanism.'])
    (directory/'batch_schedule.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for name,by_cap in plans.items():
        for capacity,r in by_cap.items():
            if r['status']=='REACHABLE':
                print(name,capacity,r['batches'],'train',round(r['train_mean_W_vector_reads'],2),
                      'valid',round(r['valid_mean_W_vector_reads'],2))
            else:print(name,capacity,r['status'])


if __name__=='__main__':main()
