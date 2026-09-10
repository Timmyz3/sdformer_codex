"""Exact dependency-state schedules for the saved 10x10 temporal consumers.

This counts graph value slots and coefficient-edge uses, not hardware cycles,
SRAM bytes or a numerical equivalence result. No producer/capture is rerun.
"""
from functools import lru_cache
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
N=10
FULL=(1<<N)-1


def members(mask):
    return [i for i in range(N) if mask>>i&1]


def graph(weight):
    assert len(weight)==N and all(len(r)==N for r in weight)
    return [sum(1<<c for c,v in enumerate(row) if v!=0) for row in weight]


def active(rows,prefix):
    """Started U whose final dependency has not yet arrived."""
    return sum(1<<r for r,deps in enumerate(rows) if deps&prefix and deps&~prefix)


def retained(rows,prefix):
    """Arrived Y columns used by at least one not-yet-ready output."""
    result=0
    for deps in rows:
        if deps&~prefix:result|=deps&prefix
    return result


def transition_cost(rows,prefix,column,strategy):
    bit=1<<column;after=prefix|bit
    if strategy=='retain_Y':
        # The incoming value occupies an input register before ready consumers
        # can finish. It is not allowed to disappear from the peak count.
        return retained(rows,prefix).bit_count()+1
    before_U=active(rows,prefix);after_U=active(rows,after)
    # Finish existing U first, then singleton outputs, then start new U.
    # An output being completed still occupies its U slot in that operation.
    singleton=any(deps==bit for deps in rows)
    return max(before_U.bit_count(),after_U.bit_count(),
               (before_U&after_U).bit_count()+int(singleton))


def optimal(rows,strategy,max_push_U=None):
    # There are only 1024 prefix sets. All unfinished state is determined by
    # the set, so this recurrence covers every one of the10! column orders.
    @lru_cache(None)
    def suffix(prefix):
        if prefix==FULL:return 0
        return min((max(transition_cost(rows,prefix,c,strategy),suffix(prefix|1<<c))
                    for c in range(N) if not prefix>>c&1
                    and (max_push_U is None or transition_cost(rows,prefix,c,'push_U')<=max_push_U)),
                   default=N+1)
    cap=suffix(0);order=[];prefix=0
    assert cap<=N
    while prefix!=FULL:
        for c in range(N):
            if (not prefix>>c&1 and transition_cost(rows,prefix,c,strategy)<=cap and suffix(prefix|1<<c)<=cap
                and (max_push_U is None or transition_cost(rows,prefix,c,'push_U')<=max_push_U)):
                order.append(c);prefix|=1<<c;break
    return dict(minimum_peak=cap,order=order,DP_states=suffix.cache_info().currsize)


def simulate(rows,order,include_trace=True):
    """Construct legal serial row execution, independently check slot peaks."""
    assert sorted(order)==list(range(N)) and all(rows)
    prefix=0;u_live=set();y_live=set();done=set()
    u_peak=0;u_between=0;y_peak=0;y_between=0;whole_column_U_peak=0
    edge_updates=0;cached_reads=0;arrival_uses=0;y_stores=0;trace=[]
    for column in order:
        bit=1<<column;after=prefix|bit
        affected=[r for r,deps in enumerate(rows) if deps&bit]
        finish=[r for r in affected if not rows[r]&~after]
        finish_existing=[r for r in finish if r in u_live]
        singleton=[r for r in finish if r not in u_live]
        continuing=[r for r in affected if r in u_live and r not in finish]
        new=[r for r in affected if r not in u_live and r not in finish]
        update_order=finish_existing+singleton+continuing+new
        u_before=sorted(u_live);step_peak=len(u_live)
        whole_column_U_peak=max(whole_column_U_peak,len(u_live|set(affected)))
        for r in update_order:
            u_live.add(r);step_peak=max(step_peak,len(u_live));edge_updates+=1
            if r in finish:u_live.remove(r)
        assert step_peak==transition_cost(rows,prefix,column,'push_U')
        assert u_live==set(members(active(rows,after)))
        u_peak=max(u_peak,step_peak);u_between=max(u_between,len(u_live))

        # Pull strategy: hold Y, compute each newly-ready row once, release
        # each input immediately after its last unfinished consumer completes.
        y_before=sorted(y_live);y_live.add(column);y_at_arrival=sorted(y_live)
        y_peak=max(y_peak,len(y_live));release=[]
        for r in finish:
            assert r not in done and set(members(rows[r]))<=y_live
            cached_reads+=rows[r].bit_count()-1;arrival_uses+=1;done.add(r)
            unused=[c for c in y_live if not any(s not in done and deps>>c&1
                                              for s,deps in enumerate(rows))]
            for c in unused:y_live.remove(c)
            release.append(dict(after_output=r,columns=sorted(unused)))
        assert y_live==set(members(retained(rows,after)))
        y_between=max(y_between,len(y_live));y_stores+=column in y_live
        if include_trace:
            trace.append(dict(input_column=column,completed_outputs=finish,
                push_U=dict(live_before=u_before,update_order=update_order,
                            operation_peak=step_peak,live_after=sorted(u_live)),
                retain_Y=dict(retained_before=y_before,including_new_input=y_at_arrival,
                              release_after_output=release,retained_after=sorted(y_live))))
        prefix=after
    edges=sum(x.bit_count() for x in rows)
    assert edge_updates==edges and cached_reads+arrival_uses==edges
    assert len(done)==N and not u_live and not y_live
    result=dict(order=order,
        push_U=dict(peak_U_values_including_completion_operation=u_peak,
            peak_persistent_U_between_input_columns=u_between,new_input_register_values=1,
            peak_U_if_release_only_at_whole_column_end=whole_column_U_peak,
            coefficient_edge_updates=edge_updates,
            persistent_partial_U_read_uses=edges-N,persistent_partial_U_write_uses=edges-N,
            source_column_arrivals=N),
        retain_Y=dict(peak_Y_values_including_new_input=y_peak,
            peak_retained_Y_between_columns=y_between,
            Y_storage_slots_excluding_reusable_input_register=y_between,
            new_input_register_values=1,working_U_values_including_completion=1,
            coefficient_edge_updates=edges,cached_Y_operand_uses=cached_reads,
            current_input_operand_uses=arrival_uses,retained_Y_writes=y_stores,
            source_column_arrivals=N,completed_U_outputs=N))
    if include_trace:result['trace']=trace
    return result


def main():
    fitted=json.loads((HERE/'dependency/fit.json').read_text())
    graphs={name:graph(v['weight']) for name,v in fitted['variants'].items()}
    choices={name:{s:optimal(rows,s) for s in ('push_U','retain_Y')} for name,rows in graphs.items()}
    result=dict(kind='EXACT_T10_DEPENDENCY_GRAPH_STATE_ONLY',
        input='dependency/fit.json actual nonzero FP32 coefficients; native=dense100',
        scope='One scalar position/channel temporal transform U_t=sum_c E_tc*Y_c+b_t; all10 outputs requested, no demand D, no coefficient approximation.',
        common_rules=[
            'Every Y column is produced once in the listed permutation; do not store/recompute an undisclosed second copy.',
            'After each input column, execute all enabled work before admitting the next. Completed outputs can be consumed immediately in any time-index order.',
            'Push_U finishes existing rows before allocating newly-started rows. The U being completed counts as live during its operation; input Y has a separate register.',
            'Retain_Y computes a row only when all dependencies are available, using one scalar U; last-use Y is released immediately. The arriving Y counts in the total Y peak.',
            'A newly arrived Y kept for future consumers can transfer to a free persistent slot after current completions; the transfer does not require a second live copy.',
            'Completed theta*g can be stored as gate bits plus independent static theta. Wide U is not retained until all T10 bits arrive; tau is not identified with theta.'],
        counted_units='Graph value slots per scalar coordinate and coefficient-edge/operand uses. A slot may be a whole spatial/channel vector in an implementation; Y and U need not have the same width.',
        exact_method='Subset minimax DP over all1024 input-prefix sets, equivalent to searching10! permutations under the stated completion rules. Each selected order is replayed with explicit row updates and last-use releases.',
        excluded=['Numeric bit width, FP32 summation-order differences, coefficient multiplies/bias/threshold arithmetic implementation',
            'Conv partial sums, arbitrary-column production cost or source rereads needed to reorder the actual convolution',
            'SRAM/register mapping, physical read/write ports, memory copies, accumulator pipeline temporary registers',
            'Output backpressure or an imposed in-order gate consumer; up to10 completed gate bits may require storage',
            'Cycles, PPA, complete network accuracy, and demand-masked dynamic dependency changes'],
        variants={})
    for name,rows in graphs.items():
        selected=choices[name]
        u=simulate(rows,selected['push_U']['order'],False)
        y=simulate(rows,selected['retain_Y']['order'],False)
        assert u['push_U']['peak_U_values_including_completion_operation']==selected['push_U']['minimum_peak']
        assert y['retain_Y']['peak_Y_values_including_new_input']==selected['retain_Y']['minimum_peak']
        joint=optimal(rows,'retain_Y',selected['push_U']['minimum_peak'])
        # On these actual masks there is an order attaining both separate
        # minima. Test it explicitly, rather than mixing incompatible orders.
        assert joint['minimum_peak']==selected['retain_Y']['minimum_peak']
        same=simulate(rows,joint['order'],False)
        result['variants'][name]=dict(nonzero_connections=sum(r.bit_count() for r in rows),
            row_dependencies=[members(r) for r in rows],
            natural_order=simulate(rows,list(range(N)),False),
            best_push_U=dict(**selected['push_U'],Y_peak_under_this_same_order=u['retain_Y']['peak_Y_values_including_new_input']),
            best_retain_Y=dict(**selected['retain_Y'],U_peak_under_this_same_order=y['push_U']['peak_U_values_including_completion_operation']),
            common_order_attaining_both_minima=same)
        if name=='row34':
            detail=simulate(rows,joint['order'])
            result['row34_common_order_witness']=[dict(
                column=t['input_column'],finished=t['completed_outputs'],
                U_update_order=t['push_U']['update_order'],U_peak=t['push_U']['operation_peak'],
                U_after=t['push_U']['live_after'],
                Y_with_new_input=t['retain_Y']['including_new_input'],Y_after=t['retain_Y']['retained_after'])
                for t in detail['trace']]
    common_orders={'natural':list(range(N)),
                   'row34_U_optimal':choices['row34']['push_U']['order'],
                   'row34_Y_optimal':choices['row34']['retain_Y']['order']}
    result['common_order_cross_variant_controls']={}
    for label,order in common_orders.items():
        comparison={}
        for name,rows in graphs.items():
            run=simulate(rows,order,False)
            comparison[name]=dict(U=run['push_U']['peak_U_values_including_completion_operation'],
                                  Y_including_arrival=run['retain_Y']['peak_Y_values_including_new_input'])
        result['common_order_cross_variant_controls'][label]=dict(order=order,peaks=comparison)
    result['interpretation']=[
        'row34 is not a ten-Y baseline: its optimum is four retained Y plus the current input, with one separate working U. A common order also attains its minimum seven push-U values.',
        'Contiguous334 retains a genuine graph-slot difference from optimized row34: four versus five Y including arrival, or four versus seven push-U values.',
        'This state footprint is not exclusive to block grouping: ordinary causal masked_k4/sliding_k4 also need four Y including arrival and only three push-U values when rows can retire individually.',
        'Learned fit334/closure334 reach the same4/4 optimum as ordinary contiguous groups after column reordering. Their natural-order7/5 is not a fair best-schedule baseline.',
        'All34-edge variants use34 coefficient edges,24 persistent partial/read uses, and10 source column arrivals under the respective rules; ordering changes lifetime, not the dense convolution production cost.',
        'A four-tap causal push schedule requires four U slots if retirement waits until the end of the entire column update; three assumes legal serial finish-before-start, with the finishing U still counted.']
    (HERE/'dependency/state_schedule.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,data in result['variants'].items():
        print(name,'edges',data['nonzero_connections'],
              'best_U',choices[name]['push_U']['minimum_peak'],choices[name]['push_U']['order'],
              'best_Y_incl_input',choices[name]['retain_Y']['minimum_peak'],choices[name]['retain_Y']['order'],
              'natural',data['natural_order']['push_U']['peak_U_values_including_completion_operation'],
              data['natural_order']['retain_Y']['peak_Y_values_including_new_input'])


if __name__=='__main__':main()
