"""Instrument the existing finite consumer address schedule, not a new RTL model.

Read lifetime != output completion. Gate-first is a generic control available
to BOTH students. The optional blocked DMA admission is a synthetic stress
condition, not a captured network sink or a native BN implementation.
"""
from pathlib import Path
from collections import Counter
import csv
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
sys.path.insert(0, str(STAGE))
import consumer_service as c


class ObservedEngine(c.Engine):
    def __init__(self, scenario):
        super().__init__()
        self.scenario = scenario
        self.issue_coverage = bytearray()
        self.bp = Counter()
        self.contexts = []
        self.current = None

    def issue(self, resource, count=1, latency=1, ready=None, label=None):
        start = max(self.now if ready is None else ready, self.free[resource])
        end = super().issue(resource, count, latency, ready, label)
        if len(self.issue_coverage) < end:
            self.issue_coverage.extend(b'\x00'*(end-len(self.issue_coverage)))
        self.issue_coverage[start:start+count] = b'\x01'*count
        return end

    def begin_context(self, positions):
        self.current = dict(id=len(self.contexts), positions=positions,
            anchor=all(y%2 == 0 and x%2 == 0 for y,x in positions),
            start=self.now, gate_reads={}, ped_reads={}, writes={},
            gate_done=[None, None], ped_done=[None, None], continuous_calls=0,
            gate_egress_done=[None, None])
        self.contexts.append(self.current)

    def read(self, address, size, label='state_read64', cache=False):
        end = self.now
        for word in self.words(address, size):
            if cache and self.s_cache == word:
                continue
            end = self.issue('SR', label=label)
            self.s_cache = word if cache else None
            if self.current and c.I_BASE//8 <= word < (c.I_BASE+5760)//8:
                role = {'PED_U32': 'ped_reads', 'proj_gate_cutoffs': 'gate_reads'}.get(self.stage)
                if role:
                    self.current[role][word-c.I_BASE//8] = end
        return end

    def write(self, address, size, label='state_write64'):
        super().write(address, size, label)
        if not self.current:
            return
        if c.I_BASE <= address < c.I_BASE+5760:
            for word in self.words(address, size):
                self.current['writes'][word-c.I_BASE//8] = self.now
        if label == 'proj_gate_word_write64':
            position = (address-c.GATE_BASE)//(c.C*2)
            p = self.current['positions'].index(divmod(position, c.SIDE))
            self.current['gate_done'][p] = self.now

    def dma(self, byte_count, label):
        if self.scenario == 'blocked' and label in ('continuous_output', 'proj_gate_output'):
            phase = self.now % 1024
            delay = max(0, 896-phase)
            tag = 'PED_BP' if label == 'continuous_output' else 'spike_BP'
            # Admission can stall, then a complete five-slot bus transaction
            # runs without mid-transaction stalls. One resident output H8 only.
            self.bp[tag] += delay
            self.wait(self.now+delay)
        super().dma(byte_count, label)
        if label == 'continuous_output':
            p = self.current['continuous_calls']//120
            self.current['ped_done'][p] = self.now
            self.current['continuous_calls'] += 1


def measure(data, q, gates, order, scenario):
    e = ObservedEngine(scenario)
    weights, e.bases, pool_bytes = c.matrices(q)
    e.stage = 'coefficient_cold_fill'
    for offset in range(0, pool_bytes, 32):
        e.dma(32, 'coefficient_fill')
        e.wait(e.issue('CW', label='coefficient_local_write256'))
    c.input_supply(e, data)
    anchors = [[(y,x), (y,x+2)] for y in range(0,8,2) for x in (0,4)]
    rest = [(y,x) for y in range(8) for x in range(8) if y%2 or x%2]
    for positions in anchors+[rest[i:i+2] for i in range(0,len(rest),2)]:
        e.begin_context(positions)
        e.stage = 'raw_I24_external_input'
        for offset in range(0,5760,32):
            e.dma(32,'I24_input')
            e.write(c.I_BASE+offset,32,'I24_local_write64')
        if e.current['anchor']:
            active = c.sparse_u(e,gates,positions,weights['U_conv2_theta'],e.bases['U_conv2_theta'],'K_major')
            e.stage = 'F_and_BN2_I24_merge'
            for p in range(2):
                for t in range(10):
                    for h in range(0,96,8):
                        if active[p,t]:
                            c.dense_output(e,weights['F'],e.bases['F'],c.WORK_BASE,6,p,t,h,'F')
                        else:
                            e.alu('F_source_empty_zero8')
                        e.weight(e.bases['BN2_constant']+h*4)
                        e.alu('BN_constant_add8')
                        address = c.I_BASE+((p*10+t)*96+h)*3
                        e.alu('I_branch_add8',ready=e.read(address,24))
                        e.alu('anchor_sat24')
                        e.write(address,24,'updated_I24_write64')
            if order == 'gate_first':
                c.gates_out(e,positions,q)
            e.stage = 'PED_U32'
            for p in range(2):
                for t in range(10):
                    for h in range(0,32,8):
                        c.dense_output(e,weights['U_ped'],e.bases['U_ped'],c.I_BASE,3,p,t,h,'PED_U')
                        e.write(c.WORK_BASE+((p*10+t)*32+h)*3,24,'PED_U24_write64')
            e.stage = 'PED_V32_bias_continuous_output'
            for p in range(2):
                for t in range(10):
                    for h in range(0,96,8):
                        c.dense_output(e,weights['V_ped'],e.bases['V_ped'],c.WORK_BASE,3,p,t,h,'PED_V')
                        e.weight(e.bases['PED_bias']+h*4)
                        e.alu('PED_bias_add8')
                        e.alu('PED_bias_sat24')
                        e.alu('exact24f14_to_FP32_8')
                        e.dma(32,'continuous_output')
        if not e.current['anchor'] or order == 'ped_first':
            c.gates_out(e,positions,q)
        e.current['end'] = e.now
    e.stage = 'proj_gate_cache_egress'
    for offset in range(0,12288,32):
        e.wait(e.read(c.GATE_BASE+offset,32,'proj_gate_output_read64'))
        e.dma(32,'proj_gate_output')
        position, inside = divmod(offset+32,192)
        if inside == 0:
            for ctx in e.contexts:
                for p, (y,x) in enumerate(ctx['positions']):
                    if y*8+x == position-1:
                        ctx['gate_egress_done'][p] = e.now
    progress = sum(e.issue_coverage)
    classes = dict(progress=progress, fifo=0, spike_BP=e.bp['spike_BP'], PED_BP=e.bp['PED_BP'], BN=0,
                   other=e.now-progress-sum(e.bp.values()))
    assert classes['other'] >= 0 and sum(classes.values()) == e.now
    return e, dict(service_slots=e.now, exclusive_issue_slots=classes, stage_slots=dict(e.stage_steps),
        requests=dict(e.count), resource_issue_slots_not_additive=dict(e.busy))


def lifetime(e, path):
    rows = []
    positions = []
    release_rules_equal = True
    for ctx in e.contexts:
        for p,(y,x) in enumerate(ctx['positions']):
            words = range(p*360,(p+1)*360)
            last_gate = max((ctx['gate_reads'].get(w,0) for w in words), default=0)
            last_ped = max((ctx['ped_reads'].get(w,0) for w in words), default=0)
            physical_free = max(max(last_gate,last_ped), max(ctx['writes'][w] for w in words))
            # GeMM consumes a separately allocated gate buffer. V reads U24.
            # Neither is a future physical reader of these I24 words.
            branch_done = max(ctx['gate_done'][p],ctx['ped_done'][p] or 0)
            positions.append(dict(context=ctx['id'],p=p,y=y,x=x,anchor=ctx['anchor'],
                gate_last_I_read=last_gate,PED_last_I_read=last_ped or None,
                I_position_free=physical_free,gate_materialized=ctx['gate_done'][p],
                PED_accepted=ctx['ped_done'][p],gate_egress_accepted=ctx['gate_egress_done'][p],
                branch_done_minus_I_free=branch_done-physical_free,
                gate_vs_PED_read_lag=last_gate-last_ped if ctx['anchor'] else None))
            for w in words:
                g,r = ctx['gate_reads'].get(w,0),ctx['ped_reads'].get(w,0)
                free = max(g,r,ctx['writes'][w])
                untyped_read_ends = [reads[w] for reads in (ctx['gate_reads'],ctx['ped_reads']) if w in reads]
                untyped_free = max([ctx['writes'][w]]+untyped_read_ends)
                release_rules_equal &= free == untyped_free
                rows.append(dict(context=ctx['id'],p=p,word=w,anchor=ctx['anchor'],
                    last_gate_read=g,last_PED_read=r,last_write=ctx['writes'][w],
                    physical_free=free,local_outputs_done=branch_done,
                    output_policy_extra_hold=branch_done-free))
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    def describe(values):
        a=np.array(values,dtype=np.int64)
        return dict(n=len(a),min=int(a.min()),median=float(np.median(a)),max=int(a.max()),mean=float(a.mean()))
    return dict(positions=positions,
        anchor_read_lag_histogram=dict(Counter(str(p['gate_vs_PED_read_lag']) for p in positions if p['anchor'])),
        anchor_output_extra_hold=describe([p['branch_done_minus_I_free'] for p in positions if p['anchor']]),
        anchor_word_output_extra_hold=describe([r['output_policy_extra_hold'] for r in rows if r['anchor']]),
        actual_read_set_release_equals_untyped_refcount=bool(release_rules_equal),
        typed_label_alone_changes_this_schedule=False,
        separate_dynamic_refcount_controller_simulated=False,
        capacity_stall_measured=False)


def main():
    old=json.loads((STAGE/'consumer_service_result.json').read_text())
    result=dict(scope='Existing integer consumer address model: full K864, corner 8x8 gates, 4x4 continuous outputs; excludes source DAG/preview/native projection/full-domain BN.',
        resources='Unchanged consumer_service_result.json resource point: 32KiB state, P2 serial, 8 lanes, 64bit1R1W, 32B/5slot DMA.',
        blocked='Synthetic admission only: output request waits while slot%1024<896, then five slots run. No trace-derived sink and no FIFO.',
        classification='Union of issued resource slots is progress; no issue/blocked admission is other/dependency latency. Active DMA is progress, not backpressure. Final BN absent; fused BN2 constants are ordinary ALU progress.',
        payload_limit='Address schedule with existing independent numerical oracle, not byte-addressed dual-consumer execution or VCS.',
        arms={})
    for axis in ('ordinary','lifting_raw'):
        data=c.read_npz(STAGE/'consumer_fixtures'/f'{axis}.npz')
        gates=c.unpack(data,'sn2')[:,:,:8,:8]
        oracle=c.numerical_reference(data,data,gates,data)
        assert oracle['gate_differences']==oracle['continuous_native_check']['differences']==0
        for order in ('ped_first','gate_first'):
            for scenario in ('ready','blocked'):
                key=f'{axis}_{order}_{scenario}'
                e,run=measure(data,data,gates,order,scenario)
                if order=='ped_first' and scenario=='ready':
                    assert run['service_slots']==old['axes'][axis]['services']['K_major']['service_slots']
                    assert run['requests']==old['axes'][axis]['services']['K_major']['operations_and_word_requests']
                for name,value in oracle['expected_nonzero_products'].items():
                    assert run['requests'][name]==value
                run['lifetime']=lifetime(e,HERE/f'{key}_I_lifetime.csv')
                run['numerical_oracle']=dict(gate_differences=0,continuous_differences=0,
                    note='Shared formula oracle; gate-first never changes I after gate computation, and gate/U24/I occupy separate declared ranges.')
                result['arms'][key]=run
                print(key,run['service_slots'],run['exclusive_issue_slots'],flush=True)
    (HERE/'consumer_result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
