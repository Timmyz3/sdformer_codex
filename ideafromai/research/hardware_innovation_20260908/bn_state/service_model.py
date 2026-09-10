"""Finite queues, bank service and dependency scheduling, not measured RTL cycles.

The FC1 kernel issues actual captured support positions into lane banks. A
scoreboard enforces the four-beat accumulation dependency. All algorithms share
the same primitive service units; ordinary blocking/caching is enabled in all.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from collections import OrderedDict, defaultdict
import heapq
import json
import math
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
R = json.loads((ROOT/'resources.json').read_text())
ceil = lambda n, d: (int(n)+int(d)-1)//int(d)


class Graph:
    def __init__(self):
        self.nodes = []

    def add(self, name, resource, duration, deps=(), nbytes=0):
        deps = tuple(set(x for x in deps if x is not None))
        i = len(self.nodes)
        self.nodes.append((name, resource, max(0, int(duration)), deps, int(nbytes)))
        return i

    def run(self):
        n = len(self.nodes)
        children = [[] for _ in range(n)]
        pending = [len(x[3]) for x in self.nodes]
        ready = [0]*n
        finish = [0]*n
        queue = []
        for i, node in enumerate(self.nodes):
            for d in node[3]: children[d].append(i)
            if not pending[i]: heapq.heappush(queue, (0, i))
        free, busy, waiting, bytecounts = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
        trace = []
        while queue:
            at, i = heapq.heappop(queue)
            name, resource, duration, _, nbytes = self.nodes[i]
            start = max(at, free[resource]) if resource else at
            end = start+duration
            if resource:
                free[resource] = end
                busy[resource] += duration
                waiting[resource] += start-at
            bytecounts[name] += nbytes
            finish[i] = end
            if len(trace) < 90 and duration:
                trace.append(dict(name=name, resource=resource, ready=at, start=start, end=end))
            for c in children[i]:
                pending[c] -= 1
                ready[c] = max(ready[c], end)
                if not pending[c]: heapq.heappush(queue, (ready[c], c))
        return dict(beats=max(finish, default=0), busy=dict(busy), queued_wait=dict(waiting),
                    bytes=dict(bytecounts), finish=finish, timeline=trace)


def dram_duration(chunks):
    return sum(R['external_transaction_latency']+ceil(min(left, R['external_max_burst_bytes']), R['external_bytes_per_beat'])
               for size in chunks for left in range(int(size), 0, -R['external_max_burst_bytes']))


def kernel_trace(S, P, C, q, sizes):
    """Exact issue schedule of c-ordered sparse updates for each finite source tile.

For q output channels, L=96/q position banks issue at most one position each.
The same accumulator cannot be revisited before latency=4. Two coefficient/
bitmap staging entries let the next c prefetch overlap the current c's issues.
"""
    L = 96//q
    result = []
    for b, n in enumerate(sizes):
        x = S.reshape(10, P, C)[:, b*32:b*32+int(n)].reshape(10*int(n), C)
        parts = []
        for c0 in range(0, C, 256):
            c1 = min(c0+256, C)
            acc_ready = np.zeros(len(x), dtype=np.int64)
            fetch_end = compute_end = 0
            old_starts = [0, 0]
            issues = adds = source_reads = 0
            nonzero_rows = np.zeros(len(x), bool)
            for c in range(c0, c1):
                positions = np.flatnonzero(x[:, c])
                if not len(positions): continue
                # Consecutive sectors use bank=(c+sector)%8; q1/q4 one word.
                prep = max(ceil(max(1, ceil(q, 4)), 8), ceil(ceil(len(x), 128), 8))
                fetch_end = max(fetch_end, old_starts[0])+prep
                start = max(compute_end, fetch_end)
                end = start
                for bank in np.unique(positions % L):
                    t = start
                    for p in positions[positions % L == bank]:
                        issue = max(t, int(acc_ready[p]))
                        acc_ready[p] = issue+R['fma_latency']
                        t = issue+1
                    end = max(end, t)
                old_starts = [old_starts[1], start]
                compute_end = end
                issues += len(positions)*q
                adds += int(nonzero_rows[positions].sum())*q
                nonzero_rows[positions] = True
                source_reads += ceil(len(x), 128)*16
            duration = max(compute_end, int(acc_ready.max(initial=0)))
            parts.append(dict(c0=c0, c1=c1, beats=duration,
                              scalar_updates=issues, within_chunk_adds=adds,
                              source_scratch_bytes=source_reads))
        result.append(parts)
    return result


class Cache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.used = 0

    def request(self, key, size):
        if key in self.entries:
            entry = self.entries.pop(key)
            self.entries[key] = entry
            return entry, []
        evictions = []
        while self.used+size > self.capacity and self.entries:
            _, entry = self.entries.popitem(last=False)
            self.used -= entry['size']
            evictions.append(entry['last'])
        entry = dict(size=size, ready=None, last=None)
        self.entries[key] = entry
        self.used += size
        return entry, evictions


class Simulation:
    def __init__(self, meta, trace, kernels, axis, q, allocation='wide_first', repair_q=1, order='resident'):
        self.m, self.d, self.kernels = meta, trace, kernels
        self.axis, self.q, self.repair_q, self.order = axis, q, repair_q, order
        self.g = Graph()
        self.C, self.H, self.P, self.T = (meta[k] for k in ('C','H','P','T'))
        self.sizes = trace['sizes']
        self.nb = len(self.sizes)
        self.source_bytes = meta['N']*self.C//8
        self.weight = Cache(R['weight_cache_bytes'])
        self.source = Cache(R['source_transpose_cache_bytes'])
        self.pin_ready = {}
        self.counters = defaultdict(int)
        self.marks = dict(statistics=[], first_output=[], last_output=[])
        self.example = None
        self.repair_example = None
        # q4 canonical layout and full-domain channel blocking, never N clipping.
        self.packet_bytes = ceil(32+2*64+4*(64+5)+64, 128)*16
        self.wide_bytes = (self.nb*q*10*self.packet_bytes if axis == 'packet'
                           else meta['N']*q*8 if axis in ('save_y','save_u') else 0)
        reserved = 0
        if axis == 'gram':
            reserved = self.C*(self.C+1)//2*4 + self.C*16*8 + self.H*10*8
        usable = R['state_capacity_bytes']-reserved
        if allocation == 'source_first':
            source_budget = min(self.source_bytes, usable)
            self.wide_keep = min(self.wide_bytes, usable-source_budget)
        else:
            self.wide_keep = min(self.wide_bytes, usable)
            source_budget = min(self.source_bytes, usable-self.wide_keep)
        self.pin = set()
        used = 0
        for b,n in enumerate(self.sizes):
            amount = 10*int(n)*self.C//8
            if used+amount <= source_budget:
                self.pin.add(b); used += amount
        self.state_live_peak = reserved+used+self.wide_keep
        self.external_mutable_peak = self.source_bytes+max(0, self.wide_bytes-self.wide_keep)
        self.input = self.mem('source_retention_write', [self.source_bytes], (), external=True)
        self.barrier = self.input

    def mem(self, name, chunks, deps, external=True):
        size = sum(int(v) for v in chunks)
        return self.g.add(name, 'external' if external else 'state',
                          dram_duration(chunks) if external else ceil(size, 128), deps, size)

    def source_get(self, b, deps):
        n = int(self.sizes[b]); rawbytes = 10*n*self.C//8
        padded = self.C*ceil(10*n,128)*16
        entry, evict = self.source.request(b, padded)
        if entry['ready'] is not None:
            self.counters['source_transpose_hits'] += 1
            return entry, entry['ready']
        self.counters['source_transpose_misses'] += 1
        needed = list(deps)+evict
        if b in self.pin and b in self.pin_ready:
            read = self.mem('source_pinned_read', [rawbytes], needed+[self.pin_ready[b]], external=False)
        else:
            # Actual capture order is (t,p,c): ten separate temporal ranges.
            read = self.mem('source_backing_read', [n*self.C//8]*10, needed)
            if b in self.pin:
                pinned = self.mem('source_pin_write', [rawbytes], [read], external=False)
                self.pin_ready[b] = pinned
                read = pinned
        transpose = self.g.add('source_transpose', 'transpose',
                               10*n+ceil(padded,128), [read])
        entry['ready'] = entry['last'] = transpose
        return entry, transpose

    def weight_get(self, h0, q, c0, c1, deps):
        waits, entries = [], []
        for sector in range(h0//4, ceil(h0+q,4)):
            key = sector, c0
            entry, evict = self.weight.request(key, (c1-c0)*16)
            if entry['ready'] is None:
                load = self.mem('weight_backing_read', [(c1-c0)*16], list(deps)+evict)
                fill = self.g.add('weight_cache_fill', 'weight_fill', ceil((c1-c0)*16,128), [load])
                entry['ready'] = entry['last'] = fill
                self.counters['weight_region_misses'] += 1
            else:
                self.counters['weight_region_hits'] += 1
            waits.append(entry['ready']); entries.append(entry)
        return waits, entries

    def fc1(self, b, h0, q, deps):
        entry, source_ready = self.source_get(b, deps)
        init = self.g.add('accumulator_init', 'numeric', ceil(10*int(self.sizes[b])*q,96), list(deps)+[source_ready])
        last = init
        for part in self.kernels[q][b]:
            weights, wentries = self.weight_get(h0,q,part['c0'],part['c1'],[last])
            last = self.g.add('fc1_sparse_issue', 'numeric', part['beats'], [last,source_ready]+weights)
            self.counters['fc1_scalar_updates'] += part['scalar_updates']
            self.counters['source_scratch_read_bytes'] += part['source_scratch_bytes']
            for w in wentries:
                w['last'] = self.g.add('weight_readers_done',None,0,[w['last'],last])
        if self.m['source_theta'] != 1.0:
            last = self.g.add('source_theta_scale', 'numeric', ceil(10*int(self.sizes[b])*q,96)+4,[last])
        entry['last'] = self.g.add('source_readers_done',None,0,[entry['last'],last])
        return last

    def psn(self, b, q, deps, wide=False):
        n=int(self.sizes[b]); L=96//q; work=0
        # Strong ordinary mapping: flatten (position,h,t) outputs, so a q1
        # block also uses the temporal dimension to fill the 96 numeric lanes.
        # At most 96 (p,h) pairs retain all T inputs AND T accumulators in the
        # explicit 16KiB register budget. Only then may their Y be overwritten.
        for first in range(0,n*q,96):
            items=np.arange(first,min(first+96,n*q))
            ps,hs=items//q,items%q
            banks=((np.arange(10)[:,None]*n+ps[None,:])%L)*q+hs[None,:]
            port_beats=int(np.bincount(banks.ravel(),minlength=96).max())
            groups=ceil(len(items)*10,96)
            ready=[0]*groups; now=0
            for s in range(10):
                for g in range(groups):
                    issue=max(now,ready[g]); ready[g]=issue+4; now=issue+1
            work+=port_beats+max(ready)+(port_beats if wide else groups)
        self.counters['psn_scalar_macs'] += int(self.sizes[b])*q*self.m['A_nonzero']
        return self.g.add('psn_fullrank', 'numeric', work, deps)

    def moments(self,b,q,dep,previous):
        return self.g.add('moments_Y_sum_square', 'moments',ceil(10*int(self.sizes[b])*q,8)+4,[dep,previous])

    def tau(self,q,deps,name):
        # Eight partial moment lanes, actual sqrt service, affine threshold for
        # every t,h. No full-domain statistics or prefix threshold is free.
        reduce=self.g.add(name+'_moment_reduce','moments',ceil(17*q,8)+4,deps)
        sqrt=self.g.add(name+'_sqrt','sqrt',(q-1)*8+24,[reduce])
        return self.g.add(name+'_affine','moments',ceil(3*q*10,8)+4,[sqrt])

    def wide_access(self,b,q,deps,write=False):
        per = q*10*self.packet_bytes if self.axis=='packet' else int(self.sizes[b])*q*10*8
        offset = b*q*10*self.packet_bytes if self.axis=='packet' else b*32*q*10*8
        local = max(0,min(per,self.wide_keep-offset))
        remote = per-local
        nodes=[]
        suffix='write' if write else 'read'
        if local: nodes.append(self.mem('record_'+suffix if self.axis=='packet' else 'wide_'+suffix,[local],deps,external=False))
        if remote: nodes.append(self.mem('record_external_'+suffix if self.axis=='packet' else 'wide_external_'+suffix,[remote],deps))
        return self.g.add('storage_done',None,0,nodes)

    def output(self,b,q,dep):
        # Base descriptor plus explicit eligible-h mask for selective consumers.
        nbytes=ceil(int(self.sizes[b])*q*10,8)+32
        node=self.g.add('confirmed_theta_g_output','output',ceil(nbytes,16),[dep],nbytes)
        self.marks['first_output'].append(node)
        return node

    def gram(self):
        C=self.C
        init=self.mem('gram_zero',[C*(C+1)//2*4],[self.input],external=False)
        slots=[init,init]
        count_batches=sum(ceil(C-i,8) for i in range(C))
        last=init
        for z,start in enumerate(range(0,self.m['N'],128)):
            nr=min(128,self.m['N']-start)
            read=self.mem('gram_source_read',[nr*C//8],[slots[z%2]])
            trans=self.g.add('gram_transpose','transpose',nr+ceil(C,8),[read])
            # One i-column register; consecutive j columns spread over eight
            # banks. Every counter update needs read then write on 1RW banks.
            issue=self.g.add('gram_popcount_and_counter','gram',C+2*count_batches+6,[trans,last])
            self.counters['gram_pair_popcounts'] += C*(C+1)//2
            self.counters['gram_counter_read_write_bytes'] += C*(C+1)//2*8
            self.counters['gram_source_column_read_bytes'] += (C+C*(C+1)//2)*16
            slots[z%2]=last=issue
        # q16 contraction: six i rows x sixteen h consumers = 96 lanes.
        # The triangular G address determines actual read-bank conflicts.
        qg=16; L=6; reads=bankbeats=0
        for i0 in range(0,C,L):
            ii=np.arange(i0,min(C,i0+L))
            for j in range(C):
                a=np.minimum(ii,j); b=np.maximum(ii,j)
                idx=a*C-a*(a-1)//2+(b-a)
                words=np.unique(idx//4)
                loads=np.bincount(words%8,minlength=8)
                reads += len(words); bankbeats += int(loads.max())
        # Blocked matrix-vector contraction, then w^T v and k^T w, all online.
        # Only one q16 v block is live; no precomputed outer-product ROM.
        for h0 in range(0,self.H,qg):
            parameters=self.mem('gram_static_parameter_read',[qg*10*8+128],[last])
            waits=[]; entries=[]
            for c0 in range(0,C,256):
                a,b=self.weight_get(h0,qg,c0,min(C,c0+256),[last]); waits+=a; entries+=b
            gread=self.g.add('gram_contraction_bank_reads','state',bankbeats,[last]+waits,reads*16)
            # Reads and FMAs form a two-entry operand pipeline: slower service
            # controls issue; pipeline drain and four-way reduction are paid.
            fma=max(bankbeats,ceil(C,L)*C)+4*ceil(C,L)+ceil(5*C*qg,96)
            last=self.g.add('gram_contraction_numeric','numeric',fma,[last]+waits)
            last=self.g.add('gram_contraction_complete',None,0,[last,gread])
            for e in entries: e['last']=last
            vmem=self.mem('gram_v_write_read',[2*C*qg*8],[last],external=False)
            last=self.tau(qg,[vmem,parameters],'gram_final')
            last=self.mem('gram_threshold_write',[qg*10*8],[last],external=False)
            self.counters['gram_contraction_macs'] += C*C*qg+2*C*qg
        self.barrier=last
        self.marks['statistics'].append(last)

    def run(self):
        if self.axis=='gram': self.gram()
        for h0 in range(0,self.H,self.q):
            q=min(self.q,self.H-h0)
            # Only the current h block's immutable affine coefficients and A
            # enter the bounded parameter scratch, not a free full-layer ROM.
            parameter_bytes=(100*8+128 if self.axis=='gram' else q*10*8+100*8+128)
            phase_start=self.mem('static_parameter_read',[parameter_bytes],[self.barrier])
            if self.axis=='gram':
                phase_start=self.mem('gram_threshold_read',[q*10*8],[phase_start],external=False)
            slots=[phase_start,phase_start]
            moment=phase_start
            records=[]; first=[]
            for b in range(self.nb):
                fc=self.fc1(b,h0,q,[slots[b%2],phase_start])
                if self.axis=='gram':
                    ps=self.psn(b,q,[fc,self.barrier]); done=self.output(b,q,ps)
                else:
                    moment=self.moments(b,q,fc,moment)
                    if self.axis=='recompute': done=moment
                    elif self.axis=='save_y': done=self.wide_access(b,q,[moment],write=True)
                    elif self.axis=='save_u':
                        ps=self.psn(b,q,[moment],wide=True)
                        done=self.wide_access(b,q,[ps],write=True)
                    else:
                        pred=self.tau(q,[moment],'prefix')
                        ps=self.psn(b,q,[moment],wide=True)
                        enc=self.g.add('boundary_packet_encode','packet',ceil(q*10,8)*int(self.sizes[b])+4,[ps,pred])
                        done=self.wide_access(b,q,[enc],write=True)
                first.append(done); slots[b%2]=done; records.append(done)
            if self.axis=='gram':
                output_nodes=first
            else:
                seal=self.tau(q,[moment],'final')
                self.marks['statistics'].append(seal)
                slots=[seal,seal]; output_nodes=[]; verified=[]
                for b in range(self.nb):
                    if self.axis=='recompute':
                        fc=self.fc1(b,h0,q,[slots[b%2],seal])
                        ps=self.psn(b,q,[fc]); done=self.output(b,q,ps)
                    elif self.axis in ('save_y','save_u'):
                        rd=self.wide_access(b,q,[slots[b%2],seal,records[b]])
                        ps=self.psn(b,q,[rd]) if self.axis=='save_y' else rd
                        done=self.output(b,q,ps)
                    else:
                        rd=self.wide_access(b,q,[slots[b%2],seal,records[b]])
                        ver=self.g.add('final_packet_verify','packet',ceil(q*10,8)+4,[rd])
                        mask=self.g.add('failure_bitmap_write','metadata',ceil(q,128),[ver],ceil(q,8))
                        verified.append(mask)
                        # Commit only successful consumers now; failed T-group
                        # consumers wait for explicit repair below.
                        success=q-int(self.d['failure'][b,h0:h0+q].sum())
                        done=self.output(b,success,ver) if success else ver
                    slots[b%2]=done; output_nodes.append(done)
                if self.axis=='packet':
                    all_verified=self.g.add('failure_list_ready',None,0,verified)
                    tasks=[]
                    for rh in range(h0,h0+q,self.repair_q):
                        rq=min(self.repair_q,h0+q-rh)
                        for b in range(self.nb):
                            if self.d['failure'][b,rh:rh+rq].any(): tasks.append((b,rh,rq))
                    if self.order=='tile': tasks.sort()
                    elif self.order=='sector': tasks.sort(key=lambda x:(x[1],x[0]))
                    else:
                        capacity=max(1,R['source_transpose_cache_bytes']//(self.C*3*16))
                        tasks.sort(key=lambda x:(x[0]//capacity,x[1],x[0]))
                    if self.order=='tile': scans=self.nb
                    elif self.order=='sector': scans=ceil(q,self.repair_q)*self.nb
                    else: scans=self.nb+ceil(self.nb,capacity)*ceil(q,self.repair_q)
                    all_verified=self.g.add('failure_bitmap_scan','metadata',scans,[all_verified],scans*16)
                    repair_slots=[all_verified,all_verified]
                    for z,(b,rh,rq) in enumerate(tasks):
                        begin=len(self.g.nodes)
                        select=self.g.add('repair_descriptor_select','metadata',1,[repair_slots[z%2],all_verified])
                        fc=self.fc1(b,rh,rq,[select])
                        ps=self.psn(b,rq,[fc])
                        actual_fail=int(self.d['failure'][b,rh:rh+rq].sum())
                        done=self.output(b,actual_fail,ps)
                        repair_slots[z%2]=done; output_nodes.append(done)
                        self.counters['repaired_h_tiles'] += rq
                        self.counters['required_failed_h_tiles'] += actual_fail
                        if self.repair_example is None:
                            self.repair_example=dict(b=b,h=rh,q=rq,required=actual_fail,
                                                     begin=begin,end=len(self.g.nodes),ready=all_verified)
                    self.counters['repair_tasks'] += len(tasks)
            self.marks['last_output'] += output_nodes
            self.barrier=self.g.add('channel_block_done',None,0,output_nodes+first)
        result=self.g.run()
        finish=result.pop('finish')
        interesting=('gram_popcount_and_counter','gram_contraction_bank_reads',
                     'gram_contraction_numeric','gram_threshold_write','fc1_sparse_issue',
                     'psn_fullrank','confirmed_theta_g_output')
        marks=[]
        for name in interesting:
            ids=[i for i,x in enumerate(self.g.nodes) if x[0]==name]
            for i in sorted(set(ids[:1]+ids[-1:])):
                node=self.g.nodes[i]
                marks.append(dict(name=name,start=finish[i]-node[2],end=finish[i],resource=node[1]))
        result['execution_milestones']=marks
        if self.repair_example is not None:
            ex=self.repair_example
            events=[]
            for i in range(ex['begin'],ex['end']):
                name,res,duration,_,amount=self.g.nodes[i]
                if duration: events.append(dict(name=name,resource=res,start=finish[i]-duration,end=finish[i],bytes=amount))
            result['first_repair']=dict(tile=ex['b'],h=ex['h'],q=ex['q'],required_h=ex['required'],
                                        selector_ready=finish[ex['ready']],events=events)
        result['statistics_first_ready']=min((finish[x] for x in self.marks['statistics']),default=0)
        result['statistics_all_ready']=max((finish[x] for x in self.marks['statistics']),default=0)
        result['first_confirmed_output']=min((finish[x] for x in self.marks['first_output']),default=0)
        result.update(axis=self.axis,q=self.q,repair_q=self.repair_q if self.axis=='packet' else None,
                      order=self.order if self.axis=='packet' else None,counters=dict(self.counters),
                      state_live_peak_bytes=self.state_live_peak,
                      external_mutable_peak_bytes=self.external_mutable_peak,
                      source_pin_packets=len(self.pin),wide_onchip_kept_bytes=self.wide_keep,
                      wide_work_live_max_bytes=2*10*32*self.q*8,
                      reserved_wide_work_bytes=2*R['wide_work_buffer_bytes'],
                      source_transpose_reserved_bytes=R['source_transpose_cache_bytes'],
                      packet_record_bytes=self.packet_bytes if self.axis=='packet' else 0)
        return result


def main():
    started=time.time(); allresults=[]
    for sid in R['samples']:
        for stage in R['stages']:
            meta=json.loads((ROOT/f'numeric_s{sid}_stage{stage}.json').read_text())
            data=dict(np.load(ROOT/f'trace_s{sid}_stage{stage}.npz'))
            packed=data['source_packed']
            S=np.unpackbits(packed,axis=1,bitorder='little')[:,:meta['C']]
            kernels={q:kernel_trace(S,meta['P'],meta['C'],q,data['sizes']) for q in R['channel_blocks']}
            rows=[]
            # Discrete standard organizations, not a search for a lucky capacity.
            for axis in ('save_y','save_u','recompute','gram'):
                for q in R['channel_blocks']:
                    for allocation in (('wide_first','source_first') if axis.startswith('save') else ('source_first',)):
                        sim=Simulation(meta,data,kernels,axis,q,allocation)
                        row=sim.run(); row['allocation']=allocation; row.pop('timeline')
                        rows.append(row)
                        print('SERVICE',sid,stage,axis,q,allocation,row['beats'],flush=True)
            # Candidate first-pass block uses the best ordinary block size from
            # save/recompute. Repair ordering and granularity remain independent.
            base=min((r for r in rows if r['axis'] in ('save_y','save_u','recompute')),key=lambda x:x['beats'])
            for q in sorted({base['q'],96}):
                for rq in sorted({min(v,q) for v in (1,4,96)}):
                    for order in ('tile','sector','resident'):
                        sim=Simulation(meta,data,kernels,'packet',q,'wide_first',rq,order)
                        row=sim.run()
                        if sim.example is None and order=='resident' and rq==1:
                            (ROOT/f'timeline_s{sid}_stage{stage}_q{q}.json').write_text(json.dumps(row['timeline'],indent=2)+'\n')
                        row.pop('timeline'); row['allocation']='wide_first'; rows.append(row)
                        print('SERVICE',sid,stage,'packet',q,rq,order,row['beats'],flush=True)
            layer=dict(meta=meta,rows=rows)
            (ROOT/f'service_s{sid}_stage{stage}.json').write_text(json.dumps(layer,indent=2)+'\n')
            allresults.append(layer)
    result=dict(scope='Finite-resource scheduled service beats, not measured RTL clocks. Float64 real-parameter trace; full N=T*P statistics domain for every h. Ordinary channel blocking, source retention and weight broadcast are strong controls.',
                resources=R,layers=allresults,wall_seconds=time.time()-started,
                limitations=['No foundry memory timing/area/energy mapping.',
                             'Input source producer, downstream FC2/BN2/shortcut are outside this BN/PSN component.',
                             'The common platform provisions otherwise idle auxiliary units; no same-area claim.',
                             'Float64 observed support agreement is not a proof of original GPU FP32 equivalence.',
                             'Within each independent channel block, FIFO resource service is modeled; cross-block speculative overlap is not assumed.'])
    (ROOT/'service_result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
