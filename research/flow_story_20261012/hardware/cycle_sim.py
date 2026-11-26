#!/usr/bin/env python3
"""Cycle/event model of ONE stage0 MLP block, not a whole-network simulator.

Explicit finite buffers and nonpreemptive per-tile compute engine. Dot microcode
is clock-stepped once per spatial point and repeated exactly for valid points;
events fast-forward only intervals in which those resources remain reserved.
Representative int8 operands/int32 accumulators; no trained quantization claim.
No event/GT/weights are loaded. Synthetic risks are already available unless
predictor_cycles_per_tile is explicitly charged by the input mask contract.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
RES={'T':10,'tile':9,'channels':96,'hidden':384,'lanes':16,
     'clock_period_ns_assumed':4,'word_bytes':16,'sram_read_latency':1,
     'u_bank_bytes':80*1024,'output_bank_bytes':320*1024,'tile_slots':2,
     'scratch_bytes':32*1024,'weight_sram_bytes':128*1024,
     'descriptor_fifo_entries':2,'dma_queue_entries':4,
     'dot_outstanding':1,'dot_mac_ii':1,
     'dma_max_burst_beats':32,'dma_setup_cycles':6,'dma_read_latency_cycles':12,
     'dma_write_response_cycles':2,'dma_stall_period':17,'dma_stall_phase':7,
     'bn_affine_latency_assumed':2}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ceildiv(a,b):return (a+b-1)//b
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')

def pixel_microcode(r):
    """Replay each MAC request/response, serializer beat and affine dependency.

    Two clocks per MAC: request paired SRAM banks, then response+accumulate.
    Only one read outstanding, matching dot_residual.sv with latency=1. Output
    data stays stable while the 128-bit serializer drains each vector.
    """
    T,C,F,L,W=r['T'],r['channels'],r['hidden'],r['lanes'],r['word_bytes']
    cycles=0; phases=[]; counts=collections.Counter(); example=[]
    def tick(label,n=1,**stats):
        nonlocal cycles
        if len(example)<48:example.append({'cycle':cycles,'microstate':label,'repeat_cycles':n})
        cycles+=n;counts.update(stats)
    def dots(label,outputs,features,K,psn):
        start=cycles
        for output in range(outputs):
            for group in range(ceildiv(features,L)):
                valid=min(L,features-group*L)
                tick(label+'.command',1,dot_commands=1)
                issued=retired=0;pending=[];port_free=cycles;mac_free=cycles
                # Replay finite outstanding slots and single-ported SRAM. The
                # synthesized core is outstanding=1. >=2 is an explicitly
                # UNSYNTHESIZED ordinary pipelined control comparison using
                # the same 16 multipliers and SRAM widths/capacities.
                while retired<K:
                    before=len(pending);stats={};activities=[]
                    if pending and pending[0]<=cycles and cycles>=mac_free:
                        pending.pop(0);retired+=1;mac_free=cycles+r['dot_mac_ii']
                        stats.update(scalar_mac=valid,psn_real_mac=valid if psn else 0,
                                     binary_weighted_ac=0 if psn else valid)
                        activities.append('response_mac')
                    if issued<K and cycles>=port_free and len(pending)<r['dot_outstanding'] and not (r['dot_outstanding']==1 and before):
                        reads=ceildiv(valid,W);pending.append(cycles+reads);port_free=cycles+reads;issued+=1
                        stats.update(activation_read_beats=reads if psn else 1,
                                     weight_read_beats=0 if psn else reads)
                        activities.append('request')
                    tick(label+'.'+'+'.join(activities or ['wait']),1,**stats)
                if psn:
                    tick(label+'.threshold',1,threshold_elements=valid)
                    tick(label+'.spike_write',ceildiv(valid,W),scratch_write_beats=ceildiv(valid,W))
                else:
                    tick(label+'.int32_write',ceildiv(valid*4,W),scratch_write_beats=ceildiv(valid*4,W))
        phases.append({'phase':label,'begin':start,'end':cycles,'cycles':cycles-start})
    def affine(label,channels,residual=False):
        start=cycles
        for t in range(T):
            for group in range(ceildiv(channels,L)):
                valid=min(L,channels-group*L)
                # Frozen alpha,beta each 32 bits. Kept explicitly, not credited
                # as gate savings or assumed free folding. Numeric scale and
                # rounding remain unspecified; this is a charged timing slot.
                b=ceildiv(valid*8,W);tick(label+'.coeff_read',b,weight_read_beats=b)
                b=ceildiv(valid*4,W);tick(label+'.activation_read',b,activation_read_beats=b)
                tick(label+'.affine',r['bn_affine_latency_assumed'],affine_elements=valid)
                if residual:
                    b=ceildiv(valid,W);tick(label+'.u_read',b,u_read_beats=b)
                    tick(label+'.add_u',1,residual_elements=valid)
                    b=ceildiv(valid*4,W);tick(label+'.output_write',b,output_write_beats=b)
                else:
                    b=ceildiv(valid,W);tick(label+'.requant_write',b,scratch_write_beats=b)
        phases.append({'phase':label,'begin':start,'end':cycles,'cycles':cycles-start})
    dots('PSN1',T,C,T,True)
    dots('FC1',T,F,C,False)
    affine('BN1_requant',F)
    dots('PSN2',T,F,T,True)
    dots('FC2',T,C,F,False)
    affine('BN2_residual',C,True)
    # Independent shape-derived operation counts, including dense T*T PSN.
    assert counts['psn_real_mac']==T*T*(C+F)
    assert counts['binary_weighted_ac']==T*(C*F+F*C)
    assert counts['affine_elements']==T*(F+C)
    assert counts['residual_elements']==T*C
    bypass=0
    for t in range(T):
        for group in range(ceildiv(C,L)):
            valid=min(L,C-group*L)
            # Read u from its slot bank; sign-extend into identical int32 output
            # representation and serialize 4 beats. No dummy PSN/FC/BN service.
            bypass+=ceildiv(valid,W)+ceildiv(valid*4,W)
    scratch=T*(C+4*F+F+F+4*C)
    assert scratch<=r['scratch_bytes']
    return {'keep_cycles_per_point':cycles,'bypass_cycles_per_point':bypass,
            'phases':phases,'counters_per_kept_point':dict(counts),
            'scratch_peak_bytes_conservative':scratch,'first_microsteps':example}

def tiles(h,w):
    result=[]
    for y in range(0,h,9):
        for x in range(0,w,9):
            hh,ww=min(9,h-y),min(9,w-x)
            result.append({'id':len(result),'y':y,'x':x,'h':hh,'w':ww,'points':hh*ww})
    assert sum(t['points'] for t in result)==h*w
    # An independent coordinate coverage audit; no padded point gets scheduled.
    coverage=set()
    for t in result:
        for y in range(t['y'],t['y']+t['h']):
            for x in range(t['x'],t['x']+t['w']):
                assert (y,x) not in coverage;coverage.add((y,x))
    assert len(coverage)==h*w
    return result

def group_commands(ts,keep,members,forced=False):
    ids=list(range(len(ts))); groups=[]
    if members==4:
        q=len(ids)//4
        groups=[[i,i+q,i+2*q,i+3*q] for i in range(q)]
        groups += [[i] for i in range(4*q,len(ids))]
    else:groups=[[i] for i in ids]
    result=[]
    for g in groups:
        decision=True if forced else any(keep[i] for i in g)
        result.append({'ids':g,'keep':decision,'risks':[255 if keep[i] else 0 for i in g]})
    return result

def row_segments(h,w,t,channels,element_bytes):
    rows=[]
    for timestep in range(10):
        for y in range(t['y'],t['y']+t['h']):
            address=((timestep*h+y)*w+t['x'])*channels*element_bytes
            size=t['w']*channels*element_bytes
            assert address>=0 and address+size<=10*h*w*channels*element_bytes
            rows.append({'address':address,'bytes':size})
    assert sum(s['bytes'] for s in rows)==t['points']*10*channels*element_bytes
    return rows

def dma_duration(start,segments,write,r):
    """Beat-by-beat bus replay, no outstanding bursts and no overlap of latency.

    These declared latencies are assumptions, not measured DRAM timings. Every
    row/T segment starts a fresh burst; no fictitious dense tile contiguity.
    """
    now=start;beats=bursts=stalls=payload=0
    for nbytes in segments:
        payload+=nbytes
        left=ceildiv(nbytes,r['word_bytes'])
        while left:
            count=min(left,r['dma_max_burst_beats']);left-=count;bursts+=1
            now+=r['dma_setup_cycles']+(0 if write else r['dma_read_latency_cycles'])
            for _ in range(count):
                while r['dma_stall_period'] and now%r['dma_stall_period']==r['dma_stall_phase']:
                    now+=1;stalls+=1
                now+=1;beats+=1
            if write:now+=r['dma_write_response_cycles']
    return now-start,{'beats':beats,'bursts':bursts,'stall_cycles':stalls,'payload_bytes':payload,'bus_word_bytes':beats*r['word_bytes']}

def simulate(h,w,keep,r=None,members=1,forced=False,predictor=0,name='mask',stepwise=False):
    r=dict(RES if r is None else r);ts=tiles(h,w);assert len(keep)==len(ts)
    assert r['channels']==96 and r['hidden']==384 and r['T']==10,'stage0-only capacity/format contract'
    assert r['sram_read_latency']==1 and r['dot_outstanding']>=1 and r['dot_mac_ii']>=1
    kernel=pixel_microcode(r);groups=group_commands(ts,keep,members,forced)
    maxu=max(t['points'] for t in ts)*r['T']*r['channels']
    assert maxu<=r['u_bank_bytes'] and maxu*4<=r['output_bank_bytes']
    # W1+W2 int8, frozen BN alpha/beta 32-bit, dense two T*T matrices
    # and PSN biases (int32). PSN small tables are loaded into registers.
    weight_bytes=2*r['channels']*r['hidden']+(r['channels']+r['hidden'])*8+2*r['T']**2+2*r['T']*4
    assert weight_bytes<=r['weight_sram_bytes']
    slots=[{'state':'free'} for _ in range(r['tile_slots'])]
    descriptors=collections.deque();dma_jobs=[];dma=None;compute=None
    weights='absent';front={'state':'idle','group':0};now=0;done=0;events=[]
    counters=collections.Counter();maxima=collections.Counter();effective=[None]*len(ts)
    completed=set();compute_order=[];job_serial=0
    def enqueue(kind,slot=None):
        nonlocal job_serial
        assert len(dma_jobs)<r['dma_queue_entries'],'DMA request queue overflow'
        dma_jobs.append({'kind':kind,'slot':slot,'enqueued':now,'serial':job_serial});job_serial+=1
    def event(label,**kw):events.append(dict(cycle=now,event=label,**kw))
    while done<len(ts):
        # Completions precede next-cycle dispatch. All intervals [start,end).
        if dma and dma['end']==now:
            kind,s=dma['kind'],dma['slot'];event('dma_done',kind=kind,slot=s)
            if kind=='weights':weights='ready'
            elif kind=='input':slots[s]['state']='loaded'
            else:
                tid=slots[s]['tile'];assert tid not in completed;completed.add(tid)
                slots[s]={'state':'free'};done+=1
            dma=None
        if compute and compute['end']==now:
            s=compute['slot'];slots[s]['state']='output_ready';enqueue('output',s)
            event('compute_done',slot=s,tile=slots[s]['tile']);compute=None
        if front['state']=='wait' and front['end']==now:front['state']='issue'
        # Fixed 2-entry descriptor FIFO; new emits cannot bypass a full queue.
        if front['state']=='issue' and front.get('next',now)<=now and len(descriptors)<r['descriptor_fifo_entries']:
            group=groups[front['group']];i=front.get('member',0);tid=group['ids'][i]
            descriptors.append((tid,group['keep']));effective[tid]=group['keep'];counters['frontend_issue_cycles']+=1
            event('descriptor',tile=tid,keep=group['keep'])
            if i+1==len(group['ids']):
                front={'state':'idle','group':front['group']+1,'next':now+1}
            else:front.update(member=i+1,next=now+1)
        if front['state']=='idle' and front['group']<len(groups) and front.get('next',0)<=now:
            group=groups[front['group']];n=len(group['ids'])
            # Previous RTL proves 1+N+N (or 1+N forced) no-stall cost. Risk
            # prediction is outside that proof; explicit caller charge below.
            pre=1+(0 if forced else n)+predictor*n
            counters['frontend_command_cycles']+=1;counters['risk_collect_cycles']+=0 if forced else n
            counters['predictor_charged_cycles']+=predictor*n
            front.update(state='wait',end=now+pre,member=0)
            event('group_start',group=front['group'],members=n)
        for s,slot in enumerate(slots):
            if slot['state']=='free' and descriptors:
                tid,decision=descriptors.popleft();slots[s]={'state':'loading','tile':tid,'keep':decision,'arrival':now}
                if decision and weights=='absent':enqueue('weights');weights='loading'
                enqueue('input',s);event('allocate',slot=s,tile=tid)
        if compute is None:
            ready=[s for s,slot in enumerate(slots) if slot['state']=='loaded' and (not slot['keep'] or weights=='ready')]
            if ready:
                s=min(ready,key=lambda v:slots[v]['arrival']);slot=slots[s];t=ts[slot['tile']]
                perpoint=kernel['keep_cycles_per_point'] if slot['keep'] else kernel['bypass_cycles_per_point']
                duration=t['points']*perpoint
                compute={'end':now+duration,'slot':s};slot['state']='computing'
                counters['kept_points' if slot['keep'] else 'bypassed_points']+=t['points']
                counters['compute_busy_cycles' if slot['keep'] else 'bypass_busy_cycles']+=duration
                compute_order.append(t['id']);event('compute_start',slot=s,tile=t['id'],keep=slot['keep'],end=now+duration)
        if dma is None and dma_jobs:
            # Output-first avoids slot deadlock; weights before input. One
            # transaction (all row bursts of this tile) owns the shared bus.
            job=min(dma_jobs,key=lambda j:({'output':0,'weights':1,'input':2}[j['kind']],j['serial']));dma_jobs.remove(job)
            if job['kind']=='weights':addresses=[{'address':0,'bytes':weight_bytes}]
            else:
                t=ts[slots[job['slot']]['tile']]
                addresses=row_segments(h,w,t,r['channels'],4 if job['kind']=='output' else 1)
            segments=[a['bytes'] for a in addresses]
            duration,dc=dma_duration(now,segments,job['kind']=='output',r)
            counters['dma_busy_cycles']+=duration;counters['dma_queue_wait_cycles']+=now-job['enqueued']
            for k,v in dc.items():counters['dma_'+k]+=v
            dma=dict(job,end=now+duration);event('dma_start',kind=job['kind'],slot=job['slot'],end=now+duration,
                segment_count=len(addresses),first_segment=addresses[0],last_segment=addresses[-1],**dc)
        maxima['descriptor_fifo']=max(maxima['descriptor_fifo'],len(descriptors));maxima['dma_queue']=max(maxima['dma_queue'],len(dma_jobs))
        maxima['occupied_slots']=max(maxima['occupied_slots'],sum(s['state']!='free' for s in slots))
        if done==len(ts):break
        deadlines=[]
        if dma:deadlines.append(dma['end'])
        if compute:deadlines.append(compute['end'])
        if front['state']=='wait':deadlines.append(front['end'])
        elif front['state']=='issue' and len(descriptors)<r['descriptor_fifo_entries']:deadlines.append(max(now+1,front.get('next',now+1)))
        elif front['state']=='idle' and front['group']<len(groups):deadlines.append(max(now+1,front.get('next',now+1)))
        # A dequeue can free FIFO space in this cycle; dispatch next at +1.
        assert deadlines,'deadlock'
        future=min(d for d in deadlines if d>now);future=now+1 if stepwise else future
        dt=future-now
        if front['state']=='issue' and len(descriptors)>=r['descriptor_fifo_entries']:counters['frontend_backpressure_cycles']+=dt
        if compute is None:counters['compute_idle_cycles']+=dt
        counters['slot_occupancy_cycles']+=dt*sum(s['state']!='free' for s in slots)
        now=future
    assert completed==set(range(len(ts))) and set(compute_order)==completed
    assert counters['kept_points']+counters['bypassed_points']==h*w
    assert counters['compute_busy_cycles']+counters['bypass_busy_cycles']+counters['compute_idle_cycles']==now
    assert counters['dma_payload_bytes']==h*w*r['T']*r['channels']*5+(weight_bytes if any(effective) else 0)
    assert maxima['descriptor_fifo']<=r['descriptor_fifo_entries'] and maxima['dma_queue']<=r['dma_queue_entries']
    return {'name':name,'height':h,'width':w,'channels':96,'T':10,'tile':9,'members':members,'forced_full':forced,
          'clock_cycles':now,'time_ms_at_assumed_4ns':now*4/1e6,
          'frame_fps_ceiling_if_this_were_only_block':1e9/(now*4),'tiles':len(ts),'requested_kept_tiles':sum(keep),
            'effective_kept_tiles':sum(effective),'kept_valid_points':counters['kept_points'],
            'logical_padded_points_not_executed':len(ts)*81-h*w,'weight_load_bytes':weight_bytes if any(effective) else 0,
            'counters':dict(counters),'queue_maxima':dict(maxima),'kernel':kernel,'resource_contract':r,
            'effective_keep':effective,'events':events,'assumptions':['No trained gate; risk already provided unless predictor charge >0',
             'int8/int32 hypothetical numerical contract; no optical-flow accuracy proof',
             'Only stage0 MLP block; attention/backbone/decoder excluded',
             'No inferred whole-network speedup/energy; affine pipeline and SRAM timing are design assumptions']}

def mask_cases(h,w):
    ts=tiles(h,w);n=len(ts);k=n//2
    hardest=set(t['id'] for t in sorted(ts,key=lambda t:(-t['points'],t['id']))[:k])
    return {'dense':[True]*n,'uniform':[i%2==0 for i in range(n)],'clustered':[i<k for i in range(n)],
            'adversarial':[i in hardest for i in range(n)],'all_bypass':[False]*n}

def load_mask(path):
    m=json.loads(Path(path).read_text())
    assert m['schema']=='flow-mlp-mask-v1' and m['batch_size']==1 and m['T']==10
    assert m['channels']==96 and m['tile_size']==9
    assert m['stage_index']==0 and m['block_index'] in (0,1)
    assert m['origin'] in ('runtime_predictor','synthetic','gt_oracle_diagnostic')
    assert isinstance(m['provenance'],dict) and m['provenance']
    h,w=m['height'],m['width'];assert type(h) is int and type(w) is int and h>0 and w>0
    keep=m['keep'];assert isinstance(keep,list) and len(keep)==len(tiles(h,w)) and all(type(v) is bool for v in keep)
    predictor=m.get('predictor_cycles_per_tile',0);assert type(predictor) is int and predictor>=0
    return m

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mask',type=Path);p.add_argument('--output',type=Path,default=HERE/'results/cycle_results.json');a=p.parse_args()
    if a.mask:
        m=load_mask(a.mask);result=simulate(m['height'],m['width'],m['keep'],predictor=m.get('predictor_cycles_per_tile',0),name=m['origin'])
        result['mask_provenance']=m;result['mask_sha256']=digest(a.mask);dump(a.output,result);return
    records=[]
    for h,w in [(10,11),(120,160)]:
        cases=mask_cases(h,w)
        for name,keep in cases.items():
            for members in ([1,4] if name!='all_bypass' else [1]):
                r=simulate(h,w,keep,members=members,name=name)
                records.append(r)
        records.append(simulate(h,w,cases['dense'],forced=True,name='dense_forced_frontend'))
    # Check event fast-forward against the same FSM advanced one clock at a
    # time. This is scheduler validation, NOT independent full-MLP arithmetic.
    replay_checks=[]
    for h,w,decisions,members in [(1,1,[True],1),(1,73,[False]*9,1),(1,73,[False]*9,4)]:
        fast=simulate(h,w,decisions,members=members);slow=simulate(h,w,decisions,members=members,stepwise=True)
        assert fast['clock_cycles']==slow['clock_cycles'] and fast['counters']==slow['counters'] and fast['events']==slow['events']
        replay_checks.append({'height':h,'width':w,'members':members,'cycles':fast['clock_cycles'],
                              'frontend_backpressure':fast['counters'].get('frontend_backpressure_cycles',0)})
    # Resource perturbation: remove second tile slot, and constrain DMA bandwidth
    # with 8-byte beats, on identical partial-edge geometry and masks.
    sensitivity=[]
    for changes in [{'tile_slots':1},{'word_bytes':8},{'dma_stall_period':2,'dma_stall_phase':1}]:
        r=dict(RES,**changes)
        sensitivity.append({'changes':changes,'result':simulate(10,11,mask_cases(10,11)['uniform'],r=r)})
    stronger=[]
    for changes in [{'dot_outstanding':2},{'dot_outstanding':4},{'dot_outstanding':4,'dot_mac_ii':2}]:
        r=dict(RES,**changes)
        stronger.append({'changes':changes,'rtl_verified':False,'same_lanes_and_sram':True,
                         'dense':simulate(120,160,mask_cases(120,160)['dense'],r=r,forced=True,name='ordinary_pipeline_dense'),
                         'uniform':simulate(120,160,mask_cases(120,160)['uniform'],r=r,name='ordinary_pipeline_uniform')})
    data={'schema':'flow-cycle-evidence-v1','status':'passed','script_sha256':digest(__file__),
          'unit_clock_vs_event_exact_match':True,'unit_clock_replay_cases':replay_checks,
          'records':records,'resource_sensitivity':sensitivity,
          'ordinary_stronger_control_comparison':stronger,
          'scope':'one stage0 MLP block; synthetic masks; finite-resource timing model plus separate representative core RTL'}
    dump(a.output,data)
    for r in records:
        print('{}x{} {} group{} keep={}/{} cycles={} DMA={} FEwait={}'.format(r['height'],r['width'],r['name'],r['members'],r['effective_kept_tiles'],r['tiles'],r['clock_cycles'],r['counters']['dma_busy_cycles'],r['counters'].get('frontend_backpressure_cycles',0)))

if __name__=='__main__':main()
