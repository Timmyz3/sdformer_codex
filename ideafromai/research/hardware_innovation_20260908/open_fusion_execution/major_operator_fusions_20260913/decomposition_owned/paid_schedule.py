"""Finite P1/T10/H48 schedule, same ports and resources; not RTL cycles."""
from pathlib import Path
import json,math
import numpy as np
from prototype import HERE,sparse_nm
from count_basis import group

def schedule(groups):
    # Two live group-code slots, two coefficient tile slots, one pipeline for
    # each of source/count preparation, coefficient reads, selection, add array.
    t=0;next_group=0;active=[];tiles=[];running={};done=0;total=sum(len(g['tiles']) for g in groups)
    busy=dict(prep=0,weight=0,select=0,add=0);fifo_peaks=dict(groups=0,weights=0)
    while done<total:
        for kind in list(running):
            end,obj=running[kind]
            if end!=t:continue
            del running[kind]
            if kind=='prep':obj['ready']=True
            elif kind=='weight':obj['w_done']=True
            elif kind=='select':obj['s_done']=True
            else:
                obj['done']=True;done+=1;obj['group']['remaining']-=1
        active=[g for g in active if g['remaining']]
        # Completed coefficient buffers are released only after their add stage.
        tiles=[v for v in tiles if not v.get('done')]
        if 'add' not in running:
            candidates=[v for v in tiles if v.get('s_done') and not v.get('a_started')]
            if candidates:
                obj=candidates[0];obj['a_started']=True;dur=max(1,obj['add']);running['add']=(t+dur,obj);busy['add']+=dur
        if 'select' not in running:
            candidates=[v for v in tiles if v.get('w_done') and not v.get('s_started')]
            if candidates:
                obj=candidates[0];obj['s_started']=True;dur=obj['select']
                if dur:running['select']=(t+dur,obj);busy['select']+=dur
                else:obj['s_done']=True
        if 'weight' not in running and len(tiles)<2:
            candidates=[g for g in active if g.get('ready') and g['next_tile']<len(g['tiles'])]
            if candidates:
                g=candidates[0];obj=dict(g['tiles'][g['next_tile']],group=g);g['next_tile']+=1;tiles.append(obj)
                dur=max(1,obj['weight']);running['weight']=(t+dur,obj);busy['weight']+=dur
        if 'prep' not in running and len(active)<2 and next_group<len(groups):
            g=dict(groups[next_group]);next_group+=1;g.update(remaining=len(g['tiles']),next_tile=0,ready=False);active.append(g)
            dur=g['prep'];running['prep']=(t+dur,g);busy['prep']+=dur
        fifo_peaks['groups']=max(fifo_peaks['groups'],len(active));fifo_peaks['weights']=max(fifo_peaks['weights'],len(tiles))
        # A zero-duration select may make an add stage ready at this instant.
        if 'add' not in running and any(v.get('s_done') and not v.get('a_started') for v in tiles):continue
        if done==total:break
        if not running:raise RuntimeError('deadlock')
        t=min(end for end,_ in running.values())
    return t,busy,fifo_peaks

def strict_rf_schedule(groups):
    # Literal shared-RF policy: an add uses two reads (coefficient and psum)
    # and one psum write in every lane. Coefficient fill or count selection
    # therefore cannot quietly receive extra RF ports. Serialize these RF
    # instructions, granting exactly the same policy to all arms.
    # Count producer also serializes its small-ALU work in this conservative
    # policy; hiding those ALUs behind RF instructions is a later mapping.
    busy=dict(prep=0,weight=0,select=0,walk=0,add=0)
    for g in groups:
        busy['prep']+=g['prep']
        for v in g['tiles']:
            busy['weight']+=max(1,v['weight'])
            busy['select']+=v['select']
            busy['walk']+=v['walk']
            busy['add']+=max(1,v['add'])
    return sum(busy.values()),busy,dict(groups=1,weights=1)

def make_groups(g,pair,signs,retained,mode,o_start,trace=False):
    # g[group216,time10,channel4]; outputs in a 48-wide wave.
    out=[];stream_bits=0;work=dict(source_reads=0,metadata_bits=0,coefficient_bits=0,active_groups=0,
        pair_generator_small_adds=0,consumer_selector_requests=0,fullwidth_adds=0)
    for k in range(216):
        inp=g[k];live_t=int(np.count_nonzero(np.any(inp,axis=-1)))
        # No free nonzero directory: every group is read/tested in each wave.
        if not live_t:
            work['source_reads']+=1
            out.append(dict(prep=1,tiles=[]))
            continue
        work['active_groups']+=1;work['source_reads']+=1
        shared=mode in ('signed_shared','unsigned_shared','unsigned_h8_shared')
        count_mode=shared or mode.endswith('_independent')
        # Six 2-bit count ALUs can generate the six pair values in parallel
        # each live t. Shared code bundle commits to RF in one extra cycle.
        prep=(2+live_t) if shared else 1
        if shared:work['pair_generator_small_adds']+=6*live_t
        tt=[]
        for first in range(o_start,min(o_start+48,96),8):
            outputs=np.arange(first,first+8)
            if mode=='dense_reconstructed':
                events=np.full(8,int(inp.sum()));slot_live=np.any(inp,axis=0)
                slots=int(slot_live.sum());meta=0;select=0
                operands=np.broadcast_to(inp[None],(8,10,4))
            elif mode in ('plain34','plain34_h8_shared','plain34_dense_zero','plain34_h8_dense_zero'):
                masks=group(retained)[outputs,k]!=0
                keep=np.stack([np.flatnonzero(v) for v in masks])
                bits=inp[:,keep].transpose(1,0,2)
                events=bits.sum((1,2));slots=int(np.any(bits,axis=(0,1)).sum())
                operands=bits
                if mode=='plain34_h8_shared':meta=2;select=2;work['consumer_selector_requests']+=2
                elif mode in ('plain34_dense_zero','plain34_h8_dense_zero'):
                    slots=int(np.any(inp,axis=0).sum());meta=0;select=0
                    operands=inp[None]*masks[:,None,:]
                else:meta=8*2;select=live_t;work['consumer_selector_requests']+=8*live_t
            else:
                pp=pair[outputs,k];ss=signs[k]
                chosen=inp[:,pp].transpose(1,0,2)
                cnt=np.sum(chosen*ss[pp][:,None,:],axis=-1)
                rest=np.stack([np.array([j for j in range(4) if j not in p]) for p in pp])
                rb=inp[:,rest].transpose(1,0,2)
                operands=np.concatenate([cnt[...,None],rb],axis=-1)
                events=np.count_nonzero(cnt,axis=-1)+rb.sum((1,2))
                slots=int(np.any(cnt!=0))+int(np.any(rb,axis=(0,1)).sum())
                # 3-bit pair ID/output; shared signs paid once per source group.
                meta=8*3+(4 if first==o_start and mode.startswith('signed') else 0)
                # One 3-bit mux result/lane/cycle: count and the two
                # exception bits require separate passes. Both write their
                # T10 masks into the two already-budgeted control words.
                select=2*live_t if shared else 3*live_t
                if mode=='unsigned_h8_shared':
                    # Unsigned counts have low/high T10 masks: 20 bits fit
                    # the same 8x3-bit selector bank. Select once per H8;
                    # two exception columns retain native source coordinates.
                    meta=3;select=2;work['consumer_selector_requests']+=2
                else:work['consumer_selector_requests']+=16*live_t
                if not shared:work['pair_generator_small_adds']+=8*live_t
            cbits=slots*8*32;before=math.ceil(stream_bits/256);stream_bits+=cbits+meta
            beats=math.ceil(stream_bits/256)-before
            work['metadata_bits']+=meta;work['coefficient_bits']+=cbits;work['fullwidth_adds']+=int(events.sum())
            # All lanes now execute the SAME coefficient slot and time.
            # A single 10-bit priority encoder walks each slot's union mask.
            # Every slot costs one setup; each issued event costs one RF
            # operand-mask read/address step, then one coefficient+psum add.
            # No lane-private event compaction or private addresses assumed.
            issues=int(np.any(operands!=0,axis=0).sum())
            walk=issues+slots
            tt.append(dict(weight=beats,select=select,walk=walk,add=issues))
        out.append(dict(prep=prep,tiles=tt))
    return out,work

def main():
    file=HERE.parent/'root_owned/sttmultires_unet_encoders_swin3d_patch_embed_residual_encoding_resblocks_0_conv2_0.npz'
    z=np.load(file);x=z['input'];w=z['weight'].astype(float);g=x.reshape(10,64,96,3,3).transpose(1,3,4,2,0).reshape(64,216,4,10).transpose(0,1,3,2).astype(np.int8)
    # Current input is exactly {0,1}. Counters consume source bits, not amplitudes.
    assert np.array_equal(g,g.astype(bool))
    axes={}
    for mode in ['dense_reconstructed','plain34','plain34_dense_zero','unsigned_independent','unsigned_shared','signed_independent','signed_shared','plain34_h8_shared','plain34_h8_dense_zero','unsigned_h8_shared']:
        name='unsigned' if mode.startswith('unsigned') else 'signed'
        with np.load(HERE/'count_results'/(name+'_selected_pair_w32.npz')) as q:
            pair=q['pair'];signs=q['signs'];retained=q['retained']
        if mode.startswith('plain34'):retained=sparse_nm(w,3)
        if mode=='unsigned_h8_shared':
            with np.load(HERE/'count_results/unsigned_h8_shared_pair_w32.npz') as q:pair=q['pair'];signs=q['signs'];retained=q['retained']
        if mode in ('plain34_h8_shared','plain34_h8_dense_zero'):
            with np.load(HERE/'count_results/plain34_h8_shared_w32.npz') as q:retained=q['weight']
        positions=[];total_busy=dict(prep=0,weight=0,select=0,walk=0,add=0,clear=0);totals={};maxrf=0
        for p in range(64):
            cycles=216 # load 216 real four-channel T10 words via 48-bit source port
            bodies=[]
            for o in [0,48]:
                groups,work=make_groups(g[p],pair,signs,retained,mode,o)
                body,busy,peaks=strict_rf_schedule(groups)
                # 480 FP32 outputs per wave, one 256-bit write port.
                # Explicit 60 SIMD8 accumulator zero-writes per wave.
                cycles+=60+body+60;total_busy['clear']+=60;bodies.append(body)
                for k,v in busy.items():total_busy[k]+=v
                for k,v in work.items():totals[k]=totals.get(k,0)+v
                assert peaks['groups']<=2 and peaks['weights']<=2
            positions.append(dict(position=int(z['positions'][p]),service_steps=cycles,wave_body_steps=bodies))
        axes[mode]=dict(total_service_steps=sum(v['service_steps'] for v in positions),mean_service_steps=float(np.mean([v['service_steps'] for v in positions])),
            positions=positions,unit_busy_steps=total_busy,logical_work=totals)
        print(mode,axes[mode]['total_service_steps'],total_busy,flush=True)
    base=axes['dense_reconstructed']['total_service_steps']
    for mode,r in axes.items():r['service_steps_vs_dense']=r['total_service_steps']/base
    report=dict(complete=True,current_capture=str(file),actual_scope='All N96/K864/T10 for each of 64 sampled spatial positions. A finite in-order tile model; not RTL, ASIC frequency, layer latency, or AEE.',
        resources=dict(fullwidth_add_lanes=8,fullwidth_acc_bits=48,coefficient_and_descriptor_port_bits=256,source_port_bits=48,output_port_bits=256,
            scratch_words_per_lane=96,scratch_word_bits=48,rf_ports_per_lane='2R1W',pair_count_ALUs=6,pair_count_ALU_bits=2,
            consumer_count_selectors=8,consumer_count_selector='6-to-1 three-bit mux',group_code_slots=2,coefficient_tile_slots=2,global_lookup_bytes=576),
        rf_allocation_per_lane=dict(source_216_words_striped_over_8_lanes=27,acc_H48_T10_striped=60,double_coefficient_tiles_32b_in_48b_words=6,
            two_group_count_code_bundles=1,descriptor_in_spare_16b_of_coefficient_word=0,stream_carry_and_control=2,total=96,capacity=96),
        policy='P1, full T10, H48 waves. All 216 source groups cached once; two waves reread local source. Six pair counts produced once per active group per wave and reused by six H8 consumers. Literal shared 2R1W RF instruction schedule serializes source/count preparation, weight+metadata fills, selector reads, and fullwidth add updates; no uncharged RF ports. Tight bitstream packing granted to all metadata arms. All-zero groups skip after paid source-cache read/setup; coefficient slots skip only if no lane needs them for any t. No cross-position weight cache assumed. Two group/tile storage slots are budgeted but this serialized policy needs only one.',
        event_controller=dict(count=1,priority_encoder_bits=10,mask_and_address_register_bits=64,operand_latch_bits=24,source_response_latch_bits=48,coefficient_response_and_carry_bits=512,semantics='One common slot and time for all eight lanes. Slot setup plus one mask-read/address cycle and one add cycle per union-live event. Selected per-lane count and exception masks fit the existing two 48-bit control words per lane. No lane-private walker.'),
        explicit_limits='Timing-independent deterministic service slots, not an RTL or bit-accurate fixed-point execution. FP32 numerical exports and 32-bit coefficient/48-bit accumulator service widths are separate assumptions; theta-folded quantization, RNE and overflow have not been verified. No physical RF crossbar/critical path, DMA latency, clock, area or energy validated. Packed-source formation/gathering and P4/global mapping are excluded; each cached source group including all-zero groups is nevertheless read/tested per wave. 32-bit output stores are service extents, not validated 48-to-32 rounding. Dense control gets the same reconstructed values and no pair metadata; plain3:4 is a different function with the same FP32 coefficient precision.',axes=axes)
    (HERE/'paid_schedule.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
