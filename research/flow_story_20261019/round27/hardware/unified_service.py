#!/usr/bin/env python3
"""R27 finite-resource, uncalibrated service schedule. No model/GPU forward.

One PE reservation calendar, one DMA calendar, bounded buffers, explicit
matrix tiling/precision, native current-BN barriers, and temporal PSN work.
This is a declared abstract machine, not RTL or measured chip cycles.
"""
import argparse, collections, dataclasses, hashlib, json, math
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
R26=REPO/'research/flow_story_20261017/window_precision_round26'
SCOPE=R26/'hardware/WHOLE_NET_SCOPE_V1.json'
ACTUAL=R26/'actual_v1/run_v1/results.json'
P='sttmultires_unet.'
E=P+'encoders.swin3d.'

def require(x,m):
    if not x: raise ValueError(m)
def ceil(n,d): return (n+d-1)//d
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())

@dataclasses.dataclass(frozen=True)
class Machine:
    output_lanes:int=16
    fp_reduction_lanes:int=16
    int8_reduction_lanes:int=16
    binary_reduction_lanes:int=128
    vector_lanes:int=16
    fp_accum_latency:int=4
    int_accum_latency:int=1
    dot_drain:int=4
    dma_bytes_tick:int=64
    dma_startup:int=32
    input_port_bytes_tick:int=64
    output_port_bytes_tick:int=64
    weight_port_bytes_tick:int=256
    resident_input_bytes:int=4*1024*1024
    io_slot_bytes:int=512*1024
    weight_slot_bytes:int=1024*1024
    weight_slots:int=2
    weight_RF_bytes:int=32*1024
    accumulator_RF_bytes:int=256*1024
    stats_RF_bytes:int=256*1024
    row_batch:int=128
    psn_feature_batch:int=4096
    bn_chunk_elements:int=65536
    reciprocal_sqrt_latency:int=16
    reciprocal_sqrt_II:int=1
    small_output_row_packing:bool=False

def byte_size(n,bits):return ceil(n*bits,8)
def burst(n):return ceil(n,64)*64 if n else 0

def dot_ticks(rows,k,n,mode,bits,m):
    """Output/reduction tails occupy slots; independent rows hide recurrence."""
    lanes=m.binary_reduction_lanes if mode=='bitserial' else m.int8_reduction_lanes if mode=='int8_parallel' else m.fp_reduction_lanes
    chunks=ceil(k,lanes)*(bits if mode=='bitserial' else 1)
    packed_rows=max(1,m.output_lanes//n) if m.small_output_row_packing and n<m.output_lanes else 1
    contexts=ceil(rows,packed_rows)*ceil(n,m.output_lanes)
    latency=m.int_accum_latency if mode in ('bitserial','int8_parallel') else m.fp_accum_latency
    return (chunks-1)*max(contexts,latency)+contexts+m.dot_drain

def pipeline(jobs,m,trace=False):
    """Constructive two-input/two-output-buffer pipeline, shared DMA.

    At compute j: retire j-1 then prefetch j+1. These overlap compute j.
    No cross-operator overlap is presumed. Weight prefetch uses the same DMA.
    Each job carries already port-bounded compute duration and PE work.
    """
    jobs=list(jobs)
    if not jobs:return dict(ticks=0,PE_arithmetic_ticks=0,PE_reserved_ticks=0,DMA_bytes=0,local_input_bytes=0,local_output_bytes=0,weight_SRAM_bytes=0,jobs=0,trace=[])
    require(m.weight_slots==2,'Two explicitly paid 1MiB weight pingpong slots')
    dma=pe=0; loads={}; end={};out_free=[0,0];in_free=[0,0];weight_free=[0,0];events=[]
    sums=collections.Counter()
    def transfer(kind,j,ready,n):
        nonlocal dma
        start=max(dma,ready);duration=ceil(burst(n),m.dma_bytes_tick)+m.dma_startup if n else 0
        dma=start+duration
        if trace:events.append(dict(kind=kind,job=j,resource='DMA',start=start,end=dma,bytes=burst(n)))
        sums['DMA_bytes']+=burst(n)
        return dma
    def load(j):
        q=jobs[j]
        require(q['input_slot_bytes']<=m.io_slot_bytes and q['output_slot_bytes']<=m.io_slot_bytes,'Finite I/O slots')
        require(q.get('weight_slot_bytes',0)<=m.weight_slot_bytes,'Finite resident weight slot')
        require(q.get('accumulator_bytes',0)<=m.accumulator_RF_bytes,'Finite accumulator RF')
        weight_ready=weight_free[q.get('weight_group',0)%2] if q.get('weight_load_bytes',0) else 0
        loads[j]=transfer('load',j,max(in_free[j%2],weight_ready),q['load_bytes'])
    load(0)
    for j,q in enumerate(jobs):
        start=max(pe,loads[j],out_free[j%2]);pe=start+q['ticks'];end[j]=pe;in_free[j%2]=pe
        weight_free[q.get('weight_group',0)%2]=pe
        require(q['ticks']>=ceil(q['local_input_bytes'],m.input_port_bytes_tick),'Input-port capacity')
        require(q['ticks']>=ceil(q['local_output_bytes'],m.output_port_bytes_tick),'Output-port capacity')
        require(q['ticks']>=ceil(q['weight_SRAM_bytes'],m.weight_port_bytes_tick),'Weight-port capacity')
        if trace:events.append(dict(kind='compute',job=j,resource='PE' if q.get('reserves_PE',True) else 'LOCAL_COPY',start=start,end=pe,depends_load=loads[j],out_slot_release=out_free[j%2],input_slot=j%2,output_slot=j%2))
        for key in ('PE_arithmetic_ticks','local_input_bytes','local_output_bytes','weight_SRAM_bytes'):sums[key]+=q[key]
        sums['PE_reserved_ticks']+=q['ticks'] if q.get('reserves_PE',True) else 0
        if j:
            out_free[(j-1)%2]=transfer('store',j-1,end[j-1],jobs[j-1]['store_bytes'])
        if j+1<len(jobs):load(j+1)
    last=len(jobs)-1;out_free[last%2]=transfer('store',last,end[last],jobs[last]['store_bytes'])
    # Resource reservation and causal checks are independent of aggregate sums.
    if trace:
        for resource in ('PE','DMA'):
            stream=sorted((e for e in events if e['resource']==resource),key=lambda e:(e['start'],e['end']))
            require(all(a['end']<=b['start'] for a,b in zip(stream,stream[1:])),'Exclusive resource overlap')
    return dict(ticks=max(pe,dma),jobs=len(jobs),**sums,trace=events if trace else [])

def make_job(arithmetic,local_in,local_out,weight_bytes,m,**kw):
    # Separate physical input and output bank groups, each 16 x 32b 1RW.
    # Weight-RF prefill then overlapped tile refills; bank-address conflicts
    # within a group are NOT established by this aggregate-port model.
    prefill=ceil(min(weight_bytes,m.weight_RF_bytes//2),m.weight_port_bytes_tick)
    duration=prefill+max(arithmetic,ceil(local_in,m.input_port_bytes_tick),ceil(local_out,m.output_port_bytes_tick),ceil(weight_bytes,m.weight_port_bytes_tick))
    return dict(ticks=duration,PE_arithmetic_ticks=arithmetic,local_input_bytes=local_in,local_output_bytes=local_out,weight_SRAM_bytes=weight_bytes,**kw)

def inventory():
    require(sha(SCOPE)=='7b8a0a981a345aac1ae0900cd79e38d7e7800d3ca6c1e326a0f23e719d46375e','Frozen R26 scope')
    require(sha(ACTUAL)=='ca16d2b0c491841f45e43a9ace4010de83706585950903ac22da4d2a4a42ce5c','Audited R26 actual geometry')
    source=read(SCOPE);obs=read(ACTUAL)['forward_records'][0];bn=obs['native_BN_inputs'];ops=[];psns=[];misc=[]
    def add_psn(name,t,e,diagnostic=False):
        require(e%t==0,'Whole temporal PSN input');psns.append(dict(name=name,kind='PSN',T=t,elements=e,features=e//t,diagnostic_only=diagnostic))
    def vec(name,e,ops_per_element=1,inputs=1,output_bits=32,kind='vector'):
        misc.append(dict(name=name,kind=kind,elements=e,ops_per_element=ops_per_element,inputs=inputs,output_bits=output_bits))
    for r in source['affine_records']:
        name=r['name'];w=r['weight'];kind=r['kind'];q=dict(name=name,weight=w,kind=kind,binary_input=True,scope='common',bias=False)
        if kind=='conv':
            h,ww,ci=r['input_HWC'];oh,ow,co=r['output_HWC'];shape=r['weight_shape'];trans='deconv' in name
            k=shape[-1];q.update(T=10,H=oh,W=ow,Cin=ci,Cout=co,rows=10*oh*ow,K=ci*k*k,input_elements=10*h*ww*ci,input_shape=[10,h,ww,ci],kernel=k,transpose=trans)
            q['binary_input']=name not in ('patch.head','patch.PED.conv_res')
            q['bias']=name.startswith('pred')
            if name.startswith('patch.res'):q['scope']='patch4'
            if trans:
                # Native direct output gather: only real kernel/input pairs.
                counts=collections.Counter()
                for y in range(oh):
                    vy=sum(0<=(y+1-j)//2<h and (y+1-j)%2==0 for j in range(3))
                    for x in range(ow):
                        vx=sum(0<=(x+1-j)//2<ww and (x+1-j)%2==0 for j in range(3));counts[vy*vx]+=10
                q['row_domains']=[dict(rows=n,K=ci*taps,complete_rows=0,taps=taps) for taps,n in sorted(counts.items())]
                require(sum(n*taps for taps,n in counts.items())==10*r['valid_spatial_taps'],'Transpose coordinate coverage')
            else:q['row_domains']=[dict(rows=q['rows'],K=q['K'],complete_rows=10*(oh//9*9)*(ow//9*9) if q['scope']=='patch4' else 0)]
        else:
            if kind=='MLP_FC':
                parts=name.split('.');s=int(parts[parts.index('layers')+1]);b=int(parts[parts.index('swin_blocks')+1]);_,h,ww,c=next(v['shape'] for v in source['stage_layer_precision_domain'] if v['stage']==s and v['block']==b)[1:]
                ci,co=(c,4*c) if name.endswith('fc1') else (4*c,c);rows=10*h*ww;q.update(T=10,H=h,W=ww,scope='MLP12')
            elif kind=='attention_projection':
                shape=r['actual_output_BN_shape'];co=ci=shape[2];rows=math.prod(shape)//co;q.update(T=2,H=9,W=9);q['bias']=name.endswith('.proj')
            else:
                shape=bn[name.rsplit('.',1)[0]+'.norm.norm_layer'][0]['shape'];co=shape[2];ci=co*2;rows=math.prod(shape)//co;q.update(T=10,H=shape[-2],W=shape[-1])
            q.update(Cin=ci,Cout=co,rows=rows,K=ci,input_elements=rows*ci,input_shape=None,kernel=1,transpose=False)
            q['row_domains']=[dict(rows=rows,K=ci,complete_rows=10*(q['H']//9*9)*(q['W']//9*9) if q['scope']=='MLP12' else 0)]
        if kind=='conv':q['producer_BN_name']=None if name=='patch.PED.conv_res' else r['actual_output_BN_name']
        elif kind=='MLP_FC':q['producer_BN_name']=name.rsplit('.',1)[0]+('.bn1.norm_layer' if name.endswith('fc1') else '.bn2.norm_layer')
        elif kind=='attention_projection':q['producer_BN_name']=r['actual_output_BN_name']
        else:q['producer_BN_name']=name.rsplit('.',1)[0]+'.norm.norm_layer'
        ops.append(q)
    # Full current-BN domains are read from actual observations, not old folded-BN costs.
    bns=[dict(name=n,kind='current_BN',shape=v[0]['shape'],elements=math.prod(v[0]['shape']),channels=v[0]['shape'][2] if len(v[0]['shape'])==5 else v[0]['shape'][1]) for n,v in bn.items()]
    add_psn(E+'patch_embed.head.sn',10,10*480*640*48)
    for i in range(2):
        for j in (1,2):add_psn(E+f'patch_embed.residual_encoding.resblocks.{i}.sn{j}',10,10*240*320*96)
        vec(f'patch.res{i}.residual_add',10*240*320*96,inputs=2)
    add_psn(E+'patch_embed.proj.sn',10,10*240*320*96);vec('patch.PED.residual_add',10*120*160*96,inputs=2)
    for l in source['stage_layer_precision_domain']:
        t,b,h,ww,c=l['shape'];q=E+f'layers.{l["stage"]}.swin_blocks.{l["block"]}.'
        packed=math.prod(bn[q+'attn.bn_q.norm_layer'][0]['shape']);e=t*b*h*ww*c
        for name,ee in (('proj_sn',packed),('sn_q',packed),('sn_k',packed),('sn2_q',packed//32),('attn_sn',packed)):
            add_psn(q+'attn.'+name,2,ee,name=='attn_sn')
        add_psn(q+'mlp.sn1',10,e);add_psn(q+'mlp.sn2',10,e*4)
        vec(q+'attention_pos_add',packed);misc[-1]['parameter_elements']=2*9*9*c
        # q is binary: ordinary hardware uses exact integer popcount32, not
        # a deliberately slow sequence of31 FP32 additions.
        vec(q+'attention_sum32',packed//32,1,inputs=32,output_bits=32,kind='attention_reduce')
        vec(q+'attention_binary_gate',packed,inputs=2,output_bits=1,kind='attention_gate')
        vec(q+'attention_residual_add',e,inputs=2);vec(q+'MLP_residual_add',e,inputs=2)
        # Static permutation/padding and unpadding/shift. Two materialized remaps.
        vec(q+'partition_and_shift',packed,0,kind='remap');vec(q+'reverse_unpad_and_unshift',e,0,kind='remap')
    for s,(h,ww,c) in enumerate(((120,160,96),(60,80,192),(30,40,384))):
        ee=10*ceil(h,2)*ceil(ww,2)*4*c;add_psn(E+f'layers.{s}.downsample.sn',10,ee);vec(f'merge{s}.gather',ee,0,kind='remap')
    for i in range(2):
        for j in (1,2):add_psn(P+f'resblocks.{i}.sn{j}',10,10*15*20*768)
        vec(f'bottleneck.res{i}.residual_add',10*15*20*768,inputs=2)
    for i,(h,ww,ci,co) in enumerate(((15,20,1536,384),(30,40,770,192),(60,80,386,96),(120,160,194,96))):
        ee=10*h*ww*ci;vec(f'decoder{i}.concat',ee,0,kind='remap');add_psn(P+f'decoders.{i}.sn',10,ee)
        add_psn(P+f'preds.{i}.sn',10,10*2*h*2*ww*co)
        vec(f'pred{i}.sumT10',2*h*2*ww*2,9,inputs=10,kind='output_temporal_sum')
        vec(f'pred{i}.nearest_full_output',480*640*2,0,kind='remap')
    require(len(ops)==83 and len(bns)==78 and len(psns)==105,'Whole graph inventory')
    require({o['producer_BN_name'] for o in ops if o['producer_BN_name']}==set(bn),'Each78BN has one actual affine producer, not PED residual twice')
    require(sum(v['elements'] for v in bns)==1414344960,'Actual all78 BN domain')
    require(sum(x['diagnostic_only'] for x in psns)==12,'12 discarded attention diagnostic PSNs')
    return dict(affines=ops,BN=bns,PSN=psns,misc=misc,source_scope_sha256=sha(SCOPE),actual_geometry_sha256=sha(ACTUAL),native_source_sha256=source['native_source_sha256'])

def affine_service(op,m,mode='native',precision='Q8',trace=False):
    """Resident weights or bounded output stripes; no full im2col materialization.

    Regular conv uses padded dense K (zero taps cost slots); transpose uses
    valid-tap buckets. Line-buffered input pass per weight/domain sweep is
    explicit. This is a hardware addressing assumption, not torch._int_mm.
    """
    n=op['Cout'];quant=mode!='native';input_bits=1 if op['binary_input'] else 32
    weight_bits=8 if quant else 32
    input_bytes=byte_size(op['input_elements'],input_bits)
    resident=input_bytes<=m.resident_input_bytes
    linebuf=0
    if op['kind']=='conv':linebuf=byte_size(op['kernel']*op['input_shape'][2]*op['Cin'],input_bits)
    require(linebuf<=m.io_slot_bytes,'Conv halo/line buffer fits declared slot')
    jobs=[];matrix_ticks=scale_ticks=bias_ticks=mean_ticks=0;rowplane=0;weight_offchip=0
    domains=[]
    for d in op['row_domains']:
        if mode=='bitserial' and precision=='Q6' and d['complete_rows']:
            domains.extend([dict(**d,active_rows=d['complete_rows'],planes=6),dict(**d,active_rows=d['rows']-d['complete_rows'],planes=8)])
        else:domains.append(dict(**d,active_rows=d['rows'],planes=8))
    # All8 weights remain required by tails. One representation is stored;
    # bitparallel and bitserial formats are alternative implementations.
    kmax=max(d['K'] for d in domains)
    per_output_weight_bytes=byte_size(kmax,weight_bits)+(4 if op['bias'] else 0)+(4 if quant else 0)
    stripe=max(m.output_lanes,(m.weight_slot_bytes//per_output_weight_bytes//m.output_lanes)*m.output_lanes)
    stripe=min(n,stripe)
    require(byte_size(kmax*stripe,weight_bits)<=m.weight_slot_bytes,'Legal weight stripe')
    initial_resident_load=input_bytes if resident else 0
    for out0 in range(0,n,stripe):
        nn=min(stripe,n-out0);weight_load=byte_size(kmax*nn,weight_bits)+(nn*4 if op['bias'] else 0)+(nn*4 if quant else 0);weight_offchip+=weight_load
        for di,d in enumerate(domains):
            rows=d['active_rows'];k=d['K'];planes=d['planes'];rowplane+=rows*planes if mode=='bitserial' else 0
            # Avoid double charging logical row-plane work across output stripes.
            if not rows:continue
            batch=min(m.row_batch,max(1,m.accumulator_RF_bytes//(nn*4)),max(1,m.io_slot_bytes//(nn*4)))
            # Each streaming pass feeds the physical source once. Q6 full/tail
            # separate sweeps trade simple packing for a second pass. No free
            # implicit gather/im2col or external-weight cancellation is claimed.
            external_pass=0 if resident else input_bytes
            prev_load=0
            for start in range(0,rows,batch):
                mm=min(batch,rows-start);mat=dot_ticks(mm,k,nn,mode,planes,m);matrix_ticks+=mat
                scale=2*ceil(mm*nn,m.vector_lanes) if quant else 0;scale_ticks+=scale
                bias=ceil(mm*nn,m.vector_lanes) if op['bias'] else 0;bias_ticks+=bias
                mean=ceil(mm*nn,m.vector_lanes) if op['producer_BN_name'] else 0;mean_ticks+=mean
                # Binary packed input is reread per output16 block; full T is
                # retained. Ordinary native gate-add receives the same reuse.
                local_in=byte_size(mm*k*ceil(nn,m.output_lanes),input_bits)
                local_out=4*mm*nn
                weight_read=byte_size(k*nn*(planes if mode=='bitserial' else weight_bits),1)+(nn*4 if op['bias'] else 0)+(nn*4 if quant else 0)
                # Integer->FP32 conversion and scale use the same PE vector
                # lanes; accumulator RF reads/writes are additional local work.
                local_in+=4*mm*nn if quant else 0
                now_load=ceil((start+mm)*external_pass,rows);loadbytes=now_load-prev_load;prev_load=now_load
                if start==0 and di==0:loadbytes+=weight_load+initial_resident_load;initial_resident_load=0
                jobs.append(make_job(mat+scale+bias+mean,local_in,local_out,weight_read,m,
                    load_bytes=loadbytes,store_bytes=local_out,input_slot_bytes=linebuf+byte_size(mm*k,input_bits),
                    output_slot_bytes=local_out,weight_slot_bytes=weight_load,accumulator_bytes=mm*nn*4,
                    weight_group=out0//stripe,weight_load_bytes=weight_load if start==0 and di==0 else 0))
    result=pipeline(jobs,m,trace);result.update(matrix_PE_ticks=matrix_ticks,scale_conversion_PE_ticks=scale_ticks,bias_PE_ticks=bias_ticks,producer_BN_mean_PE_ticks=mean_ticks,
        rows=op['rows'],logical_row_planes=rowplane//ceil(n,stripe) if mode=='bitserial' else None,
        source_weight_DMA_bytes=weight_offchip,input_resident=resident,source_input_bytes=input_bytes,
        line_buffer_bytes=linebuf,mode=mode,precision=precision,weight_format='packed8_bitplanes' if mode=='bitserial' else 'int8' if quant else 'FP32')
    return result

def vector_service(elements,m,*,operations=1,inputs=1,output_bits=32,input_bits=32,throughput=None,parameter_elements=0,trace=False):
    jobs=[]
    rate=throughput or m.vector_lanes
    for start in range(0,elements,m.bn_chunk_elements):
        n=min(m.bn_chunk_elements,elements-start);ib=byte_size(n*inputs,input_bits);ob=byte_size(n,output_bits)
        # Large reductions are split into chunks so both slots remain bounded.
        if ib>m.io_slot_bytes:
            # Recursive smaller chunks are represented by a single equivalent
            # sequence, never by reserving more physical SRAM.
            local=dataclasses.replace(m,bn_chunk_elements=max(1,m.io_slot_bytes//max(1,byte_size(inputs,input_bits))))
            # This branch is rarely used; return chunk streams via helper.
            for substart in range(0,n,local.bn_chunk_elements):
                nn=min(local.bn_chunk_elements,n-substart);ii=byte_size(nn*inputs,input_bits);oo=byte_size(nn,output_bits)
                jobs.append(make_job(ceil(nn*operations,rate)+4 if operations else 0,ii,oo,nn*4 if parameter_elements else 0,m,load_bytes=ii+(parameter_elements*4 if start==substart==0 else 0),store_bytes=oo,input_slot_bytes=ii,output_slot_bytes=oo,reserves_PE=bool(operations)))
        else:jobs.append(make_job(ceil(n*operations,rate)+4 if operations else 0,ib,ob,n*4 if parameter_elements else 0,m,load_bytes=ib+(parameter_elements*4 if start==0 else 0),store_bytes=ob,input_slot_bytes=ib,output_slot_bytes=ob,reserves_PE=bool(operations)))
    return pipeline(jobs,m,trace)

def bn_service(op,m,trace=False):
    e,c=op['elements'],op['channels'];size=e*4
    require(12*c*4<=m.stats_RF_bytes,'Four mean and four variance carry vectors + affine factors fit stats RF')
    # Four partial sums/channel hide assumed FP32 recurrence latency. Mean is
    # accumulated while producer emits Z: paid PE work, no extra external read.
    # Element-wise mean updates are already charged in each affine producer's
    # PE jobs, before releasing its accumulators. Only final carry merge here.
    mean_ticks=ceil(3*c,m.vector_lanes)+4
    mean_parameter_ticks=ceil(c,m.vector_lanes)
    retained=size<=m.resident_input_bytes
    phase=[]
    for pname,arith in [('variance',2),('normalization',1)]:
        jobs=[]
        for start in range(0,e,m.bn_chunk_elements):
            n=min(m.bn_chunk_elements,e-start);ib=n*4;ob=n*4 if pname=='normalization' else 0
            jobs.append(make_job(ceil(n*arith,m.vector_lanes)+4,ib,ob,0,m,load_bytes=0 if retained else ib,store_bytes=ob,input_slot_bytes=ib,output_slot_bytes=ob))
        phase.append((pname,pipeline(jobs,m,trace)))
    final=ceil(3*c,m.vector_lanes)+ceil(c,m.vector_lanes)*4+c*m.reciprocal_sqrt_II+m.reciprocal_sqrt_latency
    total=collections.Counter()
    for _,p in phase:
        for key in ('ticks','PE_arithmetic_ticks','PE_reserved_ticks','DMA_bytes','local_input_bytes','local_output_bytes','weight_SRAM_bytes','jobs'):total[key]+=p.get(key,0)
    param_bytes=burst(c*2*4);param_dma=ceil(param_bytes,m.dma_bytes_tick)+m.dma_startup
    total['DMA_bytes']+=param_bytes;total['ticks']+=param_dma+mean_ticks+mean_parameter_ticks+final;total['PE_arithmetic_ticks']+=mean_ticks+mean_parameter_ticks+final;total['PE_reserved_ticks']+=mean_ticks+mean_parameter_ticks+final
    # Producer's Z store is already in affine output traffic. If retained, this
    # schedule still writes that copy: legal, conservative, not minimal traffic.
    return dict(total,elements=e,channels=c,Z_retained=retained,global_barriers=2,mean_fused_with_producer=True,mean_PE_ticks=mean_ticks+mean_parameter_ticks,variance_finalize_PE_ticks=final,retention_bytes=size if retained else 0,phases=[dict(name=n,**p) for n,p in phase] if trace else [])

def psn_service(op,m):
    t=op['T'];f=op['features'];jobs=[];matrix=0
    batch=min(m.psn_feature_batch,m.accumulator_RF_bytes//(t*4),m.io_slot_bytes//(t*4))
    for start in range(0,f,batch):
        rows=min(batch,f-start);ib=rows*t*4;ob=byte_size(rows*t,1);mat=dot_ticks(rows,t,t,'native',8,m);matrix+=mat
        # Temporal learned real matrix, bias, threshold and packed-bit write.
        arith=mat+2*ceil(rows*t,m.vector_lanes);wb=(t*t+t)*4 if start==0 else 0
        jobs.append(make_job(arith,ib,ob,wb,m,load_bytes=ib+wb,store_bytes=ob,input_slot_bytes=ib,output_slot_bytes=ob,weight_slot_bytes=wb,accumulator_bytes=rows*t*4))
    r=pipeline(jobs,m);r.update(matrix_PE_ticks=matrix,elements=op['elements'],T=t,temporal_real_products=op['elements']*t)
    return r

_SERVICE_CACHE={}
def scenario(inv,m,scope,implementation,include_diagnostic=False):
    def cached(op,category,mode=None,precision=None):
        key=(m,category,op['name'],mode,precision)
        if key not in _SERVICE_CACHE:
            if category=='affine':value=affine_service(op,m,mode,precision)
            elif category=='BN':value=bn_service(op,m)
            elif category=='PSN':value=psn_service(op,m)
            else:value=vector_service(op['elements'],m,operations=op['ops_per_element'],inputs=op['inputs'],output_bits=op['output_bits'],input_bits=1 if op['kind'] in ('attention_gate','attention_reduce') else 32,throughput=m.output_lanes*m.binary_reduction_lanes if op['kind']=='attention_gate' else None,parameter_elements=op.get('parameter_elements',0))
            _SERVICE_CACHE[key]=value
        return _SERVICE_CACHE[key]
    records=[]
    for op in inv['affines']:
        selected=op['scope']=='MLP12' or (scope=='MLP12_patch4' and op['scope']=='patch4')
        if implementation=='native' or not selected:mode='native';precision='FP32'
        elif implementation=='Q8_parallel':mode='int8_parallel';precision='Q8'
        elif implementation=='Q8_serial':mode='bitserial';precision='Q8'
        elif implementation=='Q6_serial':mode='bitserial';precision='Q6'
        elif implementation=='Q8_strong':
            p=cached(op,'affine','int8_parallel','Q8');s=cached(op,'affine','bitserial','Q8')
            # Ordinary static per-layer implementation chosen by primary PE
            # reservation count, before reading quality; total ticks tie-break.
            mode='int8_parallel' if (p['PE_reserved_ticks'],p['ticks'])<=(s['PE_reserved_ticks'],s['ticks']) else 'bitserial';precision='Q8'
        else:raise ValueError(implementation)
        r=cached(op,'affine',mode,precision);records.append(dict(name=op['name'],category='affine',scope=op['scope'],selected=selected,**r))
    for op in inv['BN']:records.append(dict(name=op['name'],category='current_BN',scope='common',**cached(op,'BN')))
    for op in inv['PSN']:
        if op['diagnostic_only'] and not include_diagnostic:continue
        records.append(dict(name=op['name'],category='PSN',scope='common',**cached(op,'PSN')))
    for op in inv['misc']:
        r=cached(op,'misc')
        records.append(dict(name=op['name'],category=op['kind'],scope='common',**r))
    summary=collections.Counter();bycat={}
    keys=('PE_arithmetic_ticks','PE_reserved_ticks','ticks','DMA_bytes','local_input_bytes','local_output_bytes','weight_SRAM_bytes','jobs')
    for r in records:
        for key in keys:summary[key]+=r.get(key,0)
        bc=bycat.setdefault(r['category'],collections.Counter())
        for key in keys:bc[key]+=r.get(key,0)
    return dict(scope=scope,implementation=implementation,include_discarded_diagnostic_PSN=include_diagnostic,summary=dict(summary),by_category=bycat,records=records)

def comparisons(results):
    out=[]
    for scope in ('MLP12','MLP12_patch4'):
        pick=lambda mode:next(x for x in results if x['scope']==scope and x['implementation']==mode)
        candidate=pick('Q6_serial')
        for mode in ('native','Q8_parallel','Q8_serial','Q8_strong'):
            base=pick(mode);row=dict(scope=scope,baseline=mode,candidate='Q6_serial')
            for metric in ('PE_arithmetic_ticks','PE_reserved_ticks','ticks','DMA_bytes'):
                b,c=base['summary'][metric],candidate['summary'][metric]
                row[metric]=dict(baseline=b,candidate=c,reduction_fraction=1-c/b,incremental_scorer_budget_for_10percent=.9*b-c)
            row['scorer_budget_scope']='Necessary budget on this one assumed PE/calendar; no scorer, no area/timing calibration, no quality qualification.'
            out.append(row)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--quick',action='store_true');a=p.parse_args()
    require(not a.output.exists(),'Exclusive output; freeze versions immutable');inv=inventory();m=Machine()
    results=[scenario(inv,m,s,mode) for s in ('MLP12','MLP12_patch4') for mode in ('native','Q8_parallel','Q8_serial','Q6_serial','Q8_strong')]
    diagnostics=[psn_service(x,m) for x in inv['PSN'] if x['diagnostic_only']]
    sensitivity=[]
    if not a.quick:
        for label,mm in [('binary64',dataclasses.replace(m,binary_reduction_lanes=64)),('binary256',dataclasses.replace(m,binary_reduction_lanes=256)),('DMA16',dataclasses.replace(m,dma_bytes_tick=16)),('DMA32',dataclasses.replace(m,dma_bytes_tick=32)),('smallN_context_packing',dataclasses.replace(m,small_output_row_packing=True))]:
            rr=[scenario(inv,mm,'MLP12_patch4',mode) for mode in ('Q8_parallel','Q8_serial','Q6_serial','Q8_strong')]
            sensitivity.append(dict(label=label,machine=dataclasses.asdict(mm),results=[{k:v for k,v in r.items() if k!='records'} for r in rr]))
    result=dict(schema='flow_r27_unified_abstract_service_v1',status='EXECUTABLE_UNCALIBRATED_RESOURCE_SCHEDULE',machine=dataclasses.asdict(m),inventory=inv,
        results=results,comparisons=comparisons(results),executed_diagnostic_PSN_addendum=dict(count=12,apply_identically_to_all_scenarios=True,summary={key:sum(x.get(key,0) for x in diagnostics) for key in ('PE_arithmetic_ticks','PE_reserved_ticks','ticks','DMA_bytes')},source_trace_PSN_count=105,flow_live_optimized_PSN_count=93),
        sensitivity=sensitivity,source_sha256={str(Path(__file__)):sha(__file__),str(SCOPE):sha(SCOPE),str(ACTUAL):sha(ACTUAL)},model_forwards=0,GT_read=False,GPU_used=False,
        scope_notes=['PE_arithmetic_ticks are shape/latency/dependency-issued slots, not MACs divided by throughput. PE_reserved adds declared local-port/RF stalls; elapsed ticks also include one shared DMA and double-buffer dependencies.',
        'FP32 gate-add, FP32 temporal FMA, int8 parallel and binary popcount are assumed modes of one resource-normalized machine. 16x128 binary versus16x16 int8/FP32 is NOT area/timing calibrated; stronger ordinaryQ8 may choose parallel. No native-FP32 toQ6 speedup is claimed without quality and physical calibration.',
        'The whole-layer schedule materializes operator boundaries except producer mean and retained BN Z. Weight/input residency and intra-operator double buffering are implemented; graph-wide liveness cache/streaming fusion, addresses/bank conflicts and complete RTL are not. Elapsed ticks are a constructive conservative schedule, not an optimized chip latency.',
        'Both ordinary and candidate keep all78 current BN, all spatial locations and allT. BN centered two-pass algebra and assumed FP32 functional units are not bitwise validated against PyTorch current BN. Quantized op scale/conversion and signed plane accumulation are included.',
        'Conv line-buffer/window address generation and one-bit activation/eight-bit weight packing are explicit hardware assumptions. Software int8-carried planes, F.unfold materialization, training quantization, hashing/captures are not mistaken for packed hardware traffic. Truepadding zero taps cost regular-conv slots; transpose uses valid coordinate buckets.',
        'Off-chip selected weights remain all8 bits because true tail outputs requireQ8. No automatic reduction of off-chip weight storage/traffic is credited toQ6.',
        'All105 original PSNs execute in the real source. Twelve attention attn_sn outputs are diagnostics discarded on the flow path; ordinary dead-output elimination removes the same12 from every primary scenario. Their executed-inclusive cost is separately reported.',
        'This is ordinary fixed precision, not motion-aware allocation. No scorer is present; positive remaining10percent budget is necessary, not sufficient. Local gate budgets cannot establish whole-net speedup or journal contribution.'])
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],comparisons=result['comparisons'],diagnostic=result['executed_diagnostic_PSN_addendum']),indent=2))

if __name__=='__main__':main()
