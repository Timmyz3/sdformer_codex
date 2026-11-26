#!/usr/bin/env python3
"""R30 fixed-machine, discrete mixed quota. CPU metadata/scheduling only.

The inherited scheduler models each operator separately; this is not a whole
graph DAG trace, calibrated silicon performance, or a quality result.
"""
import argparse, collections, copy, dataclasses, fractions, hashlib, importlib.util, json, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
OLD=REPO/'research/flow_story_20261019/round27/hardware'
sys.path.insert(0,str(OLD))
import unified_service as s
import q4_cost_addendum as q
SOURCE_SHA=q.SOURCE_SHA
RESULT_SHA=q.RESULT_SHA
PREFIX='sttmultires_unet.encoders.swin3d.'
KEYS=('PE_arithmetic_ticks','PE_reserved_ticks','ticks','DMA_bytes','local_input_bytes','local_output_bytes','weight_SRAM_bytes','jobs')
RESERVE_NUM=3652955059
RESERVE_DEN=100

def group(op):
    if op['scope']=='MLP12':return op['name'].rsplit('.',1)[0]
    a=op['name'].split('.')
    return PREFIX+f'patch_embed.residual_encoding.resblocks.{a[1][-1]}.conv{int(a[2][-1])+1}.0'

def geometry(inv,crop=False):
    out={}
    for op in inv['affines']:
        if op['scope'] not in ('MLP12','patch4'):continue
        h,w=op['H'],op['W']
        if crop:h,w=h*3//5,w*3//5
        name=group(op)
        row=dict(H=h,W=w,complete_tiles=(h//9)*(w//9),T=10,B=1)
        if name in out:s.require(out[name]==row,'FC1/2 geometry shared')
        out[name]=row
    s.require(len(out)==16,'Exactly12 MLP groups and4 conv groups')
    return out

def quotas(geom,f):
    return {g:dict(v,Q8_complete_tiles=(f.numerator*v['complete_tiles'])//f.denominator) for g,v in geom.items()}

def uniform_tiles(v):
    """Centered, equally spaced row-major strata; no score or labels."""
    n,k=v['complete_tiles'],v['Q8_complete_tiles'];w=v['W']//9
    ids=[((2*j+1)*n)//(2*k) for j in range(k)] if k else []
    s.require(len(set(ids))==k and all(0<=i<n for i in ids),'Unique uniform coordinates')
    return [[i//w,i%w] for i in ids]

def config(op,k,m):
    low=op['row_domains'][0]['complete_rows']-810*k
    s.require(0<=low<=op['rows'],'Whole-tile quota rows')
    perout=s.byte_size(op['K'],8)+(4 if op['bias'] else 0)+4
    stripe=min(op['Cout'],max(m.output_lanes,(m.weight_slot_bytes//perout//m.output_lanes)*m.output_lanes))
    return low,stripe

def fast_core_pe(op,k,m):
    """Closed-form sum of the inherited per-job dot/scale/BN-mean schedule."""
    low,stripe=config(op,k,m)
    total=reserved=jobs=0
    for o in range(0,op['Cout'],stripe):
        n=min(stripe,op['Cout']-o)
        batch=min(m.row_batch,m.accumulator_RF_bytes//(n*4),m.io_slot_bytes//(n*4))
        for rows,bits in ((low,4),(op['rows']-low,8)):
            if not rows:continue
            full,tail=divmod(rows,batch)
            def cost(r):
                pe=s.dot_ticks(r,op['K'],n,'bitserial',bits,m)+2*s.ceil(r*n,m.vector_lanes)+(s.ceil(r*n,m.vector_lanes) if op['bias'] else 0)+(s.ceil(r*n,m.vector_lanes) if op['producer_BN_name'] else 0)
                local_in=s.byte_size(r*op['K']*s.ceil(n,m.output_lanes),1)+4*r*n
                local_out=4*r*n
                weights=s.byte_size(op['K']*n*bits,1)+(4*n if op['bias'] else 0)+4*n
                return pe,s.make_job(pe,local_in,local_out,weights,m)['ticks']
            p,r=cost(batch);total+=full*p;reserved+=full*r
            if tail:
                p,r=cost(tail);total+=p;reserved+=r
            jobs+=full+bool(tail)
    return total,jobs,reserved

def obligations(op,k,m):
    """Paid conservative aggregate-port staging, on top of inherited work.

    Two domain scans, explicit bit-packed row gather write/read, full output
    scatter descriptors, bitmap bytes, and one mode setup per matrix job.
    No extra physical SRAM: staging reuses the inherited input/output slots
    before compute. No bank-address correctness claim is made.
    """
    _,stripe=config(op,k,m);stripes=s.ceil(op['Cout'],stripe)
    rows=op['rows'];h,w=op['H'],op['W']
    _,jobs,_=fast_core_pe(op,k,m)
    bitmap=s.ceil(s.ceil(h,9)*s.ceil(w,9),8)
    s.require(bitmap+64<=2048,'Bitmap and address counters fit2KiB reservation within existing stats RF')
    # Every row in both scans pays an address/classify operation pair; scatter
    # address creation pays another pair. Conservative shared-vector billing.
    address_ops=rows*6*stripes
    address_pe=s.ceil(address_ops,m.vector_lanes)+4
    mode_pe=4*jobs
    bitmap_reads=2*rows*stripes  # one byte per row per domain scan, no cache credit
    gathered=s.byte_size(rows*op['K'],1)*stripes
    # Extra gather read and packed staging write; computation's existing reads
    # and the ordinary full-grid output write remain in the inherited service.
    local_r=bitmap_reads+gathered
    local_w=gathered
    staging=s.ceil(local_r,m.input_port_bytes_tick)+s.ceil(local_w,m.output_port_bytes_tick)+8
    # Do not merge adjacent tiles. Each true 9-wide output span has its own
    # DMA scatter descriptor; incomplete bottom rows use one full-width span.
    spans=10*((h//9*9)*(w//9+int(w%9!=0))+(h%9))*stripes
    scatter_start=spans*m.dma_startup
    mask_dma=s.burst(bitmap)
    mask_time=s.ceil(mask_dma,m.dma_bytes_tick)+m.dma_startup
    pe=address_pe+mode_pe
    return dict(PE_arithmetic_ticks=pe,PE_reserved_ticks=pe,ticks=pe+staging+scatter_start+mask_time,
                DMA_bytes=mask_dma,local_input_bytes=local_r,local_output_bytes=local_w,
                weight_SRAM_bytes=0,jobs=0,bitmap_storage_bytes=bitmap,
                classification_and_scatter_integer_ops=address_ops,mode_setup_ticks=mode_pe,
                domain_scans=2,gather_staging_bytes=gathered,bitmap_read_bytes=bitmap_reads,
                scatter_spans=spans,scatter_DMA_startup_ticks=scatter_start,
                staging_port_ticks=staging,output_stripes=stripes,
                allocation='Reuses inherited finite I/O slots; no additional full-grid buffer')

def evaluate(inv,m,quota,common,q4fun,full=True):
    records=[];total=collections.Counter(common)
    for op in inv['affines']:
        if op['scope'] not in ('MLP12','patch4'):continue
        k=quota[group(op)]['Q8_complete_tiles']
        more=obligations(op,k,m);p,jobs,reserved=fast_core_pe(op,k,m)
        if not full:
            total['PE_arithmetic_ticks']+=p+more['PE_arithmetic_ticks']
            total['PE_reserved_ticks']+=reserved+more['PE_reserved_ticks']
            continue
        private=copy.deepcopy(op);private['row_domains'][0]['complete_rows']-=810*k
        core=q4fun(private,m,'bitserial','Q4')
        # The inherited Q4 function only splits if low rows>0. If no low rows,
        # its unsplit8-plane branch is exactly the intended all-Q8 domain.
        s.require(core['PE_arithmetic_ticks']==p and core['jobs']==jobs and core['PE_reserved_ticks']==reserved,'Analytic screening equals actual job sums including port reservations')
        expected=8*op['rows']-4*(op['row_domains'][0]['complete_rows']-810*k)
        s.require(core['logical_row_planes']==expected,'Actual domain/plane conservation')
        for key in KEYS:total[key]+=core.get(key,0)+more.get(key,0)
        records.append(dict(name=op['name'],group=group(op),H=op['H'],W=op['W'],T=10,
                            Q8_complete_tiles=k,rows=op['rows'],restored_rows=810*k,
                            tail_rows=op['rows']-op['row_domains'][0]['complete_rows'],
                            high8_rows=op['rows']-op['row_domains'][0]['complete_rows']+810*k,
                            row_planes=expected,core=core,mixed_obligations=more))
    return dict(summary=dict(total),records=records)

def build():
    s.require(s.sha(OLD/'unified_service.py')==SOURCE_SHA and s.sha(OLD/'UNIFIED_SERVICE_V1.json')==RESULT_SHA,'Frozen source/results')
    raw=s.read(OLD/'UNIFIED_SERVICE_V1.json');inv=raw['inventory'];m=s.Machine();q4fun,_=q.build_private_function()
    s.require(max(12*b['channels']*4 for b in inv['BN'])+2048<=m.stats_RF_bytes,'Existing stats RF fits BN carry plus explicit metadata reservation')
    pick=lambda mode:next(x for x in raw['results'] if x['scope']=='MLP12_patch4' and x['implementation']==mode)
    ordinary={k:pick(k)['summary'] for k in ('native','Q8_strong')}
    q4=s.read(OLD/'Q4_COST_ADDENDUM_V1.json')['variants'][0]
    common=q4['common'];geoms={n:geometry(inv,n=='crop_center9') for n in ('full9','crop_center9')}
    ns={v['complete_tiles'] for geom in geoms.values() for v in geom.values()}
    fractions_list=sorted({fractions.Fraction(k,n) for n in ns for k in range(n+1)})
    reserve=fractions.Fraction(RESERVE_NUM,RESERVE_DEN)
    budgets={metric:min(fractions.Fraction(9,10)*v[metric] for v in ordinary.values())-reserve for metric in ('PE_arithmetic_ticks','PE_reserved_ticks')}
    screening=[]
    for f in fractions_list:
        qq=quotas(geoms['full9'],f)
        totals=evaluate(inv,m,qq,common,q4fun,False)['summary']
        screening.append(dict(fraction=[f.numerator,f.denominator],**{k:totals[k] for k in budgets},passes_reserve=all(totals[k]<=budgets[k] for k in budgets)))
    # Enumerate every discrete breakpoint, rather than assume monotonic drains.
    admitted=[i for i,x in enumerate(screening) if x['passes_reserve']]
    s.require(admitted,'Even zero-restoration with paid obligations cannot fit reserved budget')
    chosen=max(admitted);f=fractions_list[chosen]
    quota={n:quotas(v,f) for n,v in geoms.items()}
    actual=evaluate(inv,m,quota['full9'],common,q4fun)
    comparisons={}
    for name,b in ordinary.items():
        comparisons[name]={}
        for metric,target in (('PE_arithmetic_ticks',.10),('PE_reserved_ticks',.10),('ticks',.08)):
            c=actual['summary'][metric]
            comparisons[name][metric]=dict(baseline=b[metric],candidate=c,reduction_fraction=1-c/b[metric],target=target,
                                         additional_allowance_at_target=(1-target)*b[metric]-c)
    s.require(all(x['ticks']['reduction_fraction']>=.08 for x in comparisons.values()),'Selected discrete PE point also meets declared8percent service condition')
    contract=dict(schema='flow_r30_mask_contract_v1',seed=20261019,tile=9,tile_size=9,tail_mode='Q8',
                  members_sha256='40eba2514b56fd33a910fda4e37d4f86e80431ffe8149efc5f79c18827991e7b',
                  training_selection='sha256_seed_step_group_row_col_topk_v1',
                  common_fraction=dict(numerator=f.numerator,denominator=f.denominator,value=float(f)),
                  quota_rule='floor(common_fraction * number_of_complete_9x9_tiles); no minimum-one exception',
                  geometries=quota,evaluation_tiles={g:uniform_tiles(v) for g,v in quota['full9'].items()},
                  evaluation_rule='Centered equally spaced row-major tile-index strata floor((2j+1)*N/(2k)); static, no scores',
                  training_rule='Choose quota tiles by SHA256(seed,step,group,row,col); serialization frozen by implementation; no GT',
                  shared_MLP_FC1_FC2_mask=True,true_tails='Always Q8; selection bitmap covers complete tiles only',
                  geometry_sources=dict(full='FrozenR26 actual shapes in R27 inventory',crop='Same graph downsample lattice for fixed288x384 input; 3/5 spatial size'),
                  selection_cost_basis='Fixed R27 primary machine; greatest discrete common fraction under both ordinary baselines and both arithmetic/reserved10percentPE bounds with the same native1percent reserve; service8percent checked',
                  future_predictor_allowance=dict(numerator=RESERVE_NUM,denominator=RESERVE_DEN,ticks=float(reserve),measured_cost=False),
                  model_source_sha256=s.sha(__file__))
    return contract,dict(schema='flow_r30_discrete_mixed_cost_v1',status='CONDITIONAL_UNCALIBRATED_OPERATOR_SERVICE',
        source_sha256={str(OLD/n):s.sha(OLD/n) for n in ('unified_service.py','UNIFIED_SERVICE_V1.json','Q4_COST_ADDENDUM_V1.json','SERVICE_FREEZE_V1.json')},
        machine=dataclasses.asdict(m),quota_fraction=contract['common_fraction'],ordinary=ordinary,common=common,
        actual=actual,comparisons=comparisons,future_allowance=contract['future_predictor_allowance'],
        screening=screening,selected_breakpoint=chosen,next_breakpoint=screening[chosen+1] if chosen+1<len(screening) else None,
        optimizer_updates=0,model_forwards=0,GPU_used=False,GT_read=False,
        limits=['Whole-net totals add independent operator schedules, not a validated topological DAG cycle trace.',
                'Fixed resource ratio, packed bit storage, FP32 BN, conv addressing and physical bank placement remain uncalibrated.',
                'Extra staging/control is a conservative constructive allowance; no unmodeled free scorer or free bitmap is credited.',
                'Full spatial activation/currentBN/PSN domains remain. All8 weight planes are fetched because tails requireQ8.',
                'Train SHA layout generation is offline experiment orchestration; no runtime selector is implemented.',
                'The nonzero future allowance is a planning reserve, not actual predictor cost or a quality-qualified gain.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    paths=[a.output_dir/'MASK_CONTRACT_V1.json',a.output_dir/'MIXED_COST_V1.json']
    s.require(not any(x.exists() for x in paths),'Exclusive outputs')
    contract,result=build()
    for path,data in zip(paths,(contract,result)):path.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(fraction=contract['common_fraction'],comparisons=result['comparisons'],quotas={n:{g:v['Q8_complete_tiles'] for g,v in x.items()} for n,x in contract['geometries'].items()}),indent=2))
