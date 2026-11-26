"""Independent E0 geometry, complete-tile mask and separate-domain integer work verification.

No model/GT import and no production core imports. Checks actual CUDA ledger;
the row-plane counts are not claimed to be hardware cycle measurements.
"""
import hashlib
import json
import math
import numpy as np

PREFIX='sttmultires_unet.encoders.swin3d.'
LINEARS={PREFIX+f'layers.{s}.swin_blocks.{b}.mlp.{f}'
         for s,d in enumerate((2,2,6,2)) for b in range(d) for f in ('fc1','fc2')}
CONVS={PREFIX+f'patch_embed.residual_encoding.resblocks.{b}.conv{j}.0' for b in (0,1) for j in (1,2)}
TARGETS=LINEARS|CONVS
GROUPS={n.rsplit('.',1)[0] if n in LINEARS else n for n in TARGETS}

def require(ok,msg):
    if not ok:raise ValueError(msg)

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def expected_selections(contract,geometry,step=None):
    rows=contract['geometries'][geometry]
    require(set(rows)==GROUPS and contract['seed']==20261019 and contract['tile_size']==9,'Exact mask groups and declared seed/tile')
    selected={}
    for group,r in rows.items():
        h,w,k=r['H'],r['W'],r['Q8_complete_tiles']
        require(type(k) is int and 0<=k<=(h//9)*(w//9),'Bounded complete tile quota')
        if step is None:cells=contract['evaluation_tiles'][group]
        else:
            require(1<=step<=512 and geometry=='crop_center9','Training masks use declared step only')
            keys=[]
            for y in range(h//9):
                for x in range(w//9):
                    body=json.dumps([20261019,step,group,y,x],separators=(',',':')).encode()
                    keys.append((hashlib.sha256(body).hexdigest(),[y,x]))
            cells=sorted(z[1] for z in sorted(keys)[:k])
        require(cells==sorted(cells) and len(cells)==k and len({tuple(x) for x in cells})==k,'Unique canonical complete-tile choices')
        require(all(len(c)==2 and all(type(v) is int for v in c) and 0<=c[0]<h//9 and 0<=c[1]<w//9 for c in cells),'Only legal whole tiles')
        selected[group]=cells
    return selected

def check_calls(calls,work,mode,geometry,contract,*,step=None,dense=False,variant='rne4'):
    require(mode in ('native','Q4','Q8','M'),'Declared R30 mode')
    require(set(calls)==TARGETS and set(work)==(set() if mode=='native' else TARGETS),'Actual28 target calls and work scope')
    geo_name='full9' if geometry==(480,640) else 'crop_center9'
    require(geometry in ((480,640),(288,384)),'Native full or fixed crop geometry')
    selections=expected_selections(contract,geo_name,step) if mode=='M' else None
    total=dict(row_planes=0,kernel_calls=0,explicit_padding_rows=0)
    for name,records in calls.items():
        require(len(records)==1,'Every affine executes once')
        call=records[0];ins,outs=call['input_shape'],call['output_shape']
        require(len(ins)==len(outs)==5 and ins[0]==outs[0]==10 and ins[1]==outs[1]==1,'Original fullT10/B1')
        if name in LINEARS:
            t,b,h,w,ci=ins;co=outs[-1];require(outs[:4]==ins[:4],'All FC output positions')
            s=int(name.split('.layers.')[1].split('.')[0]);c=96*2**s
            require((h,w)==(geometry[0]//(4*2**s),geometry[1]//(4*2**s)),'Native stage spatial grid')
            require((ci,co)==((c,4*c) if name.endswith('.fc1') else (4*c,c)),'Native FC input/output width');k=ci
            group=name.rsplit('.',1)[0]
        else:
            t,b,ci,h,w=ins;co=outs[2];require(outs[:2]+outs[3:]==ins[:2]+ins[3:],'All convolution positions')
            require((h,w,ci,co)==(geometry[0]//2,geometry[1]//2,96,96),'Native patch conv geometry');k=9*ci;group=name
        if mode=='native':continue
        require(call['module_class']=='STEAffine','Actual installed integer wrapper')
        a=work[name];rows=t*b*h*w;tail2=np.ones((h,w),dtype=bool);tail2[:h//9*9,:w//9*9]=False
        chosen=np.zeros((h,w),dtype=bool);cells=selections[group] if mode=='M' else []
        if mode=='M':
            q=contract['geometries'][geo_name][group]
            require((q['H'],q['W'])==(h,w),'Contract geometry is real executed geometry')
            for y,x in cells:chosen[y*9:y*9+9,x*9:x*9+9]=True
        high2=np.ones_like(tail2) if mode=='Q8' else tail2|chosen
        high=np.broadcast_to(high2,(t,b,h,w)).reshape(-1);tail=t*b*int(tail2.sum());nh=int(high.sum())
        require((a['mode'],a['reference'],a['rows'],a['Cin'],a['Cout'])==(mode,dense,rows,k,co),'Actual mode/domain/size')
        require((a['tail_rows'],a['complete_rows'],a['high8_rows'],a['low4_rows'],a['restored_complete_rows'])==(tail,rows-tail,nh,rows-nh,nh-tail),'True tails and full9 restored rows')
        require(a['high8_rows_sha256']==digest(high) and a['selected_complete_tiles']==cells,'Exact actual row membership and whole tile list')
        require(a['selected_spatial_mask_sha256']==(digest(chosen) if mode=='M' else None),'Exact independent spatial mask bytes')
        require(a['geometric_shape']==[t,b,h,w] and a['row_order']=='T,B,output_y,output_x','Actual row order preserves all time/spatial positions')
        require(a['binary_verified'] and a['int32_accumulation'] and a['scale_applications']==1 and a['spatial_rows_retained'] and a['full_T_and_channels'],'Original numeric contract')
        require(a['software_plane_dtype']=='int8' and a['quantization_device']=='cpu' and a['cycle_claim'] is False,'Declared non-cycle execution evidence')
        require(variant in ('rne4','midpoint'),'Reviewed ordinary variant')
        rep=a['representation'];require(rep['variant']==variant and rep['q8_shape']==[co,k] and rep['rounding']=='ties_to_even' and rep['high_scale']=='per_output_maxabs/127','Ordinary high encoding')
        require(rep['low_representation_generated'] is (mode!='Q8') and a['scale_variants']==(2 if mode!='Q8' and variant=='rne4' else 1),'Separate scales only when RNE low domain used')
        require(a['scale_applications_per_output']==1 and a['canonical_scatter_before_BN'] and a['independent_domain_execution'],'Canonical complete output after domain execution')
        offset=8 if mode!='Q8' and variant=='midpoint' else 0
        require(a['midpoint_correction']['offset']==offset and a['midpoint_correction']['rows']==(rows-nh if offset and not dense else 0),'Actual correction row count')
        if mode!='Q8':
            if variant=='rne4':
                require(a['low_bit_rule']==rep['low_rule']=='independent_rne4_maxabs7_clamp_minus8_plus7' and rep['low_scale']=='per_output_maxabs/7/16' and rep['low_encoding']=='signed_q4_left_shift4' and rep['low_offset_per_active_input']==0,'Independent RNE4 representation')
            else:
                require(a['low_bit_rule']==rep['low_rule']=='drop4_plus_8_popcount' and rep['low_scale']=='same_q8_scale' and rep['low_encoding']=='signed_q8_drop4' and rep['low_offset_per_active_input']==8,'Midpoint ordinary representation')
        else:require(a['low_bit_rule']=='drop4','Q8 legacy low-rule placeholder has no low-domain work')
        limit=max(24,min(65536,8*1024*1024//co));limit=limit//8*8;ledger=[]
        for scope,n,bits in (('high8',nh,range(8)),('low4',rows-nh,range(4,8))):
            for start in range(0,n,limit):
                count=min(limit,n-start);pad=max(24,math.ceil(count/8)*8)-count
                if dense:
                    ledger.append(dict(kind='dense_integer_reference',bit=None,rows=count,padding_rows=pad,scope=scope,coefficient=None))
                else:
                    for bit in bits:
                        ledger.append(dict(kind='binary_weight_plane',bit=bit,rows=count,padding_rows=pad,scope=scope,coefficient=-128 if bit==7 else 1<<bit))
        if dense:require(a['logical_row_plane_work'] is None,'Dense reference is not serial work')
        else:
            plane=sum(x['rows'] for x in ledger)
            require(a['logical_row_plane_work']==plane==8*nh+4*(rows-nh),'Exclusive high8/low4 work')
            total['row_planes']+=plane
        require(a['actual_integer_kernel_calls']==len(ledger) and a['explicit_padded_rows']==sum(x['padding_rows'] for x in ledger),'Exact kernel count/padding')
        if 'kernel_calls' in a:require(a['kernel_calls']==ledger,'Every actual kernel bit/scope/chunk/sign/padding')
        total['kernel_calls']+=len(ledger);total['explicit_padding_rows']+=sum(x['padding_rows'] for x in ledger)
    return total
