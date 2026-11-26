"""Independent static360 geometry/materializer/work checks; no GPU entry."""
import hashlib,json,math
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(ok,why):
    if not ok:raise ValueError(why)
def tile_order():return list(range(302,453))+list(range(209))

def validate(c,core):
    require(c['schema']=='flow_r35_static360_contract_v1','R35-specific contract')
    require(c['tile_ids_selection_order']==tile_order() and c['selected_complete_tiles_per_patch']==360,'Exact unique360 selection')
    require(len(set(tile_order()))==360 and all(0<=i<910 for i in tile_order()),'Complete26x35 IDs')
    require(set(c['geometries'])==set(core.TARGETS) and c['family_targets']==list(core.CONVS),'28 geometry and four patch authority')
    for name,g in c['geometries'].items():
        if name in core.CONVS:require(g==dict(T=10,H=240,W=320,K=864,Cout=96),'Fixed actual patch geometry')
        else:
            stage=int(name.split('.layers.')[1].split('.')[0]);k=96*2**stage if name.endswith('.fc1') else 384*2**stage
            co=384*2**stage if name.endswith('.fc1') else 96*2**stage
            require(g==dict(T=10,H=480//(4*2**stage),W=640//(4*2**stage),K=k,Cout=co),'MLP unchanged geometry')
    require(c['core_sha256']=='ce07ae04c9f3d58738ae134cdc1c8276542a41a04584720b5d69359e220d1f9f','Frozen RNE core')
    require(c['membership_sha256']=='d860612a3ae3e4620fa2a9a616df9f8e93774efadcf7dc564881e27b7e7da8cc','Same exposed120')
    require(len(c['target_ids'])==len(set(c['target_ids']))==120,'120 exact identities')
    require((c['formal_forwards'],c['engineering_forwards'],c['max_forwards'],c['wall_cap_seconds'])==(120,4,124,1200),'Hard124 and20min')
    require(c['added_four_patch_core_ticks_vs_Q4']==4*360*81*10*4*7*6==195955200,'Actual replacement arithmetic')
    return c

def materialize(c,core,selected=True):
    import torch
    validate(c,core);masks={}
    for name,g in c['geometries'].items():masks.setdefault(core.group_name(name),torch.zeros(g['H'],g['W'],dtype=torch.bool))
    if selected:
        for name in core.CONVS:
            for tid in tile_order():
                y,x=divmod(tid,35);masks[name][y*9:y*9+9,x*9:x*9+9]=True
    require(set(masks)==set(core.GROUPS),'All16 mask groups')
    return masks,dict(label='static360' if selected else 'background_Q4',mode='M' if selected else 'Q4',
        selected_tiles={n:int(m.sum())//81 for n,m in masks.items()},mask_sha256={n:core.digest(m) for n,m in masks.items()},
        tile_ids_selection_order=tile_order() if selected else [],tile_ids_canonical_spatial_order=sorted(tile_order()) if selected else [],
        added_four_patch_core_ticks_vs_Q4=195955200 if selected else 0)

class BoundCore:
    def __init__(self,core,c,selected=True):
        self.engine=core;self.masks,self.mask_record=materialize(c,core,selected);self.mode=self.mask_record['mode']
    def __getattr__(self,n):return getattr(self.engine,n)
    def installed(self,model,mode,*,ste=False,reference=False):
        require(not ste and mode==self.mode,'Inference-only new static action')
        return self.engine.installed(model,mode,masks=self.masks if mode=='M' else None,ste=False,reference=reference,variant='rne4')

def validate_work(work,calls,bns,c,core,selected=True):
    validate(c,core)
    require(set(work)==set(calls)==set(core.TARGETS) and all(v==1 for v in calls.values()),'All28 exactly once')
    require(len(bns)==78 and all(v==1 for v in bns.values()),'All78 currentBN exactly once')
    ticks=0
    for name,w in work.items():
        g=c['geometries'][name];rows=g['T']*g['H']*g['W'];complete=g['T']*(g['H']//9*9)*(g['W']//9*9)
        restored=360*81*g['T'] if selected and name in core.CONVS else 0
        expected_tiles=[list(divmod(tid,35)) for tid in sorted(tile_order())] if restored else []
        require(w['rows']==rows and w['Cin']==g['K'] and w['Cout']==g['Cout'] and w['tail_rows']==rows-complete and
            w['restored_complete_rows']==restored and w['high8_rows']==rows-complete+restored and w['low4_rows']==complete-restored and
            w['geometric_shape']==[10,1,g['H'],g['W']] and w['selected_complete_tiles']==expected_tiles,'Actual360 geometry/domain '+name)
        require(w['representation']['variant']=='rne4' and w['scale_variants']==2 and w['canonical_scatter_before_BN'] and
            not w['reference'] and w['independent_domain_execution'] and w['binary_verified'],'Original RNE domain semantics '+name)
        require(w['logical_row_plane_work']==8*w['high8_rows']+4*w['low4_rows'],'Logical row planes')
        for bit in range(8):
            expected=w['high8_rows']+(w['low4_rows'] if bit>=4 else 0)
            require(sum(k['rows'] for k in w['kernel_calls'] if k['bit']==bit)==expected,'Actual accepted rows each plane')
        ticks+=w['logical_row_plane_work']*math.ceil(g['K']/128)*math.ceil(g['Cout']/16)
    return dict(affine_count=28,BN_count=78,selected_complete_tiles_per_patch=360 if selected else 0,
        added_four_patch_core_ticks_vs_Q4=195955200 if selected else 0,all28_logical_core_ticks=ticks,
        logical_row_plane_work=sum(w['logical_row_plane_work'] for w in work.values()),net_cycle_claim=False)
