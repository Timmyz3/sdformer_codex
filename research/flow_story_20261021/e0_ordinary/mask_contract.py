"""E0 reuses external, GT-free R30 quotas and deterministic full-tile masks."""
import hashlib,json
import mixed_integer_core as core
MEMBERS_SHA='40eba2514b56fd33a910fda4e37d4f86e80431ffe8149efc5f79c18827991e7b'
require=core.require

def expected_shapes(geometry):
 require(geometry in ('full9','crop_center9'),'Fixed native geometry')
 h,w=(480,640) if geometry=='full9' else (288,384)
 out={}
 for group in core.GROUPS:
  if group in core.CONVS:out[group]=(h//2,w//2)
  else:
   stage=int(group.split('.layers.')[1].split('.')[0]);out[group]=(h//(4*2**stage),w//(4*2**stage))
 return out

def validate(contract):
 require(contract['schema']=='flow_r30_mask_contract_v1' and contract['seed']==20261019 and contract['tile_size']==9,'Explicit external R30 contract')
 require(contract['tail_mode']=='Q8' and contract['members_sha256']==MEMBERS_SHA,'Same source population and true-tail policy')
 require(contract['training_selection']=='sha256_seed_step_group_row_col_topk_v1','One fixed no-GT training mask distribution')
 require(set(contract['geometries'])=={'full9','crop_center9'},'Both actually executed geometries')
 for geometry,groups in contract['geometries'].items():
  require(set(groups)==set(core.GROUPS),'All16 independent groups; FC pair implicit')
  for group,(h,w) in expected_shapes(geometry).items():
   row=groups[group];k=row['Q8_complete_tiles']
   require(row['H']==h and row['W']==w and type(k) is int and 0<=k<=(h//9)*(w//9),'Actual geometry and bounded external quota')
 require(set(contract['evaluation_tiles'])==set(core.GROUPS),'Explicit fixed full-resolution evaluation layout')
 for group,row in contract['geometries']['full9'].items():
  cells=contract['evaluation_tiles'][group]
  require(isinstance(cells,list) and all(isinstance(x,list) and len(x)==2 and all(type(y) is int for y in x) for x in cells),'Explicit row-column coordinates')
  require(len(cells)==row['Q8_complete_tiles'] and len({tuple(x) for x in cells})==len(cells),'Unique evaluation choices exactly equal quota')
  require(cells==sorted(cells) and all(0<=y<row['H']//9 and 0<=x<row['W']//9 for y,x in cells),'Ordered complete9 tiles only')
 return contract

def mode_for(order,sequences):
 rank=sequences.index(order['sequence']);v=order['sequence_visit_index'];p=order['pass_index']
 require(1<=v<=32 and p in (1,2),'Fixed per-sequence visits/two passes')
 return ('Q4','Q4','Q8','M')[(v-1+rank+2*(p-1))%4]

def coordinates(contract,geometry,*,step=None):
 groups=contract['geometries'][geometry];out={}
 require((geometry=='crop_center9')==(step is not None),'Only training crops have step-dependent masks')
 for group,row in groups.items():
  if step is None:out[group]=contract['evaluation_tiles'][group]
  else:
   require(type(step) is int and 1<=step<=512,'Only fixed training step')
   cells=[[y,x] for y in range(row['H']//9) for x in range(row['W']//9)]
   def key(cell):
    text=json.dumps([contract['seed'],step,group,*cell],separators=(',',':'))
    return hashlib.sha256(text.encode()).hexdigest(),cell
   out[group]=sorted(sorted(cells,key=key)[:row['Q8_complete_tiles']])
 return out

def materialize(contract,geometry,*,step=None):
 import torch
 cells=coordinates(contract,geometry,step=step);masks={}
 for group,row in contract['geometries'][geometry].items():
  mask=torch.zeros(row['H'],row['W'],dtype=torch.bool)
  for y,x in cells[group]:mask[y*9:y*9+9,x*9:x*9+9]=True
  masks[group]=mask
 return masks,dict(geometry=geometry,step=step,selected_tiles=cells,mask_sha256={g:core.digest(v) for g,v in masks.items()})

class BoundCore:
 """Adapter for frozen capture helpers; masks are explicit immutable call context."""
 def __init__(self,masks=None,*,variant="rne4",engine=core):self.masks=masks;self.variant=variant;self.engine=engine
 def __getattr__(self,key):return getattr(self.engine,key)
 def installed(self,model,mode,*,ste=True,reference=False):
  kwargs=dict(masks=self.masks if mode=='M' else None,ste=ste,reference=reference)
  if self.engine is core:kwargs['variant']=self.variant
  return self.engine.installed(model,mode,**kwargs)
