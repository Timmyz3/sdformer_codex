"""Descriptive same-Q4-W Q8 comparison. No new launch or mixed-family gate."""
from pathlib import Path
import sys
R27=Path(__file__).resolve().parents[2]/'round27/independent_review'
sys.path.append(str(R27))
import analysis_core as old
old.require(Path(old.__file__).resolve()==R27/'analysis_core.py','Original fixed comparison mathematics')
STATE='Q8_on_recovered_Q4'
require=old.require

def analyze(r27,r28,new):
 require(r27['schema']=='flow_r27_independent_raw_quality_v1' and r28['schema']=='flow_r28_independent_raw_quality_v1' and r27['status']==r28['status']=='RAW_METRICS_COMPLETE','Actual independently measured reference quality')
 identity=lambda rows:[(f['target_id'],f['sequence']) for f in rows]
 require(len(new)==len({f['target_id'] for f in new})==64 and identity(new)==identity(r27['frames'])==identity(r28['frames']),'Exact64 shared development identities')
 seqs=list(dict.fromkeys(f['sequence'] for f in new));require(len(seqs)==8 and all(sum(f['sequence']==s for f in new)==8 for s in seqs),'8sequence x8')
 combined=[]
 for a,b,c in zip(r27['frames'],r28['frames'],new):
  require(set(a['states'])==set(old.STATES) and set(b['states'])=={'initial_Q4','recovered_Q4'} and set(c['states'])=={STATE},'Actual reference and only one new state')
  states={**a['states'],**b['states'],**c['states']}
  for state,regions in states.items():
   require(set(regions)==set(old.REGIONS),'Complete16 regions')
   for region in old.REGIONS:
    old.validate(regions[region]);require(regions[region]['valid_pixels']==states['original_native'][region]['valid_pixels'],'Same GT support')
  combined.append(dict(target_id=c['target_id'],sequence=c['sequence'],states=states))
 states=old.STATES+('initial_Q4','recovered_Q4',STATE);scopes=['overall']+seqs
 summaries={s:{scope:{region:old.aggregate([f['states'][s][region] for f in combined if scope=='overall' or f['sequence']==scope]) for region in old.REGIONS} for scope in scopes} for s in states}
 require(all(summaries[s]==r27['summaries'][s] for s in old.STATES) and all(summaries[s]==r28['summaries'][s] for s in ('initial_Q4','recovered_Q4')),'Prior summaries unchanged')
 comparisons={STATE+'__vs__'+ref:old.comparison(combined,summaries,STATE,ref,seqs) for ref in ('recovered_Q4','original_native','recovered_native','recovered_Q8')}
 for c in comparisons.values():c['thresholds_are_descriptive_only']=True;c['new_qualification_gate']=False
 return dict(schema='flow_r29_same_weight_scalar_quality_v1',frames=new,summaries={STATE:summaries[STATE]},comparisons=comparisons,
  new_quality_predictions=64,new_frame_region_cells=1024,new_summary_cells=144,optimizer_updates=0,
  qualification_gate_added=False,old_quality_modified=False,selector_learning_demonstrated=False,
  scope='Same Q4-trained W, full-Q8 high-cost diagnostic. Main comparison reference is same-W Q4. Different-W recovered-Q8 comparison is descriptive. No conclusion about all local/mixed actions and no automatic experiment.')
