"""Two genuine Q4 states; use frozen R27 math without changing its globals."""
from pathlib import Path
import sys
R27=Path(__file__).resolve().parents[2]/'round27/independent_review'
sys.path.append(str(R27)) # Keep this round's same-named CLI ahead of old helpers.
import analysis_core as old
old.require(Path(old.__file__).resolve()==R27/'analysis_core.py','Actual frozen R27 comparison module')
NEW_STATES=('initial_Q4','recovered_Q4')
require=old.require

def analyze(reference,new_frames):
 require(reference['schema']=='flow_r27_independent_raw_quality_v1' and reference['status']=='RAW_METRICS_COMPLETE','Actual R27 reference quality')
 prior=reference['frames'];require(len(prior)==len(new_frames)==64,'Exactly64 paired frames')
 require([(f['target_id'],f['sequence']) for f in new_frames]==[(f['target_id'],f['sequence']) for f in prior],'Exact old development identity/order')
 require(len({f['target_id'] for f in new_frames})==64,'Unique development identities')
 seqs=list(dict.fromkeys(f['sequence'] for f in prior));require(len(seqs)==8 and all(sum(f['sequence']==s for f in prior)==8 for s in seqs),'Eight sequences x8')
 combined=[]
 for p,n in zip(prior,new_frames):
  require(set(n['states'])==set(NEW_STATES),'Only two actual new Q4 states')
  require(set(p['states'])==set(old.STATES),'Original six states remain genuine prior references')
  for state in NEW_STATES:
   require(set(n['states'][state])==set(old.REGIONS),'All16 new regions')
   for region in old.REGIONS:
    old.validate(n['states'][state][region])
    require(n['states'][state][region]['valid_pixels']==p['states']['original_native'][region]['valid_pixels'],'Same original GT support')
  combined.append(dict(target_id=n['target_id'],sequence=n['sequence'],states={**p['states'],**n['states']}))
 scopes=['overall']+seqs
 summaries={state:{s:{region:old.aggregate([f['states'][state][region] for f in combined if s=='overall' or f['sequence']==s]) for region in old.REGIONS} for s in scopes} for state in old.STATES+NEW_STATES}
 require(all(summaries[s]==reference['summaries'][s] for s in old.STATES),'Old six summaries unchanged and independently reconstructed from fixed scalars')
 pairs=[('initial_Q4','original_native'),('recovered_Q4','original_native'),('recovered_Q4','recovered_native'),('recovered_Q4','initial_Q4'),('recovered_Q4','recovered_Q8')]
 comparisons={c+'__vs__'+r:old.comparison(combined,summaries,c,r,seqs) for c,r in pairs}
 passed=all(comparisons['recovered_Q4__vs__'+r]['aggregate_pass'] for r in ('original_native','recovered_native'))
 return dict(schema='flow_r28_two_state_scalar_quality_v1',frames=new_frames,summaries={s:summaries[s] for s in NEW_STATES},comparisons=comparisons,
  inherited_R27_native_recovery_stable=reference['native_recovery_stable'],inherited_R27_Q8_dual_reference_pass=reference['Q8_dual_reference_pass'],Q4_uniform_dual_reference_pass=passed,
  configuration_decision='UNIFORM_Q4_AGGREGATE_PASS_REVIEW_FLAGS_SEPARATE' if passed else 'UNIFORM_Q4_NOT_QUALIFIED_MIXED_FAMILY_UNDECIDED',
  new_quality_prediction_count=128,new_frame_region_cells=2048,new_summary_cells=288,old_quality_modified=False,selector_learning_demonstrated=False,
  scope='Two newly measured Q4 states on exposed development64. R27 reference scalars are reused with source binding, not new predictions. Uniform failure is not proof that all mixed actions fail.')
