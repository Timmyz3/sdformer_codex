"""R30 four new measured states; unchanged R27 math, explicit old references."""
from pathlib import Path
import sys

R27=Path(__file__).resolve().parents[3]/'flow_story_20261019/round27/independent_review'
sys.path.append(str(R27))
import analysis_core as old
old.require(Path(old.__file__).resolve()==R27/'analysis_core.py','Frozen R27 comparison source')
require=old.require
NEW_STATES=('initial_M','joint_Q4','joint_Q8','joint_M')
REFERENCES=old.STATES+('initial_Q4','recovered_Q4')
PRIMARY_REFERENCES=('original_native','recovered_native')

def analyze(r27,r28,new):
    require(r27['schema']=='flow_r27_independent_raw_quality_v1'
            and r28['schema']=='flow_r28_independent_raw_quality_v1'
            and r27['status']==r28['status']=='RAW_METRICS_COMPLETE',
            'Actual independently measured references')
    identity=lambda rows:[(f['target_id'],f['sequence']) for f in rows]
    require(len(new)==len({f['target_id'] for f in new})==64
            and identity(new)==identity(r27['frames'])==identity(r28['frames']),
            'Exact exposed64 development identities/order')
    seqs=list(dict.fromkeys(f['sequence'] for f in new))
    require(len(seqs)==8 and all(sum(f['sequence']==s for f in new)==8 for s in seqs),
            'Eight sequences x8')
    combined=[]
    for a,b,c in zip(r27['frames'],r28['frames'],new):
        require(set(a['states'])==set(old.STATES)
                and set(b['states'])=={'initial_Q4','recovered_Q4'}
                and set(c['states'])==set(NEW_STATES),'Eight old and four genuine new states')
        states={**a['states'],**b['states'],**c['states']}
        for state,regions in states.items():
            require(set(regions)==set(old.REGIONS),'Complete16 regions')
            for region in old.REGIONS:
                old.validate(regions[region])
                require(regions[region]['valid_pixels']==states['original_native'][region]['valid_pixels'],
                        'Same GT support for every state')
        combined.append(dict(target_id=c['target_id'],sequence=c['sequence'],states=states))
    scopes=['overall']+seqs
    summaries={s:{scope:{r:old.aggregate([f['states'][s][r] for f in combined
                    if scope=='overall' or f['sequence']==scope]) for r in old.REGIONS}
                    for scope in scopes} for s in REFERENCES+NEW_STATES}
    require(all(summaries[s]==r27['summaries'][s] for s in old.STATES)
            and all(summaries[s]==r28['summaries'][s] for s in ('initial_Q4','recovered_Q4')),
            'Historical summaries unchanged')
    pairs=[(s,r) for s in NEW_STATES for r in PRIMARY_REFERENCES]
    pairs += [('joint_M','initial_M'),('joint_Q4','initial_Q4'),('joint_Q8','initial_Q8')]
    pairs += [(s,r) for s in ('joint_Q4','joint_Q8','joint_M')
              for r in ('recovered_Q4','recovered_Q8','recovered_Q6')]
    pairs += [('joint_M','joint_Q4'),('joint_M','joint_Q8'),('joint_Q8','joint_Q4')]
    require(len(pairs)==len(set(pairs))==23,'Predeclared complete comparison matrix')
    comparisons={}
    for c,r in pairs:
        result=old.comparison(combined,summaries,c,r,seqs)
        primary=c=='joint_M' and r in PRIMARY_REFERENCES
        result.update(primary_M_quality_gate=primary,
                      thresholds_are_descriptive_only=not primary,
                      Q8_is_extra_M_veto=False)
        comparisons[c+'__vs__'+r]=result
    passes={s:all(comparisons[s+'__vs__'+r]['aggregate_pass'] for r in PRIMARY_REFERENCES)
            for s in NEW_STATES}
    return dict(schema='flow_r30_joint_mode_scalar_quality_v1',frames=new,
        summaries={s:summaries[s] for s in NEW_STATES},comparisons=comparisons,
        descriptive_native_dual_reference_pass=passes,M_native_dual_reference_pass=passes['joint_M'],
        configuration_decision=('FIXED_M_AGGREGATE_PASS_COST_AND_REVIEW_FLAGS_SEPARATE'
          if passes['joint_M'] else 'FIXED_M_NOT_QUALIFIED_OTHER_MIXED_UNDECIDED'),
        Q8_is_extra_M_veto=False,hardware_qualification_evaluated=False,
        new_quality_predictions=256,new_frame_region_cells=4096,new_summary_cells=576,
        old_quality_modified=False,selector_learning_demonstrated=False,
        scope='One ordinary512-update joint-mode recovery; exposed dev64, no new confirmation. '
              'Only fixed M has the predeclared dual-native quality gate; full Q8 is compatibility '
              'diagnosis. All historical reference scalars explicitly reused. No local signed-gain '
              'or flow-specific superiority claim and no automatic next experiment.')
