#!/usr/bin/env python3
"""Standard-library fixture checks, no model/data arrays or genuine new AEE."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import summarize_recovery as s

HERE=Path(__file__).resolve().parent
DEV=json.loads((HERE.parent/'breadth/gate_dev32.json').read_text())['target_ids']
CHECK=json.loads((HERE/'manifests/check64.json').read_text())['target_ids']
FIT=json.loads((HERE.parents[1]/'flow_story_20261011/preflight/data_protocol/artifacts/bn_calibration_schedule.json').read_text())['stages']['initial']['target_ids']


def metric_fixture(ids, base):
    frames=[]
    for i,target in enumerate(ids):
        regions={}
        for key,n,offset in [('all_valid',2,0),('B',1,1),('L',i%2,10),('B_unknown',1,3),('voxel_zero',1,4)]:
            regions[key]={'valid_pixels':n,'error_sum':n*(base+offset),'AEE':base+offset if n else None}
        frames.append({'target_id':target,'sequence':target.rsplit('_',1)[0],'regions':regions})
    def aggregate(part):
        regions={}
        for key in frames[0]['regions']:
            vals=[f['regions'][key] for f in part]; good=[v for v in vals if v['valid_pixels']]
            pixels=sum(v['valid_pixels'] for v in vals);error=sum(v['error_sum'] for v in vals)
            regions[key]={'frames_with_region':len(good),'empty_region_frames':len(vals)-len(good),
                'valid_pixels':pixels,'error_sum':error,'frame_mean_AEE':sum(v['AEE'] for v in good)/len(good) if good else None,
                'pixel_weighted_AEE':error/pixels if pixels else None}
        return {'frame_count':len(part),'regions':regions}
    summary=aggregate(frames)
    summary['per_sequence']={seq:aggregate([f for f in frames if f['sequence']==seq]) for seq in {f['sequence'] for f in frames}}
    return frames,summary


def candidate(arm,stage):
    checkpoint=s.digest([arm,stage]); visits=0 if stage=='initial' else 4573
    value={'schema':'flow_recovery_candidate_evaluation_v1','status':'CANDIDATE_EVALUATION_COMPLETE',
        'stage':stage,'geometry':'full9','identity':{'arm':arm,'source_checkpoint_sha256':s.START_SHA,
            'final_valid_used':False,'candidate_checkpoint_sha256':checkpoint,
            'candidate_sidecar':{'checkpoint_sha256':checkpoint},'recovery_contract_sha256':s.digest(arm),
            'architecture_sha256':s.digest(arm),'recovery_training':{'visited_targets':visits,
            'completed_optimizer_steps':visits-2 if visits else 0,'zero_valid_crops':2 if visits else 0}},
        'populations':{'gate_dev32':list(DEV)},'candidate_BN_calibration_completed':True,
        'all_evaluation_BN_states_unchanged':True,
        'calibration':{'non_BN_state_unchanged':True,'count_per_BN':176,'fit_ids':FIT},
        'native_preflight':{'data_contract':{'synthetic':True},'source_sha256':{'synthetic':'a'*64},
            'regions_sha256':'b'*64},'evaluations':{}}
    if stage=='endpoint':value['populations']['check64']=list(CHECK)
    for population,ids in value['populations'].items():
        for bn in s.BNS:
            base=1.+(3. if stage=='endpoint' else 0)+(10. if bn==s.BNS[1] else 0)+(20. if population=='check64' else 0)+s.ARMS.index(arm)/10.
            frames,summary=metric_fixture(ids,base)
            value['evaluations'][bn+'__'+population]={'status':'COMPLETE','BN_policy':bn,
                'population':population,'frames':frames,'summary':summary}
    return value


def write(path,value): path.write_text(json.dumps(value,allow_nan=False)+'\n')


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='flow_summary_cpu_');self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def populate(self):
        write(self.root/'queue_status.json',{'schema':'flow_recovery_queue_v1','status':'COMPLETE_BASELINE_REMAINDER_LAUNCHED',
            'experiment_status':'FOUR_ARMS_COMPLETE','active_job':None})
        write(self.root/'smoke_audit.json',{'status':'ALL_FOUR_ARMS_PASS','arms':[{'arm':a,'updates':8} for a in s.ARMS]})
        write(self.root/'svs_stability.json',{'schema':'flow_recovery_svs_stability_v1','sampling_only_comparison':True})
        for arm in s.ARMS:
            for stage in ('initial','endpoint'):write(self.root/(arm+'_'+stage+'_eval.json'),candidate(arm,stage))
    def test_complete_and_matched_dense_deltas(self):
        self.populate();report=s.summarize(self.root)
        self.assertEqual(report['status'],'RECOVERY_EXPERIMENT_COMPLETE')
        self.assertEqual(len(report['quality_rows']),24)
        for row in report['quality_rows']:
            self.assertEqual(row['delta_status'],'MATCHED_DENSE')
            for region in row['delta_regions'].values():
                self.assertAlmostEqual(region['frame_mean_AEE'],s.ARMS.index(row['arm'])/10.)
            self.assertEqual(row['delta_reference']['BN_policy'],row['BN_policy'])
            self.assertEqual(row['delta_reference']['population'],row['population'])
            self.assertEqual(row['delta_reference']['stage'],row['stage'])
        text=s.markdown(report)
        for word in ('old dev32','check64','P_paper','D_candidate_CMA176','B_unknown','135930','4,573','像素加权'):
            self.assertIn(word,text)
    def test_partial_source_excluded_and_dense_never_borrowed(self):
        self.populate()
        (self.root/'dense_initial_eval.json').unlink()
        partial=candidate('svs_hidden25','endpoint');partial['status']='RUNNING'
        write(self.root/'svs_hidden25_endpoint_eval.json',partial)
        report=s.summarize(self.root)
        self.assertEqual(report['status'],'RECOVERY_EVIDENCE_INCOMPLETE')
        self.assertEqual(report['completion']['complete_candidate_files'],6)
        self.assertFalse(any(r['arm']=='svs_hidden25' and r['stage']=='endpoint' for r in report['quality_rows']))
        for row in report['quality_rows']:
            if row['stage']=='initial':
                self.assertEqual(row['delta_status'],'PENDING_MATCHED_COMPLETE_DENSE')
                self.assertEqual(row['delta_regions'],{})
    def test_invalid_endpoint_metric_and_member_refused(self):
        self.populate()
        cases=[('lowrank_param25',lambda v:v['identity']['recovery_training'].update(visited_targets=4572)),
            ('svs_hidden25',lambda v:v['evaluations']['P_paper__check64']['summary']['regions']['all_valid'].update(frame_mean_AEE=123)),
            ('magnitude_hidden25',lambda v:v['populations']['check64'].__setitem__(0,DEV[0]))]
        for arm,mutate in cases:
            value=candidate(arm,'endpoint');mutate(value);write(self.root/(arm+'_endpoint_eval.json'),value)
        report=s.summarize(self.root)
        self.assertEqual(len(report['pending_candidates']),3)
        self.assertTrue(all(r['status']=='INVALID_COMPLETE_RESULT' for r in report['pending_candidates']))
    def test_changed_region_protocol_refuses_delta_without_hiding_raw_AEE(self):
        self.populate();value=candidate('lowrank_param25','endpoint')
        value['native_preflight']['regions_sha256']='c'*64
        write(self.root/'lowrank_param25_endpoint_eval.json',value)
        report=s.summarize(self.root)
        rows=[r for r in report['quality_rows'] if r['arm']=='lowrank_param25' and r['stage']=='endpoint']
        self.assertEqual(len(rows),4)
        self.assertTrue(all(r['delta_status']=='REFUSED_BINDING_OR_GT_COVERAGE_MISMATCH' and not r['delta_regions'] for r in rows))
        self.assertFalse(report['completion']['matched_dense_deltas_complete'])
        self.assertEqual(report['status'],'RECOVERY_EVIDENCE_INCOMPLETE')
    def test_empty_run_CLI_outputs_only_pending(self):
        with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(s.main(['--run-dir',str(self.root)]),0)
        result=json.loads((self.root/'summary/recovery_summary.json').read_text())
        self.assertEqual(result['quality_rows'],[])
        self.assertEqual(len(result['pending_candidates']),8)
        self.assertFalse(result['scope']['hardware_cycle_or_energy_claim'])
        self.assertIn('待完成',(self.root/'summary/RECOVERY_SUMMARY.md').read_text())
    def test_existing_frozen_metric_JSON_schema_accepted(self):
        path=HERE.parent/'breadth/results_calibrated42/calibrated42_results.json'
        raw=path.read_bytes()
        self.assertEqual(__import__('hashlib').sha256(raw).hexdigest(),
            '440b97e44058751db570a340635afdd0709a1ac00b76e9fadd3f67c11afe0032')
        group=json.loads(raw)['arms']['dense_P_paper']
        summary,coverage=s.metric_summary(group,DEV)
        self.assertEqual(summary['frame_count'],32)
        self.assertEqual(summary['regions']['all_valid']['valid_pixels'],2556536)
        self.assertEqual(len(coverage),32)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    report={'schema':'flow_recovery_summary_cpu_checks_v1','passed':result.wasSuccessful(),'tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'new_results_synthetic_only':True,
        'old_frozen_metric_JSON_schema_checked':True,'GPU_or_model_or_real_arrays_used':False,
        'source_sha256':{n:__import__('hashlib').sha256((HERE/n).read_bytes()).hexdigest() for n in ('summarize_recovery.py','check_summary_cpu.py')}}
    write(HERE/'summary_cpu_checks.json',report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
