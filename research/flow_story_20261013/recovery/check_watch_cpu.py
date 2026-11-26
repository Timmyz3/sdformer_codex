#!/usr/bin/env python3
"""No SSH or real observer is launched; temporary JSON fixtures only."""
import base64
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import watch_recovery_results as w


class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='flow_watch_cpu_');self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def packet(self,status='RUNNING'):
        q={'schema':'flow_recovery_queue_v1','status':status,'active_job':None}
        raw=json.dumps(q).encode()
        return {'schema':'flow_recovery_json_transport_v1','run_directory':w.RUN,'remote_utc':'synthetic-utc',
            'files':{'queue_status.json':{'sha256':w.sha(raw),'bytes':len(raw),'base64':base64.b64encode(raw).decode()}},
            'missing':[p for p in w.FILES if p!='queue_status.json'],
            'queue':{'status':status,'experiment_status':None,'active_job':None}}
    def test_actual_remote_reader_code_delta_only_and_whitelist(self):
        # Run the exact embedded read-only code locally with only its fixed root
        # substituted into an isolated temp fixture. No SSH, no native data.
        (self.root/'queue_status.json').write_text(json.dumps({'schema':'flow_recovery_queue_v1','status':'RUNNING'}))
        (self.root/'forbidden.pt').write_bytes(b'NOT JSON\x00\xff')
        code=w.REMOTE_CODE.replace("RUN='/data/flow_a/runs/recovery_20261013'",'RUN='+repr(str(self.root)))
        def run(known):
            result=subprocess.run([sys.executable,'-c',code],input=json.dumps({'known_sha256':known}).encode(),
                                  stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=5)
            self.assertEqual(result.returncode,0,result.stderr.decode())
            return json.loads(result.stdout)
        first=run({});self.assertEqual(set(first['files']),{'queue_status.json'})
        self.assertIn('base64',first['files']['queue_status.json'])
        known={p:r['sha256'] for p,r in first['files'].items()}
        second=run(known);self.assertNotIn('base64',second['files']['queue_status.json'])
        self.assertNotIn('forbidden.pt',second['missing'])
    def test_transport_rejects_hash_inventory_and_disappeared_evidence(self):
        original=self.packet();packet,changed=w.validate_packet(json.dumps(original).encode(),{})
        self.assertEqual(set(changed),{'queue_status.json'})
        bad=json.loads(json.dumps(original));bad['files']['queue_status.json']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'hash/length'):w.validate_packet(json.dumps(bad).encode(),{})
        bad=json.loads(json.dumps(original));bad['files']['../forbidden.pt']=bad['files'].pop('queue_status.json')
        with self.assertRaisesRegex(ValueError,'inventory'):w.validate_packet(json.dumps(bad).encode(),{})
        with self.assertRaisesRegex(ValueError,'missing remotely'):
            w.validate_packet(json.dumps(original).encode(),{'smoke_audit.json':'a'*64})
    def test_mocked_SSH_poll_hash_writes_and_real_local_summary(self):
        local=self.root/'local';local.mkdir()
        fake=types.SimpleNamespace(returncode=0,stdout=json.dumps(self.packet()).encode(),stderr=b'')
        with patch.object(w,'LOCAL',local),patch.object(w.subprocess,'run',return_value=fake) as call:
            result=w.poll('/tmp/existing_fixture_socket')
            self.assertEqual(result['changed_files'],['queue_status.json'])
            self.assertEqual(call.call_args[1]['timeout'],30)
            self.assertIn('ProxyCommand=false',call.call_args[0][0])
            self.assertTrue((local/'summary/RECOVERY_SUMMARY.md').exists())
            self.assertEqual(json.loads((local/'queue_status.json').read_text())['status'],'RUNNING')
            self.assertEqual(w.local_hashes()['queue_status.json'],self.packet()['files']['queue_status.json']['sha256'])
    def test_terminal_failure_limit_and_deadline_status_without_SSH(self):
        update={'remote_utc':'synthetic-utc','queue':{'status':w.TERMINAL[0]},'changed_files':[]}
        with patch.object(w,'LOCAL',self.root/'terminal'),patch.object(w,'poll',return_value=update) as poll:
            result=w.watch('/tmp/fixture')
            self.assertEqual(result['status'],'TERMINAL_SYNCED');self.assertEqual(poll.call_count,1)
            self.assertIsNotNone(result['last_sync_utc'])
        with patch.object(w,'LOCAL',self.root/'failed'),patch.object(w,'poll',side_effect=OSError('synthetic disconnect')) as poll, \
             patch.object(w.time,'sleep'),contextlib.redirect_stdout(io.StringIO()):
            result=w.watch('/tmp/fixture')
            self.assertEqual(result['status'],'FAILED_CONSECUTIVE_POLLS');self.assertEqual(poll.call_count,3)
        with patch.object(w,'LOCAL',self.root/'expired'),patch.object(w,'MAX_SECONDS',0),patch.object(w,'poll') as poll:
            result=w.watch('/tmp/fixture')
            self.assertEqual(result['status'],'TIMED_OUT');poll.assert_not_called()


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    report={'schema':'flow_recovery_watch_cpu_checks_v1','passed':result.wasSuccessful(),'tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'SSH_or_actual_watcher_started':False,
        'GPU_or_native_data_or_weights_used':False,'fixtures_only':True,
        'source_sha256':{n:w.sha((w.HERE/n).read_bytes()) for n in ('watch_recovery_results.py','check_watch_cpu.py')}}
    (w.HERE/'watch_cpu_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
