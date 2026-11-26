#!/usr/bin/env python3
"""Native schema guard tests; synthetic metadata and CPU tensors only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
import native_data_acceptance as n
import train_native_baseline as entry

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parents[1] / 'flow_story_20261011'
CANONICAL = PREVIOUS / 'preflight/data_protocol/artifacts/common_targets.json'
AUDIT = PREVIOUS / 'preflight/data_protocol/archive_data_audit.json'


def encoding():
    return {'encoding_id': n.ENCODING_ID, 'canonical_manifest_sha256': n.CANONICAL_SHA,
            'source_pins': n.SOURCE_PINS, 'script_sha256': n.GENERATOR_SHA, 'CPU_threads': 1,
            'bins': 10, 'shape': [10, 480, 640], 'dtype': 'float32',
            'time': '(t-t[0]).astype(float32); t/t[-1]; native convert_CHW renormalization',
            'spatial': 'all sliced events; native rectify and tap-wise validity',
            'polarity': 'signed 2p-1; no post-voxel z-score', 'slice': '[t_beg,t_end)',
            'torch': '2.5.1+cu124', 'numpy': '1.26.4', 'h5py': '3.15.1'}


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def create_fixture(directory):
    records = n.canonical_records(CANONICAL)
    expected = n.expected_members(records, n.pinned_json(AUDIT, n.AUDIT_SHA))
    rows, hashes, verified = [], [], []
    for relative, e in expected.items():
        h = e['known_sha256'] or n.digest(relative)
        row = dict(relative_path=relative, **{k: e[k] for k in ('category', 'split', 'target', 'sequence', 'bytes')}, sha256=h,
                   proof_kind='generated_native_input_receipt' if e['category'] == 'input' else 'verified_source_transfer',
                   generation_receipt_sha256=n.digest('receipt:' + relative) if e['category'] == 'input' else '')
        rows.append(row); hashes.append({k: row[k] for k in ('relative_path', 'bytes', 'sha256')})
        v = {'relative_path': relative, 'category': e['category'], 'passed': True, 'bytes': e['bytes'], 'sha256': h}
        if e['category'] in n.ARRAYS:
            shape, dtype, _ = n.ARRAYS[e['category']]
            v.update(npy_header={'shape': shape, 'dtype': dtype, 'fortran_order': False, 'version': [1, 0], 'header_bytes': 128},
                     all_elements_finite=True, tensor_C_order_bytes_sha256=n.digest('tensor:' + relative))
        verified.append(v)
    for name, content in [('native_deployment_manifest.csv', rows), ('verified_hash_inventory.csv', hashes)]:
        with (directory/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(content[0])); writer.writeheader(); writer.writerows(content)
    counts = dict(n.COUNTS, categories={c: sum(e['category'] == c for e in expected.values())
                                      for c in {e['category'] for e in expected.values()}})
    provenance = {'canonical_manifest_sha256': n.CANONICAL_SHA, 'archive_audit_sha256': n.AUDIT_SHA,
                  'generator_sha256': n.GENERATOR_SHA, 'validator_sha256': n.VALIDATOR_SHA,
                  'generator_source_pins': n.SOURCE_PINS,
                  'generation_reports_sha256': {str(i): str(i) * 64 for i in range(4)},
                  'generation_receipts_SHA_inventory_digest': 'a' * 64,
                  'recovery': {'source_file_count': 16362, 'linked_files': 7526, 'transferred_files': 8836,
                               **{k: 'b' * 64 for k in ('recovery_plan_sha256', 'recovery_status_sha256',
                                                       'source_hash_inventory_sha256', 'verified_local_links_sha256')}}}
    manifest = {'schema': 'flow-native-deployment-v1', 'encoding_id': n.ENCODING_ID,
                'canonical_manifest_sha256': n.CANONICAL_SHA, 'counts': counts, 'encoding_contract': encoding(),
                'encoding_contract_sha256': n.digest(encoding()), 'files_csv': 'native_deployment_manifest.csv',
                'files_csv_sha256': n.sha(directory/'native_deployment_manifest.csv'),
                'hash_inventory': 'verified_hash_inventory.csv', 'hash_inventory_sha256': n.sha(directory/'verified_hash_inventory.csv'),
                'npy_contracts': {k: {'shape': s, 'dtype': d} for k, (s, d, _) in n.ARRAYS.items()}, 'provenance': provenance}
    dump(directory/'native_deployment_manifest.json', manifest)
    dump(directory/'verification_files.json', {'schema': 'flow-native-file-verification-v1', 'files': verified})
    accepted = {'schema': 'flow-native-dataset-acceptance-v1', 'status': 'VERIFIED_NATIVE_DATASET', 'passed': True,
                'root': str(directory/'DSEC_native_test'), 'encoding_id': n.ENCODING_ID,
                'canonical_manifest_sha256': n.CANONICAL_SHA, 'encoding_contract_sha256': n.digest(encoding()),
                'counts': counts, 'canonical_membership_complete': True,
                'all_generated_inputs_match_generation_receipts': True,
                'all_recovered_source_files_match_verified_transfer_hashes': True,
                'all_NPY_headers_shapes_dtypes_and_finite_verified': True,
                'deployment_manifest': 'native_deployment_manifest.json',
                'deployment_manifest_sha256': n.sha(directory/'native_deployment_manifest.json'),
                'deployment_csv_sha256': manifest['files_csv_sha256'],
                'hash_inventory': manifest['hash_inventory'], 'hash_inventory_sha256': manifest['hash_inventory_sha256'],
                'verification_files_sha256': n.sha(directory/'verification_files.json'),
                'validator_sha256': n.VALIDATOR_SHA, 'CPU_only': True, 'training_started': False,
                'legacy_validator_schema_compatible': False}
    dump(directory/'acceptance.json', accepted)


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory(); cls.directory = Path(cls.fixture.name)
        create_fixture(cls.directory)

    @classmethod
    def tearDownClass(cls): cls.fixture.cleanup()

    def bundle(self, check_files=False, root=None):
        return n.validate_native_bundle(root or self.directory/'DSEC_native_test', CANONICAL,
            self.directory/'acceptance.json', n.sha(self.directory/'acceptance.json'), AUDIT, check_files)

    @contextlib.contextmanager
    def mutation(self, name, operation):
        path = self.directory/name; before = path.read_bytes()
        marker = self.directory/'acceptance.json'; old_marker = marker.read_bytes()
        value = json.loads(before); operation(value); dump(path, value)
        if name != 'acceptance.json':
            accepted = json.loads(old_marker)
            key = {'verification_files.json': 'verification_files_sha256',
                   'native_deployment_manifest.json': 'deployment_manifest_sha256'}[name]
            accepted[key] = n.sha(path); dump(marker, accepted)
        try: yield
        finally: path.write_bytes(before); marker.write_bytes(old_marker)

    def test_full_exact_metadata_acceptance(self):
        rows, hashes, contract, audit = self.bundle()
        self.assertEqual((len(rows), len(hashes)), (6000, 18000))
        self.assertEqual(contract['valid_targets'], 2152)
        self.assertFalse(audit['current_file_stat_checked'])
        self.assertTrue(all(r['protocol_split'] == 'train' for r in rows))

    def test_actual_file_presence_not_skipped_by_formal_validation(self):
        with self.assertRaises(FileNotFoundError): self.bundle(check_files=True)

    def test_legacy_marker_wrong_root_and_pin_rejected(self):
        with self.mutation('acceptance.json', lambda x: x.update(schema='old_phase1')):
            with self.assertRaisesRegex(ValueError, 'legacy'): self.bundle()
        with self.assertRaisesRegex(ValueError, 'root'): self.bundle(root=self.directory/'old_DSEC')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            n.validate_native_bundle(self.directory/'DSEC_native_test', CANONICAL, self.directory/'acceptance.json', '0'*64, AUDIT, False)

    def test_counts_finite_and_member_coverage_rejected(self):
        with self.mutation('acceptance.json', lambda x: x['counts'].update(generated_inputs=8151)):
            with self.assertRaisesRegex(ValueError, 'counts'): self.bundle()
        with self.mutation('verification_files.json', lambda x: x['files'][0].update(all_elements_finite=False)):
            with self.assertRaisesRegex(ValueError, 'finite'): self.bundle()
        with self.mutation('verification_files.json', lambda x: x['files'].pop()):
            with self.assertRaisesRegex(ValueError, 'membership'): self.bundle()
        with self.mutation('verification_files.json', lambda x: x['files'].append(x['files'][0])):
            with self.assertRaisesRegex(ValueError, 'duplicate'): self.bundle()

    def test_wrong_encoding_source_and_dtype_rejected(self):
        with self.mutation('native_deployment_manifest.json', lambda x: x['encoding_contract'].update(time='absolute_float32')):
            with self.assertRaisesRegex(ValueError, 'encoding'): self.bundle()
        with self.mutation('native_deployment_manifest.json', lambda x: x['provenance'].update(generator_sha256='f'*64)):
            with self.assertRaisesRegex(ValueError, 'source pins'): self.bundle()
        with self.mutation('verification_files.json', lambda x: x['files'][0]['npy_header'].update(dtype='float16')):
            with self.assertRaisesRegex(ValueError, 'header'): self.bundle()

    def test_hash_inventory_tamper_rejected(self):
        p = self.directory/'verified_hash_inventory.csv'; before = p.read_bytes()
        try:
            p.write_bytes(before+b'\n')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'): self.bundle()
        finally: p.write_bytes(before)

    def test_new_core_config_contract_and_unchanged_optimizer_recipe(self):
        binding = {'encoding_id': n.ENCODING_ID, 'data_acceptance_sha256': '1'*64}
        core = entry.load_core(binding)
        class Parser:
            def reset_config(self): self.config = {}
            def parse_dict(self, raw): self.config = raw
            @staticmethod
            def combine_entries(config): return config
        class Args:
            data_root = Path('/new'); seed = 0; workers = 2; device = 0
        raw = {k: {} for k in ('data', 'loader', 'swin_transformer', 'optimizer')}
        config = core.effective_config(raw, Args(), Parser)
        self.assertEqual(config['data_contract'], binding)
        self.assertEqual(config['optimizer']['lr'], .001)
        self.assertFalse(config['optimizer']['use_amp'])
        self.assertEqual(config['loader']['n_epochs'], 80)
        self.assertEqual(n.sha(HERE/'train_baseline.py'), entry.CORE_SHA)
        pilot = Args(); pilot.phase = 'pilot'; pilot.max_steps = 32; pilot.wall_hours = 1
        continued = Args(); continued.phase = 'continue'; continued.max_steps = 480000; continued.wall_hours = 12
        self.assertEqual(core.effective_config(raw, pilot, Parser), core.effective_config(raw, continued, Parser))

    def test_new_checkpoint_roundtrip_and_old_checkpoint_namespace_rejection(self):
        core = entry.load_core(); contract = {'schema_version': entry.SCHEMA, 'scope': 'canonical6000',
                                              'effective_config': {'data_contract': {'encoding_id': n.ENCODING_ID}}}
        payload = {'schema_version': entry.SCHEMA, 'contract': contract, 'contract_sha256': core.digest(contract),
                   'provenance': {'created_by': entry.CREATOR, 'random_init': True, 'old_checkpoint_loaded': False,
                                  'training_scope': 'canonical6000'},
                   'training': {'completed_optimizer_steps': 32}, 'model_state_dict': {'w': torch.arange(4)},
                   'rng_state': core.capture_rng(torch, np)}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); core.save_checkpoint(torch, directory, payload)
            dump(directory/'run_contract.json', {'contract': contract, 'contract_sha256': core.digest(contract)})
            restored, _ = core.load_checkpoint(torch, directory/'latest.json', contract)
            self.assertTrue(torch.equal(restored['model_state_dict']['w'], payload['model_state_dict']['w']))
            info = entry.validate_resume_metadata(directory/'latest.json', contract['effective_config']['data_contract'], core)
            self.assertEqual(info['completed_optimizer_steps'], 32)
            with self.assertRaisesRegex(ValueError, 'contract differs'):
                entry.validate_resume_metadata(directory/'latest.json', {'encoding_id': 'old'}, core)
            side_path = directory/'checkpoint_step_000000032.json'; side = json.loads(side_path.read_text())
            side.update(schema_version='flow_a_native_v2_checkpoint_v1', created_by='flow_a_20261012_native_baseline')
            dump(side_path, side)
            with self.assertRaisesRegex(ValueError, 'Old/engineering'):
                entry.validate_resume_metadata(directory/'latest.json', contract['effective_config']['data_contract'], core)
            with self.assertRaisesRegex(ValueError, 'legacy imports forbidden'):
                core.load_checkpoint(torch, directory/'latest.json', contract)

    def test_sample_read_sha_and_symlink_still_guarded(self):
        core = entry.load_core()
        row = {'target_id': 'example', 'sequence': 'example', 'v2_target_filename': 'x.npy', 'gt_filename': 'x.npy', 'mask_filename': 'x.npy'}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); hashes = {}
            for category, relative in core.target_files(row).items():
                path = root/relative; path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, np.zeros(core.SHAPES[category], dtype=core.DTYPES[category]))
                hashes[relative] = {'bytes': path.stat().st_size, 'sha256': core.sha(path)}
            ds = core.VerifiedTrainDataset(root, [row], hashes)
            self.assertEqual(ds[0][-1], 'example')
            path = root/core.target_files(row)['input']; data = bytearray(path.read_bytes()); data[-1] = 1; path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'changed'): ds[0]
            link = root/'link'; link.symlink_to('saved_flow_data')
            with self.assertRaisesRegex(ValueError, 'Symlink'): core.under_root(root, 'link/anything')

    def test_cli_pilot_limit_no_subset_or_automatic_long_run(self):
        args = []
        for flag in ('source-root', 'data-root', 'run-dir', 'manifest', 'source-manifest', 'data-acceptance', 'archive-audit'):
            args += ['--'+flag, '/tmp/example']
        args += ['--acceptance-sha256', 'a'*64, '--max-steps', '32']
        parsed = entry.parse_args(args)
        self.assertEqual(parsed.scope, 'canonical6000'); self.assertFalse(parsed.run_gpu)
        self.assertEqual(parsed.max_steps, 32)
        for extra in (['--max-steps', '501'], ['--scope', 'engineering_subset'], ['--wall-hours', '13']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): entry.parse_args(args+extra)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            entry.parse_args(args+['--phase','continue','--max-steps','480000'])
        continued = entry.parse_args(args+['--phase','continue','--max-steps','480000','--wall-hours','12',
            '--resume','/tmp/example/latest.json','--pilot-marker','/tmp/example/pilot.json','--pilot-marker-sha256','b'*64])
        self.assertEqual(continued.phase, 'continue'); self.assertFalse(continued.run_gpu)

    def pilot_fixture(self, directory):
        """Synthetic CPU guard fixture ONLY: never invoke main/run_training with it."""
        core = entry.load_core(); binding = {'encoding_id': n.ENCODING_ID, 'fixture_only': True}
        contract = {'schema_version': entry.SCHEMA, 'scope': 'canonical6000',
                    'effective_config': {'data_contract': binding}, 'runtime': {'cuda': 'test-fixture', 'device_name': 'CPU-fixture'}}
        provenance = {'created_by': entry.CREATOR, 'random_init': True, 'old_checkpoint_loaded': False,
                      'training_scope': 'canonical6000'}
        payload = {'schema_version': entry.SCHEMA, 'contract': contract, 'contract_sha256': core.digest(contract),
                   'provenance': provenance, 'training': {'completed_optimizer_steps': 32},
                   'model_state_dict': {'w': torch.arange(4)}, 'rng_state': core.capture_rng(torch, np)}
        side = core.save_checkpoint(torch, directory, payload)
        run = {'contract': contract, 'contract_sha256': core.digest(contract), 'provenance': provenance, 'preflight': {}}
        dump(directory/'run_contract.json', run)
        segment = {'status': 'SEGMENT_COMPLETE', 'scope': 'canonical6000', 'stop_reason': 'max_steps',
                   'GPU_executed': True, 'validation_used': False, 'old_checkpoint_loaded': False, 'precision': 'fp32',
                   'segment_completed_steps': 32, 'training': {'completed_optimizer_steps': 32}, 'checkpoint': side}
        (directory/'segments').mkdir(); dump(directory/'segments/unit_fixture.json', segment)
        records = [{'step': i, 'target_id': 'unit-fixture-%d'%i, 'loss': 1., 'grad_norm_preclip': 2., 'lr': .001,
                    'step_seconds_excluding_loader_wait': .2, 'epoch_0based': 0, 'index_0based': i-1} for i in range(1,33)]
        (directory/'train_steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        marker = {'schema': entry.PILOT_SCHEMA, 'status': 'PILOT_COMPLETE', 'created_by': entry.CREATOR,
                  'execution_phase': 'pilot', 'GPU_executed': True, 'synthetic': False,
                  'completed_optimizer_steps': 32, 'requested_max_steps': 32,
                  'contract_sha256': core.digest(contract), 'data_contract_sha256': core.digest(binding),
                  'segment_file': 'segments/unit_fixture.json', 'segment_sha256': n.sha(directory/'segments/unit_fixture.json'),
                  'checkpoint_file': side['checkpoint_file'], 'checkpoint_sha256': side['checkpoint_sha256'],
                  'finite_pilot_records_sha256': core.digest(records)}
        dump(directory/'pilot_fixture.json', marker)
        args = SimpleNamespace(phase='continue', run_dir=directory, resume=directory/'latest.json',
                               pilot_marker=directory/'pilot_fixture.json', pilot_marker_sha256=n.sha(directory/'pilot_fixture.json'), max_steps=501)
        resume = entry.validate_resume_metadata(args.resume, binding, core)
        return core, binding, payload, run, marker, args, resume

    def test_continuation_same_contract_and_later_segment_authority(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            core,binding,payload,run,marker,args,resume = self.pilot_fixture(directory)
            gate = entry.validate_continuation(args, binding, resume, core)
            self.assertTrue(gate['single_segment_only']); self.assertFalse(gate['automatic_next_segment'])
            # A later step requires the same recorded authority, not merely a matching recipe.
            payload['training']['completed_optimizer_steps'] = 4000
            core.save_checkpoint(torch, directory, payload); args.max_steps = 480000
            later = entry.validate_resume_metadata(args.resume, binding, core)
            with self.assertRaisesRegex(ValueError, 'authority'): entry.validate_continuation(args,binding,later,core)
            run['preflight']['continuation_gate'] = gate; dump(directory/'run_contract.json',run)
            self.assertEqual(entry.validate_continuation(args,binding,later,core),gate)
            # The immutable pilot prefix remains bound when later training appends rows.
            with (directory/'train_steps.jsonl').open('a') as f: f.write(json.dumps({'step':4000})+'\n')
            self.assertEqual(entry.validate_continuation(args,binding,later,core),gate)

    def test_continuation_wrong_marker_incomplete_pilot_and_old_namespace_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            core,binding,payload,run,marker,args,resume=self.pilot_fixture(directory)
            args.pilot_marker_sha256='0'*64
            with self.assertRaisesRegex(ValueError,'SHA mismatch'): entry.validate_continuation(args,binding,resume,core)
            for key,value in [('status','INCOMPLETE'),('synthetic',True),('GPU_executed',False),
                              ('requested_max_steps',500),('data_contract_sha256','f'*64),('checkpoint_sha256','e'*64)]:
                altered=copy.deepcopy(marker);altered[key]=value;dump(args.pilot_marker,altered)
                args.pilot_marker_sha256=n.sha(args.pilot_marker)
                with self.assertRaises(ValueError): entry.validate_continuation(args,binding,resume,core)
            dump(args.pilot_marker,marker);args.pilot_marker_sha256=n.sha(args.pilot_marker)
            bad=dict(resume,checkpoint_sha256='d'*64)
            with self.assertRaisesRegex(ValueError,'anchor'):entry.validate_continuation(args,binding,bad,core)
            args.max_steps=32
            with self.assertRaisesRegex(ValueError,'step bound'):entry.validate_continuation(args,binding,resume,core)
            args.max_steps=501
            segment_path=directory/'segments/unit_fixture.json';segment=json.loads(segment_path.read_text())
            segment['stop_reason']='wall_budget';dump(segment_path,segment)
            marker['segment_sha256']=n.sha(segment_path);dump(args.pilot_marker,marker);args.pilot_marker_sha256=n.sha(args.pilot_marker)
            with self.assertRaisesRegex(ValueError,'incomplete'):entry.validate_continuation(args,binding,resume,core)
            side=directory/'checkpoint_step_000000032.json';metadata=json.loads(side.read_text())
            metadata['schema_version']='flow_a_native_v2_checkpoint_v1';dump(side,metadata)
            with self.assertRaisesRegex(ValueError,'Old/engineering'):entry.validate_resume_metadata(args.resume,binding,core)

    def test_cpu_fixture_cannot_publish_real_gpu_success_marker(self):
        with self.assertRaisesRegex(ValueError,'Only actual initialized CUDA'):
            entry.publish_pilot_marker(SimpleNamespace(phase='pilot',run_gpu=True),{},entry.load_core())

    def test_old_run_records_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); path = directory/'preflight.json'
            dump(path, {'scope': 'engineering_subset'})
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'never overwrite'): entry.guard_run_directory(directory)
            self.assertEqual(path.read_bytes(), before)
            dump(path, {'native_data_contract': {'schema': 'flow_native_training_data_contract_v1', 'encoding': 'x'}})
            with self.assertRaisesRegex(ValueError, 'different data/code'):
                entry.guard_run_directory(directory, {'native_data_contract': {'schema': 'flow_native_training_data_contract_v1', 'encoding': 'y'}})


if __name__ == '__main__':
    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeTests))
    report = {'schema': 'flow_native_baseline_cpu_tests_v1', 'passed': result.wasSuccessful(),
              'tests_run': result.testsRun, 'errors': len(result.errors), 'failures': len(result.failures),
              'GPU_initialized': torch.cuda.is_initialized(), 'old_training_core_sha256': n.sha(HERE/'train_baseline.py'),
              'native_entrypoint_sha256': n.sha(HERE/'train_native_baseline.py'),
              'native_adapter_sha256': n.sha(HERE/'native_data_acceptance.py'),
              'scope': 'Synthetic complete24514-member metadata rejection checks and CPU checkpoint/Dataset tests; no real training or new full-data acceptance'}
    dump(HERE/'native_baseline_cpu_tests.json', report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
