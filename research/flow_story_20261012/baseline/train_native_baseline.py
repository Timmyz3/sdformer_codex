#!/usr/bin/env python3
"""Native-v1 canonical6000: default CPU preflight; explicit bounded GPU phases.

Uses a SHA-pinned original training core without editing its file. New encoding,
acceptance, wrapper and adapter hashes enter the checkpoint training contract.
No subset, old checkpoint import, auto long-run, validation, AMP or full15.
"""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode = True
import native_data_acceptance as native

HERE = Path(__file__).resolve().parent
CORE_SHA = '9628cad1c7dc7e20416ec9465303692f1b3860aaf6c6f3413345cc243b9e11e0'
SCHEMA = 'flow_a_native_encoding_v2_checkpoint_v1'
CREATOR = 'flow_a_20261012_native_encoding_baseline'
PILOT_SCHEMA = 'flow_native_gpu_pilot_success_v1'


def load_core(data_contract=None):
    """Only controlled in-memory substitutions; original source stays pinned."""
    path = HERE / 'train_baseline.py'
    native.read_pinned(path, CORE_SHA)
    spec = importlib.util.spec_from_file_location('flow_native_pinned_training_core', path)
    core = importlib.util.module_from_spec(spec); sys.modules[spec.name] = core; spec.loader.exec_module(core)
    core.SCHEMA, core.CREATOR = SCHEMA, CREATOR
    # Strengthen old per-read path handling, retaining its actual bytes SHA,
    # finite/dtype/shape checks, native preprocessing and random DataLoader.
    core.under_root = native.checked_path
    original_config = core.effective_config
    def effective_config(raw, args, parser_class):
        if data_contract is None:
            raise ValueError('Native data contract required before model/config construction')
        config = original_config(raw, args, parser_class)
        config['data_contract'] = copy.deepcopy(data_contract)
        return config
    core.effective_config = effective_config
    return core


def preflight(a):
    core = load_core()
    sources = core.checked_json(a.source_manifest, core.SOURCE_MANIFEST_SHA)
    native.require(len(sources) == 63, 'Expected pinned63-file native source')
    for relative, expected in sources.items():
        native.require(native.sha(native.checked_path(a.source_root, relative)) == expected, 'Native source changed: ' + relative)
    rows, hashes, data_contract, audit = native.validate_native_bundle(
        a.data_root, a.manifest, a.data_acceptance, a.acceptance_sha256, a.archive_audit, check_current_files=True)
    native.require(audit['current_file_stat_checked'] is True and len(rows) == 6000 and len(hashes) == 18000,
                   'Canonical native full data preflight incomplete')
    data_contract.update(entrypoint_sha256=native.sha(__file__), adapter_sha256=native.sha(native.__file__),
                         training_core_sha256=CORE_SHA, checkpoint_schema=SCHEMA, checkpoint_creator=CREATOR)
    resume_metadata = validate_resume_metadata(a.resume, data_contract, core) if a.resume else None
    if resume_metadata:
        native.require(resume_metadata['completed_optimizer_steps'] < a.max_steps,
                       'Requested cumulative step bound must exceed the resume checkpoint')
    continuation = validate_continuation(a, data_contract, resume_metadata, core) if a.phase == 'continue' else None
    ids = [r['target_id'] for r in rows]
    plan = {'scope': 'canonical6000', 'targets': len(rows), 'target_ids': ids,
            'training_target_ids_sha256': core.digest(ids), 'training_files_sha256': core.digest(hashes),
            'canonical_manifest_sha256': native.CANONICAL_SHA, 'source_manifest_sha256': core.SOURCE_MANIFEST_SHA,
            'source_verified_files': len(sources), 'native_data_contract': data_contract, 'acceptance_check': audit,
            'sample_SHA_checked_on_every_Dataset_read': True, 'validation_used': False, 'GPU_executed': False,
            'resume_metadata_checked': resume_metadata,
            'execution_phase': a.phase, 'continuation_gate': continuation,
            'segment_limits': {'cumulative_steps_at_most': 500 if a.phase == 'pilot' else 480000,
                               'requested_max_steps': a.max_steps, 'wall_hours': a.wall_hours},
            'no_auto_long_training': True}
    return rows, hashes, sources, plan


def validate_resume_metadata(path, data_contract, core):
    """Reject old/schema/data drift before CUDA or any pickle deserialization."""
    path = Path(path)
    if path.name == 'latest.json':
        pointer = json.loads(path.read_text()); filename = pointer['checkpoint_file']
        native.require(Path(filename).name == filename, 'Unsafe checkpoint pointer')
        path = path.parent / filename
    side = json.loads(path.with_suffix('.json').read_text())
    native.require(side.get('schema_version') == SCHEMA and side.get('created_by') == CREATOR and
                   side.get('training_scope') == 'canonical6000' and side.get('checkpoint_file') == path.name,
                   'Old/engineering checkpoint metadata forbidden in native run')
    run = json.loads((path.parent / 'run_contract.json').read_text())
    contract = run['contract']
    native.require(contract.get('schema_version') == SCHEMA and contract.get('scope') == 'canonical6000' and
                   contract.get('effective_config', {}).get('data_contract') == data_contract and
                   run.get('contract_sha256') == side.get('contract_sha256') == core.digest(contract),
                   'Resume data/code/encoding contract differs')
    native.require(native.sha(path) == side.get('checkpoint_sha256'), 'Native resume checkpoint SHA differs')
    return {'checkpoint_file': path.name, 'checkpoint_sha256': side['checkpoint_sha256'],
            'schema_version': SCHEMA, 'contract_sha256': side['contract_sha256'],
            'completed_optimizer_steps': side['completed_optimizer_steps'],
            'full_runtime_optimizer_RNG_validation': 'performed by pinned core before any resumed update'}


def pilot_records(directory, steps):
    """Bind the finite pilot prefix; later appended continuation rows are allowed."""
    records = []
    with (directory / 'train_steps.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if 'step' in row and row['step'] <= steps:
                records.append(row)
    native.require([r['step'] for r in records] == list(range(1, steps + 1)), 'Pilot log prefix is incomplete/duplicated')
    for row in records:
        native.require(all(isinstance(row.get(k), (int, float)) and not isinstance(row[k], bool) and math.isfinite(row[k])
                           for k in ('loss', 'grad_norm_preclip', 'lr', 'step_seconds_excluding_loader_wait')),
                       'Pilot log has missing/nonfinite numerical evidence')
        native.require(row['grad_norm_preclip'] >= 0 and row['lr'] == 0.001 and
                       row['step_seconds_excluding_loader_wait'] > 0 and row.get('epoch_0based') == 0 and
                       row.get('index_0based') == row['step'] - 1 and isinstance(row.get('target_id'), str),
                       'Pilot log recipe/cursor/timing differs')
    return records


def validate_pilot_evidence(directory, marker, data_contract, core):
    native.require(marker.get('schema') == PILOT_SCHEMA and marker.get('status') == 'PILOT_COMPLETE' and
                   marker.get('created_by') == CREATOR and marker.get('execution_phase') == 'pilot' and
                   marker.get('GPU_executed') is True and marker.get('synthetic') is False,
                   'A completed real native GPU pilot marker is required')
    steps = marker.get('completed_optimizer_steps')
    native.require(type(steps) is int and 1 <= steps <= 500 and marker.get('requested_max_steps') == steps,
                   'Pilot did not reach its bounded requested steps')
    native.require(marker.get('data_contract_sha256') == core.digest(data_contract), 'Pilot data/code contract differs')
    run = json.loads((directory / 'run_contract.json').read_text()); contract = run['contract']
    native.require(contract.get('schema_version') == SCHEMA and contract.get('scope') == 'canonical6000' and
                   contract.get('effective_config', {}).get('data_contract') == data_contract and
                   marker.get('contract_sha256') == run.get('contract_sha256') == core.digest(contract) and
                   run.get('provenance', {}).get('created_by') == CREATOR and
                   run['provenance'].get('random_init') is True and run['provenance'].get('old_checkpoint_loaded') is False,
                   'Pilot training contract/provenance differs')
    native.require(bool(contract.get('runtime', {}).get('cuda')) and
                   bool(contract.get('runtime', {}).get('device_name')), 'Pilot CUDA runtime evidence absent')
    segment_path = native.checked_path(directory, marker['segment_file'])
    native.require(segment_path.parent == directory / 'segments', 'Pilot must bind an immutable segment record')
    segment = native.pinned_json(segment_path, marker['segment_sha256'])
    native.require(segment.get('status') == 'SEGMENT_COMPLETE' and segment.get('scope') == 'canonical6000' and
                   segment.get('stop_reason') == 'max_steps' and segment.get('GPU_executed') is True and
                   segment.get('validation_used') is False and segment.get('old_checkpoint_loaded') is False and
                   segment.get('precision') == 'fp32' and segment.get('segment_completed_steps', 0) > 0 and
                   segment.get('training', {}).get('completed_optimizer_steps') == steps,
                   'Pilot segment is incomplete, non-native or did not reach its step bound')
    side = segment['checkpoint']
    native.require(side.get('schema_version') == SCHEMA and side.get('created_by') == CREATOR and
                   side.get('training_scope') == 'canonical6000' and side.get('completed_optimizer_steps') == steps and
                   side.get('contract_sha256') == marker['contract_sha256'] and
                   side.get('checkpoint_sha256') == marker.get('checkpoint_sha256') and
                   side.get('checkpoint_file') == marker.get('checkpoint_file') and native.valid_sha(side['checkpoint_sha256']),
                   'Pilot checkpoint metadata mismatch')
    native.require(core.digest(pilot_records(directory, steps)) == marker.get('finite_pilot_records_sha256'),
                   'Pilot log prefix digest differs')
    return segment


def publish_pilot_marker(a, plan, core):
    """Called only after actual CUDA training returns; CPU fixtures cannot publish."""
    import torch
    native.require(a.phase == 'pilot' and a.run_gpu and torch.cuda.is_initialized() and torch.cuda.is_available(),
                   'Only actual initialized CUDA pilot execution can publish a success marker')
    segment = json.loads((a.run_dir / 'segment_result.json').read_text())
    if segment.get('stop_reason') != 'max_steps' or segment.get('training', {}).get('completed_optimizer_steps') != a.max_steps:
        return {'pilot_success_marker': None, 'reason': 'Pilot stopped before its explicitly requested step bound'}
    resume = validate_resume_metadata(a.run_dir / 'latest.json', plan['native_data_contract'], core)
    native.require(resume['completed_optimizer_steps'] == a.max_steps, 'Pilot latest checkpoint step differs')
    segment_file = 'segments/' + segment['segment_id'] + '.json'
    marker = {'schema': PILOT_SCHEMA, 'status': 'PILOT_COMPLETE', 'created_by': CREATOR,
              'execution_phase': 'pilot', 'GPU_executed': True, 'synthetic': False,
              'completed_optimizer_steps': a.max_steps, 'requested_max_steps': a.max_steps,
              'contract_sha256': resume['contract_sha256'], 'data_contract_sha256': core.digest(plan['native_data_contract']),
              'checkpoint_file': resume['checkpoint_file'], 'checkpoint_sha256': resume['checkpoint_sha256'],
              'segment_file': segment_file, 'segment_sha256': native.sha(a.run_dir / segment_file),
              'finite_pilot_records_sha256': core.digest(pilot_records(a.run_dir, a.max_steps)),
              'scope': 'Execution/finite/resume eligibility only; no AEE reproduction or compression claim'}
    validate_pilot_evidence(a.run_dir, marker, plan['native_data_contract'], core)
    destination = a.run_dir / ('pilot_step_%09d.json' % a.max_steps)
    native.require(not destination.exists(), 'Pilot success marker is immutable')
    core.atomic_json(destination, marker)
    return {'pilot_success_marker': destination.name, 'sha256': native.sha(destination)}


def validate_continuation(a, data_contract, resume, core):
    native.require(a.resume and resume and a.pilot_marker and native.valid_sha(a.pilot_marker_sha256),
                   'Continue requires explicit resume and trusted pilot marker SHA')
    native.require(a.pilot_marker.parent == a.run_dir and a.resume.parent == a.run_dir,
                   'Pilot marker, resume checkpoint and continuation must belong to the same run directory')
    marker = native.pinned_json(a.pilot_marker, a.pilot_marker_sha256)
    validate_pilot_evidence(a.run_dir, marker, data_contract, core)
    native.require(resume['contract_sha256'] == marker['contract_sha256'] and
                   marker['completed_optimizer_steps'] <= resume['completed_optimizer_steps'] < a.max_steps,
                   'Continuation checkpoint/step bound differs from successful pilot contract')
    if resume['completed_optimizer_steps'] == marker['completed_optimizer_steps']:
        native.require(resume['checkpoint_sha256'] == marker['checkpoint_sha256'], 'Pilot anchor checkpoint SHA mismatch')
    else:
        run = json.loads((a.run_dir / 'run_contract.json').read_text())
        prior = run.get('preflight', {}).get('continuation_gate', {}) or {}
        native.require(prior.get('pilot_marker_sha256') == a.pilot_marker_sha256 and
                       prior.get('pilot_checkpoint_sha256') == marker['checkpoint_sha256'],
                       'Later continuation checkpoint lacks the same recorded pilot authority')
    return {'pilot_marker_file': a.pilot_marker.name, 'pilot_marker_sha256': a.pilot_marker_sha256,
            'pilot_completed_optimizer_steps': marker['completed_optimizer_steps'],
            'pilot_checkpoint_sha256': marker['checkpoint_sha256'], 'same_training_contract': True,
            'single_segment_only': True, 'automatic_next_segment': False}


def guard_run_directory(directory, plan=None):
    """Do not overwrite historical engineering or differently bound run records."""
    run = directory / 'run_contract.json'
    if run.exists():
        prior = json.loads(run.read_text())
        native.require(prior.get('contract', {}).get('schema_version') == SCHEMA and
                       prior.get('provenance', {}).get('created_by') == CREATOR,
                       'Run directory contains old or foreign training records; choose a fresh directory')
    previous = directory / 'preflight.json'
    if previous.exists():
        old = json.loads(previous.read_text())
        native.require(old.get('native_data_contract', {}).get('schema') == 'flow_native_training_data_contract_v1',
                       'Run directory contains old engineering preflight; never overwrite it')
        if plan is not None:
            native.require(old['native_data_contract'] == plan['native_data_contract'],
                           'Existing native run binds different data/code; use a new run directory')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-root', 'data-root', 'run-dir', 'manifest', 'source-manifest', 'data-acceptance', 'archive-audit'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--acceptance-sha256', required=True, help='Explicit trusted SHA of the completed NEW acceptance.json')
    parser.add_argument('--phase', choices=('pilot', 'continue'), default='pilot')
    parser.add_argument('--max-steps', type=int, required=True, help='Cumulative bound: pilot<=500; explicit continue<=480000')
    parser.add_argument('--wall-hours', type=float, default=1, help='One bounded invocation, no auto continuation; at most12h')
    parser.add_argument('--resume', type=Path, help='Only this new native schema and identical data/code/recipe/runtime contract')
    parser.add_argument('--pilot-marker', type=Path, help='Continue only: immutable success marker from this real native GPU pilot')
    parser.add_argument('--pilot-marker-sha256', help='Continue only: explicitly recorded pilot marker SHA')
    parser.add_argument('--checkpoint-steps', type=int, default=250)
    parser.add_argument('--keep-last', type=int, default=2)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run-gpu', action='store_true')
    a = parser.parse_args(argv)
    if not (1 <= a.max_steps <= (500 if a.phase == 'pilot' else 480000) and 0 < a.wall_hours <= 12 and
            0 <= a.workers <= 2 and 1 <= a.cpu_threads <= 2 and 1 <= a.checkpoint_steps <= 1000 and
            2 <= a.keep_last <= 5 and 1 <= a.log_every <= 100 and 0 <= a.seed < 2**31 and a.device >= 0):
        parser.error('Arguments outside bounded native phase limits')
    if a.phase == 'continue' and not (a.resume and a.pilot_marker and native.valid_sha(a.pilot_marker_sha256)):
        parser.error('Explicit continue requires --resume, --pilot-marker and --pilot-marker-sha256')
    if a.phase == 'pilot' and (a.pilot_marker or a.pilot_marker_sha256):
        parser.error('Pilot marker authority is used only with --phase continue')
    if not native.valid_sha(a.acceptance_sha256):
        parser.error('Explicit full native acceptance SHA required')
    for key, value in vars(a).items():
        if isinstance(value, Path): setattr(a, key, value.resolve())
    a.scope = 'canonical6000'
    return a


def main(argv=None):
    a = parse_args(argv); started = time.monotonic()
    guard_run_directory(a.run_dir)
    a.run_dir.mkdir(parents=True, exist_ok=True)
    with (a.run_dir / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        core = load_core()
        try:
            rows, hashes, sources, plan = preflight(a)
            guard_run_directory(a.run_dir, plan)
            core.atomic_json(a.run_dir / 'preflight.json', plan)
            if not a.run_gpu:
                print(json.dumps(plan, indent=2)); return 0
            engine = load_core(plan['native_data_contract'])
            engine.run_training(a, rows, hashes, sources, plan, started)
            if a.phase == 'pilot':
                print(json.dumps(publish_pilot_marker(a, plan, engine)), flush=True)
            return 0
        except BaseException as exc:
            core.atomic_json(a.run_dir / 'failure.json', {'status': 'FAILED', 'exception': type(exc).__name__,
                'message': str(exc), 'traceback': traceback.format_exc(), 'wall_seconds': time.monotonic() - started,
                'scope': 'new_native_canonical6000', 'execution_phase': a.phase, 'automatic_long_training': False})
            raise


if __name__ == '__main__':
    raise SystemExit(main())
