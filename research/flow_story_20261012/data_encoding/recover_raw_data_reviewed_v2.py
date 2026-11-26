#!/usr/bin/env python3
"""Recover raw DSEC inputs + GT/masks to a NEW dataset root, never old voxels.

Reuses the audited 20261011 streaming/CRC/SHA/atomic-receipt transport unchanged.
Locally available GT/mask/metadata files are hardlinked ONLY after rehash against
a completed old transfer receipt. Raw events and remaining files stream from the
archive. This script performs no voxel generation or GPU training.
"""
import argparse
import concurrent.futures
import csv
import importlib.util
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('transport', 'audit', 'canonical', 'source-csv', 'old-root', 'old-receipts', 'target-root', 'out-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--source-socket', required=True)
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location('audited_transfer', a.transport)
    transport = importlib.util.module_from_spec(spec); spec.loader.exec_module(transport)
    # This imported source remains an immutable dependency, with SHA in every plan.
    audit = json.loads(a.audit.read_text())
    if transport.sha_file(a.canonical) != '6829f9ac8181f8eb5e7ce606c23915c2109c28447a0342bde8696cda045a7b22':
        raise ValueError('Canonical target manifest changed')
    if transport.sha_file(a.source_csv) != audit['transfer_manifest']['sha256']:
        raise ValueError('Archive NPY inventory changed')
    rows = json.loads(a.canonical.read_text())['records']
    sequences = {r['sequence'] for r in rows}
    raw = audit['raw_sequences']
    if len(raw) != 18 or {r['sequence'] for r in raw} != sequences:
        raise ValueError('Required raw sequences differ')
    source_root = Path(audit['remote_root'])
    a.old_root, a.target_root = a.old_root.resolve(), a.target_root.resolve()
    if a.old_root == a.target_root or a.old_root in a.target_root.parents or a.target_root in a.old_root.parents:
        raise ValueError('New dataset root must be a separate sibling')
    metadata = [dict(relative_path=r['relative_path'], bytes=int(r['bytes']), sha256=r.get('sha256'))
                for r in audit['phase1_metadata_transfer']['files']]
    raw_files = [{'relative_path': str(Path(r['left_events']['path']).relative_to(source_root)),
                  'bytes': int(r['left_events']['bytes']), 'sha256': None} for r in raw]
    if sum(r['bytes'] for r in raw_files) != audit['raw_required_left_events_bytes']:
        raise ValueError('Raw byte total differs')
    # 01_a first provides the first independent regeneration sample quickly.
    raw_files.sort(key=lambda r: (0 if '/zurich_city_01_a/' in r['relative_path'] else 1, r['bytes']))
    with a.source_csv.open() as stream:
        gt = [{'relative_path': r['relative_path'], 'bytes': int(r['bytes']), 'sha256': None}
              for r in csv.DictReader(stream) if r['category'] in ('gt', 'mask')]
    wanted_gt = {'saved_flow_data/' + folder + '/' + r['v2_target_filename']
                 for r in rows for folder in ('gt_tensors', 'mask_tensors')}
    if len(gt) != 16304 or {r['relative_path'] for r in gt} != wanted_gt:
        raise ValueError('GT/mask inventory differs from canonical members')
    files = metadata + raw_files + gt
    if len(files) != 16362 or len({r['relative_path'] for r in files}) != len(files):
        raise ValueError('Recovery scope count/uniqueness')
    known = {}
    old_plan = json.loads((a.old_receipts.parent/'full_plan.json').read_text())
    old_batches = {m['batch_id']: m for m in old_plan['batches']}
    for path in sorted(a.old_receipts.glob('*.json')):
        receipt = json.loads(path.read_text())
        if (receipt.get('schema') != 'flow-dsec-receipt-v1' or receipt.get('status') != 'verified'
                or receipt.get('target_root') != str(a.old_root) or receipt.get('source_root') != str(source_root)):
            raise ValueError('Not a verified old-root receipt')
        # Original manifests may omit as-yet-unknown SHA values, so reconstruct
        # their identity from the immutable plan, not the completed hash list.
        identity = old_batches.get(receipt['batch_id'])
        if not identity or identity['manifest_sha256'] != receipt['manifest_sha256']:
            raise ValueError('Receipt not in original transfer plan')
        original_manifest = json.loads((a.old_receipts.parent/'manifests'/(receipt['batch_id']+'.json')).read_text())
        transport.remote_functions()['verify_manifest'](original_manifest)
        transport.remote_functions()['check_receipt'](original_manifest, receipt)
        if receipt['source_hashes']['files'] != receipt['files'] or transport.remote_functions()['digest'](receipt['source_hashes']) != receipt['source_hashes_sha256']:
            raise ValueError('Old receipt source digest mismatch')
        for item in receipt['files']:
            previous = known.get(item['relative_path'])
            if previous and previous != item: raise ValueError('Conflicting old receipts')
            known[item['relative_path']] = item
    for item in files:
        prior = known.get(item['relative_path'])
        if prior:
            if prior['bytes'] != item['bytes'] or (item['sha256'] and item['sha256'] != prior['sha256']):
                raise ValueError('Old receipt content conflicts with expected inventory')
            item['sha256'] = prior['sha256']
    a.out_dir.mkdir(parents=True, exist_ok=True)
    plan = {'schema': 'flow_a_raw_recovery_v1', 'source_root': str(source_root),
        'old_root': str(a.old_root), 'target_root': str(a.target_root), 'files': files,
        'file_count': len(files), 'total_bytes': sum(r['bytes'] for r in files),
        'raw_events_bytes': sum(r['bytes'] for r in raw_files),
        'old_voxels_included': False, 'GPU_training': False, 'voxel_generation': False,
        'transport_sha256': transport.sha_file(a.transport), 'script_sha256': transport.sha_file(__file__),
        'audit_sha256': transport.sha_file(a.audit), 'canonical_sha256': transport.sha_file(a.canonical),
        'source_csv_sha256': transport.sha_file(a.source_csv)}
    plan_path = a.out_dir / 'recovery_plan.json'
    if plan_path.exists():
        prior_plan = json.loads(plan_path.read_text())
        if {k:v for k,v in prior_plan.items() if k != 'script_sha256'} != {k:v for k,v in plan.items() if k != 'script_sha256'}:
            raise ValueError('Existing semantic plan changed')
    else: transport.write_json(plan_path, plan)
    if not a.run:
        print(json.dumps({k:v for k,v in plan.items() if k != 'files'})); return
    import fcntl
    lock = (a.out_dir/'recovery.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    endpoint = ('root@ssh.sd5ai.scnet.cn', 10037, a.source_socket)
    transport.socket_check(endpoint)
    started = time.monotonic(); linked = []; pending = []
    for item in files:
        relative = item['relative_path']
        old, new = a.old_root / relative, a.target_root / relative
        if relative in known and old.is_file():
            if old.is_symlink() or old.stat().st_size != item['bytes'] or transport.sha_file(old) != item['sha256']:
                raise ValueError('Previously accepted file changed: ' + relative)
            new.parent.mkdir(parents=True, exist_ok=True)
            try: os.link(old, new)
            except FileExistsError:
                if new.is_symlink() or new.stat().st_size != item['bytes'] or transport.sha_file(new) != item['sha256']:
                    raise ValueError('New-root hardlink conflict: ' + relative)
            linked.append(item)
        else:
            pending.append(item)
    transport.write_json(a.out_dir/'verified_local_links.json', {'files': linked, 'count': len(linked),
        'bytes': sum(r['bytes'] for r in linked), 'scope': 'Rehashed against source transfer receipts before linking; immutable shared contents'})
    batches = [transport.batch_manifest(batch, str(source_root), str(a.target_root))
               for batch in transport.split_batches(pending, 768*1024*1024, False)]
    results = []; failure = []
    def status(state):
        transport.write_json(a.out_dir/'recovery_status.json', {'status': state, 'elapsed_seconds': time.monotonic()-started,
            'linked_files': len(linked), 'transferred_verified_files': sum(r['files'] for r in results),
            'transferred_verified_bytes': sum(r['bytes'] for r in results), 'results': results, 'failures': failure,
            'voxel_generation_started': False, 'GPU_training_started': False,
            'current_script_sha256': plan['script_sha256']})
    status('transferring_raw_and_missing_GT')
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            tasks = [pool.submit(transport.transfer_one, m, endpoint, None, a.out_dir) for m in batches]
            try:
                for task in concurrent.futures.as_completed(tasks):
                    results.append(task.result()); status('transferring_raw_and_missing_GT')
            except BaseException:
                transport.STOP.set()
                for task in tasks: task.cancel()
                raise
        collected = {r['relative_path']: r for r in linked}
        for m in batches:
            receipt = json.loads((a.out_dir/'receipts'/(m['batch_id']+'.json')).read_text())
            transport.remote_functions()['check_receipt'](m, receipt)
            for r in receipt['files']: collected[r['relative_path']] = r
        if set(collected) != {r['relative_path'] for r in files}: raise ValueError('Recovery incomplete')
        import io
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=['relative_path','bytes','sha256']); writer.writeheader()
        writer.writerows(collected[r['relative_path']] for r in files)
        inventory = a.out_dir/'source_hashes.csv'
        if inventory.exists() and inventory.read_text() != stream.getvalue().replace('\r\n', '\n'):
            raise ValueError('Existing final recovery inventory differs')
        if not inventory.exists():
            import tempfile
            fd, temporary = tempfile.mkstemp(prefix='.source_hashes.', dir=str(a.out_dir))
            try:
                with os.fdopen(fd, 'w', newline='') as destination:
                    destination.write(stream.getvalue()); destination.flush(); os.fsync(destination.fileno())
                os.link(temporary, inventory)
            finally: os.unlink(temporary)
        status('raw_and_GT_verified_ready_for_separate_CPU_generation')
    except BaseException as exc:
        transport.STOP.set(); failure.append(repr(exc)); status('failed'); raise


if __name__ == '__main__': main()
