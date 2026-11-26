#!/usr/bin/env python3
"""Independent, CPU-only global acceptance of the NEW native-v1 DSEC dataset.

All8152 generated voxels plus all16362 recovered source files are rehashed.
NPY headers/dtypes/shapes/finite values and generation/transfer evidence are
checked separately. Publishes a NEW acceptance schema only after every check.
Never calls training, imports torch, writes dataset arrays, or changes old data.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import concurrent.futures
import csv
import datetime
import fcntl
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import time
sys.dont_write_bytecode = True

CANONICAL_SHA = '6829f9ac8181f8eb5e7ce606c23915c2109c28447a0342bde8696cda045a7b22'
AUDIT_SHA = '888bdbb22a67a06d4baedcfd0adf957235d998e5ba0685d800519157fb33ceb9'
GENERATOR_SHA = 'd0d45517cc8a43b59c02afafd030bb976de0656dd4876b3286f522c59abe4251'
ENCODING_ID = 'sdformerflow_upstream_voxel_10bins_cpu_v1'
SOURCE_PINS = {
    'DSEC_dataloader/event_representations.py': 'f02493e7440e12828b487b272d0c17f96a1a2a6c785fd8a6764981b09825403a',
    'DSEC_dataloader/DSEC_dataset_preprocess.py': '63fa4eaf500910d798e2d81fc17de85a0d4b6cd52f0ae6ade4d841be16256294',
}
ARRAYS = {'input': ((10, 480, 640), 'float32', 12288128),
          'gt': ((2, 480, 640), 'float32', 2457728),
          'mask': ((480, 640), 'bool', 307328)}
READY_STATUS = 'raw_and_GT_verified_ready_for_separate_CPU_generation'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def safe_relative(value):
    require(isinstance(value, str) and value and '\\' not in value and '\x00' not in value,
            'Unsafe relative path')
    require(not value.startswith('/') and all(x not in ('', '.', '..') for x in value.split('/')),
            'Unsafe relative path components: ' + value)
    return Path(value)


def checked_path(root, relative):
    path = root
    for part in safe_relative(relative).parts:
        path = path / part
        require(not path.is_symlink(), 'Symlink is not accepted: ' + str(path))
    require(path.resolve().is_relative_to(root.resolve()), 'Dataset path escapes root')
    return path


def pinned_json(path, expected):
    require(sha(path) == expected, 'Pinned input SHA mismatch: ' + str(path))
    return json.loads(Path(path).read_text())


def unique_records(rows, label):
    require(isinstance(rows, list), label + ': records must be a list')
    out = {}
    for row in rows:
        relative = row['relative_path']; safe_relative(relative)
        require(relative not in out, label + ': duplicate member ' + relative)
        require(type(row['bytes']) is int and row['bytes'] >= 0, label + ': invalid bytes')
        out[relative] = row
    return out


def load_hash_inventory(path):
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row['bytes'] = int(row['bytes'])
        require(valid_sha(row['sha256']), 'Invalid full source SHA')
    return unique_records(rows, 'hash inventory')


def canonical_members(path):
    records = pinned_json(path, CANONICAL_SHA)['records']
    ids = [r['target_id'] for r in records]
    require(len(ids) == len(set(ids)) == 8152, 'Canonical unique target count changed')
    require(sum(r['protocol_split'] == 'train' for r in records) == 6000 and
            sum(r['protocol_split'] == 'valid' for r in records) == 2152, 'Canonical split changed')
    require(len({r['sequence'] for r in records}) == 18, 'Canonical sequence membership changed')
    return records


def source_members(records, audit):
    """Independently construct the recovery set; no old event voxel is included."""
    result = {}
    for row in records:
        for category, filename in (('gt', row['gt_filename']), ('mask', row['mask_filename'])):
            rel = 'saved_flow_data/' + ('gt_tensors/' if category == 'gt' else 'mask_tensors/') + filename
            result[rel] = {'relative_path': rel, 'category': category, 'split': row['protocol_split'],
                           'target': row['target_id'], 'sequence': row['sequence'],
                           'bytes': ARRAYS[category][2], 'known_sha256': None}
    source_root = Path(audit['remote_root'])
    require(len(audit['raw_sequences']) == 18 and
            {r['sequence'] for r in audit['raw_sequences']} == {r['sequence'] for r in records},
            'Pinned raw sequence set differs')
    for item in audit['raw_sequences']:
        raw = item['left_events']; relative = str(Path(raw['path']).relative_to(source_root))
        result[relative] = {'relative_path': relative, 'category': 'raw_left_events', 'split': 'source',
                            'target': '', 'sequence': item['sequence'], 'bytes': raw['bytes'], 'known_sha256': None}
    for item in audit['phase1_metadata_transfer']['files']:
        relative = item['relative_path']
        require(relative not in result, 'Pinned source metadata overlap')
        result[relative] = {'relative_path': relative, 'category': item['category'], 'split': 'source',
                            'target': item.get('target', ''), 'sequence': item.get('sequence', ''),
                            'bytes': item['bytes'], 'known_sha256': item.get('sha256')}
    # Retain the eight previously raw-PNG-bound GT/mask hashes; old event hashes
    # are intentionally excluded because these inputs are being regenerated.
    for sample in audit['sample_content_checks']:
        for category in ('gt', 'mask'):
            array = sample['arrays'][category]
            relative = str(Path(array['path']).relative_to(source_root))
            require(relative in result and result[relative]['bytes'] == array['bytes'], 'Audited GT/mask mismatch')
            result[relative]['known_sha256'] = array['sha256']
    require(len(result) == 16362, 'Recovery must contain exactly16362 source files')
    require(sum(r['category'] in ('gt', 'mask') for r in result.values()) == 16304,
            'GT/mask count differs')
    require(sum(r['bytes'] for r in result.values()) == 61749620933, 'Recovery source byte total differs')
    return result


def same_file(record, reference, label):
    require(record['relative_path'] == reference['relative_path'] and record['bytes'] == reference['bytes'] and
            record['sha256'] == reference['sha256'], label + ': source content differs')


def validate_transport_receipt(manifest, receipt, plan, inventory):
    require(manifest.get('schema') == 'flow-dsec-batch-v1', 'Bad transfer manifest schema')
    core = {k: v for k, v in manifest.items() if k not in ('batch_id', 'manifest_sha256')}
    msha = digest(core)
    require(manifest.get('manifest_sha256') == msha and manifest.get('batch_id') == msha[:24],
            'Transfer manifest digest differs')
    require(receipt.get('schema') == 'flow-dsec-receipt-v1' and receipt.get('status') == 'verified',
            'Transfer receipt not verified')
    for key in ('batch_id', 'manifest_sha256', 'source_root', 'target_root'):
        require(receipt.get(key) == manifest[key], 'Transfer receipt identity differs: ' + key)
    require(manifest['source_root'] == plan['source_root'] and manifest['target_root'] == plan['target_root'],
            'Transfer endpoint differs from recovery')
    expected = unique_records(manifest['files'], 'batch manifest')
    received = unique_records(receipt['files'], 'batch receipt')
    require(list(expected) == list(received), 'Transfer membership/order differs')
    require(receipt['file_count'] == len(received) and receipt['total_bytes'] == sum(r['bytes'] for r in received.values()),
            'Transfer receipt counts differ')
    nested = receipt['source_hashes']
    require(nested.get('schema') == 'flow-dsec-source-hashes-v1' and nested['manifest_sha256'] == msha and
            nested['files'] == receipt['files'] and receipt['source_hashes_sha256'] == digest(nested),
            'Source trailer receipt digest differs')
    for relative, row in received.items():
        require(relative in inventory, 'Transfer includes out-of-scope source file')
        same_file(row, inventory[relative], relative)
        require(expected[relative]['bytes'] == row['bytes'] and
                (not expected[relative].get('sha256') or expected[relative]['sha256'] == row['sha256']),
                'Transfer contradicts known source content')
    return received


def validate_recovery(root, directory, expected, audit):
    plan_path, status_path, inventory_path = (directory / n for n in ('recovery_plan.json', 'recovery_status.json', 'source_hashes.csv'))
    plan, status = json.loads(plan_path.read_text()), json.loads(status_path.read_text())
    inventory = load_hash_inventory(inventory_path)
    require(plan.get('schema') == 'flow_a_raw_recovery_v1' and plan['canonical_sha256'] == CANONICAL_SHA and
            plan['audit_sha256'] == AUDIT_SHA, 'Recovery provenance differs')
    require(Path(plan['target_root']).resolve() == root and plan['source_root'] == audit['remote_root'] and
            Path(plan['old_root']).resolve() != root and plan.get('old_voxels_included') is False,
            'Recovery root/old-voxel scope differs')
    require(plan['source_csv_sha256'] == audit['transfer_manifest']['sha256'], 'Recovery source inventory pin differs')
    planned = unique_records(plan['files'], 'recovery plan')
    require(set(planned) == set(inventory) == set(expected) and len(inventory) == 16362,
            'Recovery inventory exact membership differs')
    expected_bytes = sum(r['bytes'] for r in expected.values())
    require(plan['file_count'] == 16362 and plan['total_bytes'] == expected_bytes and
            plan['raw_events_bytes'] == audit['raw_required_left_events_bytes'], 'Recovery totals differ')
    for relative, row in expected.items():
        require(inventory[relative]['bytes'] == planned[relative]['bytes'] == row['bytes'], 'Recovery size differs')
        for known in (row['known_sha256'], planned[relative].get('sha256')):
            require(not known or inventory[relative]['sha256'] == known, 'Known source SHA differs: ' + relative)
    require(status.get('status') == READY_STATUS and status.get('failures') == [], 'Recovery not completed cleanly')
    links_path = directory / 'verified_local_links.json'
    links_doc = json.loads(links_path.read_text()); links = unique_records(links_doc['files'], 'verified local links')
    require(links_doc['count'] == len(links) and links_doc['bytes'] == sum(r['bytes'] for r in links.values()),
            'Hardlink receipt totals differ')
    for relative, row in links.items():
        require(relative in inventory, 'Hardlink outside recovery membership')
        same_file(row, inventory[relative], 'hardlink')
        require(planned[relative].get('sha256') == row['sha256'], 'Hardlink lacks prior source SHA in recovery plan')
    collected = dict(links); receipt_files = {}
    transferred_files = transferred_bytes = 0
    results = status.get('results', [])
    batch_ids = [r['batch_id'] for r in results]
    require(len(set(batch_ids)) == len(batch_ids), 'Duplicate recovery batch results')
    for batch in results:
        require(batch.get('status') == 'verified', 'Recovery batch incomplete')
        batch_id = batch['batch_id']; require(len(batch_id) == 24 and all(c in '0123456789abcdef' for c in batch_id), 'Unsafe batch ID')
        mp, rp = directory / 'manifests' / (batch_id + '.json'), directory / 'receipts' / (batch_id + '.json')
        manifest, receipt = json.loads(mp.read_text()), json.loads(rp.read_text())
        received = validate_transport_receipt(manifest, receipt, plan, inventory)
        require(not (set(collected) & set(received)), 'Recovery transfer/hardlink records overlap')
        require(batch['files'] == len(received) and batch['bytes'] == sum(r['bytes'] for r in received.values()),
                'Recovery result/receipt totals differ')
        collected.update(received); transferred_files += len(received); transferred_bytes += batch['bytes']
        receipt_files[str(mp)] = sha(mp); receipt_files[str(rp)] = sha(rp)
    require(set(collected) == set(inventory), 'Source receipt coverage is incomplete')
    require(status['linked_files'] == len(links) and status['transferred_verified_files'] == transferred_files and
            status['transferred_verified_bytes'] == transferred_bytes, 'Recovery status counters differ')
    evidence = {'recovery_plan_sha256': sha(plan_path), 'recovery_status_sha256': sha(status_path),
                'source_hash_inventory_sha256': sha(inventory_path), 'verified_local_links_sha256': sha(links_path),
                'transfer_evidence_files_sha256': receipt_files, 'linked_files': len(links),
                'transferred_files': transferred_files, 'source_file_count': len(inventory)}
    return inventory, evidence


def expected_shards(records, count=4):
    counts = {}
    for row in records:
        counts[row['sequence']] = counts.get(row['sequence'], 0) + 1
    loads, mapping = [0] * count, {}
    for sequence in sorted(counts, key=lambda s: (-counts[s], s)):
        worker = min(range(count), key=lambda i: (loads[i], i))
        mapping[sequence] = worker; loads[worker] += counts[sequence]
    return mapping, loads


def check_encoding_contract(contract):
    expected = {'encoding_id': ENCODING_ID, 'canonical_manifest_sha256': CANONICAL_SHA,
                'source_pins': SOURCE_PINS, 'script_sha256': GENERATOR_SHA, 'CPU_threads': 1,
                'bins': 10, 'shape': [10, 480, 640], 'dtype': 'float32',
                'time': '(t-t[0]).astype(float32); t/t[-1]; native convert_CHW renormalization',
                'spatial': 'all sliced events; native rectify and tap-wise validity',
                'polarity': 'signed 2p-1; no post-voxel z-score', 'slice': '[t_beg,t_end)'}
    require(set(contract) == set(expected) | {'torch', 'numpy', 'h5py'}, 'Encoding contract keys changed')
    for key, value in expected.items():
        require(contract.get(key) == value, 'Native encoding contract differs: ' + key)
    require(all(isinstance(contract[k], str) and contract[k] for k in ('torch', 'numpy', 'h5py')),
            'Missing generator runtime versions')


def validate_shard_membership(reports, records, root, source_inventory_sha):
    require(len(reports) == 4, 'Exactly four completed shard reports are required')
    mapping, loads = expected_shards(records)
    found, target_ids, common_contract = set(), set(), None
    for report in reports:
        shard = report['shard']; index = shard['index']
        require(type(index) is int and 0 <= index < 4 and index not in found, 'Duplicate/invalid shard index')
        found.add(index)
        require(report.get('status') == 'SHARD_COMPLETE' and report.get('GPU_initialized') is False and
                report.get('GT_mask_modified') is False, 'Shard not cleanly completed')
        require(report['encoding_id'] == ENCODING_ID and report['canonical_manifest_sha256'] == CANONICAL_SHA and
                report['source_pins'] == SOURCE_PINS and report['script_sha256'] == GENERATOR_SHA and
                report['raw_hash_inventory_sha256'] == source_inventory_sha and
                Path(report['output_root']).resolve() == root, 'Shard provenance/root differs')
        require(shard == {'index': index, 'count': 4, 'whole_sequence_assignment': mapping,
                          'target_loads': loads, 'global_canonical_targets': len(records)}, 'Shard assignment changed')
        check_encoding_contract(report['contract'])
        require(report['contract_sha256'] == digest(report['contract']), 'Shard encoding contract digest differs')
        if common_contract is None:
            common_contract = report['contract']
        require(report['contract'] == common_contract, 'Mixed generator source/runtime contracts')
        expected = [r for r in records if mapping[r['sequence']] == index]
        expected_ids = {r['target_id'] for r in expected}
        listed, receipts = report['target_ids'], report['records']
        actual = [r['target_id'] for r in receipts]
        require(report['target_count'] == len(expected_ids) == len(listed) == len(actual) and
                len(set(listed)) == len(listed) and len(set(actual)) == len(actual) and
                set(listed) == set(actual) == expected_ids, 'Shard target membership incomplete/duplicated')
        require(not target_ids & expected_ids, 'Overlapping shard targets')
        target_ids.update(expected_ids)
    require(target_ids == {r['target_id'] for r in records}, 'Global generated membership incomplete')
    return common_contract


def generated_members(root, records, reports, inventory, inventory_sha):
    import numpy as np
    contract = validate_shard_membership(reports, records, root, inventory_sha)
    contract_sha = digest(contract)
    by_id = {r['target_id']: r for r in records}
    timestamps = {}
    for seq in sorted({r['sequence'] for r in records}):
        rel = 'train_optical_flow/' + seq + '/flow/forward_timestamps.txt'
        timestamps[seq] = np.loadtxt(checked_path(root, rel), dtype=np.int64, delimiter=',')
    output, receipt_hashes = {}, {}
    for report in reports:
        local_raw = {}
        for seq in {by_id[t]['sequence'] for t in report['target_ids']}:
            for rel in ('train_events/' + seq + '/events/left/events.h5',
                        'train_events/' + seq + '/events/left/rectify_map.h5',
                        'train_optical_flow/' + seq + '/flow/forward_timestamps.txt'):
                local_raw[rel] = {k: inventory[rel][k] for k in ('bytes', 'sha256')}
        require(report['raw_files'] == local_raw, 'Shard raw provenance differs from full recovery inventory')
        for item in report['records']:
            target = item['target_id']; row = by_id[target]; seq = row['sequence']
            relative = 'saved_flow_data/event_tensors/10bins/left/' + seq + '/' + row['v2_target_filename']
            rp = checked_path(root, '.native_encoding_receipts/' + target + '.json')
            receipt = json.loads(rp.read_text()); normalized = {k: v for k, v in item.items() if k != 'reused'}
            require(normalized == receipt, 'Shard report differs from per-target receipt: ' + target)
            pair = timestamps[seq][row['target_index_1based'] - 1].tolist()
            expected_raw = {rel: local_raw[rel] for rel in (
                'train_events/' + seq + '/events/left/events.h5',
                'train_events/' + seq + '/events/left/rectify_map.h5',
                'train_optical_flow/' + seq + '/flow/forward_timestamps.txt')}
            require(receipt.get('schema_version') == 'flow_native_input_receipt_v1' and
                    receipt['contract_sha256'] == contract_sha and receipt['protocol_split'] == row['protocol_split'] and
                    receipt['relative_path'] == relative and receipt['timestamp_index_1based'] == row['target_index_1based'] and
                    receipt['timestamp_pair_us'] == pair and receipt['source_hashes'] == expected_raw,
                    'Generated receipt identity/source binding differs: ' + target)
            require(receipt['bytes'] == ARRAYS['input'][2] and receipt['shape'] == [10, 480, 640] and
                    receipt['dtype'] == 'float32' and receipt['finite'] is True and valid_sha(receipt['sha256']) and
                    valid_sha(receipt['tensor_C_order_bytes_sha256']), 'Generated tensor metadata differs')
            raw_slice = receipt['raw_slice_sha']
            require(set(raw_slice) == {'p', 'x', 'y', 't'} and
                    len({r['count'] for r in raw_slice.values()}) == 1 and raw_slice['t']['count'] >= 2 and
                    all(valid_sha(r['sha256_C_order_bytes']) and isinstance(r['dtype'], str) for r in raw_slice.values()),
                    'Missing or inconsistent raw-slice evidence')
            require(relative not in output, 'Duplicate generated file')
            receipt_hash = sha(rp); receipt_hashes[target] = receipt_hash
            output[relative] = {'relative_path': relative, 'category': 'input', 'split': row['protocol_split'],
                                 'target': target, 'sequence': seq, 'bytes': receipt['bytes'], 'sha256': receipt['sha256'],
                                 'tensor_sha256': receipt['tensor_C_order_bytes_sha256'],
                                 'generation_receipt_sha256': receipt_hash,
                                 'proof_kind': 'generated_native_input_receipt'}
    return output, contract, receipt_hashes


def stable_stat(a, b):
    return (a.st_dev, a.st_ino, a.st_size, a.st_mtime_ns, a.st_ctime_ns) == \
           (b.st_dev, b.st_ino, b.st_size, b.st_mtime_ns, b.st_ctime_ns)


def verify_file(root, row):
    """Read every byte; NPY parse and finite checks use the same hashed bytes."""
    import numpy as np
    result = {'relative_path': row['relative_path'], 'category': row['category'], 'passed': False}
    try:
        path = checked_path(root, row['relative_path'])
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as f:
            before = os.fstat(f.fileno())
            require(stat.S_ISREG(before.st_mode) and before.st_size == row['bytes'], 'File type/size changed')
            if row['category'] in ARRAYS:
                raw = f.read()
                require(len(raw) == before.st_size, 'Short NPY read')
                observed_sha = hashlib.sha256(raw).hexdigest()
            else:
                h = hashlib.sha256()
                for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                    h.update(block)
                observed_sha = h.hexdigest()
            require(stable_stat(before, os.fstat(f.fileno())), 'File changed during verification')
        require(observed_sha == row['sha256'], 'Current full SHA differs from receipt/source')
        result.update(bytes=before.st_size, sha256=observed_sha)
        if row['category'] in ARRAYS:
            shape, dtype, _ = ARRAYS[row['category']]
            header = io.BytesIO(raw); version = np.lib.format.read_magic(header)
            require(version in ((1, 0), (2, 0)), 'Unsupported NPY version')
            actual_shape, fortran, actual_dtype = (np.lib.format.read_array_header_1_0(header) if version == (1, 0)
                                                  else np.lib.format.read_array_header_2_0(header))
            header_bytes = header.tell()
            require(actual_shape == shape and str(actual_dtype) == dtype and actual_dtype.isnative and
                    not actual_dtype.hasobject and fortran is False, 'NPY header shape/dtype/layout differs')
            require(header_bytes + int(np.prod(shape)) * actual_dtype.itemsize == len(raw), 'NPY trailing/truncated payload')
            stream = io.BytesIO(raw); array = np.load(stream, allow_pickle=False)
            require(stream.tell() == len(raw) and array.shape == shape and np.isfinite(array).all(),
                    'NPY payload contains nonfinite/trailing content')
            tensor_sha = hashlib.sha256(memoryview(raw)[header_bytes:]).hexdigest()
            require(not row.get('tensor_sha256') or row['tensor_sha256'] == tensor_sha, 'Tensor SHA differs from generation receipt')
            result.update(npy_header={'shape': list(shape), 'dtype': dtype, 'fortran_order': False,
                                     'version': list(version), 'header_bytes': header_bytes},
                          all_elements_finite=True, tensor_C_order_bytes_sha256=tensor_sha)
        result['passed'] = True
    except Exception as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
    return result


def publish_outputs(directory, rows, results, contract, provenance, root):
    """No success marker is created unless exact membership and every check pass."""
    require(len(rows) == 24514 and sum(r['category'] == 'input' for r in rows.values()) == 8152 and
            sum(r['category'] == 'gt' for r in rows.values()) == 8152 and
            sum(r['category'] == 'mask' for r in rows.values()) == 8152,
            'Refuse acceptance publication: global source/input counts differ')
    require(len(results) == len(rows) and {r['relative_path'] for r in results} == set(rows) and
            len({r['relative_path'] for r in results}) == len(results) and all(r.get('passed') is True for r in results),
            'Refuse acceptance publication: incomplete/failed verification')
    directory.mkdir(parents=True, exist_ok=True)
    require(not (directory / 'acceptance.json').exists(), 'Acceptance already published; use a new output directory')
    by_path = {r['relative_path']: r for r in results}
    ordered = sorted(rows)
    csv_path, inventory_path = directory / 'native_deployment_manifest.csv', directory / 'verified_hash_inventory.csv'
    fieldnames = ['relative_path', 'category', 'split', 'target', 'sequence', 'bytes', 'sha256',
                  'proof_kind', 'generation_receipt_sha256']
    for path, fields in ((csv_path, fieldnames), (inventory_path, ['relative_path', 'bytes', 'sha256'])):
        tmp = path.with_name(path.name + '.tmp')
        with tmp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            for rel in ordered:
                item = dict(rows[rel]); item.update(bytes=by_path[rel]['bytes'], sha256=by_path[rel]['sha256'])
                writer.writerow({key: item.get(key, '') for key in fields})
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    categories = {key: sum(r['category'] == key for r in rows.values()) for key in sorted({r['category'] for r in rows.values()})}
    counts = {'files': len(rows), 'bytes': sum(r['bytes'] for r in rows.values()),
              'generated_inputs': categories.get('input', 0), 'recovered_source_files': len(rows) - categories.get('input', 0),
              'npy_headers_and_all_finite_checked': sum('npy_header' in r for r in results),
              'full_file_SHA_checked': len(results), 'categories': categories}
    manifest = {'schema': 'flow-native-deployment-v1', 'encoding_id': ENCODING_ID,
                'canonical_manifest_sha256': CANONICAL_SHA, 'encoding_contract': contract,
                'encoding_contract_sha256': digest(contract), 'counts': counts,
                'files_csv': csv_path.name, 'files_csv_sha256': sha(csv_path),
                'hash_inventory': inventory_path.name, 'hash_inventory_sha256': sha(inventory_path),
                'npy_contracts': {k: {'shape': list(v[0]), 'dtype': v[1]} for k, v in ARRAYS.items()},
                'provenance': provenance,
                'scope': 'Raw/meta/GT/mask match source transfer evidence; native input NPYs match new generation receipts. These new voxels are not claimed to be archived legacy files.'}
    manifest_path = directory / 'native_deployment_manifest.json'; atomic_json(manifest_path, manifest)
    atomic_json(directory / 'verification_files.json', {'schema': 'flow-native-file-verification-v1', 'files': results})
    acceptance = {'schema': 'flow-native-dataset-acceptance-v1', 'status': 'VERIFIED_NATIVE_DATASET',
                  'passed': True, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'root': str(root), 'encoding_id': ENCODING_ID, 'canonical_manifest_sha256': CANONICAL_SHA,
                  'encoding_contract_sha256': digest(contract), 'counts': counts,
                  'canonical_membership_complete': True,
                  'all_generated_inputs_match_generation_receipts': True,
                  'all_recovered_source_files_match_verified_transfer_hashes': True,
                  'all_NPY_headers_shapes_dtypes_and_finite_verified': True,
                  'deployment_manifest': manifest_path.name, 'deployment_manifest_sha256': sha(manifest_path),
                  'deployment_csv_sha256': sha(csv_path),
                  'hash_inventory': inventory_path.name, 'hash_inventory_sha256': sha(inventory_path),
                  'verification_files_sha256': sha(directory / 'verification_files.json'),
                  'validator_sha256': sha(__file__), 'CPU_only': True, 'training_started': False,
                  'legacy_validator_schema_compatible': False}
    atomic_json(directory / 'acceptance.json', acceptance)
    return acceptance


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'canonical', 'recovery-dir', 'output-dir'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--shard-report', type=Path, action='append', required=True)
    p.add_argument('--archive-audit', type=Path, help='Default: recovery-dir.parent/archive_data_audit.json')
    p.add_argument('--threads', type=int, choices=(1, 2), default=2)
    a = p.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.resolve())
    a.shard_report = [p.resolve() for p in a.shard_report]
    a.archive_audit = a.archive_audit or a.recovery_dir.parent / 'archive_data_audit.json'
    require(len(a.shard_report) == 4 and len(set(a.shard_report)) == 4, 'Need four distinct shard reports')
    return a


def main():
    a = parse_args(); started = time.monotonic(); a.output_dir.mkdir(parents=True, exist_ok=True)
    require(not (a.output_dir / 'acceptance.json').exists(), 'Use a new output-dir; existing acceptance is immutable')
    locks = []
    with (a.output_dir / '.validator.lock').open('a') as output_lock:
        fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            records = canonical_members(a.canonical)
            audit = pinned_json(a.archive_audit, AUDIT_SHA)
            expected_sources = source_members(records, audit)
            source_inventory, recovery_evidence = validate_recovery(a.root, a.recovery_dir, expected_sources, audit)
            for sequence in sorted({r['sequence'] for r in records}):
                lock = checked_path(a.root, '.native_encoding_receipts/.sequence_' + sequence + '.lock').open('rb')
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(lock)
            reports = [json.loads(path.read_text()) for path in a.shard_report]
            generated, contract, generation_receipt_hashes = generated_members(
                a.root, records, reports, source_inventory, recovery_evidence['source_hash_inventory_sha256'])
            all_rows = dict(generated)
            for relative, row in expected_sources.items():
                require(relative not in all_rows, 'Recovered source overlaps generated inputs')
                all_rows[relative] = dict(row, sha256=source_inventory[relative]['sha256'],
                                          proof_kind='verified_source_transfer')
            require(len(all_rows) == 24514 and len(generated) == 8152, 'Full native dataset size differs')
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.threads) as pool:
                for index, result in enumerate(pool.map(lambda r: verify_file(a.root, r),
                                                       (all_rows[p] for p in sorted(all_rows))), 1):
                    results.append(result)
                    if index % 500 == 0 or index == len(all_rows):
                        print(json.dumps({'checked': index, 'total': len(all_rows),
                                          'failed': sum(not r['passed'] for r in results)}), flush=True)
            failures = [r for r in results if not r['passed']]
            if failures:
                atomic_json(a.output_dir / 'failed_files.json', {'files': failures})
                raise ValueError('Native dataset verification failed for%d files' % len(failures))
            provenance = {'canonical_manifest_sha256': CANONICAL_SHA, 'archive_audit_sha256': AUDIT_SHA,
                          'recovery': recovery_evidence, 'generator_sha256': GENERATOR_SHA,
                          'generation_reports_sha256': {str(p): sha(p) for p in a.shard_report},
                          'generation_receipts_SHA_inventory_digest': digest(generation_receipt_hashes),
                          'generator_source_pins': SOURCE_PINS, 'validator_sha256': sha(__file__),
                          'verification_threads': a.threads, 'elapsed_seconds': time.monotonic() - started}
            acceptance = publish_outputs(a.output_dir, all_rows, results, contract, provenance, a.root)
            print(json.dumps(acceptance, indent=2))
        except BaseException as exc:
            atomic_json(a.output_dir / 'verification_failure.json', {'schema': 'flow-native-verification-failure-v1',
                        'passed': False, 'exception': type(exc).__name__, 'message': str(exc),
                        'elapsed_seconds': time.monotonic() - started, 'acceptance_published': False})
            raise
        finally:
            for lock in locks:
                lock.close()


if __name__ == '__main__':
    main()
