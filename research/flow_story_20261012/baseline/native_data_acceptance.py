"""Strict new-native acceptance adapter; no torch/GPU import or data mutation."""
import csv
import hashlib
import io
import json
from pathlib import Path
import stat

CANONICAL_SHA = '6829f9ac8181f8eb5e7ce606c23915c2109c28447a0342bde8696cda045a7b22'
AUDIT_SHA = '888bdbb22a67a06d4baedcfd0adf957235d998e5ba0685d800519157fb33ceb9'
VALIDATOR_SHA = '49129001d24df33048d25080e81d8bcd3f0ac7615455d9e38725b9bd6c7fef9e'
GENERATOR_SHA = 'd0d45517cc8a43b59c02afafd030bb976de0656dd4876b3286f522c59abe4251'
ENCODING_ID = 'sdformerflow_upstream_voxel_10bins_cpu_v1'
SOURCE_PINS = {
    'DSEC_dataloader/event_representations.py': 'f02493e7440e12828b487b272d0c17f96a1a2a6c785fd8a6764981b09825403a',
    'DSEC_dataloader/DSEC_dataset_preprocess.py': '63fa4eaf500910d798e2d81fc17de85a0d4b6cd52f0ae6ade4d841be16256294',
}
ARRAYS = {'input': ([10, 480, 640], 'float32', 12288128),
          'gt': ([2, 480, 640], 'float32', 2457728),
          'mask': ([480, 640], 'bool', 307328)}
COUNTS = {'files': 24514, 'bytes': 161922440389, 'generated_inputs': 8152,
          'recovered_source_files': 16362, 'npy_headers_and_all_finite_checked': 24456,
          'full_file_SHA_checked': 24514}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_parts(relative):
    require(isinstance(relative, str) and relative and '\\' not in relative and '\x00' not in relative,
            'Unsafe relative path')
    parts = relative.split('/')
    require(not relative.startswith('/') and all(p not in ('', '.', '..') for p in parts), 'Unsafe path components')
    return parts


def checked_path(root, relative):
    parts = safe_parts(relative)
    path = Path(root)
    for part in parts:
        path = path / part
        require(not path.is_symlink(), 'Symlink not accepted: ' + str(path))
    require(path.resolve().is_relative_to(Path(root).resolve()), 'Dataset path escapes root')
    return path


def read_pinned(path, expected):
    require(valid_sha(expected), 'A complete explicit artifact SHA is required')
    data = Path(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, 'Artifact SHA mismatch: ' + str(path))
    return data


def pinned_json(path, expected):
    return json.loads(read_pinned(path, expected))


def canonical_records(path):
    records = pinned_json(path, CANONICAL_SHA)['records']
    ids = [r['target_id'] for r in records]
    require(len(ids) == len(set(ids)) == 8152 and
            sum(r['protocol_split'] == 'train' for r in records) == 6000 and
            sum(r['protocol_split'] == 'valid' for r in records) == 2152,
            'Exact canonical6000/2152 membership required')
    require(len({r['sequence'] for r in records}) == 18, 'Canonical sequence count changed')
    return records


def target_files(row):
    return {'input': 'saved_flow_data/event_tensors/10bins/left/' + row['sequence'] + '/' + row['v2_target_filename'],
            'gt': 'saved_flow_data/gt_tensors/' + row['gt_filename'],
            'mask': 'saved_flow_data/mask_tensors/' + row['mask_filename']}


def expected_members(records, audit):
    result = {}
    for r in records:
        for category, relative in target_files(r).items():
            require(relative not in result, 'Canonical file alias')
            result[relative] = {'category': category, 'split': r['protocol_split'], 'target': r['target_id'],
                                'sequence': r['sequence'], 'bytes': ARRAYS[category][2], 'known_sha256': None}
    remote = Path(audit['remote_root'])
    require(len(audit['raw_sequences']) == 18 and
            {r['sequence'] for r in audit['raw_sequences']} == {r['sequence'] for r in records}, 'Raw sequence scope differs')
    for r in audit['raw_sequences']:
        item = r['left_events']; relative = str(Path(item['path']).relative_to(remote))
        require(relative not in result, 'Duplicate raw member')
        result[relative] = {'category': 'raw_left_events', 'split': 'source', 'target': '',
                            'sequence': r['sequence'], 'bytes': item['bytes'], 'known_sha256': None}
    for item in audit['phase1_metadata_transfer']['files']:
        relative = item['relative_path']; require(relative not in result, 'Duplicate metadata')
        result[relative] = {'category': item['category'], 'split': 'source', 'target': item.get('target', ''),
                            'sequence': item.get('sequence', ''), 'bytes': item['bytes'], 'known_sha256': item.get('sha256')}
    for sample in audit['sample_content_checks']:
        for category in ('gt', 'mask'):
            item = sample['arrays'][category]; relative = str(Path(item['path']).relative_to(remote))
            require(result[relative]['bytes'] == item['bytes'], 'Audited known file size differs')
            result[relative]['known_sha256'] = item['sha256']
    require(len(result) == 24514 and sum(r['bytes'] for r in result.values()) == 161922440389, 'New native inventory scope changed')
    return result


def validate_encoding(contract):
    expected = {'encoding_id': ENCODING_ID, 'canonical_manifest_sha256': CANONICAL_SHA,
                'source_pins': SOURCE_PINS, 'script_sha256': GENERATOR_SHA, 'CPU_threads': 1,
                'bins': 10, 'shape': [10, 480, 640], 'dtype': 'float32',
                'time': '(t-t[0]).astype(float32); t/t[-1]; native convert_CHW renormalization',
                'spatial': 'all sliced events; native rectify and tap-wise validity',
                'polarity': 'signed 2p-1; no post-voxel z-score', 'slice': '[t_beg,t_end)'}
    require(set(contract) == set(expected) | {'torch', 'numpy', 'h5py'}, 'Encoding contract fields differ')
    require(all(contract[k] == value for k, value in expected.items()), 'Wrong native encoding or generator/source pin')
    require(all(isinstance(contract[k], str) and contract[k] for k in ('torch', 'numpy', 'h5py')), 'Missing generator runtime')


def unique_rows(rows, label):
    out = {}
    for row in rows:
        relative = row['relative_path']
        safe_parts(relative)
        require(relative not in out, label + ': duplicate relative path')
        out[relative] = row
    return out


def _artifact_path(directory, name, required_name):
    require(name == required_name, 'Unexpected acceptance artifact name')
    return checked_path(directory, name)


def validate_native_bundle(root, canonical, acceptance_path, acceptance_sha, archive_audit, check_current_files=True):
    """Independently validate new artifacts; formal entry always checks files.

    Does not rehash161.9GB: relies on pinned independent complete acceptance,
    inspects all24514 current file types/sizes, then Dataset rehashes and checks
    finite/shape/dtype on EVERY actual training sample read.
    False is reserved for local artifact-only audit/tests, never training.
    """
    root = Path(root).resolve(); acceptance_path = Path(acceptance_path)
    accepted = pinned_json(acceptance_path, acceptance_sha)
    require(accepted.get('schema') == 'flow-native-dataset-acceptance-v1' and
            accepted.get('status') == 'VERIFIED_NATIVE_DATASET' and accepted.get('passed') is True,
            'New native acceptance required; legacy phase1/engineering markers are forbidden')
    require(Path(accepted['root']).resolve() == root, 'Acceptance root differs')
    require(accepted.get('canonical_manifest_sha256') == CANONICAL_SHA and accepted.get('encoding_id') == ENCODING_ID and
            accepted.get('validator_sha256') == VALIDATOR_SHA, 'Acceptance canonical/encoding/validator pin differs')
    for key in ('canonical_membership_complete', 'all_generated_inputs_match_generation_receipts',
                'all_recovered_source_files_match_verified_transfer_hashes', 'all_NPY_headers_shapes_dtypes_and_finite_verified', 'CPU_only'):
        require(accepted.get(key) is True, 'Incomplete independent acceptance: ' + key)
    require(accepted.get('training_started') is False and accepted.get('legacy_validator_schema_compatible') is False,
            'Unexpected acceptance/training schema state')
    require(all(type(accepted['counts'].get(k)) is int and accepted['counts'][k] == value for k, value in COUNTS.items()),
            'Acceptance global counts incomplete')
    directory = acceptance_path.parent
    mp = _artifact_path(directory, accepted['deployment_manifest'], 'native_deployment_manifest.json')
    manifest = pinned_json(mp, accepted['deployment_manifest_sha256'])
    require(manifest.get('schema') == 'flow-native-deployment-v1' and manifest.get('encoding_id') == ENCODING_ID and
            manifest.get('canonical_manifest_sha256') == CANONICAL_SHA and manifest['counts'] == accepted['counts'],
            'Wrong native deployment schema/counts')
    validate_encoding(manifest['encoding_contract'])
    require(digest(manifest['encoding_contract']) == manifest['encoding_contract_sha256'] == accepted['encoding_contract_sha256'],
            'Encoding contract digest mismatch')
    require(manifest['npy_contracts'] == {k: {'shape': s, 'dtype': dt} for k, (s, dt, _) in ARRAYS.items()}, 'NPY contract changed')
    prov = manifest['provenance']; recovery = prov['recovery']
    require(prov.get('canonical_manifest_sha256') == CANONICAL_SHA and prov.get('archive_audit_sha256') == AUDIT_SHA and
            prov.get('generator_sha256') == GENERATOR_SHA and prov.get('validator_sha256') == VALIDATOR_SHA and
            prov.get('generator_source_pins') == SOURCE_PINS, 'Independent provenance source pins differ')
    require(recovery.get('source_file_count') == 16362 and
            recovery.get('linked_files', -1) + recovery.get('transferred_files', -1) == 16362,
            'Source receipt coverage incomplete')
    for key in ('recovery_plan_sha256', 'recovery_status_sha256', 'source_hash_inventory_sha256', 'verified_local_links_sha256'):
        require(valid_sha(recovery.get(key)), 'Source recovery proof digest missing')
    reports = prov['generation_reports_sha256']
    require(len(reports) == 4 and all(valid_sha(h) for h in reports.values()) and
            valid_sha(prov.get('generation_receipts_SHA_inventory_digest')), 'Generation report/receipt coverage proof missing')
    csv_path = _artifact_path(directory, manifest['files_csv'], 'native_deployment_manifest.csv')
    require(manifest['files_csv_sha256'] == accepted['deployment_csv_sha256'], 'Deployment CSV digest binding differs')
    rows = list(csv.DictReader(io.StringIO(read_pinned(csv_path, manifest['files_csv_sha256']).decode())))
    deployment = unique_rows(rows, 'deployment')
    inventory_path = _artifact_path(directory, accepted['hash_inventory'], 'verified_hash_inventory.csv')
    require(manifest['hash_inventory'] == inventory_path.name and manifest['hash_inventory_sha256'] == accepted['hash_inventory_sha256'],
            'Hash inventory binding differs')
    hash_rows = list(csv.DictReader(io.StringIO(read_pinned(inventory_path, accepted['hash_inventory_sha256']).decode())))
    inventory = unique_rows(hash_rows, 'hash inventory')
    vp = checked_path(directory, 'verification_files.json')
    verification = pinned_json(vp, accepted['verification_files_sha256'])
    require(verification.get('schema') == 'flow-native-file-verification-v1', 'Wrong verification-file schema')
    verified = unique_rows(verification['files'], 'verified file records')
    records = canonical_records(canonical); audit = pinned_json(archive_audit, AUDIT_SHA)
    expected = expected_members(records, audit)
    require(set(deployment) == set(inventory) == set(verified) == set(expected), 'Native exact membership differs')
    categories = {c: sum(r['category'] == c for r in expected.values()) for c in {r['category'] for r in expected.values()}}
    require(accepted['counts']['categories'] == categories, 'Native category counts differ')
    clean_hashes = {}
    for relative, exp in expected.items():
        row, hashed, checked = deployment[relative], inventory[relative], verified[relative]
        require(all(row.get(k) == exp[k] for k in ('category', 'split', 'target', 'sequence')), 'Member role/target differs: ' + relative)
        require(int(row['bytes']) == int(hashed['bytes']) == checked.get('bytes') == exp['bytes'], 'Member bytes differ')
        require(valid_sha(hashed['sha256']) and row['sha256'] == checked.get('sha256') == hashed['sha256'] and
                checked.get('passed') is True and checked.get('category') == exp['category'], 'Unverified member SHA/category')
        require(not exp['known_sha256'] or exp['known_sha256'] == hashed['sha256'], 'Previously known source SHA differs')
        if exp['category'] == 'input':
            require(row['proof_kind'] == 'generated_native_input_receipt' and valid_sha(row['generation_receipt_sha256']) and
                    valid_sha(checked.get('tensor_C_order_bytes_sha256')), 'New input generation proof missing')
        else:
            require(row['proof_kind'] == 'verified_source_transfer' and not row['generation_receipt_sha256'], 'Recovery proof category differs')
        if exp['category'] in ARRAYS:
            shape, dtype, _ = ARRAYS[exp['category']]; header = checked.get('npy_header', {})
            require(checked.get('all_elements_finite') is True and header.get('shape') == shape and
                    header.get('dtype') == dtype and header.get('fortran_order') is False and
                    header.get('version') in ([1, 0], [2, 0]), 'NPY finite/header acceptance missing: ' + relative)
        if check_current_files:
            path = checked_path(root, relative); current = path.stat()
            require(stat.S_ISREG(current.st_mode) and current.st_size == exp['bytes'], 'Accepted data file now missing/wrong size')
        clean_hashes[relative] = {'bytes': exp['bytes'], 'sha256': hashed['sha256']}
    train = [r for r in records if r['protocol_split'] == 'train']
    train_hashes = {rel: clean_hashes[rel] for r in train for rel in target_files(r).values()}
    binding = {'schema': 'flow_native_training_data_contract_v1', 'encoding_id': ENCODING_ID,
               'canonical_manifest_sha256': CANONICAL_SHA, 'archive_audit_sha256': AUDIT_SHA,
               'data_acceptance_sha256': acceptance_sha, 'deployment_manifest_sha256': accepted['deployment_manifest_sha256'],
               'deployment_csv_sha256': accepted['deployment_csv_sha256'], 'reference_hashes_sha256': accepted['hash_inventory_sha256'],
               'verification_files_sha256': accepted['verification_files_sha256'],
               'encoding_contract_sha256': accepted['encoding_contract_sha256'], 'encoding_contract': manifest['encoding_contract'],
               'validator_sha256': VALIDATOR_SHA, 'generator_sha256': GENERATOR_SHA,
               'all_accepted_files': 24514, 'train_targets': 6000, 'valid_targets': 2152,
               'train_files': len(train_hashes), 'all_NPY_finite_verified': True,
               'sample_SHA_shape_dtype_finite_checked_on_every_Dataset_read': True}
    return train, train_hashes, binding, {'current_file_stat_checked': check_current_files,
                                        'accepted_files': len(expected), 'train_files': len(train_hashes)}
