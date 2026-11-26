#!/usr/bin/env python3
"""CPU-only, read-only dataset checks; only reports/hash inventories are written.

quick: all existence/size + every previously known SHA256.
header: quick plus every NPY header shape/dtype/payload length.
fullhash: header plus SHA256 of every file, optionally compared to an independent
source inventory. A newly generated inventory alone is NOT source equivalence.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import collections
import concurrent.futures
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import time


def sha256(path, block_bytes=8*1024*1024):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(block_bytes), b''):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path):
    meta = json.loads(path.read_text())
    if meta['schema_version'] != 1:
        raise ValueError('unsupported manifest schema')
    csv_name = Path(meta['files_csv'])
    if csv_name.name != str(csv_name):
        raise ValueError('manifest files_csv must be a sibling filename')
    csv_path = path.parent/csv_name
    if sha256(csv_path) != meta['files_csv_sha256']:
        raise ValueError('deployment CSV hash mismatch')
    rows = list(csv.DictReader(csv_path.open(newline='')))
    seen = set()
    for row in rows:
        rel = PurePosixPath(row['relative_path'])
        if rel.is_absolute() or '..' in rel.parts or str(rel) != row['relative_path']:
            raise ValueError('unsafe or noncanonical relative path')
        if row['relative_path'] in seen:
            raise ValueError('duplicate file in manifest')
        seen.add(row['relative_path'])
        row['bytes'] = int(row['bytes'])
        if row['bytes'] < 0:
            raise ValueError('negative expected file size')
        if row['known_sha256'] and not re.fullmatch('[0-9a-f]{64}', row['known_sha256']):
            raise ValueError('invalid known SHA256')
        if row['relative_path'].endswith('.npy') and row['category'] not in meta['npy_contracts']:
            raise ValueError('NPY file has no shape/dtype contract')
    if len(rows) != meta['expected_files'] or sum(r['bytes'] for r in rows) != meta['expected_bytes']:
        raise ValueError('manifest count or byte total mismatch')
    if sum(bool(r['known_sha256']) for r in rows) != meta['expected_known_hashes']:
        raise ValueError('known hash count mismatch')
    for key, column in [('expected_by_split', 'split'), ('expected_by_category', 'category')]:
        if dict(collections.Counter(r[column] for r in rows)) != meta[key]:
            raise ValueError('manifest membership counts changed: '+key)
    return meta, rows, csv_path


def load_reference(path, rows):
    refs = {}
    for row in csv.DictReader(path.open(newline='')):
        rel = row['relative_path']
        if rel in refs or not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError('duplicate or invalid reference hash')
        refs[rel] = {'sha256': row['sha256'], 'bytes': int(row['bytes'])}
    if set(refs) != {r['relative_path'] for r in rows}:
        raise ValueError('reference inventory must cover exactly every deployment file')
    for row in rows:
        ref = refs[row['relative_path']]
        if ref['bytes'] != row['bytes']:
            raise ValueError('reference byte count disagrees with sealed deployment manifest')
        if row['known_sha256'] and row['known_sha256'] != ref['sha256']:
            raise ValueError('reference conflicts with previously audited trusted SHA256')
    return refs


def npy_header(path, contract):
    import numpy as np
    with path.open('rb') as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError('unsupported NPY header version '+repr(version))
        offset = f.tell()
    if list(shape) != contract['shape'] or dtype != np.dtype(contract['dtype']) or dtype.hasobject:
        raise ValueError('NPY shape/dtype mismatch: '+repr((shape, str(dtype))))
    payload_bytes = math.prod(shape)*dtype.itemsize
    if offset+payload_bytes != path.stat().st_size:
        raise ValueError('NPY truncated payload or unexpected trailing bytes')
    return {'shape': list(shape), 'dtype': str(dtype), 'fortran_order': fortran,
            'header_bytes': offset, 'payload_bytes': payload_bytes}


def finite_array(path, chunk_bytes):
    import numpy as np
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    flat = a.ravel(order='K')
    per_chunk = max(1, chunk_bytes//a.dtype.itemsize)
    nonfinite = 0
    for begin in range(0, flat.size, per_chunk):
        nonfinite += int(np.count_nonzero(~np.isfinite(flat[begin:begin+per_chunk])))
    return {'checked_elements': int(a.size), 'nonfinite_elements': nonfinite}


def check_file(row, root, meta, mode, finite, chunk_bytes, reference):
    result = {'relative_path': row['relative_path'], 'category': row['category'],
              'split': row['split'], 'expected_bytes': row['bytes'], 'errors': []}
    path = root/row['relative_path']
    try:
        resolved = path.resolve()
        if os.path.commonpath([str(root), str(resolved)]) != str(root):
            raise ValueError('file symlink escapes supplied dataset root')
        if not path.is_file():
            raise FileNotFoundError('expected regular file missing')
        before = path.stat()
        result['exists'] = True
        result['observed_bytes'] = before.st_size
        if before.st_size != row['bytes']:
            raise ValueError('file size mismatch')
        is_npy = row['relative_path'].endswith('.npy')
        do_finite = is_npy and (finite == 'all' or (finite == 'sample' and bool(row['known_sha256'])))
        if is_npy and (mode != 'quick' or do_finite):
            result['npy_header'] = npy_header(path, meta['npy_contracts'][row['category']])
        if do_finite:
            result['finite'] = finite_array(path, chunk_bytes)
            if result['finite']['nonfinite_elements']:
                result['errors'].append('nonfinite NPY values')
        if mode == 'fullhash' or row['known_sha256']:
            value = sha256(path)
            result['sha256'] = value
            if row['known_sha256']:
                result['known_hash_matches'] = value == row['known_sha256']
                if not result['known_hash_matches']:
                    result['errors'].append('previously audited SHA256 mismatch')
            if reference is not None:
                result['source_hash_matches'] = value == reference[row['relative_path']]['sha256']
                if not result['source_hash_matches']:
                    result['errors'].append('source SHA256 mismatch')
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            result['errors'].append('file changed during validation')
    except Exception as e:
        result['errors'].append(type(e).__name__+': '+str(e))
    result['passed'] = not result['errors']
    return result


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True, help='DSEC root containing saved_flow_data and train_*')
    p.add_argument('--manifest', type=Path, default=Path(__file__).with_name('phase1_deployment_manifest.json'))
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--mode', choices=['quick', 'header', 'fullhash'], default='header')
    p.add_argument('--finite', choices=['none', 'sample', 'all'], default='sample', help='sample=12 previously hashed NPY files; all scans every NPY payload')
    p.add_argument('--threads', type=int, choices=[1, 2], default=1)
    p.add_argument('--finite-chunk-mib', type=int, default=8)
    p.add_argument('--reference-hashes', type=Path, help='complete source inventory CSV; requires fullhash')
    p.add_argument('--hashes-output', type=Path, help='fullhash only; default REPORT_STEM.sha256.csv')
    args = p.parse_args()
    if args.finite_chunk_mib < 1 or args.finite_chunk_mib > 64:
        p.error('--finite-chunk-mib must be between 1 and 64')
    if (args.reference_hashes or args.hashes_output) and args.mode != 'fullhash':
        p.error('--reference-hashes/--hashes-output require --mode fullhash')
    started = time.monotonic()
    safe_report = False
    try:
        meta, rows, csv_path = load_manifest(args.manifest)
        reference = load_reference(args.reference_hashes, rows) if args.reference_hashes else None
        root = args.root.resolve()
        if not root.is_dir():
            raise ValueError('dataset root is not an existing directory')
        hashes_output = args.hashes_output or args.report.with_name(args.report.stem+'.sha256.csv')
        inputs = {args.manifest.resolve(), csv_path.resolve()}
        if args.reference_hashes:
            inputs.add(args.reference_hashes.resolve())
        outputs = [args.report.resolve()]
        if args.mode == 'fullhash':
            outputs.append(hashes_output.resolve())
        if len(set(outputs)) != len(outputs) or any(x in inputs for x in outputs):
            raise ValueError('report/hash output would overwrite an input or each other')
        protected_data = {root/r['relative_path'] for r in rows}
        if any(x in protected_data for x in outputs):
            raise ValueError('report/hash output would overwrite dataset file')
        safe_report = True
        if args.mode != 'quick' or args.finite != 'none':
            import numpy  # Import once before worker threads; no Torch/CUDA imports.
        results = []
        def work(row):
            return check_file(row, root, meta, args.mode, args.finite,
                              args.finite_chunk_mib*1024*1024, reference)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as pool:
            for i, result in enumerate(pool.map(work, rows), 1):
                results.append(result)
                if i % 500 == 0 or i == len(rows):
                    print('checked {}/{} files; failures={}'.format(i, len(rows), sum(not r['passed'] for r in results)), file=sys.stderr, flush=True)
        counts = {
            'expected_files': len(rows), 'passed_files': sum(r['passed'] for r in results),
            'failed_files': sum(not r['passed'] for r in results),
            'existing_files': sum(r.get('exists', False) for r in results),
            'expected_bytes': meta['expected_bytes'],
            'observed_bytes': sum(r.get('observed_bytes', 0) for r in results),
            'npy_headers_checked': sum('npy_header' in r for r in results),
            'finite_arrays_checked': sum('finite' in r for r in results),
            'hashes_computed': sum('sha256' in r for r in results),
            'previously_known_hashes_verified': sum(r.get('known_hash_matches', False) for r in results),
            'source_hashes_verified': sum(r.get('source_hash_matches', False) for r in results)}
        passed = counts['failed_files'] == 0 and counts['previously_known_hashes_verified'] == meta['expected_known_hashes']
        inventory = None
        if args.mode == 'fullhash' and counts['hashes_computed'] == len(rows) and passed:
            hashes_output.parent.mkdir(parents=True, exist_ok=True)
            tmp = hashes_output.with_name(hashes_output.name+'.tmp')
            with tmp.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['relative_path', 'bytes', 'sha256'])
                writer.writeheader()
                for r in results:
                    writer.writerow({'relative_path': r['relative_path'], 'bytes': r['observed_bytes'], 'sha256': r['sha256']})
            tmp.replace(hashes_output)
            inventory = {'path': str(hashes_output.resolve()), 'sha256': sha256(hashes_output), 'files': len(rows)}
        report = {
            'schema_version': 1, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'root': str(root), 'mode': args.mode, 'finite_mode': args.finite,
            'threads': args.threads, 'CPU_only': True, 'source_data_modified': False,
            'passed_requested_checks': passed,
            'complete_source_destination_hash_equivalence_verified': passed and reference is not None and counts['source_hashes_verified'] == len(rows),
            'all_file_hash_inventory_generated': inventory is not None,
            'scope_warning': 'Existence/size/header checks are not full content equivalence. fullhash without a trusted independent source inventory only generates an inventory. No AEE, encoding regeneration or GPU checks.',
            'manifest_sha256': sha256(args.manifest), 'manifest_csv_sha256': sha256(csv_path),
            'validator_sha256': sha256(Path(__file__).resolve()),
            'reference_hashes_sha256': sha256(args.reference_hashes) if args.reference_hashes else None,
            'elapsed_seconds': time.monotonic()-started, 'counts': counts,
            'hash_inventory': inventory, 'files': results}
        write_json(args.report, report)
        print(json.dumps({k: report[k] for k in ['passed_requested_checks', 'complete_source_destination_hash_equivalence_verified', 'elapsed_seconds', 'counts']}))
        return 0 if passed else 1
    except Exception as e:
        if safe_report:
            write_json(args.report, {'passed_requested_checks': False,
                                    'complete_source_destination_hash_equivalence_verified': False,
                                    'fatal_error': type(e).__name__+': '+str(e),
                                    'elapsed_seconds': time.monotonic()-started})
        print(type(e).__name__+': '+str(e), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
