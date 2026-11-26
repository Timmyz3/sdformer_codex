#!/usr/bin/env python3
"""Small synthetic corruption tests; never touches real DSEC files or CUDA."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    validator = here/'validate_received_data.py'
    tests = []
    with tempfile.TemporaryDirectory(prefix='flow-received-data-test-') as tmp:
        tmp = Path(tmp)
        data = tmp/'dataset'
        data.mkdir()
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        np.save(data/'sample.npy', a)
        np.save(data/'other.npy', a+1)
        (data/'timestamps.txt').write_text('100,200\n')
        original_other = (data/'other.npy').read_bytes()
        rows = []
        for name in ['sample.npy', 'other.npy', 'timestamps.txt']:
            rows.append({'split': 'test', 'category': 'input' if name.endswith('.npy') else 'metadata',
                         'target': name, 'relative_path': name, 'bytes': (data/name).stat().st_size,
                         'known_sha256': '' if name == 'other.npy' else sha(data/name)})
        csvpath = tmp/'manifest.csv'
        with csvpath.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        meta = {'schema_version': 1, 'files_csv': csvpath.name, 'files_csv_sha256': sha(csvpath),
                'expected_files': 3, 'expected_bytes': sum(r['bytes'] for r in rows),
                'expected_known_hashes': 2, 'expected_by_split': {'test': 3},
                'expected_by_category': dict(collections.Counter(r['category'] for r in rows)),
                'npy_contracts': {'input': {'shape': [2, 3], 'dtype': 'float32'}}}
        manifest = tmp/'manifest.json'
        manifest.write_text(json.dumps(meta))
        def run(name, options, expected, report=None):
            report = report or tmp/(name+'.json')
            process = subprocess.run([sys.executable, str(validator), '--root', str(data),
                                      '--manifest', str(manifest), '--report', str(report), *options],
                                     text=True, capture_output=True)
            if process.returncode != expected:
                raise AssertionError(name+' unexpected exit: '+process.stderr+' '+process.stdout)
            tests.append({'name': name, 'exit_code': process.returncode, 'passed': True})
            return json.loads(report.read_text()) if expected != 2 and report.is_file() else None
        quick = run('quick_good', ['--mode', 'quick'], 0)
        assert quick['counts']['hashes_computed'] == 2
        assert not quick['complete_source_destination_hash_equivalence_verified']
        assert quick['counts']['npy_headers_checked'] == 1
        source_hashes = tmp/'source.csv'
        src = run('source_fullhash_inventory_only', ['--mode', 'fullhash', '--finite', 'all',
                  '--hashes-output', str(source_hashes)], 0)
        assert src['all_file_hash_inventory_generated']
        assert not src['complete_source_destination_hash_equivalence_verified']
        dst = run('destination_fullhash_match', ['--mode', 'fullhash', '--threads', '2',
                  '--reference-hashes', str(source_hashes)], 0)
        assert dst['complete_source_destination_hash_equivalence_verified']
        np.save(data/'other.npy', (a+1).reshape(3, 2))
        run('same_size_wrong_shape', ['--mode', 'header'], 1)
        (data/'other.npy').write_bytes(original_other)
        changed = a+1; changed[0, 0] = np.nan
        np.save(data/'other.npy', changed)
        run('finite_all_rejects_nan', ['--mode', 'header', '--finite', 'all'], 1)
        run('source_hash_rejects_same_size_corruption', ['--mode', 'fullhash', '--finite', 'none',
                  '--reference-hashes', str(source_hashes)], 1)
        q = run('quick_does_not_claim_unhashed_content', ['--mode', 'quick'], 0)
        assert not q['complete_source_destination_hash_equivalence_verified']
        (data/'other.npy').write_bytes(original_other)
        (data/'timestamps.txt').write_text('100,201\n')
        run('known_metadata_hash_required_in_quick', ['--mode', 'quick'], 1)
        (data/'timestamps.txt').write_text('100,200\n')
        (data/'other.npy').unlink()
        run('missing_file', ['--mode', 'header'], 1)
        (data/'other.npy').write_bytes(original_other)
        before = manifest.read_bytes()
        run('cannot_overwrite_manifest_via_report', ['--mode', 'quick'], 2, manifest)
        assert manifest.read_bytes() == before
        before = (data/'sample.npy').read_bytes()
        run('cannot_overwrite_data_via_report', ['--mode', 'quick'], 2, data/'sample.npy')
        assert (data/'sample.npy').read_bytes() == before
    result = {'passed': True, 'CPU_only': True, 'real_DSEC_accessed': False,
              'validator_sha256': sha(validator), 'tests': tests}
    output = here/'validator_selftest_results.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': True, 'tests': len(tests), 'output': str(output)}))


if __name__ == '__main__':
    main()
