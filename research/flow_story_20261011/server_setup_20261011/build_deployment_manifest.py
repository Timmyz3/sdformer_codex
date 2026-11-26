#!/usr/bin/env python3
"""Freeze the already agreed phase-1 file set; no SSH or data reads/writes."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    protocol = here.parent / 'preflight/data_protocol'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v2-csv', type=Path, default=protocol/'archive_v2_transfer_manifest.csv')
    p.add_argument('--audit', type=Path, default=protocol/'archive_data_audit.json')
    p.add_argument('--output-dir', type=Path, default=here)
    args = p.parse_args()
    audit = json.loads(args.audit.read_text())
    assert digest(args.v2_csv) == audit['transfer_manifest']['sha256'], 'v2 CSV changed'
    rows = list(csv.DictReader(args.v2_csv.open(newline='')))
    assert len(rows) == 24456
    assert sum(int(r['bytes']) for r in rows) == 122713555968
    assert collections.Counter(r['split'] for r in rows) == {'train': 18000, 'valid': 6456}
    hashes = {}
    source_root = PurePosixPath(audit['remote_root'])
    for sample in audit['sample_content_checks']:
        for item in sample['arrays'].values():
            hashes[str(PurePosixPath(item['path']).relative_to(source_root))] = item['sha256']
    for item in audit['phase1_metadata_transfer']['files']:
        rows.append({'split': 'metadata', 'category': item['category'],
                     'target': item.get('target', item.get('sequence', '')),
                     'relative_path': item['relative_path'], 'bytes': str(item['bytes'])})
        if item.get('sha256'):
            hashes[item['relative_path']] = item['sha256']
    assert len(rows) == len({r['relative_path'] for r in rows}) == 24496
    assert len(hashes) == 34
    assert sum(int(r['bytes']) for r in rows) == 122759202642
    for row in rows:
        rel = PurePosixPath(row['relative_path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        row['known_sha256'] = hashes.get(row['relative_path'], '')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir/'phase1_deployment_manifest.csv'
    columns = ['split', 'category', 'target', 'relative_path', 'bytes', 'known_sha256']
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r['relative_path']))
    meta = {
        'schema_version': 1,
        'membership': 'frozen canonical train6000/valid2152 v2 triplets plus18 rectify+18 timestamps+4 sample PNG; no auxiliary membership changes',
        'files_csv': csv_path.name, 'files_csv_sha256': digest(csv_path),
        'expected_files': len(rows), 'expected_bytes': sum(int(r['bytes']) for r in rows),
        'expected_known_hashes': len(hashes),
        'expected_by_split': dict(collections.Counter(r['split'] for r in rows)),
        'expected_by_category': dict(collections.Counter(r['category'] for r in rows)),
        'npy_contracts': {
            'input': {'shape': [10, 480, 640], 'dtype': 'float32'},
            'gt': {'shape': [2, 480, 640], 'dtype': 'float32'},
            'mask': {'shape': [480, 640], 'dtype': 'bool'}},
        'source_root_hint': audit['remote_root'],
        'source_v2_csv_sha256': digest(args.v2_csv),
        'source_archive_audit_sha256': digest(args.audit),
        'canonical_manifest_sha256': audit['canonical_manifest_sha256'],
        'builder_sha256': digest(Path(__file__).resolve()),
        'hash_scope': '34 previously audited hashes; other files require source fullhash inventory before destination content-equivalence verification'}
    meta_path = args.output_dir/'phase1_deployment_manifest.json'
    meta_path.write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps({'metadata': str(meta_path), 'csv': str(csv_path),
                      'files': len(rows), 'bytes': meta['expected_bytes'], 'known_hashes': len(hashes)}))


if __name__ == '__main__':
    main()
