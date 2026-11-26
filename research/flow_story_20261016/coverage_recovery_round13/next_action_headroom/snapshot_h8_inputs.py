#!/usr/bin/env python3
"""Archive fixed H8 input bytes for local independent reduction; no inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'preparation'))
from prepare_common_state import prepare, require, sha, write


def main():
    parser = argparse.ArgumentParser()
    for key in ('runtime', 'membership'):
        parser.add_argument('--' + key, type=Path, required=True)
        parser.add_argument('--' + key + '-sha256', required=True)
    parser.add_argument('--h8', type=Path, required=True)
    parser.add_argument('--h8-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    a = parser.parse_args()
    require(sha(a.h8) == a.h8_sha256, 'H8 SHA differs')
    h8 = json.loads(a.h8.read_text())
    runtime, members, base, b, context, roles = prepare(a)
    require(h8['membership_sha256'] == a.membership_sha256, 'H8 member binding differs')
    chosen = []
    for sequence in members['split_sequences']['train']:
        rows = [r for r in members['selector_members'] if r['sequence'] == sequence]
        chosen.append(min(rows, key=lambda r: (hashlib.sha256(('flow-headroom-v1:' + r['target_id']).encode()).hexdigest(), r['target_id'])))
    require(chosen == h8['members'] and len(chosen) == 8, 'H8 fixed identity rule differs')
    native, _, _, _, hashes, *_ = context
    a.output_dir.mkdir(parents=True, exist_ok=False)
    entries = []
    archive = a.output_dir / 'inputs.tar'
    with tarfile.open(str(archive), 'w') as tar:
        for row in chosen:
            for key, relative in native.target_files(row).items():
                rel = Path(relative)
                require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe path')
                path = native.checked_path(b.data_root, relative)
                actual, expected = sha(path), hashes[relative]['sha256']
                require(actual == expected, 'Input SHA differs')
                entries.append({'target_id': row['target_id'], 'kind': 'events' if key == 'input' else key,
                                'relative_path': relative, 'expected_sha256': expected,
                                'sha256': actual, 'bytes': path.stat().st_size})
                tar.add(str(path), arcname=relative, recursive=False)
    require(len(entries) == 24, 'Expected24 input arrays')
    result = {'schema': 'flow_h8_input_snapshot_v1', 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'source_root': str(b.data_root), 'all_files': entries,
              'archive_bytes': archive.stat().st_size, 'archive_sha256': sha(archive),
              'membership_sha256': a.membership_sha256, 'h8_sha256': a.h8_sha256,
              'script_sha256': sha(__file__),
              'scope': 'Opaque bytes from metadata-selected H8, no numerical GT/events inspection or model inference.'}
    write(a.output_dir / 'receipt.json', result)
    print(json.dumps({'status': 'SNAPSHOT_COMPLETE', 'files': 24, 'receipt_sha256': sha(a.output_dir / 'receipt.json')}))


if __name__ == '__main__':
    main()
