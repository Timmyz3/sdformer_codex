#!/usr/bin/env python3
"""Copy only the four audited samples using an existing authenticated SSH socket.

No credentials, training, torch import, remote writes, or whole-dataset transfer.
Every incoming file is checked against the SHA recorded by the read-only audit.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--socket', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    destination = Path(args.destination).resolve()
    audit_path = here / 'archive_data_audit.json'
    audit = json.loads(audit_path.read_text())
    source_root = Path(audit['remote_root'])
    selected = {}
    sequences = set()
    for sample in audit['sample_content_checks']:
        sequences.add(sample['sequence'])
        for item in list(sample['arrays'].values()) + [sample['raw_png']]:
            selected[item['path']] = item
    for row in audit['raw_sequences']:
        if row['sequence'] in sequences:
            item = row['timestamps']
            selected[item['path']] = item
    result = {'scope': 'Four audited samples and their timestamps only; no full staging',
              'audit_sha256': sha(audit_path), 'source_root': str(source_root),
              'destination': str(destination), 'files': [], 'gpu_used': False}
    for source, item in sorted(selected.items()):
        relative = Path(source).relative_to(source_root)
        if not item.get('sha256'):
            raise ValueError('Missing audited hash: ' + source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        already_valid = target.is_file() and sha(target) == item['sha256']
        if not already_valid:
            if target.exists():
                raise ValueError('Refusing to replace existing different file: ' + str(target))
            partial = target.with_name(target.name + '.partial')
            command = ['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ControlPath=' + args.socket,
                       '-P', '10037', 'root@ssh.sd5ai.scnet.cn:' + source, str(partial)]
            subprocess.run(command, check=True, timeout=180)
            if partial.stat().st_size != item['bytes'] or sha(partial) != item['sha256']:
                raise ValueError('Incoming size/hash mismatch: ' + str(partial))
            os.replace(str(partial), str(target))
        result['files'].append({'relative_path': str(relative), 'bytes': item['bytes'],
                                'sha256': item['sha256'], 'matched': True,
                                'already_present': already_valid})
        print('Verified', str(relative), flush=True)
    result.update({'status': 'PASS', 'file_count': len(result['files']),
                   'total_bytes': sum(x['bytes'] for x in result['files']),
                   'completed_utc': datetime.datetime.utcnow().isoformat() + 'Z'})
    (here / 'archived_sample_staging.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS:', result['file_count'], 'files,', result['total_bytes'], 'bytes', flush=True)


if __name__ == '__main__':
    main()
