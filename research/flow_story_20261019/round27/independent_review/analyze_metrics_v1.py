#!/usr/bin/env python3
"""R27 scalar-quality CLI; no model, GPU, prediction or GT file access."""
import argparse
import hashlib
import json
from pathlib import Path
import analysis_core as core

HERE = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics', required=True, type=Path)
    p.add_argument('--membership', default=HERE/'membership_v1.json', type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    core.require(not a.output.exists(), 'Fresh output only')
    core.require(sha(a.membership) == '40eba2514b56fd33a910fda4e37d4f86e80431ffe8149efc5f79c18827991e7b', 'Fixed R27 roles')
    m, x = json.loads(a.membership.read_text()), json.loads(a.metrics.read_text())
    rows = x['frames']
    core.require([(r['target_id'], r['sequence']) for r in rows] ==
                 [(r['target_id'], r['sequence']) for r in m['dev_members']], 'Exact new64 order')
    out = core.analyze(rows)
    out['input_metrics_sha256'] = sha(a.metrics)
    out['membership_sha256'] = sha(a.membership)
    out['source_sha256'] = {name:sha(HERE/name) for name in ('analysis_core.py', 'analyze_metrics_v1.py')}
    out['scope_note'] = 'Scalar reaggregation only. Actual prediction/GT provenance needs the separate raw-array audit.'
    a.output.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:out[k] for k in ('resource_decision','quality_prediction_count','frame_region_cells','summary_cells')}))

if __name__ == '__main__':
    main()
