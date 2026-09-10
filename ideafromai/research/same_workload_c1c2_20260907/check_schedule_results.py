"""Compare the independently requested occupancy audit with all40 original points."""
from pathlib import Path
import hashlib
import json

BASE = Path(__file__).resolve().parent


def main():
    old = json.loads((BASE/'c1_cache_attempt_r1/result.json').read_text())['points']
    new = json.loads((BASE/'c1_audit_attempt_r1/result.json').read_text())['points']
    assert len(old) == len(new) == 40
    for a, b in zip(old, new):
        assert a == {k:v for k,v in b.items() if not k.startswith('audit_')}
        assert b['audit_peak_live_temporary_vectors'] <= 4
    record = {
        'status':'PASS_40_IDENTICAL_CYCLE_AND_COUNT_POINTS_WITH_REGISTER_INTERVAL_ASSERTIONS',
        'source_sha256':hashlib.sha256((BASE/'c1_schedule_audit.cpp').read_bytes()).hexdigest(),
        'peak_live_vectors_observed':sorted({p['audit_peak_live_temporary_vectors'] for p in new}),
        'slot_interval_counts_total':[sum(p['audit_vector_intervals'][i] for p in new) for i in range(4)],
        'slot_order':['source_accumulator','coefficient_transfer_and_refill','old_destination_and_final_output','pending_destination_write'],
        'boundary':'CPU scheduled interval assertions plus static review, not RTL execution, all-input proof, or mapped timing.'}
    target = BASE/'c1_schedule_checks.json'
    assert not target.exists()
    target.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
