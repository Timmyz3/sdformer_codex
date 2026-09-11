"""Rerun the existing byte-payload source program; exclusive issue-slot causes."""
from pathlib import Path
from collections import Counter
import csv
import json
import sys

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
sys.path.insert(0, str(STAGE / 'two_stage_writeback'))
import run_compare as source


def classify(path, scenario, total):
    exclusive = Counter({k: 0 for k in ('progress', 'fifo', 'spike_BP', 'PED_BP', 'BN', 'other')})
    overlapped = Counter()
    conf = source.CONTRACT['backpressure'][scenario]
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        ready = int(r['cycle']) % conf['period'] >= conf['blocked_prefix']
        issue = r['issue']
        if issue == 'fifo_full_wait':
            cause = 'fifo'
            overlapped['fifo_full_with_sink_unready'] += int(not ready)
        elif issue == 'drain' and r['accepted_vector'] == '' and int(r['FIFO_count']) and not ready:
            cause = 'spike_BP'
        elif issue in ('input_wait', 'RAW_wait', 'pipeline_drain') or (issue == 'drain' and r['accepted_vector'] == ''):
            cause = 'other'
        else:
            cause = 'progress'
        exclusive[cause] += 1
        overlapped['sink_unready_exposure'] += int(not ready)
        overlapped['writeback_slots'] += int(r['WB_kind'] != '')
        overlapped['output_accept_slots'] += int(r['accepted_vector'] != '')
        r['exclusive_issue_class'] = cause
    assert sum(exclusive.values()) == total == len(rows)
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return dict(exclusive_issue_slots=dict(exclusive), overlapping_observations_not_additive=dict(overlapped))


def main():
    result = dict(scope='P2/C96/T10 source only, existing two-stage payload program, 8 lanes; no PED or BN',
        classification='One primary class per issue slot; writeback and transfers are overlapping observations, not additive waits.',
        infinite_sink='Always accepts at the SAME one gate-word/slot port. Consumer storage unlimited; the physical source FIFO stays at 16 entries.',
        arms={})
    previous = json.loads((STAGE/'two_stage_writeback/result.json').read_text())
    for axis in ('ordinary', 'lifting_raw'):
        for control in ('unfused', 'fused'):
            key = axis+'_'+control
            program = json.loads((STAGE/'two_stage_writeback'/f'{key}_program.json').read_text())
            vectors, expected = source.base.captured_tile(axis)
            result['arms'][key] = {}
            for scenario in ('ready', 'blocked'):
                path = HERE/f'{key}_{scenario}.csv'
                run = source.simulate(program, scenario, path, vectors, expected)
                assert run['service_slots'] == previous['measurements'][key]['scenarios'][scenario]['service_slots']
                run.update(classify(path, scenario, run['service_slots']))
                result['arms'][key][scenario] = run
    (HERE/'source_result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: {s: dict(slots=v['service_slots'], classes=v['exclusive_issue_slots']) for s,v in arm.items()} for k,arm in result['arms'].items()}, indent=2))


if __name__ == '__main__':
    main()
