"""Read-only audit of completed result records; does not run RTL or quality jobs."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
rows = json.loads((HERE.parent / 'rtl/results.json').read_text())
index = {(r['fixture'], r['mode'], r['stall'], r['command']): r for r in rows}
assert len(index) == len(rows) == 648
checks = 0
for r in rows:
    assert r['outputs'] == 3840 and r['configuration_cycles'] == 12193
    checks += 2
    peer = index[r['fixture'], 5, r['stall'], r['command']]
    for k in ('source_words', 'weight_words'):
        assert r[k] == peer[k]
        checks += 1
    assert r['psum_reads'] == r['psum_writes'] == r['update_issues'] + 480
    checks += 1
    unstalled = index[r['fixture'], r['mode'], 0, r['command']]
    assert r['cycles'] - unstalled['cycles'] == sum(r[k] for k in (
        'source_stalls', 'weight_stalls', 'output_stalls'))
    checks += 1
    repeated = index[r['fixture'], r['mode'], r['stall'], 1-r['command']]
    for k in ('cycles', 'source_words', 'weight_words', 'psum_reads', 'psum_writes',
              'sum_issues', 'update_issues', 'pattern_copies', 'source_stalls',
              'weight_stalls', 'output_stalls'):
        assert r[k] == repeated[k]
        checks += 1

aggregate = []
for arm in ('dense', 'block_magnitude25', 'cin_magnitude25', 'cin_fullcost25',
            'mixed_retirement25'):
    for stall in (0, 1):
        group = [r for r in rows if r['scope'] == 'formal' and r['arm'] == arm
                 and r['stall'] == stall and r['command'] == 0]
        assert len(group) == 24
        cycles = {m: sum(r['cycles'] for r in group if r['mode'] == m) for m in (3, 4, 5)}
        aggregate.append(dict(arm=arm, stall=stall, cycles=cycles,
            mode4_vs5_extra_percent=100*(cycles[4]/cycles[5]-1)))

out = dict(passed=True, record_rows=len(rows), scalar_invariants=checks,
    output_records=sum(r['outputs'] for r in rows), independent_rtl_execution=False,
    scope='Independent JSON identities and aggregates; functional gold comparison is by the reviewed RTL-agent TB.',
    full_configuration_cycles_once_per_process=12193,
    second_command_reuses_configuration=True, aggregate=aggregate)
(HERE/'result_record_audit.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='aggregate'}, indent=2))
