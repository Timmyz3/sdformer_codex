"""Summarize the fixed 36 final same-Machine executions, without new runs."""
from pathlib import Path
import csv
import json

HERE = Path(__file__).resolve().parent
AXES = ('dense_twopot', 'lifting_twopot', 'contiguous34')
MODES = ('baseline', 'consumer', 'producer', 'consumer_both', 'producer_both')


def read(axis, label, mode, stress=False):
    suffix = '_stress' if stress else ''
    return json.loads((HERE / f'{axis}_{label}_{mode}{suffix}.json').read_text())


def main():
    rows = []
    comparisons = []
    for axis in AXES:
        for label, stress in (('corner', False), ('interior', False), ('corner', True)):
            modes = ('baseline', 'producer_both') if stress else MODES
            cases = {mode: read(axis, label, mode, stress) for mode in modes}
            baseline = cases['baseline']['service_slots']
            for mode, case in cases.items():
                assert case['common_consumer'].startswith('Immediate ordinary P1 retained Z')
                checks = [v for scope in ('producer', 'consumer')
                          for v in case[scope]['checks'].values()]
                assert all(c['differences'] == 0 for c in checks)
                assert sum(case['stages'].values()) == case['service_slots']
                counts = case['counts']
                rows.append(dict(axis=axis, label=label, stress=stress, mode=mode,
                    service_slots=case['service_slots'],
                    saving_percent_vs_same_function_baseline=100*(baseline-case['service_slots'])/baseline,
                    producer_end=case['producer_end'],
                    SR64_reads=counts['SR64_reads'], SW64_writes=counts['SW64_writes'],
                    CR256_reads=counts['CR256_reads'], CW256_writes=counts['CW256_writes'],
                    SR_bytes=8*counts['SR64_reads'], SW_bytes=8*counts['SW64_writes'],
                    CR_bytes=32*counts['CR256_reads'], CW_bytes=32*counts['CW256_writes'],
                    metadata_bytes=case['metadata_bytes']+case['sn2_metadata_bytes'],
                    differences=sum(c['differences'] for c in checks),
                    compared_output_values=sum(c['values'] for c in checks)))
            if not stress:
                for suffix in ('', '_both'):
                    scan, production = cases['consumer'+suffix], cases['producer'+suffix]
                    comparisons.append(dict(axis=axis, label=label,
                        interface='source_and_sn2' if suffix else 'source_only',
                        strongest_scan_slots=scan['service_slots'],
                        producer_slots=production['service_slots'],
                        saved_slots=scan['service_slots']-production['service_slots'],
                        saving_percent=100*(scan['service_slots']-production['service_slots'])/scan['service_slots']))
    with (HERE/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    result = dict(cases=len(rows), rows=rows, producer_increment=comparisons,
        evidence='CPU actual payload same finite Machine; complete local source/preview/sn2/integer/PED; common P1 retained-Z suffix.',
        precision='Each mode preserves its own source function. No new AEE, no cross-parent quality inheritance.',
        limitations=['Two halos from one frame, three source functions; stress only corner baseline and producer_both.',
                     'Not full layer/network. Native projection and global BN/join are outside.',
                     'Metadata op latency modeled, RTL leaf only functionally checked; no ASIC PPA/same-area claim.',
                     'Source words and continuous I24 remain materialized. Full K control scan remains.'],
        decision='Retain ordinary occupancy format as common implementation; producer-only increment does not justify a title.')
    (HERE/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print('36 final cases checked; summary.csv and summary.json written.')


if __name__ == '__main__':
    main()
