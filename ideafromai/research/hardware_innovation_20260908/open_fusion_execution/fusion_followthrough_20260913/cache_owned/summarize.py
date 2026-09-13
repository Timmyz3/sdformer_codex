"""Aggregate the six fixed runs without mixing directed or offline data."""
from pathlib import Path
import csv
import json

HERE = Path(__file__).resolve().parent


def main():
    rows, totals = [], []
    for pressure in ('ready', 'stress'):
        loaded = {mode:json.loads((HERE/f'{mode}_{pressure}.json').read_text())
                  for mode in ('raw', 'exact', 'certified')}
        base = loaded['raw']['total_service_slots']
        for mode, data in loaded.items():
            assert len(data['frames']) == 4
            assert data['total_service_slots'] == sum(f['service_slots'] for f in data['frames'])
            total = dict(mode=mode, pressure=pressure, total_service_slots=data['total_service_slots'],
                         delta_vs_raw=data['total_service_slots']-base,
                         percent_more_service_vs_raw=100*(data['total_service_slots']/base-1),
                         source_service_slots=0, selected_H8_service_slots=0,
                         natural_warm_queries=72 if mode != 'raw' else 0, natural_hits=0,
                         nonzero_change_hits=0, SR_bytes=0, SW_bytes=0, CR_bytes=0, CW_bytes=0,
                         port_or_writeback_wait=0, actual_source_dot_values_checked=0)
            for f in data['frames']:
                for group in f['checks'].values():
                    assert all(v['differences'] == 0 for v in group.values())
                assert all(v['differences'] == 0 for v in f['captured_endpoint_comparison'].values())
                s = f['source']
                assert s['checks']['differences'] == 0 and s['selected_H8'] == 24 and s['total_H8'] == 1452
                row = dict(mode=mode, pressure=pressure, frame=f['frame'], service_slots=f['service_slots'],
                           source_service_slots=s['service_slots'],
                           selected_H8_service_slots=sum(r['service_slots'] for r in s['rows']),
                           natural_hits=s['hits'], nonzero_change_hits=s['nonzero_change_hits'],
                           actual_source_dot_values_checked=s['actual_dot_values_checked'])
                for label, count, width in [('SR_bytes','SR64_reads',8),('SW_bytes','SW64_writes',8),
                                             ('CR_bytes','CR256_reads',32),('CW_bytes','CW256_writes',32)]:
                    row[label] = f['counts'].get(count, 0)*width
                row['port_or_writeback_wait'] = f['counts'].get('port_or_writeback_wait', 0)
                rows.append(row)
                for key in total:
                    if key in row and key not in ('mode', 'pressure'):
                        total[key] += row[key]
            totals.append(total)
    for name, values in [('frames.csv', rows), ('summary.csv', totals)]:
        with (HERE/name).open('w') as out:
            writer = csv.DictWriter(out, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    result = dict(complete=True, CPU_service_model=True, natural_cases=24,
                  directed_source_cases=12, natural_hit_count=0,
                  gate_preview_updated_U_PED_and_actual_egress_zero_difference=True,
                  decision='Stop this 24-entry/three-level whole-H8 layout; retain other temporal representations/interfaces.',
                  totals=totals)
    (HERE/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    main()
