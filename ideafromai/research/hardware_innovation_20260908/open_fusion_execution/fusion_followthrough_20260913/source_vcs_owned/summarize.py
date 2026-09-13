"""Summarize only the completed fixed VCS source batch, without rerunning tools."""
from pathlib import Path
import csv
import json

HERE = Path(__file__).resolve().parent


def main():
    result = json.loads((HERE/'results.json').read_text())
    assert result['complete'] and result['VCS_completed'] and len(result['rows']) == 20
    manifest = json.loads((HERE/'manifest.json').read_text())
    length = {r['name']: r['program_length'] for r in manifest['families']}
    rows = []
    for row in result['rows']:
        assert row['gate_and_RF_exact'] and row['all_cycle_port_stall_counts_equal']
        assert not row['field_mismatches'] and row['measured'] == row['archived_Verilator']
        rows.append(dict(family=row['family'], program_length=length[row['family']],
                         window=row['window'], **row['measured'], archived_Verilator_equal=True))
    with (HERE/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = dict(scope='Five new source programs on unchanged temporal_source RTL, VCS full64, two existing halos and two fixed handshake modes.',
                   tool=result['tool_probe']['vcs_version'], totals=result['totals'], rows=rows,
                   all_source_gate_and_RF_values_exact=True, all_archived_cycle_port_stall_counts_equal=True,
                   no_RTL_functional_changes=True, no_PPA_no_full_chain=True,
                   compile_attempts=1, simulation_cases=20, failed_compile_or_simulation_attempts=0,
                   initial_tool_detection_note='Default-width vcs -ID selected absent linux32 compiler; explicit -full64 found V-2023.12-SP1. This preceded the sole compile campaign.',
                   mutex_released=True)
    (HERE/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary['totals']))


if __name__ == '__main__':
    main()
