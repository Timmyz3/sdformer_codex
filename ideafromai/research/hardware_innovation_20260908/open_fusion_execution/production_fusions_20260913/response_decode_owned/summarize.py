"""Summarize the two fixed C2 conditions; no simulation or new configurations."""
from pathlib import Path
import csv
import json

HERE = Path(__file__).resolve().parent


def main():
    rows = []
    checked = 0
    for condition in ('ready', 'stress'):
        result = json.loads((HERE/(condition+'.json')).read_text())
        assert result['complete']
        assert len(result['rows']) == 3
        for row in result['rows']:
            assert all(c['differences'] == 0 for c in row['checks'].values())
            checked += sum(c['values'] for c in row['checks'].values())
            out = dict(condition=condition, mode=row['mode'],
                       service_slots=row['service_slots'],
                       consumer_service_slots=row['consumer_service_slots'],
                       percent_vs_expanded16=row['percent_vs_expanded16'],
                       SR64_bytes=row['physical_port_bytes']['SR64'],
                       SW64_bytes=row['physical_port_bytes']['SW64'],
                       CR256_bytes=row['physical_port_bytes']['CR256'],
                       CW256_bytes=row['physical_port_bytes']['CW256'],
                       decoder_issue_slots=row['counts'].get('response_W8_decode_issue', 0),
                       decoder_wait_slots=row['counts'].get('response_W8_decode_wait', 0),
                       RF84_unpack_issues=row['counts'].get('IUNPACK_SIGNED_issues', 0),
                       row_scale_product_issues=row['counts'].get('row_scale_existing_16x24_product', 0))
            rows.append(out)
    summary = dict(complete=True, configurations=6,
                   source_condition='ordinary/interior ready and original fixed stress only',
                   checked_updated_gate_PED_values=checked, differences=0,
                   no_new_quantizer_no_AEE=True,
                   scope='Executed real I24 -> preview/sn2 -> complete K864 -> Conv2/merge -> projection gate + U32/V96 PED. No native/globalBN/join.',
                   decoder_latency='2 paid slots per H8: one shared issue + one wait; unvalidated physical timing hypothesis.',
                   decision='Stop this fixed2-slot response-local W8 placement as an acceleration claim; retain numerical and charged dataflow result. No W8/MiLo family verdict.',
                   rows=rows)
    (HERE/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (HERE/'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(configurations=6, checked_values=checked, differences=0)))


if __name__ == '__main__':
    main()
