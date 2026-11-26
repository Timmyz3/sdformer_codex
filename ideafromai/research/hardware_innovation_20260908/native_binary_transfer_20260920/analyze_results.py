"""Aggregate measured RTL cycles; keep cold/warm and supply calendars separate."""
from pathlib import Path
from collections import defaultdict
import csv
import json

HERE = Path(__file__).resolve().parent
metadata = {m['label']: m for m in json.loads((HERE/'case_metadata.json').read_text())['cases']}
groups = defaultdict(lambda: defaultdict(int))
all_rows = []
fields = ['cycles', 'config_cycles', 'total_cycles', 'source_reads', 'coef_reads',
          'add_cycles', 'build_cycles', 'out_stalls', 'match_cycles', 'sort_cycles',
          'parent_reads', 'parent_writes', 'y_reads', 'y_writes', 'k_tiles']
for filename in ('fc_results.csv', 'gp_results.csv', 'pr_results.csv', 'gp4_results.csv'):
    path = HERE/filename
    if not path.exists():
        continue
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        m = metadata[row['case']]
        for f in fields:
            row[f] = int(row[f])
        row['calendar'] = int(row['calendar'])
        row['table_writes'] = int(row['table_writes'])
        row['coefficient_transactions'] = row['coef_reads']+row['table_writes']
        assert row['source_reads'] == 80*((int(row['C'])+127)//128), row
        if row['mode'].startswith('AAC'):
            assert row['add_cycles'] == m['nnz'] == row['coef_reads'], row
        if row['mode'] in ('GP', 'GP_reloaded'):
            assert row['add_cycles'] == m['nnz'], row
        all_rows.append(row)
        if not m['real']:
            continue
        for stage in ('all', f'S{m["stage"]}'):
            key = (stage, row['calendar'], row['mode'])
            groups[key]['cases'] += 1
            for field in fields+['table_writes', 'coefficient_transactions']:
                groups[key][field] += row[field]

records = []
for (stage, calendar, mode), values in sorted(groups.items()):
    record = dict(stage=stage, calendar=calendar, mode=mode, **values)
    base_mode = 'AAC_stream' if mode.endswith('stream') else ('AAC_reloaded' if (mode.endswith('warm') or mode.endswith('reloaded')) else 'AAC')
    base = groups[stage, calendar, base_mode]
    record['cycle_reduction_vs_matching_AAC'] = 1-values['cycles']/base['cycles']
    record['total_cycle_reduction_vs_matching_AAC'] = 1-values['total_cycles']/base['total_cycles']
    records.append(record)

out = dict(scope='Selected native ep34 linear slices only, new Q8 weight contract, Verilator component simulation; no BN/PSN/AEE/PPA result.',
           calendars={0:'source/coefficient/output always available',
                      1:'source 3/4, coefficient 2/3, output 4/7 deterministic availability'},
           verified_tasks=len(all_rows), verified_scalar_outputs=len(all_rows)*80*8,
           records=records)
(HERE/'rtl_summary.json').write_text(json.dumps(out, indent=2)+'\n')
lines = ['# RTL原生切片测量表', '',
         out['scope'], '',
         '周期从接受start到最后Y被接受，包含输出排空。total另加真实配置；暖表只换源并循环移动一个完整P8组。', '',
         '冷表写事务已计入执行周期，另在总系数事务中显式列出。GP的build指NRV构包；G4的build指构表，两者不可相减作为同一工作。', '',
         '|域|供数|模式|任务数|执行拍|含配置拍|系数读|表写|相对同冷暖AAC省执行拍|',
         '|---|---|---|---:|---:|---:|---:|---:|---:|']
for r in records:
    lines.append(f'|{r["stage"]}|{r["calendar"]}|{r["mode"]}|{r["cases"]}|{r["cycles"]:,}|{r["total_cycles"]:,}|{r["coef_reads"]:,}|{r["table_writes"]:,}|{r["cycle_reduction_vs_matching_AAC"]:.2%}|')
lines += ['', f'整数核对：{out["verified_tasks"]:,}任务、{out["verified_scalar_outputs"]:,}个输出；包括诊断用例，性能表排除诊断用例。', '']
(HERE/'RTL_MEASUREMENTS.md').write_text('\n'.join(lines))
print('\n'.join(lines))
