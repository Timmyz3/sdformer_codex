"""T11a 合并校验：7 个分桶 agent 输出 → 全库处置表 + 桶统计。

校验：(1) 覆盖率——每篇输入 paper 都有恰好一条处置；
     (2) 枚举合法；(3) APPLICABLE_NEW 必须带 how/trial_type。
输出：t11_triage/t11a_merged.json（按桶分组）、t11_triage/t11a_summary.json（统计）。
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INP, OUT = ROOT / 'input', ROOT / 'output'

VALID = {'INCORPORATED', 'FAMILY_COVERED', 'APPLICABLE_NEW',
         'APPLICABLE_BLOCKED', 'NOT_APPLICABLE'}

expected = {}
for f in sorted(INP.glob('batch_*.json')):
    for rec in json.loads(f.read_text()):
        expected[rec['id']] = rec

dispositions = {}
problems = []
for f in sorted(OUT.glob('batch_*.jsonl')):
    for ln in f.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError as e:
            problems.append(f'{f.name}: bad json: {e}: {ln[:80]}')
            continue
        pid = rec.get('id')
        if pid not in expected:
            problems.append(f'{f.name}: unknown id {pid}')
            continue
        if pid in dispositions:
            problems.append(f'{pid}: duplicate disposition')
            continue
        d = rec.get('disposition')
        if d not in VALID:
            problems.append(f'{pid}: invalid disposition {d!r}')
            continue
        if d == 'APPLICABLE_NEW' and (not rec.get('how') or not rec.get('trial_type')):
            problems.append(f'{pid}: APPLICABLE_NEW missing how/trial_type')
            continue
        dispositions[pid] = rec

missing = sorted(set(expected) - set(dispositions))
print('papers: %d, dispositioned: %d, missing: %d, problems: %d' %
      (len(expected), len(dispositions), len(missing), len(problems)))
for p in problems[:20]:
    print('  !', p)

cnt = Counter(r['disposition'] for r in dispositions.values())
fam_cnt = defaultdict(list)
for pid, r in dispositions.items():
    if r['disposition'] == 'APPLICABLE_NEW':
        fam_cnt[r.get('family') or ''].append(pid)

merged = {'dispositions': dispositions}
merged_path = ROOT / 't11a_merged.json'
merged_path.write_text(json.dumps(merged, indent=1, ensure_ascii=False) + '\n')

summary = {
    'total_papers': len(expected),
    'covered': len(dispositions),
    'missing': missing,
    'problems': problems,
    'bucket_counts': dict(cnt),
    'applicable_new_by_family': {k: v for k, v in sorted(fam_cnt.items())},
}
(ROOT / 't11a_summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n')

print('\nbucket counts:')
for k in VALID:
    print('  %-18s %d' % (k, cnt.get(k, 0)))
print('\nAPPLICABLE_NEW families:')
for fam, ids in sorted(fam_cnt.items()):
    print('  %-40s %d  e.g. %s' % (fam or '(none)', len(ids), ids[0]))
