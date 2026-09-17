"""把 7 个批次压成紧凑 TSV 供主会话人工分类（省上下文）。

列：id | name | venue | cat | ABX(1/0) | A摘要或untried前段
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def clip(s, n):
    s = (s or '').replace('\t', ' ').replace('\n', ' ').strip()
    return s[:n]


for f in sorted((ROOT / 'input').glob('batch_*.json')):
    recs = json.loads(f.read_text())
    out = []
    for r in recs:
        abx = bool(clip(r.get('A'), 5) or clip(r.get('X'), 5))
        sig = clip(r.get('A') or r.get('untried'), 150)
        cat = clip(r.get('cat'), 30)
        out.append('%s\t%s\t%s\t%s\t%d\t%s' %
                   (r['id'], clip(r.get('name'), 45), clip(r.get('venue'), 22),
                    cat, abx, sig))
    dst = ROOT / 'input' / (f.stem + '.tsv')
    dst.write_text('\n'.join(out) + '\n')
    print(dst.name, len(recs))
