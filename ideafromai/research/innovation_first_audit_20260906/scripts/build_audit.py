#!/usr/bin/env python3.12
"""Assemble reviewed source records; fail on missing original files or hash drift."""
from pathlib import Path
import json
import hashlib
import html
import re
import markdown

BASE = Path(__file__).resolve().parents[1]
IDEAS = BASE.parents[1]


def h(x):
    if isinstance(x, (dict, list)):
        x = json.dumps(x, ensure_ascii=False)
    return html.escape(str(x))


def main():
    inventory = json.loads((BASE / 'inventory.json').read_text())['files']
    original = {f['path']: f for f in inventory}
    catalog_file = BASE / 'catalog_updates.json'
    catalog = {e['path']: e for e in json.loads(catalog_file.read_text())['updates']} if catalog_file.exists() else {}
    assert set(catalog) <= {'README.md', 'INDEX.json', 'MANIFEST.txt'}, 'unexpected source mutation'
    all_files = []
    for name in ['root_audit.json', 'c1_audit.json', 'c2_audit.json', 'new_audit.json']:
        obj = json.loads((BASE / 'records' / name).read_text())
        for f in obj['files']:
            f['reviewer_record'] = name
            all_files.append(f)
    by_path = {f['path']: f for f in all_files}
    assert len(by_path) == len(all_files), 'duplicate file review'
    missing = sorted(set(original) - set(by_path))
    assert not missing, ('missing', missing)
    ids = set()
    for f in all_files:
        p = IDEAS / f['path']
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        assert f['read_complete'] is True, ('incomplete', f['path'])
        reviewed_digest = digest
        if f['path'] in catalog:
            c = catalog[f['path']]
            assert digest == c['after_sha256'], ('catalog drift', f['path'])
            snapshot = BASE / c['snapshot']
            reviewed_digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            assert reviewed_digest == c['before_sha256'], ('snapshot drift', f['path'])
            f['current_catalog_sha256'] = digest
            f['reviewed_snapshot'] = str(snapshot)
            f['coverage_note'] = str(f.get('coverage_note', '')) + ' 审阅后仅修订研究入口；本条 SHA 对应已保留的审阅前快照，当前入口 SHA 另记。'
        assert f['sha256'] == reviewed_digest, ('read hash differs', f['path'])
        if f['path'] in original:
            assert original[f['path']]['sha256'] == reviewed_digest, ('inventory drift', f['path'])
        for i in f['ideas']:
            assert i['id'] not in ids, ('duplicate id', i['id'])
            ids.add(i['id'])
            for key in ['original_name', 'section_or_lines', 'mechanism', 'nearest_prior',
                        'frozen_fit', 'actual_increment', 'cost_or_fatal', 'migration', 'scores', 'verdict']:
                assert key in i, (f['path'], i['id'], key)
            for s in ['N', 'F', 'H', 'T']:
                value = i['scores'][s]
                assert value is None or 0 <= value <= 10, (i['id'], s, value)
    all_files.sort(key=lambda f: f['path'])
    merged = {'date': '2026-09-06', 'original_files': len(original),
              'reviewed_files_including_prior_round': len(all_files),
              'entry_count_including_repeat_occurrences_and_evidence_rows': len(ids),
              'score_boundary': 'N/F/H/T are research judgments, not acceptance probabilities. Null means non-mechanism.',
              'files': all_files}
    (BASE / 'audit-ledger.json').write_text(json.dumps(merged, ensure_ascii=False, indent=2) + '\n')
    intro = (BASE / 'synthesis.md').read_text()
    md = [intro, '\n## 逐文件逐条审阅\n',
          '重复机制保留各文件出现位置；记录数不代表独立创新数量。N=新意，F=冻结适配，H=实现可行性，T=TCAS-II主机制潜力；均为0–10。空分表示入口或工程史料。\n']
    chunks = []
    sources = []
    for f in all_files:
        file_link = f.get('reviewed_snapshot', str(IDEAS / f['path']))
        md += [f"\n### {f['path']}\n", f"精读完成；SHA256 `{f['sha256']}`。\n"]
        if f.get('coverage_note'):
            md += [str(f['coverage_note']) + '\n']
        entries = []
        for i in f['ideas']:
            scores = ' / '.join('—' if i['scores'][s] is None else str(i['scores'][s]) for s in ['N', 'F', 'H', 'T'])
            md += [f"\n#### {i['id']} · {i['original_name']}\n",
                   f"定位：{i['section_or_lines']}。N/F/H/T：**{scores}**。\n"]
            cells = []
            labels = [('mechanism','原机制'), ('nearest_prior','最近先验与访问状态'),
                      ('frozen_fit','冻结适配'), ('actual_increment','真正增量'),
                      ('cost_or_fatal','代价与反证'), ('migration','迁移方式'), ('verdict','结论')]
            for key, label in labels:
                value = i[key]
                if key == 'nearest_prior':
                    ps = value if isinstance(value, list) else [value]
                    textparts = []
                    htmlparts = []
                    for prior in ps:
                        if isinstance(prior, dict):
                            url = prior.get('url')
                            title = prior.get('title', '来源')
                            access = prior.get('access', '未声明访问状态')
                            textparts.append((f'[{title}]({url})' if url else str(title)) + '；' + str(access))
                            htmlparts.append((f'<a href="{h(url)}" target="_blank" rel="noopener">{h(title)}</a>' if url else h(title)) + '；' + h(access))
                            sources.append({'idea_id':i['id'],'file':f['path'], **prior})
                        else:
                            textparts.append(str(prior)); htmlparts.append(h(prior))
                    plain = '；'.join(textparts); rendered = '<br>'.join(htmlparts)
                else:
                    plain = str(value); rendered = h(value).replace('\n','<br>')
                md += [f'- **{label}：** {plain}\n']
                cells.append(f'<dt>{label}</dt><dd>{rendered}</dd>')
            t = i['scores']['T']
            badge = 'na' if t is None else ('high' if t >= 6 else 'low')
            searchable = h(f['path'] + ' ' + json.dumps(i, ensure_ascii=False)).lower()
            entries.append(f'<details class="idea" data-text="{searchable}" data-score="{t if t is not None else -1}"><summary><span class="identity">{h(i["id"])} · {h(i["original_name"])}</span><span class="score {badge}">{h(scores)}</span></summary><p class="loc">{h(i["section_or_lines"])}</p><dl>{"".join(cells)}</dl></details>')
        chunks.append(f'<section class="file"><h3><a href="{h(file_link)}">{h(f["path"])}</a></h3><p class="loc">已精读 · {len(f["ideas"])} 条记录 · SHA256 {f["sha256"][:12]}</p><p>{h(f.get("coverage_note",""))}</p>{"".join(entries) if entries else "<p class=loc>入口或身份合同；没有独立新增机制。</p>"}</section>')
    (BASE / 'report-source.md').write_text('\n'.join(md) + '\n')
    review_sources = []
    review_hashes = []
    def collect_sources(value, record_file, trail=''):
        if isinstance(value, dict):
            if value.get('url') and ('title' in value or 'name' in value):
                review_sources.append({'review_file': record_file, 'field': trail, **value})
            for k, v in value.items():
                collect_sources(v, record_file, trail + '/' + str(k))
        elif isinstance(value, list):
            for j, v in enumerate(value):
                collect_sources(v, record_file, trail + '/' + str(j))
    for review_file in sorted((BASE / 'records').glob('*.json')):
        if review_file.name in {'root_audit.json', 'c1_audit.json', 'c2_audit.json', 'new_audit.json'}:
            continue
        review_hashes.append({'path': str(review_file), 'sha256': hashlib.sha256(review_file.read_bytes()).hexdigest()})
        collect_sources(json.loads(review_file.read_text()), review_file.name)
    (BASE / 'claim-source-ledger.json').write_text(json.dumps({'boundary':'每条记录的来源和访问状态；未访问全文不升级为已核全文。','records':sources, 'candidate_review_sources':review_sources, 'candidate_review_hashes':review_hashes}, ensure_ascii=False, indent=2) + '\n')
    introduction = markdown.markdown(intro, extensions=['tables', 'fenced_code'])
    doc = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C1/C2 创新逐条复审</title><style>
    :root{color-scheme:light;--ink:#162c3f;--muted:#536879;--line:#d9e2e8;--blue:#135e81}*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:var(--ink);font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1160px;margin:0 auto;padding:32px 22px 80px}h1{font-size:32px;line-height:1.35}h2{font-size:24px;margin-top:36px}h3{font-size:18px}a{color:var(--blue);text-underline-offset:3px;overflow-wrap:anywhere}p,li,dd{overflow-wrap:anywhere}.lead,.file{background:white;border:1px solid var(--line);border-radius:12px;padding:24px;margin:20px 0}.loc{font-size:13px;color:var(--muted)}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border:1px solid var(--line);text-align:left;vertical-align:top}th{background:#edf4f8}code{background:#f0f4f6;padding:2px 4px;font-size:.9em;overflow-wrap:anywhere}pre{white-space:pre-wrap}details.idea{border-top:1px solid var(--line);padding:14px 0}summary{cursor:pointer;display:flex;align-items:start;gap:14px;justify-content:space-between;font-weight:600}.identity{max-width:80%}.score{font-size:13px;white-space:nowrap;padding:2px 8px;border-radius:5px;background:#edf2f4}.high{background:#e4f3ef;color:#235749}.low{color:#66524b}.na{color:var(--muted)}dl{display:grid;grid-template-columns:155px 1fr;gap:12px;margin-top:16px;font-size:14px}dt{font-weight:600;color:var(--muted)}dd{margin:0}.filters{position:sticky;top:0;background:#e7eff4f5;padding:16px;border:1px solid var(--line);border-radius:8px;z-index:2;display:flex;gap:12px;align-items:center;flex-wrap:wrap}input,select,button{font:inherit;padding:7px 10px;border:1px solid #93a7b6;border-radius:5px;background:white}input{flex:1;min-width:230px}.filters label{font-size:13px}button{cursor:pointer;font-size:13px}[hidden]{display:none!important}.totals{font-size:14px;color:var(--muted)}@media(max-width:700px){main{padding:20px 12px}.lead,.file{padding:16px}dl{grid-template-columns:1fr;gap:3px}dd{margin-bottom:12px}summary{display:block}.score{display:inline-block;margin-top:8px}.identity{max-width:none}table{display:block;overflow:auto}}@media print{.filters{display:none}body{background:white}.file{break-inside:auto}summary{display:block}details>dl{display:grid!important}.lead,.file{border:0;padding:0}a{color:inherit}}
    </style></head><body><main>'''
    doc += '<article class="lead">' + introduction + '</article>'
    doc += f'<h2>逐文件审阅台账</h2><p class="totals">原始文件 {len(original)}/{len(original)}；另审上一轮研究入口。共 {len(ids)} 条机制出现位置与证据记录，含重复与工程史料，不代表独立创新数量。N / F / H / T 均为研究判断。</p>'
    doc += '<p class="totals">以下检索与分数筛选作用于原文件台账；V1–V5 的新候选及独立评分见上方结论表。原包条目最高 T=5。</p>'
    doc += '<div class="filters"><label for="q">检索机制、先验或文件</label><input id="q" type="search" placeholder="例如：Prosperity、动态 BN、OP-STW"><label for="min">最低 T 分</label><select id="min"><option value="-1">全部（含无评分）</option><option value="4">4</option><option value="6">6</option><option value="7">7</option></select><button id="expand" type="button">展开可见条目</button><button id="collapse" type="button">收起条目</button><span id="count" aria-live="polite"></span></div>'
    doc += ''.join(chunks)
    doc += '''<script>
    const q=document.getElementById('q'),min=document.getElementById('min');
    const entries=[...document.querySelectorAll('.idea')];
    function filter(){const s=q.value.toLowerCase().trim(),v=Number(min.value);let n=0;for(const e of entries){e.hidden=!(e.dataset.text.includes(s)&&Number(e.dataset.score)>=v);if(!e.hidden)n++;}for(const f of document.querySelectorAll('.file')){const es=[...f.querySelectorAll('.idea')];f.hidden=es.length?!es.some(e=>!e.hidden):(Boolean(s)||v>=0);}document.getElementById('count').textContent=n+' 条可见';}
    q.addEventListener('input',filter);min.addEventListener('change',filter);
    document.getElementById('expand').onclick=()=>entries.forEach(e=>{if(!e.hidden)e.open=true;});
    document.getElementById('collapse').onclick=()=>entries.forEach(e=>e.open=false);filter();
    </script></main></body></html>'''
    (BASE / 'C1C2创新逐条复审.html').write_text(doc)
    qa={'original_file_coverage':f'{len(original)}/{len(original)}','unique_record_ids':len(ids),
        'read_hash_checks':'PASS','required_fields_and_score_ranges':'PASS',
        'catalog_updates_with_verified_original_snapshots':sorted(catalog),
        'visual_check':'浏览器渲染工具不可用；本构建仅完成结构检查，未声称视觉全检。',
        'outputs':['audit-ledger.json','report-source.md','claim-source-ledger.json','C1C2创新逐条复审.html']}
    (BASE / 'QA.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(qa,ensure_ascii=False))


if __name__ == '__main__':
    main()
