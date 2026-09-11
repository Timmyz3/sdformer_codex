#!/usr/bin/env python3.12
"""Aggregate existing public-work inventories and idea records; no network required."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

OUT = Path(__file__).resolve().parent
IDEAS = next(p for p in OUT.parents if p.name == 'ideafromai')
SURVEY = 'research/hardware_innovation_20260908/survey_ab_fusion_20260910'
MAIN = 'research/literature_audit_20260909/literature_inventory.json'
MUSHA = 'research/literature_audit_mushaolong_20260909/literature_inventory.json'
AUDIT = 'research/audit_comparison_20260909'
URL_RE = re.compile(r'https?://[^\s<>\]\[\)\(\"\'`；，。]+')
SKIP = {'.git', '.venv_train312', '__pycache__', 'node_modules', 'original_snapshots'}


def text(value):
    if value is None:
        return ''
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalized(value):
    return re.sub(r'[^\w]+', '', text(value).casefold())


def uniq(items):
    return list(dict.fromkeys(x for x in items if x))


def join(items):
    return ' || '.join(uniq(map(text, items)))


def urls(value):
    return uniq(x.rstrip('.,;:') for x in URL_RE.findall(text(value)))


def repo_url(url):
    p = urlsplit(url)
    if p.hostname not in {'github.com', 'gitlab.com', 'codeberg.org'}:
        return ''
    parts = p.path.strip('/').split('/')
    if len(parts) < 2 or parts[0] in {'topics', 'search', 'orgs', 'users', 'features', 'settings'}:
        return ''
    return f'https://{p.hostname}/{parts[0]}/{parts[1].removesuffix(".git")}'.casefold()


def paper_id(row):
    if row['entity_type'] != 'paper':
        return ''
    found = set(re.findall(r'arxiv\.org/(?:abs|html|pdf)/(\d{4}\.\d{4,5})', ' '.join(row['links'])))
    return 'arxiv:' + next(iter(found)) if len(found) == 1 else ''


def csv_write(name, rows, fields):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def load(rel):
    return json.loads((IDEAS / rel).read_text())


def canonical_path(value):
    return text(value).replace('/home/zhumd/work/ideafromai/', str(IDEAS) + '/')


records = []
source_counts = {}


def add_record(raw, source, ref, kind=None):
    title = raw.get('full_title') or raw.get('title') or raw.get('name', '')
    venue = raw.get('venue') or ' '.join(map(str, filter(None, [raw.get('venue_name'), raw.get('venue_year')])))
    if isinstance(venue, dict):
        venue = ' '.join(map(str, filter(None, [venue.get('name'), venue.get('year')])))
    name = raw.get('name') or title
    row = {
        'record_ref': ref, 'source_file': source, 'name': name, 'full_title': title,
        'venue': text(venue), 'entity_type': kind or raw.get('entity_type', 'paper'),
        'category': raw.get('category', raw.get('domain', '')),
        'links': uniq(urls(raw.get('source_url', [])) +
                      ([f"https://doi.org/{raw['doi']}"] if raw.get('doi') else [])),
        'reading': join([raw.get(k) for k in ('depth', 'depth_raw', 'reading_depth', 'reading_status', 'depth_bucket', 'read_flag')]),
        'reading_level': raw.get('reading_level', ''),
        'attempt': join([raw.get(k) for k in ('status', 'migration_status', 'disposition', 'artifact_status')]),
        'reason': join([raw.get(k) for k in ('reason', 'stop_or_include_reason')]),
        'untried': join([raw.get(k) for k in ('what_untried', 'untried_parts', 'capabilities_unknown', 'missing_recipe')]),
        'evidence': canonical_path(raw.get('evidence_file', raw.get('evidence_paths', raw.get('local_pdf', '')))),
        'priority': text(raw.get('precision_tier', '')),
        'A': raw.get('core_mechanism', raw.get('A', '')),
        'B': raw.get('gap', raw.get('B', '')), 'X': raw.get('X_hint', ''), 'cards': [],
    }
    records.append(row)
    source_counts[source] = source_counts.get(source, 0) + 1
    return row


for d in load(SURVEY + '/literature_merged_300plus.json'):
    add_record(d, SURVEY + '/literature_merged_300plus.json', d['uid'])
for d in load(MAIN):
    add_record(d, MAIN, 'MAIN-' + d['id'])
for d in load(MUSHA)['entries']:
    add_record(d, MUSHA, 'MUSHA-' + d['id'])
direct = 'research/hardware_innovation_20260908/literature/direct_priors_20260909/method_inventory.json'
for key in ('cicc', 'cfmp'):
    add_record(load(direct)[key], direct, 'DIRECT-' + key)

# The comparison audit is a separate source, not a replacement for either
# original audit. Read its authored supplement, including the eight table rows
# and the two explicitly linked companion priors. Levels below describe that
# September 9 source's stated reading, not new PDF reading by this builder.
cicc_source = AUDIT + '/CICC_SUPPLEMENT.md'
cicc_body = (IDEAS / cicc_source).read_text()
cicc_programs = {year: next((u for u in urls(cicc_body)
                            if f'CICC-{year}-Program' in u), '')
                 for year in ('2022', '2023', '2024', '2025')}
cicc_programs['2026'] = 'https://www.ieee-cicc.org/technicalprogram/'


def supplement_record(title, name, venue, body, reading, level, ref, line_number):
    source_urls = urls(body)
    year = next(iter(re.findall(r'20\d{2}', venue)), '')
    if 'CICC' in venue and cicc_programs.get(year):
        source_urls.append(cicc_programs[year])
    doi = re.search(r'10\.\d{4,9}/[A-Za-z0-9._/-]+', body)
    bullets = [line.strip('- ').replace('**', '') for line in body.splitlines()
               if line.startswith('- ')]
    statuses = [line for line in bullets if line.startswith(('纳入', '处置'))]
    untried = [line for line in bullets if line.startswith(('尚未尝试', '未证实部分'))]
    mechanism = next((line.removeprefix('A：').strip() for line in bullets if line.startswith('A：')), '')
    row = add_record(dict(name=name, full_title=title, venue=venue,
        source_url=source_urls, doi=doi.group(0) if doi else None,
        reading_depth=reading, reading_level=level,
        status=join(statuses) or '补充原文保留为对照或待验证线索；尚未迁入',
        reason=join(bullets) or body,
        what_untried=join(untried) or body,
        core_mechanism=mechanism,
        evidence_file=f'{IDEAS / cicc_source}:{line_number}'), cicc_source, ref)
    return row


cicc_levels = ['full_text_reported', 'primary_abstract_reported', 'primary_abstract_reported',
               'primary_demo_abstract_reported', 'primary_identity_only', 'primary_identity_only',
               'primary_identity_only', 'primary_abstract_reported']
for match in re.finditer(r'^### (\d+)\. ([^\n]+)\n(.*?)(?=^### |^## |\Z)', cicc_body, re.M | re.S):
    number, heading, body = int(match[1]), match[2], match[3]
    title = re.search(r'^\*\*(.+?)\*\*', body, re.M)[1]
    reading = re.search(r'\*\*阅读：(.*?)(?:\n\n|\Z)', body, re.S)[1].replace('**', '')
    year = re.search(r'CICC (20\d{2})', heading)[1]
    supplement_record(title, heading.split(' — ')[0], 'CICC '+year, body, reading,
        cicc_levels[number-1], f'CICC-SUPP-{number:02d}', cicc_body[:match.start()].count('\n')+1)

table_number = 8
for line_number, line in enumerate(cicc_body.splitlines(), 1):
    if not line.startswith('| **'):
        continue
    table_number += 1
    cells = [x.strip() for x in line.strip('|').split('|')]
    title = re.search(r'\*\*(.+?)\*\*', cells[0])[1]
    year = re.search(r'20\d{2}', cells[0])[0]
    level = ('partial_text_reported' if '搜索正文片段' in cells[1] else
             'author_description_reported' if '介绍' in cells[1] else 'primary_identity_only')
    supplement_record(title, title.split(':')[0], 'CICC '+year, line,
        cells[1], level, f'CICC-SUPP-{table_number:02d}', line_number)

for index, line in enumerate(cicc_body.splitlines(), 1):
    if not line.startswith('- **') or not re.search(r'\*\*(JSSC|ISSCC) 20\d{2}', line):
        continue
    title = re.search(r'^- \*\*(.+?)\*\*', line)[1]
    venue = re.search(r'\*\*((?:JSSC|ISSCC) 20\d{2})\*\*', line)[1]
    is_jssc = venue.startswith('JSSC')
    supplement_record(title, 'conditional-computing CNN' if is_jssc else 'C-DNN', venue, line,
        ('已读一手摘要，未完整复现；补充原文的既有阅读声明。' if is_jssc else
         '作者实验室介绍；已核异构分工及梯度驱动稀疏生成，未完整读方法。'),
        'primary_abstract_reported' if is_jssc else 'author_description_reported',
        'CICC-COMPANION-JSSC2021' if is_jssc else 'CICC-COMPANION-CDNN', index)

other_source = AUDIT + '/venue_supplement_other.json'
other_supplement = load(other_source)
other_levels = ['full_text_reported', 'full_text_reported', 'primary_abstract_reported',
                'primary_abstract_reported', 'partial_text_reported', 'author_method_text_reported',
                'primary_identity_only', 'primary_identity_only']
for index, raw in enumerate(other_supplement['papers'], 1):
    raw = dict(raw, venue=f"{raw['venue']} {raw['year']}", reading_level=other_levels[index-1],
        evidence_file=f'{IDEAS / other_source}#papers[{index-1}]',
        reason=join([raw.get('reason'), raw.get('mount'), raw.get('strong_control')]))
    add_record(raw, other_source, f'VENUE-SUPP-{index:02d}')
unread_levels = ['primary_identity_only', 'identity_unconfirmed', 'primary_abstract_reported',
                 'primary_abstract_reported', 'primary_identity_only', 'not_read_in_source_scope',
                 'not_read_in_source_scope']
for index, raw in enumerate(other_supplement['important_unread'], 1):
    raw = dict(raw, reading_depth=raw['reason'], reading_level=unread_levels[index-1],
        status='原补表 important_unread：保留相关线索，缺读不作失败判决',
        what_untried=raw['reason'], evidence_file=f'{IDEAS / other_source}#important_unread[{index-1}]')
    add_record(raw, other_source, f'VENUE-UNREAD-{index:02d}')

# The September 10 appendix explicitly does not alter the older CSV count; load
# its six later additions separately so the CSV freeze does not hide new priors.
appendix = 'research/literature_audit_20260909/LITERATURE_INVENTORY.md'
active = False
for line_no, line in enumerate((IDEAS / appendix).read_text().splitlines(), 1):
    if line.startswith('## 9月10日'):
        active = True
    if active and line.startswith('| ['):
        cells = [x.strip() for x in line.strip('|').split('|')]
        name = re.match(r'\[([^\]]+)\]', cells[0]).group(1)
        venue = re.sub(r'\[[^\]]+\]\([^\)]+\)', '', cells[0]).strip('；， ').replace('**', '')
        if name == 'DMP-SNN':
            venue = 'Nature Machine Intelligence 2026（原记录出版页）'
        add_record({'name': name, 'title': name, 'venue': venue, 'source_url': urls(cells[0]),
                    'reading_depth': cells[1], 'status': cells[2], 'reason': cells[1], 'what_untried': cells[2]},
                   appendix, f'APPENDIX-{line_no}')

# Precision CSV is an annotation of the same source UIDs, not 732 new works.
precision = {}
with (IDEAS / SURVEY / 'literature_precision_732.csv').open(encoding='utf-8-sig') as f:
    for d in csv.DictReader(f):
        precision[d['uid']] = d
for row in records:
    if row['record_ref'] in precision:
        row['priority'] = precision[row['record_ref']]['precision_tier']

# Pro literature tables add named records missing from the inventories. Their
# prose claims remain source claims, not new venue/full-text verification.
pro_files = sorted((IDEAS / 'gptpro').glob('*.md'))
for p in pro_files:
    rel = str(p.relative_to(IDEAS))
    active = False
    for line_no, line in enumerate(p.read_text().splitlines(), 1):
        if line.startswith('#'):
            active = ('重点文献矩阵' in line or '与当前挂点直接相关的文献' in line)
        if not active or not line.startswith('| **'):
            continue
        cells = [x.strip() for x in line.strip('|').split('|')]
        name = cells[0].replace('**', '')
        first_round = '硬件idea深挖' in p.name
        if first_round:
            bits = re.split(r'[，,]', name, maxsplit=1)
            name, venue = bits[0], bits[1] if len(bits) == 2 else ''
            role = cells[1]
        else:
            venue, role = cells[1].replace('**', ''), cells[2]
        kind = 'paper_family' if name in {'NeuFlow / NeuFlow v2', 'FireFly-S / FireFly-T', 'BitVert / BBS'} else 'paper'
        raw = {'name': name, 'full_title': name, 'venue': venue, 'source_url': urls(line),
               'reading_depth': 'Pro 调研中的文献矩阵；汇总本轮未重新全文核验',
               'status': 'Pro 建议/对照线索；详见引用段落，不等于本地已迁入',
               'reason': role + '；' + cells[-1]}
        add_record(raw, rel, f'PRO-{p.stem}-{line_no}', kind)

# Primary-source corrections are local catalogue overlays; source audits stay
# unchanged. MICRO and NeurIPS channel-gating papers have different titles.
followup = 'research/hardware_innovation_20260908/open_fusion_execution/literature_followup.md'
for row in records:
    if row['record_ref'] == 'MAIN-R132':
        old_title = row['full_title']
        row['full_title'] = 'Boosting the Performance of CNN Accelerators with Dynamic Fine-Grained Channel Gating'
        row['reason'] += '；2026-09-12按作者MICRO PDF首页补正题名；源表原题名为 '+old_title+'，另有NeurIPS2019同族论文，不合并两篇。'
for ref, raw in [
    ('FOLLOWUP-CGNET-NIPS2019', dict(name='Channel Gating Neural Networks',
        venue='NeurIPS 2019', source_url=['https://proceedings.neurips.cc/paper/2019/file/68b1fbe7f16e4ae3024973f12f3cb313-Paper.pdf', 'https://github.com/cornell-zhang/dnn-gating'],
        reading_depth='本轮官方论文首页/摘要与作者代码README；非整篇精读',
        status='纳入phase剪枝与条件计算最近邻；未完整迁入',
        reason='空间-通道条件执行及算法/硬件联合已有，不允许把phase mask本身说成首创。',
        what_untried='原作者完整训练与条件分支硬件；当前先测试静态消费者phase/H8源生产删除。')),
    ('FOLLOWUP-SPSRC-WACV2024', dict(name='SPSRC',
        full_title='Towards Better Structured Pruning Saliency by Reorganizing Convolution',
        venue='WACV 2024', source_url=['https://openaccess.thecvf.com/content/WACV2024/papers/Sun_Towards_Better_Structured_Pruning_Saliency_by_Reorganizing_Convolution_WACV_2024_paper.pdf', 'https://github.com/AlexSunNik/SPSRC'],
        reading_depth='本轮官方论文摘要/方法入口与作者README，非全文精读',
        status='新纳入更强剪枝saliency对照，未实验',
        reason='重组相邻卷积以表达空间saliency已有；真实消费者损失优于幅值不能独自充当X。',
        what_untried='作者conv_to_mat与谱/核/Frobenius范数、相同预算恢复；对比完整RNE后gate/PED真实误差目标。')),
]:
    add_record(raw, followup, ref)

# Append new supplement entities after the existing inventory so unrelated
# existing W identifiers remain stable on this regeneration.
records.sort(key=lambda r: r['source_file'] in (cicc_source, other_source))

# Current execution priors are deliberately appended after the historical
# inventory and comparison supplements; the existing W0001-W0754 IDs remain.
local_execution_source = str((OUT / 'local_execution_priors.json').relative_to(IDEAS))
local_execution_priors = load(local_execution_source)
for raw in local_execution_priors['works']:
    add_record(raw, local_execution_source, raw['record_ref'])

# Merge only exact source identity, exact full title + compatible entity type,
# single identical arXiv ID for paper records, or one identical repo URL for code.
# A paper and its code are separate entities. No fuzzy title/name similarity.
parent = list(range(len(records)))


def root(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def union(a, b):
    parent[root(b)] = root(a)


keys = defaultdict(list)
for i, row in enumerate(records):
    keys[('ref', row['record_ref'])].append(i)
    typ = row['entity_type']
    title = normalized(row['full_title'])
    if len(title) > 10:
        keys[('title', typ, title)].append(i)
    pid = paper_id(row)
    if pid:
        keys[('paper_id', pid)].append(i)
    repos = uniq(repo_url(u) for u in row['links'])
    if typ == 'open_source' and len(repos) == 1:
        keys[('repository', repos[0])].append(i)
for kind_key, ids in keys.items():
    for i in ids[1:]:
        union(ids[0], i)

# Cross-audit descriptions sometimes replace a title by a mechanism gloss. The
# following exact IDs were inspected against name, venue/year, mechanism and the
# original record. This is an explicit alias map, not fuzzy title matching.
# Conflicting/family entries (e.g. Avalanche's venue and FlashAttention / FA2 / FA3)
# are intentionally left separate and appear in possible_aliases.csv.
audited_aliases = {
    'R001': 'SP002', 'R002': 'SP001', 'R018': 'SP015', 'R026': 'SP006',
    'R027': 'SP005', 'R031': 'SP010', 'R037': 'SP013',
    'R051': 'QT001', 'R052': 'QT002', 'R053': 'QT003',
    'R054': 'SN005', 'R055': 'SN006', 'R056': 'SN007', 'R057': 'SN008',
    'R062': 'SN012', 'R068': 'SN011', 'R075': 'AT004', 'R077': 'AT003',
    'R078': 'AT005', 'R098': 'SP026', 'R100': 'SP024', 'R102': 'CR002',
    'R103': 'CR001', 'R106': 'SP016', 'R129': 'SP009', 'R147': 'CR003',
    'R148': 'CR004', 'R199': 'SN018', 'R200': 'OF009', 'R218': 'OF008',
    'R229': 'OF011', 'R230': 'SP028', 'R236': 'SP027', 'R241': 'OF015',
    'R261': 'OF010', 'R268': 'CM001', 'R269': 'SN004', 'R273': 'OF017',
    'R284': 'CM002', 'R285': 'CM003', 'R286': 'CM004', 'R290': 'OF012',
    'R292': 'OF002', 'R329': 'IS006', 'R330': 'IS007', 'R333': 'IS010',
    'R334': 'IS011', 'R347': 'SP014',
}
ref_index = {r['record_ref']: i for i, r in enumerate(records)}
for main_id, musha_id in audited_aliases.items():
    a, b = ref_index.get('MAIN-' + main_id), ref_index.get('MUSHA-' + musha_id)
    if a is not None and b is not None:
        union(a, b)

# These two identities are explicitly reconciled in the authored comparison
# supplement: Zhang was formerly a placeholder; its ISSCC companion was only
# a short-name lead in the other venue table. Keep all old provenance rows.
for a, b in [('MAIN-R270', 'CICC-SUPP-01'),
             ('CICC-COMPANION-CDNN', 'VENUE-UNREAD-07')]:
    union(ref_index[a], ref_index[b])

# A short-name Pro table with a named, same-year exact inventory alias can be
# attached to its already-identified paper. Do not apply this to unknown venues,
# multi-work families, or the general inventory (where acronym collisions exist).
for i, row in enumerate(records):
    if not row['source_file'].startswith('gptpro') or row['entity_type'] != 'paper':
        continue
    candidates = [j for j, other in enumerate(records)
                  if other['source_file'] == MAIN and other['entity_type'] == 'paper'
                  and normalized(other['name']) == normalized(row['name'])]
    if len(candidates) == 1:
        j = candidates[0]
        years_a = set(re.findall(r'20\d{2}', row['venue']))
        years_b = set(re.findall(r'20\d{2}', records[j]['venue']))
        if years_a & years_b:
            union(j, i)

grouped = defaultdict(list)
for i, row in enumerate(records):
    grouped[root(i)].append(row)

clusters = []
ref_to_cluster = {}
for members in grouped.values():
    # Prefer inventory's full title to a later short-name table.
    members.sort(key=lambda r: (r['source_file'] != MAIN, r['source_file'].startswith('gptpro'), not r['record_ref'].startswith('MAIN'), r['record_ref']))
    first = members[0]
    full_title = first['full_title']
    if '待核' in full_title:
        full_title = next((m['full_title'] for m in members
                           if m['source_file'] == cicc_source and '待核' not in m['full_title']), full_title)
    item = {
        'catalog_id': f'W{len(clusters) + 1:04d}', 'name': first['name'],
        'full_title': full_title, 'entity_type': first['entity_type'],
        'venue': join(m['venue'] for m in members), 'category': join(m['category'] for m in members),
        'source_refs': join(m['record_ref'] for m in members),
        'source_files': join(m['source_file'] for m in members),
        'source_urls': join(u for m in members for u in m['links']),
        'reading_status_by_source': join(f"{m['record_ref']}: {m['reading']}" for m in members if m['reading']),
        'reading_levels_by_source': join(f"{m['record_ref']}: {m['reading_level']}" for m in members if m['reading_level']),
        'attempt_status_by_source': join(f"{m['record_ref']}: {m['attempt']}" for m in members if m['attempt']),
        'include_stop_reasons': join(m['reason'] for m in members),
        'untried_interfaces': join(m['untried'] for m in members),
        'evidence_paths': join(m['evidence'] for m in members),
        'precision_tier': join(m['priority'] for m in members),
        'A': join(m['A'] for m in members), 'B': join(m['B'] for m in members),
        'X_hint': join(m['X'] for m in members), 'idea_card_paths': '',
        'code_status': '', 'code_urls': '', 'code_scope': '',
        'identity_note': '历史记录原样保留；当前 AT-LIF={0,θ}，推理 θ 可折权。',
    }
    clusters.append(item)
    for m in members:
        ref_to_cluster[m['record_ref']] = item

local_author_aliases = {
    'GustavSNN_HPCA2026_public_mirror.txt': 'MAIN-R002',
    'FlexSpIM_ISCAS2025_author.txt': 'MAIN-R347',
    'SCNN_ISCA2017_author.txt': 'MAIN-R031',
    'LoopTree_TCASAI2024_author.txt': 'MAIN-R106',
    'RISCSparse_ICCAD2024_author.txt': 'MAIN-R018',
    'ISSCC2025_23_2_ConvFormer_author.txt': 'DIRECT-cfmp',
}
for filename, ref in local_author_aliases.items():
    if ref in ref_to_cluster:
        ref_to_cluster['local_author_txt:' + filename] = ref_to_cluster[ref]

# Structured idea table and authored cards are views of ideas, not distinct
# title-level contributions and not newly measured results.
idea_rows = []
extract_path = SURVEY + '/idea_extract_per_paper.csv'
with (IDEAS / extract_path).open(encoding='utf-8-sig') as f:
    for d in csv.DictReader(f):
        item = ref_to_cluster.get(d['uid'])
        if item:
            for dst, src in [('A', 'A'), ('B', 'B'), ('X_hint', 'X_hint')]:
                item[dst] = join([item[dst], d[src]])
        idea_rows.append({'idea_ref': d['uid'], 'name': d['name'], 'view_type': 'per_paper_fusion_hint',
                          'catalog_id': item['catalog_id'] if item else '', 'A': d['A'], 'B': d['B'],
                          'X': d['X_hint'], 'status': d['f_or_stageB'], 'boundary': d['boundary'],
                          'next_or_stop': d['kill_gate'], 'source': extract_path, 'detail': d['idea_bullets']})
source_counts[extract_path] = len(idea_rows)

card_paths = sorted((IDEAS / SURVEY / 'idea_cards').glob('*.md'))
for p in card_paths:
    body = p.read_text()
    refs = uniq(re.findall(r'\b(?:MAIN-R\d+|MUSHA-[A-Z]+\d+|OS2?-\d+)\b', body))
    for ref in refs:
        item = ref_to_cluster.get(ref)
        if item:
            item['idea_card_paths'] = join([item['idea_card_paths'], str(p.relative_to(IDEAS))])

screen_path = 'research/idea_screen_20260907/scoreboard.json'
for d in load(screen_path)['cards']:
    idea_rows.append({'idea_ref': 'SCREEN-' + d['id'], 'name': d['pitch'], 'view_type': 'screen_20260907',
                      'catalog_id': '', 'A': '', 'B': '', 'X': d['pitch'],
                      'status': join([d.get('s1_status'), d.get('highest_stage_reached')]),
                      'boundary': '2026-09-07 历史筛查；QK 统计为 ep35，不能当 ep34 或当前执行结论',
                      'next_or_stop': text(d.get('next_action', d.get('next', ''))), 'source': screen_path,
                      'detail': text(d)})

kdense_path = 'research/idea_screen_20260907/kdense/session.json'
for d in load(kdense_path)['ideas']:
    idea_rows.append({'idea_ref': 'KDENSE-' + d['id'], 'name': d['statement'], 'view_type': 'brainstorm_record',
                      'catalog_id': '', 'A': '', 'B': '', 'X': d['statement'], 'status': d['status'],
                      'boundary': '原记录的科研假设/打分，不是实验或独立多人复核',
                      'next_or_stop': join(d.get('disconfirming_evidence', [])), 'source': kdense_path,
                      'detail': text(d)})

for p in pro_files:
    lines = p.read_text().splitlines()
    for line_no, line in enumerate(lines, 1):
        if line.startswith(('# 三、新增候选', '# 四、F1', '# 五、F2', '# 六、F5', '# 七、F3', '## 6.1 候选', '## 6.2 候选', '## 7.1 F2', '## 7.2 F3')):
            idea_rows.append({'idea_ref': f'PRO-IDEA-{p.stem}-{line_no}', 'name': line.lstrip('# '),
                              'view_type': 'pro_candidate_section', 'catalog_id': '', 'A': '', 'B': '', 'X': line.lstrip('# '),
                              'status': '候选/建议；最新本地测试见主执行页',
                              'boundary': '保留家族；旧建议不是当前测试结论', 'next_or_stop': '',
                              'source': f'{p.relative_to(IDEAS)}:{line_no}', 'detail': ''})

# Preserve every text entrypoint so ideas outside the normalized source tables
# remain discoverable. Scan is indexing, not a claim to have read all documents.
documents = []
repository_mentions = defaultdict(list)
all_links = defaultdict(list)
for p in sorted(IDEAS.rglob('*.md')):
    if OUT in p.parents or any(s in p.parts for s in SKIP):
        continue
    rel = str(p.relative_to(IDEAS))
    body = p.read_text(errors='replace')
    headings = [line.lstrip('# ').strip() for line in body.splitlines() if re.match(r'^#{1,3} ', line)]
    documents.append({'path': rel, 'title': headings[0] if headings else p.stem,
                      'headings': join(headings[1:]), 'indexing_status': 'machine_indexed_not_full_read',
                      'url_count': len(urls(body))})
    for line_no, line in enumerate(body.splitlines(), 1):
        for url in urls(line):
            all_links[url].append(f'{rel}:{line_no}')
            repo = repo_url(url)
            if repo:
                repository_mentions[repo].append((rel, line_no, line.strip()))
for item in clusters:
    for url in urls(item['source_urls']):
        repo = repo_url(url)
        if repo:
            repository_mentions[repo].append((item['source_files'].split(' || ')[0], 0, item['name']))

# Explicit repository attribution already recorded in the Pro public-code table.
# These are reported author-code links, not a fresh clone/license/fullness audit.
pro_code_source = 'gptpro/ChatGPTpro-#第二轮跨领域调研：.md:267'
known_code = {
    'https://github.com/fangwei123456/parallel-spiking-neuron': ('PSN', 'author_code_reported', '神经元算法实现'),
    'https://github.com/vainf/torch-pruning': ('DepGraph', 'author_code_reported', '剪枝依赖图/算法工具'),
    'https://github.com/ruokaiyin/duogpt': ('DuoGPT', 'author_code_reported', '校准、fake_prune、评估；非 RTL'),
    'https://github.com/ruokaiyin/loas': ('LoAS', 'author_code_reported', 'artifact/pruning_gen/训练分析；不宣称完整生产 RTL'),
    'https://github.com/dubcyfor3/prosperity': ('Prosperity', 'author_code_reported', '周期模拟器/CUDA/数据；原记录称 Synopsys 脚本未公开'),
    'https://github.com/facebookresearch/deltacnn': ('DeltaCNN', 'author_code_reported', 'CUDA/PyTorch 差分执行；非 ASIC RTL'),
    'https://github.com/sfmth/openspike': ('OpenSpike', 'author_code_reported', 'Verilog/OpenLane/OpenRAM 工程；不能借用为本地 PPA'),
    'https://github.com/ridgerchu/tcja': ('TCJA-SNN', 'author_code_reported', '时间-通道注意力算法'),
    'https://github.com/neufieldrobotics/neuflow_v2': ('NeuFlow v2', 'author_code_reported', '光流算法实现'),
    'https://github.com/ruokaiyin/celty': ('Celty', 'author_code_partial_reported', '原记录明确 README 仍称将上传，按部分公开'),
    'https://github.com/pulp-platform/stream-ebpc': ('EBPC', 'author_code_reported', '流式无损压缩/解码 RTL；主审计 R043 来源声明'),
    'https://github.com/calad0i/da4ml': ('da4ml', 'author_code_reported', '官方 CMVM/CSE/硬件生成；本地已安装并调用，非本地完整执行核'),
    'https://github.com/yc2367/bbs-micro': ('BitVert / BBS', 'author_code_reported', 'BBS 位列剪枝算法作者工件；完整硬件另计'),
    'https://github.com/mostafaelhoushi/deepshift': ('DeepShift', 'author_code_reported', 'Q/PS 训练、量化、压缩、CUDA；非 ASIC RTL'),
    'https://github.com/cornell-zhang/dnn-gating': ('Precision Gating', 'author_code_reported', '精度门控算法和训练代码'),
    'https://github.com/alexsunnik/spsrc': ('SPSRC', 'author_code_reported', '卷积重组saliency、剪枝、微调与评价；本轮读作者README，未迁入或宣称硬件工件'),
    'https://github.com/seolabcornell/im_snn-spqunat_snn': ('IM-SNN', 'author_code_reported', 'SNN 权重/膜量化作者算法仓；原拼写 SpQunat 保留'),
}
code_evidence = {
    'https://github.com/pulp-platform/stream-ebpc': MAIN,
    'https://github.com/calad0i/da4ml': 'research/hardware_innovation_20260908/psn/cmvm_20260909/README.md:7',
    'https://github.com/yc2367/bbs-micro': 'research/hardware_innovation_20260908/algorithm/patch_probe/residual_consumer_probe/projection_fusion_one_page.md:7',
    'https://github.com/mostafaelhoushi/deepshift': 'research/literature_audit_20260909/LITERATURE_INVENTORY.md:478',
    'https://github.com/cornell-zhang/dnn-gating': 'research/literature_audit_20260909/LITERATURE_INVENTORY.md:476',
    'https://github.com/alexsunnik/spsrc': followup,
    'https://github.com/seolabcornell/im_snn-spqunat_snn': 'research/literature_audit_20260909/LITERATURE_INVENTORY.md:475',
}
code_rows = []
for repo, mentions in sorted(repository_mentions.items()):
    known = known_code.get(repo)
    # Do not infer official authorship simply from a GitHub URL or a paper mirror.
    origin = known[1] if known else 'repository_link_owner_unverified'
    if not known and all(re.search(r'第三方|third.party|非官方', m[2], re.I) for m in mentions):
        origin = 'third_party_reported'
    code_rows.append({'repository': repo, 'associated_work_reported': known[0] if known else '',
                      'origin_status': origin, 'scope_reported': known[2] if known else '现有材料中的仓库入口；实现/许可/作者关系待核',
                      'attribution_source': code_evidence.get(repo, pro_code_source) if known else '',
                      'mention_count': len(mentions),
                      'source_locations': join(f'{p}:{line}' for p, line, _ in mentions),
                      'example_context': join(m[2] for m in mentions[:3]),
                      'verification_this_run': 'local aggregation only; no clone or completeness claim'})

for item in clusters:
    direct_repos = uniq(repo_url(u) for u in urls(item['source_urls']))
    short = normalized(item['name'])
    attached = [repo for repo, value in known_code.items()
                if short == normalized(value[0]) or short == normalized(value[0]) + 'authorcode']
    repos = uniq(direct_repos + attached)
    item['code_urls'] = join(repos)
    statuses = []
    for repo in repos:
        statuses.append(known_code[repo][1] if repo in known_code else 'repository_link_owner_unverified')
    item['code_status'] = join(statuses) if statuses else 'code_not_located_in_aggregated_sources'
    item['code_scope'] = join(known_code[r][2] for r in repos if r in known_code)
    if item['entity_type'] in {'internal_idea', 'internal_mechanism', 'component_alias'}:
        item['code_status'] = 'internal_candidate_not_public_paper; ' + item['code_status']
    if item['name'] == 'GustavSNN-public-mirror':
        item['code_scope'] = '论文镜像/本地迁移入口；原 open_source 标签不能证明作者代码已开源'

fields = list(clusters[0])
csv_write('works.csv', clusters, fields)
csv_write('idea_views.csv', idea_rows, list(idea_rows[0]))
csv_write('repositories.csv', code_rows, list(code_rows[0]))
csv_write('document_index.csv', documents, list(documents[0]))
csv_write('source_records.csv', [{**r, 'links': join(r['links']), 'cards': join(r['cards'])} for r in records], list(records[0]))
csv_write('source_links.csv', [{'url': u, 'source_locations': join(loc)} for u, loc in sorted(all_links.items())], ['url', 'source_locations'])

# Full title/venue variants stay visible; here users can inspect identifier joins.
merge_rows = []
for members in grouped.values():
    titles = uniq(m['full_title'] for m in members)
    if len(titles) > 1:
        item = ref_to_cluster[members[0]['record_ref']]
        merge_rows.append({'catalog_id': item['catalog_id'], 'names': join(m['name'] for m in members),
                           'titles': join(titles), 'source_refs': item['source_refs'],
                           'basis': 'source UID / exact title / single arXiv paper ID / same repo / explicit cross-audit alias / Pro exact-name same-year alias; no fuzzy title'})
csv_write('identity_variants.csv', merge_rows, ['catalog_id', 'names', 'titles', 'source_refs', 'basis'])

by_name = defaultdict(list)
for item in clusters:
    if item['entity_type'] == 'paper':
        by_name[normalized(item['name'])].append(item)
possible_aliases = [{'name': rows[0]['name'], 'catalog_ids': join(r['catalog_id'] for r in rows),
                     'titles': join(r['full_title'] for r in rows), 'venues': join(r['venue'] for r in rows),
                     'reason_unmerged': 'same short name only; source family, venue or identity remains unresolved'}
                    for rows in by_name.values() if len(rows) > 1]
csv_write('possible_aliases.csv', possible_aliases, ['name', 'catalog_ids', 'titles', 'venues', 'reason_unmerged'])

venue_patterns = {
    'HPCA': r'\bHPCA\b', 'ISCA': r'\bISCA\b', 'MICRO': r'\bMICRO\b',
    'ASPLOS': r'\bASPLOS\b', 'DATE': r'\bDATE\b', 'ICCAD': r'\bICCAD\b',
    'ISSCC': r'\bISSCC\b', 'JSSC': r'\bJSSC\b|Journal of Solid.State Circuits',
    'CICC': r'\bCICC\b',
    'TCAS（含系列）': r'\bTCAS(?:[-– ]?(?:I+|AI))?\b|Transactions on Circuits and Systems',
    'TCAS-II': r'\b(?:TCAS|TCS)[-– ]?II\b|Circuits and Systems II\b',
    'TCAS-I': r'\b(?:TCAS|TCS)[-– ]?I\b|Circuits and Systems I\b',
    'ISCAS': r'\bISCAS\b', 'ESSCIRC': r'\bESSCIRC\b', 'ESSERC': r'\bESSERC\b',
    'A-SSCC': r'\bA[-– ]SSCC\b|Asian Solid.State Circuits Conference',
    'VLSI Symposium': r'(?:Symposium|Symp\.).*VLSI|\bVLSI (?:20\d{2}|Symposium|Technology|Circuits)',
    'TVLSI（期刊）': r'\bTVLSI\b|Transactions on Very Large Scale Integration',
    'DAC': r'\bDAC\b', 'FPGA': r'\bFPGA\b', 'FPL': r'\bFPL\b',
    'ICLR': r'\bICLR\b', 'CVPR': r'\bCVPR\b', 'ICCV': r'\bICCV\b',
    'WACV': r'\bWACV\b', 'AAAI': r'\bAAAI\b', 'NeurIPS': r'\b(?:NeurIPS|NIPS)\b',
    'ICML': r'\bICML\b', 'MLSys': r'\bMLSys\b', 'OOPSLA': r'\bOOPSLA\b',
    'ICDE': r'\bICDE\b',
}
# Historical fields also contain "ISCA2020" and "CVPR2021 Workshops".
# ASCII-letter boundaries admit attached years while still separating ISCAS.
for venue in ('HPCA', 'ISCA', 'MICRO', 'ASPLOS', 'DATE', 'ICCAD', 'ISSCC', 'CICC',
              'ISCAS', 'ESSCIRC', 'ESSERC', 'DAC', 'FPGA', 'FPL', 'ICLR', 'CVPR',
              'ICCV', 'WACV', 'AAAI', 'ICML', 'MLSys', 'OOPSLA', 'ICDE'):
    venue_patterns[venue] = rf'(?<![A-Za-z]){venue}(?![A-Za-z])'
venue_patterns['NeurIPS'] = r'(?<![A-Za-z])(?:NeurIPS|NIPS)(?![A-Za-z])'
venue_items = {venue: [i for i in clusters if i['entity_type'] == 'paper' and
                re.search(pattern, i['venue'], re.I)] for venue, pattern in venue_patterns.items()}
supplement_records = [r for r in records if r['source_file'] in (cicc_source, other_source)]
supplement_clusters = {ref_to_cluster[r['record_ref']]['catalog_id'] for r in supplement_records}
supplement_existing = {ref_to_cluster[r['record_ref']]['catalog_id'] for r in records
                       if r['source_file'] not in (cicc_source, other_source)}
execution_clusters = {ref_to_cluster[r['record_ref']]['catalog_id']
                      for r in records if r['source_file'] == local_execution_source}
execution_existing = {ref_to_cluster[r['record_ref']]['catalog_id']
                      for r in records if r['source_file'] != local_execution_source}
summary = {'work_entities': len(clusters), 'input_work_records': len(records),
           'entity_types': dict(Counter(i['entity_type'] for i in clusters)),
           'repository_urls': len(code_rows), 'repository_origin_status': dict(Counter(i['origin_status'] for i in code_rows)),
           'idea_views': len(idea_rows), 'idea_view_types': dict(Counter(i['view_type'] for i in idea_rows)),
           'indexed_documents': len(documents), 'idea_card_documents': len(card_paths),
           'source_urls': len(all_links), 'source_counts': source_counts,
           'identity_variant_groups': len(merge_rows),
           'unmerged_same_name_groups': len(possible_aliases),
           'venue_paper_counts': {v: len(items) for v, items in venue_items.items()},
           'local_execution_priors': dict(source_records=len(local_execution_priors['works']),
                represented_work_entities=len(execution_clusters),
                new_work_entities=len(execution_clusters-execution_existing),
                existing_work_entities_enriched=len(execution_clusters & execution_existing),
                scope=local_execution_priors['scope']),
           'comparison_supplement': dict(source_records=len(supplement_records),
                represented_work_entities=len(supplement_clusters),
                new_work_entities=len(supplement_clusters-supplement_existing),
                existing_work_entities_enriched=len(supplement_clusters & supplement_existing),
                cicc_records=sum(r['source_file'] == cicc_source and r['venue'].startswith('CICC') for r in supplement_records),
                cicc_reading_levels=dict(Counter(r['reading_level'] for r in supplement_records
                    if r['source_file'] == cicc_source and r['venue'].startswith('CICC'))),
                levels_meaning='Reading levels reported by the September9 authored supplement; no new PDF reading by this builder.'),
           'meaning': 'Existing-record aggregation. Counts are not full reads, unique novel mechanisms, executable artifacts, or experimental successes.'}
(OUT / 'counts.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')


def cell(value):
    return text(value).replace('|', '\\|').replace('\n', ' ')


lines = ['# 公开工作统一目录', '',
         '本表由现有库存和 Pro 文献矩阵聚合。阅读、去留与来源冲突保留在 CSV；不把“已公开论文”冒充“已开源实现”。', '',
         '| ID | 工作 / 类型 | 会议期刊（来源原称） | 代码状态 | 已有 idea 线索 |',
         '|---|---|---|---|---|']
for i in clusters:
    links = urls(i['source_urls'])
    name = f"[{cell(i['name'])}]({links[0]})" if links else cell(i['name'])
    hint = i['X_hint'][:200] or i['untried_interfaces'][:200] or '详见来源记录'
    lines.append(f"| {i['catalog_id']} | {name} / {i['entity_type']} | {cell(i['venue'])} | {cell(i['code_status'])} | {cell(hint)} |")
(OUT / 'WORKS.md').write_text('\n'.join(lines) + '\n')

lines = ['# 仓库入口：作者代码与第三方/未核身份分开', '',
         '链接来自全树 Markdown 和已有库存。未逐仓 clone，未做本轮许可证或完整性审计；作者代码标记依赖列出的本地来源声明。', '',
         '| 仓库 | 来源身份 | 内容范围（已有记录） |', '|---|---|---|']
for i in code_rows:
    lines.append(f"| [{cell(i['repository'].removeprefix('https://github.com/'))}]({i['repository']}) | {i['origin_status']} | {cell(i['scope_reported'])} |")
(OUT / 'REPOSITORIES.md').write_text('\n'.join(lines) + '\n')

lines = ['# 按会议期刊查询当前目录', '',
    '由 build_catalog.py 与 WORKS/counts 同步生成。按论文实体的 venue 字段检索；多刊会/待核项可能重复出现，不代表全文精读数。ISCA/ISCAS、TCAS-I/II、VLSI Symposium/TVLSI 分开匹配。', '',
    '| 会议/期刊 | 当前论文实体 | 对应 ID |', '|---|---:|---|']
for venue, items in venue_items.items():
    lines.append(f"| {venue} | {len(items)} | {', '.join(i['catalog_id'] for i in items)} |")
lines += ['', f"CICC 定向补表的16项现已全部纳入；当前 CICC 论文实体为 {len(venue_items['CICC'])}，不再因漏读补表显示为2篇。补表只有1项明确记为完整原文阅读，其他层级见 source_records.csv/works.csv；本轮没有重新阅读这些 PDF。",
    '', '同目录其他补表的8条主项与7条明确保留线索也已进入实体与来源表，重复项保留来源并归并。原补检为2022–2026定向检索，不是逐篇 proceedings 审计；2026部分节目不可完整取得、若干题名仅核身份或摘要，缺正文/完整工件仍是实际覆盖边界。',
    '', '[返回总表](WORKS.md) · [仓库入口](REPOSITORIES.md) · [CICC原补表]('+str(IDEAS/cicc_source)+') · [其他会刊原补表]('+str(IDEAS/other_source)+')']
(OUT / 'VENUES.md').write_text('\n'.join(lines)+'\n')

coverage_rows = [dict(venue='CICC', years='2022–2026',
    scope='September9 authored supplement: 2022–2025 official relevant program sessions plus2026 author/institution/IEEE targeted lookup.',
    reading_boundary='16 CICC records: 1 full text, 3 primary abstracts, 1 companion-demo abstract, 9 identity/title only, 1 partial text, 1 author description; stated prior reading, not new PDF work.',
    gap='2026 dynamic technical program was not fully retrievable; no complete annual proceedings review; most mechanism/control and original artifact transfers remain untried.',
    source=cicc_source, source_urls=join(cicc_programs.values()))]
for row in other_supplement['searched_venue_years']:
    coverage_rows.append(dict(venue=row['venue'], years=join(row['years']), scope=row['method'],
        reading_boundary='Per-paper original reading preserved in source_records.csv, including important_unread; not all full text.',
        gap=row['gap'], source=other_source, source_urls=join(row['sources'])))
csv_write('venue_coverage.csv', coverage_rows, list(coverage_rows[0]))

readme = f'''# 统一公开工作 / 开源实现 / 开放 idea 入口

这是可直接用于融合试做的导航汇总，不是等待全部文献核完才准实验的门槛。先从适配接口挑组合，在隔离原型中试，再把真实结果写回执行计划；历史负结果只约束当时的布局和身份。

当前聚合得到 **{len(clusters)} 个保守归并的工作实体、{len(code_rows)} 个仓库 URL、{len(idea_rows)} 条结构化 idea 视图、{len(documents)} 份 Markdown 入口**。另有 **{len(possible_aliases)} 组同名候选尚未消歧**，所以实体数不是已最终核净的论文数。实体包含论文、代码、内部别名等；idea 视图有重叠，不能称 {len(idea_rows)} 个独立新机制。全树索引是机器导航，不是本人全文阅读计数。

原表中“先等某门、暂不训练/RTL、停标题”等为当时的状态与建议，不构成本轮继续试做的限制。用户当前已授权先把适配组合做成隔离原型、测性能再决定保留；借入来源和本地新增部分仍需分别列明。

## 从哪里用

- [WORKS.md](WORKS.md) / [works.csv](works.csv)：全部归并实体、来源、阅读和尝试状态、A/B/X、未试接口、代码身份。
- [REPOSITORIES.md](REPOSITORIES.md) / [repositories.csv](repositories.csv)：全树仓库入口。`author_code_reported` 为原调研明确报告的作者代码；`third_party_reported` 为第三方声明；`repository_link_owner_unverified` 表示有链接但作者关系未核。有论文无代码使用 `code_not_located_in_aggregated_sources`，不等于不存在代码。
- [VENUES.md](VENUES.md)：与本表同步生成的会刊论文实体计数；补表阅读层级按原来源保留，不将实体数充作全文数。
- [venue_coverage.csv](venue_coverage.csv)：当前比较审计明确记录的会刊/年份检索边界、未取得正文与未试接口；不是全球覆盖证明。
- [idea_views.csv](idea_views.csv)：逐篇融合提取表、32 卡筛查、科研流程记录和 Pro 候选段落集中查询；状态带来源/时间，不假装统一裁决。
- [document_index.csv](document_index.csv)：全树 Markdown 的标题、章节入口；包括 Grok 新增目录、Pro、原始 Card 和具体试验结果。
- [source_records.csv](source_records.csv)、[source_links.csv](source_links.csv)、[identity_variants.csv](identity_variants.csv)：追溯归并与链接。[possible_aliases.csv](possible_aliases.csv) 保留未消歧的同名记录；代码实体不与论文实体合并。
- [counts.json](counts.json)：覆盖计数，含实体类型与 idea 视图类型。

## 归并方法与边界

自动归并依据同一源 UID、相同完整标题、论文唯一相同 arXiv 编号、或代码唯一相同仓库 URL。另对本轮实际核对了名称、刊会/年份、机制和原记录的跨审计别名，在脚本中逐个列明 MAIN/MUSHA ID；Pro 的精确同名同年记录可以挂回唯一库存项。CICC补表的Zhang完整题名与旧MAIN-R270占位明确归并，C-DNN短名线索与补表完整题名明确归并；旧来源和阅读冲突仍保留。没有模糊题名相似度归并。共享 arXiv 等别名保留在 `identity_variants.csv`，刊会冲突/多代作品族等未消歧项保留为不同实体。正文引用了很多论文的内部 Card 不参与论文 arXiv 归并。公开工作与开源实现是两个维度；GustavSNN-public-mirror 的旧 `open_source` 标签只是论文镜像/本地迁移入口，未被提升为作者完整工件。

旧材料中的身份冲突保留为历史来源陈述；本轮统一身份是 **AT-LIF={{0,θ}}，推理 θ 可吸收进下一层 W**，连续 PSN/PED/残差义务另算。旧 ep35 统计和旧 AEE 不能直接变成 ep34 或当前学生结果。这里不改写源材料，不生成新的性能/新颖性分数。

## 直接聚合的库存

| 输入 | 本轮读取条数 | 作用 |
|---|---:|---|
'''
for source, count in source_counts.items():
    readme += f'| [{source}]({IDEAS / source}) | {count} | 原记录/注释；非新增全文阅读 |\n'
readme += '''
## 本轮执行先验补录

五项均来自已使用的实际先验，不是新增广扫；BNFF 与 FlexAcc 归回原 ID，其余续排。底座/接口近邻与本地新增机制分开；阅读层级不自动升级。

| ID | 原工作与主来源 | 实际阅读范围 | 执行角色 / 未试部分 |
|---|---|---|---|
'''
for raw in local_execution_priors['works']:
    item = ref_to_cluster[raw['record_ref']]
    link = urls(raw['source_url'])[0]
    readme += (f"| {item['catalog_id']} | [{cell(raw['name'])}: {cell(raw['full_title'])}]({link}) "
               f"| {cell(raw['reading_depth'])} "
               f"| {cell(raw['status'])}；{cell(raw['what_untried'])} |\n")
readme += f'''
另外加载同 UID 的 `literature_precision_732.csv` 作为精度分层注释，关联 **{len(card_paths)}** 篇 `idea_cards/*.md`。全部 Markdown 还扫描了文中 URL，使未进入历史 CSV 的新 Grok/Pro 内容仍可被找到。

## 主要目录入口

- [{MAIN.rsplit('/', 1)[0]}]({IDEAS / MAIN.rsplit('/', 1)[0] / 'README.md'})：Codex 去留审计及领域子清单。
- [{MUSHA.rsplit('/', 1)[0]}]({IDEAS / MUSHA.rsplit('/', 1)[0] / 'README.md'})：Grok 独立审计；未覆盖或替换。
- [audit_comparison_20260909]({IDEAS / AUDIT / 'README.md'})：CICC16条与2条伴随先验、其他电路会刊8条主项及7条明确保留缺读线索全部作为来源记录纳入；阅读层级是原补表声明。
- [{SURVEY}]({IDEAS / SURVEY})：732 库存、精度分层、逐篇 idea 提取、P0 补读、box RTL 和融合建议。
- [gptpro]({IDEAS / 'gptpro'})：两轮跨领域调研；文献矩阵和候选段落分别进入目录。
- [idea_screen_20260907]({IDEAS / 'research/idea_screen_20260907/README.md'})：32 卡、ep35 普查、Orchestra/K-Dense 原记录。
- [exploration_tcasii_20260911]({IDEAS / 'research/hardware_innovation_20260908/exploration_tcasii_20260911/CODEX_HANDOFF.md'})：可吸收身份后的 R2/R3/R4；文档与 URL 纳入导航，未凭 prose 自动宣称新论文身份。
- [codex_cards]({IDEAS / 'codex_cards'})、[Grok 4.6]({IDEAS / 'research/grok46_20260905/00_READ_THIS_FIRST.md'})、[早期独立包]({IDEAS / 'codex_independent_20260905/README.md'})：原始机制家族仍可查，不用当前单布局失败抹掉。

## 尚有缺口

1. 这是现有目录的全树入口和结构化记录归并，不是全球开源工作的穷尽搜索。新 prose 中未给出稳定题名/编号的条目只进入文档/链接索引，尚未逐条提升为作品实体。
2. 仓库 URL 不保证有效、完整、许可允许复用或和论文作者对应；除明确来源声明外均保留未核状态。借入时应读该仓库 README/代码和许可，不能继承其 PPA。
3. 历史阅读状态可能冲突（例如已有全文记录与后来某轮未取得全文）。保留每个来源，未自动取“最乐观”状态。
4. 论文族、内部组件别名、工具框架与单篇作品分开，尚不能把它们当同粒度的创新候选排名。
5. 本脚本未打开 PDF 全文、未执行任何候选、不取代主线程的最新性能结果。最新试验请从主执行目录进入。

复跑：

```bash
/usr/bin/python3.12 {OUT / 'build_catalog.py'}
```
'''
(OUT / 'README.md').write_text(readme)
print(json.dumps(summary, ensure_ascii=False, indent=2))
