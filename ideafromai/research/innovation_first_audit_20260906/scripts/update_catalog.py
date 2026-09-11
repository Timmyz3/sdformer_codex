#!/usr/bin/env python3.12
"""Revise only three navigation files, preserving their reviewed snapshots."""
from pathlib import Path
import json
import hashlib

BASE = Path(__file__).resolve().parents[1]
IDEAS = BASE.parents[1]
REPORT = 'research/innovation_first_audit_20260906/'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    receipt = BASE / 'catalog_updates.json'
    assert not receipt.exists(), 'catalog update already recorded'
    inventory = {x['path']: x for x in json.loads((BASE/'inventory.json').read_text())['files']}
    originals = {p: (IDEAS/p).read_bytes() for p in ['README.md', 'INDEX.json', 'MANIFEST.txt']}
    for p, data in originals.items():
        assert sha(data) == inventory[p]['sha256'], ('unexpected catalog change', p)
    readme = originals['README.md'].decode()
    start = readme.index('**2026-09-06 最新推进：**')
    end = readme.index('\n\n', start)
    replacement = '**2026-09-06 最新：创新优先逐条复审。** [可检索审阅台账](' + REPORT + 'C1C2创新逐条复审.html) · [研究说明](' + REPORT + 'README.md)。原始 42/42 文件、620 条定位记录已审（含重复、来源与工程史料），五个重构候选接受独立评审。当前优先研究浅层 FC1 的二值源统计先行，独立主机制潜力为 6/10；C1 尚无通过主创新筛选的方案。没有稳 accept、RTL 速度或 PPA 准入。\n\n**上一轮包降为实现探索：** [父槽重算与原始权重暂存](' + 'research/codex_deep_rebuild_20260906/README.md' + ') 的 CPU 结果原样保留，按用户创新优先要求，不再作为主创新推荐。'
    readme = readme[:start] + replacement + readme[end:]
    readme = readme.replace('The two external survey packs are listed below. Read **both** alongside the latest Codex screening pack, then follow the frozen-ep34 identity.', '下列两个外部包均已逐条复审；先看最新复审结论，再读各包原始假说。冻结 ep34 身份优先。')
    index = json.loads(originals['INDEX.json'])
    index['packs'].insert(0, {'id':'innovation_first_audit_20260906', 'author':'Codex with independent cross-reviews', 'subdir':REPORT, 'synthesis':REPORT+'README.md', 'interactive_audit':REPORT+'C1C2创新逐条复审.html', 'coverage':{'original_files':42,'location_records_including_repeats_and_evidence':620}, 'headline':'Innovation-first audit: source-first dynamic-BN moments in shallow FC1 remain a conditional candidate; no C1 headline has passed', 'evidence':'Fixed sample0 CPU source statistics and independent research reviews only; RTL/PPA admission 0', 'acceptance_claim':False})
    for pack in index['packs']:
        if pack['id'] == 'codex_deep_rebuild_20260906':
            pack['status'] = 'IMPLEMENTATION_EXPLORATION_NOT_HEADLINE_INNOVATION'
            pack['priority_override'] = REPORT+'synthesis.md'
    manifest = originals['MANIFEST.txt'].decode()
    header = '# Canonical: /home/zhumd/work/sdformer_codex/ideafromai/\n# Latest: innovation-first audit; older experiments are preserved as implementation evidence.\n\n'
    paths = [REPORT+'README.md', REPORT+'C1C2创新逐条复审.html', REPORT+'report-source.md', REPORT+'audit-ledger.json', REPORT+'QA.json']
    body = [line for line in manifest.splitlines() if line and not line.startswith('#')]
    for path in inventory:
        absolute = str(IDEAS/path)
        if absolute not in body:
            body.append(absolute)
    updated = {
        'README.md':readme.encode(),
        'INDEX.json':(json.dumps(index,ensure_ascii=False,indent=2)+'\n').encode(),
        'MANIFEST.txt':(header+'\n'.join([str(IDEAS/p) for p in paths]+body)+'\n').encode()
    }
    updates=[]
    for p, data in originals.items():
        snapshot = BASE/'original_snapshots'/p
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        assert not snapshot.exists(), ('snapshot exists',p)
        snapshot.write_bytes(data)
        (IDEAS/p).write_bytes(updated[p])
        assert (IDEAS/p).read_bytes() == updated[p]
        updates.append({'path':p,'before_sha256':sha(data),'after_sha256':sha(updated[p]),'snapshot':str(snapshot.relative_to(BASE)), 'scope':'navigation priority only; original proposals and experiment results unchanged'})
    receipt.write_text(json.dumps({'date':'2026-09-06','updates':updates},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'updated':[u['path'] for u in updates], 'snapshots_verified':True},ensure_ascii=False))

if __name__ == '__main__':
    main()
