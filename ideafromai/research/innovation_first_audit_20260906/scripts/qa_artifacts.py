#!/usr/bin/env python3.12
"""Check coverage, local links and actual filter JS without claiming visual QA."""
from pathlib import Path
from urllib.parse import urlparse, unquote
import json
import re
import hashlib
import subprocess
from bs4 import BeautifulSoup

BASE = Path(__file__).resolve().parents[1]
REPO = Path('/home/zhumd/work/sdformer_codex')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ledger = json.loads((BASE/'audit-ledger.json').read_text())
    html_path = BASE/'C1C2创新逐条复审.html'
    doc = BeautifulSoup(html_path.read_text(), 'html.parser')
    entries = doc.select('details.idea')
    sections = doc.select('section.file')
    assert len(entries) == ledger['entry_count_including_repeat_occurrences_and_evidence_rows'] == 620
    assert len(sections) == ledger['reviewed_files_including_prior_round'] == 43
    expected = {i['id']:i for f in ledger['files'] for i in f['ideas']}
    seen = set()
    for e in entries:
        ident = e.select_one('.identity').get_text().split(' · ',1)[0]
        assert ident in expected and ident not in seen, ident
        seen.add(ident)
        score = expected[ident]['scores']['T']
        assert int(e['data-score']) == (-1 if score is None else score)
        assert len(e.select('dt')) == len(e.select('dd')) == 7
    local_links = set()
    for a in doc.find_all('a',href=True):
        href = a['href']
        if not href or href.startswith('#') or urlparse(href).scheme:
            continue
        name = re.sub(r':\d+$','',unquote(href).split('#')[0])
        p = Path(name)
        p = p if p.is_absolute() else BASE/p
        assert p.exists(), ('broken local link',href)
        local_links.add(str(p))
    indices = {id(e):j for j,e in enumerate(entries)}
    payload = {
        'entries':[{'text':e['data-text'],'score':e['data-score']} for e in entries],
        'sections':[[indices[id(e)] for e in f.select('.idea')] for f in sections],
        'script':doc.find('script').string,
    }
    node = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const data=JSON.parse(fs.readFileSync(0,'utf8'));
const entries=data.entries.map(e=>({dataset:e,hidden:false,open:false}));
const files=data.sections.map(ix=>({hidden:false,querySelectorAll:()=>ix.map(i=>entries[i])}));
const controls={q:{value:''},min:{value:'-1'},expand:{},collapse:{},count:{}};
for(const c of Object.values(controls))c.addEventListener=(name,fn)=>{c[name]=fn};
const ctx={document:{getElementById:id=>controls[id],querySelectorAll:s=>s==='.idea'?entries:files}};
vm.createContext(ctx);vm.runInContext(data.script,ctx);
const count=()=>entries.filter(e=>!e.hidden).length;
assert.strictEqual(count(),620);
controls.min.value='6';controls.min.change();assert.strictEqual(count(),0);
controls.min.value='4';controls.min.change();assert.strictEqual(count(),29);
controls.min.value='-1';controls.q.value='Prosperity';controls.q.input();assert(count()>0&&count()<620);
controls.expand.onclick();assert(entries.filter(e=>!e.hidden).every(e=>e.open));
controls.collapse.onclick();assert(entries.every(e=>!e.open));
controls.q.value='no_such_idea_xyz_20260906';controls.q.input();assert.strictEqual(count(),0);
controls.q.value='';controls.q.input();assert.strictEqual(count(),620);
process.stdout.write(JSON.stringify({filter_search_expand_collapse:'PASS',scope:'actual JS in a minimal DOM; not browser rendering'}));
'''
    js = subprocess.run(['node','-e',node],input=json.dumps(payload,ensure_ascii=False),text=True,capture_output=True,check=True,timeout=15)
    js_result = json.loads(js.stdout)
    protected = REPO/'SDformer/hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md'
    digest = sha(protected)
    assert digest == 'dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4'
    status = subprocess.run(['git','status','--short'],cwd=REPO,text=True,capture_output=True,check=True).stdout
    expected_status = ' M SDformer/hw_autoresearch_nts07/reviews/tcasii_accelerator_story_and_next_ideas_20260905.md\n'
    assert status == expected_status, ('main repository status changed',status)
    qa_path = BASE/'QA.json'
    qa = json.loads(qa_path.read_text())
    qa.update({
        'html_structure':{'status':'PASS','file_sections':43,'entries':620,'seven_analysis_fields_per_entry':True},
        'local_links':{'status':'PASS','unique_checked':len(local_links)},
        'interaction':js_result,
        'protected_docs359_sha256':digest,
        'main_repository_status':{'status':'UNCHANGED_FROM_HANDOFF','preexisting_dirty_file_only':status.strip()},
        'artifact_sha256':{p:sha(BASE/p) for p in ['synthesis.md','report-source.md','audit-ledger.json','claim-source-ledger.json','C1C2创新逐条复审.html']},
        'boundary':'Content independent QA records refer to their recorded input snapshot. Final additions are checked by the coordinator; browser visual rendering was not available.'
    })
    independent = BASE/'records/synthesis_independent_qa.json'
    if independent.exists():
        qa['independent_content_qa']={'path':str(independent),'sha256':sha(independent)}
    qa_path.write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'structure':'PASS','local_links':'PASS','interaction':'PASS','protected_repo':'PASS','visual_browser_check':'NOT_AVAILABLE'},ensure_ascii=False))

if __name__ == '__main__':
    main()
