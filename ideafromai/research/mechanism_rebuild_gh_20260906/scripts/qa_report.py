"""Validate the delivered artifact and its explicitly frozen read scope."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import json
import subprocess

BASE = Path(__file__).resolve().parents[1]
REPO = Path('/home/zhumd/work/sdformer_codex')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Parser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        if tag == 'a' and 'href' in attrs:
            self.links.append(attrs['href'])


def main():
    errors = []
    jsons = sorted((BASE/'records').glob('*.json')) + [BASE/'source-registry.json']
    for path in jsons:
        json.loads(path.read_text())
    for name in ['threshold_real_screen_review.json', 'source_gram_circuit_review.json',
                 'gh_concurrent_update_review.json', 'card_i_ideas_independent_review.json',
                 'root_card_i_static_review.json']:
        if not (BASE/'records'/name).exists():
            errors.append('missing final review '+name)
    parser = Parser()
    parser.feed((BASE/'report.html').read_text())
    if len(parser.ids) != len(set(parser.ids)):
        errors.append('duplicate HTML ids')
    for href in parser.links:
        if not href.startswith(('https://', 'http://', '#')) and href != 'artifact-qa.json' and not (BASE/href).exists():
            errors.append('missing report link '+href)
    snapshot = json.loads((BASE/'records/gh_final_scope_snapshot.json').read_text())
    live_drift = []
    for item in snapshot['files']:
        if sha(BASE/item['snapshot']) != item['sha256']:
            errors.append('reviewed snapshot changed '+item['snapshot'])
        if sha(Path(item['original_path'])) != item['sha256']:
            live_drift.append(item['original_path'])
    # The old six-file drift remains historical evidence; never rewrite its hashes.
    update = json.loads((BASE/'records/gh_concurrent_update_review.json').read_text())
    if sha(BASE/'records/gh_rtl_audit.json') != update['old_record_sha256']:
        errors.append('original independent audit was modified')
    screen = json.loads((BASE/'records/threshold_packet_sample0_screen.json').read_text())
    workbook = json.loads((BASE/'records/opportunity_workbook_qa.json').read_text())
    if workbook['status'] != 'PASS_EXPORTED_VALUES_AND_RATIO_CHECKS':
        errors.append('workbook export did not pass')
    if workbook['workbook_sha256'] != sha(BASE/'opportunity-data.xlsx'):
        errors.append('workbook changed after read-back validation')
    if workbook['source_sha256'] != sha(BASE/'records/threshold_packet_sample0_screen.json'):
        errors.append('workbook source changed after export')
    for key, filename in [('script_sha256', 'scripts/screen_threshold_packets.py'),
                          ('reader_sha256', 'scripts/checkpoint_numpy.py'),
                          ('plan_sha256', 'records/threshold_packet_screen_plan.json')]:
        if sha(BASE/filename) != screen[key]:
            errors.append('screen provenance '+key)
    assert len(screen['layers']) == 2
    assert sum(x['reference_binary_count'] for x in screen['layers']) == 82944000
    assert all(x['float64_proxy_vs_archived_FC2_source_mismatches'] == 0 for x in screen['layers'])
    assert all(len(x['rows']) == 12 for x in screen['layers'])
    result = subprocess.run(['node', str(BASE/'scripts/qa_interaction.js')], capture_output=True, text=True, check=True)
    interaction = json.loads(result.stdout)
    status = subprocess.run(['git', 'status', '--short'], cwd=REPO, capture_output=True, text=True, check=True).stdout
    expected = ' M SDformer/hw_autoresearch_nts07/reviews/tcasii_accelerator_story_and_next_ideas_20260905.md\n'
    if status != expected:
        errors.append('main repository status differs from the starting dirty file')
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    protected = REPO/'SDformer/hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md'
    if sha(protected) != 'dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4':
        errors.append('protected docs359 mismatch')
    artifacts = {str(p.relative_to(BASE)): sha(p) for p in sorted(BASE.rglob('*'))
                 if p.is_file() and '__pycache__' not in p.parts and p.name != 'artifact-qa.json'}
    qa = {'status': 'PASS_STRUCTURAL_AND_INTERACTION_QA_VISUAL_UNVERIFIED' if not errors else 'FAIL',
          'errors': errors, 'json_files_parsed': len(jsons),
          'local_links': 'All report-relative links exist; the QA self-link is generated here.',
          'review_scope_snapshot_files_verified': len(snapshot['files']),
          'current_external_mirror_drift_after_snapshot': live_drift,
          'version_policy': 'Report is explicitly scoped to preserved read content. External subsequent sync is not silently included. Initial six-file drift was reviewed and preserved in a historical failure receipt.',
          'screen_provenance': 'Script/reader/predeclared-plan hashes match the saved result.',
          'screen_boundary': 'Two layers,12 configurations each;82,944,000 output bits matched. Not frozen FP proof.',
          'interaction': interaction,
          'visual_qa': {'status': 'NOT_VERIFIED',
                        'attempt': 'Optional headless Chromium133 exited133 after sandbox socket permission error; no screenshot produced.',
                        'boundary': 'No browser visual pass is claimed.'},
          'main_repo_HEAD': head, 'main_repo_status': status,
          'docs359_sha256': sha(protected), 'artifacts': artifacts,
          'PPA_ADMISSION': 0, 'RTL_SPEEDUP_ADMISSION': 0, 'acceptance_claim': False}
    (BASE/'artifact-qa.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key: qa[key] for key in ['status', 'errors', 'json_files_parsed',
                                              'review_scope_snapshot_files_verified', 'interaction',
                                              'current_external_mirror_drift_after_snapshot']}, ensure_ascii=False, indent=2))
    assert not errors


if __name__ == '__main__':
    main()
