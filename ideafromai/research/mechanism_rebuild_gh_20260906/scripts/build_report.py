"""Build a local, self-contained research report and evidence registry."""
from pathlib import Path
import hashlib
import html
import json
import re
import markdown

BASE = Path(__file__).resolve().parents[1]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registry():
    by_url = {}
    for path in sorted((BASE/'records').glob('*.json')):
        data = json.loads(path.read_text())

        def walk(value, loc):
            if isinstance(value, dict):
                for key, val in value.items():
                    if key in ('url', 'read_url', 'read_source', 'published_url', 'full_text_read') and isinstance(val, str) and val.startswith('https://'):
                        entry = by_url.setdefault(val, {'url': val, 'records': []})
                        entry['records'].append({'file': str(path.relative_to(BASE)), 'json_location': loc,
                                                 'title': value.get('title', value.get('name', value.get('id'))),
                                                 'access': value.get('access', 'See the original record; access level is not upgraded here.'),
                                                 'role': key})
                    walk(val, loc+'/'+key)
            elif isinstance(value, list):
                for i, val in enumerate(value):
                    walk(val, loc+'/'+str(i))
        walk(data, '')
    raw = {'status': 'AUDITABLE_SOURCE_POINTERS_NOT_ALL_FULLTEXT',
           'date': '2026-09-06', 'sources': list(by_url.values()),
           'limits': 'Source mirrors, indexed sections, abstracts, unresolved full text and root-relayed readings remain explicitly distinguished in the underlying records. URL count is not paper full-read count.'}
    (BASE/'source-registry.json').write_text(json.dumps(raw, ensure_ascii=False, indent=2)+'\n')


def main():
    registry()
    text = (BASE/'report-source.md').read_text().replace('本輪', '本轮')
    (BASE/'report-source.md').write_text(text)
    body = markdown.markdown(text, extensions=['tables', 'fenced_code'])
    data = json.loads((BASE/'records/threshold_packet_sample0_screen.json').read_text())
    embedded = json.dumps(data['layers'], ensure_ascii=False).replace('</', '<\\/')
    css = '''
:root{--ink:#172b35;--muted:#576872;--paper:#f5f3ed;--panel:#fff;--line:#d9dfde;--green:#086c60;--blue:#236699;--orange:#a75522}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.85 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}
header{background:#112d36;color:white;padding:42px max(24px,calc((100vw - 1120px)/2));border-bottom:6px solid #1d9c88}
header p{max-width:820px;margin:8px 0;color:#c9dbdd}header h1{font-size:32px;line-height:1.35;margin:14px 0}.eyebrow{letter-spacing:.13em;font-size:12px}.tag{display:inline-block;border:1px solid #678489;border-radius:4px;padding:2px 9px;margin:3px 8px 3px 0;font-size:12px}
main{max-width:1168px;margin:28px auto;padding:0 24px 60px}a{color:var(--blue);text-underline-offset:3px}header a{color:#9fe7d5}
.panel,article{background:var(--panel);border:1px solid var(--line);padding:26px;margin-bottom:24px;border-radius:8px}.panel h2{font-size:23px;margin:0 0 8px}.muted{color:var(--muted);font-size:14px}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}.card{border-top:4px solid var(--green);background:white;padding:20px;border-radius:5px}.card:nth-child(2){border-color:var(--blue)}.card:nth-child(3){border-color:var(--orange)}.card h2{font-size:20px;line-height:1.5;margin:4px 0}.card p{font-size:14px;margin:8px 0}.score{font-weight:700;font-size:21px}
.diagram{overflow-x:auto}.diagram svg{display:block;min-width:950px;width:100%;height:auto}.diagram text{font-family:inherit;fill:var(--ink)}
.controls{display:flex;gap:18px;flex-wrap:wrap;padding:14px 0}label{font-size:14px}select{display:block;font:inherit;padding:6px 14px;border:1px solid #9cadaa;border-radius:4px;background:white;color:var(--ink);min-width:130px}
.bars{display:grid;gap:13px;margin:20px 0}.barline{display:grid;grid-template-columns:190px 1fr 85px;gap:12px;align-items:center;font-size:14px}.track{background:#e8eeee;height:20px;border-radius:3px;overflow:hidden}.fill{height:100%;min-width:1px;background:var(--green);transition:width .2s}.barline:nth-child(2) .fill{background:var(--blue)}.barline:nth-child(3) .fill{background:var(--orange)}.value{text-align:right;font-variant-numeric:tabular-nums}.facts{display:grid;grid-template-columns:1fr 1fr;gap:10px}.fact{padding:13px;background:#f0f5f4}.fact strong{display:block;font-size:22px}
table{width:100%;border-collapse:collapse;font-size:14px;display:block;overflow-x:auto;margin:20px 0}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}th{background:#edf2f1;white-space:nowrap}tr:nth-child(even){background:#fafbf9}article h1{font-size:27px;line-height:1.5}article p{margin:16px 0}article p>strong:only-child{display:block;margin:35px 0 8px;font-size:23px}code{font-size:.92em;background:#edf1ef;border-radius:3px;padding:2px 5px;overflow-wrap:anywhere}pre{background:#edf1ef;padding:18px;overflow-x:auto;font-size:14px;line-height:1.7}pre code{padding:0}li{margin:9px 0}footer{font-size:13px;color:var(--muted);padding:18px 0}button:focus,select:focus,a:focus{outline:3px solid #59bca9;outline-offset:3px}
@media(max-width:720px){header{padding:28px 22px}header h1{font-size:27px}main{padding:0 13px 40px}.cards{grid-template-columns:1fr}.panel,article{padding:19px}.barline{grid-template-columns:135px 1fr 72px;gap:7px;font-size:12px}.facts{grid-template-columns:1fr}article h1{font-size:23px}}
@media print{body{background:white}header{background:white;color:black;border-bottom:2px solid black;padding:15px}header p,header a{color:black}main{max-width:none;padding:0}.controls{display:none}.panel,article{border:0}.card{break-inside:avoid}a{color:inherit}.diagram svg{min-width:0}table{font-size:11px}.fill{print-color-adjust:exact}}
'''
    diagram = '''<div class="diagram"><svg viewBox="0 0 1080 300" role="img" aria-labelledby="flow-title flow-desc"><title id="flow-title">晚知阈值的二值提交因果图</title><desc id="flow-desc">FC1首遍同时求统计和计算U。保存候选位、区间及K个数值，最终统计到齐后确认。证书失败时读取原二值源重放完整T，所有输出都经确认后提交。</desc><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#52716f"/></marker></defs>
<g fill="#eff6f3" stroke="#80a9a2" stroke-width="1.5"><rect x="15" y="130" width="170" height="82" rx="7"/><rect x="245" y="130" width="182" height="82" rx="7"/><rect x="486" y="130" width="184" height="82" rx="7"/><rect x="730" y="130" width="154" height="82" rx="7"/><rect x="950" y="130" width="110" height="82" rx="7"/><rect x="295" y="16" width="430" height=" sixty" rx="7" style="height:65px"/><rect x="686" y="247" width="236" height="42" rx="7" fill="#fff1e5"/></g>
<g font-size="19" text-anchor="middle"><text x="100" y="164">完整首遍 FC1</text><text x="100" y="191" font-size="14">保留二值源供回放</text><text x="336" y="164">完整 T → U</text><text x="336" y="191" font-size="14">包内预测阈值固定</text><text x="578" y="164">候选位＋区间</text><text x="578" y="191" font-size="14">仅留 K 个边界数值</text><text x="807" y="164">最终阈值判定</text><text x="807" y="191" font-size="14">a &lt; τ ≤ b</text><text x="1005" y="164">修补后</text><text x="1005" y="191">提交</text><text x="510" y="44">同时累加当前域统计</text><text x="510" y="68" font-size="14">全域封存后生成最终阈值</text><text x="804" y="274" font-size="16">失败：源回放完整 T，再提交</text></g>
<g fill="none" stroke="#52716f" stroke-width="2" marker-end="url(#arrow)"><path d="M185 171H241"/><path d="M427 171H482"/><path d="M670 171H726"/><path d="M884 171H946"/><path d="M100 130V49H290"/><path d="M725 49H807V126"/><path d="M807 212V243"/><path d="M922 269H1005V216"/></g><text x="913" y="157" font-size="13" fill="#086c60">通过</text></svg></div>'''
    # Use a numeric SVG attribute, not a CSS workaround.
    diagram = diagram.replace('height=" sixty" rx="7" style="height:65px"', 'height="65" rx="7"')
    js = '''
const rows=JSON.parse(document.getElementById('snapshot').textContent);
const fmt=(x,d=4)=>Number(x).toLocaleString('zh-CN',{minimumFractionDigits:d,maximumFractionDigits:d});
function update(){
 const l=rows[Number(document.getElementById('layer').value)],B=Number(document.getElementById('block').value),K=Number(document.getElementById('keep').value);
 const r=l.rows.find(x=>x.B===B&&x.K===K);
 const metrics=[r.packet_failure_fraction,r.replay_FC1_fraction_by_h,r.replay_FC1_fraction_fixed96];
 metrics.forEach((v,i)=>{document.getElementById('bar'+i).style.width=(100*v)+'%';document.getElementById('value'+i).textContent=fmt(100*v)+'%';});
 document.getElementById('packets').textContent=fmt(r.bare_payload_bytes_by_assumed_U_width['32']/1e6,3)+' MB';
 document.getElementById('wide').textContent=fmt(l.N_includes_T*l.H*4/1e6,3)+' MB';
 document.getElementById('source-read').textContent=fmt(r.replay_PSN_terms_by_h*l.C/(8*l.T)/1e6,3)+' MB';
 document.getElementById('changes').textContent=r.certified_patch_bits.toLocaleString('zh-CN')+' 位';
 document.getElementById('identity').textContent='sample0 · '+l.module.split('layers.')[1]+' · N='+l.N_includes_T+'（含 T=10）· C='+l.C+' · H='+l.H;
}
['layer','block','keep'].forEach(id=>document.getElementById(id).addEventListener('change',update));update();
'''
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C1/C2 新机制研究 · 2026-09-06</title><style>{css}</style></head><body>
<header><div class="eyebrow">CIRCUIT RESEARCH / TCAS-II / 2026-09-06</div><h1>把二值网络的真实依赖，变成可检验的电路问题</h1><p>本轮精读 G/H、审阅新增 RTL、交叉独立评审，并用同 SHA 的 ep34 参数完成一次预声明 CPU 筛查。</p><div><span class="tag">研究证据</span><span class="tag">无 RTL 倍率</span><span class="tag">无 PPA 准入</span></div><p><a href="report-source.md">完整 Markdown</a> · <a href="source-registry.json">来源登记</a> · <a href="artifact-qa.json">交付 QA</a></p></header>
<main><div class="cards"><section class="card"><div class="muted">主攻候选</div><h2>晚知阈值下的二值提交</h2><p>丢弃多数宽值，保存候选位、区间和有限边界数值；最终统计到齐后证明、修补或回放。</p><div class="score">独立 T：5–6 / 10</div></section><section class="card"><div class="muted">浅层竞争路线</div><h2>二值源统计驱动前馈</h2><p>先构造源 Gram 与当前域统计，再产生宽位输出。在线收缩和单口存储是主要费用。</p><div class="score">独立 T：5 / 10</div></section><section class="card"><div class="muted">独立组件备选</div><h2>二维见证的运动搜索</h2><p>投影作上界，真实二维轨迹作见证。投影饱和、完整历史与先验重合仍未解决。</p><div class="score">独立 T：5 / 10</div></section></div>
<section class="panel"><h2>主攻机制的因果链</h2><p class="muted">所有输出经最终确认后提交。首遍 FC1、PSN、统计、编码与失败回放全部收费。</p>{diagram}</section>
<section class="panel"><h2>真实输入：小失败率会变成多少补算</h2><p>两个预设层共 82,944,000 个输出位与对应封存二值源零差异。下图显示算项机会与回放放大，不能换算成周期、能量或全网精度。</p>
<div class="controls"><label>层<select id="layer"><option value="0">stage0 / block0</option><option value="1">stage3 / block0</option></select></label><label>空间包 B<select id="block"><option>16</option><option selected>32</option><option>64</option></select></label><label>保留宽值 K<select id="keep"><option>0</option><option>2</option><option selected>4</option><option>8</option></select></label></div>
<p id="identity" class="muted"></p><div class="bars"><div class="barline"><span>证书失败包</span><div class="track"><div class="fill" id="bar0"></div></div><span class="value" id="value0"></span></div><div class="barline"><span>按 h 回放 FC1 加项</span><div class="track"><div class="fill" id="bar1"></div></div><span class="value" id="value1"></span></div><div class="barline"><span>固定 96 组回放加项</span><div class="track"><div class="fill" id="bar2"></div></div><span class="value" id="value2"></span></div></div>
<div class="facts"><div class="fact"><span>裸编码载荷（假设 U 为 32 位）</span><strong id="packets"></strong></div><div class="fact"><span>完整 U 的 32 位载荷</span><strong id="wide"></strong></div><div class="fact"><span>逐 h 独立重读源的逻辑字节</span><strong id="source-read"></strong></div><div class="fact"><span>证书通过包内实际修补的位</span><strong id="changes"></strong></div></div>
<p class="muted">容量不含源、局部 Y/U、描述符、队列、宏对齐和端口。源重读是未跨 h 共享的模型，缓存可降低，同时增加权重重读或调度费用。CPU 为 float64 数学重建；32 位容量假设尚未成为等价部署合同。</p><noscript><p>交互需要 JavaScript；完整 24 行结果见 <a href="records/threshold_packet_sample0_screen.json">数据文件</a>，默认配置也列于下方报告。</p></noscript></section>
<article>{body}</article><footer>评分表示当前研究潜力，不能承诺录用。所有候选 PPA_ADMISSION=0、RTL_SPEEDUP_ADMISSION=0。来源与未闭问题均保留在本地记录。</footer></main><script type="application/json" id="snapshot">{embedded}</script><script>{js}</script></body></html>'''
    (BASE/'report.html').write_text(page)
    (BASE/'scripts/report-interaction.js').write_text(js)
    print(json.dumps({'report_bytes': (BASE/'report.html').stat().st_size,
                      'source_bytes': (BASE/'report-source.md').stat().st_size,
                      'report_sha256': fingerprint(BASE/'report.html')}, indent=2))


if __name__ == '__main__':
    main()
