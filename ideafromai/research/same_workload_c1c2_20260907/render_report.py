"""Produce one self-contained HTML research report from the canonical source."""
from pathlib import Path
import markdown

BASE = Path(__file__).resolve().parent
CSS = """
:root{color-scheme:light;--ink:#142d3d;--muted:#586775;--line:#d9e2e8;--blue:#165a7b;--soft:#f1f6f9}
*{box-sizing:border-box}body{margin:0;background:#edf2f5;color:var(--ink);font-family:'Noto Sans CJK SC','WenQuanYi Micro Hei','Microsoft YaHei',sans-serif;line-height:1.82;font-size:16px}
main{max-width:1060px;margin:32px auto;padding:52px 62px 44px;background:white;border-top:7px solid var(--blue);box-shadow:0 3px 20px #1232}
h1{font-size:32px;line-height:1.45;letter-spacing:.3px;margin:0 0 12px}h2{font-size:24px;line-height:1.5;margin:44px 0 18px;border-bottom:2px solid var(--line);padding-bottom:10px}h3{font-size:19px;margin:28px 0 12px}p{margin:14px 0}a{color:#086b95;text-underline-offset:3px;overflow-wrap:anywhere}a[id]{scroll-margin-top:24px}.subtitle{color:var(--muted);font-size:14px;margin-bottom:30px}.briefbox,.equation{background:var(--soft);border-left:4px solid #3e849c;padding:16px 21px;margin:23px 0}.equation{font-family:'Noto Sans CJK SC',sans-serif;font-size:18px;text-align:center;line-height:2.1}.flow{display:flex;align-items:stretch;gap:10px;margin:25px 0;padding:16px 0}.flow div{flex:1;background:#eaf4f0;border:1px solid #bfd6cb;border-radius:6px;padding:14px 8px;text-align:center;font-weight:600;line-height:1.7}.flow span{align-self:center;font-size:22px;color:#406879}.flow small{font-size:12px;font-weight:400;color:#52665e}table{border-collapse:collapse;width:100%;font-size:14px;margin:22px 0;line-height:1.7}th{background:#edf3f7;font-weight:650}td,th{padding:11px 13px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}td:not(:first-child){font-variant-numeric:tabular-nums}tr:nth-child(even) td{background:#fbfcfd}code{font-family:'DejaVu Sans Mono',monospace;font-size:13px;padding:2px 4px;background:#f2f4f6;overflow-wrap:anywhere}li{margin:8px 0}.footer-note{border-top:1px solid var(--line);padding-top:22px;margin-top:38px;color:var(--muted);font-size:13px}
@media(max-width:760px){main{margin:0;padding:26px 18px;box-shadow:none}h1{font-size:25px}h2{font-size:21px}.flow{gap:5px}.flow div{padding:10px 4px;font-size:13px}.flow small{font-size:10px}td,th{padding:8px 6px;font-size:12px}}
@media print{body{background:white}main{margin:0;padding:15mm 13mm;max-width:none;box-shadow:none}h2,h3{break-after:avoid}table,figure,.equation,.briefbox{break-inside:avoid}a{color:inherit}}
"""


def main():
    target = BASE/'report.html'
    assert not target.exists(), 'Render once; revise only to fix an observed defect.'
    source = (BASE/'report-source.md').read_text()
    appendix = (BASE/'five_ideas_reassessment.md').read_text()
    appendix = appendix.replace('## ', '### ').replace('# 五项思想复审', '## 五项思想复审', 1)
    content = markdown.markdown(source+'\n\n<a id="ideas"></a>\n\n'+appendix,extensions=['tables','fenced_code'])
    html = '<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>C1 / C2 同负载净收益与思想复审 · 2026-09-07</title><style>'+CSS+'</style></head><body><main>'+content+'</main></body></html>\n'
    target.write_text(html)
    print(str(target))


if __name__=='__main__':
    main()
