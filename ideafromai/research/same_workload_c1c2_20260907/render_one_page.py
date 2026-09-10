from pathlib import Path
import markdown

BASE = Path(__file__).resolve().parent
target = BASE/'net_benefit.html'
assert not target.exists()
css = '''
@page {size:A4 landscape;margin:10mm}
*{box-sizing:border-box}
body{font-family:'Noto Sans CJK SC','WenQuanYi Micro Hei',sans-serif;color:#183345;background:#eef3f6;margin:0;font-size:12px;line-height:1.5}
main{max-width:1180px;margin:25px auto;background:white;padding:28px 35px;border-top:5px solid #176485}
h1{font-size:24px;line-height:1.3;margin:0 0 12px}p{margin:10px 0}a{color:#076b90;text-decoration:none;overflow-wrap:anywhere}
table{width:100%;border-collapse:collapse;font-size:11.5px;line-height:1.45;margin:12px 0}
th,td{padding:7px 8px;border-bottom:1px solid #d4e0e7;text-align:left;vertical-align:top}th{background:#e9f1f5}tr:nth-child(even)td{background:#f7fafc}
td:nth-child(3){white-space:nowrap;font-variant-numeric:tabular-nums}strong{font-weight:700}code{font-size:11px}
@media print{body{background:white;font-size:9.3pt;line-height:1.4}main{padding:0;margin:0;border-top:3px solid #176485;max-width:none;padding-top:4mm}h1{font-size:17pt;margin-bottom:3mm}p{margin:2.5mm 0}table{font-size:8.7pt;line-height:1.35;margin:3mm 0}th,td{padding:2mm}a{color:inherit}}
'''
body = markdown.markdown((BASE/'net_benefit.md').read_text(),extensions=['tables'])
target.write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>C1 / C2 同负载净收益表</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>\n')
print(target)
