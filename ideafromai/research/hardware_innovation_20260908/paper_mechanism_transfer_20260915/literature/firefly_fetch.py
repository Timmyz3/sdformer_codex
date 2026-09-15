from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests,subprocess,json
H=Path(__file__).resolve().parent
items={'original':'https://arxiv.org/pdf/2301.01905v5','v2':'https://floyedshen.github.io/pdf/li2024fireflyv2.pdf','s':'https://arxiv.org/pdf/2408.15578v3','t':'https://arxiv.org/pdf/2505.12771'}
def fetch(kv):
 key,url=kv;r=requests.get(url,timeout=60);r.raise_for_status();assert r.content[:4]==b'%PDF',(key,r.headers)
 p=H/f'firefly_{key}.pdf';p.write_bytes(r.content)
 subprocess.run(['pdftotext','-layout',str(p),str(p.with_suffix('.txt'))],check=True)
 return dict(key=key,url=url,bytes=len(r.content))
with ThreadPoolExecutor(max_workers=4) as ex:results=list(ex.map(fetch,items.items()))
(H/'firefly_sources.json').write_text(json.dumps(results,indent=2)+'\n');print(results)
