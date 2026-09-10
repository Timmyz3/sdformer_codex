"""Render local research artifacts in an isolated browser and capture bounded QA."""
from pathlib import Path
import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.request
import websocket

BASE = Path(__file__).resolve().parent
profile = Path(tempfile.mkdtemp(prefix='tcasii_same_workload_qa_'))
pdf = BASE/'net_benefit.pdf'
assert not pdf.exists()
cmd = ['/usr/bin/chromium-browser','--headless','--disable-gpu',
       '--disable-background-networking','--disable-extensions','--disable-sync',
       '--password-store=basic','--use-mock-keychain','--no-first-run',
       '--no-default-browser-check','--hide-scrollbars','--remote-debugging-address=127.0.0.1',
       '--remote-debugging-port=0','--user-data-dir='+str(profile),'about:blank']
env = dict(os.environ, XDG_DATA_HOME=str(profile/'xdg'))
proc = subprocess.Popen(cmd,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
try:
    active = profile/'DevToolsActivePort'
    for _ in range(200):
        if active.exists():
            break
        assert proc.poll() is None, 'Isolated browser exited before startup'
        time.sleep(.05)
    port = active.read_text().splitlines()[0]
    pages = json.loads(urllib.request.urlopen('http://127.0.0.1:'+port+'/json').read())
    ws = websocket.create_connection(next(p['webSocketDebuggerUrl'] for p in pages if p['type']=='page'),suppress_origin=True,timeout=20)
    serial = 0
    def call(method,params=None):
        global serial
        serial += 1
        ws.send(json.dumps({'id':serial,'method':method,'params':params or {}}))
        while True:
            result = json.loads(ws.recv())
            if result.get('id') == serial:
                assert 'error' not in result, result
                return result.get('result',{})
    def navigate(path):
        url = path.as_uri()
        call('Page.navigate',{'url':url})
        for _ in range(100):
            ready = call('Runtime.evaluate',{'expression':'document.readyState === "complete" && location.href === '+json.dumps(url),'returnByValue':True})
            if ready['result'].get('value'):
                break
            time.sleep(.05)
        else:
            raise RuntimeError('Page did not finish loading')
        call('Runtime.evaluate',{'expression':'document.fonts.ready.then(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))))','awaitPromise':True})
    call('Page.enable')
    call('Emulation.setDeviceMetricsOverride',{'width':1280,'height':1800,'deviceScaleFactor':1,'mobile':False})
    navigate(BASE/'report.html')
    views = []
    for anchor in ['opening','c1','c2','ideas']:
        y = call('Runtime.evaluate',{'expression':'document.getElementById('+json.dumps(anchor)+').getBoundingClientRect().top+window.scrollY','returnByValue':True})['result']['value']
        call('Runtime.evaluate',{'expression':f'window.scrollTo(0,{y}); new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))','awaitPromise':True})
        snap = call('Page.captureScreenshot',{'format':'png','fromSurface':True})
        target = profile/(anchor+'.png')
        target.write_bytes(base64.b64decode(snap['data']))
        views.append({'anchor':anchor,'path':str(target),'y':y,'bytes':target.stat().st_size})
    geometry = call('Runtime.evaluate',{'expression':'({scrollWidth:document.documentElement.scrollWidth,clientWidth:document.documentElement.clientWidth,scrollHeight:document.documentElement.scrollHeight})','returnByValue':True})['result']['value']
    assert geometry['scrollWidth'] == geometry['clientWidth'], geometry
    navigate(BASE/'net_benefit.html')
    call('Emulation.setDeviceMetricsOverride',{'width':1280,'height':1000,'deviceScaleFactor':1,'mobile':False})
    snap = call('Page.captureScreenshot',{'format':'png','fromSurface':True})
    target = profile/'one_page.png';target.write_bytes(base64.b64decode(snap['data']))
    views.append({'anchor':'one_page','path':str(target),'bytes':target.stat().st_size})
    export = call('Page.printToPDF',{'preferCSSPageSize':True,'printBackground':True,'displayHeaderFooter':False})
    pdf.write_bytes(base64.b64decode(export['data']))
    receipt = {'status':'RENDERED_PENDING_VISUAL_INSPECTION','views':views,'report_geometry':geometry,'pdf':str(pdf)}
    (BASE/'render_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    ws.close()
finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill();proc.wait()
