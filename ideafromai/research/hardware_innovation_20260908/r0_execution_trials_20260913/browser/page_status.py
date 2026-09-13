"""Read only the visible page of the isolated research browser through CDP."""
import json
import urllib.request
import websocket

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
pages = json.load(opener.open('http://127.0.0.1:9222/json', timeout=5))
for page in pages:
    if page.get('type') != 'page' or not page.get('url', '').startswith('https://chatgpt.com/'):
        continue
    ws = websocket.create_connection(page['webSocketDebuggerUrl'], suppress_origin=True, timeout=10)
    ws.send(json.dumps({'id': 1, 'method': 'Runtime.evaluate', 'params': {
        'expression': 'JSON.stringify({title:document.title,url:location.href,text:document.body?.innerText.slice(0,2000)})',
        'returnByValue': True}}))
    while True:
        message = json.loads(ws.recv())
        if message.get('id') == 1:
            print(message['result']['result'].get('value', 'No visible page text'))
            break
    ws.close()
