"""Start this user's isolated, visible browser for manual ChatGPT login.

Browser data and VNC authentication stay in ~/.local/state, outside Git.
The browser, VNC and noVNC listeners are loopback-only. Use an SSH tunnel.
"""
from pathlib import Path
import json
import os
import secrets
import socket
import string
import subprocess
import time

STATE = Path.home() / '.local/state/codex-pro-browser'
SHARE = Path.home() / '.local/share/codex-pro-browser'

def launch(name, args, env):
    with (STATE / (name + '.log')).open('ab') as out:
        p = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL,
                             stdout=out, stderr=out, start_new_session=True)
    return p.pid

def wait_port(port):
    for _ in range(50):
        with socket.socket() as s:
            s.settimeout(.1)
            if s.connect_ex(('127.0.0.1', port)) == 0:
                return
        time.sleep(.1)
    raise RuntimeError(f'Port {port} did not start; see {STATE}/*.log')

def main():
    os.umask(0o077)
    STATE.mkdir(parents=True, exist_ok=True)
    STATE.chmod(0o700)
    if (STATE / 'processes.json').exists():
        pids = json.loads((STATE / 'processes.json').read_text())
        if any(Path('/proc', str(pid)).exists() for pid in pids.values()):
            print('Browser service already running; see', STATE / 'processes.json')
            return
    password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(8))
    (STATE / 'viewer-password.txt').write_text(password + '\n')
    encoded = subprocess.run(['/usr/bin/vncpasswd', '-f'], input=(password + '\n').encode(),
                             check=True, capture_output=True).stdout
    (STATE / 'vnc-password').write_bytes(encoded)
    authority = STATE / 'Xauthority'
    authority.touch(mode=0o600)
    subprocess.run(['/usr/bin/xauth', '-f', str(authority)],
                   input=f'add :88 . {secrets.token_hex(16)}\n'.encode(), check=True,
                   stdout=subprocess.DEVNULL)
    env = os.environ.copy()
    env.update(DISPLAY=':88', XAUTHORITY=str(authority))
    pids = {}
    pids['xvnc'] = launch('xvnc', ['/usr/bin/Xvnc', ':88', '-geometry', '1440x1000',
        '-depth', '24', '-auth', str(authority), '-rfbport', '5988', '-localhost',
        '-SecurityTypes', 'VncAuth', '-PasswordFile', str(STATE / 'vnc-password'),
        '-desktop', 'Codex Pro research browser', '-nolisten', 'tcp', '-noreset'], env)
    (STATE / 'processes.json').write_text(json.dumps(pids, indent=2) + '\n')
    wait_port(5988)
    pids['chromium'] = launch('chromium', ['/usr/bin/chromium-browser',
        '--user-data-dir=' + str(STATE / 'chromium-profile'), '--no-first-run',
        '--no-default-browser-check', '--disable-gpu', '--window-size=1440,1000',
        '--window-position=0,0', '--remote-debugging-address=127.0.0.1',
        '--remote-debugging-port=9222', '--proxy-server=http://127.0.0.1:7897',
        'https://chatgpt.com/'], env)
    web_env = env.copy()
    web_env['PYTHONPATH'] = str(SHARE / 'python')
    pids['novnc'] = launch('novnc', ['/opt/anaconda3/bin/python3.12', '-m', 'websockify',
        '--web', str(SHARE / 'noVNC'), '127.0.0.1:6088', '127.0.0.1:5988'], web_env)
    (STATE / 'processes.json').write_text(json.dumps(pids, indent=2) + '\n')
    wait_port(6088)
    wait_port(9222)
    print('Ready: http://127.0.0.1:6088/vnc.html (forward port 6088 through SSH)')
    print('VNC viewer password file:', STATE / 'viewer-password.txt')
    print('Login must be completed by the user in this browser.')

if __name__ == '__main__':
    main()
