#!/usr/bin/env python3
"""在 sd5ai 上执行远程命令（非交互密码登录，纯标准库 pty 驱动）。

本工作站没有 sshpass / paramiko（且用户不希望装包），但 ssh 需要从 /dev/tty 读密码。
本脚本用 pty.fork 给 ssh 一个伪终端，检测到 "password:" 提示后写入密码。

密码来源（按优先级）：$SD5AI_PASS，否则 ~/.sd5ai_pass（应为 0600）。

用法：
  python tools/sd5ai_ssh.py 'hostname; nvidia-smi'
  python tools/sd5ai_ssh.py --put local.txt /root/remote.txt   # 上传单文件
  python tools/sd5ai_ssh.py --get /root/remote.txt local.txt   # 下载单文件
"""
from __future__ import annotations

import argparse
import os
import pty
import select
import signal
import sys
import time
from pathlib import Path

HOST = 'ssh.sd5ai.scnet.cn'
PORT = 10037
USER = 'root'
PASS_FILE = Path.home() / '.sd5ai_pass'
SSH_OPTS = ['-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=15',
            '-o', 'PreferredAuthentications=password', '-o', 'PubkeyAuthentication=no',
            '-o', 'NumberOfPasswordPrompts=1']


def get_pass() -> str:
    p = os.environ.get('SD5AI_PASS')
    if p:
        return p
    if not PASS_FILE.exists():
        sys.exit('no password: set $SD5AI_PASS or create %s (chmod 600)' % PASS_FILE)
    if PASS_FILE.stat().st_mode & 0o077:
        sys.exit('%s is group/other-readable; chmod 600 it' % PASS_FILE)
    return PASS_FILE.read_text().strip()


def talk(argv: list[str], password: str, timeout: float = 300.0) -> int:
    """跑 argv，遇到密码提示就喂密码；stdout 原样转发到本进程 stdout。"""
    pid, fd = pty.fork()
    if pid == 0:                                   # 子进程：把 pty 当控制终端
        os.execvp(argv[0], argv)
        os._exit(127)

    deadline = time.time() + timeout
    sent = False
    tail = b''            # 仅用于识别提示，避免重复匹配
    status = None
    try:
        while True:
            if time.time() > deadline:
                os.kill(pid, signal.SIGKILL)
                sys.stderr.write('\n[sd5ai_ssh] timeout after %.0fs\n' % timeout)
                status = 124
                break
            r, _, _ = select.select([fd], [], [], 1.0)
            if r:
                try:
                    data = os.read(fd, 65536)
                except OSError:
                    data = b''
                if not data:
                    break
                if not sent:
                    tail = (tail + data)[-256:]
                    if b'assword:' in tail:
                        os.write(fd, (password + '\n').encode())
                        sent = True
                        tail = b''
                sys.stdout.buffer.write(data.replace(b'\r\n', b'\n'))
                sys.stdout.buffer.flush()
            done, st = os.waitpid(pid, os.WNOHANG)
            if done:
                status = st
                # 把 pty 里剩下的输出读干净
                while True:
                    r, _, _ = select.select([fd], [], [], 0.2)
                    if not r:
                        break
                    try:
                        d = os.read(fd, 65536)
                    except OSError:
                        break
                    if not d:
                        break
                    sys.stdout.buffer.write(d.replace(b'\r\n', b'\n'))
                sys.stdout.buffer.flush()
                break
    finally:
        if status is None:
            try:
                _, status = os.waitpid(pid, 0)
            except ChildProcessError:
                status = 1
        os.close(fd)
    return os.waitstatus_to_exitcode(status) if isinstance(status, int) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--put', nargs=2, metavar=('LOCAL', 'REMOTE'))
    ap.add_argument('--get', nargs=2, metavar=('REMOTE', 'LOCAL'))
    ap.add_argument('--timeout', type=float, default=300.0)
    ap.add_argument('command', nargs='*')
    a = ap.parse_args()
    pw = get_pass()

    if a.put:
        local, remote = a.put
        argv = ['scp', '-P', str(PORT)] + SSH_OPTS + [local, '%s@%s:%s' % (USER, HOST, remote)]
    elif a.get:
        remote, local = a.get
        argv = ['scp', '-P', str(PORT)] + SSH_OPTS + ['%s@%s:%s' % (USER, HOST, remote), local]
    else:
        if not a.command:
            ap.error('need a remote command (or --put/--get)')
        argv = ['ssh', '-p', str(PORT)] + SSH_OPTS + ['%s@%s' % (USER, HOST),
                                                     ' '.join(a.command)]
    sys.exit(talk(argv, pw, a.timeout))


if __name__ == '__main__':
    main()
