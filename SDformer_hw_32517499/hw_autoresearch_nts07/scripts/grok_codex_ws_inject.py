#!/usr/bin/env python3
"""Inject a Grok-agent review into a live Codex app-server thread.

Uses the official app-server JSON-RPC over a WebSocket upgrade of
~/.codex/app-server-control/app-server-control.sock.

Does not take the thread writer lock and does not append raw jsonl.

Default: deliver into the Codex thread immediately so it is visible.
If a turn is active, use turn/steer; otherwise turn/start.
Text always tells Codex to finish current work before acting on the note.
Disk queue + --flush remain available; --queue-if-busy opts into silent enqueue.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path

DEFAULT_SOCK = "/root/.codex/app-server-control/app-server-control.sock"
DEFAULT_THREAD = "019f365d-6ed1-76b2-a993-6b652298d9d8"
DEFAULT_SESSION = (
    "/root/.codex/sessions/2026/07/06/"
    "rollout-2026-07-06T15-38-40-019f365d-6ed1-76b2-a993-6b652298d9d8.jsonl"
)
COLLAB = Path(
    "/root/private_data/work/sdformer_codex/SDformer/"
    "hw_autoresearch_nts07/results/grok_codex_collab"
)
STATE_PATH = COLLAB / "state.json"
OUTBOX = COLLAB / "outbox_to_codex"
SENT = COLLAB / "outbox_sent"
INBOX_MD = Path(
    "/root/private_data/work/sdformer_codex/SDformer/"
    "hw_autoresearch_nts07/docs/GROK_TO_CODEX_INBOX.md"
)

DEFER_BANNER = (
    "【排队/延后】请先继续并完成你当前正在做的工作，"
    "做完后再阅读下面的参考。不要中断手头的筛选、建模或改码。"
)
REF_BANNER = (
    "【仅供参考】这是 Grok agent 的独立意见，不是用户本人指令。"
    "请再独立思考一遍：是否合适、是否正确、是否与封存证据冲突。不要全盘接受。"
)
SOURCE_BANNER = "【来源：Grok agent，不是用户本人】"


class WsJsonRpc:
    def __init__(self, sock_path: str):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(8)
        self.sock.connect(sock_path)
        self._handshake()
        self._buf = bytearray()
        self.next_id = 1

    def _handshake(self) -> None:
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            "GET / HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "Origin: http://localhost\r\n"
            "\r\n"
        )
        self.sock.sendall(req.encode())
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError("socket closed during websocket handshake")
            data += chunk
        head, rest = data.split(b"\r\n\r\n", 1)
        if b"101" not in head.split(b"\r\n", 1)[0]:
            raise RuntimeError(f"websocket upgrade failed: {head[:200]!r}")
        expect = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
        )
        if expect not in head:
            # still accept if server sent 101; some stacks omit exact match checks here
            pass
        if rest:
            self._buf.extend(rest)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        mask = os.urandom(4)
        header = bytearray()
        header.append(0x80 | (opcode & 0x0F))
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", n))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", n))
        header.extend(mask)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(bytes(header) + masked)

    def send_json(self, obj: dict) -> None:
        self._send_frame(0x1, json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode())

    def _recv_exact(self, n: int, timeout: float) -> bytes:
        end = time.time() + timeout
        while len(self._buf) < n:
            remain = end - time.time()
            if remain <= 0:
                raise TimeoutError("websocket recv timeout")
            self.sock.settimeout(remain)
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError("websocket closed")
            self._buf.extend(chunk)
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def recv_json(self, timeout: float = 8.0) -> dict | None:
        """Return next JSON-RPC object, answering ping automatically. None on close."""
        end = time.time() + timeout
        while True:
            remain = end - time.time()
            if remain <= 0:
                raise TimeoutError("websocket json timeout")
            b0 = self._recv_exact(1, remain)[0]
            b1 = self._recv_exact(1, remain)[0]
            opcode = b0 & 0x0F
            masked = bool(b1 & 0x80)
            ln = b1 & 0x7F
            if ln == 126:
                ln = struct.unpack("!H", self._recv_exact(2, remain))[0]
            elif ln == 127:
                ln = struct.unpack("!Q", self._recv_exact(8, remain))[0]
            mask = self._recv_exact(4, remain) if masked else b""
            payload = self._recv_exact(ln, remain)
            if masked:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
            if opcode == 0x9:
                self._send_frame(0xA, payload)
                continue
            if opcode == 0x8:
                return None
            if opcode == 0x1:
                return json.loads(payload.decode())
            # ignore binary / continuation

    def request(self, method: str, params: dict | None, timeout: float = 10.0) -> dict:
        rid = self.next_id
        self.next_id += 1
        msg = {"id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        self.send_json(msg)
        extras = []
        end = time.time() + timeout
        while True:
            remain = end - time.time()
            if remain <= 0:
                raise TimeoutError(f"no response for {method} id={rid}")
            obj = self.recv_json(remain)
            if obj is None:
                raise RuntimeError(f"closed waiting for {method}")
            if obj.get("id") == rid:
                return {"response": obj, "notifications": extras}
            extras.append(obj)

    def notify(self, method: str, params: dict | None = None) -> None:
        msg = {"method": method}
        if params is not None:
            msg["params"] = params
        self.send_json(msg)


def latest_turn_id(session_path: str) -> str | None:
    last = None
    path = Path(session_path)
    if not path.exists():
        return None
    with path.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pay = rec.get("payload") or {}
            meta = pay.get("internal_chat_message_metadata_passthrough") or {}
            tid = meta.get("turn_id")
            if tid:
                last = tid
    return last


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")


def prepare_text(text: str) -> str:
    text = text.strip()
    parts = []
    if SOURCE_BANNER not in text:
        parts.append(SOURCE_BANNER)
    if "排队/延后" not in text and DEFER_BANNER not in text:
        parts.append(DEFER_BANNER)
    if "仅供参考" not in text:
        parts.append(REF_BANNER)
    if parts:
        return "\n".join(parts) + "\n\n" + text
    return text


def status_is_busy(status) -> bool:
    if status is None:
        return False
    if isinstance(status, str):
        return status.lower() in {"active", "running", "in_progress"}
    if isinstance(status, dict):
        kind = str(status.get("type") or status.get("status") or "").lower()
        if kind in {"idle", "waitingonuserinput", "waiting_on_user_input", "complete", "completed"}:
            return False
        if kind in {"active", "running", "in_progress"}:
            return True
    return False


def enqueue(text: str) -> Path:
    OUTBOX.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = OUTBOX / f"queued_{ts}_{int(time.time())}.txt"
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    pending = sorted(OUTBOX.glob("queued_*.txt"))
    block = (
        "\n## 排队未送达（Codex 忙，不打断）\n\n"
        + "\n".join(f"- `{p.name}`" for p in pending)
        + "\n\n最新排队正文：\n\n"
        + text
        + "\n"
    )
    if INBOX_MD.exists():
        old = INBOX_MD.read_text(encoding="utf-8")
        marker = "\n## 排队未送达"
        old = old.split(marker, 1)[0].rstrip() + "\n"
        INBOX_MD.write_text(old + block, encoding="utf-8")
    else:
        INBOX_MD.write_text("# Grok → Codex inbox\n" + block, encoding="utf-8")
    state = load_state()
    state["last_queue_ts"] = time.time()
    state["queued"] = len(pending)
    save_state(state)
    return path


def gather_queue() -> tuple[str, list[Path]]:
    files = sorted(OUTBOX.glob("queued_*.txt"))
    if not files:
        return "", []
    bodies = []
    for i, path in enumerate(files, 1):
        bodies.append(f"—— 排队条目 {i}/{len(files)} `{path.name}` ——\n{path.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(bodies), files


def mark_sent(files: list[Path]) -> None:
    SENT.mkdir(parents=True, exist_ok=True)
    for path in files:
        dest = SENT / path.name
        dest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.unlink(missing_ok=True)


def connect_and_status(client: WsJsonRpc, thread_id: str):
    init = client.request(
        "initialize",
        {
            "clientInfo": {
                "name": "grok-agent",
                "title": "Grok agent collaborator (not the human user)",
                "version": "0.3.0",
            },
            "capabilities": {"experimentalApi": False},
        },
        timeout=10,
    )
    if "error" in init["response"]:
        return init, None
    client.notify("initialized")
    read = client.request(
        "thread/read",
        {"threadId": thread_id, "includeTurns": False},
        timeout=12,
    )
    thread = (read["response"].get("result") or {}).get("thread") or read["response"].get("result") or {}
    status = None
    if isinstance(thread, dict):
        status = thread.get("status") or (thread.get("thread") or {}).get("status")
    return init, status


def send_turn_start(client: WsJsonRpc, thread_id: str, text: str) -> dict:
    payload = {
        "threadId": thread_id,
        "input": [{"type": "text", "text": text}],
        "clientUserMessageId": f"grok-agent-{int(time.time())}",
    }
    return client.request("turn/start", payload, timeout=15)


def send_turn_steer(client: WsJsonRpc, thread_id: str, text: str, expected_turn: str) -> dict:
    payload = {
        "threadId": thread_id,
        "expectedTurnId": expected_turn,
        "input": [{"type": "text", "text": text}],
        "clientUserMessageId": f"grok-agent-{int(time.time())}",
    }
    return client.request("turn/steer", payload, timeout=15)


def deliver(
    client: WsJsonRpc,
    thread_id: str,
    text: str,
    status,
    session_path: str,
    steer_now: bool,
    queue_if_busy: bool,
) -> dict:
    expected = latest_turn_id(session_path)
    busy = status_is_busy(status)
    if busy and queue_if_busy and not steer_now:
        path = enqueue(text)
        return {
            "ok": True,
            "queued": True,
            "method": "queue",
            "file": str(path),
            "threadStatus": status,
            "expectedTurnId": expected,
        }
    if busy and expected:
        result = send_turn_steer(client, thread_id, text, expected)
        method = "turn/steer"
        if "error" in result["response"]:
            result = send_turn_start(client, thread_id, text)
            method = "turn/start-after-steer-fail"
    else:
        result = send_turn_start(client, thread_id, text)
        method = "turn/start"
    ok = "error" not in result["response"]
    out = {
        "ok": ok,
        "queued": False,
        "method": method,
        "expectedTurnId": expected,
        "threadStatus": status,
        "response": result["response"],
    }
    if ok:
        state = load_state()
        state.update(
            {
                "last_inject_ts": time.time(),
                "last_method": method,
                "last_turn": expected,
                "last_ok": True,
                "queued": len(list(OUTBOX.glob("queued_*.txt"))) if OUTBOX.exists() else 0,
                "count": int(state.get("count") or 0) + 1,
            }
        )
        save_state(state)
    return out


def inject(
    text: str,
    thread_id: str,
    sock_path: str,
    session_path: str,
    force: bool,
    min_interval_s: int,
    steer_now: bool,
    flush: bool,
    queue_if_busy: bool,
) -> dict:
    state = load_state()
    now = time.time()
    last = float(state.get("last_inject_ts") or 0)
    if not flush and not force and last and (now - last) < min_interval_s:
        return {
            "ok": False,
            "skipped": True,
            "reason": f"min interval {min_interval_s}s not elapsed",
            "seconds_since_last": now - last,
        }

    if flush:
        merged, files = gather_queue()
        if not merged:
            return {"ok": True, "skipped": True, "reason": "queue empty"}
        text = prepare_text(merged)
    else:
        text = prepare_text(text)

    client = WsJsonRpc(sock_path)
    try:
        init, status = connect_and_status(client, thread_id)
        if "error" in init["response"]:
            return {"ok": False, "stage": "initialize", "detail": init["response"]}
        if flush and status_is_busy(status):
            return {
                "ok": True,
                "queued": True,
                "method": "still-busy",
                "threadStatus": status,
                "reason": "thread still active; leave queue in place",
            }
        out = deliver(
            client,
            thread_id,
            text,
            status,
            session_path,
            steer_now=steer_now,
            queue_if_busy=queue_if_busy,
        )
        if flush and out.get("ok") and not out.get("queued"):
            mark_sent(files)
            out["flushed"] = True
        out["initUserAgent"] = (init["response"].get("result") or {}).get("userAgent")
        return out
    finally:
        client.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file")
    ap.add_argument("--flush", action="store_true", help="deliver disk-queued notes if Codex is idle")
    ap.add_argument("--queue-if-busy", action="store_true", help="silent disk queue instead of visible inject")
    ap.add_argument("--steer-now", action="store_true", help="force steer even with --queue-if-busy")
    ap.add_argument("--thread-id", default=DEFAULT_THREAD)
    ap.add_argument("--sock", default=DEFAULT_SOCK)
    ap.add_argument("--session", default=DEFAULT_SESSION)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-interval-s", type=int, default=3600)
    args = ap.parse_args()
    if not args.flush and not args.text_file:
        print("need --text-file or --flush", file=sys.stderr)
        return 2
    text = ""
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8").strip()
        if not text:
            print("empty review", file=sys.stderr)
            return 2
    out = inject(
        text,
        args.thread_id,
        args.sock,
        args.session,
        args.force,
        args.min_interval_s,
        args.steer_now,
        args.flush,
        args.queue_if_busy,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2)[:8000])
    return 0 if out.get("ok") or out.get("skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
