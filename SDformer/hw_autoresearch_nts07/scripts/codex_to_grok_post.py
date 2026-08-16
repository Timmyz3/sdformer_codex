#!/usr/bin/env python3
"""Post a Codex-agent message into the Grok inbound mailbox.

Grok polls this file. Do not grok --resume the live TUI session.
Do not use Codex multi_agent send_input against a Grok session id.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path("/root/private_data/work/sdformer_codex/SDformer/hw_autoresearch_nts07")
INBOX = ROOT / "docs" / "CODEX_TO_GROK_INBOX.md"
QUEUE = ROOT / "results" / "grok_codex_collab" / "from_codex"
STATE = ROOT / "results" / "grok_codex_collab" / "from_codex_state.json"
HEADER = "【来源：Codex agent，不是用户本人】"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", required=True)
    args = ap.parse_args()
    text = Path(args.text_file).read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit("empty message")
    if HEADER not in text.splitlines()[0]:
        text = HEADER + "\n" + text
    QUEUE.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    stamp = f"{ts}_{int(time.time())}"
    path = QUEUE / f"msg_{stamp}.md"
    body = text + "\n"
    path.write_text(body, encoding="utf-8")
    INBOX.write_text(
        "# Codex → Grok inbox\n\n"
        "这不是用户本人输入。来源永远是 **Codex agent**。\n\n"
        f"## latest `{path.name}`\n\n"
        + body,
        encoding="utf-8",
    )
    state = {"last_post_ts": time.time(), "last_file": str(path)}
    STATE.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({"ok": True, "file": str(path), "inbox": str(INBOX)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
