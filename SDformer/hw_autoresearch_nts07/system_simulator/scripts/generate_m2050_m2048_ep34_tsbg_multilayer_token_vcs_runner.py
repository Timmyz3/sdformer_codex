#!/opt/anaconda3/bin/python
"""Derive the non-retry M2050 runner after the sealed M2049 LRU-model failure."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "dc_handoff/scripts/run_m2049_m2048_ep34_tsbg_multilayer_token_vcs_one_shot.sh"
OUTPUT = HW / "dc_handoff/scripts/run_m2050_m2048_ep34_tsbg_multilayer_token_vcs_one_shot.sh"
SOURCE_SHA = "2920427afb5bda133665d36474f8dfc91159b20067b39076554fa94b7650eb4b"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"runner anchor cardinality drift: {old!r}")
    return text.replace(old, new)


def main() -> int:
    if sha256(SOURCE) != SOURCE_SHA:
        raise RuntimeError("M2049 runner SHA drift")
    text = SOURCE.read_text(encoding="utf-8")
    text = text.replace("m2049_m2048", "m2050_m2048")
    text = text.replace("M2049", "M2050")
    text = replace_once(
        text,
        "b45c2c34153bb9152498a16d0c9102db8f7a3defa6d7a44c730df2f3896cba34",
        "1858c94b1fc411e691152f848f8dd3a5b5955001828236cba902d04b9639014b",
    )
    text = replace_once(
        text,
        "92da4ab451b2859c39097717e2a0b87db27a460ff56db959c0b305b5dd0adfb5",
        "940f243fa218a6154887f129e622e7a219f96f97b25744b93a68d2ef60532900",
    )
    text = replace_once(
        text,
        "50c2ad913492a085746e89583b086b9a24178feb5e28e8babe838d2752f787ad",
        "bd951471b272d4ddcb6dfa61904003e94a59576265e17b06ff687bf84052886b",
    )
    OUTPUT.write_text(text, encoding="utf-8")
    OUTPUT.chmod(0o775)
    print(f"PASS output={OUTPUT} sha256={sha256(OUTPUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
