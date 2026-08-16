#!/usr/bin/env python3
"""统计Fixed-TTB与RQTB层次的VCD位翻转活动代理。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def known_binary(value: str) -> bool:
    return value and all(bit in "01" for bit in value)


def normalize(value: str, width: int) -> str:
    if len(value) < width:
        value = value[0] * (width - len(value)) + value
    return value[-width:]


def analyze(path: Path) -> dict[str, object]:
    scopes: list[str] = []
    signals: dict[str, dict[str, object]] = {}
    previous: dict[str, str] = {}
    toggles: defaultdict[str, int] = defaultdict(int)
    signal_toggles: defaultdict[str, int] = defaultdict(int)
    definitions_done = False

    with path.open("r", encoding="ascii", errors="strict") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if not definitions_done:
                if line.startswith("$scope"):
                    fields = line.split()
                    scopes.append(fields[2])
                elif line.startswith("$upscope"):
                    scopes.pop()
                elif line.startswith("$var"):
                    fields = line.split()
                    width = int(fields[2])
                    code = fields[3]
                    name = fields[4]
                    full_name = ".".join(scopes + [name])
                    if code not in signals:
                        signals[code] = {"width": width, "names": [full_name]}
                    else:
                        aliases = signals[code]["names"]
                        assert isinstance(aliases, list)
                        aliases.append(full_name)
                        signals[code]["width"] = max(int(signals[code]["width"]), width)
                elif line.startswith("$enddefinitions"):
                    definitions_done = True
                continue
            if line[0] in "01xXzZ":
                value = line[0].lower()
                code = line[1:]
            elif line[0] in "bB":
                fields = line.split()
                if len(fields) != 2:
                    continue
                value = fields[0][1:].lower()
                code = fields[1]
            else:
                continue
            if code not in signals:
                continue
            width = int(signals[code]["width"])
            names = signals[code]["names"]
            assert isinstance(names, list)
            value = normalize(value, width)
            old = previous.get(code)
            previous[code] = value
            if old is None or not known_binary(old) or not known_binary(value):
                continue
            delta = (int(old, 2) ^ int(value, 2)).bit_count()
            if not delta:
                continue
            groups = set()
            for full_name in names:
                if ".u_fixed." in f".{full_name}.":
                    groups.add("fixed")
                if ".u_rqtb." in f".{full_name}.":
                    groups.add("rqtb")
            if len(groups) != 1:
                continue
            group = next(iter(groups))
            group_names = [
                name for name in names if f".u_{group}." in f".{name}."
            ]
            full_name = min(group_names)
            toggles[group] += delta
            signal_toggles[f"{group}:{full_name}"] += delta

    shared_alias_codes = []
    for code, signal in signals.items():
        names = signal["names"]
        assert isinstance(names, list)
        groups = {
            group
            for group in ("fixed", "rqtb")
            if any(f".u_{group}." in f".{name}." for name in names)
        }
        if groups == {"fixed", "rqtb"}:
            shared_alias_codes.append(code)

    top_signals: dict[str, list[dict[str, object]]] = {}
    for group in ("fixed", "rqtb"):
        ranked = sorted(
            (
                {"signal": key.split(":", 1)[1], "bit_toggles": value}
                for key, value in signal_toggles.items()
                if key.startswith(group + ":")
            ),
            key=lambda item: int(item["bit_toggles"]),
            reverse=True,
        )
        top_signals[group] = ranked[:20]

    fixed = toggles["fixed"]
    rqtb = toggles["rqtb"]
    return {
        "schema": "h67_rqtb_vcd_activity_proxy_v1",
        "status": "PASS" if fixed > 0 and rqtb > 0 else "FAIL",
        "source_vcd": str(path.resolve()),
        "source_vcd_sha256": sha256(path),
        "scope": "单个真实T450 row、两级层次VCD位翻转；不是SAIF功耗或ASIC能量",
        "alias_policy": "同时别名到fixed/rqtb层次的VCD identifier作为共享网络排除",
        "shared_alias_codes_excluded": len(shared_alias_codes),
        "bit_toggles": {"fixed": fixed, "rqtb": rqtb},
        "rqtb_reduction_ratio": 1.0 - rqtb / fixed if fixed else None,
        "top_signals": top_signals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.vcd)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
