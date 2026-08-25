#!/usr/bin/env python3
"""Generate auditable PrimeTime RTL-SAIF to mapped-register name mappings."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


INSTANCE_RE = re.compile(r"^\s*\S+\s+([^\s(]+)\s*\(", re.MULTILINE)
INSTANCE_BLOCK_RE = re.compile(
    r"^\s*\S+\s+([^\s(]+)\s*\((.*?)\);\s*$", re.MULTILINE | re.DOTALL
)
Q_NET_RE = re.compile(r"\.Q\(\s*([^\s)]+)\s*\)")
INDEXED_REG_RE = re.compile(r"^(?P<base>[A-Za-z_$][A-Za-z0-9_$]*)_reg_(?P<idx>\d+(?:__\d+)*)_$")
BRACKET_REG_RE = re.compile(r"^(?P<base>[A-Za-z_$][A-Za-z0-9_$]*)_reg\[(?P<idx>\d+)\]$")
SCALAR_REG_RE = re.compile(r"^(?P<base>[A-Za-z_$][A-Za-z0-9_$]*)_reg$")
SAIF_SIGNAL_RE = re.compile(r"^\s*\(([^()\s]+)\s*$", re.MULTILINE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rtl_name(instance: str) -> str | None:
    match = INDEXED_REG_RE.fullmatch(instance)
    if match:
        indices = "".join(f"[{value}]" for value in match.group("idx").split("__"))
        return f"{match.group('base')}{indices}"
    match = BRACKET_REG_RE.fullmatch(instance)
    if match:
        return f"{match.group('base')}[{match.group('idx')}]"
    match = SCALAR_REG_RE.fullmatch(instance)
    if match:
        return match.group("base")
    return None


def saif_signal_names(text: str) -> set[str]:
    names: set[str] = set()
    for encoded in SAIF_SIGNAL_RE.findall(text):
        decoded = encoded.replace(r"\[", "[").replace(r"\]", "]")
        if decoded not in {"NET", "PORT", "INSTANCE", "SAIFILE"}:
            names.add(decoded)
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--netlist", required=True, type=Path)
    parser.add_argument("--saif", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--gate-target",
        choices=("cell", "qnet"),
        default="cell",
        help="map the RTL signal to the sequential cell or its Q net",
    )
    args = parser.parse_args()

    netlist_text = args.netlist.read_text(encoding="utf-8", errors="replace")
    saif_text = args.saif.read_text(encoding="utf-8", errors="replace")
    saif_names = saif_signal_names(saif_text)
    instances = INSTANCE_RE.findall(netlist_text)
    qnet_by_instance: dict[str, str] = {}
    if args.gate_target == "qnet":
        for instance, body in INSTANCE_BLOCK_RE.findall(netlist_text):
            q_match = Q_NET_RE.search(body)
            if q_match:
                qnet_by_instance[instance] = q_match.group(1)

    mappings: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    derived_register_instances = 0
    for instance in instances:
        rtl = rtl_name(instance)
        if rtl is None:
            continue
        derived_register_instances += 1
        gate = instance
        if args.gate_target == "qnet":
            gate = qnet_by_instance.get(instance, "")
            if not gate:
                continue
        pair = (rtl, gate)
        if rtl in saif_names and pair not in seen:
            seen.add(pair)
            mappings.append(pair)
    mappings.sort()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write("# Generated explicit RTL backward-SAIF to gate-register map.\n")
        handle.write(f"set rtl_gate_map_entries {len(mappings)}\n")
        for rtl, gate in mappings:
            handle.write(f"set_rtl_to_gate_name -rtl {{{rtl}}} -gate {{{gate}}}\n")

    mapped_rtl = {rtl for rtl, _ in mappings}
    manifest = {
        "schema": "date.rtl_to_gate_saif_map.v1",
        "netlist": str(args.netlist.resolve()),
        "netlist_sha256": sha256(args.netlist),
        "saif": str(args.saif.resolve()),
        "saif_sha256": sha256(args.saif),
        "mapping_tcl": str(args.output.resolve()),
        "mapping_tcl_sha256": sha256(args.output),
        "gate_target": args.gate_target,
        "netlist_instance_count": len(instances),
        "derived_register_instance_count": derived_register_instances,
        "saif_signal_count": len(saif_names),
        "mapped_pair_count": len(mappings),
        "mapped_unique_rtl_count": len(mapped_rtl),
        "replicated_gate_mapping_count": len(mappings) - len(mapped_rtl),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(args.manifest)


if __name__ == "__main__":
    main()
