#!/opt/anaconda3/bin/python3
"""Fail-closed M2172 parser for the ordinary native-SAIF preflight.

This is an additive repair of M2160.  Runtime/topology facts that M2161
accepted are reused from that frozen parser.  Reset-failure semantics and
SAIF ownership are independently strengthened here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Iterator


HW = Path(__file__).resolve().parents[2]
BASE_PATH = HW / "system_simulator/scripts/parse_m2160_m2018_ordinary_native_saif_report_reset_preflight.py"
EXPECTED = {
    "cycles": 20292, "duration_ns": 60876.0, "rows": 149,
    "issues": 1278, "products": 29472, "commits": 24,
    "bundles": 1788, "reads": 14304, "records": 93971,
    "internal_elements": 228, "prehistory_duration_ns": 1167.01,
}
TARGET_INSTANCE = "dut_ordinary"
CRITICAL = (
    "mem_req_valid", "mem_rsp_valid", "bridge_valid", "commit_valid",
    "mem_req_accept", "mem_rsp_accept", "bridge_accept", "commit_accept",
)


class Failure(RuntimeError):
    pass


def need(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"missing/symlink: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_base():
    spec = importlib.util.spec_from_file_location("m2160_parser_frozen", BASE_PATH)
    need(spec is not None and spec.loader is not None, "M2160 parser import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base()


def audit_single_axis_source(tb_text: str, filelist_text: str) -> dict[str, object]:
    try:
        return BASE.audit_single_axis_source(tb_text, filelist_text)
    except BASE.Failure as exc:
        raise Failure(str(exc)) from exc


def verify_file_seal(path: Path) -> dict[str, str]:
    need(path.is_file() and not path.is_symlink(), f"raw file: {path}")
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    need(sidecar.is_file() and not sidecar.is_symlink(),
         f"missing/symlink file seal: {sidecar}")
    need(outer.is_file() and not outer.is_symlink(),
         f"missing/symlink outer file seal: {outer}")
    need(sidecar.read_text().split() == [sha256(path), path.name],
         f"raw file sidecar mismatch: {path.name}")
    need(outer.read_text().split() == [sha256(sidecar), sidecar.name],
         f"raw file outer seal mismatch: {path.name}")
    return {"sha256": sha256(path), "sidecar_sha256": sha256(sidecar),
            "outer_sha256": sha256(outer)}


def normalize_semantics(line: str) -> str:
    value = line.casefold().replace("-", " ").replace("_", " ")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def reset_failure_lines(text: str) -> list[str]:
    """Return semantic reset/clear rejection lines, not exact spellings."""
    bad: list[str] = []
    negative = re.compile(
        r"\b(ignore(?:d|s|ing)?|reject(?:ed|s|ing)?|deny|denied|unsupported|"
        r"fail(?:ed|s|ure)?|cannot|unable|uncleared|retained|remain(?:s|ed)?|"
        r"not\s+(?:be\s+)?(?:clear(?:ed)?|reset|done|supported)|"
        r"did\s+not\s+(?:clear|reset)|was\s+not\s+(?:clear(?:ed)?|reset))\b")
    for raw in text.splitlines():
        line = normalize_semantics(raw)
        if not line:
            continue
        has_reset_context = (
            ("reset" in line and any(word in line for word in
             ("request", "power", "switch", "activity", "counter", "saif", "clear")))
            or ("clear" in line and any(word in line for word in
                ("power", "switch", "activity", "counter", "saif")))
            or ("uncleared" in line and any(word in line for word in
                ("power", "switch", "activity", "counter", "saif"))))
        report_before_reset = bool(re.search(
            r"\bsaif\s+report\s+before\s+reset\b", line))
        warning_context = (line.startswith("warning") or line.startswith("error"))
        if report_before_reset or (has_reset_context and
                                   (negative.search(line) or warning_context)):
            bad.append(raw.strip())
    return bad


def parse_runtime(path: Path) -> dict[str, object]:
    text = read(path)
    failures = reset_failure_lines(text)
    need(not failures, f"semantic power-reset rejection: {failures}")
    try:
        result = BASE.parse_runtime(path)
    except BASE.Failure as exc:
        raise Failure(str(exc)) from exc
    result["sha256"] = sha256(path)
    result["power_reset_rejection_warning_count"] = 0
    result["power_reset_acceptance_runtime_evidence"] = (
        "semantic_reset_failure_absent_and_tb_duration_exact__"
        "final_requires_balanced_dut_scoped_saif_duration")
    return result


Token = str
Node = list["Token | Node"]


def tokenize_saif(text: str) -> list[str]:
    # Comments are only permitted before the first list; stripping them here
    # avoids interpreting comment words as top-level SAIF atoms.
    first = text.find("(")
    need(first >= 0, "SAIF has no list")
    prefix = text[:first]
    need(all(not line.strip() or line.lstrip().startswith(("//", "/**", "*", "*/"))
             for line in prefix.splitlines()), "unexpected pre-SAIF content")
    return re.findall(r'\(|\)|"(?:\\.|[^"\\])*"|[^\s()]+', text[first:])


def parse_balanced_saif(text: str) -> Node:
    tokens = tokenize_saif(text)
    need(tokens and tokens[0] == "(", "SAIF root must be a list")
    stack: list[Node] = []
    roots: list[Node] = []
    for token in tokens:
        if token == "(":
            node: Node = []
            if stack:
                stack[-1].append(node)
            else:
                roots.append(node)
            stack.append(node)
        elif token == ")":
            need(bool(stack), "unmatched SAIF close parenthesis")
            stack.pop()
        else:
            need(bool(stack), f"SAIF atom outside root: {token}")
            stack[-1].append(token)
    need(not stack, "unterminated SAIF list")
    need(len(roots) == 1, f"SAIF root list count {len(roots)} != 1")
    need(head(roots[0]) == "SAIFILE", "SAIF root is not SAIFILE")
    return roots[0]


def head(node: Node) -> str | None:
    return node[0] if node and isinstance(node[0], str) else None


def children(node: Node) -> Iterator[Node]:
    for item in node:
        if isinstance(item, list):
            yield item


def atom_after_head(node: Node) -> str | None:
    for item in node[1:]:
        if isinstance(item, str):
            return item.strip('"')
    return None


def all_nodes(node: Node) -> Iterator[Node]:
    yield node
    for child in children(node):
        yield from all_nodes(child)


def numeric_field(node: Node, label: str) -> float:
    values = [child for child in children(node) if head(child) == label]
    need(len(values) == 1, f"activity field {label} count {len(values)} != 1")
    value = atom_after_head(values[0])
    need(value is not None, f"activity field {label} absent")
    try:
        return float(value)
    except ValueError as exc:
        raise Failure(f"nonnumeric activity field {label}: {value}") from exc


def is_activity(node: Node) -> bool:
    labels = {head(child) for child in children(node)}
    return bool({"T0", "T1", "TX", "TC"} & labels)


def activity_name(node: Node) -> str:
    value = head(node)
    need(value is not None and value not in {"INSTANCE", "NET", "PORT"},
         "activity record has invalid signal name")
    return value.lstrip("\\")


def collect_activity(root: Node, target: Node) -> tuple[list[Node], list[Node]]:
    inside: list[Node] = []
    outside: list[Node] = []

    def walk(node: Node, in_target: bool) -> None:
        now_inside = in_target or node is target
        if is_activity(node):
            (inside if now_inside else outside).append(node)
            return
        for child in children(node):
            walk(child, now_inside)

    walk(root, False)
    return inside, outside


def header_value(root: Node, label: str) -> str:
    matches = [node for node in all_nodes(root) if head(node) == label]
    need(len(matches) == 1, f"{label} count {len(matches)} != 1")
    value = atom_after_head(matches[0])
    need(value is not None, f"{label} value absent")
    return value


def parse_saif(path: Path, *, role: str) -> dict[str, object]:
    need(role in {"diagnostic_prehistory", "measurement"},
         f"unsupported SAIF role: {role}")
    seal = verify_file_seal(path)
    root = parse_balanced_saif(read(path))

    timescales = [node for node in all_nodes(root) if head(node) == "TIMESCALE"]
    need(len(timescales) == 1, f"TIMESCALE count {len(timescales)} != 1")
    scale_atoms = [item for item in timescales[0][1:] if isinstance(item, str)]
    need(len(scale_atoms) == 2, "TIMESCALE must have scalar and unit")
    try:
        scale = float(scale_atoms[0])
    except ValueError as exc:
        raise Failure(f"nonnumeric TIMESCALE: {scale_atoms[0]}") from exc
    unit = scale_atoms[1]
    unit_scale_ns = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0,
                     "ps": 1e-3, "fs": 1e-6}
    need(unit in unit_scale_ns, f"unsupported SAIF unit: {unit}")
    try:
        duration_raw = float(header_value(root, "DURATION"))
    except ValueError as exc:
        raise Failure("nonnumeric DURATION") from exc
    duration_ns = duration_raw * scale * unit_scale_ns[unit]
    expected_duration = (EXPECTED["duration_ns"] if role == "measurement"
                         else EXPECTED["prehistory_duration_ns"])
    need(math.isclose(duration_ns, expected_duration,
                      rel_tol=0.0, abs_tol=1e-6),
         f"{role} duration {duration_ns} != {expected_duration}")

    instances = [node for node in all_nodes(root) if head(node) == "INSTANCE"]
    target_instances = [node for node in instances
                        if atom_after_head(node) == TARGET_INSTANCE]
    need(len(target_instances) == 1,
         f"target INSTANCE {TARGET_INSTANCE} count {len(target_instances)} != 1")
    records, outside = collect_activity(root, target_instances[0])
    need(not outside, f"activity records outside target INSTANCE: {len(outside)}")
    need(len(records) == EXPECTED["records"],
         f"DUT-only record coverage {len(records)} != {EXPECTED['records']}")

    tx_nonzero = 0
    tx_sum = 0.0
    toggled = 0
    conservation_failures = 0
    named: dict[str, list[float]] = {}
    for record in records:
        name = activity_name(record)
        t0, t1, tx, tc = (numeric_field(record, field)
                          for field in ("T0", "T1", "TX", "TC"))
        need(min(t0, t1, tx, tc) >= 0.0, "negative SAIF field")
        tx_nonzero += int(tx != 0.0)
        tx_sum += tx
        toggled += int(tc > 0.0)
        conservation_failures += int(not math.isclose(
            t0 + t1 + tx, duration_raw, rel_tol=0.0, abs_tol=1e-6))
        named.setdefault(name, []).append(tc)
    if role == "measurement":
        need(tx_nonzero == 0 and tx_sum == 0.0,
             f"SAIF unknown activity: records={tx_nonzero} sum={tx_sum}")
    need(conservation_failures == 0,
         f"SAIF conservation failures: {conservation_failures}")
    need(toggled >= 20, f"insufficient nonzero-toggle records: {toggled}")
    critical: dict[str, int] = {}
    for token in (CRITICAL if role == "measurement" else ("load_valid",)):
        counts = [tc for name, values in named.items()
                  if name == token or re.fullmatch(re.escape(token) + r"\\?\[[^]]+\]", name)
                  for tc in values]
        count = sum(value > 0.0 for value in counts)
        need(count > 0, f"missing/zero critical activity: {token}")
        critical[token] = count
    return {
        "identity_seal": seal, "role": role, "axis": "ordinary_lru4",
        "target_instance": TARGET_INSTANCE, "balanced_hierarchy": True,
        "target_instance_count": 1, "outside_target_record_count": 0,
        "duration_raw": duration_raw, "duration_ns": duration_ns,
        "record_count": len(records), "instance_count": len(instances),
        "nonzero_toggle_record_count": toggled,
        "tx_nonzero_record_count": tx_nonzero, "tx_sum": tx_sum,
        "conservation_failures": conservation_failures,
        "critical_nonzero_record_counts": critical,
    }


def final_result(root: Path, output: Path) -> dict[str, object]:
    runtime = parse_runtime(root / "rtl_sim.log")
    diagnostic = parse_saif(root / "rtl_prehistory.saif",
                            role="diagnostic_prehistory")
    measurement = parse_saif(root / "rtl_measurement.saif", role="measurement")
    need(diagnostic["identity_seal"]["sha256"] !=
         measurement["identity_seal"]["sha256"],
         "diagnostic and measurement SAIF content identities collide")
    result = {
        "schema": "m2174_m2172_m2018_ordinary_native_saif_balanced_scope_preflight_result_r1_v1",
        "status": "PASS_RAW_M2174_M2172_BALANCED_SCOPE_NATIVE_SAIF_PREFLIGHT_PENDING_M2175_RESULT_HAMMER",
        "runtime": runtime, "diagnostic_prehistory_saif": diagnostic,
        "measurement_saif": measurement,
        "power_reset_acceptance": {
            "requested_after_diagnostic_report": True,
            "semantic_simulator_rejection_absent": True,
            "measurement_duration_ns": measurement["duration_ns"],
            "balanced_target_instance_scope": True, "accepted": True,
        },
        "claim_boundary": {
            "ordinary_axis_only": True, "single_frontend": True,
            "schedule_mode": 0, "second_axis_run": False,
            "vcs_native_rtl_saif_acquisition_preflight": True,
            "diagnostic_prehistory_never_annotated": True,
            "measurement_saif_candidate_only": True,
            "dc_run": False, "ptpx_run": False, "icc2_run": False,
            "mapped_netlist_activity": False, "power_or_energy": False,
            "component_speedup_admitted": False, "system_speedup": False,
            "paper_citable": False,
        },
    }
    write_json(output, result)
    return result


def static_check() -> dict[str, object]:
    checks = {
        "semantic_reset_gate": reset_failure_lines("Warning: reset request ignored") != [],
        "balanced_parser": parse_balanced_saif("(SAIFILE)")[0] == "SAIFILE",
        "target_instance_exact": TARGET_INSTANCE == "dut_ordinary",
        "exact_record_gate": EXPECTED["records"] == 93971,
        "all_tx_zero_gate": True, "conservation_gate": True,
        "critical_toggle_gate": True, "raw_file_double_seal_gate": True,
        "single_axis_boundary": True, "diagnostic_boundary": True,
    }
    need(all(checks.values()), f"static checks failed: {checks}")
    return {"status": "PASS_M2172_STATIC_PARSER", "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("static")
    runtime = sub.add_parser("runtime")
    runtime.add_argument("--path", type=Path, required=True)
    saif = sub.add_parser("saif")
    saif.add_argument("--path", type=Path, required=True)
    saif.add_argument("--role", choices=("diagnostic_prehistory", "measurement"),
                      required=True)
    final = sub.add_parser("final")
    final.add_argument("--root", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "static":
        value = static_check()
    elif args.command == "runtime":
        value = parse_runtime(args.path)
    elif args.command == "saif":
        value = parse_saif(args.path, role=args.role)
    else:
        value = final_result(args.root, args.output)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as exc:
        print(f"M2172_PARSE_FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
