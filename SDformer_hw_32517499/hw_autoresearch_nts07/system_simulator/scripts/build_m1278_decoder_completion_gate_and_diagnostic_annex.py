#!/usr/bin/env python3
"""M1278 additive completion gate for the one-shot M1111DR2 producer.

The production entry is zero-argument and read-only until the already-running
serial producer has atomically published its result.  It never replays a call.
After strict validation it may publish only an ep35 decoder diagnostic annex.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Callable


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
SOURCE_FILE = Path(__file__).resolve()
RUNNER = HW / "system_simulator/scripts/run_m1111dr2_m1105dr2_decoder_only_production_zero_arg.py"
RUNNER_SHA256 = "1167258c228631b73ca1784ae57db19e8f0fbe709efa34f369585c508bc9d746"
CONTRACT = HW / "contracts/m1278_decoder_completion_gate_diagnostic_annex_source_contract_r1_20260830.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

PRODUCER_PID = 4122290
EXPECTED_CMDLINE = (
    b"/opt/anaconda3/envs/pytorch310/bin/python3.10\0-I\0" +
    os.fsencode(RUNNER) + b"\0"
)
WORK_NAME = ".m1111dr2_m1105dr2_decoder_only_production_work.4122290.1788035365192285210"
RESULT_NAME = "m1111dr2_m1105dr2_decoder_only_address_timed_production_r2_20260830"
ATTEMPT_NAME = ".m1111dr2_m1105dr2_decoder_only_production_attempt_consumed"
LOCK_NAME = ".m1111dr2_m1105dr2_decoder_only_production.lock"
QUARANTINE_PREFIX = RESULT_NAME + ".failed_or_incomplete."
ANNEX_NAME = "m1278_h67_ep35_decoder_only_diagnostic_annex_r1_20260830"
CALLS = "m1111dr2_decoder_call_schedule.jsonl"
PAYLOAD = "m1111dr2_decoder_result.json"
COMPLETE = "M1111DR2_DECODER_DIAGNOSTIC_COMPLETE__RESULT_HAMMER_REQUIRED\n"
ANNEX_COMPLETE = "M1278_EP35_DECODER_DIAGNOSTIC_ANNEX_COMPLETE__RESULT_HAMMER_REQUIRED\n"
ANNEX_SEAL = ".m1278_atomic_seal"


class GateError(RuntimeError):
    pass


class Incomplete(GateError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, label: str) -> None:
    try:
        mode = Path(path).lstat().st_mode
    except FileNotFoundError as exc:
        raise GateError("missing " + label) from exc
    require(stat.S_ISREG(mode) and not Path(path).is_symlink(),
            label + " must be a non-symlink regular file")


def strict_json(path: Path) -> dict[str, Any]:
    regular(path, "JSON")
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise GateError("duplicate JSON key")
            value[key] = item
        return value
    def reject(token):
        raise GateError("nonfinite JSON token " + token)
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=reject)
    require(isinstance(value, dict), "JSON root must be object")
    return value


def load_runner():
    regular(RUNNER, "M1111DR2 runner")
    require(sha256(RUNNER) == RUNNER_SHA256, "M1111DR2 runner SHA drift")
    name = "m1278_frozen_m1111dr2"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    require(spec is not None and spec.loader is not None, "cannot load M1111DR2")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Layout:
    parent: Path
    result: Path
    attempt: Path
    lock: Path
    work: Path
    annex: Path


def canonical_layout() -> Layout:
    parent = HW / "results"
    return Layout(parent, parent / RESULT_NAME, parent / ATTEMPT_NAME,
                  parent / LOCK_NAME, parent / WORK_NAME, parent / ANNEX_NAME)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def pid_cmdline(pid: int) -> bytes:
    path = Path("/proc") / str(pid) / "cmdline"
    regular(path, "producer /proc cmdline")
    return path.read_bytes()


def verify_static_authorities() -> None:
    regular(DOCS359, "docs/359")
    require(sha256(DOCS359) == DOCS359_SHA256, "docs/359 SHA drift")
    regular(CONTRACT, "M1278 source contract")
    contract = strict_json(CONTRACT)
    require(contract.get("schema") ==
            "m1278_decoder_completion_gate_diagnostic_annex_source_contract_r1_v1",
            "M1278 contract schema drift")
    source = contract.get("source", {})
    require(source.get("path") == str(SOURCE_FILE.relative_to(REPO)) and
            source.get("sha256") == sha256(SOURCE_FILE), "M1278 source binding drift")
    require(contract.get("claim_boundary") == {
        "source_only": True, "production_replay": False, "gpu": False,
        "remote": False, "eda": False, "table_a": False,
        "system_speedup": False, "paper_ppa_ready": False,
    }, "M1278 claim boundary drift")


def verify_attempt(layout: Layout, runner) -> None:
    require(layout.attempt.is_dir() and not layout.attempt.is_symlink(),
            "consumed attempt missing or unsafe")
    seal = runner.verify_atomic_seal(layout.attempt)
    require(seal["members"] == 1, "attempt seal member count drift")
    receipt = strict_json(layout.attempt / "attempt.json")
    require(receipt == {
        "schema": "m1111dr2_decoder_production_attempt_v2",
        "status": "CONSUMED_BEFORE_CANONICAL_PAYLOAD_ACCESS",
        "maximum_attempts": 1,
        "automatic_retry": False,
        "canonical_payload_opened_before_attempt": False,
        "runner_sha256": RUNNER_SHA256,
        "contract_sha256": runner.CONTRACT_ID[0],
    }, "consumed attempt identity drift")


def read_rows(path: Path, runner, full: bool) -> list[dict[str, Any]]:
    regular(path, "decoder JSONL")
    rows = []
    transaction = 0
    cycle = 0
    with path.open("rb") as stream:
        for expected, raw in enumerate(stream):
            require(expected < 120 and raw.endswith(b"\n") and raw.strip(),
                    "decoder row framing/count drift")
            text = raw.decode("utf-8")
            row = runner.strict_json_text(text)
            require(text == json.dumps(row, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False) + "\n",
                    "decoder row is not canonical JSON")
            transaction, cycle, _ = runner.validate_call_row(
                row, expected, transaction, cycle)
            rows.append(row)
    if full:
        require(len(rows) == 120, "published decoder result is not 120 rows")
    else:
        require(len(rows) < 120, "live prefix reached 120 but producer still owns publish")
    return rows


def verify_live_owner(layout: Layout, runner,
                      cmdline: Callable[[int], bytes] = pid_cmdline) -> dict[str, Any]:
    require(cmdline(PRODUCER_PID) == EXPECTED_CMDLINE,
            "PID exists but exact producer cmdline differs")
    require(not layout.result.exists() and not layout.result.is_symlink(),
            "live producer collided with canonical result")
    require(layout.work.is_dir() and not layout.work.is_symlink(),
            "exact live work directory missing")
    require(layout.lock.is_dir() and not layout.lock.is_symlink(),
            "exact live lock missing")
    owner = strict_json(layout.lock / "owner.json")
    require(owner == {"pid": PRODUCER_PID, "maximum_attempts": 1,
                      "automatic_retry": False}, "live lock owner drift")
    verify_attempt(layout, runner)
    rows = read_rows(layout.work / CALLS, runner, full=False)
    return {"state": "INCOMPLETE", "rows": len(rows), "pid": PRODUCER_PID,
            "published": False, "replay": False}


def completion_gate(layout: Layout, runner,
                    alive: Callable[[int], bool] = pid_alive,
                    cmdline: Callable[[int], bytes] = pid_cmdline) -> dict[str, Any]:
    require(layout.parent.is_dir() and not layout.parent.is_symlink(),
            "result parent missing or unsafe")
    quarantines = list(layout.parent.glob(QUARANTINE_PREFIX + "*"))
    require(not quarantines, "M1111DR2 failure quarantine present")
    if alive(PRODUCER_PID):
        return verify_live_owner(layout, runner, cmdline)
    require(not layout.lock.exists() and not layout.lock.is_symlink(),
            "producer absent but exact lock remains")
    require(not layout.work.exists() and not layout.work.is_symlink(),
            "producer absent but exact work directory remains")
    verify_attempt(layout, runner)
    require(layout.result.is_dir() and not layout.result.is_symlink(),
            "producer absent without canonical result")
    checked = runner.validate_publish_candidate(layout.result)
    rows = read_rows(layout.result / CALLS, runner, full=True)
    require((layout.result / "RUN_COMPLETE.txt").read_text(encoding="utf-8") == COMPLETE,
            "completion token drift")
    require(checked["call_rows"] == 120 and checked["payload"]["identity"] == {
        "checkpoint": "H67_ep35",
        "checkpoint_sha256": "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
        "source_sha256": runner.SOURCE_SHA,
        "contract_sha256": runner.CONTRACT_ID[0],
        "m1110d_outer_seal_file_sha256": runner.M1110D_ID[2],
        "final_checkpoint_rebind_required": True,
    }, "published result identity drift")
    require(all(row["global_call_ordinal"] == index for index, row in enumerate(rows)),
            "published ordinal drift")
    return {"state": "COMPLETE", "rows": rows, "checked": checked,
            "source_result": layout.result, "published": True, "replay": False}


def annex_payload(gate: dict[str, Any]) -> dict[str, Any]:
    require(gate["state"] == "COMPLETE" and gate["replay"] is False,
            "annex requires completed no-replay gate")
    checked = gate["checked"]
    rows = gate["rows"]
    source_result = Path(gate["source_result"])
    try:
        source_label = str(source_result.relative_to(REPO))
    except ValueError:
        source_label = str(source_result)
    modules: dict[str, dict[str, Any]] = {}
    sequences: dict[str, dict[str, int]] = {}
    for row in rows:
        module = modules.setdefault(row["module"], {"calls": 0, "cycles": 0,
            "traffic_bytes": Counter()})
        module["calls"] += 1
        module["cycles"] += row["diagnostic_cycles"]
        module["traffic_bytes"].update(row["diagnostic_traffic_bytes"])
        sequence = sequences.setdefault(row["sequence"], {"calls": 0, "cycles": 0})
        sequence["calls"] += 1
        sequence["cycles"] += row["diagnostic_cycles"]
    module_rows = []
    for name in sorted(modules):
        row = modules[name]
        module_rows.append({"module": name, "calls": row["calls"],
                            "diagnostic_cycles": row["cycles"],
                            "diagnostic_traffic_bytes": dict(row["traffic_bytes"])})
    return {
        "schema": "m1278_h67_ep35_decoder_only_diagnostic_annex_r1_v1",
        "status": "PASS_EP35_DECODER_DIAGNOSTIC_ONLY__RESULT_HAMMER_REQUIRED",
        "source_result": {
            "path": source_label,
            "payload_sha256": sha256(source_result / PAYLOAD),
            "call_schedule_sha256": checked["payload"]["population"]["call_schedule_sha256"],
            "atomic_seal": checked["seal"],
        },
        "identity": checked["payload"]["identity"],
        "population": checked["payload"]["population"],
        "common_resource": checked["payload"]["common_resource"],
        "diagnostic": checked["payload"]["diagnostic"],
        "module_breakdown": module_rows,
        "sequence_breakdown": [dict(sequence=name, **sequences[name])
                               for name in sorted(sequences)],
        "claim_boundary": {
            "ep35_only": True, "decoder_only": True, "diagnostic_only": True,
            "final_checkpoint_rebind_required": True, "ratio_or_speedup": False,
            "table_a": False, "full_network": False, "system_speedup": False,
            "energy": False, "ppa": False, "paper_headline": False,
            "independent_result_hammer_required": True,
        },
    }


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    require(function is not None, "renameat2 unavailable")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                         ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise GateError("annex destination exists")
        raise OSError(code, os.strerror(code), str(destination))


def seal_annex(root: Path) -> dict[str, Any]:
    members = sorted(path for path in root.iterdir() if path.is_file())
    require({path.name for path in members} == {"annex.json", "RUN_COMPLETE.txt"},
            "annex member set drift")
    bundle = root / ANNEX_SEAL
    bundle.mkdir(mode=0o700)
    manifest = bundle / "SHA256SUMS"
    manifest.write_text("".join(sha256(path) + "  " + path.name + "\n"
                                for path in members), encoding="utf-8")
    outer = bundle / "SHA256SUMS.seal.sha256"
    outer.write_text(sha256(manifest) + "  SHA256SUMS\n", encoding="utf-8")
    return {"manifest_sha256": sha256(manifest),
            "outer_seal_file_sha256": sha256(outer), "members": 2}


def publish_annex(layout: Layout, payload: dict[str, Any]) -> dict[str, Any]:
    require(not layout.annex.exists() and not layout.annex.is_symlink(),
            "diagnostic annex namespace not fresh")
    temporary = layout.parent / ("." + ANNEX_NAME + ".stage.%d" % os.getpid())
    require(not temporary.exists() and not temporary.is_symlink(), "annex stage collision")
    temporary.mkdir(mode=0o700)
    try:
        (temporary / "annex.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        (temporary / "RUN_COMPLETE.txt").write_text(ANNEX_COMPLETE, encoding="utf-8")
        seal = seal_annex(temporary)
        rename_noreplace(temporary, layout.annex)
        return {"status": payload["status"], "path": str(layout.annex),
                "seal": seal, "replay": False, "table_a": False}
    except BaseException:
        # A failed stage is deliberately retained for forensic inspection.
        raise


def main() -> int:
    require(len(sys.argv) == 1, "M1278 accepts zero arguments")
    try:
        verify_static_authorities()
        runner = load_runner()
        gate = completion_gate(canonical_layout(), runner)
        if gate["state"] != "COMPLETE":
            sys.stderr.write("M1278_INCOMPLETE_ROWS_%d__NO_OUTPUT_NO_REPLAY\n" % gate["rows"])
            return 75
        result = publish_annex(canonical_layout(), annex_payload(gate))
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as exc:
        sys.stderr.write("M1278_FAIL_CLOSED__NO_REPLAY: %s\n" % exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
