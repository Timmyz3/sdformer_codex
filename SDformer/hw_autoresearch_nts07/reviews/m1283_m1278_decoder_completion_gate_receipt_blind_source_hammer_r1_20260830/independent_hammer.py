#!/usr/bin/env python3
"""Receipt-blind, synthetic-only hammer of M1278. Never opens live work."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
OUT = Path(__file__).resolve().parent
SOURCE = HW / "system_simulator/scripts/build_m1278_decoder_completion_gate_and_diagnostic_annex.py"
CONTRACT = HW / "contracts/m1278_decoder_completion_gate_diagnostic_annex_source_contract_r1_20260830.json"
DOCS = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    SOURCE: "52c0829927fb32211df86e0781049f202b2ed63297b3743f121267a6bfa5471d",
    CONTRACT: "6987400c9adc638905675f1b1c3794095ec0ed2d63b887efadefa35ab105edfb",
    DOCS: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class HammerError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise HammerError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha(path) == expected,
            "identity drift: " + str(path))


def load_source():
    name = "m1283_receipt_blind_m1278"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    require(spec is not None and spec.loader is not None, "cannot import M1278")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Fixture:
    def __init__(self, module, runner):
        self.module = module; self.runner = runner
        self.temp = tempfile.TemporaryDirectory(prefix="m1283_m1278.")
        parent = Path(self.temp.name)
        self.layout = module.Layout(parent, parent / module.RESULT_NAME,
            parent / module.ATTEMPT_NAME, parent / module.LOCK_NAME,
            parent / module.WORK_NAME, parent / module.ANNEX_NAME)

    def close(self):
        self.temp.cleanup()

    def attempt(self, maximum: Any = 1):
        self.layout.attempt.mkdir()
        receipt = {"schema": "m1111dr2_decoder_production_attempt_v2",
            "status": "CONSUMED_BEFORE_CANONICAL_PAYLOAD_ACCESS",
            "maximum_attempts": maximum, "automatic_retry": False,
            "canonical_payload_opened_before_attempt": False,
            "runner_sha256": self.module.RUNNER_SHA256,
            "contract_sha256": self.runner.CONTRACT_ID[0]}
        (self.layout.attempt / "attempt.json").write_text(
            json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
        self.runner.atomic_seal(self.layout.attempt)

    def result(self):
        self.runner.build_publish_self_test_candidate(self.layout.result)
        self.runner.atomic_seal(self.layout.result)

    def reseal_result(self):
        shutil.rmtree(self.layout.result / self.runner.SEAL_DIR)
        self.runner.atomic_seal(self.layout.result)

    def complete(self):
        return self.module.completion_gate(self.layout, self.runner,
                                           alive=lambda _: False)

    def live(self, rows: int):
        source = self.layout.parent / "source"
        self.runner.build_publish_self_test_candidate(source)
        self.layout.work.mkdir()
        lines = (source / self.runner.CALLS).read_text(encoding="utf-8").splitlines(True)
        (self.layout.work / self.runner.CALLS).write_text(
            "".join(lines[:rows]), encoding="utf-8")
        self.layout.lock.mkdir()
        (self.layout.lock / "owner.json").write_text(json.dumps({
            "pid": self.module.PRODUCER_PID, "maximum_attempts": 1,
            "automatic_retry": False}, sort_keys=True) + "\n", encoding="utf-8")

    def live_gate(self, cmdline=None):
        return self.module.completion_gate(
            self.layout, self.runner, alive=lambda _: True,
            cmdline=(cmdline or (lambda _: self.module.EXPECTED_CMDLINE)))


def rejected(name: str, function, bucket: list[str]) -> None:
    try:
        function()
    except Exception:
        bucket.append(name)
    else:
        raise HammerError("required rejection escaped: " + name)


def run_rejections(module, runner) -> list[str]:
    passed: list[str] = []

    fx = Fixture(module, runner)
    try:
        fx.attempt(); fx.live(120)
        rejected("forged_live_120_rows", fx.live_gate, passed)
    finally: fx.close()

    fx = Fixture(module, runner)
    try:
        fx.attempt(); fx.live(3); fx.layout.lock.rename(fx.layout.parent / "wrong.lock")
        rejected("missing_exact_live_lock", fx.live_gate, passed)
    finally: fx.close()

    fx = Fixture(module, runner)
    try:
        fx.attempt(); fx.live(3)
        rejected("wrong_pid_cmdline", lambda: fx.live_gate(lambda _: b"python\0wrong\0"), passed)
    finally: fx.close()

    fx = Fixture(module, runner)
    try:
        fx.attempt()
        rejected("missing_completed_result", fx.complete, passed)
    finally: fx.close()

    fx = Fixture(module, runner)
    try:
        fx.attempt(); fx.result(); fx.layout.lock.mkdir()
        rejected("completed_lock_remains", fx.complete, passed)
    finally: fx.close()

    fx = Fixture(module, runner)
    try:
        fx.attempt(); fx.result(); fx.layout.work.mkdir()
        rejected("completed_work_remains", fx.complete, passed)
    finally: fx.close()

    fx = Fixture(module, runner)
    try:
        fx.attempt(); fx.result()
        with (fx.layout.result / runner.CALLS).open("ab") as stream: stream.write(b" ")
        rejected("bad_result_seal", fx.complete, passed)
    finally: fx.close()

    fx = Fixture(module, runner)
    try:
        fx.attempt(); fx.result()
        payload = fx.layout.result / runner.PAYLOAD
        value = json.loads(payload.read_text(encoding="utf-8"))
        value["identity"]["checkpoint"] = "final"
        payload.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        fx.reseal_result()
        rejected("final_checkpoint_identity_promotion", fx.complete, passed)
    finally: fx.close()

    for name, text in (("duplicate_attempt_key", '{"schema":1,"schema":2}\n'),
                       ("nonfinite_attempt", '{"schema":NaN}\n')):
        fx = Fixture(module, runner)
        try:
            fx.layout.attempt.mkdir()
            (fx.layout.attempt / "attempt.json").write_text(text, encoding="utf-8")
            runner.atomic_seal(fx.layout.attempt); fx.result()
            rejected(name, fx.complete, passed)
        finally: fx.close()

    fx = Fixture(module, runner)
    try:
        fx.attempt(); fx.result()
        row = json.loads((fx.layout.result / runner.CALLS).read_text(encoding="utf-8").splitlines()[0])
        row["global_call_ordinal"] = True
        rejected("boolean_call_ordinal", lambda: runner.validate_call_row(row, 0, 0, 0), passed)
    finally: fx.close()
    return passed


def run_escape_audit(module, runner) -> list[dict[str, Any]]:
    escapes = []

    # Python equality accepts True as integer 1 in M1278's exact-dict attempt gate.
    fx = Fixture(module, runner)
    try:
        fx.attempt(maximum=True); fx.result()
        try:
            fx.complete()
        except Exception:
            pass
        else:
            escapes.append({"id": "P1_01_BOOL_AS_INT_ATTEMPT",
                "detail": "maximum_attempts=true passes equality against integer 1"})
    finally: fx.close()

    # The inherited validator accepts integer 0 where a boolean false is required.
    fx = Fixture(module, runner)
    try:
        fx.attempt(); fx.result()
        payload = fx.layout.result / runner.PAYLOAD
        value = json.loads(payload.read_text(encoding="utf-8"))
        value["claim_boundary"]["speedup_admitted"] = 0
        payload.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        fx.reseal_result()
        try:
            fx.complete()
        except Exception:
            pass
        else:
            escapes.append({"id": "P1_02_INT_AS_BOOL_RESULT_CLAIM",
                "detail": "speedup_admitted=0 passes equality against false"})
    finally: fx.close()

    # Last-mile publisher accepts a mutated promoted payload without revalidation.
    fx = Fixture(module, runner)
    try:
        fx.attempt(); fx.result(); gate = fx.complete()
        payload = module.annex_payload(gate)
        payload["identity"]["checkpoint"] = "final"
        payload["claim_boundary"]["ep35_only"] = False
        payload["claim_boundary"]["table_a"] = True
        payload["claim_boundary"]["full_network"] = True
        payload["claim_boundary"]["system_speedup"] = True
        try:
            module.publish_annex(fx.layout, payload)
        except Exception:
            pass
        else:
            escapes.append({"id": "P1_03_LAST_MILE_PROMOTION",
                "detail": "publish_annex writes final-checkpoint/Table-A/system flags"})
    finally: fx.close()

    # The publisher carries no completion capability and writes from incomplete state.
    fx = Fixture(module, runner)
    try:
        fx.attempt(); fx.live(3)
        forged = {"status": "FORGED_INCOMPLETE", "claim_boundary": {
            "diagnostic_only": False, "table_a": True, "system_speedup": True}}
        try:
            module.publish_annex(fx.layout, forged)
        except Exception:
            pass
        else:
            escapes.append({"id": "P1_04_INCOMPLETE_DIRECT_PUBLISH",
                "detail": "publish_annex has no completed-gate capability argument"})
    finally: fx.close()
    return escapes


def write_review(rejections: list[str], escapes: list[dict[str, Any]]) -> None:
    require(len(rejections) == 11, "rejection inventory drift")
    require([row["id"] for row in escapes] == [
        "P1_01_BOOL_AS_INT_ATTEMPT",
        "P1_03_LAST_MILE_PROMOTION", "P1_04_INCOMPLETE_DIRECT_PUBLISH"],
        "expected P1 escapes changed")
    review = {
        "schema": "m1283_m1278_decoder_completion_gate_receipt_blind_source_hammer_r1_v1",
        "status": "STOP_M1283_M1278_THREE_P1_BOUNDARY_ESCAPES__NO_RELEASE_NO_LIVE_PREFLIGHT",
        "verdict": "STOP_REPAIR_REQUIRED",
        "score": 80,
        "issue_counts": {"P0": 0, "P1": 3, "P2": 0},
        "identity": {"source_sha256": EXPECTED[SOURCE],
                     "contract_sha256": EXPECTED[CONTRACT],
                     "docs359_sha256": EXPECTED[DOCS],
                     "author_receipt_read": False},
        "passing_attacks": rejections,
        "findings": escapes,
        "required_repairs": [
            "use type(value) is int/bool gates before all scalar equality",
            "add strict final annex schema/identity/claim validator inside publish_annex",
            "require an unforgeable completed-gate capability or re-run completion validation inside publisher"
        ],
        "execution": {"live_work_read": False, "m1278_live_preflight": False,
                      "replay": False, "eda": False, "gpu": False,
                      "remote": False, "canonical_annex_written": False,
                      "docs359_modified": False},
        "admission": {"source_release": False, "diagnostic_annex": False,
                      "table_a": False, "system_speedup": False,
                      "paper_ppa_ready": False}
    }
    checks = {"schema": "m1283_m1278_receipt_blind_mechanical_r1_v1",
              "status": "PASS_HAMMER_FOUND_THREE_P1", "rejections": rejections,
              "escapes": escapes, "synthetic_temp_only": True,
              "live_work_open_count": 0, "canonical_write_count": 0}
    (OUT / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "mechanical_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "review.md").write_text(
        "# M1283 receipt-blind hammer of M1278\n\n"
        "**STOP, score 80/100; P0/P1/P2 = 0/3/0.** Eleven malformed-state "
        "attacks were rejected, including forged live 120 rows, missing PID/lock/result/work "
        "transitions, bad seal, final-checkpoint identity, duplicate/nonfinite JSON and a "
        "boolean call ordinal.\n\n"
        "Three P1 escapes remain: `true` passes the integer-one attempt check; "
        "`publish_annex` accepts a mutated "
        "final-checkpoint/Table-A/system payload; and that publisher has no completed-gate "
        "capability, so a direct caller can write from an incomplete synthetic state. The "
        "zero-argument main path did not itself promote a claim, hence no P0. Repair scalar "
        "types and add last-mile validation/capability binding before another hammer.\n\n"
        "This hammer was receipt-blind and synthetic-only. It did not open the growing live "
        "work file, run M1278 live preflight, or launch replay/EDA/GPU/remote work.\n",
        encoding="utf-8")
    (OUT / "RUN_COMPLETE.txt").write_text(
        "STOP_M1283_M1278_THREE_P1_BOUNDARY_ESCAPES__NO_RELEASE\n", encoding="utf-8")
    members = sorted(path for path in OUT.iterdir()
                     if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    manifest = OUT / "SHA256SUMS"
    manifest.write_text("".join(sha(path) + "  " + path.name + "\n" for path in members),
                        encoding="utf-8")
    (OUT / "SHA256SUMS.seal.sha256").write_text(
        sha(manifest) + "  SHA256SUMS\n", encoding="utf-8")


def main() -> int:
    before = {str(path): sha(path) for path in EXPECTED}
    for path, digest in EXPECTED.items(): regular(path, digest)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    require(contract["source"]["sha256"] == EXPECTED[SOURCE] and
            contract["claim_boundary"]["table_a"] is False,
            "source contract drift")
    module = load_source(); runner = module.load_runner()
    rejections = run_rejections(module, runner)
    escapes = run_escape_audit(module, runner)
    after = {str(path): sha(path) for path in EXPECTED}
    require(before == after, "source/contract/docs changed")
    write_review(rejections, escapes)
    print("STOP_M1283_M1278_THREE_P1_BOUNDARY_ESCAPES__NO_RELEASE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
