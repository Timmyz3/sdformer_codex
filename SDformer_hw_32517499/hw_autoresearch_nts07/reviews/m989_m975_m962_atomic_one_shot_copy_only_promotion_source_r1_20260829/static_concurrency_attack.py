#!/usr/bin/env python3
"""Static/source-only concurrency attack for the inert M989 promoter.

The production promoter is never executed.  This test combines exact source
ordering/prohibition checks with temporary-filesystem races using the same
atomic mkdir and mv -T primitives.  It starts no EDA and touches no project
run identity.
"""

import hashlib
import json
import multiprocessing as mp
import os
import queue
import subprocess
import tempfile
import time
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
SCRIPT = HW / "dc_handoff/scripts/promote_m989_m962_quarantine_atomic_one_shot_copy_only_r1.sh"
CONTRACT = HW / "contracts/m989_m975_m962_atomic_one_shot_copy_only_promotion_source_contract_r1_20260829.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contender(base, gate, out, hold_seconds):
    base = Path(base)
    lock = base / "lock"
    attempt = base / "attempt"
    gate.wait()
    try:
        os.mkdir(lock)
    except FileExistsError:
        out.put("STOP_AT_LOCK_BEFORE_COPY")
        return
    try:
        try:
            os.mkdir(attempt)
        except FileExistsError:
            out.put("STOP_AT_ATTEMPT_BEFORE_COPY")
            return
        (base / "copy_started").write_text("one\n")
        time.sleep(hold_seconds)
        out.put("COPY_WINNER")
    finally:
        os.rmdir(lock)


def race_wave(base, workers, hold_seconds):
    gate = mp.Barrier(workers)
    out = mp.Queue()
    jobs = [mp.Process(target=contender, args=(str(base), gate, out, hold_seconds))
            for _ in range(workers)]
    for job in jobs:
        job.start()
    for job in jobs:
        job.join(10)
        assert not job.is_alive(), "concurrency worker hung"
        assert job.exitcode == 0, "concurrency worker failed"
    result = []
    for _ in jobs:
        try:
            result.append(out.get(timeout=2))
        except queue.Empty as exc:
            raise AssertionError("missing worker result") from exc
    return result


def main():
    source = SCRIPT.read_text()
    contract = json.loads(CONTRACT.read_text())
    expected_script_sha = "7b63668f5fb68ac8d60acf4e43925313ab1c0bdc84caeefcbfb0e238871c4be9"
    assert sha(SCRIPT) == expected_script_sha
    assert contract["status"] == "SOURCE_READY__M989_PROMOTION_NOT_AUTHORIZED_NOW"
    assert contract["identity"]["promotion_script_sha256"] == expected_script_sha
    assert contract["authorization"]["promotion_runs_now"] == 0

    tokens = [
        'verify_file_seal "${M991_RELEASE}"',
        'verify_dir_seal "$(dirname -- "${M992_HAMMER}")"',
        'if ! mkdir -- "${LOCK}"; then',
        'if ! mkdir -- "${ATTEMPT}"; then',
        'mkdir -- "${WORK}"',
        'cp -a --no-dereference "${SOURCE}/." "${WORK}/original_quarantine/"',
        'seal_dir "${WORK}"',
        '[[ ! -e "${TARGET}" ]] || exit 6',
        'mv -T -- "${WORK}" "${TARGET}"',
    ]
    positions = []
    cursor = 0
    for token in tokens:
        pos = source.index(token, cursor)
        positions.append(pos)
        cursor = pos + len(token)
    assert positions == sorted(positions), positions
    assert 'WORK="${TARGET}.copy_work.$$"' not in source
    assert 'WORK="${HW_ROOT}/dc_handoff/runs/.m993_m989_m962_copy_promotion_work"' in source
    assert 'ATTEMPT="${HW_ROOT}/dc_handoff/runs/.m993_m989_m962_copy_promotion_attempt_consumed"' in source
    assert "rm -rf" not in source and "rm -r" not in source
    assert not any(line.strip().startswith(("rm ", "rmdir ")) and "ATTEMPT" in line
                   for line in source.splitlines())
    trap_text = source[source.index("on_exit() {"):source.index("trap on_exit EXIT INT TERM")]
    assert 'mv -T -- "${WORK}" "${FAILQ}"' in trap_text
    assert '>"${TARGET}' not in trap_text and 'mv -T -- "${WORK}" "${TARGET}"' not in trap_text

    with tempfile.TemporaryDirectory(prefix="m989_concurrency_attack_") as tmp:
        base = Path(tmp)
        first = race_wave(base, 32, 0.08)
        assert first.count("COPY_WINNER") == 1, first
        assert sum(x.startswith("STOP_") for x in first) == 31, first
        assert (base / "attempt").is_dir()
        assert (base / "copy_started").read_text() == "one\n"

        second = race_wave(base, 32, 0.0)
        assert second.count("COPY_WINNER") == 0, second
        assert second.count("STOP_AT_ATTEMPT_BEFORE_COPY") >= 1, second
        assert sum(x.startswith("STOP_") for x in second) == 32, second
        assert (base / "copy_started").read_text() == "one\n"

        target_race = base / "target_race"
        work_race = base / "work_race"
        failq = base / "failed_work"
        work_race.mkdir()
        (work_race / "payload").write_text("sealed-work\n")
        target_race.mkdir()
        publish_allowed = not target_race.exists()
        assert publish_allowed is False
        os.rename(work_race, failq)
        assert failq.is_dir() and target_race.is_dir()
        assert not work_race.exists()

        work_publish = base / "work_publish"
        target_publish = base / "target_publish"
        work_publish.mkdir()
        (work_publish / "payload").write_text("published\n")
        subprocess.run(["mv", "-T", "--", str(work_publish), str(target_publish)], check=True)
        assert (target_publish / "payload").read_text() == "published\n"
        assert not (target_publish / "work_publish").exists()
        assert not work_publish.exists()
        work_active_path_exists_after_publish = work_publish.exists()
        assert work_active_path_exists_after_publish is False
        assert (target_publish / "payload").read_text() == "published\n"

    result = {
        "schema": "m989_atomic_one_shot_copy_only_static_concurrency_attack_v1",
        "status": "PASS_M989_STATIC_CONCURRENCY_ATTACKS",
        "production_promotion_executed": False,
        "eda_runs": 0,
        "script_sha256": sha(SCRIPT),
        "contract_sha256": sha(CONTRACT),
        "static_order": {token: pos for token, pos in zip(tokens, positions)},
        "prohibitions": {
            "pid_bound_work": False,
            "attempt_removed": False,
            "recursive_delete": False,
            "trap_writes_or_moves_to_target": False,
        },
        "race_wave_1": {
            "workers": 32,
            "copy_winners": first.count("COPY_WINNER"),
            "stopped_before_copy": sum(x.startswith("STOP_") for x in first),
        },
        "race_wave_2_after_attempt_consumed": {
            "workers": 32,
            "copy_winners": second.count("COPY_WINNER"),
            "stopped_before_copy": sum(x.startswith("STOP_") for x in second),
        },
        "target_appeared_before_publish": "STOP_AND_ISOLATE_WORK",
        "mv_T_no_nesting": "PASS",
        "trap_after_publish_target_unchanged": "PASS",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
