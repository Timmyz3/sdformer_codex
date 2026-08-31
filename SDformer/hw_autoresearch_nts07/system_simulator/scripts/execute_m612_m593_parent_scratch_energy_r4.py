#!/usr/libexec/platform-python3.6
"""M612 r4 exact runner: M606 closures plus path-safe failure evidence."""

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time


def require(value, message):
    if not value:
        raise RuntimeError(message)


def lexical_absolute(path):
    return Path(os.path.abspath(os.fspath(path)))


def lexists(path):
    return os.path.lexists(str(path))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plain_chain(path, directory=False):
    """Walk the caller's lexical path with lstat before any realpath call."""
    lexical = lexical_absolute(path)
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current = current / part
        require(lexists(current), "missing lexical path: " + str(current))
        mode = os.lstat(str(current)).st_mode
        require(not stat.S_ISLNK(mode), "symlink lexical path: " + str(current))
        final = current == lexical
        require(stat.S_ISDIR(mode) if (not final or directory) else stat.S_ISREG(mode),
                "wrong lexical path type: " + str(current))
    require(Path(os.path.realpath(str(lexical))) == lexical,
            "lexical/real path drift after lstat walk")
    return lexical


SELF = plain_chain(__file__)
REPO = SELF.parents[3]
HW = REPO / "hw_autoresearch_nts07"
PYTHON = Path("/usr/libexec/platform-python3.6")
PYTHON_SHA = "9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f"
CORE_REL = "hw_autoresearch_nts07/system_simulator/scripts/execute_m606_m593_parent_scratch_energy_r3.py"
CORE_SHA = "3896c348b809b3094396bc64f63ffc7802866b3a5034e222c8addba8b21640fa"
ADAPTER_REL = "hw_autoresearch_nts07/system_simulator/scripts/analyze_m612_m597_m593_parent_scratch_generated_macro_energy_r4.py"
ADAPTER_SHA = "65f6f006c62a5e7732eefc62106af14b76eb708567da995a3b45ad9a9d78daba"
M607_REL = "hw_autoresearch_nts07/reviews/m607_m606_m593_parent_scratch_energy_exact_runner_static_hammer_r1_20260828"
M607_REVIEW_SHA = "06414d5ccdbddc532e364e349f4dcedbe0b5fe550c7d9cb46b0f180fd1e0553c"
M607_MANIFEST_SHA = "6d9a1b8224d6037c489ce248bd3be8240fd5645d2ba41a6c77e55d30f2c7c42c"
M607_OUTER_SHA = "cd5aaf6ac063620465059f4417019883caf5d7646c260aaa8281ef9071fb905e"
RESULT = HW / "results/m612_m597_m593_parent_scratch_generated_macro_energy_r4_20260828"
ATTEMPT = HW / "results/m612_m597_m593_parent_scratch_generated_macro_energy_r4_20260828.attempt"
CONSUMED = HW / "results/m612_m597_m593_parent_scratch_generated_macro_energy_r4_20260828.attempt.consumed"
AUTH = HW / "contracts/m614_m612_m593_parent_scratch_energy_true_launch_admission_r1_20260828.json"
M613_REVIEW = HW / "reviews/m613_m612_m593_parent_scratch_energy_exact_runner_static_hammer_r1_20260828/review.json"


core_path = REPO / CORE_REL
plain_chain(core_path)
require(sha(core_path) == CORE_SHA, "M606 core runner SHA drift")
spec = importlib.util.spec_from_file_location("m612_exact_m606_runner_core", str(core_path))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
CORE_VERIFY_STATIC = core.verify_static


# Rebind the already-reviewed M606 protocol to immutable M612 identities.
core.ADAPTER_REL = ADAPTER_REL
core.ADAPTER_SHA = ADAPTER_SHA
core.RESULT = RESULT
core.ATTEMPT = ATTEMPT
core.CONSUMED = CONSUMED
core.AUTH = AUTH
core.M607_REVIEW = M613_REVIEW
core.plain_chain = plain_chain


def strict_json(path):
    def pairs(items):
        value = {}
        for key, child in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = child
        return value
    with Path(path).open("r", encoding="utf-8") as handle:
        result = json.load(handle, object_pairs_hook=pairs,
                           parse_constant=lambda token: (_ for _ in ()).throw(RuntimeError(token)))
    require(isinstance(result, dict), "top-level JSON is not object")
    def finite(node):
        if isinstance(node, float): require(math.isfinite(node), "non-finite JSON")
        elif isinstance(node, dict):
            for child in node.values(): finite(child)
        elif isinstance(node, list):
            for child in node: finite(child)
    finite(result)
    return result


def verify_static(shell_path, python_runner_path):
    CORE_VERIFY_STATIC(shell_path, python_runner_path)
    plain_chain(core_path)
    require(sha(core_path) == CORE_SHA, "M606 core runner SHA drift")
    m607 = REPO / M607_REL
    manifest_sha, outer_sha = core.verify_seal(m607, {"review.json", "review.md"})
    require(manifest_sha == M607_MANIFEST_SHA and outer_sha == M607_OUTER_SHA and
            sha(m607 / "review.json") == M607_REVIEW_SHA, "M607 failed-review identity drift")
    review = strict_json(m607 / "review.json")
    require(review.get("schema") == "m607_m606_m593_parent_scratch_energy_exact_runner_static_hammer_v1" and
            review.get("status") == "FAIL_RUNNER_STATIC__TRUE_LAUNCH_ADMISSION_FORBIDDEN__R4_REPAIR_REQUIRED" and
            (review.get("p0_count"), review.get("p1_count"), review.get("p2_count")) == (0, 1, 1) and
            review.get("authorization", {}).get("true_launch_admission_authoring_allowed") is False,
            "M607 failed-review predicate drift")


def stale_quarantine_entries():
    parent = HW / "results"
    prefixes = (".m612_energy.failed_quarantine.staging.", ".m612_energy.failed_raw.")
    return [Path(entry.path) for entry in os.scandir(str(parent))
            if entry.name.startswith(prefixes)]


def verify_coordinates(staging):
    plain_chain(HW / "results", directory=True)
    for path in (RESULT, ATTEMPT, CONSUMED, staging):
        require(path.parent == HW / "results", "coordinate parent drift")
        require(not lexists(path), "coordinate exists: " + str(path))
    require(not stale_quarantine_entries(), "stale M612 quarantine staging/raw entry")


def mode_type(mode):
    if stat.S_ISREG(mode): return "regular_file"
    if stat.S_ISDIR(mode): return "directory"
    if stat.S_ISLNK(mode): return "symlink"
    if stat.S_ISFIFO(mode): return "fifo"
    if stat.S_ISSOCK(mode): return "socket"
    if stat.S_ISCHR(mode): return "character_device"
    if stat.S_ISBLK(mode): return "block_device"
    return "unknown"


def bytes_b64(value):
    return base64.b64encode(value).decode("ascii")


def snapshot_entry(path, relative_bytes=b"."):
    """Serialize arbitrary filesystem evidence without following special entries."""
    encoded_path = os.fsencode(str(path))
    st = os.lstat(encoded_path)
    mode = st.st_mode
    receipt = {
        "relative_path_b64": bytes_b64(relative_bytes),
        "type": mode_type(mode),
        "mode_octal": "%07o" % stat.S_IMODE(mode),
        "device": st.st_dev,
        "inode": st.st_ino,
        "uid": st.st_uid,
        "gid": st.st_gid,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }
    if stat.S_ISLNK(mode):
        receipt["link_target_b64"] = bytes_b64(os.readlink(encoded_path))
    elif stat.S_ISREG(mode):
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(encoded_path, flags)
        try:
            opened = os.fstat(fd)
            require(opened.st_dev == st.st_dev and opened.st_ino == st.st_ino and
                    stat.S_ISREG(opened.st_mode), "regular evidence race")
            digest = hashlib.sha256()
            while True:
                block = os.read(fd, 1 << 20)
                if not block: break
                digest.update(block)
            receipt["content_sha256"] = digest.hexdigest()
        finally:
            os.close(fd)
    elif stat.S_ISDIR(mode):
        children = []
        with os.scandir(encoded_path) as scan:
            entries = sorted(list(scan), key=lambda entry: entry.name)
        for entry in entries:
            child_name = entry.name
            child_rel = child_name if relative_bytes == b"." else relative_bytes + b"/" + child_name
            children.append(snapshot_entry(Path(os.fsdecode(encoded_path + b"/" + child_name)), child_rel))
        receipt["children"] = children
    return receipt


def remove_entry_nofollow(path):
    encoded = os.fsencode(str(path))
    mode = os.lstat(encoded).st_mode
    if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
        with os.scandir(encoded) as scan:
            names = [entry.name for entry in scan]
        for name in names:
            remove_entry_nofollow(Path(os.fsdecode(encoded + b"/" + name)))
        os.rmdir(encoded)
    else:
        os.unlink(encoded)


def quarantine_failure(staging, shell_path, python_runner_path, error, stage, auth_sha):
    parent = HW / "results"
    stamp = "%d.%d" % (int(time.time() * 1000000), os.getpid())
    qstage = parent / (".m612_energy.failed_quarantine.staging." + stamp)
    qfinal = parent / ("m612_energy.failed_or_incomplete." + stamp)
    raw_prefix = ".m612_energy.failed_raw." + stamp + "."
    require(not lexists(qstage) and not lexists(qfinal), "quarantine collision")
    coordinates = [("canonical_result", RESULT), ("attempt", ATTEMPT),
                   ("consumed_attempt", CONSUMED), ("runner_staging", staging)]
    adapter_prefix = "." + staging.name + ".m606_staging_"
    for entry in sorted(os.scandir(str(parent)), key=lambda item: item.name):
        if entry.name.startswith(adapter_prefix):
            coordinates.append(("adapter_internal_staging_%d" % len(coordinates), Path(entry.path)))
    evidence = []
    raw_paths = []
    for index, (name, coordinate) in enumerate(coordinates):
        if not lexists(coordinate):
            evidence.append({"name": name, "original_path": str(coordinate), "present": False})
            continue
        raw = parent / (raw_prefix + str(index))
        require(not lexists(raw), "raw quarantine collision")
        core.rename_noreplace(coordinate, raw)
        raw_paths.append(raw)
        item = snapshot_entry(raw)
        evidence.append({"name": name, "original_path": str(coordinate),
                         "present": True, "filesystem_evidence": item})
        remove_entry_nofollow(raw)
        raw_paths.remove(raw)
    require(all(not lexists(path) for _, path in coordinates), "canonical survived evidence capture")
    require(not raw_paths and not any(lexists(path) for path in raw_paths), "raw evidence survived")
    os.mkdir(str(qstage), 0o700)
    core.write_exclusive(qstage / "filesystem_evidence.json",
        json.dumps({"schema": "m612_arbitrary_filesystem_evidence_v1", "entries": evidence},
                   sort_keys=True, indent=2, ensure_ascii=True) + "\n")
    receipt = {
        "schema": "m612_m593_energy_failed_attempt_quarantine_v1",
        "status": "FAILED_OR_INTERRUPTED_ALL_COORDINATES_SERIALIZED_AND_REMOVED",
        "failure_stage": stage,
        "exception_type": type(error).__name__,
        "message": str(error),
        "runner": {"shell_path": str(shell_path), "shell_sha256": sha(shell_path),
                   "python_path": str(python_runner_path), "python_sha256": sha(python_runner_path)},
        "authorization_sha256_start": auth_sha,
        "canonical_coordinates_absent": all(not lexists(path) for _, path in coordinates),
        "arbitrary_entry_policy": "lstat metadata plus regular-file SHA or symlink target bytes; no special entry is placed inside sealed tree",
    }
    core.write_exclusive(qstage / "failure_receipt.json",
                         json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    core.seal_tree(qstage)
    core.verify_seal(qstage, {"filesystem_evidence.json", "failure_receipt.json"})
    core.rename_noreplace(qstage, qfinal)
    core.verify_seal(qfinal, {"filesystem_evidence.json", "failure_receipt.json"})
    require(not lexists(qstage), "quarantine staging survived publish")
    require(all(not lexists(path) for _, path in coordinates), "canonical survived quarantine")
    require(not stale_quarantine_entries(), "stale raw/staging survived quarantine")
    require(lexists(qfinal), "unique final quarantine missing")


AUTH_KEYS = {"admission_id", "date", "status", "launch_now", "release",
             "runner", "source_static_hammer", "upstream", "canonical", "claim_boundary"}


def verify_authorization(shell_path, python_runner_path):
    plain_chain(AUTH); plain_chain(Path(str(AUTH) + ".sha256")); plain_chain(Path(str(AUTH) + ".sha256.seal.sha256"))
    side = Path(str(AUTH) + ".sha256"); outer = Path(str(AUTH) + ".sha256.seal.sha256")
    auth_sha = sha(AUTH)
    require(side.read_text().strip().split() == [auth_sha, AUTH.name], "auth sidecar drift")
    require(outer.read_text().strip().split() == [sha(side), side.name], "auth outer drift")
    value = strict_json(AUTH)
    require(set(value) == AUTH_KEYS and
            value["admission_id"] == "m614_m612_m593_parent_scratch_energy_true_launch_admission_r1_20260828" and
            value["status"] == "TRUE_LAUNCH_ADMISSION__FRESH_M613_P0_P1_ZERO_REQUIRED" and
            value["launch_now"] is True and value["release"] is True, "authorization predicate drift")
    require(value["runner"] == {
        "shell_path": str(Path(shell_path).relative_to(REPO)), "shell_sha256": sha(shell_path),
        "python_path": str(Path(python_runner_path).relative_to(REPO)), "python_sha256": sha(python_runner_path),
        "adapter_path": ADAPTER_REL, "adapter_sha256": ADAPTER_SHA}, "authorization runner drift")
    require(value["canonical"] == {"result_dir": str(RESULT.relative_to(HW)),
        "attempt_dir": str(ATTEMPT.relative_to(HW)),
        "consumed_attempt_dir": str(CONSUMED.relative_to(HW))}, "authorization coordinates drift")
    require(value["upstream"] == {"m597_contract_sha256": core.CONTRACT_SHA,
        "m597_analyzer_sha256": core.UPSTREAM_SHA, "m612_adapter_sha256": ADAPTER_SHA,
        "m607_failed_review_sha256": M607_REVIEW_SHA}, "authorization upstream drift")
    hammer = value["source_static_hammer"]
    require(set(hammer) == {"path", "sha256", "manifest_sha256", "outer_seal_file_sha256"} and
            REPO / hammer["path"] == M613_REVIEW, "M613 review coordinate drift")
    manifest_sha, outer_sha = core.verify_seal(M613_REVIEW.parent, {"review.json", "review.md"})
    require(sha(M613_REVIEW) == hammer["sha256"] and manifest_sha == hammer["manifest_sha256"] and
            outer_sha == hammer["outer_seal_file_sha256"], "M613 review seal drift")
    review = strict_json(M613_REVIEW)
    require(review.get("schema") == "m613_m612_m593_parent_scratch_energy_exact_runner_static_hammer_v1" and
            review.get("status") == "PASS_RUNNER_STATIC__TRUE_LAUNCH_ADMISSION_AUTHORING_ONLY__NO_EXECUTION" and
            review.get("score_0_to_100") == 100 and
            (review.get("p0_count"), review.get("p1_count")) == (0, 0) and
            review.get("authorization", {}).get("true_launch_admission_authoring_allowed") is True,
            "M613 review predicate drift")
    require(value["claim_boundary"] == {"component_only": True, "paper_data": False,
        "system_energy": False, "result_hammer_pending": True}, "authorization claim drift")
    return auth_sha


core.verify_static = verify_static
core.verify_coordinates = verify_coordinates
core.verify_authorization = verify_authorization
core.quarantine_failure = quarantine_failure


def main(argv):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight-only", action="store_true")
    group.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization")
    parser.add_argument("--shell-path", required=True, type=Path)
    args = parser.parse_args(argv)
    shell_path = plain_chain(args.shell_path)
    python_runner = SELF
    staging = RESULT.parent / (RESULT.name + ".staging." + str(os.getpid()))
    verify_static(shell_path, python_runner)
    verify_coordinates(staging)
    if args.preflight_only:
        require(args.authorization is None, "preflight received authorization")
        cp = subprocess.run([str(PYTHON), str(REPO / ADAPTER_REL), "--source-contract",
            str(REPO / core.CONTRACT_REL), "--self-test"], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True)
        require(cp.returncode == 0 and cp.stdout.splitlines()[-1] ==
                "PASS_M612_PATH_HARDENED_NOREPLACE_ADAPTER_STATIC_SELF_TEST",
                "M612 adapter self-test failed: " + cp.stderr)
        verify_static(shell_path, python_runner); verify_coordinates(staging)
        print("PASS_M612_M593_SOURCE_PREFLIGHT_ONLY__NO_RESULT_ATTEMPT_OR_LAUNCH")
        return 0
    require(args.authorization is not None, "--execute requires future authorization")
    supplied_auth = lexical_absolute(args.authorization)
    require(supplied_auth == AUTH, "authorization path drift")
    core.execute(shell_path, python_runner, staging)
    print("PASS_M612_M593_ATOMIC_COMPONENT_RESULT_PENDING_INDEPENDENT_RESULT_HAMMER " + str(RESULT))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as error:
        print("M612_FAIL_CLOSED: " + str(error), file=sys.stderr)
        sys.exit(70)
