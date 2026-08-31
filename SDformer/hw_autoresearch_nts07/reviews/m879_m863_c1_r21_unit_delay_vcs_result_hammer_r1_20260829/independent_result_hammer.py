#!/usr/bin/env python3
"""Fresh independent M863/C1 R21 canonical VCS result hammer.

Standard-library-only.  It never executes VCS, simv, a license query, EDA, a
remote command, or a workload.  Every mutation is confined to a temporary
copy; the canonical result and frozen source chain are read-only.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile


HW = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RESULT = HW / "results/m863_m533_m528_dead_write_only_1rw_unit_delay_vcs_r21_20260829"
RELEASE = HW / "contracts/m863_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_release_r1_20260829.json"
FINAL = HW / "reviews/m863_m533_r21_unit_delay_vcs_final_launch_release_hammer_r1_20260829"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m863_m533_m528_dead_write_only_1rw_unit_delay_r21_exact_sha.sh"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

RESULT_MANIFEST_SHA = "f3c858c4506778da8268ad464f062c025ba3b265d98df6e71a966d0411fc24e8"
RESULT_OUTER_SHA = "2bc063f29cf44e6d311132c2c14e7858929734d4e32c13f5f1c84e6d7c822510"
RECEIPT_SHA = "7b10955cf7aabaa9d42e631236013f1f68692a165684132df6c463c3d6ce0de1"
INVENTORY_SHA = "662ec47cbc4b441d3a3e9210be92cde1c08ad6ec1faf1020f30191ec64d0d8ee"
RUNNER_SHA = "456a07a05f6dd1447819c84aacfe6bba5ed5541c428c907284f26cf60748b920"
RELEASE_SHA = "2ee62a3069b058235720164a31b145bf96865c175a280570c2098aebb4cd05f6"
RELEASE_OUTER_SHA = "532e0b33da17c09364eab40e13f89a71f11006e79fca6e9b9664c511b2d5a14f"
FINAL_REVIEW_SHA = "5284b097884d2d34ce8e91392372747a43cef27a745a5f9882c77fc1bdeb8683"
FINAL_MANIFEST_SHA = "23364e0096a97c8a1e568dc0bbbf21a207bfc336269b696de1ac85d7eadda5e4"
FINAL_OUTER_SHA = "cac33f1350c8af002e2fc645abe13f836730098a6b724eb945a65a2916c0cdc4"
DOC359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

TERMINAL_EXCLUDED_FROM_INVENTORY = {
    "ARTIFACT_INVENTORY.json", "RUN_COMPLETE.json", "RUN_COMPLETE.txt",
    "RUN_FAILED_OR_INCOMPLETE.json", "FAILED_DO_NOT_CITE",
    "SHA256SUMS", "SHA256SUMS.seal.sha256",
}
NORMAL_COVERS = [
    "dead_plus_read", "deadline_read_write", "same_address_forward",
    "pending_plus_forward", "full_no_credit", "liveness_sequences",
    "parent_modes", "stalled_raw_recovery", "stalled_raw_forward_recovery",
    "stalled_raw_response_recovery", "pingpong_overlap", "endpoint_rows",
    "all_slices",
]
ASSERTION_COVERS = [
    "cp_dead_plus_read", "cp_same_address_forward", "cp_pending_plus_forward",
    "cp_full_then_consume_no_credit", "cp_exact_parent", "cp_pingpong_overlap",
    "cp_row_zero", "cp_row_sixty_three", "cp_all_slices_nonzero",
    "cp_live_deadline_read_then_write", "cp_three_dead",
    "cp_alternating_dead_live", "cp_partial_parent_multibeat",
    "cp_back_to_back_completion", "cp_stalled_same_address",
]


class AuditFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditFailure(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AuditFailure("duplicate JSON key: " + key)
        result[key] = value
    return result


def strict_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=reject_duplicates,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          AuditFailure("nonfinite JSON token: " + token)))


def parse_manifest(path):
    entries = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/].*)", line)
        require(match is not None, "malformed manifest line")
        name = match.group(2)
        require(name not in entries and not name.startswith("../") and
                "/../" not in name and not name.endswith("/.."),
                "duplicate/escaping manifest member")
        entries[name] = match.group(1)
    return entries


def verify_file_double_seal(path, expected_sha, expected_outer=None):
    path = Path(path)
    inner = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(path.is_file() and not path.is_symlink(), "payload not regular")
    require(inner.is_file() and not inner.is_symlink(), "inner seal not regular")
    require(outer.is_file() and not outer.is_symlink(), "outer seal not regular")
    require(sha(path) == expected_sha, "payload SHA drift: " + str(path))
    require(inner.read_text(encoding="utf-8") == expected_sha + "  " + path.name + "\n",
            "inner seal content drift")
    require(outer.read_text(encoding="utf-8") == sha(inner) + "  " + inner.name + "\n",
            "outer seal content drift")
    if expected_outer:
        require(sha(outer) == expected_outer, "outer seal SHA drift")


def verify_flat_double_seal(root, review_name, expected_review_sha,
                            expected_manifest_sha, expected_outer_sha):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "review root invalid")
    for member in root.iterdir():
        require(member.is_file() and not member.is_symlink(),
                "review package contains nonregular member")
    manifest = parse_manifest(root / "SHA256SUMS")
    actual_payload = {p.name for p in root.iterdir()} - {
        "SHA256SUMS", "SHA256SUMS.seal.sha256"}
    require(set(manifest) == actual_payload, "review manifest population drift")
    for name, expected in manifest.items():
        require(sha(root / name) == expected, "review payload drift: " + name)
    require(sha(root / review_name) == expected_review_sha, "review identity drift")
    require(sha(root / "SHA256SUMS") == expected_manifest_sha, "review manifest SHA drift")
    require((root / "SHA256SUMS.seal.sha256").read_text(encoding="utf-8") ==
            expected_manifest_sha + "  SHA256SUMS\n", "review outer content drift")
    require(sha(root / "SHA256SUMS.seal.sha256") == expected_outer_sha,
            "review outer SHA drift")


def live_inventory(root):
    root = Path(root)
    root_resolved = root.resolve(strict=True)
    items = []
    for path in sorted(root.rglob("*"), key=lambda p: str(p.relative_to(root))):
        rel = str(path.relative_to(root))
        if rel in TERMINAL_EXCLUDED_FROM_INVENTORY:
            continue
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            items.append({"path": rel, "type": "directory"})
        elif stat.S_ISREG(mode):
            data = path.read_bytes()
            items.append({"path": rel, "type": "regular", "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})
        elif stat.S_ISLNK(mode):
            raw = os.readlink(path)
            try:
                resolved = path.resolve(strict=True)
                inside = resolved == root_resolved or root_resolved in resolved.parents
                target_regular = resolved.is_file() and not resolved.is_symlink()
            except (FileNotFoundError, RuntimeError, OSError):
                resolved = None
                inside = False
                target_regular = False
            require(inside and target_regular,
                    "PASS tree has external/broken/nonregular symlink: " + rel)
            data = resolved.read_bytes()
            items.append({
                "path": rel, "type": "symlink", "readlink_target": raw,
                "resolved_inside_result": True,
                "resolved_target": str(resolved.relative_to(root_resolved)),
                "target_regular": True, "target_bytes": len(data),
                "target_sha256": hashlib.sha256(data).hexdigest(),
            })
        else:
            raise AuditFailure("special object in PASS tree: " + rel)
    return items


def verify_result_tree(root, pin_identity=True):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "canonical result root invalid")
    manifest = parse_manifest(root / "SHA256SUMS")
    # The terminal manifest deliberately seals regular files only.  Symlink
    # objects and their internal targets are independently bound by inventory.
    live_regular = set()
    for path in root.rglob("*"):
        if path.name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        if stat.S_ISREG(path.lstat().st_mode):
            live_regular.add(str(path.relative_to(root)))
    require(set(manifest) == live_regular, "terminal manifest population drift")
    require(len(manifest) == 123, "terminal regular payload count drift")
    for name, expected in manifest.items():
        path = root / name
        require(path.is_file() and not path.is_symlink(), "manifest member nonregular")
        require(sha(path) == expected, "terminal payload SHA drift: " + name)
    manifest_sha = sha(root / "SHA256SUMS")
    require((root / "SHA256SUMS.seal.sha256").read_text(encoding="utf-8") ==
            manifest_sha + "  SHA256SUMS\n", "terminal outer content drift")
    if pin_identity:
        require(manifest_sha == RESULT_MANIFEST_SHA, "canonical manifest identity drift")
        require(sha(root / "SHA256SUMS.seal.sha256") == RESULT_OUTER_SHA,
                "canonical outer identity drift")
        require(sha(root / "RUN_COMPLETE.json") == RECEIPT_SHA,
                "canonical receipt identity drift")
        require(sha(root / "ARTIFACT_INVENTORY.json") == INVENTORY_SHA,
                "canonical inventory identity drift")
    inventory = strict_json(root / "ARTIFACT_INVENTORY.json")
    require(inventory.get("schema") == "m863_m533_r21_unit_delay_artifact_inventory_v1",
            "inventory schema drift")
    require(inventory.get("terminal_kind") == "success" and
            inventory.get("all_pass_symlinks_internal_regular_and_content_bound") is True,
            "inventory PASS policy drift")
    rebuilt = live_inventory(root)
    require(inventory.get("items") == rebuilt, "inventory/live topology drift")
    counts = {kind: sum(item["type"] == kind for item in rebuilt)
              for kind in ("directory", "regular", "symlink")}
    require(counts == {"directory": 19, "regular": 120, "symlink": 2},
            "inventory type-count drift")
    return {"manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": sha(root / "SHA256SUMS.seal.sha256"),
            "regular_payload_files": len(manifest), "inventory_items": len(rebuilt),
            "inventory_type_counts": counts}


NEGATIVE = re.compile(
    r"failed at|Offending|Assertion\s+fail|^Error(?:-|\[|:)|^Fatal|Fatal:|"
    r"watchdog|Timing violation|Error-\[SVA|\$error|\$fatal|"
    r"normal scoreboard errors|protocol attack not detected", re.I | re.M)


def parse_key_values(line):
    result = {}
    for token in line.split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            require(key not in result, "duplicate token field: " + key)
            result[key] = value
    return result


def assertion_match_count(text, name):
    values = re.findall(r"\." + re.escape(name) +
                        r",\s+\d+ attempts,\s+(\d+) match", text)
    require(values, "assertion cover absent: " + name)
    return max(map(int, values))


def verify_authorities_and_receipt(root):
    verify_file_double_seal(RUNNER, RUNNER_SHA)
    verify_file_double_seal(RELEASE, RELEASE_SHA, RELEASE_OUTER_SHA)
    verify_flat_double_seal(FINAL, "review.json", FINAL_REVIEW_SHA,
                            FINAL_MANIFEST_SHA, FINAL_OUTER_SHA)
    require(sha(DOC359) == DOC359_SHA, "docs/359 drift")
    release = strict_json(RELEASE)
    final = strict_json(FINAL / "review.json")
    require(release.get("schema") ==
            "m863_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_release_v1",
            "release schema drift")
    require(release.get("launch_now") is True and
            release.get("authorization", {}).get("vcs_runs") == 1 and
            release.get("authorization", {}).get("simv_runs") == 1,
            "release authorization drift")
    unique = release.get("unique_attempt", {})
    require(unique.get("result_path") ==
            "results/m863_m533_m528_dead_write_only_1rw_unit_delay_vcs_r21_20260829" and
            unique.get("attempt_identity") ==
            "atomic canonical result-directory mkdir owned by frozen runner" and
            unique.get("max_attempts") == 1,
            "release unique-attempt drift")
    require(final.get("status") ==
            "PASS_M875_M863_C1_R21_FINAL_LAUNCH_RELEASE_HAMMER__EXACTLY_ONE_NOARG_VCS_AUTHORIZED" and
            final.get("verdict") == "PASS" and final.get("score_100") == 100 and
            [final.get(k) for k in ("p0_count", "p1_count", "p2_count")] == [0, 0, 0],
            "final hammer semantics drift")
    require(final.get("identity", {}).get("final_release_sha256") == RELEASE_SHA and
            final.get("identity", {}).get("runner_sha256") == RUNNER_SHA,
            "final hammer binding drift")

    receipt = strict_json(Path(root) / "RUN_COMPLETE.json")
    expected_terminal = {
        "schema": "m863_m533_r21_unit_delay_atomic_terminal_receipt_v1",
        "status": "PASS_FUNCTIONAL_VCS_ONLY", "kind": "success",
        "phase": "success_terminal_seal", "runner_exit_rc": 0,
        "child_rc": "0", "monitor_status": "final_sample_ack_pass",
        "failure_message": "", "preflight_cleanup_rc": "0",
        "paper_citable": False, "macro_model_mode": "foundry_UNIT_DELAY_functional",
    }
    for key, expected in expected_terminal.items():
        require(receipt.get(key) == expected, "receipt terminal drift: " + key)
    require(receipt.get("claim_boundary") == {
        "energy": False, "functional_vcs_only": True,
        "paper_citable_timing": False, "ppa": False, "speedup": False,
        "system_or_paper_headline": False, "timing_verified": False,
    }, "receipt claim boundary drift")
    require(receipt.get("artifact_inventory") == {
        "path": "ARTIFACT_INVENTORY.json", "sha256": INVENTORY_SHA},
        "receipt inventory binding drift")
    require((Path(root) / "RUN_COMPLETE.txt").read_text(encoding="utf-8") ==
            "PASS_FUNCTIONAL_VCS_ONLY phase=success_terminal_seal runner_rc=0 child_rc=0 monitor_status=final_sample_ack_pass\n",
            "terminal text drift")
    bindings = receipt.get("exact_live_launch_bindings", {})
    require(bindings.get("runner_r21_unit_delay", {}).get("sha256") == RUNNER_SHA,
            "receipt runner binding drift")
    require(bindings.get("launch_release", {}).get("sha256") == RELEASE_SHA,
            "receipt release binding drift")
    require(bindings.get("final_release_hammer_review", {}).get("sha256") == FINAL_REVIEW_SHA,
            "receipt final-hammer binding drift")
    # Independently check every recorded launch binding, not just the three
    # headline identities.
    require(len(bindings) >= 50, "live binding population unexpectedly small")
    for name, binding in bindings.items():
        path = Path(binding.get("path", ""))
        require(path.is_file() and not path.is_symlink(),
                "live binding nonregular: " + name)
        require(sha(path) == binding.get("sha256"),
                "live binding SHA drift: " + name)
    return receipt, len(bindings)


def verify_attempt_population():
    matches = sorted(p.name for p in (HW / "results").iterdir()
                     if p.name.startswith(
                         "m863_m533_m528_dead_write_only_1rw_unit_delay_vcs_r21_20260829"))
    require(matches == [RESULT.name], "R21 result/quarantine/attempt population drift")
    return {"canonical_result_directories": 1, "failure_quarantines": 0,
            "max_attempts": 1,
            "attempt_identity": "atomic canonical result-directory mkdir owned by frozen runner"}


def verify_logs(root):
    compile_log = (Path(root) / "compile.log").read_text(encoding="utf-8")
    sim_log = (Path(root) / "sim.log").read_text(encoding="utf-8")
    require("Version V-2023.12-SP1_Full64" in compile_log and
            "All of 4 modules done" in compile_log and
            re.search(r"CPU time: .* to compile .* to elab .* to link", compile_log),
            "compile completion identity missing")
    require(not NEGATIVE.search(compile_log + "\n" + sim_log),
            "error/fatal/assert/watchdog token found")
    coverage_lines = re.findall(r"^COVERAGE_M533_M528_DW1RW_R8 .*$", sim_log, re.M)
    p2_lines = re.findall(r"^P2_STRENGTH_M533_M528_DW1RW_R3 .*$", sim_log, re.M)
    held_lines = re.findall(r"^HELD_FINAL_RECOVERY_M533_M528_DW1RW_R10 .*$", sim_log, re.M)
    pass_lines = re.findall(r"^PASS_M533_M528_DW1RW_R8_DIRECTED_RANDOM_AND_ATTACKS .*$", sim_log, re.M)
    require(len(coverage_lines) == len(p2_lines) == len(held_lines) == len(pass_lines) == 1,
            "terminal functional token cardinality drift")
    coverage = parse_key_values(coverage_lines[0])
    require(coverage.get("normal_covers") == "13" and coverage.get("minima") == "1",
            "normal coverage aggregate drift")
    cover_counts = {name: int(coverage.get(name, "0")) for name in NORMAL_COVERS}
    require(all(value > 0 for value in cover_counts.values()), "zero normal cover")
    p2 = parse_key_values(p2_lines[0])
    require(p2 == {"consecutive_distinct_reads": "19",
                   "response_identity_checks": "189", "minima_pairs": "1",
                   "minima_responses": "2"}, "P2 strength drift")
    held = parse_key_values(held_lines[0])
    require(held == {"preedge_handshake": "1", "accept_edges": "1",
                     "psum_delta": "1", "row_delta": "1", "cover": "1"},
            "held-final recovery drift")
    passed = parse_key_values(pass_lines[0])
    require(passed.get("commits") == "271" and passed.get("done") == "7" and
            passed.get("attacks") == "6", "PASS aggregate drift")
    attack_fields = ["dirty_reserved", "stale_epoch", "overflow", "wrong_parent",
                     "read_before_write", "parent_only_nonzero"]
    require(all(passed.get(name) == "1" for name in attack_fields),
            "six protocol attack classes drift")
    require(passed.get("functional_vcs_only") == "true" and
            all(passed.get(name) == "false" for name in
                ("timing_verified", "trace_recurrence", "speedup", "ppa",
                 "energy", "full_network", "headline")),
            "PASS claim boundary drift")
    assertion_covers = {name: assertion_match_count(sim_log, name)
                        for name in ASSERTION_COVERS}
    require(all(value > 0 for value in assertion_covers.values()),
            "zero mandatory assertion cover")
    require("$finish at simulation time" in sim_log and
            "V C S   S i m u l a t i o n   R e p o r t" in sim_log,
            "simulation completion report missing")
    return {
        "compile_rc": 0, "sim_rc": 0,
        "rc_derivation": "runner_exit_rc=0 and child_rc=0 after source-enforced VCS/tee and simv/tee rc gates",
        "normal_cover_count": 13, "normal_cover_values": cover_counts,
        "assertion_cover_values": assertion_covers,
        "p2": {key: int(value) for key, value in p2.items()},
        "held_final": {key: int(value) for key, value in held.items()},
        "attacks": {name: int(passed[name]) for name in attack_fields},
        "attack_count": 6, "commits": 271, "done": 7,
        "final_pass_token_count": 1,
    }


def reseal(root):
    root = Path(root)
    names = []
    for path in root.rglob("*"):
        if path.name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        if stat.S_ISREG(path.lstat().st_mode):
            names.append(str(path.relative_to(root)))
    lines = [sha(root / name) + "  " + name + "\n" for name in sorted(names)]
    (root / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    (root / "SHA256SUMS.seal.sha256").write_text(
        sha(root / "SHA256SUMS") + "  SHA256SUMS\n", encoding="utf-8")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")


def expect_reject(label, mutate):
    with tempfile.TemporaryDirectory(prefix="m879_c1_r21_result_attack_") as temp:
        copy = Path(temp) / "copy"
        shutil.copytree(RESULT, copy, symlinks=True)
        mutate(copy)
        rejected = False
        try:
            verify_result_tree(copy, pin_identity=False)
            verify_authorities_and_receipt(copy)
            verify_logs(copy)
        except (AuditFailure, OSError, UnicodeError, json.JSONDecodeError,
                ValueError, KeyError):
            rejected = True
        require(rejected, "mutation accepted: " + label)
    return label


def run_mutation_attacks():
    attacks = []
    attacks.append(expect_reject("compile_payload_flip_unsealed",
        lambda root: (root / "compile.log").write_bytes(
            (root / "compile.log").read_bytes() + b"X")))
    attacks.append(expect_reject("extra_regular_file",
        lambda root: (root / "extra.txt").write_text("x\n", encoding="utf-8")))
    attacks.append(expect_reject("manifest_flip",
        lambda root: (root / "SHA256SUMS").write_bytes(
            (root / "SHA256SUMS").read_bytes() + b"X")))
    attacks.append(expect_reject("outer_seal_flip",
        lambda root: (root / "SHA256SUMS.seal.sha256").write_bytes(
            (root / "SHA256SUMS.seal.sha256").read_bytes() + b"X")))

    def receipt_mutation(root, key, value):
        receipt = strict_json(root / "RUN_COMPLETE.json")
        receipt[key] = value
        write_json(root / "RUN_COMPLETE.json", receipt)
        reseal(root)
    attacks.append(expect_reject("receipt_status_resealed",
        lambda root: receipt_mutation(root, "status", "PASS_BUT_WRONG")))
    attacks.append(expect_reject("receipt_runner_rc_nonzero_resealed",
        lambda root: receipt_mutation(root, "runner_exit_rc", 1)))

    def binding_mutation(root):
        receipt = strict_json(root / "RUN_COMPLETE.json")
        receipt["exact_live_launch_bindings"]["runner_r21_unit_delay"]["sha256"] = "0" * 64
        write_json(root / "RUN_COMPLETE.json", receipt)
        reseal(root)
    attacks.append(expect_reject("receipt_runner_binding_resealed", binding_mutation))

    def log_mutation(root, old, new):
        path = root / "sim.log"
        text = path.read_text(encoding="utf-8")
        require(old in text, "attack source token absent")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        reseal(root)
    attacks.append(expect_reject("p2_strength_drift_resealed",
        lambda root: log_mutation(root, "consecutive_distinct_reads=19",
                                  "consecutive_distinct_reads=18")))
    attacks.append(expect_reject("held_final_accept_drift_resealed",
        lambda root: log_mutation(root, "accept_edges=1", "accept_edges=2")))
    attacks.append(expect_reject("attack_count_drift_resealed",
        lambda root: log_mutation(root, "attacks=6", "attacks=5")))
    attacks.append(expect_reject("normal_cover_zero_resealed",
        lambda root: log_mutation(root, "pending_plus_forward=1",
                                  "pending_plus_forward=0")))
    attacks.append(expect_reject("assertion_failure_resealed",
        lambda root: log_mutation(root, "$finish at simulation time",
                                  "Assertion failed at 1ns\n$finish at simulation time")))

    def external_symlink(root):
        path = root / "csrc/_1957651_archive_1.so"
        path.unlink()
        path.symlink_to("/etc/passwd")
        reseal(root)
    attacks.append(expect_reject("external_symlink_resealed", external_symlink))
    return attacks


def verify_output_seal():
    expected = {"RUN_COMPLETE.txt", "independent_result_hammer.py",
                "mechanical_checks.txt", "review.json", "review.md"}
    actual = {path.name for path in OUT.iterdir()} - {
        "SHA256SUMS", "SHA256SUMS.seal.sha256"}
    require(actual == expected, "review output population drift")
    for path in OUT.iterdir():
        require(path.is_file() and not path.is_symlink(), "review output nonregular")
    manifest = parse_manifest(OUT / "SHA256SUMS")
    require(set(manifest) == expected, "review manifest population drift")
    for name, expected_sha in manifest.items():
        require(sha(OUT / name) == expected_sha, "review payload drift: " + name)
    require((OUT / "SHA256SUMS.seal.sha256").read_text(encoding="utf-8") ==
            sha(OUT / "SHA256SUMS") + "  SHA256SUMS\n", "review outer drift")


def seal_output():
    members = ["RUN_COMPLETE.txt", "independent_result_hammer.py",
               "mechanical_checks.txt", "review.json", "review.md"]
    (OUT / "SHA256SUMS").write_text(
        "".join(sha(OUT / name) + "  " + name + "\n" for name in members),
        encoding="utf-8")
    (OUT / "SHA256SUMS.seal.sha256").write_text(
        sha(OUT / "SHA256SUMS") + "  SHA256SUMS\n", encoding="utf-8")
    verify_output_seal()


def main():
    result_identity = verify_result_tree(RESULT)
    receipt, binding_count = verify_authorities_and_receipt(RESULT)
    attempt = verify_attempt_population()
    evidence = verify_logs(RESULT)
    attacks = run_mutation_attacks()
    review = {
        "schema": "m879_m863_c1_r21_unit_delay_vcs_result_hammer_v1",
        "date": "2026-08-29",
        "status": "PASS100_M863_C1_R21_SYNOPSYS_VCS_E3_FUNCTIONAL_RESULT_ADMITTED",
        "verdict": "PASS", "score_out_of_100": 100,
        "p0_count": 0, "p1_count": 0, "p2_count": 0,
        "p0": [], "p1": [], "p2": [],
        "reviewer_role": "Fresh independent receipt/result hammer; no VCS, simv, license, DC, PT, FM, PTPX, EDA, remote, GPU, or workload execution.",
        "identity": {
            "canonical_result": str(RESULT.relative_to(HW)),
            "canonical_result_seal": result_identity,
            "receipt_sha256": RECEIPT_SHA, "inventory_sha256": INVENTORY_SHA,
            "runner_sha256": RUNNER_SHA, "release_sha256": RELEASE_SHA,
            "release_outer_seal_file_sha256": RELEASE_OUTER_SHA,
            "final_hammer_review_sha256": FINAL_REVIEW_SHA,
            "final_hammer_manifest_sha256": FINAL_MANIFEST_SHA,
            "final_hammer_outer_seal_file_sha256": FINAL_OUTER_SHA,
            "docs359_sha256": DOC359_SHA,
            "live_launch_binding_count": binding_count,
        },
        "attempt": attempt,
        "vcs_evidence": dict({"tool": "Synopsys VCS V-2023.12-SP1",
                              "macro_model_mode": "foundry_UNIT_DELAY_functional"},
                             **evidence),
        "mutation_attacks": {"passed": len(attacks), "failed": 0,
                             "names": attacks, "canonical_modified": False},
        "execution_receipt": {
            "canonical_result_modified": False, "frozen_source_modified": False,
            "docs359_modified": False, "vcs_runs_by_reviewer": 0,
            "simv_runs_by_reviewer": 0, "license_queries_by_reviewer": 0,
            "dc_runs_by_reviewer": 0, "pt_runs_by_reviewer": 0,
            "formality_runs_by_reviewer": 0, "ptpx_runs_by_reviewer": 0,
            "all_eda_runs_by_reviewer": 0, "remote_or_network_actions": 0,
        },
        "claim_boundary": {
            "directed_component_synopsys_vcs_e3_functional_citable": True,
            "functional_scope": "C1/M528 dead-write-only single-1RW parent-capture island with foundry UNIT_DELAY functional macro model",
            "rtl_cycle_speedup_verified": False, "timing_verified": False,
            "dc_or_physical_ppa": False, "energy": False,
            "full_network_or_system_speedup": False, "paper_headline": False,
            "m528_cpu_same_ledger_speedup": 1.746753,
            "m528_cpu_same_ledger_speedup_is_not_rtl_cycle_or_ppa_or_system": True,
            "paper_usage": "May cite only as directed component-level Synopsys VCS E3 functional/coverage evidence. The 1.746753x number remains CPU same-ledger only and cannot be relabelled as RTL cycle, PPA, energy, full-network, or system performance.",
        },
        "required_next_gate": "A separately authorized C1 macro-inclusive DC/PT/PTPX and a cycle-accounted RTL-to-ledger bridge are required before any C1 PPA, energy, timing, or RTL speedup statement.",
    }
    mechanical = [
        "PASS canonical 123-regular-payload terminal double seal and inventory/live topology",
        "PASS 120 regular inventory payloads, 19 directories, 2 internal regular content-bound symlinks",
        "PASS receipt/release/final-hammer/runner identities and %d live launch bindings" % binding_count,
        "PASS exactly one atomic R21 canonical attempt and zero failure quarantines",
        "PASS compile and simulation rc=0; no Error/Fatal/assertion failure/watchdog/timing-violation token",
        "PASS 13/13 normal covers nonzero; P2 consecutive/identity=19/189 minima=1/2",
        "PASS held-final preedge/accept/psum/row/cover=1/1/1/1/1",
        "PASS six attack classes exact one each and unique final PASS token",
        "PASS sealed-copy mutation attacks %d/%d" % (len(attacks), len(attacks)),
        "PASS P0=0 P1=0 P2=0",
    ]
    (OUT / "mechanical_checks.txt").write_text("\n".join(mechanical) + "\n",
                                                encoding="utf-8")
    (OUT / "review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    (OUT / "review.md").write_text(
        "# M879 — M863/C1 R21 canonical VCS result hammer\n\n"
        "**PASS 100/100; P0/P1/P2 = 0/0/0.**\n\n"
        "The canonical result is terminally double-sealed and independently bound to the exact runner, release, final launch hammer, 123 regular payloads, and an inventory that admits only two internal regular content-bound tool symlinks. The atomic R21 identity has exactly one canonical result and no failure quarantine.\n\n"
        "Synopsys VCS E3 compiled and simulated successfully. Independent log parsing admits the directed C1 functional result: all 13 normal cover fields are nonzero; P2 reports 19 consecutive distinct reads, 189 response identity checks, and 1/2 minima; held-final recovery is exactly 1/1/1/1/1; all six attack classes are exercised once; and the final PASS token is unique.\n\n"
        "The claim is functional/coverage only. The CPU same-ledger `1.746753x` remains non-RTL-cycle, non-PPA, non-energy, and non-system evidence. Thirteen sealed-copy mutations were rejected without modifying canonical evidence or docs/359.\n",
        encoding="utf-8")
    (OUT / "RUN_COMPLETE.txt").write_text(
        "PASS100_M863_C1_R21_SYNOPSYS_VCS_E3_FUNCTIONAL_RESULT_ADMITTED\n",
        encoding="utf-8")
    seal_output()
    print("PASS100 M879 M863/C1 R21 independent result hammer")


if __name__ == "__main__":
    main()
