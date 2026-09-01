#!/usr/bin/env python3
"""M1627 read-only hammer for the published M1613 registered-fault VCS result."""
from __future__ import print_function

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RESULT = HW / "results/m1613_c2_m1609_registered_fault_directed_vcs_r1_20260901"
ATTEMPT = HW / "results/.m1613_c2_m1609_registered_fault_directed_vcs_attempt_consumed"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1621_m1613_c2_m1609_registered_fault_directed_exact_sha_r1.sh"
RTL = HW / "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv"
TB = HW / "dc_handoff/tb/tb_m1613_c2_m1609_registered_fault_directed.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1613_c2_m1609_registered_fault_directed_vcs.f"
SOURCE_CONTRACT = HW / "contracts/m1613_c2_m1609_registered_fault_directed_source_contract_r1_20260901.json"
RUNNER_CONTRACT = HW / "contracts/m1621_m1613_python36_regular_path_runner_successor_source_contract_r1_20260901.json"
SOURCE_HAMMER = HW / "reviews/m1622_m1621_m1613_c2_registered_fault_directed_runner_source_hammer_r1_20260901"
RELEASE = HW / "contracts/m1623_m1622_m1621_m1613_c2_registered_fault_directed_vcs_launch_release_r1_20260901.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

TOP = "tb_m1613_c2_m1609_registered_fault_directed"
PASS_TOKEN = (
    "PASS M1613 M1609 registered-fault directed "
    "legal_terminal_no_false_pulse=1 legal_descriptor_accepts=1 "
    "illegal_header_latched=1 illegal_raw_latched=1 sticky_checks=3 "
    "source_only=false performance=false"
)
PREDECESSOR = "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"
EXPECTED_SYMLINKS = {
    "csrc/_4162708_archive_1.so": ".//../simv.daidir//_4162708_archive_1.so",
    "simv.vdb/snps/coverage/db/testdata/test/assert.verilog.shape.xml":
        "../../common/assert.verilog.shape.xml",
}
EXPECTED_DIRS = {
    "csrc", "csrc/archive.0", "csrc/diag", "csrc/hsim", "csrc/objs",
    "simv.daidir", "simv.daidir/debug_dump", "simv.vdb", "simv.vdb/snps",
    "simv.vdb/snps/coverage", "simv.vdb/snps/coverage/db",
    "simv.vdb/snps/coverage/db/auxiliary",
    "simv.vdb/snps/coverage/db/common",
    "simv.vdb/snps/coverage/db/design",
    "simv.vdb/snps/coverage/db/shape",
    "simv.vdb/snps/coverage/db/testdata",
    "simv.vdb/snps/coverage/db/testdata/test",
}

EXPECTED = {
    RESULT / "SHA256SUMS": "2553aa4ac73f6445c9ceeae3e7a43297a6c98f2bda475dde5f07e30114f85cb8",
    RESULT / "SHA256SUMS.seal.sha256": "824473041e995686cd9fdb6a848132ae3b16d3f8ae8d8cf28b84e0b0dce4a85a",
    RESULT / "receipt.json": "1d11ba89467dd6a6475b70c4dfaa899ed06e44540608c7345593e1e8a537e438",
    RESULT / "runner.log": "f4d82ef0b23c339d4840c388ab66e64d466c6fc04dcae039355267943e37729c",
    RESULT / "compile.log": "2e1fb2d75ffc64fee470f5905444b1e34263562738626691b517b2fc927f67cf",
    RESULT / "compile.rc": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    RESULT / "sim.log": "4fcf716617fa7feed801a984ebf853225c8170d777b1ff70dc40a791e3a79ed2",
    RESULT / "sim.rc": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    RESULT / "RUN_COMPLETE.txt": "d14afdc207ff2c0e0efc37c212c227d430b5c1c3e5385a6840ea0dc0886b4a7f",
    RESULT / "assert.report": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ATTEMPT / "attempt.txt": "dc56944d9d5131ff8a7b27833d810cf58f33d528305c5140b497f6433247f2e0",
    RUNNER: "11da68ff4eb9da70c83b56ae7dd2dbff26f125833224beb08f165fe97a0ea30b",
    RTL: "7ee28b3912ae34c99c795a48e80be29df2b59b363e5de2d2b359175ec9dda931",
    TB: "096f32095a81abbedf4c0bda59a2a146df764d18bdc2136c31e0c7a7319a57a4",
    FILELIST: "071e37d731988a12c3b0adc6380e179de55260c34325fce944daabba6c58671d",
    SOURCE_CONTRACT: "248c9065d81608a8fc2aacdd8539a3287462653e411ee545a8f320a98a8a5f8d",
    RUNNER_CONTRACT: "ccda9594402154f3aeb3105da014cdb2134015017b2c113aa604ed4e385a9fdb",
    SOURCE_HAMMER / "review.json": "da2b75d96feffb3437734a4b5dabb85e4b54f4a313448ba0e9c58db65631f059",
    SOURCE_HAMMER / "SHA256SUMS": "e61bbc785d669cf30a6847cb00c1731543dcc49f82ff207e564a61298ddc1f3a",
    SOURCE_HAMMER / "SHA256SUMS.seal.sha256": "26ba2170172dd381934f39d5ce366c7b0e7794c441e3df7ff31e3c4a2f029090",
    RELEASE: "770196732c5b89d894a50add263a1f74af0dd69d98216d9f104f8904be28b198",
    Path(str(RELEASE) + ".sha256"): "b820c76aa748daa8cb4b353c386d73bebd7f3c4d93cfcf991edd2251391ad38d",
    Path(str(RELEASE) + ".sha256.seal.sha256"):
        "46c7298e2284d3757df9434d2b1435d10a30b585c8bb2c11d74bc4b1f3c375c5",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class Failure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Failure(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out

    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            Failure("non-finite JSON constant: " + value)))


def parse_manifest(root):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and not manifest.is_symlink(), "manifest nonregular")
    require(outer.is_file() and not outer.is_symlink(), "outer seal nonregular")
    require(outer.read_text(encoding="utf-8") == sha(manifest) + "  SHA256SUMS\n",
            "outer seal mismatch")
    rows = manifest.read_text(encoding="utf-8").splitlines()
    require(rows and rows == sorted(rows, key=lambda row: row.split("  ", 1)[1]),
            "manifest missing or unsorted")
    members = {}
    for row in rows:
        require(re.match(r"^[0-9a-f]{64}  \./[^\n]+$", row) is not None,
                "malformed manifest row")
        digest, raw_name = row.split("  ", 1)
        name = raw_name[2:]
        parts = Path(name).parts
        require(name not in members, "duplicate manifest member: " + name)
        require(not Path(name).is_absolute() and parts and
                all(part not in ("", ".", "..") for part in parts) and
                "\\" not in name, "unsafe manifest member: " + name)
        members[name] = digest
    return members


def inventory(root):
    regular = set()
    directories = set()
    links = {}
    special = set()
    for current, dirnames, filenames in os.walk(str(root), topdown=True,
                                                 followlinks=False):
        current_path = Path(current)
        for name in list(dirnames) + list(filenames):
            path = current_path / name
            rel = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                links[rel] = os.readlink(str(path))
                if name in dirnames:
                    dirnames.remove(name)
            elif stat.S_ISDIR(mode):
                directories.add(rel)
            elif stat.S_ISREG(mode):
                regular.add(rel)
            else:
                special.add(rel)
    return regular, directories, links, special


def verify_exact_result_tree(root, pin_manifest):
    members = parse_manifest(root)
    if pin_manifest:
        require(sha(root / "SHA256SUMS") == EXPECTED[RESULT / "SHA256SUMS"],
                "canonical manifest identity drift")
        require(sha(root / "SHA256SUMS.seal.sha256") ==
                EXPECTED[RESULT / "SHA256SUMS.seal.sha256"],
                "canonical outer seal identity drift")
    regular, directories, links, special = inventory(root)
    require(not special, "special result member: " + repr(sorted(special)))
    require(directories == EXPECTED_DIRS,
            "directory topology drift: " + repr(sorted(directories ^ EXPECTED_DIRS)))
    require(links == EXPECTED_SYMLINKS, "symlink topology/target drift")
    expected_regular = set(members) | {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    require(regular == expected_regular,
            "regular-file topology drift: " + repr(sorted(regular ^ expected_regular)))
    root_resolved = root.resolve()
    for name, target_text in links.items():
        link = root / name
        require(os.readlink(str(link)) == target_text, "symlink target text drift")
        target = (link.parent / target_text).resolve(strict=True)
        require(os.path.commonpath([str(root_resolved), str(target)]) == str(root_resolved),
                "symlink escapes sealed result")
        require(target.is_file() and not target.is_symlink(), "symlink target nonregular")
        require(target.relative_to(root_resolved).as_posix() in members,
                "symlink target is not a sealed member")
    for name, digest in members.items():
        path = root / name
        require(path.is_file() and not path.is_symlink(),
                "manifest member nonregular: " + name)
        require(sha(path) == digest, "manifest member digest mismatch: " + name)
    return members


def verify_result_semantics(root):
    receipt = strict_json(root / "receipt.json")
    require(set(receipt) == {
        "schema", "status", "vcs_compiles", "simv_runs", "seed", "performance",
        "dc", "power", "runner_sha256", "successor_sha256", "testbench_sha256",
        "filelist_sha256"}, "receipt schema/key drift")
    require(receipt == {
        "schema": "m1613_c2_m1609_registered_fault_directed_vcs_receipt_r1_v1",
        "status": "PASS", "vcs_compiles": 1, "simv_runs": 1, "seed": 1613,
        "performance": False, "dc": False, "power": False,
        "runner_sha256": EXPECTED[RUNNER], "successor_sha256": EXPECTED[RTL],
        "testbench_sha256": EXPECTED[TB], "filelist_sha256": EXPECTED[FILELIST],
    }, "receipt content drift")
    require((root / "compile.rc").read_text(encoding="ascii") == "0\n",
            "compile rc is not exact zero")
    require((root / "sim.rc").read_text(encoding="ascii") == "0\n",
            "simulation rc is not exact zero")
    require((root / "runner.log").read_text(encoding="ascii") ==
            "VCS_COMPILE vcs_compiles=1\nSIMV_RUN simv_runs=1 seed=1613\n",
            "runner count/seed transcript drift")
    require((root / "RUN_COMPLETE.txt").read_text(encoding="ascii") ==
            "PASS_M1613_M1609_REGISTERED_FAULT_DIRECTED_VCS\n",
            "completion token drift")
    require((root / "assert.report").read_bytes() == b"", "assertion failures recorded")
    require(sha(root / "assert.report.disablelog") ==
            "aecf0c9dffcd66646682ee289bbc1def497a1a7e1b3e3a8fd60586b46ad7c056",
            "assert disable report drift")
    require((root / "simv").stat().st_mode & 0o111, "simv is not executable")

    compile_log = (root / "compile.log").read_text(encoding="utf-8", errors="strict")
    sim_log = (root / "sim.log").read_text(encoding="utf-8", errors="strict")
    parsed = re.findall(r"^Parsing design file '([^']+)'$", compile_log, re.M)
    require(parsed == [
        "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv",
        "dc_handoff/tb/tb_m1613_c2_m1609_registered_fault_directed.sv",
    ], "compiled source set/order drift")
    require(PREDECESSOR not in compile_log and "rtl_m214/" not in compile_log,
            "frozen predecessor appears in compile evidence")
    require(re.search(r"Top Level Modules:\s*\n\s*" + re.escape(TOP) + r"\s*$",
                      compile_log, re.M) is not None, "top identity drift")
    require("2 modules and 0 UDP read." in compile_log and
            "CPU time:" in compile_log and "simv up to date" in compile_log,
            "compile completion/source count drift")
    require(sim_log.count(PASS_TOKEN) == 1 and
            len(re.findall(r"^PASS(?: |$)", sim_log, re.M)) == 1,
            "PASS token is absent or not unique")
    require("$finish called from file \"dc_handoff/tb/tb_m1613_c2_m1609_registered_fault_directed.sv\", line 238." in sim_log and
            "$finish at simulation time                61501" in sim_log and
            "V C S   S i m u l a t i o n   R e p o r t" in sim_log,
            "normal simulation completion drift")
    forbidden = re.compile(
        r"(?:Error-\[|^Error|^Fatal|Fatal:|\$fatal|failed at|Offending|watchdog|"
        r"assert(?:ion)?[^\n]*fail|\bFAIL(?:ED)?\b)", re.I | re.M)
    require(forbidden.search(compile_log) is None and
            forbidden.search(sim_log) is None and
            forbidden.search((root / "assert.report.disablelog").read_text(
                encoding="utf-8", errors="strict")) is None,
            "error/fatal/assertion/watchdog/failure evidence found")


def verify_external_authority():
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "frozen identity drift: " + str(path))
    source_rows = [row for row in FILELIST.read_text(encoding="utf-8").splitlines()
                   if row.strip()]
    require(source_rows == [
        "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv",
        "dc_handoff/tb/tb_m1613_c2_m1609_registered_fault_directed.sv",
    ] and PREDECESSOR not in source_rows, "successor-only filelist drift")
    source_contract = strict_json(SOURCE_CONTRACT)
    require(source_contract["rtl_selection"]["selection"] == "successor_only" and
            source_contract["rtl_selection"]["predecessor_in_new_filelist"] is False,
            "source contract predecessor/selection drift")
    hammer = strict_json(SOURCE_HAMMER / "review.json")
    require(hammer["status"] ==
            "PASS_M1622_M1621_M1613_C2_REGISTERED_FAULT_RUNNER_HAMMER__AUTHORIZE_ONE_FUTURE_VCS_ATTEMPT" and
            hammer["score"] >= 95 and hammer["p0_count"] == 0 and
            hammer["p1_count"] == 0, "source-hammer admission drift")
    release = strict_json(RELEASE)
    require(release["status"] ==
            "AUTHORIZE_ONE_M1621_M1613_C2_REGISTERED_FAULT_DIRECTED_VCS_ATTEMPT",
            "release status drift")
    require(release["authorization"] == {
        "vcs_compiles": 1, "simv_runs": 1, "all_other_eda_runs": 0,
    }, "release execution budget drift")
    require(release["execution_contract"] == {
        "top": TOP, "seed": 1613, "successor_only_filelist": True,
        "attempt_consumed_before_vcs": True, "same_uid_vcs_collision_tolerance": 0,
        "automatic_retry": False, "atomic_no_replace_publication": True,
    }, "release execution contract drift")
    require(release["identity"]["runner_sha256"] == EXPECTED[RUNNER] and
            release["identity"]["source_contract_sha256"] == EXPECTED[SOURCE_CONTRACT] and
            release["identity"]["hammer_review_sha256"] == EXPECTED[SOURCE_HAMMER / "review.json"] and
            release["identity"]["successor_sha256"] == EXPECTED[RTL] and
            release["identity"]["testbench_sha256"] == EXPECTED[TB] and
            release["identity"]["filelist_sha256"] == EXPECTED[FILELIST],
            "release identity binding drift")
    require(release["claim_boundary"]["cycle_performance"] is False and
            release["claim_boundary"]["speedup"] is False and
            release["claim_boundary"]["area"] is False and
            release["claim_boundary"]["timing"] is False and
            release["claim_boundary"]["power"] is False and
            release["claim_boundary"]["energy"] is False and
            release["claim_boundary"]["paper_result"] is False,
            "release claim boundary drift")
    release_name = RELEASE.name
    require(Path(str(RELEASE) + ".sha256").read_text(encoding="ascii") ==
            EXPECTED[RELEASE] + "  " + release_name + "\n", "release inner seal drift")
    require(Path(str(RELEASE) + ".sha256.seal.sha256").read_text(encoding="ascii") ==
            EXPECTED[Path(str(RELEASE) + ".sha256")] + "  " +
            release_name + ".sha256\n", "release outer seal drift")


def verify_attempt():
    regular, directories, links, special = inventory(ATTEMPT)
    require(regular == {"attempt.txt"} and not directories and not links and not special,
            "attempt namespace topology drift")
    expected = ("M1613_ATTEMPT_CONSUMED runner_sha256=" + EXPECTED[RUNNER] +
                " automatic_retry=false\n")
    require((ATTEMPT / "attempt.txt").read_text(encoding="ascii") == expected,
            "attempt content drift")
    require((ATTEMPT / "attempt.txt").stat().st_mtime_ns <
            (RESULT / "compile.log").stat().st_mtime_ns <
            (RESULT / "sim.log").stat().st_mtime_ns,
            "attempt-before-compile-before-simulation timestamp order absent")
    runner_text = RUNNER.read_text(encoding="utf-8")
    require(runner_text.index('mkdir "${attempt}"') <
            runner_text.index('"${vcs}" -full64') <
            runner_text.index('"${simv}" +ntb_random_seed=1613'),
            "runner attempt/tool ordering drift")


def reseal_member(root, name):
    manifest = root / "SHA256SUMS"
    rows = manifest.read_text(encoding="utf-8").splitlines()
    target = "./" + name
    replaced = []
    found = False
    for row in rows:
        digest, raw_name = row.split("  ", 1)
        if raw_name == target:
            digest = sha(root / name)
            found = True
        replaced.append(digest + "  " + raw_name)
    require(found, "mutation member is absent from manifest: " + name)
    manifest.write_text("\n".join(replaced) + "\n", encoding="utf-8")
    (root / "SHA256SUMS.seal.sha256").write_text(
        sha(manifest) + "  SHA256SUMS\n", encoding="ascii")


def run_mutations():
    labels = []
    with tempfile.TemporaryDirectory(prefix="m1627_result_hammer_") as temporary:
        temp = Path(temporary)

        def reject(label, mutator, semantic=False):
            case = temp / ("case_%02d" % (len(labels) + 1))
            shutil.copytree(str(RESULT), str(case), symlinks=True)
            mutator(case)
            try:
                verify_exact_result_tree(case, False)
                if semantic:
                    verify_result_semantics(case)
            except Failure:
                labels.append(label)
                return
            raise Failure("mutation escaped rejection: " + label)

        reject("extra_flat_file", lambda root: (root / "EXTRA").write_text("x"))
        reject("extra_nested_file", lambda root: (
            (root / "unexpected/deep").mkdir(parents=True),
            (root / "unexpected/deep/member").write_text("x")))
        reject("extra_empty_directory", lambda root: (root / "empty_extra").mkdir())
        reject("extra_symlink", lambda root: (root / "extra_link").symlink_to("sim.rc"))
        reject("manifested_member_replaced_by_symlink", lambda root: (
            (root / "compile.rc").unlink(), (root / "compile.rc").symlink_to("sim.rc")))
        reject("known_symlink_target_drift", lambda root: (
            (root / "csrc/_4162708_archive_1.so").unlink(),
            (root / "csrc/_4162708_archive_1.so").symlink_to("../sim.rc")))

        def duplicate_manifest(root):
            manifest = root / "SHA256SUMS"
            first = manifest.read_text(encoding="utf-8").splitlines()[0]
            manifest.write_text(manifest.read_text(encoding="utf-8") + first + "\n",
                                encoding="utf-8")
            (root / "SHA256SUMS.seal.sha256").write_text(
                sha(manifest) + "  SHA256SUMS\n", encoding="ascii")
        reject("duplicate_manifest_member", duplicate_manifest)

        def unsorted_manifest(root):
            manifest = root / "SHA256SUMS"
            rows = manifest.read_text(encoding="utf-8").splitlines()
            rows[0], rows[1] = rows[1], rows[0]
            manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
            (root / "SHA256SUMS.seal.sha256").write_text(
                sha(manifest) + "  SHA256SUMS\n", encoding="ascii")
        reject("unsorted_manifest", unsorted_manifest)

        def special_fifo(root):
            os.mkfifo(str(root / "special_fifo"))
        reject("special_node", special_fifo)

        def mutate_text(name, old, new):
            def mutate(root):
                path = root / name
                content = path.read_text(encoding="utf-8")
                require(old in content, "mutation needle absent")
                path.write_text(content.replace(old, new, 1), encoding="utf-8")
                reseal_member(root, name)
            return mutate

        reject("duplicate_receipt_json_key", mutate_text(
            "receipt.json", '"status":"PASS"', '"status":"PASS","status":"FAIL"'), True)
        reject("receipt_compile_count", mutate_text(
            "receipt.json", '"vcs_compiles":1', '"vcs_compiles":2'), True)
        reject("receipt_sim_count", mutate_text(
            "receipt.json", '"simv_runs":1', '"simv_runs":2'), True)
        reject("receipt_seed", mutate_text(
            "receipt.json", '"seed":1613', '"seed":1614'), True)
        reject("receipt_performance_inflation", mutate_text(
            "receipt.json", '"performance":false', '"performance":true'), True)
        reject("receipt_runner_identity", mutate_text(
            "receipt.json", EXPECTED[RUNNER], "0" * 64), True)
        reject("compile_rc_nonzero", mutate_text("compile.rc", "0\n", "1\n"), True)
        reject("sim_rc_nonzero", mutate_text("sim.rc", "0\n", "1\n"), True)
        reject("runner_extra_compile", mutate_text(
            "runner.log", "VCS_COMPILE vcs_compiles=1\n",
            "VCS_COMPILE vcs_compiles=1\nVCS_COMPILE vcs_compiles=2\n"), True)
        reject("runner_wrong_seed", mutate_text(
            "runner.log", "seed=1613", "seed=1614"), True)
        reject("duplicate_pass_token", mutate_text(
            "sim.log", PASS_TOKEN, PASS_TOKEN + "\n" + PASS_TOKEN), True)
        reject("simulation_error_injection", mutate_text(
            "sim.log", PASS_TOKEN, "Error-[MUTATION]\n" + PASS_TOKEN), True)
        reject("predecessor_compile_injection", mutate_text(
            "compile.log", "Parsing design file 'dc_handoff/tb/",
            "Parsing design file '" + PREDECESSOR + "'\nParsing design file 'dc_handoff/tb/"), True)
        reject("completion_token_drift", mutate_text(
            "RUN_COMPLETE.txt", "PASS_M1613", "FAIL_M1613"), True)
        def assertion_failure(root):
            (root / "assert.report").write_text("Assertion failed at 1ns\n",
                                                encoding="utf-8")
            reseal_member(root, "assert.report")
        reject("assertion_report_failure", assertion_failure, True)
    return labels


def main():
    verify_external_authority()
    members = verify_exact_result_tree(RESULT, True)
    verify_result_semantics(RESULT)
    verify_attempt()
    mutations = run_mutations()
    require(len(mutations) == 24, "mutation-count drift")
    output = {
        "schema": "m1627_m1613_c2_registered_fault_directed_vcs_result_mechanical_checks_r1_v1",
        "status": "PASS_M1627_M1613_C2_REGISTERED_FAULT_DIRECTED_VCS_RESULT_HAMMER",
        "score": 99,
        "p0_count": 0,
        "p1_count": 0,
        "p2_count": 0,
        "result": {
            "manifest_members": len(members),
            "directories": len(EXPECTED_DIRS),
            "pinned_internal_symlinks": len(EXPECTED_SYMLINKS),
            "compile_rc": 0,
            "sim_rc": 0,
            "vcs_compiles": 1,
            "simv_runs": 1,
            "seed": 1613,
            "pass_tokens": 1,
            "legal_terminal_no_false_pulse": 1,
            "legal_descriptor_accepts": 1,
            "illegal_header_latched": 1,
            "illegal_raw_latched": 1,
            "sticky_checks": 3,
            "predecessor_compiled": False,
            "assertion_or_error_or_fatal": False,
        },
        "attempt": {
            "consumed": True,
            "automatic_retry": False,
            "mtime_before_compile": True,
            "runner_order_before_vcs": True,
        },
        "mutation_hammer": {
            "attacks": len(mutations), "rejections": len(mutations),
            "labels": mutations,
            "extra_flat": "REJECTED", "extra_nested": "REJECTED",
            "extra_or_replaced_symlink": "REJECTED",
            "duplicate_manifest": "REJECTED",
        },
        "claim_boundary": {
            "directed_compactor_local_registered_fault_behavior": True,
            "integration_outer_error_or_chain": False,
            "performance": False,
            "speedup": False,
            "area": False,
            "timing": False,
            "power": False,
            "energy": False,
            "paper_headline": False,
        },
        "eda_launched_by_hammer": False,
        "docs359_sha256": EXPECTED[DOCS359],
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
