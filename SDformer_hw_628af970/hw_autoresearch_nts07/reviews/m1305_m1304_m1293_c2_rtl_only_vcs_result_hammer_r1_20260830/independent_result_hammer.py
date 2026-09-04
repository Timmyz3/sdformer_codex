#!/usr/bin/env python3
"""Read-only M1305 hammer of the already-completed M1304 VCS result."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RUNS = HW / "dc_handoff/runs"
RUN = RUNS / "m1304_m1293_c2_dual_dut_rtl_only_vcs_r1_20260830"
SOURCE_FILELIST = HW / "dc_handoff/filelists/date_m1293_c2_dual_dut_source_only_vcs.f"
CONTRACT = HW / "contracts/m1293_c2_semantic_tap_dual_dut_repair_source_contract_r1_20260830.json"
M1300 = HW / "reviews/m1300_m1293_c2_semantic_tap_repair_receipt_blind_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
TOP = "tb_m1293_c2_dual_dut_functional_reachability"
PASS = ("PASS_M1293_DUAL_DUT_FUNCTIONAL_REACHABILITY "
        "classification=BOTH_CLEAN_FUNCTIONALLY_EQUAL req=48 result=6 done=1 first=4/58/64")

EXPECTED = {
    SOURCE_FILELIST: "af7c3e4394d35ba037280c82b6395769ed10203a73b91446ac24f7251a169564",
    CONTRACT: "1c50a862e02aeda009d52850f00ba8befa96c19b6599077e61951b36929299f5",
    M1300 / "SHA256SUMS.seal.sha256":
        "89b64ba957ee4440a4ef384cc69c3fd998101519080aaa159d734fe599fb7cdd",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    RUN / "absolute_filelist.f": "82862c680b9586809028dae2318f1982ffead2f7fbda8ecb00721b8b579fa8d2",
    RUN / "compile.log": "4381e8533636e3f0f6daa879b4c38e3085057ebe9c1335dde35b5e602e6bdfee",
    RUN / "sim.log": "47faab5b33be7943e302f5863c70d654d4478ecda07a3f08c5b42d5d629c4462",
    RUN / "simv": "947c6dbe37aa9fb09a1760f8274f1e468bb6d0b2470254350a18133a2b39edd1",
    RUN / "input_sha256.txt": "47228741b8cf257f38f3e6c38d54e263a4e33d8d97857bd402fdbb165f19a12f",
    RUN / "output_sha256.txt": "35cb8ef33a44fa4626c34d52d7bd84471378fb17c80503958471f7e9f87ad47f",
}


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          Failure("nonfinite JSON " + value)))


def verify_manifest(root: Path) -> None:
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(outer.read_text(encoding="utf-8") == sha(manifest) + "  SHA256SUMS\n",
            "M1300 outer seal drift")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        require(sha(root / name) == digest, "M1300 member drift: " + name)


def main() -> int:
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "identity drift: " + str(path))
    verify_manifest(M1300)
    release = strict_json(M1300 / "review.json")["release"]
    require(release["fresh_rtl_only_vcs_authorized"] is True and
            release["maximum_fresh_runs"] == 1, "M1300 release authority drift")

    run_names = sorted(path.name for path in RUNS.glob("*m1293*c2*dual*dut*")
                       if path.is_dir())
    require(run_names == [RUN.name], "fresh/single run namespace drift")
    relative = [line.strip() for line in SOURCE_FILELIST.read_text().splitlines()
                if line.strip()]
    absolute = [line.strip() for line in (RUN / "absolute_filelist.f").read_text().splitlines()
                if line.strip()]
    require(absolute == [str(HW / name) for name in relative],
            "absolute filelist does not exactly project M1293 filelist")
    contract = strict_json(CONTRACT)
    sources = {row["path"]: row["sha256"] for row in contract["sources"]}
    support_sources = {
        "rtl_m218/m218_fc2_tagged_slice_service_island.sv":
            "f6537081977e9dc09e968fad800b333604b4573ee2e9361960483349fe1e8ad1",
        "tb_m349/m349_fc2_scalar_bank_memory_model.sv":
            "4375072b6bd09ada3dc3fd585c12102346ea897192a13630b0c44acf72ff63fa",
    }
    require(set(relative) == set(sources) & set(relative) | set(support_sources),
            "compiled member authority set drift")
    for name in relative:
        require(sha(HW / name) == sources.get(name, support_sources.get(name)),
                "compiled source SHA drift: " + name)
    require(relative[-1] ==
            "dc_handoff/tb/tb_m1293_c2_dual_dut_functional_reachability.sv",
            "top source/filelist tail drift")

    compile_log = (RUN / "compile.log").read_text(encoding="utf-8", errors="strict")
    sim_log = (RUN / "sim.log").read_text(encoding="utf-8", errors="strict")
    require(compile_log.count("Command:") == 1 and sim_log.count("Command:") == 1,
            "not exactly one compile and one simulation command")
    require("-top " + TOP in compile_log and
            re.search(r"Top Level Modules:\s*\n\s*" + TOP, compile_log),
            "exact M1293 top not compiled")
    require("CPU time:" in compile_log and (RUN / "simv").stat().st_mode & 0o111,
            "compile completion/executable missing")
    require(sim_log.count(PASS) == 1 and "$finish at simulation time" in sim_log and
            "V C S   S i m u l a t i o n   R e p o r t" in sim_log,
            "simulation PASS/normal completion drift")
    forbidden = re.compile(r"(?:\$fatal|Fatal:|Error-|Assertion[^\n]*fail|assertion[^\n]*fail|"
                           r"FIRST_X|X_COERCION|UNKNOWN_ESCAPE|\bFAIL\b)", re.I)
    require(forbidden.search(compile_log) is None and forbidden.search(sim_log) is None,
            "fatal/assert/X/failure evidence present")

    expected_input = (sha(SOURCE_FILELIST) + "  " + str(SOURCE_FILELIST) + "\n" +
                      sha(RUN / "absolute_filelist.f") + "  " +
                      str(RUN / "absolute_filelist.f") + "\n")
    require((RUN / "input_sha256.txt").read_text() == expected_input,
            "input SHA receipt drift")
    output_lines = (RUN / "output_sha256.txt").read_text().splitlines()
    output_map = {name: digest for digest, name in
                  (line.split("  ", 1) for line in output_lines)}
    require(output_map == {"compile.log": sha(RUN / "compile.log"),
                           "sim.log": sha(RUN / "sim.log"),
                           "simv": sha(RUN / "simv")}, "output SHA receipt drift")

    result = {
        "schema": "m1305_m1304_m1293_c2_rtl_only_vcs_result_hammer_r1_v1",
        "status": "GO_DIRECTED_K1_DIAGNOSTIC_RTL_VCS_FUNCTIONAL_RECEIPT",
        "score": 96,
        "fresh_single_run": True,
        "compile": {"commands": 1, "top": TOP, "executable_produced": True,
            "normal_tool_completion": True, "numeric_exit_code_persisted": False},
        "simulation": {"commands": 1, "normal_finish": True, "pass_tokens": 1,
            "classification": "BOTH_CLEAN_FUNCTIONALLY_EQUAL", "requests": 48,
            "results": 6, "done": 1, "first_cycles": [4, 58, 64],
            "fatal_or_assert_fail_or_x_escape": False,
            "numeric_exit_code_persisted": False},
        "input_receipt": {"file_sha256": EXPECTED[RUN / "input_sha256.txt"],
            "m1293_filelist_sha256": EXPECTED[SOURCE_FILELIST]},
        "output_receipt": {"file_sha256": EXPECTED[RUN / "output_sha256.txt"],
            "compile_log_sha256": EXPECTED[RUN / "compile.log"],
            "sim_log_sha256": EXPECTED[RUN / "sim.log"],
            "simv_sha256": EXPECTED[RUN / "simv"]},
        "claim_boundary": {"directed_k1_diagnostic_only": True,
            "rtl_functional_vcs": True, "mapped_functionality": False,
            "performance": False, "power": False, "energy": False,
            "system_speedup": False, "paper_headline": False},
        "remaining_p2": "Numeric shell exit codes were not separately persisted; normal compile/link and VCS $finish plus post-run output SHA provide completion evidence.",
        "eda_launched_by_hammer": False,
        "docs359_sha256": EXPECTED[DOCS359],
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
