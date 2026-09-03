#!/opt/anaconda3/bin/python
"""Fail-closed audit/merge for the M2057 three-slot successor to failed M2053."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
OLD_RAW = HW / "dc_handoff/runs/m2053_m2051_ep34_tsbg_full40_vcs_raw.77266"
OLD_ATTEMPT = HW / "results/.m2053_m2051_ep34_tsbg_full40_vcs_attempt_consumed"
OLD_RESULT = HW / "results/m2053_m2051_ep34_tsbg_full40_vcs_r1_20260903"
OLD_QUARANTINE = HW / (
    "results/m2053_m2051_ep34_tsbg_full40_vcs_r1_20260903."
    "failed_or_incomplete.77266.quarantine"
)
OLD_PARSER = HW / "system_simulator/scripts/parse_m2053_ep34_tsbg_full40_vcs.py"
FIXTURE_JSON = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

MISSING = (86, 893, 1755)
MISSING_SHA256 = {
    86: "b96eefccfd3797faccfa67e993b4d2b0b6f5977390940b70a435d7ae1cc53ec7",
    893: "12ac7ad2fd938bd3f9432b1739661cf1ae3f68255a69ad5148a9691b82b20032",
    1755: "499028a3b7134af758465b35080ade050a56c4804024665d123831932606c326",
}
OLD_LOG_POPULATION_SHA256 = (
    "4cc5b35eac768625488afb796a3693f84d83ddb30837fa8579451efe3bf218ba"
)
OLD_LOG_TOTAL_BYTES = 55014753
OLD_SIMV_SHA256 = "80887d96cd4bf3c037eb53f383474f29ab7f35a7406f4c4a175a4ed7f8099789"
OLD_SIMV_DAIDIR_SHA256 = (
    "5262b6845a1c4743c6f44fee0ec7be28f078802c4e231cc11adf24ca9e528da8"
)
OLD_COMPILE_SHA256 = (
    "fb774d9d15276c56e02423b3fed31dd767a3124334c21920ac863fbae936a86e"
)
OLD_FAILED_SHA256 = (
    "50b403acabe4b04cc78988777ad3c6ca4a509149b1ca8d1b94d27fa84efa31b1"
)
OLD_ATTEMPT_SEAL_SHA256 = (
    "c42cad06df139dda9d831d4c0ed3cbf175d4987866d0ea4311c0ce24dc353bf9"
)
OLD_PARSER_SHA256 = (
    "2dfa31aaad1e1e3b2a4184eca95e4cdd99170a5c5232e4f2c47596ea15f138fd"
)
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_regular(path: Path, expected: str) -> None:
    assert path.is_file() and not path.is_symlink(), path
    assert sha256(path) == expected, path


def verify_seal(directory: Path) -> None:
    seal = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    assert directory.is_dir() and not directory.is_symlink()
    require_regular(outer, OLD_ATTEMPT_SEAL_SHA256)
    outer_tokens = outer.read_text().strip().split()
    assert len(outer_tokens) == 2 and outer_tokens[1] == "SHA256SUMS"
    assert sha256(seal) == outer_tokens[0]
    listed = []
    for line in seal.read_text().splitlines():
        digest, rel = line.split("  ", 1)
        path = directory / rel
        require_regular(path, digest)
        listed.append(rel)
    actual = sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    )
    assert sorted(listed) == actual


def tree_digest(directory: Path) -> str:
    lines = []
    for path in sorted(directory.rglob("*"), key=lambda p: p.relative_to(directory).as_posix()):
        rel = path.relative_to(directory).as_posix()
        if path.is_symlink():
            lines.append(f"{rel}\tL\t{os.readlink(path)}\n")
        elif path.is_file():
            lines.append(f"{rel}\tF\t{path.stat().st_size}\t{sha256(path)}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def load_old_parser():
    require_regular(OLD_PARSER, OLD_PARSER_SHA256)
    spec = importlib.util.spec_from_file_location("m2053_frozen_parser", OLD_PARSER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_old() -> tuple[object, dict, dict[int, dict]]:
    assert OLD_RAW.is_dir() and not OLD_RAW.is_symlink()
    assert not OLD_RESULT.exists()
    assert OLD_QUARANTINE.is_dir() and not OLD_QUARANTINE.is_symlink()
    assert not any(OLD_QUARANTINE.iterdir())
    verify_seal(OLD_ATTEMPT)
    assert OLD_ATTEMPT.joinpath("ATTEMPT_CONSUMED.txt").read_text() == (
        "status=M2053_ATTEMPT_CONSUMED\nlicense_queries=1\nvcs_compiles=1\n"
        "simv_runs=1920\nsimv_parallelism=4\nretry=false\n"
    )
    require_regular(OLD_RAW / "simv", OLD_SIMV_SHA256)
    assert os.access(OLD_RAW / "simv", os.X_OK)
    assert tree_digest(OLD_RAW / "simv.daidir") == OLD_SIMV_DAIDIR_SHA256
    require_regular(OLD_RAW / "vcs_compile.log", OLD_COMPILE_SHA256)
    require_regular(OLD_RAW / "RUN_FAILED_OR_INCOMPLETE.txt", OLD_FAILED_SHA256)
    assert OLD_RAW.joinpath("RUN_FAILED_OR_INCOMPLETE.txt").read_text() == (
        "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=123\nretry=false\n"
    )

    parser = load_old_parser()
    parser.parse_compile(OLD_RAW / "vcs_compile.log")
    fixture = json.loads(FIXTURE_JSON.read_text())
    assert len(fixture["rows"]) == 1920
    population = []
    total_bytes = 0
    rows: dict[int, dict] = {}
    for slot in range(1920):
        path = OLD_RAW / f"sim_slot{slot}.log"
        assert path.is_file() and not path.is_symlink()
        size = path.stat().st_size
        digest = sha256(path)
        population.append(f"{slot}\t{size}\t{digest}\n")
        total_bytes += size
        if slot in MISSING:
            assert size == 480 and digest == MISSING_SHA256[slot]
            text = path.read_text(errors="replace")
            assert text.count("ASLR will be switched off and simv re-executed") == 1
            assert text.count("Please use '-no_save' simv switch") == 1
            assert "PASS_M2051_EP34_TSBG_FULL40_CYCLE" not in text
            assert not re.search(r"Fatal:|\$(?:error|fatal)|Assertion[^\n]*failed", text, re.I)
        else:
            rows[slot] = parser.parse_sim(path, fixture["rows"][slot])
    digest = hashlib.sha256("".join(population).encode()).hexdigest()
    assert digest == OLD_LOG_POPULATION_SHA256
    assert total_bytes == OLD_LOG_TOTAL_BYTES
    assert len(rows) == 1917 and set(range(1920)) - set(rows) == set(MISSING)
    return parser, fixture, rows


def expected_commands() -> str:
    return "".join(
        f"slot={slot} simv_sha256={OLD_SIMV_SHA256} "
        f"argv=-no_save +WORKLOAD_SLOT={slot} -assert global_finish_maxfail=1\n"
        for slot in MISSING
    )


def merge(new_sim_dir: Path, merged_sim_dir: Path, output: Path) -> None:
    parser, fixture, rows_by_slot = audit_old()
    command_file = new_sim_dir / "M2057_RUN_COMMANDS.txt"
    assert command_file.is_file() and not command_file.is_symlink()
    assert command_file.read_text() == expected_commands()
    discovered_slots = sorted(
        int(path.stem.removeprefix("sim_slot"))
        for path in new_sim_dir.glob("sim_slot*.log")
    )
    assert discovered_slots == list(MISSING)
    new_log_identity = {}
    for slot in MISSING:
        path = new_sim_dir / f"sim_slot{slot}.log"
        text = path.read_text(errors="replace")
        assert "ASLR will be switched off and simv re-executed" not in text
        assert "Please use '-no_save' simv switch" not in text
        rows_by_slot[slot] = parser.parse_sim(path, fixture["rows"][slot])
        new_log_identity[str(slot)] = sha256(path)

    rows = [rows_by_slot[slot] for slot in range(1920)]
    assert len(rows) == 1920
    for slot in range(1920):
        merged = merged_sim_dir / f"sim_slot{slot}.log"
        source = (new_sim_dir if slot in MISSING else OLD_RAW) / f"sim_slot{slot}.log"
        assert merged.is_file() and not merged.is_symlink()
        assert sha256(merged) == sha256(source)
    assert sha256(merged_sim_dir / "vcs_compile.log") == OLD_COMPILE_SHA256
    assert merged_sim_dir.joinpath("M2057_RUN_COMMANDS.txt").read_text() == expected_commands()

    result = {
        "schema": "m2057_m2053_ep34_tsbg_full40_missing3_successor_result_r1_v1",
        "status": "RAW_PASS_PENDING_INDEPENDENT_REVIEW",
        "selection": fixture["selection_rule"],
        "selection_uses_performance": False,
        "workload_scope": {
            "checkpoint": "motion_ep34_live93",
            "sequences": 4,
            "samples": list(range(40)),
            "supported_fc_layers": 16,
            "fc1_layers": 12,
            "fc2_layers": 4,
            "token_regions": ["first", "middle", "last"],
            "workloads": 1920,
            "contexts_per_workload": 4,
            "physical_source_groups": 48,
            "unsupported_fc2_layer_ids_over_g48": fixture[
                "unsupported_fc2_layer_ids_over_g48"
            ],
            "real_activity_and_sign_descriptors": True,
            "real_weights": False,
        },
        "axes": {
            "baseline": "ordinary-LRU4 schedule_mode=0",
            "candidate": "TSBG-B4 schedule_mode=1",
            "same_parametric_rtl": True,
            "same_physical_g48_engine": True,
            "same_external_ports": True,
            "same_cache_capacity": True,
            "same_backpressure_schedule": True,
            "descriptor_preload_excluded_from_execute_cycles": True,
        },
        "rows": rows,
        "aggregate": parser.summarize(rows),
        "breakdown": {
            "target": parser.breakdown(rows, "target"),
            "layer_id": parser.breakdown(rows, "layer_id"),
            "sequence": parser.breakdown(rows, "sequence"),
            "token_role": parser.breakdown(rows, "token_role"),
            "source_groups": parser.breakdown(rows, "source_groups"),
        },
        "cross_attempt_boundary": {
            "failed_parent_attempt": "M2053",
            "failed_parent_raw": OLD_RAW.name,
            "parent_exit_code": 123,
            "parent_status_preserved": "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
            "parent_logs_inherited": 1917,
            "parent_logs_excluded": list(MISSING),
            "successor_attempt": "M2057",
            "successor_slots": list(MISSING),
            "successor_logs": 3,
            "successor_parallelism": 1,
            "successor_runtime_switch": "-no_save",
            "successor_license_queries": 0,
            "successor_vcs_compiles": 0,
            "compiled_simv_reused": True,
            "automatic_retry": False,
            "merged_logs_double_sealed_after_parse": True,
        },
        "identity": {
            "inherited_simv_sha256": OLD_SIMV_SHA256,
            "inherited_simv_daidir_tree_sha256": OLD_SIMV_DAIDIR_SHA256,
            "inherited_compile_log_sha256": OLD_COMPILE_SHA256,
            "inherited_old_log_population_sha256": OLD_LOG_POPULATION_SHA256,
            "inherited_old_attempt_outer_seal_sha256": OLD_ATTEMPT_SEAL_SHA256,
            "frozen_m2053_parser_sha256": OLD_PARSER_SHA256,
            "new_log_sha256": new_log_identity,
            "fixture_json_sha256": sha256(FIXTURE_JSON),
            "docs359_sha256": sha256(DOCS359),
        },
        "claim_boundary": {
            "directed_real_descriptor_component_cycle_distribution": True,
            "cross_attempt_evidence": True,
            "all_fc1_layers_supported": True,
            "all_fc2_layers_supported": False,
            "full_fc_population": False,
            "real_weights": False,
            "same_area": False,
            "energy": False,
            "system_speedup": False,
            "headline": False,
            "paper_admitted": False,
        },
    }
    assert result["aggregate"]["weighted_cycle_speedup"] >= 1.15
    assert result["identity"]["docs359_sha256"] == DOCS359_SHA256
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight-old", action="store_true")
    ap.add_argument("--new-sim-dir", type=Path)
    ap.add_argument("--merged-sim-dir", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.preflight_old:
        assert args.new_sim_dir is None and args.merged_sim_dir is None and args.output is None
        audit_old()
        print("PASS_M2057_M2053_OLD_EVIDENCE_PREFLIGHT slots=1917 missing=86,893,1755")
        return
    assert args.new_sim_dir and args.merged_sim_dir and args.output
    merge(args.new_sim_dir, args.merged_sim_dir, args.output)
    print("PASS_M2057_M2053_MISSING3_MERGE workloads=1920 successor_logs=3")


if __name__ == "__main__":
    main()
