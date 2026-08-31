#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent mutation hammer for M1340.  Never emits a real Table-A row."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "hw_autoresearch_nts07/system_simulator/scripts/build_m1340_ep34_table_a_common_charge_compiler.py"
AUTHOR_TEST = REPO / "hw_autoresearch_nts07/system_simulator/tests/test_m1340_ep34_table_a_common_charge_compiler.py"
DOCS359 = REPO / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"
AUTHOR = REPO / "hw_autoresearch_nts07/reviews/m1340_ep34_table_a_common_charge_compiler_source_author_r1_20260831"
SPEC = importlib.util.spec_from_file_location("m1340_blind_target", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture:
    def __init__(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="m1341_blind_")
        self.root = Path(self.temp.name)
        (self.root / "inputs").mkdir()
        (self.root / "docs").mkdir()
        protected = self.root / "docs/359_DATE终局冻结_20260813.md"
        protected.write_bytes(DOCS359.read_bytes()); protected.chmod(0o444)
        self.population = []
        for sequence, stratum in (("interlaken_01_a", "low"),
                                  ("thun_01_b", "medium"),
                                  ("zurich_city_12_a", "high")):
            for sample in range(10):
                self.population.append({"sequence_id": sequence,
                    "sample_id": sample, "density_stratum": stratum,
                    "weight": 1.0 / 30.0})
        media = {"checkpoint": "application/octet-stream",
                 "config": "application/yaml", "profile": "application/json",
                 "capture_result": "application/json",
                 "capture_result_hammer": "application/json"}
        self.identity = {name: self.write_spec("identity_%s.dat" % name,
                         ("self-forged-%s" % name).encode(), kind)
                         for name, kind in media.items()}
        self.common = {name: self.charge_spec("common", name, 10, 1)
                       for name in M.COMMON_CATEGORIES}
        self.branches = {}
        row_cycles = {"B0": 100, "B1": 92, "B2": 88,
                      "B3": 84, "C2": 82, "Ours": 70}
        for branch_index, branch in enumerate(M.DIRECT_BRANCHES):
            self.branches[branch] = {}
            for row, cycles in row_cycles.items():
                self.branches[branch][row] = self.charge_spec(
                    "direct", "%s.%s" % (branch, row),
                    cycles + branch_index, 1)
        energy = {"schema": M.ENERGY_SCHEMA,
            "identity": "Motion-C12-ep34-final",
            "native_mapped_activity_coverage": 0.96,
            "logic_pj_per_cycle": {row: 2.0 for row in M.ROWS},
            "dram_pj_per_byte": {"read": 3.7, "write": 3.9},
            "sram_pj_per_byte": {macro: {"read": 0.2, "write": 0.3}
                                  for macro in M.SRAM_MACROS}}
        self.energy = self.write_json_spec("energy.json", energy)
        self.config = {"schema": M.SCHEMA, "status": "SOURCE_FIXTURE",
            "identity": {"name": "Motion-C12-ep34-final", **self.identity},
            "resource": copy.deepcopy(M.RESOURCE),
            "population": self.population, "rows": list(M.ROWS),
            "common_operators": self.common,
            "direct_branches": self.branches,
            "energy_authority": self.energy,
            "claim_boundary": {"same_denominator": True,
                "common_charge_identical_all_rows": True,
                "component_speedups_not_multiplied": True,
                "external_prosperity_not_ours": True,
                "independent_hammer_required": True,
                "paper_headline_admitted": False},
            "protected_file": {"path": "docs/359_DATE终局冻结_20260813.md",
                "sha256": M.PROTECTED_SHA256, "media_type": "text/markdown"}}

    def close(self) -> None:
        self.temp.cleanup()

    def write_spec(self, name: str, content: bytes,
                   media: str = "application/json") -> dict[str, str]:
        path = self.root / "inputs" / name
        path.write_bytes(content); path.chmod(0o444)
        return {"path": path.relative_to(self.root).as_posix(),
                "sha256": digest(path), "media_type": media}

    def write_json_spec(self, name: str, payload: Any) -> dict[str, str]:
        return self.write_spec(name, (json.dumps(payload, sort_keys=True,
            separators=(",", ":"), allow_nan=False) + "\n").encode())

    def charge(self, cycles: int, numerator: int) -> dict[str, Any]:
        return {"cycles": cycles, "fixed_numerator": numerator,
            "dram_read_bytes": cycles * 2, "dram_write_bytes": cycles,
            "sram_bytes": {macro: {"read_bytes": cycles + index,
                                    "write_bytes": index}
                           for index, macro in enumerate(M.SRAM_MACROS)}}

    def charge_spec(self, kind: str, name: str, cycles: int,
                    numerator: int) -> dict[str, str]:
        payload = {"schema": M.CHARGE_SCHEMA, "kind": kind, "name": name,
                   "identity": "Motion-C12-ep34-final", "population": {}}
        for point in self.population:
            key = M.population_key(point["sequence_id"], point["sample_id"])
            payload["population"][key] = self.charge(cycles + point["sample_id"],
                                                       numerator)
        return self.write_json_spec("charge_%s_%s.json" %
                                    (kind, name.replace(".", "_")), payload)

    def mutate_json_spec(self, spec: dict[str, str],
                         mutation: Callable[[dict[str, Any]], None]) -> None:
        path = self.root / spec["path"]
        path.chmod(0o644); payload = json.loads(path.read_text())
        mutation(payload)
        path.write_text(json.dumps(payload, sort_keys=True,
                                   separators=(",", ":"), allow_nan=False) + "\n")
        path.chmod(0o444); spec["sha256"] = digest(path)

    def config_path(self, raw: str | None = None) -> Path:
        path = self.root / "config.json"
        if path.lexists() if hasattr(path, "lexists") else path.exists():
            path.chmod(0o644); path.unlink()
        path.write_text(raw if raw is not None else json.dumps(self.config),
                        encoding="utf-8"); path.chmod(0o444)
        return path

    def build(self) -> dict[str, Any]:
        return M.build(self.config_path(), self.root)


results: list[dict[str, Any]] = []


def run_case(name: str, expect: str, body: Callable[[Fixture], Any]) -> Any:
    fixture = Fixture()
    try:
        try:
            value = body(fixture)
            observed = "ACCEPT"
            detail = value if isinstance(value, (str, int, float, bool, dict, list)) else str(value)
        except M.CompileError as exc:
            observed = "REJECT"
            detail = str(exc)
        passed = observed == expect
        results.append({"name": name, "expected": expect,
                        "observed": observed, "test_pass": passed,
                        "detail": detail})
        if not passed:
            raise AssertionError("%s expected %s observed %s: %s" %
                                 (name, expect, observed, detail))
        return value if observed == "ACCEPT" else None
    finally:
        fixture.close()


def mutate_first_charge(payload: dict[str, Any], mutation: Callable[[dict[str, Any]], None]) -> None:
    mutation(next(iter(payload["population"].values())))


def main() -> int:
    run_case("M01_missing_common_category", "REJECT",
             lambda f: (f.config["common_operators"].pop("fc1"), f.build())[1])
    run_case("M02_missing_row", "REJECT",
             lambda f: (f.config["rows"].remove("B2"), f.build())[1])
    run_case("M03_missing_direct_branch", "REJECT",
             lambda f: (f.config["direct_branches"].pop("attention"), f.build())[1])

    def different_row_numerator(f: Fixture) -> Any:
        spec = f.branches["decoder"]["Ours"]
        f.mutate_json_spec(spec, lambda p: [row.__setitem__("fixed_numerator", 2)
                                           for row in p["population"].values()])
        return f.build()
    run_case("M04_different_row_fixed_numerator", "REJECT", different_row_numerator)

    def rowized_common(f: Fixture) -> Any:
        f.mutate_json_spec(f.common["fc1"], lambda p: p.__setitem__("rows", ["B0"]))
        return f.build()
    run_case("M05_common_charge_rowized", "REJECT", rowized_common)

    def missing_dram(f: Fixture) -> Any:
        f.mutate_json_spec(f.common["fc1"],
            lambda p: mutate_first_charge(p, lambda c: c.pop("dram_read_bytes")))
        return f.build()
    run_case("M06_missing_dram_field", "REJECT", missing_dram)

    def missing_sram(f: Fixture) -> Any:
        f.mutate_json_spec(f.common["fc1"],
            lambda p: mutate_first_charge(p,
                lambda c: c["sram_bytes"].pop("parent_00")))
        return f.build()
    run_case("M07_missing_charge_sram_macro", "REJECT", missing_sram)

    def missing_energy_macro(f: Fixture) -> Any:
        f.mutate_json_spec(f.energy,
            lambda p: p["sram_pj_per_byte"].pop("parent_00"))
        return f.build()
    run_case("M08_missing_energy_sram_macro", "REJECT", missing_energy_macro)

    def low_coverage(f: Fixture) -> Any:
        f.mutate_json_spec(f.energy,
            lambda p: p.__setitem__("native_mapped_activity_coverage", 0.949))
        return f.build()
    run_case("M09_coverage_below_95", "REJECT", low_coverage)
    run_case("M10_population_not_3x10", "REJECT",
             lambda f: (f.config["population"].pop(), f.build())[1])
    run_case("M11_weight_sum_wrong", "REJECT",
             lambda f: (f.config["population"][0].__setitem__("weight", 0.5), f.build())[1])
    run_case("M12_density_stratum_missing", "REJECT",
             lambda f: ([p.__setitem__("density_stratum", "low")
                         for p in f.config["population"]], f.build())[1])
    run_case("M13_resource_lane_drift", "REJECT",
             lambda f: (f.config["resource"].__setitem__("source_lanes", 95), f.build())[1])
    run_case("M14_resource_port_drift", "REJECT",
             lambda f: (f.config["resource"].__setitem__("external_read_ports_per_bank", 2), f.build())[1])
    run_case("M15_resource_queue_drift", "REJECT",
             lambda f: (f.config["resource"].__setitem__("group_fifo_depth", 5), f.build())[1])
    run_case("M16_identity_name_drift", "REJECT",
             lambda f: (f.config["identity"].__setitem__("name", "fake"), f.build())[1])
    run_case("M17_sha_drift", "REJECT",
             lambda f: (f.config["identity"]["profile"].__setitem__("sha256", "0" * 64), f.build())[1])

    def writable(f: Fixture) -> Any:
        (f.root / f.identity["profile"]["path"]).chmod(0o644)
        return f.build()
    run_case("M18_writable_input", "REJECT", writable)

    def hardlink(f: Fixture) -> Any:
        original = f.root / f.identity["profile"]["path"]
        link = f.root / "inputs/profile_hardlink.dat"; os.link(original, link)
        f.identity["profile"]["path"] = link.relative_to(f.root).as_posix()
        return f.build()
    run_case("M19_hard_link_input", "REJECT", hardlink)

    def direct_symlink(f: Fixture) -> Any:
        original = f.root / f.identity["profile"]["path"]
        target = f.root / "inputs/profile_target.dat"
        shutil.copyfile(original, target); target.chmod(0o444)
        original.unlink(); original.symlink_to(target.name)
        f.identity["profile"]["sha256"] = digest(target)
        return f.build()
    run_case("M20_direct_symlink_input", "REJECT", direct_symlink)

    def duplicate_json(f: Fixture) -> Any:
        M.load_json(f.config_path(raw='{"schema":1,"schema":2}'))
    run_case("M21_duplicate_json", "REJECT", duplicate_json)

    def nan_json(f: Fixture) -> Any:
        M.load_json(f.config_path(raw='{"x":NaN}'))
    run_case("M22_nonfinite_json", "REJECT", nan_json)

    def o_excl_cli(f: Fixture) -> Any:
        config = f.config_path(); output = f.root / "out.json"
        command = [sys.executable, str(SCRIPT), "--config", str(config),
                   "--workspace-root", str(f.root), "--output", str(output)]
        first = subprocess.run(command, text=True, capture_output=True)
        second = subprocess.run(command, text=True, capture_output=True)
        if first.returncode != 0 or second.returncode == 0:
            raise M.CompileError("O_EXCL no-replace behavior missing")
        return {"first": first.returncode, "second": second.returncode,
                "mode": oct(output.stat().st_mode & 0o777)}
    run_case("M23_o_excl_no_replace_runtime", "ACCEPT", o_excl_cli)

    # P0 exploit: every claimed authority below is fabricated inside this
    # temporary directory, yet status alone upgrades the output to production.
    def self_forged_production(f: Fixture) -> Any:
        f.config["status"] = "PRODUCTION_CANDIDATE"
        result = f.build()
        if result["status"] != "PASS_PRODUCTION_CANDIDATE_UNHAMMERED":
            raise M.CompileError("exploit did not reach production status")
        return {"status": result["status"], "identity_bytes": "self-forged",
                "rows": len(result["rows"])}
    run_case("X01_self_forged_json_production_candidate", "ACCEPT", self_forged_production)

    # P0 exploit: each row has a different per-point numerator, but weighted
    # totals are equal and therefore pass the sole denominator check.
    def weighted_numerator_cancellation(f: Fixture) -> Any:
        for index, row_id in enumerate(M.ROWS):
            spec = f.branches["decoder"][row_id]
            def mutation(payload: dict[str, Any], target=index) -> None:
                key = sorted(payload["population"])[target]
                payload["population"][key]["fixed_numerator"] = 2
            f.mutate_json_spec(spec, mutation)
        result = f.build()
        per_key = {row["row_id"]: row["per_population"][0]["charge"]["fixed_numerator"]
                   for row in result["rows"]}
        if len(set(per_key.values())) == 1:
            raise M.CompileError("exploit failed to create per-point mismatch")
        return per_key
    run_case("X02_per_point_numerator_mismatch_weighted_cancel", "ACCEPT",
             weighted_numerator_cancellation)

    # P0 exploit: row-specific energy rates are applied to common work, so a
    # self-authored rate table can manufacture a near-100% energy reduction.
    def unfair_weighted_energy(f: Fixture) -> Any:
        def mutation(payload: dict[str, Any]) -> None:
            payload["logic_pj_per_cycle"] = {row: (1e6 if row == "B0" else 1e-9)
                                              for row in M.ROWS}
            payload["dram_pj_per_byte"] = {"read": 1e-12, "write": 1e-12}
            payload["sram_pj_per_byte"] = {macro: {"read": 1e-12, "write": 1e-12}
                                            for macro in M.SRAM_MACROS}
        f.mutate_json_spec(f.energy, mutation)
        result = f.build(); reduction = result["rows"][-1]["energy_reduction_vs_B0"]
        if reduction < 0.999:
            raise M.CompileError("energy exploit did not dominate")
        return reduction
    run_case("X03_row_specific_rate_charges_common_energy_unfairly", "ACCEPT",
             unfair_weighted_energy)

    # P1 exploit: resolve_spec rejects a leaf symlink but accepts a regular leaf
    # reached through a symlinked parent inside the workspace.
    def parent_symlink(f: Fixture) -> Any:
        alias = f.root / "alias"; alias.symlink_to("inputs", target_is_directory=True)
        f.identity["profile"]["path"] = "alias/identity_profile.dat"
        result = f.build(); return result["status"]
    run_case("X04_parent_directory_symlink", "ACCEPT", parent_symlink)

    # P0 exploit: weights/strata/sequence identities are not pinned to an
    # authority.  Any three names and any positive weights summing to one pass.
    def population_cherry_pick(f: Fixture) -> Any:
        for index, point in enumerate(f.config["population"]):
            point["sequence_id"] = "invented_%d" % (index // 10)
            point["density_stratum"] = ("low", "medium", "high")[index % 3]
            point["weight"] = 0.5 if index == 0 else 0.5 / 29.0
        # Charge file keys must be forged to match the new population.
        specs = list(f.common.values()) + [spec for branch in f.branches.values()
                                           for spec in branch.values()]
        for spec in specs:
            def mutation(payload: dict[str, Any]) -> None:
                old = list(payload["population"].values())
                payload["population"] = {
                    M.population_key(point["sequence_id"], point["sample_id"]): old[i]
                    for i, point in enumerate(f.config["population"])}
            f.mutate_json_spec(spec, mutation)
        result = f.build(); return result["population"]
    run_case("X05_unpinned_population_weights_and_strata", "ACCEPT", population_cherry_pick)

    # P0 exploit: presence of all fields/macros is treated as accounting even
    # when every SRAM and DRAM access is self-reported as zero.
    def zero_memory_traffic(f: Fixture) -> Any:
        specs = list(f.common.values()) + [spec for branch in f.branches.values()
                                           for spec in branch.values()]
        for spec in specs:
            def mutation(payload: dict[str, Any]) -> None:
                for charge in payload["population"].values():
                    charge["dram_read_bytes"] = 0; charge["dram_write_bytes"] = 0
                    for access in charge["sram_bytes"].values():
                        access["read_bytes"] = 0; access["write_bytes"] = 0
            f.mutate_json_spec(spec, mutation)
        f.config["status"] = "PRODUCTION_CANDIDATE"
        result = f.build()
        ours = result["rows"][-1]["aggregate"]
        if ours["dram_read_bytes"] != 0:
            raise M.CompileError("zero-traffic exploit failed")
        return {"status": result["status"], "dram": 0,
                "sram": sum(v["read_bytes"] + v["write_bytes"]
                            for v in ours["sram_bytes"].values())}
    run_case("X06_zero_sram_dram_self_report_production", "ACCEPT", zero_memory_traffic)

    source = SCRIPT.read_text()
    invariant_ok = ("os.O_EXCL" in source and
                    "os.open(str(args.output), flags, 0o444)" in source)
    mutant = source.replace(" | os.O_EXCL", "", 1)
    mutant_caught = not ("os.O_EXCL" in mutant and
                          "os.open(str(args.output), flags, 0o444)" in mutant)
    results.append({"name": "M24_o_excl_source_replacement", "expected": "REJECT",
                    "observed": "REJECT" if invariant_ok and mutant_caught else "ACCEPT",
                    "test_pass": invariant_ok and mutant_caught,
                    "detail": "independent static publish invariant"})

    failed_tests = [row for row in results if not row["test_pass"]]
    exploits = [row for row in results if row["name"].startswith("X")
                and row["observed"] == "ACCEPT"]
    report = {"schema": "m1341.m1340.blind_hammer.r1",
              "status": "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED",
              "mutation_count": len(results),
              "mechanical_failures": len(failed_tests),
              "accepted_exploits": [row["name"] for row in exploits],
              "results": results,
              "production_candidate_self_forge_p0":
                  "X01_self_forged_json_production_candidate" in
                  [row["name"] for row in exploits]}
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if not failed_tests and len(exploits) >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
