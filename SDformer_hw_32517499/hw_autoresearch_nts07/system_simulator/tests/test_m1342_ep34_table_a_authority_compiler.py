from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_m1342_ep34_table_a_authority_compiler.py"
SPEC = importlib.util.spec_from_file_location("m1342_authority", SCRIPT)
N = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(N)
M = N.M
REPO = Path(__file__).resolve().parents[3]
DOCS359 = REPO / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture:
    def __init__(self, numerator_cancel: bool = False,
                 unfair_energy: bool = False,
                 population_manifest_mismatch: bool = False,
                 zero_memory: bool = False,
                 transaction_mismatch: bool = False) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="m1342_test_")
        self.root = Path(self.temp.name)
        self.auth_root = self.root / "authorities"; self.auth_root.mkdir()
        (self.root / "docs").mkdir()
        docs = self.root / "docs/359_DATE终局冻结_20260813.md"
        docs.write_bytes(DOCS359.read_bytes()); docs.chmod(0o444)
        self.points = []
        for sequence, stratum in (("interlaken_01_a", "low"),
                                  ("thun_01_b", "medium"),
                                  ("zurich_city_12_a", "high")):
            for sample in range(10):
                self.points.append({"sequence_id": sequence, "sample_id": sample,
                    "density_stratum": stratum, "weight": 1.0 / 30.0})
        self.role_payloads = {role: {} for role in N.ROLES}
        self.role_dirs = {role: self.auth_root / role for role in N.ROLES}
        for directory in self.role_dirs.values(): directory.mkdir()

        identity_media = {"checkpoint": "application/octet-stream",
            "config": "application/yaml", "profile": "application/json",
            "capture_result": "application/json",
            "capture_result_hammer": "application/json"}
        identity_specs = {}
        for name, media in identity_media.items():
            identity_specs[name] = self.payload("final_identity", name + ".dat",
                ("fixture-%s" % name).encode(), media, name)

        manifest_points = copy.deepcopy(self.points)
        if population_manifest_mismatch:
            manifest_points[0]["weight"] = 0.04
            manifest_points[1]["weight"] -= 0.04 - 1.0 / 30.0
        population = {"schema": N.POPULATION_SCHEMA,
                      "identity": "Motion-C12-ep34-final",
                      "points": manifest_points}
        self.payload_json("population_manifest", "population.json", population,
                          "population_manifest")

        self.charge_specs = {"common": {}, "direct": {}}
        self.charge_data = {"common": {}, "direct": {}}
        for category in M.COMMON_CATEGORIES:
            spec, data = self.charge_payload("common", category, 10, zero_memory)
            self.charge_specs["common"][category] = spec
            self.charge_data["common"][category] = data
        row_cycles = {"B0": 100, "B1": 92, "B2": 88,
                      "B3": 84, "C2": 82, "Ours": 70}
        for branch_index, branch in enumerate(M.DIRECT_BRANCHES):
            self.charge_specs["direct"][branch] = {}
            self.charge_data["direct"][branch] = {}
            for row_index, (row, cycles) in enumerate(row_cycles.items()):
                offsets = ({row_index: 1} if numerator_cancel and branch == "decoder"
                           else {})
                spec, data = self.charge_payload("direct", "%s.%s" % (branch, row),
                    cycles + branch_index, zero_memory, offsets)
                self.charge_specs["direct"][branch][row] = spec
                self.charge_data["direct"][branch][row] = data

        common_rate = 2.0
        logic_rates = {row: (1000.0 if unfair_energy and row == "B0" else
                             0.001 if unfair_energy else common_rate)
                       for row in M.ROWS}
        dram = {"read": 3.7, "write": 3.9}
        sram = {macro: {"read": 0.2, "write": 0.3} for macro in M.SRAM_MACROS}
        m1340_energy = {"schema": M.ENERGY_SCHEMA,
            "identity": "Motion-C12-ep34-final",
            "native_mapped_activity_coverage": 0.96,
            "logic_pj_per_cycle": logic_rates,
            "dram_pj_per_byte": dram, "sram_pj_per_byte": sram}
        energy_spec = self.payload_json("energy_producer", "m1340_energy.json",
                                        m1340_energy, "m1340_energy")
        partitioned = {"schema": N.PARTITIONED_ENERGY_SCHEMA,
            "identity": "Motion-C12-ep34-final",
            "native_mapped_activity_coverage": 0.96,
            "common_logic_pj_per_cycle": common_rate,
            "direct_logic_pj_per_cycle": {
                branch: {row: 2.0 + index * 0.1 for index, row in enumerate(M.ROWS)}
                for branch in M.DIRECT_BRANCHES},
            "dram_pj_per_byte": dram, "sram_pj_per_byte": sram}
        self.payload_json("energy_producer", "partitioned_energy.json", partitioned,
                          "partitioned_energy")

        transaction = {"schema": N.TRANSACTION_SCHEMA,
            "identity": "Motion-C12-ep34-final",
            "address_trace_sha256": hashlib.sha256(b"fixture-address-trace").hexdigest(),
            "rows": {row: {} for row in M.ROWS}}
        for row in M.ROWS:
            for point in self.points:
                key = M.population_key(point["sequence_id"], point["sample_id"])
                total = M.new_charge()
                for category in M.COMMON_CATEGORIES:
                    M.add_charge(total, self.charge_data["common"][category][key])
                for branch in M.DIRECT_BRANCHES:
                    M.add_charge(total, self.charge_data["direct"][branch][row][key])
                transaction["rows"][row][key] = total
        if transaction_mismatch:
            first = next(iter(transaction["rows"]["Ours"].values()))
            first["dram_read_bytes"] += 1
        self.payload_json("transaction_receipt", "transaction.json", transaction,
                          "transaction_receipt")

        base = {"schema": M.SCHEMA, "status": "SOURCE_FIXTURE",
            "identity": {"name": "Motion-C12-ep34-final", **identity_specs},
            "resource": copy.deepcopy(M.RESOURCE), "population": self.points,
            "rows": list(M.ROWS),
            "common_operators": self.charge_specs["common"],
            "direct_branches": self.charge_specs["direct"],
            "energy_authority": energy_spec,
            "claim_boundary": {"same_denominator": True,
                "common_charge_identical_all_rows": True,
                "component_speedups_not_multiplied": True,
                "external_prosperity_not_ours": True,
                "independent_hammer_required": True,
                "paper_headline_admitted": False},
            "protected_file": {"path": "docs/359_DATE终局冻结_20260813.md",
                "sha256": M.PROTECTED_SHA256, "media_type": "text/markdown"}}
        base_spec = self.payload_json("final_identity", "base_config.json", base,
                                      "base_config")
        self.config = {"schema": N.SCHEMA, "status": "SOURCE_FIXTURE",
            "base_config": base_spec,
            "authority_roots": {role: self.role_dirs[role].relative_to(self.root).as_posix()
                                for role in N.ROLES},
            "claim_boundary": {"same_denominator": True,
                "per_population_numerator_equal": True,
                "common_energy_row_invariant": True,
                "transaction_conservation": True,
                "independent_hammer_required": True,
                "paper_headline_admitted": False}}
        self.allowlist = {}
        for role in N.ROLES:
            self.seal(role)

    def close(self) -> None:
        self.temp.cleanup()

    def payload(self, role: str, member: str, content: bytes,
                media: str, name: str) -> dict[str, str]:
        path = self.role_dirs[role] / member
        path.write_bytes(content); path.chmod(0o444)
        self.role_payloads[role][name] = {"member": member,
            "sha256": digest(path), "media_type": media}
        return {"path": path.relative_to(self.root).as_posix(),
                "sha256": digest(path), "media_type": media}

    def payload_json(self, role: str, member: str, payload: object,
                     name: str) -> dict[str, str]:
        return self.payload(role, member, (json.dumps(payload, sort_keys=True,
            separators=(",", ":"), allow_nan=False) + "\n").encode(),
            "application/json", name)

    def charge(self, cycles: int, numerator: int, zero_memory: bool) -> dict:
        return {"cycles": cycles, "fixed_numerator": numerator,
            "dram_read_bytes": 0 if zero_memory else cycles * 2,
            "dram_write_bytes": 0 if zero_memory else cycles,
            "sram_bytes": {macro: {"read_bytes": 0 if zero_memory else cycles + i,
                                    "write_bytes": 0 if zero_memory else i}
                           for i, macro in enumerate(M.SRAM_MACROS)}}

    def charge_payload(self, kind: str, name: str, cycles: int,
                       zero_memory: bool, numerator_offsets: dict[int, int] | None = None):
        payload = {"schema": M.CHARGE_SCHEMA, "kind": kind, "name": name,
                   "identity": "Motion-C12-ep34-final", "population": {}}
        numerator_offsets = numerator_offsets or {}
        data = {}
        for index, point in enumerate(self.points):
            key = M.population_key(point["sequence_id"], point["sample_id"])
            data[key] = self.charge(cycles + point["sample_id"],
                                    1 + numerator_offsets.get(index, 0), zero_memory)
            payload["population"][key] = data[key]
        label = ("common:%s" % name if kind == "common" else
                 "direct:%s" % name.replace(".", ":"))
        spec = self.payload_json("charge_producer",
            "charge_%s_%s.json" % (kind, name.replace(".", "_")), payload, label)
        return spec, data

    def seal(self, role: str) -> None:
        directory = self.role_dirs[role]
        for name in ("SHA256SUMS", "SHA256SUMS.seal.sha256", "review.json",
                     "producer.bin", "tool.bin"):
            path = directory / name
            if path.exists(): path.chmod(0o644); path.unlink()
        producer = directory / "producer.bin"; producer.write_bytes(("producer-" + role).encode())
        tool = directory / "tool.bin"; tool.write_bytes(("tool-" + role).encode())
        producer.chmod(0o444); tool.chmod(0o444)
        review = {"schema": N.AUTHORITY_SCHEMA, "role": role,
            "status": "ADMITTED_SOURCE_FIXTURE_AUTHORITY",
            "identity": "Motion-C12-ep34-final",
            "producer": {"member": "producer.bin", "sha256": digest(producer)},
            "tool": {"member": "tool.bin", "sha256": digest(tool)},
            "payloads": self.role_payloads[role],
            "claim_boundary": {"production_admitted": False,
                "source_fixture_only": True, "independent_hammer_pass": True}}
        review_path = directory / "review.json"
        review_path.write_text(json.dumps(review, sort_keys=True,
                                          separators=(",", ":")) + "\n")
        review_path.chmod(0o444)
        members = sorted(path for path in directory.iterdir() if path.is_file())
        manifest = directory / "SHA256SUMS"
        manifest.write_text("".join("%s  %s\n" % (digest(path), path.name)
                                    for path in members))
        manifest.chmod(0o444)
        outer = directory / "SHA256SUMS.seal.sha256"
        outer.write_text("%s  SHA256SUMS\n" % digest(manifest)); outer.chmod(0o444)
        self.allowlist[role] = {"root": directory.relative_to(self.root).as_posix(),
            "review_sha256": digest(review_path), "manifest_sha256": digest(manifest),
            "outer_file_sha256": digest(outer), "producer_sha256": digest(producer),
            "tool_sha256": digest(tool)}

    def config_path(self) -> Path:
        path = self.root / "m1342_config.json"
        if path.exists(): path.chmod(0o644); path.unlink()
        path.write_text(json.dumps(self.config), encoding="utf-8"); path.chmod(0o444)
        return path

    def build(self):
        return N.build(self.config_path(), self.root, self.allowlist)


class Tests(unittest.TestCase):
    def fixture(self, **kwargs):
        fixture = Fixture(**kwargs); self.addCleanup(fixture.close); return fixture

    def test_01_good_source_fixture_binds_authorities_and_fair_energy(self):
        fixture = self.fixture(); result = fixture.build()
        self.assertEqual(result["status"], "PASS_SOURCE_FIXTURE_NOT_PRODUCTION")
        self.assertEqual(set(result["authority_digests"]), set(N.ROLES))
        common = {row["energy_split"]["common_weighted_pj"] for row in result["rows"]}
        self.assertEqual(len(common), 1)
        self.assertEqual(result["population_manifest_sha256"],
                         fixture.role_payloads["population_manifest"]
                         ["population_manifest"]["sha256"])

    def test_02_production_self_forge_is_impossible_with_empty_code_allowlist(self):
        fixture = self.fixture(); fixture.config["status"] = "PRODUCTION_CANDIDATE"
        with self.assertRaisesRegex(N.CompileError, "allowlist is not populated"):
            fixture.build()

    def test_03_weighted_numerator_cancellation_is_rejected_per_key(self):
        fixture = self.fixture(numerator_cancel=True)
        with self.assertRaisesRegex(N.CompileError, "population key"):
            fixture.build()

    def test_04_row_specific_common_energy_rate_is_rejected(self):
        fixture = self.fixture(unfair_energy=True)
        with self.assertRaisesRegex(N.CompileError, "row invariant"):
            fixture.build()

    def test_05_population_substitution_differs_from_sealed_manifest(self):
        fixture = self.fixture(population_manifest_mismatch=True)
        with self.assertRaisesRegex(N.CompileError, "sealed population manifest"):
            fixture.build()

    def test_06_parent_directory_symlink_is_rejected(self):
        fixture = self.fixture()
        alias = fixture.root / "alias"; alias.symlink_to("authorities", target_is_directory=True)
        fixture.config["authority_roots"]["final_identity"] = "alias/final_identity"
        fixture.allowlist["final_identity"]["root"] = "alias/final_identity"
        with self.assertRaisesRegex(N.CompileError, "symlink path component"):
            fixture.build()

    def test_07_all_zero_memory_transactions_are_rejected(self):
        fixture = self.fixture(zero_memory=True)
        with self.assertRaisesRegex(N.CompileError, "all-zero"):
            fixture.build()

    def test_08_transaction_receipt_mismatch_is_rejected(self):
        fixture = self.fixture(transaction_mismatch=True)
        with self.assertRaisesRegex(N.CompileError, "conservation mismatch"):
            fixture.build()

    def test_09_missing_authority_role_is_rejected(self):
        fixture = self.fixture(); del fixture.config["authority_roots"]["energy_producer"]
        with self.assertRaisesRegex(N.CompileError, "role set"):
            fixture.build()

    def test_10_allowlist_sha_drift_is_rejected(self):
        fixture = self.fixture(); fixture.allowlist["charge_producer"]["review_sha256"] = "0" * 64
        with self.assertRaisesRegex(N.CompileError, "review manifest identity"):
            fixture.build()

    def test_11_outer_semantics_are_checked_not_only_outer_sha(self):
        fixture = self.fixture(); role = "energy_producer"
        outer = fixture.role_dirs[role] / "SHA256SUMS.seal.sha256"
        outer.chmod(0o644); outer.write_text("0" * 64 + "  SHA256SUMS\n"); outer.chmod(0o444)
        fixture.allowlist[role]["outer_file_sha256"] = digest(outer)
        with self.assertRaisesRegex(N.CompileError, "outer semantic"):
            fixture.build()

    def test_12_producer_sha_is_code_pinned(self):
        fixture = self.fixture(); fixture.allowlist["charge_producer"]["producer_sha256"] = "1" * 64
        with self.assertRaisesRegex(N.CompileError, "producer/tool"):
            fixture.build()

    def test_13_m1340_missing_common_gate_is_retained(self):
        fixture = self.fixture()
        base_path = fixture.role_dirs["final_identity"] / "base_config.json"
        base_path.chmod(0o644); base = json.loads(base_path.read_text())
        del base["common_operators"]["fc1"]
        base_path.write_text(json.dumps(base)); base_path.chmod(0o444)
        fixture.role_payloads["final_identity"]["base_config"]["sha256"] = digest(base_path)
        fixture.config["base_config"]["sha256"] = digest(base_path)
        fixture.seal("final_identity")
        with self.assertRaises((N.CompileError, M.CompileError)):
            fixture.build()

    def test_14_m1340_resource_gate_is_retained(self):
        fixture = self.fixture()
        base_path = fixture.role_dirs["final_identity"] / "base_config.json"
        base_path.chmod(0o644); base = json.loads(base_path.read_text())
        base["resource"]["group_fifo_depth"] = 5
        base_path.write_text(json.dumps(base)); base_path.chmod(0o444)
        fixture.role_payloads["final_identity"]["base_config"]["sha256"] = digest(base_path)
        fixture.config["base_config"]["sha256"] = digest(base_path)
        fixture.seal("final_identity")
        with self.assertRaises((N.CompileError, M.CompileError)):
            fixture.build()

    def test_15_output_contains_all_authority_and_config_digests(self):
        fixture = self.fixture(); result = fixture.build()
        self.assertEqual(len(result["config_sha256"]), 64)
        self.assertEqual(len(result["base_config_sha256"]), 64)
        self.assertEqual(len(result["m1342_source_sha256"]), 64)
        for authority in result["authority_digests"].values():
            self.assertEqual(set(authority), {"review_sha256", "manifest_sha256",
                "outer_file_sha256", "producer_sha256", "tool_sha256"})

    def test_16_claim_boundary_never_admits_headline(self):
        fixture = self.fixture(); result = fixture.build()
        self.assertFalse(result["claim_boundary"]["paper_headline_admitted"])
        self.assertTrue(result["claim_boundary"]["requires_fresh_independent_bundle_hammer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
