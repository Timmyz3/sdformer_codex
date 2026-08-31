#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Authority-bound additive successor to the M1340 common-charge compiler.

M1340 remains frozen and supplies its strict structural accounting gates.  This
successor adds provenance, per-population denominator equality, partitioned
energy fairness, immutable ancestry, and transaction conservation.  The
production allowlist is intentionally empty in this source-only milestone;
production cannot run until an additive release pins every exact authority.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Iterable


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
HW = REPO / "hw_autoresearch_nts07"
M1340_PATH = HW / "system_simulator/scripts/build_m1340_ep34_table_a_common_charge_compiler.py"
M1340_SHA256 = "9cbf2262d2f391754ffff2eb77d4d7798d28c535f1c9b59fe6262e4702c52d54"
SPEC = importlib.util.spec_from_file_location("m1342_frozen_m1340", M1340_PATH)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)

SCHEMA = "m1342.ep34.table_a.authority.compiler.r1"
POPULATION_SCHEMA = "m1342.table_a.population_manifest.r1"
TRANSACTION_SCHEMA = "m1342.table_a.transaction_receipt.r1"
PARTITIONED_ENERGY_SCHEMA = "m1342.table_a.partitioned_energy.r1"
AUTHORITY_SCHEMA = "m1342.table_a.authority.review.r1"
ROLES = ("final_identity", "population_manifest", "charge_producer",
         "energy_producer", "transaction_receipt")
PRODUCTION_AUTHORITY_ALLOWLIST: dict[str, dict[str, str]] = {}
PRODUCTION_ALLOWLIST_STATE = "SOURCE_ONLY_UNPOPULATED__ADDITIVE_RELEASE_REQUIRED"
STATIC_PARENT_AUTHORITIES = {
    "m1336_readiness": {
        "root": HW / "reviews/m1336_strong_accept_table_a_post_ep34_capture_readiness_readonly_audit_r1_20260831",
        "review": "c8ada414c63a378b3b996614d14040271b46047803689a53216c25e9c214c60a",
        "manifest": "1d778ed9fc9cf0aa866f9a9452d0571d5fcfaaa8669692973b5fac2394dfd064",
        "outer": "8e11f6b87abfd3cae2660b9e31fbcc281f13c9c899a906c452c59642b8bd3e8b",
    },
    "m1341_failure": {
        "root": HW / "reviews/m1341_m1340_table_a_common_charge_compiler_source_blind_hammer_r1_20260831",
        "review": "823afc06b63cc24548136bb36f52f79967125a7e559eb113613a49ebe14b2844",
        "manifest": "fb77a27e122a55d71ce4b9188b370cd4799c7442d9e5d966ac257d357f915e2e",
        "outer": "2822e7e7e1f357f8474fd44127d7fbdcae39a3dcd055a32f755ceadb3eb46d75",
    },
}


class CompileError(ValueError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise CompileError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    return M.load_json(path)


def exact_object(value: Any, fields: Iterable[str], label: str) -> dict[str, Any]:
    return M.exact_object(value, fields, label)


def no_symlink_ancestry(root: Path, path: Path, leaf_must_exist: bool = True) -> Path:
    root_abs = root.absolute()
    path_abs = path if path.is_absolute() else root_abs / path
    try:
        relative = path_abs.relative_to(root_abs)
    except ValueError:
        raise CompileError("path escapes workspace: %s" % path)
    current = root_abs
    chain = [current]
    for part in relative.parts:
        current = current / part
        chain.append(current)
    for index, member in enumerate(chain):
        if not member.exists() and index == len(chain) - 1 and not leaf_must_exist:
            continue
        try:
            mode = member.lstat().st_mode
        except OSError as exc:
            raise CompileError("path ancestry missing %s: %s" % (member, exc))
        if stat.S_ISLNK(mode):
            raise CompileError("symlink path component forbidden: %s" % member)
    return path_abs


def regular_readonly_single(root: Path, path: Path, label: str) -> Path:
    target = no_symlink_ancestry(root, path)
    mode = target.lstat()
    require(stat.S_ISREG(mode.st_mode) and mode.st_nlink == 1,
            "%s must be single-link regular" % label)
    require((mode.st_mode & 0o222) == 0, "%s must be read-only" % label)
    return target


def parse_manifest(root: Path, expected_sha: str) -> dict[str, str]:
    manifest = regular_readonly_single(root, root / "SHA256SUMS", "authority manifest")
    require(sha256(manifest) == expected_sha, "authority manifest SHA drift")
    rows: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, "malformed authority manifest row")
        digest, name = fields; name = name.lstrip("*")
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts and name not in rows,
                "unsafe authority manifest member")
        member = regular_readonly_single(root, root / rel, "authority member")
        require(sha256(member) == digest, "authority member SHA drift: %s" % name)
        rows[name] = digest
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in
              {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(rows), "authority recursive population drift")
    return rows


def verify_simple_static_seal(spec: dict[str, Any]) -> None:
    root = spec["root"]
    require(root.is_dir() and not root.is_symlink(), "static authority root invalid")
    require(sha256(root / "SHA256SUMS") == spec["manifest"],
            "static manifest drift")
    require(sha256(root / "SHA256SUMS.seal.sha256") == spec["outer"],
            "static outer drift")
    require((root / "SHA256SUMS.seal.sha256").read_text().split() ==
            [spec["manifest"], "SHA256SUMS"], "static outer semantic drift")
    rows = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        member = root / name
        require(member.is_file() and not member.is_symlink()
                and sha256(member) == digest, "static sealed member drift")
        rows[name] = digest
    require(rows.get("review.json") == spec["review"], "static review identity drift")


def verify_static_parents() -> None:
    require(sha256(M1340_PATH) == M1340_SHA256, "frozen M1340 source drift")
    for spec in STATIC_PARENT_AUTHORITIES.values():
        verify_simple_static_seal(spec)
    readiness = strict_json(STATIC_PARENT_AUTHORITIES["m1336_readiness"]["root"] /
                            "review.json")
    failure = strict_json(STATIC_PARENT_AUTHORITIES["m1341_failure"]["root"] /
                          "review.json")
    require(readiness["status"] ==
            "PASS_READONLY_AUDIT__TABLE_A_NOT_READY__NO_NEW_MECHANISM",
            "M1336 semantic status drift")
    require(failure["status"] == "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED"
            and failure["production_candidate_self_forge_p0"] is True,
            "M1341 semantic status drift")


def verify_authority(root: Path, role: str, root_rel: str,
                     allowlist: dict[str, dict[str, str]],
                     production: bool) -> dict[str, Any]:
    require(role in allowlist, "authority role not code-pinned: %s" % role)
    allowed = allowlist[role]
    require(set(allowed) == {"root", "review_sha256", "manifest_sha256",
                             "outer_file_sha256", "producer_sha256", "tool_sha256"},
            "allowlist entry fields drift")
    require(root_rel == allowed["root"], "authority root is not allowlisted")
    authority_root = no_symlink_ancestry(root, Path(root_rel))
    require(authority_root.is_dir() and not authority_root.is_symlink(),
            "authority root invalid")
    outer = regular_readonly_single(authority_root,
                                    authority_root / "SHA256SUMS.seal.sha256",
                                    "authority outer seal")
    require(sha256(outer) == allowed["outer_file_sha256"],
            "authority outer SHA drift")
    require(outer.read_text().split() == [allowed["manifest_sha256"], "SHA256SUMS"],
            "authority outer semantic drift")
    rows = parse_manifest(authority_root, allowed["manifest_sha256"])
    require(rows.get("review.json") == allowed["review_sha256"],
            "authority review manifest identity drift")
    review = strict_json(authority_root / "review.json")
    review = exact_object(review, ("schema", "role", "status", "identity",
                                   "producer", "tool", "payloads", "claim_boundary"),
                          "authority review")
    required_status = ("ADMITTED_PRODUCTION_AUTHORITY" if production
                       else "ADMITTED_SOURCE_FIXTURE_AUTHORITY")
    require(review["schema"] == AUTHORITY_SCHEMA and review["role"] == role
            and review["status"] == required_status
            and review["identity"] == "Motion-C12-ep34-final",
            "authority semantic identity/status mismatch")
    required_claim = {"production_admitted": production,
                      "source_fixture_only": not production,
                      "independent_hammer_pass": True}
    require(review["claim_boundary"] == required_claim,
            "authority claim boundary drift")
    producer = exact_object(review["producer"], ("member", "sha256"), "producer")
    tool = exact_object(review["tool"], ("member", "sha256"), "tool")
    require(producer["sha256"] == allowed["producer_sha256"]
            and tool["sha256"] == allowed["tool_sha256"],
            "producer/tool allowlist drift")
    for label, item in (("producer", producer), ("tool", tool)):
        require(rows.get(item["member"]) == item["sha256"],
                "%s is not sealed" % label)
    payloads = review["payloads"]
    require(isinstance(payloads, dict) and payloads, "authority payloads missing")
    normalized = {}
    for name, raw in payloads.items():
        item = exact_object(raw, ("member", "sha256", "media_type"),
                            "authority payload")
        require(rows.get(item["member"]) == item["sha256"],
                "authority payload is not sealed: %s" % name)
        normalized[name] = {"path": (Path(root_rel) / item["member"]).as_posix(),
                            "sha256": item["sha256"],
                            "media_type": item["media_type"]}
    return {"review_sha256": allowed["review_sha256"],
            "manifest_sha256": allowed["manifest_sha256"],
            "outer_file_sha256": allowed["outer_file_sha256"],
            "producer_sha256": allowed["producer_sha256"],
            "tool_sha256": allowed["tool_sha256"], "payloads": normalized}


def spec_equal(left: Any, right: Any, label: str) -> None:
    require(left == right, "%s is not authority-bound" % label)


def validate_population_manifest(path: Path, base_population: Any) -> list[dict[str, Any]]:
    payload = exact_object(strict_json(path), ("schema", "identity", "points"),
                           "population manifest")
    require(payload["schema"] == POPULATION_SCHEMA
            and payload["identity"] == "Motion-C12-ep34-final",
            "population manifest identity mismatch")
    points, _ = M.validate_population(payload["points"])
    canonical = [{key: point[key] for key in
                  ("sequence_id", "sample_id", "density_stratum", "weight")}
                 for point in points]
    require(base_population == canonical,
            "base population differs from sealed population manifest")
    return points


def validate_partitioned_energy(path: Path, m1340_energy: dict[str, Any]) -> dict[str, Any]:
    payload = exact_object(strict_json(path),
        ("schema", "identity", "native_mapped_activity_coverage",
         "common_logic_pj_per_cycle", "direct_logic_pj_per_cycle",
         "dram_pj_per_byte", "sram_pj_per_byte"), "partitioned energy")
    require(payload["schema"] == PARTITIONED_ENERGY_SCHEMA
            and payload["identity"] == "Motion-C12-ep34-final",
            "partitioned energy identity mismatch")
    coverage = M.finite_nonnegative_number(
        payload["native_mapped_activity_coverage"], "coverage")
    require(0.95 <= coverage <= 1.0, "partitioned coverage below 95 percent")
    common_rate = M.finite_nonnegative_number(
        payload["common_logic_pj_per_cycle"], "common logic rate", positive=True)
    direct_raw = payload["direct_logic_pj_per_cycle"]
    require(isinstance(direct_raw, dict) and set(direct_raw) == set(M.DIRECT_BRANCHES),
            "direct energy branch set differs")
    direct = {}
    for branch in M.DIRECT_BRANCHES:
        require(isinstance(direct_raw[branch], dict)
                and set(direct_raw[branch]) == set(M.ROWS),
                "direct energy row set differs")
        direct[branch] = {row: M.finite_nonnegative_number(
            direct_raw[branch][row], "direct logic rate", positive=True)
            for row in M.ROWS}
    dram_raw = exact_object(payload["dram_pj_per_byte"], ("read", "write"),
                            "partitioned DRAM rate")
    dram = {key: M.finite_nonnegative_number(value, "DRAM rate", positive=True)
            for key, value in dram_raw.items()}
    sram_raw = payload["sram_pj_per_byte"]
    require(isinstance(sram_raw, dict) and set(sram_raw) == set(M.SRAM_MACROS),
            "partitioned energy missing SRAM macro")
    sram = {}
    for macro in M.SRAM_MACROS:
        rates = exact_object(sram_raw[macro], ("read", "write"), "SRAM rate")
        sram[macro] = {key: M.finite_nonnegative_number(value, "SRAM rate", positive=True)
                       for key, value in rates.items()}
    # The compatibility authority consumed by frozen M1340 is not allowed to
    # smuggle a row-dependent rate onto common work.
    require(all(math.isclose(m1340_energy["logic"][row], common_rate,
                             rel_tol=0.0, abs_tol=0.0) for row in M.ROWS),
            "M1340 compatibility logic rates are not row invariant")
    require(m1340_energy["dram"] == dram and m1340_energy["sram"] == sram,
            "M1340 compatibility memory rates differ")
    return {"coverage": coverage, "common_logic": common_rate,
            "direct_logic": direct, "dram": dram, "sram": sram}


def memory_energy(charge: dict[str, Any], energy: dict[str, Any]) -> float:
    value = (charge["dram_read_bytes"] * energy["dram"]["read"]
             + charge["dram_write_bytes"] * energy["dram"]["write"])
    for macro in M.SRAM_MACROS:
        value += (charge["sram_bytes"][macro]["read_bytes"]
                  * energy["sram"][macro]["read"]
                  + charge["sram_bytes"][macro]["write_bytes"]
                  * energy["sram"][macro]["write"])
    return value


def validate_transactions(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = exact_object(strict_json(path),
                           ("schema", "identity", "address_trace_sha256", "rows"),
                           "transaction receipt")
    require(payload["schema"] == TRANSACTION_SCHEMA
            and payload["identity"] == "Motion-C12-ep34-final",
            "transaction receipt identity mismatch")
    trace_sha = payload["address_trace_sha256"]
    require(isinstance(trace_sha, str) and len(trace_sha) == 64,
            "address trace SHA grammar invalid")
    receipt_rows = payload["rows"]
    require(isinstance(receipt_rows, dict) and set(receipt_rows) == set(M.ROWS),
            "transaction receipt row set differs")
    all_memory = 0
    for row in rows:
        row_id = row["row_id"]
        expected = {point["key"]: point["charge"] for point in row["per_population"]}
        actual = receipt_rows[row_id]
        require(isinstance(actual, dict) and set(actual) == set(expected),
                "transaction population coverage differs")
        for key, charge in expected.items():
            receipt = M.validate_charge(actual[key], "transaction[%s][%s]" % (row_id, key))
            require(receipt == charge, "transaction conservation mismatch: %s %s" %
                    (row_id, key))
            all_memory += receipt["dram_read_bytes"] + receipt["dram_write_bytes"]
            all_memory += sum(access["read_bytes"] + access["write_bytes"]
                              for access in receipt["sram_bytes"].values())
    require(all_memory > 0, "all-zero SRAM/DRAM production accounting forbidden")
    return trace_sha


def split_charges(root: Path, base: dict[str, Any], points: list[dict[str, Any]]) -> tuple[Any, Any]:
    keys = {point["key"] for point in points}
    common = {name: M.read_charge_file(root, base["common_operators"][name],
                                      "common", name, keys)
              for name in M.COMMON_CATEGORIES}
    branches = {branch: {row: M.read_charge_file(
        root, base["direct_branches"][branch][row], "direct",
        "%s.%s" % (branch, row), keys) for row in M.ROWS}
        for branch in M.DIRECT_BRANCHES}
    return common, branches


def fair_energy_rows(points: list[dict[str, Any]], common: Any, branches: Any,
                     energy: dict[str, Any]) -> dict[str, Any]:
    common_weighted = 0.0
    common_by_key = {}
    for point in points:
        charge = M.new_charge()
        for category in M.COMMON_CATEGORIES:
            M.add_charge(charge, common[category][point["key"]])
        value = charge["cycles"] * energy["common_logic"] + memory_energy(charge, energy)
        common_by_key[point["key"]] = value
        common_weighted += point["weight"] * value
    rows = {}
    for row_id in M.ROWS:
        direct_weighted = {branch: 0.0 for branch in M.DIRECT_BRANCHES}
        for point in points:
            for branch in M.DIRECT_BRANCHES:
                charge = branches[branch][row_id][point["key"]]
                value = (charge["cycles"] * energy["direct_logic"][branch][row_id]
                         + memory_energy(charge, energy))
                direct_weighted[branch] += point["weight"] * value
        total = common_weighted + sum(direct_weighted.values())
        rows[row_id] = {"common_weighted_pj": common_weighted,
                        "direct_weighted_pj": direct_weighted,
                        "total_weighted_pj": total}
    return rows


def build(config_path: Path, workspace_root: Path,
          fixture_allowlist: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    verify_static_parents()
    root = workspace_root.absolute()
    config_path = regular_readonly_single(root, config_path, "M1342 config")
    config = exact_object(strict_json(config_path),
        ("schema", "status", "base_config", "authority_roots", "claim_boundary"),
        "M1342 config")
    require(config["schema"] == SCHEMA and config["status"] in
            ("SOURCE_FIXTURE", "PRODUCTION_CANDIDATE"), "M1342 schema/status mismatch")
    production = config["status"] == "PRODUCTION_CANDIDATE"
    if production:
        require(set(PRODUCTION_AUTHORITY_ALLOWLIST) == set(ROLES),
                "production authority allowlist is not populated")
        allowlist = PRODUCTION_AUTHORITY_ALLOWLIST
    else:
        require(fixture_allowlist is not None and set(fixture_allowlist) == set(ROLES),
                "source fixture requires explicit non-production allowlist")
        allowlist = fixture_allowlist
    roots = config["authority_roots"]
    require(isinstance(roots, dict) and set(roots) == set(ROLES),
            "authority role set differs")
    authorities = {role: verify_authority(root, role, roots[role], allowlist,
                                          production) for role in ROLES}
    base_spec = exact_object(config["base_config"], ("path", "sha256", "media_type"),
                             "base config spec")
    spec_equal(base_spec, authorities["final_identity"]["payloads"].get("base_config"),
               "base config")
    base_path = regular_readonly_single(root, Path(base_spec["path"]), "base config")
    require(sha256(base_path) == base_spec["sha256"], "base config SHA drift")
    base = strict_json(base_path)
    require(base["status"] == "SOURCE_FIXTURE",
            "frozen M1340 must run only in source-fixture mode under M1342")
    for field in ("checkpoint", "config", "profile", "capture_result",
                  "capture_result_hammer"):
        require(isinstance(base.get("identity"), dict) and field in base["identity"],
                "base identity field missing: %s" % field)
        spec_equal(base["identity"][field],
                   authorities["final_identity"]["payloads"].get(field),
                   "identity.%s" % field)
    population_spec = authorities["population_manifest"]["payloads"].get(
        "population_manifest")
    require(population_spec is not None, "population manifest payload missing")
    population_path = regular_readonly_single(root, Path(population_spec["path"]),
                                               "population manifest")
    points = validate_population_manifest(population_path, base["population"])
    require(isinstance(base.get("common_operators"), dict)
            and set(base["common_operators"]) == set(M.COMMON_CATEGORIES),
            "base common operator categories differ")
    for category in M.COMMON_CATEGORIES:
        spec_equal(base["common_operators"][category],
                   authorities["charge_producer"]["payloads"].get(
                       "common:%s" % category), "common charge %s" % category)
    require(isinstance(base.get("direct_branches"), dict)
            and set(base["direct_branches"]) == set(M.DIRECT_BRANCHES),
            "base direct branch set differs")
    for branch in M.DIRECT_BRANCHES:
        require(isinstance(base["direct_branches"][branch], dict)
                and set(base["direct_branches"][branch]) == set(M.ROWS),
                "base direct branch rows differ: %s" % branch)
        for row in M.ROWS:
            spec_equal(base["direct_branches"][branch][row],
                       authorities["charge_producer"]["payloads"].get(
                           "direct:%s:%s" % (branch, row)),
                       "direct charge %s.%s" % (branch, row))
    spec_equal(base["energy_authority"],
               authorities["energy_producer"]["payloads"].get("m1340_energy"),
               "M1340 compatibility energy")
    base_result = M.build(base_path, root)
    # Per-population equality, not merely equality after weighted aggregation.
    by_row = {row["row_id"]: {point["key"]: point["charge"]["fixed_numerator"]
                              for point in row["per_population"]}
              for row in base_result["rows"]}
    for key in sorted(by_row["B0"]):
        require(len({by_row[row][key] for row in M.ROWS}) == 1,
                "fixed numerator differs at population key %s" % key)
    transaction_spec = authorities["transaction_receipt"]["payloads"].get(
        "transaction_receipt")
    require(transaction_spec is not None, "transaction receipt payload missing")
    trace_sha = validate_transactions(regular_readonly_single(
        root, Path(transaction_spec["path"]), "transaction receipt"),
        base_result["rows"])
    m1340_energy = M.validate_energy(root, base["energy_authority"])
    partitioned_spec = authorities["energy_producer"]["payloads"].get(
        "partitioned_energy")
    require(partitioned_spec is not None, "partitioned energy payload missing")
    energy = validate_partitioned_energy(regular_readonly_single(
        root, Path(partitioned_spec["path"]), "partitioned energy"), m1340_energy)
    common, branches = split_charges(root, base, points)
    fair = fair_energy_rows(points, common, branches, energy)
    output_rows = copy.deepcopy(base_result["rows"])
    b0_energy = fair["B0"]["total_weighted_pj"]
    for row in output_rows:
        row.pop("aggregate_energy", None); row.pop("weighted_energy", None)
        row["energy_split"] = fair[row["row_id"]]
        row["energy_reduction_vs_B0"] = (1.0 -
            fair[row["row_id"]]["total_weighted_pj"] / b0_energy)
    require(len({fair[row]["common_weighted_pj"] for row in M.ROWS}) == 1,
            "common weighted energy differs across rows")
    expected_claim = {"same_denominator": True,
        "per_population_numerator_equal": True,
        "common_energy_row_invariant": True,
        "transaction_conservation": True,
        "independent_hammer_required": True,
        "paper_headline_admitted": False}
    require(config["claim_boundary"] == expected_claim, "M1342 claim boundary drift")
    authority_digests = {role: {key: value for key, value in authority.items()
        if key != "payloads"} for role, authority in authorities.items()}
    return {"schema": "m1342.ep34.table_a.authority.output.r1",
        "status": ("PASS_PRODUCTION_CANDIDATE_UNHAMMERED" if production else
                   "PASS_SOURCE_FIXTURE_NOT_PRODUCTION"),
        "production_allowlist_state": PRODUCTION_ALLOWLIST_STATE,
        "identity": "Motion-C12-ep34-final",
        "config_sha256": sha256(config_path),
        "base_config_sha256": sha256(base_path),
        "m1340_source_sha256": M1340_SHA256,
        "m1342_source_sha256": sha256(SCRIPT),
        "authority_digests": authority_digests,
        "population_manifest_sha256": population_spec["sha256"],
        "address_trace_sha256": trace_sha,
        "resource": base_result["resource"], "rows": output_rows,
        "claim_boundary": {**expected_claim,
            "paper_headline_admitted": False,
            "requires_fresh_independent_bundle_hammer": True}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(args.config, args.workspace_root)
        output = no_symlink_ancestry(args.workspace_root.absolute(), args.output,
                                     leaf_must_exist=False)
        require(output.parent.is_dir(), "output parent must already exist")
        encoded = json.dumps(result, ensure_ascii=False, sort_keys=True,
                             indent=2, allow_nan=False) + "\n"
        descriptor = os.open(str(output), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
        print("M1342_AUTHORITY_COMPILER_PASS status=%s rows=6 headline=false" %
              result["status"])
        return 0
    except (CompileError, M.CompileError, OSError, ValueError) as exc:
        print("M1342_AUTHORITY_COMPILER_FAIL: %s" % exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
