#!/usr/bin/env python3
"""M722-r2 preflight repair: exclude only root seals from population.

The frozen r1 model is reused without arithmetic, cycle, storage or decision
changes.  r1 incorrectly excluded every nested file named ``SHA256SUMS`` from
the actual population; M686 intentionally contains nested sealed weight and
runtime-receipt directories.  This wrapper replaces only the directory-seal
checker and then invokes r1.
"""

import importlib.util
from pathlib import Path


R1_PATH = Path(__file__).resolve().with_name(
    "analyze_m722_lb_fuse_decoder_cpu_fastkill.py")
SPEC = importlib.util.spec_from_file_location("m722_r1", R1_PATH)
R1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R1)
R1_STRICT_JSON = R1.strict_json


def verify_directory(path):
    path = Path(path)
    R1.require(path.is_dir() and not path.is_symlink(),
               "bad sealed directory")
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    R1.require(manifest.is_file() and not manifest.is_symlink() and
               outer.is_file() and not outer.is_symlink(), "missing seals")
    expected_names = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        R1.require(len(fields) == 2 and len(fields[0]) == 64,
                   "malformed manifest")
        expected, name = fields
        R1.require(name not in expected_names, "duplicate sealed member")
        expected_names.add(name)
        member = path.joinpath(*R1.safe_member(name).parts)
        R1.require(member.is_file() and not member.is_symlink() and
                   R1.sha256(member) == expected,
                   "sealed member mismatch: " + name)
    root_seals = {manifest.resolve(), outer.resolve()}
    actual_names = set()
    for member in path.rglob("*"):
        R1.require(not member.is_symlink(), "symlink in sealed directory")
        if member.is_file() and member.resolve() not in root_seals:
            actual_names.add(member.relative_to(path).as_posix())
    R1.require(actual_names == expected_names, "sealed population mismatch")
    fields = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    R1.require(fields == [R1.sha256(manifest), "SHA256SUMS"],
               "outer seal mismatch")
    return {"manifest_sha256": R1.sha256(manifest),
            "outer_seal_file_sha256": R1.sha256(outer)}


def strict_json_with_r2_overlay(path):
    """Expand the r2 repair overlay onto the exact frozen r1 contract."""
    path = Path(path)
    value = R1_STRICT_JSON(path)
    if path.name != "m722r2_lb_fuse_decoder_cpu_fastkill_contract_r1_20260828.json":
        return value
    R1.require(value.get("schema") == R1.CONTRACT_SCHEMA,
               "r2 overlay schema")
    base = path.with_name(
        "m722_lb_fuse_decoder_cpu_fastkill_contract_r1_20260828.json")
    R1.require(R1.sha256(base) ==
               value.get("base_contract", {}).get("sha256"),
               "r1 base contract identity")
    merged = R1_STRICT_JSON(base)
    R1.require(merged.get("schema") == R1.CONTRACT_SCHEMA,
               "r1 base schema")
    merged["status"] = value["status"]
    merged["r2_overlay"] = value
    return merged


R1.verify_directory = verify_directory
R1.strict_json = strict_json_with_r2_overlay
# Make the canonical receipt bind this r2 wrapper while retaining the r1 model
# as an explicitly frozen dependency in the r2 contract.
R1.__file__ = __file__


if __name__ == "__main__":
    raise SystemExit(R1.main())
