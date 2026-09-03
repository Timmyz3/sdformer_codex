#!/usr/bin/env python3
"""Independent, read-only M1833 identity/seal/source inventory audit."""
from __future__ import print_function

import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CONTRACT = HW / "contracts/m1831_m1830_m1811_c2_fresh_mapped_production_energy_source_contract_r1_20260902.json"
AUTHOR = HW / "reviews/m1831_m1830_m1811_c2_fresh_mapped_production_energy_source_author_receipt_r1_20260902"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key " + key)
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_file_seal(path):
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    need(path.is_file() and not path.is_symlink(), "contract file")
    need(sidecar.is_file() and not sidecar.is_symlink(), "contract sidecar")
    need(outer.is_file() and not outer.is_symlink(), "contract outer")
    need(sidecar.read_text().split() == [sha(path), path.name], "contract sidecar content")
    need(outer.read_text().split() == [sha(sidecar), sidecar.name], "contract outer content")
    return {"file": sha(path), "sidecar": sha(sidecar), "outer": sha(outer)}


def verify_directory(root, expected_manifest=None, expected_outer=None):
    root = Path(root)
    need(root.is_dir() and not root.is_symlink(), "directory absent " + str(root))
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(manifest.is_file() and not manifest.is_symlink(), "manifest")
    need(outer.is_file() and not outer.is_symlink(), "outer")
    if expected_manifest is not None:
        need(sha(manifest) == expected_manifest, "manifest identity")
    if expected_outer is not None:
        need(sha(outer) == expected_outer, "outer identity")
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"], "outer content")
    mapping = {}
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        name = rel.as_posix()
        need(not rel.is_absolute() and ".." not in rel.parts and name not in mapping,
             "manifest path")
        target = root / rel
        need(target.is_file() and not target.is_symlink() and sha(target) == fields[0],
             "member identity " + name)
        mapping[name] = fields[0]
    actual = set()
    for path in root.rglob("*"):
        need(not path.is_symlink(), "symlink " + str(path))
        if path.is_file() and path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
            actual.add(path.relative_to(root).as_posix())
    need(actual == set(mapping), "manifest not exhaustive")
    return mapping, sha(manifest), sha(outer)


def main():
    contract_seal = verify_file_seal(CONTRACT)
    contract = strict_json(CONTRACT)
    canonical = contract["canonical_identity"]
    m1811 = HW / canonical["m1811_canonical_directory"]
    m1830 = HW / canonical["m1830_review_directory"]
    m1811_map, m1811_manifest, m1811_outer = verify_directory(
        m1811, canonical["m1811_manifest_sha256"], canonical["m1811_outer_file_sha256"])
    m1830_map, m1830_manifest, m1830_outer = verify_directory(
        m1830, canonical["m1830_manifest_sha256"], canonical["m1830_outer_file_sha256"])
    need(m1811_map.get("receipt.json") == canonical["m1811_receipt_sha256"],
         "M1811 receipt sealed identity")
    need(m1830_map.get("review.json") == canonical["m1830_review_sha256"],
         "M1830 review sealed identity")
    review = strict_json(m1830 / "review.json")
    need(review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0},
         "M1830 admission")

    author_map, author_manifest, author_outer = verify_directory(AUTHOR)
    author_receipt = strict_json(AUTHOR / "receipt.json")
    need(author_map.get("receipt.json") == sha(AUTHOR / "receipt.json"),
         "author receipt sealed")
    need(author_receipt.get("formal_contract_identity") == {
        "contract_sha256": contract_seal["file"],
        "contract_sidecar_sha256": contract_seal["sidecar"],
        "contract_outer_file_sha256": contract_seal["outer"]},
        "author contract identity")

    inventory = contract["source_inventory"]
    need(len(inventory) == 10, "source inventory count")
    for row in inventory:
        path = HW / row["path"]
        need(path.is_file() and not path.is_symlink() and sha(path) == row["sha256"],
             "source inventory " + row["path"])
    for table_name in ("reused_source_identity",):
        for name, digest in contract[table_name].items():
            path = HW / name
            need(path.is_file() and not path.is_symlink() and sha(path) == digest,
                 table_name + " " + name)
    technology = contract["technology_identity"]
    for path_key, sha_key in (("cell_verilog_path", "cell_verilog_sha256"),
                              ("tt_db_path", "tt_db_sha256")):
        path = Path(technology[path_key])
        need(path.is_file() and not path.is_symlink() and sha(path) == technology[sha_key],
             "technology " + path_key)

    design = "m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24"
    mapped_roots = {"k8": m1811 / "k8/netlist", "k1x8": m1811 / "k1x8/netlist"}
    for axis, top in (("k8", design + "_ARCH_MODE0"),
                      ("k1x8", design + "_ARCH_MODE1")):
        netlist = mapped_roots[axis] / (design + "_mapped.v")
        sdc = mapped_roots[axis] / (design + "_mapped.sdc")
        need(sha(netlist) == canonical[axis + "_mapped_netlist_sha256"], axis + " netlist")
        need(sha(sdc) == canonical[axis + "_mapped_sdc_sha256"], axis + " sdc")
        modules = re.findall(r"(?m)^\s*module\s+([^\s(]+)", netlist.read_text())
        need(modules.count(top) == 1, axis + " derived top")

    need(sha(DOCS359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
         "docs359")
    forbidden = (
        HW / "results/.m1831_c2_fresh_mapped_production_energy_attempt_consumed",
        HW / "results/m1831_c2_fresh_mapped_production_energy_r1_20260902",
        HW / "results/m1831_c2_fresh_mapped_production_energy_r1_20260902.failed_or_incomplete.quarantine",
        HW / "results/m1831_c2_fresh_mapped_production_energy_r1_20260902.private_build.unsealed_do_not_cite",
        HW / "contracts/m1835_m1833_m1831_c2_fresh_mapped_production_energy_launch_release_r1_20260902.json",
    )
    need(not any(path.exists() or path.is_symlink() for path in forbidden),
         "execution/release namespace not fresh")

    print(json.dumps({
        "schema": "m1833_m1831_c2_energy_source_independent_identity_check_r1_v1",
        "status": "PASS_IDENTITY_AND_SEALS__SOURCE_REVIEW_STILL_FAIL_CLOSED_ON_SEMANTICS",
        "contract": contract_seal,
        "m1811_manifest_sha256": m1811_manifest,
        "m1811_outer_file_sha256": m1811_outer,
        "m1830_manifest_sha256": m1830_manifest,
        "m1830_outer_file_sha256": m1830_outer,
        "author_receipt_sha256": sha(AUTHOR / "receipt.json"),
        "author_manifest_sha256": author_manifest,
        "author_outer_file_sha256": author_outer,
        "source_inventory": "10/10",
        "mapped_netlist_sdc": "4/4",
        "derived_tops": "2/2_UNIQUE",
        "docs359_sha256": sha(DOCS359),
        "execution_and_release_namespaces": "FRESH",
        "eda_runs": 0,
        "license_queries": 0,
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
