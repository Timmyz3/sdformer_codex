#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent, local-only M1209 hammer for the inert M1208 source package."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import stat
import subprocess
import sys
import tempfile
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1208_motion_ep29_unified_hardware_symlink_root_successor_r1.py")
CONTRACT = HW / (
    "contracts/m1208_motion_ep29_unified_capture_symlink_root_successor_"
    "source_contract_r1_20260830.json")
TEST = HW / "tests/test_m1208_motion_ep29_unified_capture_symlink_root_successor_source.py"
LAUNCHER = HW / "scripts/run_m1208_motion_ep29_unified_capture_remote_one_shot_source.py"
AUTHOR = HW / (
    "reviews/m1208_motion_ep29_unified_capture_symlink_root_successor_author_r1_20260830")
OLD_LAUNCH = HW / "contracts/m1182_m1180_motion_ep29_unified_capture_launch_release_r1_20260830.json"
TECHNICAL = HW / "contracts/m1177_motion_checkpoint_parametric_unified_capture_source_contract_r2_20260830.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    SOURCE: "41b5276c39b613b6568ad7c7486abf150c3d0db86c3a905d6a30cdbbb543a049",
    CONTRACT: "dad36c0a264e3e0d3a478929549431453ced60cba84fc24b2d9de442d29faa20",
    TEST: "69de86545947d3c006dc621ddc0b618a61a8c57aa7e453478f61b56f079b3934",
    LAUNCHER: "273447a1de9708a066e7356a66ef346213cf723c96c7b505a441acc4532dcfae",
    AUTHOR / "review.json": "1efcd9561b31ba26375f6dd9c5e44e0a40fc65d7ccdd8d84e9f5345aca6a5931",
    AUTHOR / "SHA256SUMS": "165699281df615d667566c314159baa232005182bb5cae8141639eb58dad3ed1",
    AUTHOR / "SHA256SUMS.seal.sha256": "ab6de94999a1207008ffbfb985b9c6b08ea70939694e47748cee8eaa6ea78891",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path) -> None:
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(),
            "non-regular or symlink artifact: " + str(path))


def strict_json(path: Path) -> dict:
    def reject(token: str) -> None:
        raise RuntimeError("non-standard JSON token " + token)
    def pairs(items: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key " + key)
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=reject)
    require(isinstance(value, dict), "JSON root is not object")
    return value


def verify_seal(root: Path) -> dict[str, str]:
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    regular(manifest)
    regular(outer)
    require(outer.read_text(encoding="utf-8").split() == [sha256(manifest), "SHA256SUMS"],
            "outer seal mismatch")
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and len(fields[0]) == 64, "malformed manifest row")
        name = fields[1].lstrip("*")
        relative = Path(name)
        require(name == relative.as_posix() and not relative.is_absolute() and
                ".." not in relative.parts and name not in rows, "unsafe manifest member")
        member = root / relative
        regular(member)
        require(sha256(member) == fields[0], "member SHA mismatch: " + name)
        rows[name] = fields[0]
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(rows), "recursive sealed population mismatch")
    return rows


def load_source() -> object:
    spec = importlib.util.spec_from_file_location("m1209_m1208_under_hammer", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot import source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_tree(base: Path, payload: bytes = b"exact") -> tuple[Path, Path, Path, str]:
    target = base / "target"
    leaf = target / "saved_flow_data/event_tensors/10bins/left/seq/sample.npy"
    leaf.parent.mkdir(parents=True)
    leaf.write_bytes(payload)
    (base / "repo/data/Datasets").mkdir(parents=True)
    (base / "repo/data/Datasets/DSEC").symlink_to(target)
    return base / "repo", target, leaf, sha256(leaf)


def expect_fail(callback, needle: str) -> None:
    try:
        callback()
    except Exception as error:
        require(needle in str(error), "wrong rejection for {}: {}".format(needle, error))
    else:
        raise RuntimeError("mutation unexpectedly admitted: " + needle)


def r1_attribute_assignments(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "run_capture")
    result: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                        and target.value.id == "R1"):
                    result.add("R1." + target.attr)
    return result


def main() -> int:
    for path, expected in EXPECTED.items():
        regular(path)
        require(sha256(path) == expected, "pinned SHA drift: " + str(path))
    author_rows = verify_seal(AUTHOR)
    require(author_rows.get("review.json") == EXPECTED[AUTHOR / "review.json"],
            "author review member binding mismatch")
    author = strict_json(AUTHOR / "review.json")
    require(author.get("status") == "PASS_SOURCE_ONLY__FRESH_DIFFERENT_AUTHOR_HAMMERS_REQUIRED",
            "author source-only status mismatch")

    policy = strict_json(CONTRACT)
    old = strict_json(OLD_LAUNCH)
    technical = strict_json(TECHNICAL)
    require(policy["source"]["sha256"] == EXPECTED[SOURCE] and
            policy["test_sha256"] == EXPECTED[TEST], "source policy SHA binding mismatch")
    require(policy["claim_boundary"] == {
        "source_only": True, "production_authorized": False, "remote": False,
        "gpu": False, "capture_complete": False, "hardware_speedup": False,
        "system_speedup": False, "energy": False, "ppa": False,
        "paper_citable_result": False}, "source-only claim boundary drift")
    require(policy["sealed_m1180_predecessor"]["role"] ==
            "immutable_technical_and_failure_evidence__never_retry", "M1180 role drift")
    failure = policy["prior_m1180_failure"]
    require(failure["attempt_token"] == "M1180_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE" and
            failure["automatic_retry"] is False and failure["result_absent"] is True and
            failure["production_log_absent"] is True and
            failure["failure_boundary"] ==
            "R1_SELECTED_SAMPLES_REJECTED_PINNED_DSEC_SYMLINK_PRE_GPU",
            "M1180 failure/non-retry binding drift")
    future = policy["future_launch_contract"]
    require(all("m1208" in future[key] for key in
                ("canonical_attempt_marker", "canonical_result", "canonical_production_log")),
            "M1208 namespace is not disjoint")
    require(all(future[key] != failure[old_key] for key, old_key in (
        ("canonical_attempt_marker", "attempt_marker"),
        ("canonical_result", "result_path"),
        ("canonical_production_log", "production_log_path"))),
        "M1208 namespace overlaps failed M1180")

    rows = old["r1_compatible_binding"]["cohort"]["samples"]
    require(rows == old["cohort"]["samples"] == technical["frozen_samples"],
            "exact40 cohort changed")
    require(len(rows) == 40 and [row["global_sample_id"] for row in rows] == list(range(40)),
            "exact40 order/population mismatch")
    require([row["cohort"] for row in rows[:10]] == ["c1"] * 10 and
            [row["cohort"] for row in rows[10:]] == ["decoder"] * 30,
            "exact40 cohort composition mismatch")
    require(len({row["path"] for row in rows}) == len({row["sha256"] for row in rows}) ==
            len({row["sample_key"] for row in rows}) == 40,
            "exact40 identity uniqueness mismatch")
    require(all(Path(row["path"]).parts[:3] == ("data", "Datasets", "DSEC") for row in rows),
            "exact40 path prefix mismatch")

    old_source = ROOT / policy["sealed_m1180_predecessor"]["source_path"]
    require(sha256(old_source) == policy["sealed_m1180_predecessor"]["source_sha256"],
            "M1180 predecessor SHA drift")
    old_assign = r1_attribute_assignments(old_source)
    new_assign = r1_attribute_assignments(SOURCE)
    require(new_assign - old_assign == {"R1.selected_samples"} and
            old_assign - new_assign == set(),
            "technical R1 mutation surface exceeds selected_samples override")
    source_text = SOURCE.read_text(encoding="utf-8")
    require("original_selected_samples = R1.selected_samples" in source_text and
            "finally:" in source_text and
            "R1.selected_samples = original_selected_samples" in source_text,
            "selected_samples resolver is not restored in finally")
    require("contract[\"r1_compatible_binding\"] == _expected_r1_successor" in source_text,
            "exact predecessor binding comparison absent")

    module = load_source()
    with tempfile.TemporaryDirectory() as name:
        repo, target, leaf, leaf_sha = make_tree(Path(name))
        rel = "data/Datasets/DSEC/saved_flow_data/event_tensors/10bins/left/seq/sample.npy"
        resolver = lambda **kw: module._resolve_whitelisted_sample(
            kw.pop("relative", rel), kw.pop("size", 5), kw.pop("digest", leaf_sha),
            repo_root=kw.pop("repo", repo), pinned_root=kw.pop("target", target))
        require(resolver() == leaf, "exact pinned symlink fixture rejected")
        expect_fail(lambda: resolver(relative="data/Datasets/DSEC/../escape.npy"),
                    "below exact pinned")
        expect_fail(lambda: resolver(size=6), "identity drift")
        expect_fail(lambda: resolver(digest="0" * 64), "identity drift")

    with tempfile.TemporaryDirectory() as name:
        base = Path(name)
        repo, target, leaf, leaf_sha = make_tree(base)
        link = repo / "data/Datasets/DSEC"
        link.unlink()
        link.symlink_to(Path("../../../target"))
        expect_fail(lambda: module._resolve_whitelisted_sample(
            "data/Datasets/DSEC/saved_flow_data/event_tensors/10bins/left/seq/sample.npy",
            5, leaf_sha, repo_root=repo, pinned_root=target), "raw symlink target drift")

    with tempfile.TemporaryDirectory() as name:
        base = Path(name)
        repo, target, leaf, leaf_sha = make_tree(base)
        real_data = repo / "real_data"
        (repo / "data/Datasets/DSEC").unlink()
        (repo / "data/Datasets").rmdir()
        (repo / "data").rename(real_data)
        (repo / "data").symlink_to(real_data, target_is_directory=True)
        expect_fail(lambda: module._resolve_whitelisted_sample(
            "data/Datasets/DSEC/saved_flow_data/event_tensors/10bins/left/seq/sample.npy",
            5, leaf_sha, repo_root=repo, pinned_root=target), "real directory")

    with tempfile.TemporaryDirectory() as name:
        base = Path(name)
        repo, target, leaf, leaf_sha = make_tree(base)
        seq = leaf.parent
        moved = target / "real_seq"
        seq.rename(moved)
        seq.symlink_to(moved, target_is_directory=True)
        expect_fail(lambda: module._resolve_whitelisted_sample(
            "data/Datasets/DSEC/saved_flow_data/event_tensors/10bins/left/seq/sample.npy",
            5, leaf_sha, repo_root=repo, pinned_root=target), "non-whitelisted symlink")

    with tempfile.TemporaryDirectory() as name:
        base = Path(name)
        repo, target, leaf, leaf_sha = make_tree(base)
        real = leaf.with_name("real.npy")
        leaf.rename(real)
        leaf.symlink_to(real)
        expect_fail(lambda: module._resolve_whitelisted_sample(
            "data/Datasets/DSEC/saved_flow_data/event_tensors/10bins/left/seq/sample.npy",
            5, leaf_sha, repo_root=repo, pinned_root=target), "non-whitelisted symlink")

    with tempfile.TemporaryDirectory() as name:
        base = Path(name)
        real_target = base / "real_target"
        real_target.mkdir()
        alias = base / "alias_target"
        alias.symlink_to(real_target, target_is_directory=True)
        (base / "repo/data/Datasets").mkdir(parents=True)
        (base / "repo/data/Datasets/DSEC").symlink_to(alias)
        expect_fail(lambda: module._resolve_whitelisted_sample(
            "data/Datasets/DSEC/sample.npy", 1, "0" * 64,
            repo_root=base / "repo", pinned_root=alias), "resolved absolute target drift")

    original = module.R1.selected_samples
    def controlled_failure(_contract: object, _binding: object) -> Path:
        require(module.R1.selected_samples is module.selected_samples, "override not active")
        raise RuntimeError("controlled restore proof")
    with mock.patch.object(module, "frozen_inventory", return_value={}), \
         mock.patch.object(module.R1, "run_capture", side_effect=controlled_failure):
        expect_fail(lambda: module.run_capture({"r1_compatible_binding": {}}, {"policy": {}}),
                    "controlled restore proof")
    require(module.R1.selected_samples is original, "resolver not restored after exception")
    expect_fail(lambda: module.validate_launch_contract(policy, CONTRACT), "cannot launch")

    completed = subprocess.run([sys.executable, str(TEST)], cwd=ROOT, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    require(completed.returncode == 0 and "Ran 12 tests" in completed.stdout and
            completed.stdout.rstrip().endswith("OK"), "controlled test suite failed")
    for path in (SOURCE, TEST, LAUNCHER):
        with tempfile.TemporaryDirectory() as cache:
            py_compile.compile(str(path), cfile=str(Path(cache) / "compiled.pyc"), doraise=True)

    print(json.dumps({
        "status": "PASS", "pinned_artifacts": len(EXPECTED),
        "author_seal_members": len(author_rows), "controlled_tests": 12,
        "independent_fixture_classes": 7, "exact40": len(rows),
        "r1_assignment_delta": sorted(new_assign - old_assign),
        "source_only": True, "remote": False, "gpu": False,
        "capture": False, "eda": False, "docs359_sha256": sha256(DOCS359),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
