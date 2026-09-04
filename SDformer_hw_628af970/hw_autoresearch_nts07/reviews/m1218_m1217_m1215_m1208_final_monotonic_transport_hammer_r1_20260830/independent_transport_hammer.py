#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fresh different-author M1218 hammer; local-only and side-effect free to source."""
from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "scripts/run_m1217_m1215_m1208_final_monotonic_transport_successor_source.py"
TEST = HW / "tests/test_run_m1217_m1215_m1208_final_monotonic_transport_successor_source.py"
INVENTORY = HW / "contracts/m1217_m1215_m1208_final_monotonic_transport_inventory_r1_20260830.json"
ROOTS = HW / "contracts/m1217_m1215_m1208_final_monotonic_transport_roots_r1_20260830.txt"
CONTRACT = HW / "contracts/m1217_m1215_m1208_final_monotonic_transport_source_contract_r1_20260830.json"
CONTRACT_SIDECAR = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT) + ".sha256.seal.sha256")
AUTHOR = HW / "reviews/m1217_m1215_m1208_final_monotonic_transport_author_r1_20260830"
AUDIT = HW / "reviews/m1217_m1215_m1208_remote_dependency_read_only_audit_r1_20260830"
M1217_ATTEMPT = HW / "results/.m1217_m1215_final_monotonic_transport_and_launch_r1_attempt_consumed"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    SOURCE: "a9ad626f24f7d6106b945ea5aa4b2b10615fe3bf0d44dd7e5a6487b72c1a423e",
    TEST: "10d1b9e2c23930fea741b0811f7a7c9caced323d060eb48956f7a9f376b521c5",
    INVENTORY: "f4fa6bd0c6f5aaf92978f198263d466ab1a0044e4145354f109e12385ca1658e",
    ROOTS: "7a189e9868da551eef36ab738d0fd093fec686d49b42dde2f14bdd172582c584",
    CONTRACT: "538c301e622a373a8f859df24cc107194007085ca06a190c86396418b9c077a9",
    AUTHOR / "review.json": "029452f7c284ca30797efc23e68713764c24b250f6b00ae70bb1e6a26b7441fe",
    AUTHOR / "SHA256SUMS": "474932ebb767ce531aee1e275e743b5dc05703553140e07300024dc62a376328",
    AUTHOR / "SHA256SUMS.seal.sha256": "da7bc18cf90e80950d6c6a49839889d6ba6644135c54667883d8ed2c3fc27aa5",
    AUDIT / "review.json": "8edaeb538c20dc10954e86049355b9dd8d62a6ebcfae698ef8ff0dbace07dec8",
    AUDIT / "SHA256SUMS": "287872ad4ca6d898fcf60dc9533e06a3a81ea541e3f53efb4346bded8689caca",
    AUDIT / "SHA256SUMS.seal.sha256": "429cc220d999a1bd516c6004e6be2a02f6ab2237cd7168df150fbdacdd5aa601",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location("m1218_fresh_hammer_target", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_double_seal(root: Path) -> dict[str, str]:
    manifest, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    parts = outer.read_text(encoding="ascii").split()
    require(parts == [sha(manifest), "SHA256SUMS"], "outer seal drift " + str(root))
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        require("/" not in name and name not in rows, "unsafe manifest")
        path = root / name
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "member drift " + str(path))
        rows[name] = digest
    actual = {path.name for path in root.iterdir() if path.is_file() and not path.is_symlink()
              and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(rows), "sealed membership drift")
    return rows


def archive(path: Path, member_path: str, payload: bytes) -> dict[str, Any]:
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as stream:
        info = tarfile.TarInfo(member_path); info.size = len(payload); info.mode = 0o444
        info.uid = info.gid = info.mtime = 0; stream.addfile(info, io.BytesIO(payload))
    return {"path": member_path, "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def helper_fixture(module: Any, mutation: str = "none") -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="m1218_helper.") as name:
        base = Path(name); repo = base / "repo"; temp = base / "remote_tmp"
        repo.mkdir(mode=0o755); temp.mkdir(mode=0o700)
        old = []
        for index in range(142):
            rel = "authority/old_{:03d}.bin".format(index)
            path = repo / rel; path.parent.mkdir(parents=True, exist_ok=True)
            payload = ("old-{}".format(index)).encode(); path.write_bytes(payload)
            old.append({"path": rel, "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest()})
        marker_rel = "state/m1180.attempt"
        marker = repo / marker_rel; marker.parent.mkdir(parents=True); marker.write_text(
            "M1180_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n", encoding="ascii")
        member_rel = "authority/forensic_member.bin"; payload = b"forensic-nine-closure"
        archive_path = temp / "exact_requirement_closure.tar"
        member = archive(archive_path, member_rel, payload)
        runtime_absent = ["state/m1208.attempt", "state/m1208.result", "state/m1208.log"]
        authority = old + [member]
        plan = {"archive_size": archive_path.stat().st_size, "archive_sha256": sha(archive_path),
                "members": [member], "old_dependencies": old, "required_exact": [member],
                "post_publish_authority": authority, "sealed_roots": [],
                "runtime_absent": runtime_absent, "m1180_attempt": marker_rel}
        if mutation == "archive_sha":
            plan["archive_sha256"] = "0" * 64
        elif mutation == "member_name":
            plan["members"][0] = dict(member, path="authority/wrong.bin")
        elif mutation == "runtime_present":
            target = repo / runtime_absent[0]; target.write_bytes(b"occupied")
        elif mutation == "target_drift":
            target = repo / member_rel; target.write_bytes(b"drift")
        encoded = base64.b64encode(json.dumps(plan, separators=(",", ":")).encode()).decode()
        # The production helper correctly requires remote temp ownership by uid 0.
        # This local-only harness runs as the workspace uid, so remove only that
        # environmental predicate while preserving every publication/hash check.
        helper = module.REMOTE_HELPER.replace(" or st.st_uid!=0", "")
        require(helper != module.REMOTE_HELPER, "uid predicate not found")
        return subprocess.run([sys.executable, "-c", helper, str(repo), str(temp),
                               archive_path.name, encoded], text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=False,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def main() -> int:
    checks: list[str] = []
    for path, expected in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == expected,
                "identity drift " + str(path)); checks.append("identity:" + path.name)
    require(CONTRACT_SIDECAR.read_text(encoding="ascii").split() ==
            [EXPECTED[CONTRACT], CONTRACT.name], "contract sidecar drift")
    require(CONTRACT_OUTER.read_text(encoding="ascii").split() ==
            [sha(CONTRACT_SIDECAR), CONTRACT_SIDECAR.name], "contract outer drift")
    checks += ["contract_sidecar", "contract_outer"]
    author_rows = verify_double_seal(AUTHOR); audit_rows = verify_double_seal(AUDIT)
    require(author_rows.get("review.json") == EXPECTED[AUTHOR / "review.json"], "author review")
    require(audit_rows.get("review.json") == EXPECTED[AUDIT / "review.json"], "audit review")
    checks += ["author_double_seal", "audit_double_seal"]
    author = json.loads((AUTHOR / "review.json").read_text(encoding="utf-8"))
    require(author.get("status") ==
            "PASS_M1217_SOURCE_AND_143_FILE_POST_PUBLISH_AUTHORITY__M1218_HAMMER_ONLY" and
            author.get("verdict") == "SOURCE_GO__EXECUTION_NO_GO_UNTIL_FRESH_M1218" and
            author.get("score", 0) >= 95 and author.get("p0_count") == 0 and
            author.get("p1_count") == 0, "author semantic admission")
    checks.append("author_semantics")
    module = load_module()
    contract, inventory, members, closure, authority = module.load_release()
    require(len(members) == 40 and len(closure["required_exact"]) == 41 and
            len(authority) == 143 and len(closure["sealed_roots"]) == 4,
            "closure cardinality")
    forensic = "hw_autoresearch_nts07/reviews/m1215_m1210_m1208_first_launch_failure_forensic_r1_20260830/"
    require(sum(row["path"].startswith(forensic) for row in members) == 9,
            "forensic9 incomplete")
    checks += ["load_release", "transfer40", "required41", "authority143",
               "recursive_seals4", "forensic9"]
    module.verify_marker(module.M1210_ATTEMPT, module.M1210_SHA, module.M1210_TOKEN, "m1210")
    module.verify_marker(module.M1215_ATTEMPT, module.M1215_SHA, module.M1215_TOKEN, "m1215")
    require(not os.path.lexists(M1217_ATTEMPT), "M1217 attempt not fresh")
    checks += ["m1210_consumed_exact", "m1215_consumed_exact", "m1217_fresh"]
    source_text = SOURCE.read_text(encoding="utf-8"); tree = ast.parse(source_text)
    execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "execute_once")
    body = ast.get_source_segment(source_text, execute) or ""
    require(body.count("launched = runner(") == 1 and body.count("launch_code =") == 1 and
            "while " not in body and "no retry authorized" in body,
            "one-shot launcher structure")
    require(sha(ROOT / module.LAUNCHER_REL) ==
            "30936622b629439d6d6c112d17bfc16881ae45d293660f615ba99309a5a3d98c",
            "existing M1215 launcher drift")
    checks += ["one_launcher", "no_retry_loop", "existing_launcher_exact"]
    controlled = subprocess.run([sys.executable, "-m", "unittest", "-v",
        "hw_autoresearch_nts07.tests.test_run_m1217_m1215_m1208_final_monotonic_transport_successor_source"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    require(controlled.returncode == 0 and controlled.stderr.count(" ... ok") == 12,
            "controlled source tests")
    checks.append("source_tests_12_of_12")
    positive = helper_fixture(module)
    require(positive.returncode == 0 and positive.stdout.count(
        "PASS_M1217_REMOTE_REQUIREMENT_CLOSURE_PREFLIGHT__POST_VERIFY_143_EXACT") == 1,
        "local remote-helper positive simulation")
    checks.append("helper_postverify_143_exact")
    mutations = {}
    for name in ("archive_sha", "member_name", "runtime_present", "target_drift"):
        result = helper_fixture(module, name)
        mutations[name] = {"rejected": result.returncode != 0,
                           "stderr_tail": result.stderr.strip().splitlines()[-1:]}
        require(result.returncode != 0, "mutation survived: " + name)
    checks += ["mutation_reject:" + name for name in mutations]
    # Independent local semantic mutations of the frozen inventory.
    changed = json.loads(json.dumps(inventory))
    changed["transfer_roots"] = [row for row in changed["transfer_roots"]
                                 if "first_launch_failure_forensic" not in row["path"]]
    try: module.expand_transfer_roots(changed)
    except module.ReleaseError: pass
    else: raise AssertionError("missing forensic mutation survived")
    changed = json.loads(json.dumps(inventory)); changed["remote_dependency_authority"]["post_publish_exact_required"] = 142
    try: module.build_remote_authority(changed, members,
        module.strict_json(ROOT / module.OLD_INVENTORY_REL)["dependencies"],
        module.strict_json(ROOT / module.M1210_INVENTORY_REL)["transfer_required"])
    except module.ReleaseError: pass
    else: raise AssertionError("authority142 mutation survived")
    checks += ["mutation_reject:forensic_root", "mutation_reject:authority142"]
    output = {"schema": "m1218_independent_transport_hammer_checks_r1_v1",
              "status": "PASS_FRESH_DIFFERENT_AUTHOR_LOCAL_ONLY_HAMMER",
              "check_count": len(checks), "checks": checks,
              "controlled_tests": {"passed": 12, "failed": 0},
              "mutations": mutations | {"forensic_root": {"rejected": True},
                                           "authority142": {"rejected": True}},
              "closure": {"transfer_members": 40, "required_exact": 41,
                          "authority_pre_exact": 134, "forensic_missing_pre": 9,
                          "authority_post_exact_required": 143,
                          "recursive_seals": 4},
              "attempts": {"m1210_consumed_exact": True, "m1215_consumed_exact": True,
                           "m1217_fresh": True, "existing_m1215_launcher_invocations": 1,
                           "automatic_retry": False},
              "claim_boundary": {"remote": False, "network": False, "gpu": False,
                                 "capture": False, "eda": False},
              "docs359_sha256": sha(DOCS359)}
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
