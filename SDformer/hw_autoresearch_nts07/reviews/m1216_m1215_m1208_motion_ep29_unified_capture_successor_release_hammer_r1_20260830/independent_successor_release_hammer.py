#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Read-only M1216 hammer for the M1215 secure capture successor."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat


HW = Path(__file__).resolve().parents[2]
SUCCESSOR = HW / "scripts/run_m1215_motion_ep29_unified_capture_remote_one_shot_successor_source.py"
WRAPPER = HW / "scripts/run_m1215_m1208_motion_ep29_unified_capture_successor_secure_remote_one_shot_source.py"
TEST = HW / "tests/test_run_m1215_m1208_motion_ep29_unified_capture_successor_secure_remote_one_shot_source.py"
CONTRACT = HW / "contracts/m1215_m1208_motion_ep29_unified_capture_successor_secure_release_source_contract_r1_20260830.json"
INVENTORY = HW / "contracts/m1215_m1208_motion_ep29_unified_capture_successor_remote_dependency_inventory_r1_20260830.json"
TRANSFER = HW / "contracts/m1215_m1208_motion_ep29_unified_capture_successor_remote_transfer_files_r1_20260830.txt"
AUTHOR = HW / "reviews/m1215_m1210_motion_ep29_unified_capture_successor_release_author_r1_20260830"
FORENSIC = HW / "reviews/m1215_m1210_m1208_first_launch_failure_forensic_r1_20260830"
M1211 = HW / "reviews/m1211_m1210_m1208_motion_ep29_unified_capture_release_hammer_r1_20260830"
M1209 = HW / "reviews/m1209_m1208_motion_checkpoint_parametric_unified_capture_r2_source_hammer_r1_20260830"
if not M1209.exists():
    M1209 = HW / "reviews/m1209_m1208_motion_ep29_unified_capture_source_hammer_r1_20260830"
M1210_MARKER = HW / "results/.m1210_m1208_secure_transfer_and_launch_r1_attempt_consumed"
M1215_MARKER = HW / "results/.m1215_m1208_successor_secure_transfer_and_launch_r1_attempt_consumed"
M1208_ATTEMPT = HW / "results/.m1208_motion_ep29_unified_hardware_capture_s40_r1_20260830.attempt_consumed"
M1208_RESULT = HW / "results/m1208_motion_ep29_unified_hardware_capture_s40_r1_20260830"
M1208_LOG = HW / "results/.m1208_motion_ep29_unified_hardware_capture_s40_r1_20260830.production.log"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    SUCCESSOR: "30936622b629439d6d6c112d17bfc16881ae45d293660f615ba99309a5a3d98c",
    WRAPPER: "c05cf6863ac56812e5153628eb8c76e2724742ffb3e24b1af491251f7c00c9a2",
    TEST: "a7652c7bd6659c520f3d0a9fa73dc888d0f7cd27745654cc17e435255db55706",
    CONTRACT: "9f66200e8f27e1f646cac020d40efebc26e0995468ab132371cb5e9eadbe33fe",
    INVENTORY: "4b7374f6f5b55086f322da02fed32315cc17640c74685687395b20ee3445e4d5",
    TRANSFER: "bf63a587967f668a1c0b253e5f0ca774b24244a4501a7b427f2d30b8a01bf95c",
    AUTHOR / "author_receipt.json": "523e4fa5deed1ccb41626b21b8466c344965d223f83285b8f8590894d086b446",
    AUTHOR / "SHA256SUMS": "7a0490554176a8c529d3c201979ad902b56ca5e5fc8c1173bba4ca0c18926509",
    AUTHOR / "SHA256SUMS.seal.sha256": "5cdbdc9de8ba30aae002d31af515e0dd006f2921bd63d1ed62620ad97a46fad4",
    FORENSIC / "review.json": "1bc5af3d81d1cc1ee8dd7a91871ba9d31edc31cc442c256d96e23bc1bd828e65",
    FORENSIC / "SHA256SUMS": "a4cb9a3da26224b4f75bd2b3ea857c262155a81a094eb977e981ce60ea77cacb",
    FORENSIC / "SHA256SUMS.seal.sha256": "a7868aec49fb721f495e40d80d98f0d5525391ea807608a1363c04465373f995",
    M1210_MARKER: "b60af667912eae9f19fb93aaf201fc342cfdd22e9add4bfeac0e55c09268e5f6",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
OUTER_CONTRACT_SHA = "9f66200e8f27e1f646cac020d40efebc26e0995468ab132371cb5e9eadbe33fe"
INNER_CAPTURE_CONTRACT_SHA = "dad36c0a264e3e0d3a478929549431453ced60cba84fc24b2d9de442d29faa20"
CAPTURE_SHA = "41b5276c39b613b6568ad7c7486abf150c3d0db86c3a905d6a30cdbbb543a049"
CAPTURE_TEST_SHA = "69de86545947d3c006dc621ddc0b618a61a8c57aa7e453478f61b56f079b3934"
LAUNCH_CONTRACT_SHA = "5aeeaf9cab836f32e025f0c329ef1fe90caa4ee3acae691514f4793c1d143829"
M1211_REVIEW_SHA = "813eec1d3fe025a21001c03d8394f32cb646674a1687648e1be2eefb54bb6567"
checks = 0


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value: raise AssertionError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key"); out[key] = value
        return out
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           AssertionError("nonfinite " + token)))
    require(isinstance(value, dict), "JSON root")
    return value


def verify_seal(directory: Path) -> None:
    require(directory.is_dir() and not directory.is_symlink(), "sealed dir")
    manifest = directory / "SHA256SUMS"; outer = directory / "SHA256SUMS.seal.sha256"
    require(outer.read_text().split() == [sha(manifest), "SHA256SUMS"], "outer seal")
    rows = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*"); rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts and name not in rows, "safe member")
        rows[name] = digest
    actual = set()
    for root, dirs, files in os.walk(directory, followlinks=False):
        base = Path(root); dirs[:] = [n for n in dirs if not (base / n).is_symlink()]
        for name in files:
            p = base / name; rel = p.relative_to(directory).as_posix()
            if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"} or p.is_symlink(): continue
            if stat.S_ISREG(p.lstat().st_mode): actual.add(rel)
    require(actual == set(rows), "complete membership")
    for name, digest in rows.items(): require(sha(directory / name) == digest, "member SHA")


def sidecar(path: Path) -> None:
    manifest = Path(str(path) + ".sha256"); outer = Path(str(path) + ".sha256.seal.sha256")
    require(manifest.read_text().split() == [sha(path), path.name], "sidecar")
    require(outer.read_text().split() == [sha(manifest), manifest.name], "sidecar outer")


def semantic(review: dict) -> None:
    require(review.get("schema") ==
            "m1216_m1215_m1208_motion_ep29_unified_capture_successor_release_hammer_r1_v1", "schema")
    require(review.get("status") ==
            "PASS_M1215_SUCCESSOR_SECURE_TRANSFER_AND_ONE_M1208_REMOTE_LAUNCH_AUTHORIZED", "status")
    require(review.get("verdict") == "GO" and review.get("score", 0) >= 95 and
            review.get("p0_count") == 0 and review.get("p1_count") == 0, "verdict")
    b = review.get("bindings", {})
    outer = {"source_sha256": sha(WRAPPER), "source_contract_sha256": sha(CONTRACT),
             "launch_contract_sha256": LAUNCH_CONTRACT_SHA, "inventory_sha256": sha(INVENTORY),
             "transfer_list_sha256": sha(TRANSFER), "release_author_manifest_sha256": sha(AUTHOR/"SHA256SUMS"),
             "release_author_outer_file_sha256": sha(AUTHOR/"SHA256SUMS.seal.sha256"),
             "successor_launcher_sha256": sha(SUCCESSOR), "m1210_failure_marker_sha256": sha(M1210_MARKER)}
    inner = {"capture_source_sha256": CAPTURE_SHA,
             "capture_source_contract_sha256": INNER_CAPTURE_CONTRACT_SHA,
             "capture_source_test_sha256": CAPTURE_TEST_SHA,
             "source_hammer_manifest_sha256": sha(M1209/"SHA256SUMS"),
             "source_hammer_outer_file_sha256": sha(M1209/"SHA256SUMS.seal.sha256"),
             "m1211_review_sha256": M1211_REVIEW_SHA,
             "m1215_forensic_review_sha256": sha(FORENSIC/"review.json")}
    require(all(b.get(k) == v for k, v in outer.items()), "outer bindings")
    require(all(b.get(k) == v for k, v in inner.items()), "inner bindings")
    require(b["source_contract_sha256"] == OUTER_CONTRACT_SHA and
            b["capture_source_contract_sha256"] == INNER_CAPTURE_CONTRACT_SHA and
            b["source_contract_sha256"] != b["capture_source_contract_sha256"], "contract vocabulary split")
    require(review.get("authorization") == {"secure_transfer": True, "exact_remote_launch": True,
            "launch_count": 1, "automatic_retry": False, "result_hammer_required": True}, "authorization")


def main() -> None:
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "exact identity " + str(path))
    for directory in (AUTHOR, FORENSIC, M1211, M1209): verify_seal(directory)
    sidecar(CONTRACT)
    ast.parse(SUCCESSOR.read_text()); ast.parse(WRAPPER.read_text()); ast.parse(TEST.read_text())
    contract = strict_json(CONTRACT); inventory = strict_json(INVENTORY); forensic = strict_json(FORENSIC/"review.json")
    require(contract["source"]["sha256"] == sha(WRAPPER) and
            contract["successor_launcher"]["sha256"] == sha(SUCCESSOR) and
            contract["test"]["sha256"] == sha(TEST), "contract identity")
    require(inventory["transfer_required"] == [{"path":
            "hw_autoresearch_nts07/scripts/run_m1215_motion_ep29_unified_capture_remote_one_shot_successor_source.py",
            "size_bytes": SUCCESSOR.stat().st_size, "sha256": sha(SUCCESSOR)}], "exact one transfer member")
    require(TRANSFER.read_text().splitlines() == [inventory["transfer_required"][0]["path"]], "transfer list")
    require(M1210_MARKER.read_text() ==
            "M1210_TRANSFER_COMPLETE__M1208_REMOTE_LAUNCH_ATTEMPT_CONSUMED__NO_RETRY\n", "M1210 token")
    require(not M1215_MARKER.exists(), "fresh successor attempt")
    require(forensic["remote_inner_state"] == {
        "m1208_attempt_exists_before_reproduction": False,
        "m1208_result_exists_before_reproduction": False,
        "m1208_log_exists_before_reproduction": False,
        "m1208_attempt_exists_after_reproduction": False,
        "m1208_result_exists_after_reproduction": False,
        "m1208_log_exists_after_reproduction": False,
        "capture_related_processes": 0, "gpu_compute_applications": 0,
        "model_or_capture_workload_started": False}, "sealed remote fresh observation")
    wrapper = WRAPPER.read_text(); execute = ast.get_source_segment(wrapper,
        next(n for n in ast.parse(wrapper).body if isinstance(n, ast.FunctionDef) and n.name == "execute_once")) or ""
    require(execute.count("launched = runner(") == 1 and execute.count("launch_code") == 2,
            "exact one successor launch")
    require("M1210 failure marker drift; retry of M1210 forbidden" in execute and
            execute.index("load_m1216") < execute.index("os.open(LOCAL_ATTEMPT") < execute.index("launched = runner"),
            "hammer and old failure gate before new attempt before launch")
    require("no retry authorized" in execute and "retry_launch" not in execute, "no retry")
    successor = SUCCESSOR.read_text()
    for token in ("capture_source_sha256", "capture_source_contract_sha256", "capture_source_test_sha256",
                  "source_hammer_manifest_sha256", "source_hammer_outer_file_sha256"):
        require(token in successor, "inner five " + token)
    require('bindings.get("source_contract_sha256") ==' in successor and
            'bindings.get("capture_source_contract_sha256") ==' in successor,
            "outer and inner contract objects split")

    base = {"schema": "m1216_m1215_m1208_motion_ep29_unified_capture_successor_release_hammer_r1_v1",
        "status": "PASS_M1215_SUCCESSOR_SECURE_TRANSFER_AND_ONE_M1208_REMOTE_LAUNCH_AUTHORIZED",
        "verdict": "GO", "score": 99, "p0_count": 0, "p1_count": 0,
        "bindings": {"source_sha256": sha(WRAPPER), "source_contract_sha256": sha(CONTRACT),
            "launch_contract_sha256": LAUNCH_CONTRACT_SHA, "inventory_sha256": sha(INVENTORY),
            "transfer_list_sha256": sha(TRANSFER), "release_author_manifest_sha256": sha(AUTHOR/"SHA256SUMS"),
            "release_author_outer_file_sha256": sha(AUTHOR/"SHA256SUMS.seal.sha256"),
            "successor_launcher_sha256": sha(SUCCESSOR), "m1210_failure_marker_sha256": sha(M1210_MARKER),
            "capture_source_sha256": CAPTURE_SHA, "capture_source_contract_sha256": INNER_CAPTURE_CONTRACT_SHA,
            "capture_source_test_sha256": CAPTURE_TEST_SHA, "source_hammer_manifest_sha256": sha(M1209/"SHA256SUMS"),
            "source_hammer_outer_file_sha256": sha(M1209/"SHA256SUMS.seal.sha256"),
            "m1211_review_sha256": M1211_REVIEW_SHA, "m1215_forensic_review_sha256": sha(FORENSIC/"review.json")},
        "authorization": {"secure_transfer": True, "exact_remote_launch": True, "launch_count": 1,
                          "automatic_retry": False, "result_hammer_required": True}}
    semantic(base)
    mutations = 0
    mutators = [
        lambda x: x.update(status="FAIL"),
        lambda x: x["bindings"].update(source_contract_sha256=INNER_CAPTURE_CONTRACT_SHA),
        lambda x: x["bindings"].update(capture_source_contract_sha256=OUTER_CONTRACT_SHA),
        lambda x: x["bindings"].pop("capture_source_sha256"),
        lambda x: x["bindings"].pop("capture_source_test_sha256"),
        lambda x: x["bindings"].pop("source_hammer_manifest_sha256"),
        lambda x: x["bindings"].update(m1210_failure_marker_sha256="0"*64),
        lambda x: x["authorization"].update(launch_count=2),
        lambda x: x.update(p1_count=1),
        lambda x: x.update(score=94),
    ]
    for mutate in mutators:
        bad = copy.deepcopy(base); mutate(bad)
        try: semantic(bad)
        except (AssertionError, KeyError): mutations += 1
    require(mutations == len(mutators), "all mutations rejected")
    print(json.dumps({"schema": "m1216_independent_successor_release_hammer_mechanical_r1_v1",
        "status": "PASS", "checks": checks, "mutations_rejected": mutations,
        "author_forensic_m1211_m1209_recursive_seals": 4,
        "forensic_review_manifest_outer_exact": True, "controlled_tests": "10/10 PASS",
        "m1210_marker_consumed_no_retry": True, "sealed_remote_m1208_namespace_fresh": True,
        "m1215_attempt_fresh": True, "inner_five_plus_contract_split": True,
        "exactly_one_launch": True, "automatic_retry": False,
        "remote_runs": 0, "network_runs": 0, "gpu_runs": 0, "capture_runs": 0,
        "eda_runs": 0, "docs359_sha256": sha(DOCS359)}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
