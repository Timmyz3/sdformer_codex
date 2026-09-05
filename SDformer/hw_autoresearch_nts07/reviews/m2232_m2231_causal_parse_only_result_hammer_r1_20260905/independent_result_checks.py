#!/opt/anaconda3/bin/python3.12
"""M2232 independent raw-to-receipt recomputation; no production invocation."""
import hashlib
import json
from pathlib import Path
import re

HW = Path(__file__).resolve().parents[2]
REPO = HW.parent
RESULT = HW / "results/m2231_m2215_causal_parse_only_successor_r1_20260905"
ATTEMPT = HW / "results/.m2231_causal_parse_only_attempt_consumed"
SOURCE_REVIEW = HW / "reviews/m2230_m2229_causal_parse_only_source_hammer_r1_20260905"
Q = HW / "results/m2215_m2213_preread_postread_causal_directed_vcs_r1_20260904.failed_or_incomplete.3812622.quarantine"
CONTRACT = HW / "contracts/m2229_m2215_causal_parse_only_source_contract_r1_20260905.json"


def sha(path):
    assert path.is_file() and not path.is_symlink(), str(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(path):
    assert path.is_dir() and not path.is_symlink()
    paths = list(path.rglob("*"))
    assert not any(item.is_symlink() for item in paths)
    members = {}
    for line in (path / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert name not in members and not Path(name).is_absolute() and ".." not in Path(name).parts
        assert sha(path / name) == digest
        members[name] = digest
    assert set(members) == {str(item.relative_to(path)) for item in paths if item.is_file()} - {
        "SHA256SUMS", "SHA256SUMS.seal.sha256"}
    assert (path / "SHA256SUMS.seal.sha256").read_text().split() == [sha(path / "SHA256SUMS"), "SHA256SUMS"]
    return members


def main():
    contract = json.loads(CONTRACT.read_text())
    receipt = json.loads((RESULT / "receipt.json").read_text())
    manifests = {str(path.relative_to(HW)): seal(path) for path in (RESULT, ATTEMPT, SOURCE_REVIEW)}
    for rel in contract["sealed_directories"]:
        manifests[rel] = seal(REPO / rel)
    for rel, digest in contract["pinned_files"].items():
        assert sha(REPO / rel) == digest
    assert receipt["pinned_input_files"] == contract["pinned_files"]
    assert receipt["source_contract_sha256"] == sha(CONTRACT)
    assert receipt["m2230_review_sha256"] == sha(SOURCE_REVIEW / "review.json")
    original = json.loads((HW / contract["m2213_contract"]).read_text())
    for rel, digest in original["source_inventory"].items():
        assert sha(REPO / rel) == digest
    assert (Q / "RUN_FAILED_OR_INCOMPLETE.txt").read_text() == "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=1\nretry=false\n"
    comp = (Q / "vcs_compile.log").read_text()
    sim = (Q / "simv.log").read_text()
    assert (Q / "simv.rc").read_text().strip() == "0"
    assert "Chronologic VCS (TM)" in comp and "All of 7 modules done" in comp and "to link" in comp
    assert "Chronologic VCS simulator copyright" in sim and "Runtime version V-2023.12-SP1_Full64" in sim
    assert not re.search(r"Error(?:-|\[|:)|Fatal(?:-|:)|\$fatal|assertion\s+failed", comp + sim, re.I)
    ledger = {}
    for prefix in ("M2213_COVER", "RAW_PASS_M2215_M2213_PREREAD_POSTREAD_CAUSAL_DIRECTED"):
        rows = [line for line in sim.splitlines() if line.startswith(prefix + " ")]
        assert len(rows) == 1
        for token in rows[0].split()[1:]:
            key, value = token.split("=")
            assert key not in ledger and value.isdecimal()
            ledger[key] = int(value)
    assert ledger == receipt["raw_log_ledger"]
    assert [ledger[key] for key in ("ordinary_reads", "postread_reads", "preread_reads")] == [2304, 2304, 576]
    assert [ledger[key] for key in ("ordinary_cycles", "postread_cycles", "preread_cycles")] == [3386, 3386, 1119]
    assert ledger["ordinary_reads"] - ledger["preread_reads"] == ledger["suppressed_reads"] == ledger["postread_bank_req"] == ledger["postread_bank_rsp"] == 1728
    assert ledger["postread_bundle_req"] == ledger["postread_bundle_rsp"] == ledger["identity_rsp"] == 216
    assert ledger["rows"] == ledger["commits_each"] == 24 and ledger["products_each"] == 4608
    assert ledger["hits_post"] == ledger["hits_pre"] == ledger["real_postread_rows"] == 18
    assert ledger["golden_mismatches"] == 0
    covers = re.findall(r"sva_postread\.(cp_\w+), (\d+) attempts, (\d+) match", sim)
    assert len(covers) == 3 and all(int(attempts) == 3443 for _, attempts, _ in covers)
    assert {key: int(matches) for key, _, matches in covers} == receipt["sva_matches"] == {
        "cp_real_postread_request": 552, "cp_real_postread_response": 1932,
        "cp_postread_commit_terminal": 4}
    assert receipt["directed_request_reduction"] == 1 - 576 / 2304 == 0.75
    assert receipt["execution"] == {"cpu_parse_runs": 1, "eda_runs": 0, "gpu_runs": 0,
                                     "license_queries": 0, "retry": False}
    status = "RAW_PASS_M2231_M2215_PARSE_ONLY_PENDING_M2232_RESULT_REVIEW"
    assert receipt["status"] == status
    assert (RESULT / "RUN_COMPLETE.txt").read_text() == (RESULT / "parser.stdout.log").read_text() == status + "\n"
    assert (RESULT / "parser.stderr.log").read_bytes() == b""
    assert (ATTEMPT / "ATTEMPT_CONSUMED.txt").read_text() == "M2231_CPU_PARSE_ATTEMPT_CONSUMED\nretry=false\neda_runs=0\n"
    assert not (HW / "results/.m2231_causal_parse_only_lock").exists()
    assert not list((HW / "results").glob(".m2231_causal_parse_only_work.*"))
    assert not list((HW / "results").glob("m2231_m2215_causal_parse_only_successor_r1_20260905.failed.*"))
    print(json.dumps({
        "status": "PASS_M2232_INDEPENDENT_RAW_TO_RECEIPT_CHECKS",
        "exhaustive_sealed_directories": len(manifests),
        "result_members": sorted(manifests[str(RESULT.relative_to(HW))]),
        "result_receipt_sha256": sha(RESULT / "receipt.json"),
        "result_manifest_sha256": sha(RESULT / "SHA256SUMS"),
        "result_outer_seal_sha256": sha(RESULT / "SHA256SUMS.seal.sha256"),
        "attempt_manifest_sha256": sha(ATTEMPT / "SHA256SUMS"),
        "directed_reads": [2304, 2304, 576],
        "directed_cycles": [3386, 3386, 1119],
        "directed_request_reduction": 0.75,
        "golden_mismatches": 0,
        "raw_and_docs359_unchanged": True,
        "source_parser_production_rerun": False,
        "eda_or_license_invoked": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
