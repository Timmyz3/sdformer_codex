#!/usr/bin/env python3
"""Seal the M1576 independent dual-runtime verdict."""

import hashlib
import json
from pathlib import Path


OUT = Path(__file__).resolve().parent
R310 = OUT / "cpython310_rehammer.json"
R36 = OUT / "cpython36_rehammer.json"
SOURCE_COMMIT = "59be0c3f642fe91c0271e00e746594b74fc17f76"
SOURCE_SHA256 = "4bf055ff31510a41882de219898e583509ccab7e9dc841aabcb6d52b20a07bf9"
STATUS = "NO_GO_M1576_M1574_EXACT_TYPE_PROVENANCE_FORGEABLE_VIA_OBJECT_NEW__SUCCESSOR_FIX_ONLY__NO_REMOTE_NO_CAPTURE"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    require(type(value) is dict, "JSON root drift")
    return value


def compact_forgery(runtime, kind):
    row = runtime["forgery_evidence"][kind]
    return {
        "schema": row["schema"],
        "provenance": row["provenance"],
        "output": row["output"],
        "inventory_sha256": row["inventory_sha256"],
        "free_bytes_before": row["free_bytes_before"],
        "free_bytes_after_upper": row["free_bytes_after_upper"],
        "consumed": row["consumed"],
        "checkpoint_loaded": row["checkpoint_loaded"],
    }


def main():
    r310 = load(R310)
    r36 = load(R36)
    for row, version in ((r310, "3.10.18"), (r36, "3.6.8")):
        require(row["status"] == "NO_GO_M1576_PROVENANCE_FORGERY_SURVIVES",
                "runtime status drift")
        require(row["runtime"]["version"] == version, "runtime version drift")
        require(row["attacks"]["count"] == 30 and
                row["attacks"]["passed"] == 28 and row["attacks"]["failed"] == 2,
                "attack count drift")
        failed = [item["name"] for item in row["attacks"]["rows"]
                  if not item["passed"]]
        require(failed == ["12_object_new_production_forgery_rejected",
                           "13_object_new_synthetic_forgery_rejected"],
                "failure identity drift")
        disk = row["production_real_disk_usage"]
        require(disk["query_count"] == 2 and
                disk["caller_supplied_free_argument"] is False and
                disk["first_receipt_free_bytes_before"] ==
                disk["first_query"]["free"] and disk["first_query"]["free"] > 0,
                "real disk usage witness drift")
        synthetic = row["synthetic_result_disk_usage"]
        require(synthetic["result_validated"] ==
                "PASS_M1558_INCREMENTAL_BINARY_VALIDATION" and
                synthetic["provenance"] == "SYNTHETIC_CALLER_BUDGET" and
                synthetic["logical_bytes"] > 0 and synthetic["allocated_bytes_st_blocks_x_512"] > 0,
                "synthetic result witness drift")
        require(row["forgery_evidence"]["production"]["provenance"] ==
                "PRODUCTION_REAL_DISK" and
                row["forgery_evidence"]["synthetic"]["provenance"] ==
                "SYNTHETIC_CALLER_BUDGET",
                "forgery evidence drift")
        require(all(value is False for value in row["side_effects"].values()),
                "side-effect boundary drift")

    failed_names = [item["name"] for item in r310["attacks"]["rows"]
                    if not item["passed"]]
    checks = {
        "schema": "m1576_m1574_permit_provenance_independent_mechanical_checks_r1_v1",
        "status": STATUS,
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "dual_runtime": {
            "cpython310": {"version": r310["runtime"]["version"],
                           "attacks": 30, "passed": 28, "failed": 2,
                           "result_sha256": sha256(R310)},
            "cpython36": {"version": r36["runtime"]["version"],
                          "attacks": 30, "passed": 28, "failed": 2,
                          "result_sha256": sha256(R36)},
        },
        "failed_attacks": failed_names,
        "real_disk_usage": {
            "cpython310": r310["production_real_disk_usage"],
            "cpython36": r36["production_real_disk_usage"],
            "receipt_matches_real_query": True,
        },
        "synthetic_actual_disk": {
            "cpython310": r310["synthetic_result_disk_usage"],
            "cpython36": r36["synthetic_result_disk_usage"],
        },
        "side_effects": r310["side_effects"],
    }

    review = {
        "schema": "m1576_m1574_permit_provenance_independent_rehammer_r1_v1",
        "status": STATUS,
        "identity": {
            "source_commit": SOURCE_COMMIT,
            "source_sha256": SOURCE_SHA256,
            "contract_sha256": "c86f8d656824aff89a5767c83b3fe7e9468fa7f2338a9053a9985f03a9d06a52",
            "author_review_sha256": "caa944692d31067a7049209c2bc0bfc34e84daefd8e268b4973e42892774733c",
            "predecessor_review_sha256": "b77da40d87fea49aab62ee56db129fbcc42f1f8063cd2d5800f690b8afd013ed",
            "docs359_sha256": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
        },
        "verdict": {
            "m1574_independent_admission": False,
            "p0": "Both exact permit classes can be instantiated with object.__new__, their name-mangled slots can be populated without either constructor secret, and consume() then emits a typed provenance receipt because it does not revalidate closure-minted membership.",
            "production_impact": "A local caller can forge an exact _ProductionPreloadPermit and obtain a PRODUCTION_REAL_DISK receipt with caller-chosen output, inventory, estimate and free bytes without executing shutil.disk_usage.",
            "synthetic_impact": "The analogous synthetic permit is forgeable; ordinary production/synthetic cross-type checks still work but are not an authority boundary.",
            "minimal_successor": "Maintain a closure-owned minted-instance registry (separate for production and synthetic), require and atomically remove membership inside consume(), and add object.__new__ plus slot-population attacks under both runtimes. Constructor-only secrets are insufficient when consume() trusts writable instance state.",
        },
        "positive_findings": {
            "public_and_closure_signatures_output_only": True,
            "actual_shutil_disk_usage_called": True,
            "actual_free_equals_receipt_free": True,
            "ordinary_exact_type_cross_provenance_rejected": True,
            "tiny_synthetic_roundtrip_validated": True,
            "synthetic_logical_bytes": r310["synthetic_result_disk_usage"]["logical_bytes"],
            "synthetic_allocated_bytes": r310["synthetic_result_disk_usage"]["allocated_bytes_st_blocks_x_512"],
            "dual_runtime_other_attacks_passed": 28,
        },
        "forgery_evidence": {
            "cpython310_production": compact_forgery(r310, "production"),
            "cpython310_synthetic": compact_forgery(r310, "synthetic"),
            "cpython36_production": compact_forgery(r36, "production"),
            "cpython36_synthetic": compact_forgery(r36, "synthetic"),
        },
        "authorization": {
            "source_only_successor_fix": True,
            "independent_rehammer_after_fix": True,
            "remote_wrapper": False,
            "ssh": False,
            "checkpoint_load": False,
            "gpu": False,
            "capture": False,
            "production_payload": False,
            "release": False,
            "automatic_retry": False,
            "rtl": False,
            "eda": False,
        },
        "claim_boundary": {
            "source_security_review_only": True,
            "aee": False, "cycles": False, "traffic": False,
            "energy": False, "speedup": False, "system_speedup": False,
            "rtl": False, "eda": False, "paper_headline": False,
        },
    }

    review_md = """# M1576 — M1574 permit provenance independent rehammer

Status: **{status}**.

Both CPython 3.10.18 and 3.6.8 ran the same 30 independent attacks. Each passed
28/30. The two failures are identical: `object.__new__` can allocate either
exact permit class without invoking its secret-checking constructor; ordinary
name-mangled slots can then be populated. `consume()` checks exact type and the
slot values but does not check that the instance was minted by the closure.
The forged production object therefore emitted a `PRODUCTION_REAL_DISK`
receipt without executing the production issuer or `shutil.disk_usage`.

The narrower M1574 repairs are real. The public and closure production
signatures accept only `output`; the normal production path called the actual
`shutil.disk_usage` twice per runtime, and the first returned free-byte value
exactly equals its receipt. Normal production/synthetic cross-type attacks
were rejected. The tiny synthetic roundtrip validated and occupied {allocated}
allocated bytes ({logical} logical bytes). These positives do not rescue the
authority claim because an exact-type forged object bypasses the issuer.

The next authorized work is source-only: keep a closure-owned registry of
minted production and synthetic instances, require and atomically remove
membership in `consume()`, and retain the two `object.__new__` attacks in both
runtimes. Remote wrapping, SSH, checkpoint load, GPU, capture, production
payload, release, RTL and EDA remain forbidden. No accuracy or performance
claim is created.
""".format(status=STATUS,
           allocated=r310["synthetic_result_disk_usage"]["allocated_bytes_st_blocks_x_512"],
           logical=r310["synthetic_result_disk_usage"]["logical_bytes"])

    targets = [OUT / "mechanical_checks.json", OUT / "review.json",
               OUT / "review.md", OUT / "RUN_COMPLETE.txt",
               OUT / "SHA256SUMS", OUT / "SHA256SUMS.seal.sha256"]
    require(all(not path.exists() for path in targets), "refuse overwrite")
    (OUT / "mechanical_checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    (OUT / "review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    (OUT / "review.md").write_text(review_md, encoding="utf-8")
    (OUT / "RUN_COMPLETE.txt").write_text(STATUS + "\n", encoding="ascii")
    members = [OUT / "independent_rehammer.py", R310, R36, Path(__file__).resolve(),
               OUT / "mechanical_checks.json", OUT / "review.json",
               OUT / "review.md", OUT / "RUN_COMPLETE.txt"]
    sums = OUT / "SHA256SUMS"
    sums.write_text("".join("{}  {}\n".format(sha256(path), path.name)
                             for path in members), encoding="ascii")
    (OUT / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(sums)), encoding="ascii")
    print(STATUS)


if __name__ == "__main__":
    main()
