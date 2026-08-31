#!/usr/bin/env python3
"""Build and verify the deterministic source-only M1544 handoff tar.

Python 3.6 compatible; no GPU, SSH, capture, or external dependency.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import tarfile


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M1458 = HW / (
    "results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831/"
    "manifest.json")
M1458_SHA256 = "3ab8431e3d7d17d6933c0b87da4a3405e87c97ccc302a27c78491b0a02491d6d"
HANDOFF = HW / "system_handoff/m1544_ep34_sparse_capture_handoff_r1_20260831"
SAMPLE_ORDER = HANDOFF / "sample_order.json"
PACK_MANIFEST = HANDOFF / "PACK_MANIFEST.json"
OUTPUT = HW / "system_handoff/packs/m1544_ep34_sparse_capture_handoff_r1_20260831.tar"
OUTPUT_SHA = Path(str(OUTPUT) + ".sha256")

FILES = {
    "hw_autoresearch_nts07/system_handoff/scripts/validate_m1544_ep34_sparse_capture_handoff.py":
        "463fa7392fa090eda7fdb298fcc10ff896f91a961a0a529a013be2eec47ec240",
    "hw_autoresearch_nts07/tests/test_validate_m1544_ep34_sparse_capture_handoff.py":
        "39e3dd43a0364185a4d9725522ce3cd33737f5272dcac29acd8b98c51c587c3d",
    "hw_autoresearch_nts07/contracts/m1544_ep34_s2_tsbg_shared_incremental_capture_handoff_source_contract_r1_20260831.json":
        "ea1ee88ce9300eaba914d62ffea8936083132fedb23f44c7d55447c0c1c20576",
    "hw_autoresearch_nts07/system_handoff/m1544_ep34_sparse_capture_handoff_r1_20260831/capture_schema.json":
        "b8a3ed87219ac556f7f4cbb73dabe4e05f229681b2a4fa14e7f477d8a51d17db",
    "hw_autoresearch_nts07/system_handoff/m1544_ep34_sparse_capture_handoff_r1_20260831/README.md":
        "45da8c7f895649177a4e231c28b649f239fcfbdf595ed14e09ec383646b03b20",
}
SAMPLE_ORDER_SCHEMA = "m1544_ep34_m1458_sample_order_r1_v1"
ORDER_SHA256 = "88db38f9cc3f3e0b89cf332ef84958ed87e7c84873075e4399a2a54d2ce64c47"
PASS_BUILD = "PASS_M1544_SOURCE_ONLY_HANDOFF_PACK_BUILT"
PASS_VERIFY = "PASS_M1544_SOURCE_ONLY_HANDOFF_PACK_VERIFIED"


class PackError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise PackError(message)


def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha(path):
    return sha_bytes(path.read_bytes())


def safe(value):
    require(type(value) is str and value and not value.startswith("/") and "\\" not in value,
            "unsafe path")
    pure = PurePosixPath(value)
    require(str(pure) == value and all(part not in ("", ".", "..") for part in pure.parts),
            "noncanonical path")
    return value


def regular_exact(path, expected, label):
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise PackError("missing " + label) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(), label + " must be regular")
    require(sha(path) == expected, label + " SHA mismatch")


def build_sample_order():
    regular_exact(M1458, M1458_SHA256, "M1458 manifest")
    source = json.loads(M1458.read_text(encoding="utf-8"))
    rows = []
    for item in source["cohort"]["samples"]:
        rows.append({key: item[key] for key in (
            "global_sample_id", "sequence", "sequence_sample_id", "sample_key", "sha256")})
    require(len(rows) == 40 and [row["global_sample_id"] for row in rows] == list(range(40)),
            "M1458 sample population/order mismatch")
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    require(sha_bytes(canonical) == ORDER_SHA256, "M1458 order SHA mismatch")
    value = {
        "schema": SAMPLE_ORDER_SCHEMA,
        "identity": {
            "checkpoint_sha256":
                "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
            "m1458_manifest_sha256": M1458_SHA256,
            "m1458_inner_manifest_sha256":
                "f7f7a08696611875837196b990575453141b5e8edbf6d4aae61f7db1ed238b8e",
            "m1458_outer_file_sha256":
                "7cf434b834d30c003153eef8e83e70d574b1c5a7d20ca4c2208902c6e0c76eed",
        },
        "samples": rows,
    }
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    SAMPLE_ORDER.write_bytes(payload)
    return payload


def tar_info(name, payload):
    info = tarfile.TarInfo(name=safe(name))
    info.size = len(payload)
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    return info


def build():
    sample_payload = build_sample_order()
    entries = []
    payloads = {}
    for name, digest in sorted(FILES.items()):
        path = ROOT / name
        regular_exact(path, digest, name)
        payload = path.read_bytes()
        payloads[name] = payload
        entries.append({"path": name, "size_bytes": len(payload), "sha256": digest})
    sample_name = str(SAMPLE_ORDER.relative_to(ROOT))
    payloads[sample_name] = sample_payload
    entries.append({"path": sample_name, "size_bytes": len(sample_payload),
                    "sha256": sha_bytes(sample_payload)})
    manifest_name = str(PACK_MANIFEST.relative_to(ROOT))
    manifest = {
        "schema": "m1544_ep34_sparse_capture_handoff_pack_manifest_r1_v1",
        "status": "SEALED_SOURCE_ONLY_HANDOFF__NO_GPU_NO_SSH_NO_CAPTURE",
        "entries": sorted(entries, key=lambda row: row["path"]),
        "counts": {"nonmanifest_files": len(entries), "total_files": len(entries) + 1},
        "compactness": {"full_tensor_bytes": 0, "checkpoint_bytes": 0,
                        "m1458_payload_bytes": 0},
        "claim_boundary": {"remote_transfer_executed": False, "capture": False,
                           "cycles": False, "energy": False, "aee": False},
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    PACK_MANIFEST.write_bytes(manifest_payload)
    payloads[manifest_name] = manifest_payload
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(OUTPUT), mode="w", format=tarfile.PAX_FORMAT) as handle:
        for name in sorted(payloads):
            payload = payloads[name]
            handle.addfile(tar_info(name, payload), io.BytesIO(payload))
    digest = sha(OUTPUT)
    OUTPUT_SHA.write_text(digest + "  " + OUTPUT.name + "\n", encoding="ascii")
    result = {"archive": str(OUTPUT.relative_to(ROOT)), "archive_sha256": digest,
              "archive_bytes": OUTPUT.stat().st_size, "files": len(payloads),
              "capture_executed": False}
    verify(OUTPUT, digest)
    return result


def verify(path, expected_sha):
    regular_exact(path, expected_sha, "M1544 archive")
    with tarfile.open(str(path), mode="r:") as handle:
        members = handle.getmembers()
        require(len(members) == 7 and len({item.name for item in members}) == 7,
                "archive member population mismatch")
        payloads = {}
        for member in members:
            safe(member.name)
            require(member.isfile() and member.uid == 0 and member.gid == 0 and
                    member.mtime == 0 and member.mode == 0o644,
                    "archive metadata mismatch")
            stream = handle.extractfile(member)
            require(stream is not None, "cannot extract archive member")
            payloads[member.name] = stream.read()
    manifest_name = str(PACK_MANIFEST.relative_to(ROOT))
    require(manifest_name in payloads, "pack manifest missing")
    manifest = json.loads(payloads[manifest_name].decode("utf-8"))
    require(manifest["status"] == "SEALED_SOURCE_ONLY_HANDOFF__NO_GPU_NO_SSH_NO_CAPTURE" and
            manifest["counts"] == {"nonmanifest_files": 6, "total_files": 7},
            "pack manifest status/count mismatch")
    mapped = {row["path"]: row for row in manifest["entries"]}
    require(set(mapped) == set(payloads) - {manifest_name}, "pack entry set mismatch")
    for name, row in mapped.items():
        require(row["size_bytes"] == len(payloads[name]) and
                row["sha256"] == sha_bytes(payloads[name]), "pack entry identity mismatch")
    return {"archive_sha256": expected_sha, "files": 7, "capture_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    require(args.build != (args.verify is not None), "select exactly one of --build/--verify")
    if args.build:
        result = build()
        print(PASS_BUILD + " " + json.dumps(result, sort_keys=True))
    else:
        require(type(args.expected_sha256) is str and len(args.expected_sha256) == 64,
                "--expected-sha256 required")
        result = verify(args.verify.resolve(), args.expected_sha256)
        print(PASS_VERIFY + " " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
