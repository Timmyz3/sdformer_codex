#!/usr/bin/env python3
"""Independent M1548 hammer; never launches GPU, SSH, or capture."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import tarfile
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HW = ROOT / "hw_autoresearch_nts07"
BUILDER = HW / "system_handoff/scripts/build_m1544_ep34_sparse_capture_handoff_pack.py"
VALIDATOR = HW / "system_handoff/scripts/validate_m1544_ep34_sparse_capture_handoff.py"
AUTHOR_TEST = HW / "tests/test_validate_m1544_ep34_sparse_capture_handoff.py"
CONTRACT = HW / "contracts/m1544_ep34_s2_tsbg_shared_incremental_capture_handoff_source_contract_r1_20260831.json"
ARCHIVE = HW / "system_handoff/packs/m1544_ep34_sparse_capture_handoff_r1_20260831.tar"
ARCHIVE_SHA = "b111f7e81452e9eea482e3905432fb75e29e4ae08bce5d1d88b0d17cb61bce12"
EXPECTED = {
    BUILDER: "a7b0592ef21c1ea2e0c9af93ac1bfc080bfb506629e89e025276f0ebd81c2410",
    VALIDATOR: "463fa7392fa090eda7fdb298fcc10ff896f91a961a0a529a013be2eec47ec240",
    AUTHOR_TEST: "39e3dd43a0364185a4d9725522ce3cd33737f5272dcac29acd8b98c51c587c3d",
    CONTRACT: "ea1ee88ce9300eaba914d62ffea8936083132fedb23f44c7d55447c0c1c20576",
    ARCHIVE: ARCHIVE_SHA,
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


for path, expected in EXPECTED.items():
    assert digest(path) == expected, "pinned M1544 byte drift: " + str(path)

B = load("m1548_builder", BUILDER)
V = load("m1548_validator", VALIDATOR)
T = load("m1548_author_fixture", AUTHOR_TEST)


def rejected(name, function, attacks):
    try:
        function()
    except Exception:
        attacks.append(name)
        return
    raise AssertionError("attack accepted: " + name)


def original_entries():
    rows = []
    with tarfile.open(str(ARCHIVE), "r:") as handle:
        for member in handle.getmembers():
            stream = handle.extractfile(member)
            rows.append([copy.copy(member), stream.read() if stream is not None else b""])
    return rows


def write_archive(path, entries):
    with tarfile.open(str(path), "w", format=tarfile.PAX_FORMAT) as handle:
        for member, payload in entries:
            # Do not let a copied PAX path mask a mutated TarInfo.name.
            member.pax_headers = {}
            if member.isfile():
                member.size = len(payload)
                handle.addfile(member, io.BytesIO(payload))
            else:
                member.size = 0
                handle.addfile(member)


def archive_attack(name, mutation, root, attacks):
    entries = original_entries()
    mutation(entries)
    path = root / (name + ".tar")
    write_archive(path, entries)
    rejected("tar_" + name, lambda: B.verify(path, digest(path)), attacks)


def capture_attack(name, mutation, root, attacks):
    target = root / ("capture_" + name)
    target.mkdir()
    T.build_fixture(target)
    mutation(target)
    T.reseal(target)
    rejected("capture_" + name, lambda: V.validate_capture(target), attacks)


def mutate_json(path, callback):
    value = json.loads(path.read_text(encoding="utf-8"))
    callback(value)
    T.dump_json(path, value)


def main():
    attacks = []
    B.verify(ARCHIVE, ARCHIVE_SHA)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["execution"] == {
        "source_creation_only": True, "gpu_started": False,
        "ssh_started": False, "capture_started": False,
        "automatic_retry": False, "remote_free_space_floor_gib": 16,
        "preflight_estimated_result_gib_max": 12,
        "abort_before_checkpoint_load_if_estimate_exceeds_cap": True}
    assert contract["admission"]["production_capture_authorized_by_this_contract"] is False
    assert contract["admission"]["new_rtl_authorized"] is False
    assert contract["quantization_boundary"]["producer_codebook_hardware_quant_authority"] is False
    assert contract["quantization_boundary"]["tsbg_exact_scope"] == (
        "captured_codeword_and_contributor_only")

    with tarfile.open(str(ARCHIVE), "r:") as handle:
        members = handle.getmembers()
        assert len(members) == len(set(member.name for member in members)) == 7
        assert all(member.isfile() and member.uid == 0 and member.gid == 0 and
                   member.mtime == 0 and member.mode == 0o644 and
                   not member.linkname for member in members)
        payloads = {member.name: handle.extractfile(member).read() for member in members}
    pack_name = "hw_autoresearch_nts07/system_handoff/m1544_ep34_sparse_capture_handoff_r1_20260831/PACK_MANIFEST.json"
    pack = json.loads(payloads[pack_name].decode("utf-8"))
    assert pack["compactness"] == {
        "checkpoint_bytes": 0, "full_tensor_bytes": 0, "m1458_payload_bytes": 0}
    assert pack["claim_boundary"] == {
        "aee": False, "capture": False, "cycles": False, "energy": False,
        "remote_transfer_executed": False}
    assert ARCHIVE.stat().st_size == 71680

    sample_name = "hw_autoresearch_nts07/system_handoff/m1544_ep34_sparse_capture_handoff_r1_20260831/sample_order.json"
    sample_value = V.strict_json_bytes(payloads[sample_name], "packed sample order")
    samples = V.validate_sample_order(sample_value)
    assert len(samples) == 40 and [row["global_sample_id"] for row in samples] == list(range(40))

    with tempfile.TemporaryDirectory(prefix="m1548_hammer.") as directory:
        root = Path(directory)
        rejected("fixed_archive_sha", lambda: B.verify(ARCHIVE, "0" * 64), attacks)
        link = root / "archive.link"; link.symlink_to(ARCHIVE)
        rejected("archive_symlink", lambda: B.verify(link, ARCHIVE_SHA), attacks)

        archive_attack("duplicate", lambda rows: setattr(rows[-1][0], "name", rows[0][0].name), root, attacks)
        archive_attack("traversal", lambda rows: setattr(rows[0][0], "name", "../escape"), root, attacks)
        archive_attack("absolute", lambda rows: setattr(rows[0][0], "name", "/escape"), root, attacks)
        archive_attack("backslash", lambda rows: setattr(rows[0][0], "name", "bad\\name"), root, attacks)
        archive_attack("symlink_member", lambda rows: (setattr(rows[0][0], "type", tarfile.SYMTYPE), setattr(rows[0][0], "linkname", "target")), root, attacks)
        archive_attack("char_device", lambda rows: setattr(rows[0][0], "type", tarfile.CHRTYPE), root, attacks)
        archive_attack("directory", lambda rows: setattr(rows[0][0], "type", tarfile.DIRTYPE), root, attacks)
        archive_attack("uid", lambda rows: setattr(rows[0][0], "uid", 1), root, attacks)
        archive_attack("mode", lambda rows: setattr(rows[0][0], "mode", 0o755), root, attacks)
        archive_attack("mtime", lambda rows: setattr(rows[0][0], "mtime", 1), root, attacks)
        archive_attack("missing", lambda rows: rows.pop(), root, attacks)

        def add_extra(rows):
            item = tarfile.TarInfo("extra.bin")
            item.mode = 0o644; item.uid = 0; item.gid = 0; item.mtime = 0
            rows.append([item, b"extra"])
        archive_attack("extra", add_extra, root, attacks)

        def payload_drift(rows):
            rows[0][1] += b"\n"
        archive_attack("payload_drift", payload_drift, root, attacks)

        def manifest_change(rows, field, value):
            for row in rows:
                if row[0].name == pack_name:
                    manifest = json.loads(row[1].decode("utf-8"))
                    if field == "count":
                        manifest["counts"]["total_files"] = value
                    elif field == "status":
                        manifest["status"] = value
                    else:
                        manifest["entries"][0]["sha256"] = value
                    row[1] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        archive_attack("manifest_count", lambda rows: manifest_change(rows, "count", 8), root, attacks)
        archive_attack("manifest_status", lambda rows: manifest_change(rows, "status", "CAPTURE_AUTHORIZED"), root, attacks)
        archive_attack("manifest_entry_sha", lambda rows: manifest_change(rows, "sha", "0" * 64), root, attacks)

        capture_attack("checkpoint", lambda r: mutate_json(
            r / "capture_manifest.json", lambda v: v["identity"].update(
                {"checkpoint_sha256": "0" * 64})), root, attacks)
        capture_attack("m1458_outer", lambda r: mutate_json(
            r / "capture_manifest.json", lambda v: v["identity"].update(
                {"m1458_outer_file_sha256": "0" * 64})), root, attacks)
        capture_attack("sample_order", lambda r: mutate_json(
            r / "sample_order.json", lambda v: v["samples"][0].update(
                {"sample_key": "substitute.npy"})), root, attacks)
        capture_attack("s1_gate", lambda r: mutate_json(
            r / "capture_manifest.json", lambda v: v["admission_gates"]["S1"].update(
                {"metadata_plus_beta_over_saved_weight_bytes_veto": 0.26})), root, attacks)
        capture_attack("s2_gate", lambda r: mutate_json(
            r / "capture_manifest.json", lambda v: v["admission_gates"]["S2"].update(
                {"total_metadata_over_weight_bytes_max": 0.03})), root, attacks)
        capture_attack("tsbg_gate", lambda r: mutate_json(
            r / "capture_manifest.json", lambda v: v["admission_gates"]["TSBG"].update(
                {"every_sequence_cycle_speedup_min": 1.0})), root, attacks)
        capture_attack("quant_authority", lambda r: mutate_json(
            r / "layers.json", lambda v: v["layers"][0]["codebook"].update(
                {"hardware_quant_authority": True})), root, attacks)
        capture_attack("model_bit_exact", lambda r: mutate_json(
            r / "capture_manifest.json", lambda v: v["claim_boundary"].update(
                {"model_bit_exact": True})), root, attacks)
        capture_attack("exact_scope", lambda r: mutate_json(
            r / "capture_manifest.json", lambda v: v["claim_boundary"].update(
                {"tsbg_exact_scope": "model_bit_exact"})), root, attacks)

        def sign_attack(r):
            rows = T.M1544ValidatorTest.read_rows(type("X", (), {"root": r})(),
                                                   "token_source_groups.jsonl.zlib")
            rows[0]["groups"][0]["sign_hex"] = "06"
            T.dump_zlib_jsonl(r / "token_source_groups.jsonl.zlib", rows)
        capture_attack("sign_outside_support", sign_attack, root, attacks)

        def nonunit_attack(r):
            rows = T.M1544ValidatorTest.read_rows(type("X", (), {"root": r})(),
                                                   "token_source_groups.jsonl.zlib")
            rows[0]["groups"][0]["nonunit_hex"] = "00"
            T.dump_zlib_jsonl(r / "token_source_groups.jsonl.zlib", rows)
        capture_attack("nonunit_code_disagree", nonunit_attack, root, attacks)

        def tensor_attack(r):
            (r / "full_tensor.fp32").write_bytes(b"forbidden")
            # Do not reseal the extra member; population validation must reject it.
        capture_attack("full_tensor", tensor_attack, root, attacks)

    assert len(attacks) == 30
    result = {
        "schema": "m1548_m1544_handoff_independent_hammer_r1_v1",
        "status": "PASS_M1548_SOURCE_ONLY_HANDOFF__TRANSFER_AND_INTEGRATION_ONLY__NO_CAPTURE",
        "archive_sha256": ARCHIVE_SHA,
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_members": 7,
        "s40_samples": 40,
        "independent_attacks_rejected": len(attacks),
        "attack_names": attacks,
        "m1541_p1_gates": {"S1": True, "S2": True, "TSBG": True},
        "hardware_quantization_authority": False,
        "tsbg_exact_scope": "captured_codeword_and_contributor_only",
        "gpu_started": False, "ssh_started": False, "capture_started": False,
        "authorization": {
            "exact_archive_transfer": True,
            "remote_producer_integration": True,
            "gpu_or_capture_execution": False,
            "performance_claim": False,
            "rtl": False,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
