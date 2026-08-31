#!/usr/bin/env python3
"""Read-only independent hammer for the inert M1183 ep29 E8 release.

This checker never contacts the remote host, opens the checkpoint, acquires the
GPU lease, runs range capture, or invokes EDA.  It deliberately fails closed if
the release does not carry an exact local-to-remote transfer closure.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RELEASE = HW / "contracts/m1183_motion_ep29_e8_inert_launch_release_r1_20260830.json"
AUTHOR = HW / "reviews/m1183_motion_ep29_e1e8_inert_launch_release_author_r1_20260830"
EXPECTED = {
    RELEASE: "3bc14a2e45837be5e1c5f4c2f0042634b8428f6beaa2152a1f818e0531aa43f5",
    RELEASE.with_name(RELEASE.name + ".sha256"): "6a220dc29606fe89e0fa2b977b52e92746d916c0ae75f7976562d660b86331b8",
    RELEASE.with_name(RELEASE.name + ".sha256.seal.sha256"): "7ec2916d08265aeb0118ca53a34d0c24046485d786c0438caf77b8b30c4562e1",
    AUTHOR / "e8_author_receipt.json": "9d4deffa7d5bdb86e069d01856fbd6b3e9b89b6e67a5ffea1fe972e7b0c73e82",
    AUTHOR / "SHA256SUMS": "3009efce541bc816b58abf42a279e1be676237bd3ab07f95e39efd858f4db437",
    AUTHOR / "SHA256SUMS.seal.sha256": "505bd678768eef57a7ebc153132cb887d90842eee1da4f52e6d0655801b76c86",
    HW / "docs/359_DATE终局冻结_20260813.md": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def need(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def strict_json(path: Path) -> dict:
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          AssertionError("non-finite JSON token: " + token)))


def main() -> int:
    checks = 0
    for path, digest in EXPECTED.items():
        need(path.is_file() and not path.is_symlink(), "missing/non-regular authority: " + str(path))
        need(sha(path) == digest, "authority SHA drift: " + str(path))
        checks += 2

    release = strict_json(RELEASE)
    receipt = strict_json(AUTHOR / "e8_author_receipt.json")
    need(release.get("mode") == "e8" and "e1" not in release, "mode mix")
    need(release.get("contract_path") == str(RELEASE.relative_to(ROOT)), "release path redirection")
    need(receipt.get("remote_policy", {}).get("repo") ==
         "/root/private_data/work/sdformer_codex/SDformer", "remote repo drift")
    need(receipt.get("remote_policy", {}).get("interpreter") ==
         "/opt/conda/envs/sdformerflow/bin/python", "remote interpreter drift")
    checks += 4

    # A runnable release must say exactly which bytes are copied to which remote
    # repository paths.  Merely naming local dependencies in `common` is not a
    # transfer closure and cannot establish that the remote execution sees the
    # hammered bytes.
    transfer = release.get("remote_transfer_closure")
    need(isinstance(transfer, dict),
         "P0: remote_transfer_closure absent from exact M1183 E8 release")

    # Unreachable for the current release.  These assertions define the minimum
    # shape required from a non-overwriting successor release.
    need(transfer.get("remote_repo") == receipt["remote_policy"]["repo"],
         "remote transfer repository redirection")
    members = transfer.get("members")
    need(isinstance(members, list) and members, "empty remote transfer member set")
    required = {
        "hw_autoresearch_nts07/system_handoff/scripts/run_m1177r2_motion_ep29_e1e8_closure_source.py",
        "hw_autoresearch_nts07/contracts/m1177r2_motion_ep29_e1e8_source_contract_r1_20260830.json",
        "hw_autoresearch_nts07/tests/test_run_m1177r2_motion_ep29_e1e8_closure_source.py",
        "hw_autoresearch_nts07/reviews/m1175_m1171_motion_final_checkpoint_binder_result_hammer_r1_20260830/review.json",
        "hw_autoresearch_nts07/reviews/m1181_m1177r2_motion_ep29_e1e8_source_hammer_r1_20260830/review.json",
        "hw_autoresearch_nts07/reviews/m1181_m1177r2_motion_ep29_e1e8_source_hammer_r1_20260830/SHA256SUMS",
        "hw_autoresearch_nts07/reviews/m1181_m1177r2_motion_ep29_e1e8_source_hammer_r1_20260830/SHA256SUMS.seal.sha256",
        "hw_autoresearch_nts07/contracts/m1177r2_motion_ep29_e8_canonical_40_source_manifest_r1_20260830.json",
        "hw_autoresearch_nts07/contracts/m1177r2_motion_ep29_e8_canonical_40_source_manifest_r1_20260830.json.sha256",
        "hw_autoresearch_nts07/contracts/m1177r2_motion_ep29_e8_canonical_40_source_manifest_r1_20260830.json.sha256.seal.sha256",
        "neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py",
        "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
    }
    got = {row.get("local_path") for row in members if isinstance(row, dict)}
    need(required <= got, "direct pinned dependency missing from remote transfer set")
    need(sum(1 for path in got if isinstance(path, str) and path.endswith(".npz")) == 40,
         "canonical 40 source payload is not exactly transferred")
    print(json.dumps({"status": "PASS", "checks": checks}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
