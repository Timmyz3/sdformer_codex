#!/usr/bin/env python3
"""Isolated negative attacks against the production M75 catalog admission.

Unlike validate_m75_independent.py, this script intentionally imports the
production loader/installer.  It writes mutations only to a TemporaryDirectory.
"""

from __future__ import print_function

import hashlib
import json
from pathlib import Path
import sys
import tempfile

import torch
from torch import nn


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
REPO = HW.parent
OVERLAY = REPO / "neuron_experiments/H9_bipolar_self_attention/overlay"
CATALOG = HW / (
    "results/m71_h67_k16_q16_paft_codebook_dev_r1_20260823/"
    "m71_h67_k16_q16_paft_codebook.json")
PAFT_SOURCE = OVERLAY / "models/STSwinNet_SNN/pattern_paft.py"
TARGET_PAFT_SHA256 = (
    "22292b265292b4d3c00cdeb1addd3020c7b2a417adc855aa043d1394735d3bf1")
sys.path.insert(0, str(OVERLAY))

from models.STSwinNet_SNN.pattern_paft import (  # noqa: E402
    _load_catalog,
    install_pattern_paft,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Identity())
        self.conv2 = nn.Sequential(nn.Identity())


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resblocks = nn.ModuleList([Block(), Block()])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sttmultires_unet = Unet()


def expect_reject(name, function):
    try:
        function()
    except RuntimeError as error:
        return {"name": name, "rejected": True, "message": str(error)}
    return {"name": name, "rejected": False, "message": None}


def main():
    live_sha = sha256(PAFT_SOURCE)
    if live_sha != TARGET_PAFT_SHA256:
        print(json.dumps({
            "status": "TARGET_SOURCE_SUPERSEDED_REVIEW_LOG_IS_HISTORICAL",
            "target_pattern_paft_sha256": TARGET_PAFT_SHA256,
            "live_pattern_paft_sha256": live_sha,
        }, indent=2, sort_keys=True))
        raise SystemExit(3)
    catalog_sha = sha256(CATALOG)
    required = [
        expect_reject("missing_catalog_sha",
                      lambda: _load_catalog(CATALOG, {})),
        expect_reject("wrong_catalog_sha",
                      lambda: _load_catalog(
                          CATALOG, {"catalog_sha256": "0" * 64})),
        expect_reject("revoked_catalog_role",
                      lambda: _load_catalog(
                          CATALOG, {"catalog_sha256": catalog_sha})),
    ]
    if not all(row["rejected"] for row in required):
        raise AssertionError("one required fail-closed attack was accepted")

    bypass = {}
    try:
        _load_catalog(CATALOG, {
            "unit_test_allow_unpinned_revoked_catalog": True,
        })
    except RuntimeError as error:
        bypass["config_override_original_revoked_catalog"] = {
            "accepted": False, "message": str(error)}
    else:
        bypass["config_override_original_revoked_catalog"] = {
            "accepted": True, "message": None}

    with tempfile.TemporaryDirectory(prefix="m75_attack_") as temporary:
        temp = Path(temporary)
        checkpoint = temp / "checkpoint.pth"
        checkpoint.write_bytes(b"M75 independent attack checkpoint identity\n")
        checkpoint_sha = sha256(checkpoint)
        train_list_sha = "a" * 64
        payload = json.loads(CATALOG.read_text(encoding="utf-8"))
        payload["split"]["role"] = "DSEC_TRAIN_ONLY_PAFT_CALIBRATION"
        payload["split"]["train_catalog_eligible"] = True
        payload["identity"]["train_sequence_list_sha256"] = train_list_sha
        payload["identity"]["checkpoint_sha256"] = checkpoint_sha
        relabeled = temp / "relabeled_revoked_m71_catalog.json"
        relabeled.write_text(json.dumps(payload, sort_keys=True) + "\n",
                             encoding="utf-8")
        relabeled_sha = sha256(relabeled)
        operators = [str(row["operator"]) for row in payload["operators"]]
        cfg = {
            "enabled": True,
            "catalog": str(relabeled),
            "catalog_sha256": relabeled_sha,
            "train_sequence_list_sha256": train_list_sha,
            "expected_checkpoint_sha256": checkpoint_sha,
            "expected_operator_names": operators,
            "sample_vectors_per_module": 1,
            "partition_chunk": 432,
        }
        try:
            _load_catalog(relabeled, cfg)
        except RuntimeError as error:
            bypass["self_relabeled_revoked_catalog_loader"] = {
                "accepted": False, "message": str(error)}
        else:
            bypass["self_relabeled_revoked_catalog_loader"] = {
                "accepted": True, "message": None}
        try:
            installed = install_pattern_paft(
                Model(), cfg, checkpoint_path=str(checkpoint))
        except RuntimeError as error:
            bypass["self_relabeled_revoked_catalog_full_install"] = {
                "accepted": False, "message": str(error),
                "installed_operators": None}
        else:
            bypass["self_relabeled_revoked_catalog_full_install"] = {
                "accepted": True, "message": None,
                "installed_operators": installed}

    print(json.dumps({
        "required_attacks": required,
        "additional_bypass_attacks": bypass,
        "catalog_sha256": catalog_sha,
    }, indent=2, sort_keys=True))
    print("PASS_REQUIRED_ATTACKS_REJECTED")
    if any(row["accepted"] for row in bypass.values()):
        print("FAIL_FAILCLOSED_BYPASS_ACCEPTED")
        raise SystemExit(2)
    print("PASS_NO_ADDITIONAL_BYPASS")


if __name__ == "__main__":
    main()
