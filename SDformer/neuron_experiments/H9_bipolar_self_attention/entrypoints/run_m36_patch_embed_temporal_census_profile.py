#!/usr/bin/env python3
"""Run the frozen H67 profile with M36 patch-embed census hooks."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from m36_patch_embed_temporal_census import M36PatchEmbedCensus, sha256


def required_option(arguments, name):
    if arguments.count(name) != 1:
        raise ValueError("profile option {} must occur once".format(name))
    index = arguments.index(name)
    if index + 1 >= len(arguments):
        raise ValueError("missing value for {}".format(name))
    return arguments[index + 1]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile-script", type=Path, required=True)
    parser.add_argument("--m36-contract", type=Path, required=True)
    parser.add_argument("--m36-output-dir", type=Path, required=True)
    wrapper_args, profile_args = parser.parse_known_args()
    profile_script = wrapper_args.profile_script.resolve()
    contract_path = wrapper_args.m36_contract.resolve()
    output_dir = wrapper_args.m36_output_dir.resolve()
    config = Path(required_option(profile_args, "--config")).resolve()
    checkpoint = Path(required_option(profile_args, "--checkpoint")).resolve()
    profile_output_dir = Path(
        required_option(profile_args, "--output-dir")
    ).resolve()
    samples = int(required_option(profile_args, "--samples"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != "m36_patch_embed_temporal_census_contract_v1"
        or contract.get("status")
        != "FROZEN_H67_EP35_S10_PATCH_EMBED_INPUT_CENSUS"
        or int(contract.get("samples", -1)) != samples
        or samples != 10
    ):
        raise ValueError("unexpected M36 contract")
    for path in (profile_script, contract_path, config, checkpoint):
        if not path.is_file():
            raise ValueError("missing M36 input {}".format(path))
    actual_hashes = {
        "profile_script": sha256(profile_script),
        "config": sha256(config),
        "checkpoint": sha256(checkpoint),
    }
    for name, actual in sorted(actual_hashes.items()):
        if actual != contract["inputs"][name]["sha256"]:
            raise ValueError("M36 {} hash drift".format(name))
    if output_dir.exists() or profile_output_dir.exists():
        raise ValueError("refusing to overwrite M36 output")

    sys.path.insert(0, str(profile_script.parent))
    spec = importlib.util.spec_from_file_location("m36_frozen_profile", str(profile_script))
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    run_identity = {
        "profile_script": str(profile_script),
        "profile_script_sha256": actual_hashes["profile_script"],
        "config": str(config),
        "config_sha256": actual_hashes["config"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_hashes["checkpoint"],
        "contract": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "writer": str(Path(__file__).with_name(
            "m36_patch_embed_temporal_census.py").resolve()),
        "writer_sha256": sha256(Path(__file__).with_name(
            "m36_patch_embed_temporal_census.py").resolve()),
        "wrapper": str(Path(__file__).resolve()),
        "wrapper_sha256": sha256(Path(__file__).resolve()),
        "samples": samples,
    }
    writer = M36PatchEmbedCensus(output_dir, contract, run_identity)
    original_build_model = profile.build_model

    def traced_build_model(config_value, checkpoint_value, device_value):
        model = original_build_model(config_value, checkpoint_value, device_value)
        writer.attach(model)
        return model

    profile.build_model = traced_build_model
    original_argv = list(sys.argv)
    try:
        sys.argv = [str(profile_script)] + profile_args
        return_code = profile.main()
        if return_code not in (None, 0):
            raise RuntimeError("hardware profile returned {}".format(return_code))
        manifest_path, manifest = writer.close(
            profile_output_dir / "nts11_hardware_p0_profile.json",
            profile_output_dir / "sample_workload.csv",
        )
        print(manifest_path)
        return 0 if manifest["status"].startswith("PASS_") else 1
    except Exception as exc:
        writer.abort(exc)
        raise
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    sys.exit(main())
