#!/usr/bin/env python3
"""Run the frozen hardware profile with an external M32 identity tracer."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from m32_dataflow_identity_trace import M32DataflowIdentityWriter, sha256


def _required_option(arguments, name):
    if arguments.count(name) != 1:
        raise ValueError(
            "profile option {} must occur exactly once, got {}".format(
                name, arguments.count(name)
            )
        )
    index = arguments.index(name)
    if index + 1 >= len(arguments):
        raise ValueError("missing value for profile option {}".format(name))
    return arguments[index + 1]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile-script", type=Path, required=True)
    parser.add_argument("--m32-trace-contract", type=Path, required=True)
    parser.add_argument("--m32-candidate-report", type=Path, required=True)
    parser.add_argument("--m32-dataflow-digest-dir", type=Path, required=True)
    wrapper_args, profile_args = parser.parse_known_args()

    profile_script = wrapper_args.profile_script.resolve()
    trace_contract_path = wrapper_args.m32_trace_contract.resolve()
    candidate_report = wrapper_args.m32_candidate_report.resolve()
    output_dir = wrapper_args.m32_dataflow_digest_dir.resolve()
    config = Path(_required_option(profile_args, "--config")).resolve()
    checkpoint = Path(_required_option(profile_args, "--checkpoint")).resolve()
    profile_output_dir = Path(
        _required_option(profile_args, "--output-dir")
    ).resolve()
    samples = int(_required_option(profile_args, "--samples"))
    if samples != 10:
        raise ValueError("M32 frozen dataflow contract requires exactly 10 samples")
    for path in (
        profile_script, trace_contract_path, candidate_report, config, checkpoint,
    ):
        if not path.is_file():
            raise ValueError("missing M32 profile input: {}".format(path))
    if output_dir.exists() or profile_output_dir.exists():
        raise ValueError("refusing to overwrite M32 dataflow/profile output")
    if output_dir == profile_output_dir:
        raise ValueError("M32 dataflow and profile output directories must differ")

    trace_contract = json.loads(
        trace_contract_path.read_text(encoding="utf-8")
    )
    if (
        trace_contract.get("schema")
        != "m32_dataflow_trace_input_contract_v1"
        or trace_contract.get("status")
        != "FROZEN_H67_EP35_TEN_SAMPLE_DYNAMIC_IDENTITY"
        or int(trace_contract.get("samples", -1)) != samples
    ):
        raise ValueError("unexpected M32 dataflow trace contract")
    actual_hashes = {
        "candidate_report": sha256(candidate_report),
        "checkpoint": sha256(checkpoint),
        "config": sha256(config),
        "profile_script": sha256(profile_script),
    }
    for name, actual in sorted(actual_hashes.items()):
        expected = trace_contract["inputs"][name]["sha256"]
        if actual != expected:
            raise ValueError(
                "M32 trace input hash drift for {}: {} != {}".format(
                    name, actual, expected
                )
            )

    sys.path.insert(0, str(profile_script.parent))
    spec = importlib.util.spec_from_file_location("m32_frozen_profile", str(profile_script))
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)

    run_identity = {
        "profile_script": str(profile_script),
        "profile_script_sha256": actual_hashes["profile_script"],
        "trace_contract": str(trace_contract_path),
        "trace_contract_sha256": sha256(trace_contract_path),
        "candidate_report": str(candidate_report),
        "candidate_report_sha256": actual_hashes["candidate_report"],
        "wrapper_script": str(Path(__file__).resolve()),
        "wrapper_script_sha256": sha256(Path(__file__).resolve()),
        "writer_script": str(Path(__file__).with_name(
            "m32_dataflow_identity_trace.py"
        ).resolve()),
        "writer_script_sha256": sha256(Path(__file__).with_name(
            "m32_dataflow_identity_trace.py"
        ).resolve()),
        "config": str(config),
        "config_sha256": actual_hashes["config"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_hashes["checkpoint"],
        "profile_output_dir": str(profile_output_dir),
        "samples": samples,
    }
    writer = M32DataflowIdentityWriter(
        output_dir=output_dir,
        candidate_report=candidate_report,
        expected_samples=samples,
        run_identity=run_identity,
    )
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
        writer.bind_postrun_evidence(
            profile_output_dir / "nts11_hardware_p0_profile.json",
            profile_output_dir / "sample_workload.csv",
        )
        manifest = writer.close()
        print(writer.manifest_path)
        return 0 if manifest["status"].startswith("PASS_") else 1
    except Exception as exc:
        if not writer.closed:
            writer.abort(exc)
        raise
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    sys.exit(main())
