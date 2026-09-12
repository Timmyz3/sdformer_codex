"""One fresh valid825 pass for each of the two fixed signed-PoT sources."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone

import numpy as np

from parent_network import ParentNetwork, arrays
from train_matched import dump

HERE = Path(__file__).resolve().parent
STRUCTURES = ("dense", "lifting40")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    out = HERE / "source_constant_valid825"
    out.mkdir(parents=True, exist_ok=True)
    # These are two predeclared single passes, never a resume or overwrite.
    if (out / "run.json").exists():
        raise FileExistsError("source_constant_valid825/run.json already exists")
    args.output = out
    net = ParentNetwork(args)
    activity = None
    run = None
    try:
        import torch
        from fixed_structure import LiteralForward
        from run_bn_probe import read_names
        from evaluate_branch_control import evaluate_axis

        sys.path.insert(0, str(net.op / "stage_20260912/algorithm"))
        from run_combinations import Activity

        nbpath = net.op / "accuracy_baseline/source_nb0_valid825.csv"
        if not nbpath.exists():
            nbpath = net.op / "stage_20260912/algorithm/source_nb0_valid825.csv"
        with nbpath.open() as stream:
            nbrows = list(csv.DictReader(stream))
        nb0 = {row["file"]: row for row in nbrows}
        names = read_names(args.data, "valid")
        assert len(names) == len(set(names)) == len(nbrows) == len(nb0) == 825
        assert set(names) == set(nb0)
        nbmean = float(np.mean([float(nb0[name]["AEE"]) for name in names]))
        nbpixels = sum(int(float(nb0[name]["valid_pixels"])) for name in names)
        assert abs(nbmean - 1.445352534681) < 1e-12
        assert nbpixels == 48152523
        args.split = "valid"
        activity = Activity(net.model, net.modules)
        run = dict(
            complete=False, started_utc=utc_now(), new_training=False,
            new_GT_steps=0, full_valid825=True, fresh_inference=True,
            split="valid", frames=names, selected_structures=list(STRUCTURES), axes={},
            rule="Fixed nearest at most two signed powers of two per source coefficient; same exponent/cutoff/RNE/consumers; no scan or training.",
            parent="Both students inherit the identical ordinary R24+onepass parent, common1024 TRAIN-moment initialization and common320 newGT recovery; projection follows that endpoint.",
            numeric="Reload signed16 source constants and signed24 literal RNE/saturation parameters into LiteralForward; AT-LIF {0,theta} amplitudes are static and may be folded into weights.",
            precision_rule="Strictly better than same-file/valid-pixel NB0; no +0.005 or 1.259 gate.",
            NB0_source=str(nbpath), NB0_sha256=sha256(nbpath), NB0_AEE=nbmean,
            NB0_valid_pixels=nbpixels, upstream_checkpoint=str(args.checkpoint),
            config=str(args.config), TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
            TF32_cudnn=torch.backends.cudnn.allow_tf32,
            output_head="Actual coarse preds.2 summed over T, bilinear480x640 align_corners=False; NB0 is the local upstream reproduction with its original final head, not an official-author checkpoint.",
            old_valid825_not_inherited=True, hardware_cycles=False,
            previous_halo_checks="../source_constant_aee and ../../hardware/source_constant_local_chain; two halos are not full-network equivalence.",
            runner_sha256=sha256(Path(__file__)),
        )
        dump(out / "run.json", run)
        for structure in STRUCTURES:
            package = HERE.parent / "source_constant_probe" / structure
            param_path = package / "deployed_constants.npz"
            parent_path = HERE / "matched_training" / structure / "stage320/deployed_constants.npz"
            params, parent_params = arrays(param_path), arrays(parent_path)
            assert set(params) == set(parent_params)
            changed = [key for key in params if not np.array_equal(params[key], parent_params[key])]
            assert changed == (["As_q16"] if structure == "dense" else ["lifting_q12"])
            param_hash = sha256(param_path)
            run["active_structure"] = structure
            dump(out / "run.json", run)
            net.install("ordinary")
            net.helper.restore()
            helper = LiteralForward(net.controller, net.pair.temporal.theta, params, structure)
            net.helper = helper
            bn_calls = []
            bn_forward = net.bn.forward

            def count_bn(x):
                bn_calls.append(int(x.shape[0] * x.shape[2] * x.shape[3]))
                return bn_forward(x)

            net.bn.forward = count_bn
            args.output = out / structure
            args.output.mkdir(parents=True, exist_ok=True)
            helper.frames.clear()
            activity.start(names)
            try:
                summary = evaluate_axis(args, net.model, net.current, names, structure,
                                        progress_tag="SOURCE_CONSTANT_VALID825")
                measured = json.loads((args.output / (structure + "_frames.json")).read_text())
                assert len(measured) == len(helper.frames) == len(activity.frames) == len(names)
                assert [row["file"] for row in measured] == names
                assert [row["file"] for row in activity.frames] == names
                assert all(row["valid_pixels"] == int(float(nb0[row["file"]]["valid_pixels"])) for row in measured)
                assert summary["valid_pixels"] == nbpixels
                assert sha256(param_path) == param_hash
                paired = [dict(file=row["file"], valid_pixels=row["valid_pixels"],
                               AEE=row["AEE"], NB0_AEE=float(nb0[row["file"]]["AEE"]),
                               delta=row["AEE"] - float(nb0[row["file"]]["AEE"])) for row in measured]
                dump(args.output / "paired_NB0.json", paired)
                actual = activity.finish(helper.frames, [])
                actual["onepass_calls"] = len(bn_calls)
                actual["onepass_observed_domains"] = sorted(set(bn_calls))
                actual["BN_note"] = "Actual frozen gamma/beta with full current-domain onepass statistics; minimum variance not separately captured."
                assert len(bn_calls) == len(names)
                dump(args.output / "activity_summary.json", actual)
                dump(args.output / "activity_ranges.json", helper.range_report(names))
                parent_rows = json.loads((HERE / "valid825" / structure / (structure + "_frames.json")).read_text())
                assert [row["file"] for row in parent_rows] == names
                assert all(a["valid_pixels"] == b["valid_pixels"] for a, b in zip(measured, parent_rows))
                parent_mean = float(np.mean([row["AEE"] for row in parent_rows]))
                result = dict(
                    complete=True, completed_utc=utc_now(), summary=summary,
                    NB0_AEE=nbmean, delta_NB0=summary["AEE_frame_mean"] - nbmean,
                    better_than_NB0=summary["AEE_frame_mean"] < nbmean,
                    same_frame_set=True, same_per_frame_valid_pixels=True,
                    parameters=str(param_path), parameters_sha256=param_hash,
                    unchanged_parameters_verified=True, changed_parameter_keys=changed,
                    parent_parameters=str(parent_path), parent_parameters_sha256=sha256(parent_path),
                    unquantized_parent_AEE=parent_mean,
                    delta_unquantized_parent=summary["AEE_frame_mean"] - parent_mean,
                    new_GT_steps=0, common_prior_GT_steps=320, common_recovery_GT_steps=320,
                    activity="activity_summary.json", activity_ranges="activity_ranges.json",
                    fresh_inference=True, inherited_quality=False,
                )
                dump(args.output / "quality.json", result)
                run["axes"][structure] = result
                dump(out / "run.json", run)
                print("SOURCE_CONSTANT_VALID825_DONE", structure, json.dumps(result), flush=True)
            finally:
                activity.active = False
                net.release_axis()
        run["complete"] = True
        run["active_structure"] = None
        run["completed_utc"] = utc_now()
        dump(out / "run.json", run)
        dump(out / "summary.json", dict(
            complete=True, full_valid825=True, new_training=False, new_GT_steps=0,
            NB0_AEE=nbmean, frames=825, valid_pixels=nbpixels,
            rows=[dict(structure=name, AEE_frame_mean=row["summary"]["AEE_frame_mean"],
                       AEE_pixel_mean=row["summary"]["AEE_pixel_mean"], delta_NB0=row["delta_NB0"],
                       better_than_NB0=row["better_than_NB0"],
                       delta_unquantized_parent=row["delta_unquantized_parent"])
                  for name, row in run["axes"].items()],
            hardware_cycles=False, quality_not_inherited=True,
        ))
    except BaseException as error:
        if run is not None:
            run["error"] = repr(error)
            run["failed_utc"] = utc_now()
            dump(out / "run.json", run)
        raise
    finally:
        if activity is not None:
            activity.restore()
        net.close()


if __name__ == "__main__":
    main()
