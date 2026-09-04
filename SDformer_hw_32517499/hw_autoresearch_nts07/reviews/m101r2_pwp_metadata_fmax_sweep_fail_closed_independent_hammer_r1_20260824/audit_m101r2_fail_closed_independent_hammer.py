#!/usr/bin/env python3
"""Independent M101-r2 seal audit plus upgraded hostile fixtures."""

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
RUN = HW / "dc_handoff/runs/m101_pwp_metadata_fmax_sweep_r1_20260824"
AUDITOR = HW / "dc_handoff/scripts/audit_m101r2_pwp_metadata_fmax_sweep_fail_closed.py"
CONTRACT = HW / "contracts/m101_pwp_metadata_fmax_sweep_synopsys_contract_r1_20260824.json"
SEAL = HW / "contracts/m101r2_pwp_metadata_fmax_sweep_fail_closed_seal_contract_r1_20260824.json"
RESULT_DIR = HW / "results/m101r2_pwp_metadata_fmax_sweep_fail_closed_r1_20260824"
RECEIPT = RESULT_DIR / "m101r2_pwp_metadata_fmax_sweep_receipt_r1.json"
DURABLE = RESULT_DIR / "m101r2_durable_run_evidence_manifest_r1.sha256"
COMPLETE = RESULT_DIR / "SHA256SUMS.complete_r1.txt"
OLD_REVIEW_MANIFEST = HW / "reviews/m101_pwp_metadata_fmax_sweep_independent_hammer_r1_20260824/manifest.sha256"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUTPUT = REVIEW / "m101r2_fail_closed_independent_hammer_audit.json"

EXPECTED = {
    "auditor": "b6a902819d03d94fd145230ccd8583e90d1db8d793b78c588434a0dee8cf8ecf",
    "contract": "dad2b791d505b9532f7924b80e28cd899983e2b097f993f5b1df1c1a97a16c50",
    "seal": "c4a051ca48412fb35ac78b4536f8c6c43a2e4fba93703d6d7601201b729f7961",
    "receipt": "7b4bc78a1974f6fab058c16595ad99aed6e2d05ae770cf7df0cd01bef9b3dd47",
    "durable": "bdccb7cfdcad1cf137c863118ffdd12bb6d0b823f9465f53b6a31e9a5bfd30c2",
    "complete": "a65ac7916c05d56fbde67f7efee5b4d1caf8ec87ce21147f1d411f3c31885394",
    "old_review_manifest": "5ffc5106f4b88eae4cb302250d5c22723799080bea707fbf99b486ad2d61f9de",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

PERIODS = (2.750, 3.000, 3.250, 3.500, 3.750, 4.000, 4.250, 4.500)
TOPS = {
    "m85": "guarded_wordpacked_pwp_stream",
    "m99": "phase_slack_guarded_wordpacked_pwp_stream",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise RuntimeError("non-standard JSON constant " + raw)

    def pairs_hook(pairs):
        output = {}
        for key, value in pairs:
            require(key not in output, "duplicate JSON key " + key)
            output[key] = value
        return output

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def point_name(design, period):
    return ("{}_{:.3f}ns".format(design, period)
            .replace(".", "p").replace("pns", "ns"))


def qor_field(text, label, number=float):
    match = re.search(
        r"^\s*{}:\s+(-?[0-9]+(?:\.[0-9]+)?)\s*$".format(
            re.escape(label)), text, re.M)
    require(match is not None, "missing QoR field " + label)
    return number(match.group(1))


def worst_slack(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    values = [(float(value), status) for status, value in re.findall(
        r"slack\s+\((MET|VIOLATED)\)\s+(-?[0-9]+(?:\.[0-9]+)?)",
        text)]
    require(values, "no timing slack records in " + str(path))
    return min(values, key=lambda item: item[0])


def report_top(path, top):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    require(re.search(r"^Design\s*:\s*{}\s*$".format(re.escape(top)),
                      text, re.M), "report top mismatch " + str(path))


def independent_point(contract, design, period):
    contract_key = "m85_unrolled" if design == "m85" else "m99_phase_slack"
    frozen = contract["designs"][contract_key]
    top = TOPS[design]
    point = RUN / point_name(design, period)
    require(point.is_dir() and not point.is_symlink(),
            "bad independent point " + str(point))
    for path in point.rglob("*"):
        require(not path.is_symlink(), "point symlink " + str(path))
    required = [
        "dc.log", "dc_backend.rc", "BACKEND_COMPLETE.txt",
        "point_identity.txt", "reports/qor.rpt", "reports/area.rpt",
        "reports/clocks.rpt", "reports/timing_setup.rpt",
        "reports/timing_hold.rpt", "reports/constraint_violators.rpt",
        "reports/check_design_postcompile.rpt",
        "reports/check_timing_postcompile.rpt",
        "reports/references_postcompile.rpt",
        "reports/resources_precompile.rpt",
        "reports/resources_postcompile.rpt",
        "netlist/" + top + "_mapped.v",
        "netlist/" + top + "_mapped.sdc",
        "netlist/" + top + ".ddc",
        "netlist/" + top + ".svf",
    ]
    for relative in required:
        path = point / relative
        require(path.is_file() and path.stat().st_size > 0,
                "missing/empty independent evidence " + str(path))

    identity = (point / "point_identity.txt").read_text(
        encoding="utf-8").splitlines()
    require(len(identity) == 4
            and identity[0] == "design_key=" + design
            and identity[1] == "design_name=" + top
            and identity[2] == "clock_period_ns={:.3f}".format(period),
            "independent point identity drift " + str(point))
    filelist_parts = identity[3].split(None, 1)
    require(len(filelist_parts) == 2
            and filelist_parts[0] == frozen["filelist_sha256"]
            and Path(filelist_parts[1]).resolve()
            == (HW / frozen["filelist"]).resolve(),
            "independent point filelist identity drift " + str(point))
    clocks_path = point / "reports/clocks.rpt"
    report_top(clocks_path, top)
    clocks = clocks_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^core_clk\s+([0-9]+(?:\.[0-9]+)?)\s+",
                      clocks, re.M)
    require(match and abs(float(match.group(1)) - period) < 1e-12,
            "independent clock period mismatch " + str(point))
    for report in ("qor.rpt", "timing_setup.rpt", "timing_hold.rpt",
                   "references_postcompile.rpt"):
        report_top(point / "reports" / report, top)
    mapped_v = point / "netlist" / (top + "_mapped.v")
    mapped_text = mapped_v.read_text(encoding="utf-8", errors="replace")
    require(re.search(r"\bmodule\s+{}\b".format(re.escape(top)),
                      mapped_text), "independent mapped top mismatch")

    require((point / "dc_backend.rc").read_text().strip() == "0"
            and (point / "BACKEND_COMPLETE.txt").read_text().strip()
            == "backend_complete=true", "independent backend incomplete")
    log = (point / "dc.log").read_text(encoding="utf-8", errors="replace")
    require(not re.search(r"^Error:", log, re.M)
            and "Current design is now '" + top + "'." in log
            and "Using operating conditions '" + contract["frozen_sweep"]
            ["operating_condition"] + "'" in log,
            "independent DC log identity mismatch")

    qor_path = point / "reports/qor.rpt"
    qor = qor_path.read_text(encoding="utf-8", errors="replace")
    setup_slack, setup_status = worst_slack(
        point / "reports/timing_setup.rpt")
    hold_slack, hold_status = worst_slack(
        point / "reports/timing_hold.rpt")
    setup_tns = qor_field(qor, "Total Negative Slack")
    setup_violations = qor_field(qor, "No. of Violating Paths")
    hold_tns = qor_field(qor, "Total Hold Violation")
    hold_violations = qor_field(qor, "No. of Hold Violations")
    no_violations = (point / "reports/constraint_violators.rpt").read_text(
        encoding="utf-8", errors="replace").count(
            "This design has no violated constraints.")
    point_pass = (setup_status == "MET" and setup_slack >= 0.0
                  and hold_status == "MET" and hold_slack >= 0.0
                  and setup_tns == 0.0 and setup_violations == 0.0
                  and hold_tns == 0.0 and hold_violations == 0.0
                  and no_violations == 5)
    return {
        "design_key": design,
        "top": top,
        "period_ns": period,
        "clock_report_period_ns": float(match.group(1)),
        "setup_worst_slack_ns": setup_slack,
        "setup_status": setup_status,
        "hold_worst_slack_ns": hold_slack,
        "hold_status": hold_status,
        "setup_tns_ns": setup_tns,
        "setup_violating_paths": setup_violations,
        "hold_tns_ns": hold_tns,
        "hold_violating_paths": hold_violations,
        "constraint_sections_without_violations": no_violations,
        "point_pass": point_pass,
        "levels_of_logic": qor_field(qor, "Levels of Logic"),
        "critical_path_length_ns": qor_field(qor, "Critical Path Length"),
        "cell_area_um2": qor_field(qor, "Cell Area"),
        "leaf_cell_count": qor_field(qor, "Leaf Cell Count", int),
        "combinational_cell_count": qor_field(
            qor, "Combinational Cell Count", int),
        "sequential_cell_count": qor_field(qor, "Sequential Cell Count", int),
        "macro_count": qor_field(qor, "Macro Count", int),
        "point_identity_sha256": sha256(point / "point_identity.txt"),
        "qor_sha256": sha256(qor_path),
        "setup_sha256": sha256(point / "reports/timing_setup.rpt"),
        "hold_sha256": sha256(point / "reports/timing_hold.rpt"),
        "mapped_verilog_sha256": sha256(mapped_v),
        "mapped_sdc_sha256": sha256(
            point / "netlist" / (top + "_mapped.sdc")),
        "ddc_sha256": sha256(point / "netlist" / (top + ".ddc")),
        "svf_sha256": sha256(point / "netlist" / (top + ".svf")),
    }


def independent_reparse(contract, receipt):
    frozen_hashes = {}
    for design, contract_key in (("m85", "m85_unrolled"),
                                 ("m99", "m99_phase_slack")):
        frozen = contract["designs"][contract_key]
        for label, path_key, sha_key in (
                ("filelist", "filelist", "filelist_sha256"),
                ("functional_contract", "functional_contract",
                 "functional_contract_sha256"),
                ("sealed_vcs", "sealed_vcs_completion",
                 "sealed_vcs_completion_sha256")):
            path = HW / frozen[path_key]
            actual = sha256(path)
            require(actual == frozen[sha_key],
                    "independent frozen input mismatch " + str(path))
            frozen_hashes[design + "_" + label] = actual
        for relative, expected in frozen["rtl_sha256"].items():
            actual = sha256(HW / relative)
            require(actual == expected,
                    "independent RTL mismatch " + relative)
            frozen_hashes[relative] = actual
    sweep = contract["frozen_sweep"]
    for label, path_key, sha_key in (
            ("tcl", "tcl", "tcl_sha256"),
            ("sdc", "sdc", "sdc_sha256")):
        actual = sha256(HW / sweep[path_key])
        require(actual == sweep[sha_key],
                "independent sweep input mismatch " + label)
        frozen_hashes[label] = actual
    setup_library, hold_library = parse_admission_libraries()
    require(sha256(setup_library) == sweep["setup_library_sha256"]
            and sha256(hold_library) == sweep["hold_library_sha256"],
            "independent library identity mismatch")
    frozen_hashes["setup_library"] = sha256(setup_library)
    frozen_hashes["hold_library"] = sha256(hold_library)

    points = {design: [independent_point(contract, design, period)
                       for period in PERIODS] for design in TOPS}
    fastest = {}
    for design, rows in points.items():
        passing = [row for row in rows if row["point_pass"]]
        require(passing, "no independent passing point " + design)
        fastest[design] = min(passing, key=lambda row: row["period_ns"])
    ratio = fastest["m85"]["period_ns"] / fastest["m99"]["period_ns"]
    area_fraction = (fastest["m99"]["cell_area_um2"]
                     / fastest["m85"]["cell_area_um2"])

    compare_keys = (
        "period_ns", "clock_report_period_ns", "setup_worst_slack_ns",
        "setup_status", "hold_worst_slack_ns", "hold_status",
        "setup_tns_ns", "setup_violating_paths", "hold_tns_ns",
        "hold_violating_paths", "constraint_sections_without_violations",
        "point_pass", "levels_of_logic", "critical_path_length_ns",
        "cell_area_um2", "leaf_cell_count", "combinational_cell_count",
        "sequential_cell_count", "macro_count", "point_identity_sha256",
        "qor_sha256", "setup_sha256", "hold_sha256",
        "mapped_verilog_sha256", "mapped_sdc_sha256", "ddc_sha256")
    for design in TOPS:
        for independent, sealed in zip(points[design],
                                       receipt["grid_points"][design]):
            for key in compare_keys:
                require(independent[key] == sealed[key],
                        "independent receipt mismatch {} {} {}".format(
                            design, independent["period_ns"], key))
    require(fastest["m85"]["period_ns"] == 4.0
            and fastest["m99"]["period_ns"] == 2.75
            and abs(ratio - 1.4545454545454546) < 1e-15
            and abs(area_fraction - 0.4781678889259841) < 1e-15,
            "independent frontier metric mismatch")
    return {
        "all_frozen_source_filelist_tcl_sdc_library_sha_match": True,
        "frozen_hashes": frozen_hashes,
        "all_16_top_clock_wns_tns_and_artifacts_reparsed": True,
        "all_16_receipt_rows_reconciled": True,
        "points": points,
        "fastest_passing": fastest,
        "frozen_grid_target_closure_ratio": ratio,
        "fastest_point_area_fraction": area_fraction,
        "fastest_point_area_reduction_fraction": 1.0 - area_fraction,
    }


def manifest_entries(path, reject_unsafe=True):
    entries = []
    labels = set()
    pattern = re.compile(r"^([0-9a-f]{64})  (.+)$")
    for line_number, line in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), 1):
        match = pattern.match(line)
        require(match is not None,
                "malformed manifest line {}".format(line_number))
        digest, label = match.groups()
        require(label not in labels, "duplicate manifest label " + label)
        if reject_unsafe:
            require("\n" not in label and "\r" not in label,
                    "manifest control character")
            parts = Path(label).parts
            require(not Path(label).is_absolute() and ".." not in parts,
                    "unsafe manifest path " + label)
        labels.add(label)
        entries.append((digest, label))
    return entries


def parse_admission_libraries():
    text = (RUN / "admission.txt").read_text(encoding="utf-8")
    setup = re.search(r"^setup_library=(.+)$", text, re.M)
    hold = re.search(r"^hold_library=(.+)$", text, re.M)
    require(setup and hold, "library paths absent")
    return Path(setup.group(1)), Path(hold.group(1))


def verify_sealed_manifest(receipt, seal):
    entries = manifest_entries(DURABLE)
    require(len(entries) == 374, "durable manifest entry count drift")
    entry_map = {label: digest for digest, label in entries}
    run_files = sorted(path for path in RUN.rglob("*") if path.is_file())
    require(len(run_files) == 370, "run evidence file count drift")
    expected_run_labels = {"run/" + str(path.relative_to(RUN))
                           for path in run_files}
    observed_run_labels = {label for _, label in entries
                           if label.startswith("run/")}
    require(observed_run_labels == expected_run_labels,
            "durable manifest run inventory mismatch")

    setup_library, hold_library = parse_admission_libraries()
    external = {
        "contract/" + CONTRACT.name: CONTRACT,
        "auditor/" + AUDITOR.name: AUDITOR,
        "external/setup_library/" + setup_library.name: setup_library,
        "external/hold_library/" + hold_library.name: hold_library,
    }
    require(set(entry_map) == expected_run_labels | set(external),
            "durable manifest namespace inventory mismatch")
    for label, path in [("run/" + str(path.relative_to(RUN)), path)
                        for path in run_files] + list(external.items()):
        require(entry_map[label] == sha256(path),
                "durable digest mismatch " + label)

    per_point_counts = {}
    for design in TOPS:
        for period in PERIODS:
            name = point_name(design, period)
            prefix = "run/" + name + "/"
            count = sum(label.startswith(prefix) for label in entry_map)
            require(count == 23,
                    "point manifest inventory count drift " + name)
            per_point_counts[name] = count

            point = next(row for row in receipt["grid_points"][design]
                         if abs(float(row["period_ns"]) - period) < 1e-12)
            top = TOPS[design]
            receipt_bindings = {
                prefix + "point_identity.txt": point["point_identity_sha256"],
                prefix + "reports/qor.rpt": point["qor_sha256"],
                prefix + "reports/timing_setup.rpt": point["setup_sha256"],
                prefix + "reports/timing_hold.rpt": point["hold_sha256"],
                prefix + "netlist/" + top + "_mapped.v":
                    point["mapped_verilog_sha256"],
                prefix + "netlist/" + top + "_mapped.sdc":
                    point["mapped_sdc_sha256"],
                prefix + "netlist/" + top + ".ddc": point["ddc_sha256"],
            }
            for label, digest in receipt_bindings.items():
                require(entry_map[label] == digest,
                        "receipt point binding mismatch " + label)

    identity = receipt["identity"]
    require(identity["durable_run_manifest_sha256"] == EXPECTED["durable"]
            and identity["durable_run_manifest_entries"] == 374,
            "receipt durable manifest binding drift")
    require(seal["frozen_identity"]["durable_run_manifest_sha256"]
            == EXPECTED["durable"]
            and seal["frozen_identity"]["durable_run_manifest_entries"] == 374
            and seal["frozen_identity"]["r2_receipt_sha256"]
            == EXPECTED["receipt"], "seal binding drift")
    return {
        "entries": len(entries),
        "run_files": len(run_files),
        "point_files_each": per_point_counts,
        "receipt_bound_files_per_point": 7,
        "all_current_digests_match": True,
        "unique_labels": True,
        "safe_relative_labels": True,
    }


def verify_complete_manifest():
    entries = manifest_entries(COMPLETE)
    require(len(entries) == 5, "complete manifest entry count drift")
    expected = {
        "dc_handoff/scripts/" + AUDITOR.name: AUDITOR,
        "contracts/" + CONTRACT.name: CONTRACT,
        "contracts/" + SEAL.name: SEAL,
        "results/m101r2_pwp_metadata_fmax_sweep_fail_closed_r1_20260824/"
        + RECEIPT.name: RECEIPT,
        "results/m101r2_pwp_metadata_fmax_sweep_fail_closed_r1_20260824/"
        + DURABLE.name: DURABLE,
    }
    require({label for _, label in entries} == set(expected),
            "complete manifest label inventory drift")
    for digest, label in entries:
        require(digest == sha256(expected[label]),
                "complete manifest digest mismatch " + label)
    return {"entries": 5, "all_digests_match": True,
            "binds_receipt_and_durable_manifest": True,
            "binds_original_and_seal_contracts": True}


def normalize_failure(text, temporary_root):
    text = text.replace(str(temporary_root), "<TMP>")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def invoke(run_dir, contract_path, output_dir, cwd):
    output = output_dir / "receipt.json"
    manifest = output_dir / "manifest.sha256"
    command = [
        "python3", str(AUDITOR), "--run-dir", str(run_dir),
        "--contract", str(contract_path), "--output", str(output),
        "--manifest-output", str(manifest),
    ]
    completed = subprocess.run(
        command, cwd=str(cwd), universal_newlines=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    parsed = strict_json(output) if output.is_file() else None
    return {
        "rc": completed.returncode,
        "receipt_created": output.is_file(),
        "manifest_created": manifest.is_file(),
        "status": parsed.get("status") if parsed else None,
        "manifest_entries_in_receipt": (
            parsed.get("identity", {}).get("durable_run_manifest_entries")
            if parsed else None),
        "receipt": parsed,
        "output_path": output,
        "manifest_path": manifest,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def copy_frozen_input_cwd(destination, contract):
    paths = set()
    for frozen in contract["designs"].values():
        paths.add(frozen["filelist"])
        paths.add(frozen["functional_contract"])
        paths.add(frozen["sealed_vcs_completion"])
        paths.update(frozen["rtl_sha256"])
    paths.add(contract["frozen_sweep"]["tcl"])
    paths.add(contract["frozen_sweep"]["sdc"])
    paths.add(str(CONTRACT.relative_to(HW)))
    for relative in sorted(paths):
        source = HW / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(target))


def attack_campaign(sealed_receipt):
    results = {}
    with tempfile.TemporaryDirectory(prefix="m101r2-independent-") as raw:
        temporary_root = Path(raw)
        attack_run = temporary_root / "run"
        shutil.copytree(str(RUN), str(attack_run))

        def call(label, run_dir=attack_run, contract_path=CONTRACT, cwd=HW):
            output_dir = temporary_root / ("out_" + label)
            output_dir.mkdir()
            observed = invoke(run_dir, contract_path, output_dir, cwd)
            observed["failure_last_line"] = normalize_failure(
                observed["stderr"], temporary_root)
            observed.pop("stdout")
            observed.pop("stderr")
            observed.pop("output_path")
            observed.pop("manifest_path")
            return observed, output_dir

        # Nominal replay must be byte-identical for both receipt and manifest.
        nominal, nominal_dir = call("nominal", run_dir=RUN)
        nominal["receipt_byte_identical"] = (
            nominal["receipt_created"]
            and (nominal_dir / "receipt.json").read_bytes()
            == RECEIPT.read_bytes())
        nominal["manifest_byte_identical"] = (
            nominal["manifest_created"]
            and (nominal_dir / "manifest.sha256").read_bytes()
            == DURABLE.read_bytes())
        nominal.pop("receipt")
        require(nominal["rc"] == 0 and nominal["receipt_byte_identical"]
                and nominal["manifest_byte_identical"],
                "nominal replay mismatch")
        results["nominal_replay"] = nominal

        target = attack_run / "m99_2p750ns"
        source_3ns = RUN / "m99_3p000ns"
        pristine_2p75 = RUN / "m99_2p750ns"

        # Original symlink alias attack.
        shutil.rmtree(str(target))
        target.symlink_to(source_3ns)
        observed, _ = call("symlink_alias")
        observed.pop("receipt")
        require(observed["rc"] != 0 and not observed["receipt_created"],
                "symlink alias did not fail closed")
        results["original_symlink_3ns_as_2p75"] = observed
        target.unlink()
        shutil.copytree(str(pristine_2p75), str(target))

        # Upgraded copy attack: not a symlink, but identity remains 3.000 ns.
        shutil.rmtree(str(target))
        shutil.copytree(str(source_3ns), str(target))
        observed, _ = call("copy_alias_identity")
        observed.pop("receipt")
        require(observed["rc"] != 0 and not observed["receipt_created"],
                "copied point identity alias did not fail closed")
        results["copied_3ns_as_2p75_identity_unmodified"] = observed

        # Forge identity only.  The independent clocks report remains 3.00 ns
        # and must still reject the copied point.
        identity_path = target / "point_identity.txt"
        identity_text = identity_path.read_text(encoding="utf-8")
        require("clock_period_ns=3.000" in identity_text,
                "copied identity precondition drift")
        identity_path.write_text(identity_text.replace(
            "clock_period_ns=3.000", "clock_period_ns=2.750"),
            encoding="utf-8")
        observed, _ = call("copy_alias_forged_identity")
        observed.pop("receipt")
        require(observed["rc"] != 0 and not observed["receipt_created"],
                "forged identity with wrong clocks did not fail closed")
        results["copied_3ns_forged_identity_clocks_still_3ns"] = observed

        # Forge the parsed clocks table too.  The mapped SDC and all timing/QoR
        # evidence still come from 3.000 ns; the current auditor accepts it.
        clocks_path = target / "reports/clocks.rpt"
        clocks_text = clocks_path.read_text(encoding="utf-8")
        require("core_clk         3.00" in clocks_text,
                "clock forge precondition drift")
        clocks_path.write_text(clocks_text.replace(
            "core_clk         3.00", "core_clk         2.75"),
            encoding="utf-8")
        observed, _ = call("copy_alias_forged_identity_clock")
        mapped_sdc = (target / "netlist/phase_slack_guarded_wordpacked_pwp_stream_mapped.sdc").read_text(
            encoding="utf-8", errors="replace")
        observed["mapped_sdc_still_contains_3ns"] = (
            "-period 3  -waveform {0 1.5}" in mapped_sdc)
        observed["attack_exposes_fail_open"] = (
            observed["rc"] == 0 and observed["receipt_created"]
            and observed["receipt"]["fastest_passing_grid_points"]
            ["m99"]["period_ns"] == 2.75
            and observed["mapped_sdc_still_contains_3ns"])
        observed.pop("receipt")
        require(observed["attack_exposes_fail_open"],
                "full copied point forge no longer passes")
        results["copied_3ns_forged_identity_and_clocks_mapped_sdc_still_3ns"] = observed
        shutil.rmtree(str(target))
        shutil.copytree(str(pristine_2p75), str(target))

        top = TOPS["m99"]
        netlist_dir = target / "netlist"
        mapped_v = netlist_dir / (top + "_mapped.v")
        mapped_sdc_path = netlist_dir / (top + "_mapped.sdc")
        ddc = netlist_dir / (top + ".ddc")
        svf = netlist_dir / (top + ".svf")

        mapped_v.write_text("", encoding="utf-8")
        observed, _ = call("empty_mapped_v")
        observed.pop("receipt")
        require(observed["rc"] != 0 and not observed["receipt_created"],
                "empty mapped netlist did not fail closed")
        results["empty_mapped_verilog"] = observed
        shutil.copy2(str(pristine_2p75 / "netlist" / mapped_v.name),
                     str(mapped_v))

        mapped_v.unlink()
        observed, _ = call("missing_mapped_v")
        observed.pop("receipt")
        require(observed["rc"] != 0 and not observed["receipt_created"],
                "missing mapped netlist did not fail closed")
        results["missing_mapped_verilog"] = observed
        shutil.copy2(str(pristine_2p75 / "netlist" / mapped_v.name),
                     str(mapped_v))

        # Nonempty but fake artifacts satisfy presence and module-name checks.
        mapped_v.write_text("module {}(); endmodule\n".format(top),
                            encoding="utf-8")
        mapped_sdc_path.write_text("fake_mapped_sdc=true\n", encoding="utf-8")
        ddc.write_text("fake_ddc=true\n", encoding="utf-8")
        svf.write_text("fake_svf=true\n", encoding="utf-8")
        observed, _ = call("fake_mapped_artifacts")
        point = observed["receipt"]["grid_points"]["m99"][0]
        observed["attack_exposes_fail_open"] = (
            observed["rc"] == 0 and point["mapped_artifacts_present"] is True
            and point["mapped_verilog_sha256"] == sha256(mapped_v)
            and mapped_v.stat().st_size < 100 and ddc.stat().st_size < 100)
        observed["fake_mapped_verilog_bytes"] = mapped_v.stat().st_size
        observed["fake_ddc_bytes"] = ddc.stat().st_size
        observed.pop("receipt")
        require(observed["attack_exposes_fail_open"],
                "fake mapped artifact attack no longer passes")
        results["nonempty_fake_mapped_v_sdc_ddc_svf"] = observed
        for name in (mapped_v.name, mapped_sdc_path.name, ddc.name, svf.name):
            shutil.copy2(str(pristine_2p75 / "netlist" / name),
                         str(netlist_dir / name))

        # Frozen contract threshold replacement must fail before evidence use.
        hostile_contract = strict_json(CONTRACT)
        hostile_contract["acceptance_gates"][
            "m99_to_m85_achieved_grid_frequency_ratio_min"] = 99.0
        hostile_contract_path = temporary_root / "hostile_contract.json"
        hostile_contract_path.write_text(
            json.dumps(hostile_contract, indent=2) + "\n", encoding="utf-8")
        observed, _ = call("contract_threshold", contract_path=hostile_contract_path)
        observed.pop("receipt")
        require(observed["rc"] != 0 and not observed["receipt_created"],
                "contract threshold attack did not fail closed")
        results["replaced_contract_threshold_99x"] = observed

        # Replace a library through admission without touching the actual DB.
        admission = attack_run / "admission.txt"
        admission_original = admission.read_text(encoding="utf-8")
        fake_library = temporary_root / "fake_setup.db"
        fake_library.write_text("not a Liberty DB\n", encoding="utf-8")
        admission.write_text(re.sub(
            r"^setup_library=.+$", "setup_library=" + str(fake_library),
            admission_original, flags=re.M), encoding="utf-8")
        observed, _ = call("replaced_library")
        observed.pop("receipt")
        require(observed["rc"] != 0 and not observed["receipt_created"],
                "library replacement did not fail closed")
        results["replaced_setup_library"] = observed
        admission.write_text(admission_original, encoding="utf-8")

        # Clone only frozen input namespace so RTL/filelist mutation is safe.
        frozen_cwd = temporary_root / "frozen_cwd"
        frozen_cwd.mkdir()
        frozen_contract = strict_json(CONTRACT)
        copy_frozen_input_cwd(frozen_cwd, frozen_contract)
        frozen_contract_path = frozen_cwd / str(CONTRACT.relative_to(HW))
        cloned_filelist = frozen_cwd / frozen_contract["designs"][
            "m99_phase_slack"]["filelist"]
        cloned_filelist.write_text(
            cloned_filelist.read_text(encoding="utf-8") + "// hostile\n",
            encoding="utf-8")
        observed, _ = call("replaced_filelist", run_dir=RUN,
                           contract_path=frozen_contract_path, cwd=frozen_cwd)
        observed.pop("receipt")
        require(observed["rc"] != 0 and not observed["receipt_created"],
                "filelist replacement did not fail closed")
        results["replaced_filelist"] = observed
        shutil.copy2(str(HW / frozen_contract["designs"]
                         ["m99_phase_slack"]["filelist"]),
                     str(cloned_filelist))

        rtl_relative = next(iter(frozen_contract["designs"]
                                 ["m99_phase_slack"]["rtl_sha256"]))
        cloned_rtl = frozen_cwd / rtl_relative
        cloned_rtl.write_text(
            cloned_rtl.read_text(encoding="utf-8") + "// hostile\n",
            encoding="utf-8")
        observed, _ = call("replaced_rtl", run_dir=RUN,
                           contract_path=frozen_contract_path, cwd=frozen_cwd)
        observed.pop("receipt")
        require(observed["rc"] != 0 and not observed["receipt_created"],
                "RTL replacement did not fail closed")
        results["replaced_rtl"] = observed

        # Remove a generated report not in REQUIRED_REPORTS.  The dynamic
        # manifest silently shrinks and the seal still passes.
        optional = target / "reports/check_design_precompile.rpt"
        optional.unlink()
        observed, output_dir = call("manifest_optional_omission")
        observed["omitted_label_absent"] = not any(
            label.endswith("m99_2p750ns/reports/check_design_precompile.rpt")
            for _, label in manifest_entries(output_dir / "manifest.sha256"))
        observed["attack_exposes_fail_open"] = (
            observed["rc"] == 0 and observed["manifest_entries_in_receipt"] == 373
            and observed["omitted_label_absent"])
        observed.pop("receipt")
        require(observed["attack_exposes_fail_open"],
                "optional manifest omission no longer passes")
        results["manifest_inventory_omission"] = observed
        shutil.copy2(str(pristine_2p75 / "reports" / optional.name),
                     str(optional))

        # A newline in a filename injects a second valid manifest record that
        # collides with the genuine run/admission.txt label.
        injection_dir = attack_run / ("evil\n" + "0" * 64 + "  run")
        injection_dir.mkdir()
        injection = injection_dir / "admission.txt"
        injection.write_text("x\n", encoding="utf-8")
        observed, output_dir = call("manifest_path_injection")
        physical_lines = (output_dir / "manifest.sha256").read_text(
            encoding="utf-8").splitlines()
        labels = []
        for line in physical_lines:
            match = re.match(r"^[0-9a-f]{64}  (.+)$", line)
            if match:
                labels.append(match.group(1))
        observed["physical_manifest_lines"] = len(physical_lines)
        observed["duplicate_admission_labels"] = labels.count(
            "run/admission.txt")
        observed["attack_exposes_fail_open"] = (
            observed["rc"] == 0 and observed["manifest_entries_in_receipt"] == 375
            and len(physical_lines) == 376
            and observed["duplicate_admission_labels"] == 2)
        observed.pop("receipt")
        require(observed["attack_exposes_fail_open"],
                "manifest path injection no longer passes")
        results["manifest_newline_path_collision"] = observed
        shutil.rmtree(str(injection_dir))

        # Contradict timing MET/+0.0009 with QoR TNS=-0.01.  The auditor
        # silently marks only this point failed, then seals the grid at 3.0 ns.
        qor = target / "reports/qor.rpt"
        qor_original = qor.read_text(encoding="utf-8")
        require("Total Negative Slack:          0.00" in qor_original,
                "QoR conflict precondition drift")
        qor.write_text(qor_original.replace(
            "Total Negative Slack:          0.00",
            "Total Negative Slack:         -0.01", 1), encoding="utf-8")
        observed, _ = call("qor_timing_conflict")
        attacked_point = observed["receipt"]["grid_points"]["m99"][0]
        observed["attacked_setup_status"] = attacked_point["setup_status"]
        observed["attacked_setup_slack_ns"] = attacked_point[
            "setup_worst_slack_ns"]
        observed["attacked_qor_tns_ns"] = attacked_point["setup_tns_ns"]
        observed["attacked_point_pass"] = attacked_point["point_pass"]
        observed["new_fastest_m99_period_ns"] = observed["receipt"][
            "fastest_passing_grid_points"]["m99"]["period_ns"]
        observed["attack_exposes_fail_open"] = (
            observed["rc"] == 0 and observed["status"].startswith("PASS_")
            and observed["attacked_setup_status"] == "MET"
            and observed["attacked_setup_slack_ns"] > 0.0
            and observed["attacked_qor_tns_ns"] < 0.0
            and observed["new_fastest_m99_period_ns"] == 3.0)
        observed.pop("receipt")
        require(observed["attack_exposes_fail_open"],
                "QoR/timing contradiction no longer seals")
        results["qor_tns_timing_slack_contradiction"] = observed
        qor.write_text(qor_original, encoding="utf-8")

        # Boundary semantics: exact zero with MET is a pass; exact zero with
        # VIOLATED must not be a passing point even if the grid can still seal.
        baseline_zero = sealed_receipt["grid_points"]["m85"][5]
        require(baseline_zero["period_ns"] == 4.0
                and baseline_zero["setup_worst_slack_ns"] == 0.0
                and baseline_zero["setup_status"] == "MET"
                and baseline_zero["point_pass"] is True,
                "zero-slack baseline drift")
        timing = attack_run / "m85_4p000ns/reports/timing_setup.rpt"
        timing_original = timing.read_text(encoding="utf-8")
        require("slack (MET)                                                      0.0000" in timing_original,
                "zero status attack precondition drift")
        timing.write_text(timing_original.replace(
            "slack (MET)                                                      0.0000",
            "slack (VIOLATED)                                                 0.0000"),
            encoding="utf-8")
        observed, _ = call("zero_slack_violated_status")
        attacked = observed["receipt"]["grid_points"]["m85"][5]
        observed["baseline_met_zero_pass"] = True
        observed["violated_zero_point_pass"] = attacked["point_pass"]
        observed["violated_zero_status"] = attacked["setup_status"]
        observed["grid_can_still_seal_on_later_point"] = observed["rc"] == 0
        observed["boundary_correct"] = (
            attacked["setup_worst_slack_ns"] == 0.0
            and attacked["setup_status"] == "VIOLATED"
            and attacked["point_pass"] is False)
        observed.pop("receipt")
        require(observed["boundary_correct"],
                "zero slack status boundary drift")
        results["zero_slack_status_boundary"] = observed

    return results


def main():
    self_start = sha256(Path(__file__).resolve())
    paths = {
        "auditor": AUDITOR, "contract": CONTRACT, "seal": SEAL,
        "receipt": RECEIPT, "durable": DURABLE, "complete": COMPLETE,
        "old_review_manifest": OLD_REVIEW_MANIFEST, "docs359": DOC359,
    }
    observed_hashes = {label: sha256(path) for label, path in paths.items()}
    require(observed_hashes == EXPECTED,
            "frozen identity mismatch " + repr(observed_hashes))

    contract = strict_json(CONTRACT)
    seal = strict_json(SEAL)
    receipt = strict_json(RECEIPT)
    require(seal["frozen_identity"]["original_contract_sha256"]
            == EXPECTED["contract"]
            and seal["frozen_identity"]["r2_auditor_sha256"]
            == EXPECTED["auditor"], "seal input identity drift")
    require(seal["motivation"]["m101_r1_independent_review_manifest_file_sha256"]
            == EXPECTED["old_review_manifest"], "r1 review binding drift")
    require(receipt["status"]
            == "PASS_FAIL_CLOSED_FROZEN_GRID_TARGET_CLOSURE",
            "receipt status drift")

    independent_grid = independent_reparse(contract, receipt)
    durable_audit = verify_sealed_manifest(receipt, seal)
    complete_audit = verify_complete_manifest()
    attacks = attack_campaign(receipt)

    closed = [
        "original_symlink_3ns_as_2p75",
        "copied_3ns_as_2p75_identity_unmodified",
        "copied_3ns_forged_identity_clocks_still_3ns",
        "empty_mapped_verilog", "missing_mapped_verilog",
        "replaced_contract_threshold_99x",
        "replaced_setup_library",
        "replaced_filelist",
        "replaced_rtl",
    ]
    require(all(attacks[name]["rc"] != 0
                and not attacks[name]["receipt_created"] for name in closed),
            "one expected fail-closed attack passed")
    open_attacks = [
        "copied_3ns_forged_identity_and_clocks_mapped_sdc_still_3ns",
        "nonempty_fake_mapped_v_sdc_ddc_svf",
        "manifest_inventory_omission",
        "manifest_newline_path_collision",
        "qor_tns_timing_slack_contradiction",
    ]
    require(all(attacks[name]["attack_exposes_fail_open"]
                for name in open_attacks),
            "one expected open attack stopped reproducing")

    output = {
        "schema": "m101r2_fail_closed_independent_hammer_audit_v1",
        "status": "CONDITIONAL_PASS_R1_ATTACKS_CLOSED_BUT_R2_STILL_HAS_FIVE_FAIL_OPEN_CLASSES",
        "identity": observed_hashes,
        "sealed_manifest_audit": durable_audit,
        "complete_manifest_audit": complete_audit,
        "independent_16_point_reparse": independent_grid,
        "receipt_binding": {
            "contract_sha256": receipt["identity"]["contract_sha256"],
            "auditor_sha256": receipt["identity"]["auditor_sha256"],
            "durable_manifest_sha256": receipt["identity"]
            ["durable_run_manifest_sha256"],
            "durable_manifest_entries": receipt["identity"]
            ["durable_run_manifest_entries"],
            "all_16_points_seven_exposed_hashes_match_manifest": True,
            "final_complete_manifest_binds_receipt": True,
        },
        "nominal_metrics": {
            "m85_fastest_passing_grid_period_ns": receipt[
                "fastest_passing_grid_points"]["m85"]["period_ns"],
            "m99_fastest_passing_grid_period_ns": receipt[
                "fastest_passing_grid_points"]["m99"]["period_ns"],
            "frozen_grid_target_closure_ratio": receipt["comparison"]
            ["frozen_grid_target_closure_ratio"],
            "fastest_point_area_fraction": receipt["comparison"]
            ["fastest_point_standard_cell_area_fraction"],
            "m85_4ns_zero_slack_status": receipt["grid_points"]["m85"][5]
            ["setup_status"],
            "m85_4ns_zero_slack_pass": receipt["grid_points"]["m85"][5]
            ["point_pass"],
        },
        "hostile_attacks": attacks,
        "closed_attack_classes": closed,
        "remaining_fail_open_attack_classes": open_attacks,
        "claim_boundary": receipt["claim_boundary"],
        "self_sha256_at_start": self_start,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    require(sha256(Path(__file__).resolve()) == self_start,
            "audit changed during execution")
    print("PASS M101-r2 independent hostile audit closed={} open={}".format(
        len(closed), len(open_attacks)))


if __name__ == "__main__":
    main()
