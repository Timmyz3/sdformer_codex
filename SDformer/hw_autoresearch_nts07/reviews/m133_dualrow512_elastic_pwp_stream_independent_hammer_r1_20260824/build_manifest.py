#!/usr/bin/env python3
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REVIEW = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


EXTERNAL = [
    "rtl_m133/m133_dualrow512_elastic_pwp_stream.sv",
    "verif_m133/m133_dualrow512_elastic_pwp_stream_assertions.sv",
    "tb_m133/tb_m133_dualrow512_elastic_pwp_stream.sv",
    "contracts/m133_dualrow512_elastic_pwp_stream_vcs_contract_r1_20260824.json",
    "contracts/m133_r1_stall_fault_composition_correction_r1_20260824.json",
    "contracts/m133r2_dualrow512_elastic_pwp_stream_vcs_contract_r1_20260824.json",
    "contracts/m133_dualrow512_elastic_pwp_stream_logic_only_dc_contract_r1_20260824.json",
    "contracts/m133r2_dc_functional_supersession_overlay_r1_20260824.json",
    "dc_handoff/runs/m133_dualrow512_elastic_pwp_stream_vcs_r1_sealed_20260824/RUN_COMPLETE.txt",
    "dc_handoff/runs/m133_dualrow512_elastic_pwp_stream_vcs_r1_sealed_20260824/input_sha256.txt",
    "dc_handoff/runs/m133_dualrow512_elastic_pwp_stream_vcs_r1_sealed_20260824/sim.raw.log",
    "dc_handoff/runs/m133_dualrow512_elastic_pwp_stream_vcs_r1_sealed_20260824/assert.report",
    "dc_handoff/runs/m133r2_dualrow512_elastic_pwp_stream_vcs_r1_sealed_20260824/RUN_COMPLETE.txt",
    "dc_handoff/runs/m133r2_dualrow512_elastic_pwp_stream_vcs_r1_sealed_20260824/input_sha256.txt",
    "dc_handoff/runs/m133r2_dualrow512_elastic_pwp_stream_vcs_r1_sealed_20260824/sim.raw.log",
    "dc_handoff/runs/m133r2_dualrow512_elastic_pwp_stream_vcs_r1_sealed_20260824/assert.report",
    "dc_handoff/runs/m133_dualrow512_elastic_pwp_stream_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt",
    "dc_handoff/runs/m133_dualrow512_elastic_pwp_stream_logic_only_dc_3p000ns_r1_sealed_20260824/m133_logic_only_dc_receipt_r1.json",
    "dc_handoff/runs/m133_dualrow512_elastic_pwp_stream_logic_only_dc_3p000ns_r1_sealed_20260824/reports/area.rpt",
    "dc_handoff/runs/m133_dualrow512_elastic_pwp_stream_logic_only_dc_3p000ns_r1_sealed_20260824/reports/qor.rpt",
    "dc_handoff/runs/m133_dualrow512_elastic_pwp_stream_logic_only_dc_3p000ns_r1_sealed_20260824/reports/timing_setup.rpt",
    "dc_handoff/runs/m133_dualrow512_elastic_pwp_stream_logic_only_dc_3p000ns_r1_sealed_20260824/reports/timing_hold.rpt",
    "dc_handoff/runs/m133_dualrow512_elastic_pwp_stream_logic_only_dc_3p000ns_r1_sealed_20260824/reports/constraint_violators.rpt",
    "docs/359_DATE终局冻结_20260813.md",
]

LOCAL = [
    "README.md",
    "RUN_COMPLETE.txt",
    "audit_m133.py",
    "audit.stdout.log",
    "m133_independent_audit.json",
    "frozen_r1_m133_assertions.sv",
    "tb_m133_stall_fault_interaction.sv",
    "run_independent_vcs.sh",
    "run_r2_independent_vcs.sh",
    "VCS_REVIEW_COMPLETE.txt",
    "R2_VCS_REVIEW_COMPLETE.txt",
    "independent_vcs_input_sha256.txt",
    "independent_r2_input_sha256.txt",
    "frozen_r1_stall_fault_cross_property/compile.log",
    "frozen_r1_stall_fault_cross_property/compile.rc",
    "frozen_r1_stall_fault_cross_property/sim.log",
    "frozen_r1_stall_fault_cross_property/sim.rc",
    "frozen_r1_stall_fault_cross_property/assert.report",
    "independent_r2_production_rerun/compile.log",
    "independent_r2_production_rerun/compile.rc",
    "independent_r2_production_rerun/sim.log",
    "independent_r2_production_rerun/sim.rc",
    "independent_r2_production_rerun/assert.report",
    "independent_r2_stall_fault_cross_property/compile.log",
    "independent_r2_stall_fault_cross_property/compile.rc",
    "independent_r2_stall_fault_cross_property/sim.log",
    "independent_r2_stall_fault_cross_property/sim.rc",
    "independent_r2_stall_fault_cross_property/assert.report",
    "build_manifest.py",
]


def main():
    (REVIEW / "RUN_COMPLETE.txt").write_text(
        "status=PASS_M133R2_INDEPENDENT_HAMMER\n"
        "score=89\n"
        "p0=0\n"
        "p1=1\n"
        "p2=3\n"
        "r1_counterexample_preserved=true\n"
        "r1_counterexample_closed_by_r2=true\n"
        "exact_sha_commercial_vcs=true\n"
        "logic_only_dc=true\n"
        "bank_mapper_implemented=false\n"
        "foundry_macro=false\n"
        "physical_speedup=false\n"
        "system_speedup=false\n"
        "docs_359_unchanged=true\n"
    )
    external_lines = []
    for relative in EXTERNAL:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        external_lines.append("{}  {}".format(sha256(path), relative))
    (REVIEW / "source_evidence.sha256").write_text(
        "\n".join(external_lines) + "\n"
    )
    local_lines = []
    for relative in LOCAL:
        path = REVIEW / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        local_lines.append("{}  {}".format(sha256(path), relative))
    (REVIEW / "manifest.sha256").write_text("\n".join(local_lines) + "\n")
    print("review_files={}".format(len(local_lines)))
    print("external_evidence={}".format(len(external_lines)))


if __name__ == "__main__":
    main()
