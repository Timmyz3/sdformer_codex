#!/usr/bin/env python3
"""Independent no-EDA mechanical hammer for frozen M2137 source."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
REVIEW = HERE.parent
HW = REVIEW.parents[1]
REPO = HW.parent
RUNNER = HW / "dc_handoff/scripts/run_m2137_m2018_tsbg_rtl_saif_window_diagnostic_one_shot.py"
TEST = HW / "tests/test_m2137_tsbg_rtl_saif_option_aware_timing_surface.py"
CONTRACT = HW / "contracts/m2137_m2018_tsbg_rtl_saif_window_diagnostic_source_contract_r1_20260904.json"
SELFCHECK = HW / "reviews/m2137_m2018_tsbg_rtl_saif_window_diagnostic_source_selfcheck_r1_20260904"
M2126 = HW / "reviews/m2126_m2125_m2018_tsbg_rtl_saif_window_diagnostic_source_hammer_r1_20260904"
M2128 = HW / "reviews/m2128_m2127_m2125_tsbg_rtl_saif_window_diagnostic_failure_hammer_r1_20260904"
M2127_ATTEMPT = HW / "results/.m2127_m2125_tsbg_rtl_saif_window_diagnostic_attempt_consumed"
M2127_RESULT = HW / "results/m2127_m2125_m2018_tsbg_rtl_saif_window_diagnostic_r1_20260904"
M2127_LOCK = HW / "results/.m2127_m2125_tsbg_rtl_saif_window_diagnostic_launch_lock"
M2139_RESULT = HW / "results/m2139_m2137_m2018_tsbg_rtl_saif_window_diagnostic_r1_20260904"
M2139_ATTEMPT = HW / "results/.m2139_m2137_tsbg_rtl_saif_window_diagnostic_attempt_consumed"
M2139_LOCK = HW / "results/.m2139_m2137_tsbg_rtl_saif_window_diagnostic_launch_lock"

RUNNER_SHA = "a1a72dcdfbbf0f1f0cbae52424b1dac08b023edd612223236f9c2fb77e7445d4"
CONTRACT_SHA = "42d2394942f25e80a28b6b448ad966715366dc3d71ea60e5cf1899b07b89b2cd"
TEST_SHA = "1b1ccd14aca4c4560766b42615839e7e8674d09f0ae0631256e32b44e34ed744"
M2126_MANIFEST = "db8f8bd83ddc6a483baff88bd1460e8b829b51757ec524421399a45d84235bdc"
M2126_OUTER = "d3313574bf92184c6029d078dfa8010e733c0936519f76e790add24e8f6a87f7"
M2128_MANIFEST = "5ecbb1bd4fc6bf1d3851566c259b837aeaa3c94d3a1bce2631a735c28b20ae4c"
M2128_OUTER = "9a2ad99b8dfaaccb121ec391fa0c2540d7aa8c88e6ce8b6384776474edaf524e"
M2128_REVIEW = "e43f1e38b8c11b522a9d35041260d8398dfbe07b5aa3db1e312a26952ee63928"
DOC359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
BUDGET = {
    "license_queries": 1, "vcs_compiles": 1, "simv_runs": 2,
    "saif_files": 2, "dc_runs": 0, "ptpx_runs": 0,
    "automatic_retry": False, "p1_serial": True,
    "reuse_old_artifacts": False,
}


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_seal(root: Path, manifest_sha: str | None = None,
                outer_sha: str | None = None) -> int:
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(manifest.is_file() and outer.is_file(), f"seal missing: {root}")
    if manifest_sha:
        need(sha(manifest) == manifest_sha, f"manifest SHA: {root}")
    if outer_sha:
        need(sha(outer) == outer_sha, f"outer-file SHA: {root}")
    tokens = outer.read_text().split()
    need(tokens == [sha(manifest), "SHA256SUMS"], f"outer seal: {root}")
    listed: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split("  ", 1)
        need("/" not in name and name not in listed, f"manifest member: {root}")
        listed[name] = digest
    actual = sorted(p.name for p in root.iterdir()
                    if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    need(sorted(listed) == actual, f"non-exhaustive seal: {root}")
    for name, digest in listed.items():
        need(sha(root / name) == digest, f"member SHA: {root / name}")
    return len(actual)


def load_runner():
    spec = importlib.util.spec_from_file_location("m2138_exact_m2137", RUNNER)
    need(spec is not None and spec.loader is not None, "runner import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    need(sha(RUNNER) == RUNNER_SHA, "runner identity")
    need(sha(CONTRACT) == CONTRACT_SHA, "contract identity")
    need(sha(TEST) == TEST_SHA, "test identity")
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [CONTRACT_SHA, CONTRACT.name], "contract sidecar")
    need(outer.read_text().split() == [sha(sidecar), sidecar.name], "contract outer")
    selfcheck_members = verify_seal(SELFCHECK)
    m2126_members = verify_seal(M2126, M2126_MANIFEST, M2126_OUTER)
    m2128_members = verify_seal(M2128, M2128_MANIFEST, M2128_OUTER)
    need(sha(M2128 / "review.json") == M2128_REVIEW, "M2128 review SHA")

    contract = load_json(CONTRACT)
    need(contract["execution_budget"] == BUDGET, "budget")
    need(contract["single_source_delta"]["data_plane_change"] is False, "data plane delta")
    need(contract["single_source_delta"]["rtl_tb_parser_ucli_filelist_change"] is False,
         "inherited inputs")
    inventory = contract["source_inventory"]
    need(len(inventory) == 12, "inventory count")
    for rel, digest in inventory.items():
        path = REPO / rel
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             f"inventory SHA: {rel}")
    need(sha(HW / "docs/359_DATE终局冻结_20260813.md") == DOC359_SHA,
         "docs359 identity")

    m2128 = load_json(M2128 / "review.json")
    disposition = m2128["m2127_disposition"]
    need(disposition["attempt_consumed"] is True, "M2127 consumed")
    need(disposition["automatic_retry"] is False
         and disposition["retry_authorized"] is False, "M2127 no retry")
    need(disposition["paper_citable"] is False, "M2127 noncitable")
    need(M2127_ATTEMPT.is_dir() and not M2127_RESULT.exists() and not M2127_LOCK.exists(),
         "M2127 live disposition")
    need(not M2139_RESULT.exists() and not M2139_ATTEMPT.exists() and not M2139_LOCK.exists(),
         "M2139 freshness")

    runner = load_runner()
    with tempfile.TemporaryDirectory(prefix="m2138_guard_") as tmp:
        root = Path(tmp)
        sdformer = root / "SDformer"
        sdformer.mkdir()
        clean = sdformer / "clean.sv"
        clean.write_text("module clean; endmodule\n", encoding="utf-8")
        filelist = sdformer / "sources.f"
        filelist.write_text(f"{clean}\n", encoding="utf-8")
        positive = ["vcs", f"-Mdir={sdformer / 'csrc'}", "-f", str(filelist),
                    "-o", str(sdformer / "simv")]
        observed = runner.validate_timing_surface(positive, [filelist, clean])
        need(observed["active_input_count"] == 2, "positive path")
        negative_counts = {"sdf_option": 0, "unit_delay_define": 0,
                           "active_source_or_filelist": 0}
        for token in ("-sdfmax", "-SDF=min:tb.dut:file.sdf", "+sdfverbose"):
            try:
                runner.validate_timing_surface(["vcs", token], [clean])
            except runner.Failure:
                negative_counts["sdf_option"] += 1
            else:
                raise RuntimeError(f"SDF mutation admitted: {token}")
        for token in ("+define+UNIT_DELAY", "+define+FOO+UNIT_DELAY=1",
                      "+DeFiNe+unit_delay=1+BAR"):
            try:
                runner.validate_timing_surface(["vcs", token], [clean])
            except runner.Failure:
                negative_counts["unit_delay_define"] += 1
            else:
                raise RuntimeError(f"UNIT_DELAY mutation admitted: {token}")
        contaminated = {
            "source_sdf.sv": 'initial $sdf_annotate("gate.sdf", dut);\n',
            "source_unit.sv": "`ifdef UNIT_DELAY\n`endif\n",
            "filelist_sdf.f": '$sdf_annotate("gate.sdf", dut)\n',
            "filelist_unit.f": "UNIT_DELAY\n",
        }
        for name, content in contaminated.items():
            path = root / name
            path.write_text(content, encoding="utf-8")
            try:
                runner.validate_timing_surface(["vcs"], [path])
            except runner.Failure:
                negative_counts["active_source_or_filelist"] += 1
            else:
                raise RuntimeError(f"active-input mutation admitted: {name}")
        need(negative_counts == {"sdf_option": 3, "unit_delay_define": 3,
                                 "active_source_or_filelist": 4}, "mutation totals")

    sources = []
    for line in runner.M2125.FILELIST.read_text().splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            sources.append((REPO / item).resolve())
    need(len(sources) == 6, "actual active source count")
    actual_cmd = ["vcs", f"-Mdir={REPO / 'SDformer/work/csrc'}", "-f",
                  str(REPO / "SDformer/work/sources.absolute.f"), "-o",
                  str(REPO / "SDformer/work/simv")]
    actual_surface = runner.validate_timing_surface(
        actual_cmd, [runner.M2125.FILELIST, *sources])
    need(actual_surface["active_input_count"] == 7, "actual timing surface")

    text = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(text)
    production = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "production")
    first = [ast.unparse(node) for node in production.body[:7]]
    need(first[0] == "source_validation(require_review=True)", "review first")
    need(first[1].startswith("need(not RESULT.exists()")
         and "not ATTEMPT.exists()" in first[1]
         and "not LOCK.exists()" in first[1], "freshness second")
    need(first[2] == "M2125.no_same_uid_eda()", "collision third")
    need(first[3] == "LOCK.mkdir()" and first[4] == "ATTEMPT.mkdir()",
         "lock/attempt order")
    line_review = text.index("source_validation(require_review=True)")
    line_attempt = text.index("ATTEMPT.mkdir()")
    line_license = text.index('counts["license_queries"] += 1')
    line_guard = text.index("commands[\"timing_surface\"] = validate_timing_surface")
    line_compile_count = text.index('counts["vcs_compiles"] += 1')
    need(line_review < line_attempt < line_license < line_guard < line_compile_count,
         "review/attempt/license/guard/compile order")
    axis_loops = [node for node in ast.walk(production) if isinstance(node, ast.For)
                  and ast.unparse(node.iter) == "M2125.AXES.items()"]
    need(len(axis_loops) == 1, "single serial axis loop")
    need('"+vcs+initreg+random"' in text and '"+vcs+initreg+0"' in text
         and '"+WORKLOAD_SLOT=42"' in text, "fixed runtime surface")
    need("M2125.PARSER" in text and "M2125.UCLI[axis]" in text,
         "inherited parser/UCLI")
    need("automatic_retry=false" in text and "independent_result_hammer_required" in text,
         "failure/result boundary")

    parser_text = runner.M2125.PARSER.read_text(encoding="utf-8")
    tb_text = (HW / "tb_m2018/tb_m2125_m2018_tsbg_rtl_saif_window_diagnostic.sv").read_text(
        encoding="utf-8")
    inherited = {
        "slot42": "FROZEN_WORKLOAD_SLOT = 42" in tb_text,
        "settled_negedge": "settled_negedge" in tb_text,
        "completion_ledgers": "completion ledger drift" in tb_text,
        "records_93971": "EXPECTED_RECORDS = 93971" in parser_text,
        "all_tx_zero": "tx_nonzero == 0 and tx_sum == 0.0" in parser_text,
        "conservation": "conservation_failures == 0" in parser_text,
        "critical": "missing/zero critical activity" in parser_text,
        "diagnostic_nonpaper": '"paper_citable": False' in parser_text,
    }
    need(all(inherited.values()), f"inherited gates: {inherited}")

    output = {
        "schema": "m2138_m2137_independent_mechanical_checks_r1_v1",
        "status": "PASS_M2138_INDEPENDENT_MECHANICAL_CHECKS__NO_EDA",
        "execution_performed": {"license_queries": 0, "vcs_compiles": 0,
                                "simv_runs": 0, "saif_files": 0,
                                "dc_runs": 0, "ptpx_runs": 0},
        "identity": {"runner_sha256": sha(RUNNER),
                     "contract_sha256": sha(CONTRACT),
                     "test_sha256": sha(TEST),
                     "docs359_sha256": DOC359_SHA},
        "seal_member_counts": {"m2137_selfcheck": selfcheck_members,
                               "m2126": m2126_members, "m2128": m2128_members},
        "source_inventory_count": len(inventory),
        "positive_sdformer_path": True,
        "negative_mutation_counts": negative_counts,
        "actual_active_inputs_clean": actual_surface,
        "review_before_attempt_before_license": True,
        "timing_guard_before_compile": True,
        "single_serial_axis_loop": True,
        "m2139_fresh": True,
        "m2127_consumed_no_retry_noncitable": True,
        "inherited_gates": inherited,
        "future_budget": BUDGET,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
