#!/opt/anaconda3/bin/python3.12
"""Independent, source-only M2214 hammer.  This script never invokes EDA."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
CONTRACT = HW / "contracts/m2213_preread_postread_causal_ablation_source_contract_r1_20260904.json"
M803 = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
M2018 = HW / "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"
RTL = HW / "rtl_m2213/m2213_c2_tsbg_b4_postread_causal_frontend.sv"
SVA = HW / "verif_m2213/m2213_c2_tsbg_postread_causal_assertions.sv"
TB = HW / "tb_m2213/tb_m2213_c2_tsbg_preread_postread_causal_directed.sv"
FILELIST = HW / "dc_handoff/filelists/tcasii_m2213_preread_postread_causal_directed_vcs.f"
PARSER = HW / "system_simulator/scripts/parse_m2215_m2213_preread_postread_causal_directed_vcs.py"
RUNNER = HW / "dc_handoff/scripts/run_m2215_m2214_m2213_preread_postread_causal_directed_vcs_one_shot.sh"
TEST = HW / "tests/test_m2213_preread_postread_causal_ablation_source.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
AUTHOR = HW / "reviews/m2213_preread_postread_causal_ablation_source_author_receipt_r1_20260904"
RESULT = HW / "results/m2215_m2213_preread_postread_causal_directed_vcs_r1_20260904"
ATTEMPT = HW / "results/.m2215_m2213_preread_postread_causal_vcs_attempt_consumed"
LOCK = HW / "results/.m2215_m2213_preread_postread_causal_vcs_launch_lock"
PYTHON = Path("/opt/anaconda3/bin/python3.12")
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")


EXPECTED = {
    M803: "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    M2018: "96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21",
    RTL: "face67f666aebb080afaff8852ee8d32f6bd389f28ff8b0ebaf7b0e2c0e1f09f",
    SVA: "f243c6f96512700da114af8140157cc6e64a4a2469997449cc24aadf8f3c212c",
    TB: "a44efdda41f66a6c6a1911e339548c3011639eac56466b3e37389a0349d6e7ab",
    FILELIST: "93695e3744224a17e0fee4ee1c632fe0515dff32a339e5a2ab404308f68810aa",
    PARSER: "10c1b13b167851ceec36c5ece653fe548c772522bdf79d22c355a169af7b70c8",
    RUNNER: "d1c96a97561ac2d3822dc517eaadebac28b7e4845dd6af9fa5e1d0414f26e1dd",
    TEST: "861506999ed8029b450c216d0bc7042f172e9c6fad897d42f401526ad947c37e",
    CONTRACT: "1c9b9d4fb8e7cbcfaf2e11ffc864cb326dcc3e220cc8d64a5566ededfce296cd",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    PYTHON: "873a1168d6d2a7d1b406b85c2a1ea986a6f086041069ab1ee3f70b9217f10161",
    VCS: "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    LMUTIL: "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def block(text: str, start: str, end: str) -> str:
    a = text.index(start)
    b = text.index(end, a + len(start))
    return text[a:b]


def audit_semantics(parts: dict[str, str]) -> None:
    rtl = parts["rtl"]
    sva = parts["sva"]
    tb = parts["tb"]
    parser = parts["parser"]
    runner = parts["runner"]

    need("parameter int SCHEDULE_MODE" not in rtl, "post-read must be fixed group-major")
    need("localparam int ORDER_INDEX = map_group * BUNDLE + map_ctx" in rtl,
         "post-read group-major mapping")
    need("assign core_req_valid = state_q == ST_FETCH_REQ && !fault_q;" in rtl,
         "real request channel")
    need("assign core_req_bank_valid = 8'hff;" in rtl, "all eight banks requested")
    need("assign core_rsp_ready = state_q == ST_FETCH_RSP && !fault_q;" in rtl,
         "real response channel")
    hit = block(rtl, "if (find_cache_hit) begin", "end else begin")
    need("fill_cache_q <= find_cache_index;" in hit, "hit preserves resident row")
    need("postread_hit_q <= 1;" in hit, "hit marked post-read")
    need("state_q <= ST_FETCH_REQ;" in hit and "state_q <= ST_BRIDGE;" not in hit,
         "hit must fetch before bridge")
    response = block(
        rtl,
        "if (state_q == ST_FETCH_RSP && core_rsp_accept\n                    && core_rsp_identity_legal) begin",
        "if (state_q == ST_BRIDGE && bridge_accept) begin",
    )
    need(re.search(
        r"if \(!postread_hit_q\) begin\s+for \(int bank.*?cache_weight_q"
        r"\[fill_cache_q\]\[fill_half_q\]\[fill_slice_q\]",
        response, re.DOTALL) is not None,
        "cache payload write must remain inside miss-only guard")
    need("postread_bundle_response_count_q" in response and
         "postread_identity_accept_count_q" in response,
         "post-read legal-response counters")
    need("if (fill_half_q && fill_slice_q == OUTPUT_SLICES - 1) begin" in response and
         "postread_hit_q <= 0;" in response and "state_q <= ST_BRIDGE;" in response,
         "bridge only after final accepted response")
    need("postread_bank_request_count_q + $countones(mem_req_accept)" in rtl and
         "postread_bank_response_count_q + $countones(mem_rsp_accept)" in rtl,
         "physical bank accepts counted")
    need("core_rsp_valid && core_rsp_ready\n                        && !core_rsp_identity_legal" in rtl,
         "illegal response faults")

    for required in (
        "debug_postread_bundle_request_count == debug_postread_row_count * 12",
        "debug_postread_bundle_response_count == debug_postread_row_count * 12",
        "debug_postread_bank_request_count == debug_postread_row_count * 96",
        "debug_postread_bank_response_count == debug_postread_row_count * 96",
        "debug_postread_identity_accept_count == debug_postread_row_count * 12",
    ):
        need(required in re.sub(r"\s+", " ", sva), f"SVA conservation: {required}")

    need("localparam int BUNDLE = 4;" in tb, "B4 workload")
    need("`CONNECT_FROZEN(dut_ordinary, ordinary, 0, load_valid_o);" in tb,
         "ordinary frozen token-major axis")
    need("`CONNECT_FROZEN(dut_preread, preread, 1, load_valid_p);" in tb,
         "preread frozen group-major axis")
    need("m2213_c2_tsbg_b4_postread_causal_frontend" in tb,
         "additive post-read axis")
    for side in ("ordinary", "postread", "preread"):
        need(f"`CONNECT_MEMORY(memory_{side}, {side});" in tb,
             f"same bank-memory protocol for {side}")
        need(f"{side}.bridge_ready = tb_cycle % 11 != 3;" in tb,
             f"same bridge backpressure for {side}")
        need(f"{side}.commit_ready = tb_cycle % 13 != 5;" in tb,
             f"same commit backpressure for {side}")
        need(f"{side}.product_count != EXPECTED_PRODUCTS" in tb,
             f"exact products for {side}")
    need(tb.count(".load_context(load_context)") == 2 and
         tb.count(".load_group(load_group)") == 2 and
         tb.count(".load_source_active(load_source_active)") == 2 and
         tb.count(".load_source_sign(load_source_sign)") == 2,
         "shared descriptor wires on macro plus post-read instances")
    for axis in ("ordinary", "postread", "preread"):
        need(f"if ({axis}.commit_accumulator[lane]" in tb,
             f"Acc24 golden compare {axis}")
        need(f"{axis}.commit_tag !== 24'hA30000 + ctx" in tb,
             f"tag check {axis}")
    need(tb.count("commit_terminal !== (slice == 5)") == 3,
         "terminal exact on all axes")
    need("mismatch_o + mismatch_l + mismatch_p" in tb,
         "three-axis mismatch total")
    need("postread.scalar_bank_request_count\n                - preread.scalar_bank_request_count\n                != postread.postread_bank_request_count" in tb,
         "causal request identity")
    need("ordinary.memory_response_count[bank]\n                    != ordinary.memory_request_count[bank]" in tb and
         "postread.memory_response_count[bank]\n                    != postread.memory_request_count[bank]" in tb and
         "preread.memory_response_count[bank]\n                    != preread.memory_request_count[bank]" in tb,
         "physical per-bank response closure")
    need(not re.search(r"(?m)^\\`", tb + sva + rtl), "escaped SV backtick")

    need("assert len(passes) == 1" in parser and "assert len(covers) == 1" in parser,
         "single strict pass and cover")
    need("re.MULTILINE" in parser and "r\"^RAW_PASS" in parser and
         "preread_cycles=(\\d+)$\"" in parser,
         "anchored pass regex")
    need("(2304, 2304, 576, 1728)" in parser and
         "216, 216, 1728, 1728, 216)" in re.sub(r"\s+", " ", parser),
         "exact causal parser ledgers")
    need('"energy": False' in parser and '"paper_citable": False' in parser and
         '"component_speedup": False' in parser and '"system_speedup": False' in parser,
         "raw claim boundary")

    need("verify_dir_seal \"${M2214}\"" in runner, "exhaustive review seal gate")
    need("d['score_over_100']>=95" in runner and
         "d['severity_counts']=={'p0':0,'p1':0,'p2':0}" in runner,
         "score and severity gate")
    need("M2215_EXPECTED_RUNNER_SHA256" in runner and
         "M2215_EXPECTED_M2214_REVIEW_SHA256" in runner,
         "caller pins runner and review")
    need("[[ ! -e \"${RESULT}\" && ! -e \"${ATTEMPT}\" && ! -e \"${WORK}\" && ! -e \"${LOCK}\" ]]" in runner,
         "virgin result/attempt/work/lock gate")
    need(runner.count('"${LMUTIL}" lmstat') == 1, "one license query command")
    need(runner.count('"${VCS}" -full64') == 1, "one VCS compile command")
    need(runner.count('300s "${WORK}/simv"') == 1, "one simv command")
    need(runner.count('"${PYTHON}" -B "${PARSER}"') == 1, "one parser command")
    need("'all_other_eda_runs':0" in runner and "'automatic_retry':False" in runner and
         "'reuse_old_artifacts':False" in runner,
         "exact zero-retry authorization")
    need("same-UID EDA collision" in runner and "MemAvailable" in runner and
         "Committed_AS" in runner, "launch safety gates")
    need("FAILED_OR_INCOMPLETE_DO_NOT_CITE" in runner and
         "retry=false" in runner and "failed_or_incomplete" in runner,
         "failure quarantine")
    for digest in (
        EXPECTED[M803], EXPECTED[M2018], EXPECTED[RTL], EXPECTED[SVA],
        EXPECTED[TB], EXPECTED[FILELIST], EXPECTED[PARSER], EXPECTED[PYTHON],
        EXPECTED[VCS], EXPECTED[LMUTIL], EXPECTED[DOCS359],
    ):
        need(digest in runner, f"runner pin {digest[:12]}")


def imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(name.name.split(".")[0] for name in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def main() -> None:
    for path, digest in EXPECTED.items():
        need(path.is_file() and not path.is_symlink(), f"regular identity {path}")
        need(sha(path) == digest, f"SHA mismatch {path}")
    need((PARSER.stat().st_mode & 0o777) == 0o664, "parser mode 664")
    need((PYTHON.stat().st_mode & 0o777) == 0o755, "python mode 755")
    need(imports(PARSER) <= {"__future__", "argparse", "json", "re", "pathlib"},
         "production parser has no unpinned custom helper")

    contract = json.loads(CONTRACT.read_text())
    inventory = contract["source_inventory"]
    for relative, digest in inventory.items():
        path = HW.parent / relative
        need(path.is_file() and not path.is_symlink(), f"inventory regular file {relative}")
        need(sha(path) == digest, f"inventory drift {relative}")
    expected_filelist = [
        "hw_autoresearch_nts07/rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
        "hw_autoresearch_nts07/rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv",
        "hw_autoresearch_nts07/rtl_m2213/m2213_c2_tsbg_b4_postread_causal_frontend.sv",
        "hw_autoresearch_nts07/verif_m2213/m2213_c2_tsbg_postread_causal_assertions.sv",
        "hw_autoresearch_nts07/tb_m2213/tb_m2213_c2_tsbg_preread_postread_causal_directed.sv",
    ]
    need(FILELIST.read_text().splitlines() == expected_filelist,
         "exact exhaustive compile filelist")
    need(not RESULT.exists() and not ATTEMPT.exists() and not LOCK.exists(),
         "M2215 result/attempt/lock virgin")
    need(not list((HW / "results").glob(
        ".m2215_m2213_preread_postread_causal_vcs_work.*")),
        "M2215 work identity virgin")
    subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)], check=True,
                   capture_output=True, text=True, timeout=30)
    subprocess.run([str(PYTHON), "-B", str(TEST)], check=True,
                   capture_output=True, text=True, timeout=30)
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=AUTHOR,
                   check=True, capture_output=True, text=True, timeout=30)
    subprocess.run(["sha256sum", "-c", "SHA256SUMS.seal.sha256"], cwd=AUTHOR,
                   check=True, capture_output=True, text=True, timeout=30)

    parts = {
        "rtl": RTL.read_text(), "sva": SVA.read_text(), "tb": TB.read_text(),
        "parser": PARSER.read_text(), "runner": RUNNER.read_text(),
    }
    audit_semantics(parts)
    mutations = [
        ("hit_bypass", "rtl", "state_q <= ST_FETCH_REQ;", "state_q <= ST_BRIDGE;"),
        ("cache_write_guard", "rtl", "if (!postread_hit_q) begin\n                    for (int bank", "if (1'b1) begin\n                    for (int bank"),
        ("response_identity", "rtl", "&& core_rsp_identity_legal) begin", ") begin"),
        ("group_major", "rtl", "map_group * BUNDLE + map_ctx", "map_ctx * 48 + map_group"),
        ("bank_request_conservation", "sva", "debug_postread_bank_request_count\n            == debug_postread_row_count * 96", "debug_postread_bank_request_count >= 0"),
        ("bundle_response_conservation", "sva", "debug_postread_bundle_response_count\n            == debug_postread_row_count * 12", "debug_postread_bundle_response_count >= 0"),
        ("preread_axis", "tb", "dut_preread, preread, 1, load_valid_p", "dut_preread, preread, 0, load_valid_p"),
        ("postread_memory", "tb", "memory_postread, postread", "memory_postread, ordinary"),
        ("acc24_postread", "tb", "if (postread.commit_accumulator[lane]", "if (ordinary.commit_accumulator[lane]"),
        ("terminal_axis", "tb", "commit_terminal !== (slice == 5)", "commit_terminal === (slice == 5)"),
        ("product_preread", "tb", "preread.product_count != EXPECTED_PRODUCTS", "1'b0"),
        ("causal_difference", "tb", "!= postread.postread_bank_request_count", "!= 0"),
        ("parser_ledger", "parser", "(2304, 2304, 576, 1728)", "(2304, 2304, 2304, 0)"),
        ("parser_claim", "parser", '"energy": False', '"energy": True'),
        ("parser_anchor", "parser", "preread_cycles=(\\d+)$\"", "preread_cycles=(\\d+)\""),
        ("review_seal", "runner", "verify_dir_seal \"${M2214}\"", ": # review seal bypass"),
        ("severity_gate", "runner", "d['severity_counts']=={'p0':0,'p1':0,'p2':0}", "d['severity_counts']['p0']==0"),
        ("virgin_gate", "runner", "[[ ! -e \"${RESULT}\" && ! -e \"${ATTEMPT}\" && ! -e \"${WORK}\" && ! -e \"${LOCK}\" ]]", ": # virgin bypass"),
        ("automatic_retry", "runner", "'automatic_retry':False", "'automatic_retry':True"),
        ("m2018_pin", "runner", EXPECTED[M2018], "0" * 64),
    ]
    rejected = 0
    survivors = []
    for name, which, old, new in mutations:
        need(old in parts[which], f"mutation target present: {name}")
        variant = dict(parts)
        variant[which] = variant[which].replace(old, new, 1)
        try:
            audit_semantics(variant)
        except (AssertionError, ValueError):
            rejected += 1
        else:
            survivors.append(name)
    need(rejected == len(mutations) == 20,
         f"mutation rejection {rejected}/{len(mutations)} survivors={survivors}")

    print(json.dumps({
        "status": "PASS_M2214_INDEPENDENT_SOURCE_STATIC_AND_MUTATION_TESTS",
        "identities_recomputed": len(EXPECTED),
        "inventory_entries_verified": len(inventory),
        "custom_transitive_parser_helpers": 0,
        "bash_syntax": True,
        "author_seals_verified": True,
        "m2215_virgin": True,
        "mutations_rejected": rejected,
        "mutations_total": len(mutations),
        "vcs_runs": 0,
        "license_queries": 0,
        "eda_runs": 0,
        "gpu_runs": 0,
        "git_mutations": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
