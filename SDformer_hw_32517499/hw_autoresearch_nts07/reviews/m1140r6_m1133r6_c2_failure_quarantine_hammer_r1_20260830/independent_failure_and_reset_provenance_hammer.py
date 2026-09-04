#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1140r6 read-only failure quarantine and reset-provenance hammer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
RESULTS = HW / "results"
ATTEMPT = RESULTS / ".m1133r6_c2_authority_schema_repair_dc_mapped_vcs_attempt_consumed"
ATTEMPT_ID = (
    "cfc25412e18d126614768a1a39f38fba101f9a77b556b548bee17f72a13cf317",
    "684c15bdd5a3f4317115b42eeda04ba404e3ddbbe65366dfb0165bd723b38036",
    "83f5e0a0bc5215242c75940e9b2d9560dda999f9fa5e7a467d9a9b006ee9129a",
)
FAILURE = RESULTS / "m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_20260830.failed_or_incomplete.1172090.quarantine"
FAILURE_ID = (
    "e0780bf99273c497bba6ecc4d966df54138681715b5072f631922ad199c9b832",
    "cbac2199f94723aa39ec3ae2e3b535dfa03e509cedb0b6ac226269b8eab7dd7e",
    "08ed7238836c58df1d9f6ccf58e530468413df82d18db5a9d3aabce79a1f3455",
)
RESULT = RESULTS / "m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_20260830"
NETLIST = FAILURE / "dc/netlist/m1129r5_c2_k1_async_observation_shadow_wrapper_mapped.v"
NETLIST_SHA = "362e855cd3b4391d31dc7a08e5388d9545f289c81d291c512d25294a8539cbc4"
AREA = FAILURE / "dc/reports/area.rpt"
AREA_SHA = "a3cba1220145ed4ca2b108d67febc3dd7a921528ccd7e5cc7259ca2cfa7dd600"
QOR = FAILURE / "dc/reports/qor.rpt"
QOR_SHA = "50c7b933872e26f2b01ee1885f861b9ea45ab849702f62cbb2b1b3651cdaf0d8"
COMPILE = FAILURE / "dc/reports/compile_receipt.rpt"
COMPILE_SHA = "39033ebc18a74be223d7815676c04647aa5e0d38eddc3305003821bd10a0cea6"
CHECK_DESIGN = FAILURE / "dc/reports/check_design_postcompile.rpt"
CHECK_DESIGN_SHA = "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865"
CHECK_TIMING = FAILURE / "dc/reports/check_timing_postcompile.rpt"
CHECK_TIMING_SHA = "94bf7675a38cf2f7e109edbc8dc51e1112bf6d2b3427362c2e14197d434eaf38"
TERMINAL = FAILURE / "dc/TCL_PASS_TERMINAL.txt"
TERMINAL_SHA = "ff97daef1e1394e0fc52fec45afe34395ba062bc5b26b2dae8d03dfa79fda3a0"
SELECTOR = FAILURE / "dc/dc_selector_runtime_identity.json"
SELECTOR_SHA = "10b5a4a60248bae855c9e00805a58c674e2d90c2c5e884f322c9bc56dc629148"
BASE_ENGINE = HW / "dc_handoff/scripts/m1129r5_c2_real_module_async_observation_engine_source_r1.py"
BASE_ENGINE_SHA = "c8fd3366ecf6c4377b62e5717d959348c08192ea8bdbd0afd3b0e566bd6fbd0b"
R6_ENGINE = HW / "dc_handoff/scripts/m1133r6_c2_authority_schema_repair_engine_source_r1.py"
R6_ENGINE_SHA = "1f8a190d7d1c8b7804e7302c8b6a38c30a49df466b6394a82e8f0cf4cec2ee40"
CONTRACT = HW / "contracts/m1133r6_c2_authority_schema_repair_engine_source_contract_r1_20260830.json"
CONTRACT_SHA = "4dc16ffccb3c4a145f69f565500d67407ca821304ee838f93659918055a3ac8a"
CELL_LIB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/tcbn28hpcplusbwp35p140.v")
CELL_LIB_SHA = "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
checks = 0


def require(value: bool, message: str) -> None:
    global checks
    if not value:
        raise RuntimeError(message)
    checks += 1


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha(path) == expected,
            "regular identity drift: " + str(path))


def strict_json(path: Path) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite: " + token)))


def exact_flat(directory: Path, identity: tuple[str, str, str]) -> dict:
    primary_name = "attempt.json" if directory == ATTEMPT else "failure.json"
    primary = directory / primary_name; manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular(primary, identity[0]); regular(manifest, identity[1]); regular(outer, identity[2])
    require(directory.is_dir() and not directory.is_symlink() and
            outer.read_text(encoding="utf-8").split() == [identity[1], "SHA256SUMS"],
            "flat outer content")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1); name = name.lstrip("*"); relative = Path(name)
        require(name not in expected and relative.as_posix() == name and
                not relative.is_absolute() and ".." not in relative.parts,
                "safe manifest member")
        expected[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(expected), "sealed exact member census")
    for name, digest in expected.items():
        regular(directory / name, digest)
    return strict_json(primary)


def parse_instances(text: str) -> list[tuple[str, str, dict[str, str]]]:
    instances = []
    pattern = re.compile(r"(?ms)^\s*(\w+)\s+(\\?[^\s(]+)\s*\((.*?)\)\s*;")
    for match in pattern.finditer(text):
        pins = {}
        for pin, net in re.findall(r"\.(\w+)\s*\(\s*([^()\s,]+)\s*\)", match.group(3)):
            require(pin not in pins, "duplicate mapped pin")
            pins[pin] = net
        instances.append((match.group(1), match.group(2), pins))
    return instances


def inverter(cell: str, pins: dict[str, str]) -> bool:
    return (re.fullmatch(r"(?:INV(?:D(?:0P7|1P5|0|1|2|3|4|6|8|9|12|15|16|18|20|21|24|32))|CKND(?:0|1|2|3|4|6|8|12|16|20|24))BWP35P140", cell)
            is not None and set(pins) == {"I", "ZN"})


def buffer(cell: str, pins: dict[str, str]) -> bool:
    return (re.fullmatch(r"(?:BUFFD(?:0P5|0P7|1|1P5|2|3|4|6|8|12|16|20|24|32)|CKBD(?:0|1|2|3|4|6|8|12|16|20|24))BWP35P140", cell)
            is not None and set(pins) == {"I", "Z"})


def reset_provenance() -> dict[str, Any]:
    text = NETLIST.read_text(encoding="utf-8", errors="strict")
    instances = parse_instances(text)
    drivers: dict[str, list[tuple[str, str, dict[str, str]]]] = {}
    for cell, name, pins in instances:
        for output in ("ZN", "Z", "Q", "QN"):
            if output in pins:
                drivers.setdefault(pins[output], []).append((cell, name, pins))
    shadow = [(cell, name, pins) for cell, name, pins in instances
              if re.search(r"shadow_\w+_q_reg", name)]
    require(len(shadow) == 337, "shadow register census")
    paths = {}
    reset_net_register_counts: dict[str, int] = {}
    cell_types = set()

    def trace(net: str):
        current = net; parity = 0; path = []
        seen = set()
        while current != "rst_core":
            require(current not in seen and len(path) < 8, "reset trace cycle/depth")
            seen.add(current)
            source = drivers.get(current, [])
            require(len(source) == 1, "reset trace driver cardinality: " + current)
            cell, name, pins = source[0]
            if inverter(cell, pins):
                require(pins["ZN"] == current, "inverter output")
                path.append({"cell": cell, "instance": name, "input": pins["I"],
                             "output": current, "polarity": "invert"})
                parity ^= 1; current = pins["I"]
            elif buffer(cell, pins):
                require(pins["Z"] == current, "buffer output")
                path.append({"cell": cell, "instance": name, "input": pins["I"],
                             "output": current, "polarity": "preserve"})
                current = pins["I"]
            else:
                raise RuntimeError("reset trace entered non-buffer logic: " + cell)
        require(parity == 1, "active-low clear does not equal !rst_core")
        return path

    for cell, name, pins in shadow:
        clear = [pin for pin in ("CDN", "CN") if pin in pins]
        require(len(clear) == 1, "one active-low clear pin")
        clear_net = pins[clear[0]]
        reset_net_register_counts[clear_net] = reset_net_register_counts.get(clear_net, 0) + 1
        cell_types.add(cell)
        path = trace(clear_net)
        paths.setdefault(clear_net, path)
        require(paths[clear_net] == path, "reset net path instability")
        for set_pin in (pin for pin in ("SDN", "SN") if pin in pins):
            require(pins[set_pin] in {"1'b1", "1'h1", "1"}, "async set active")

    target = next((pins for _cell, name, pins in shadow
                   if name == "shadow_service_result_count_q_reg_22_"), None)
    require(target is not None and target["CDN"] == "n186651", "failure target identity")
    require(paths["n186651"] == [
        {"cell": "CKND0BWP35P140", "instance": "U114871", "input": "n111637",
         "output": "n186651", "polarity": "invert"},
        {"cell": "BUFFD1BWP35P140", "instance": "U104338", "input": "rst_core",
         "output": "n111637", "polarity": "preserve"},
    ], "target exact two-cell reset path")
    via_buffer = sum(count for net, count in reset_net_register_counts.items()
                     if len(paths[net]) == 2)
    direct = sum(count for net, count in reset_net_register_counts.items()
                 if len(paths[net]) == 1)
    require(direct + via_buffer == 337 and via_buffer > 0,
            "direct/buffered reset coverage")
    return {
        "shadow_register_bits": 337,
        "shadow_register_cell_types": sorted(cell_types),
        "active_low_clear_nets": len(paths),
        "reset_net_register_counts": dict(sorted(reset_net_register_counts.items())),
        "direct_inverter_registers": direct,
        "buffer_then_inverter_registers": via_buffer,
        "all_paths_end_rst_core": True,
        "all_paths_inversion_parity": 1,
        "all_paths_single_driver": True,
        "non_buffer_logic_paths": 0,
        "constant_paths": 0,
        "cycle_paths": 0,
        "target": {"register": "shadow_service_result_count_q_reg_22_",
                   "clear_net": "n186651", "path": paths["n186651"],
                   "logical_equation": "n186651 = NOT(BUF(rst_core)) = NOT(rst_core)"},
        "unique_paths": paths,
    }


def main() -> None:
    regular(BASE_ENGINE, BASE_ENGINE_SHA); regular(R6_ENGINE, R6_ENGINE_SHA)
    regular(CONTRACT, CONTRACT_SHA); regular(CELL_LIB, CELL_LIB_SHA)
    regular(DOCS359, DOCS359_SHA)
    attempt = exact_flat(ATTEMPT, ATTEMPT_ID)
    failure = exact_flat(FAILURE, FAILURE_ID)
    attempts = list(RESULTS.glob(".m1133r6_c2_authority_schema_repair_dc_mapped_vcs_attempt_consumed"))
    failures = list(RESULTS.glob("m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*"))
    work = list(RESULTS.glob(".m1133r6_c2_authority_schema_repair_dc_mapped_vcs_work.*"))
    require(attempts == [ATTEMPT] and failures == [FAILURE] and work == [] and
            not RESULT.exists() and not RESULT.is_symlink(), "exactly-one/result-absent namespace")
    require(set(attempt) == {"contract_sha256", "dc_attempts", "engine_sha256",
            "launcher_sha256", "m1121_outer_seal_file_sha256",
            "m1132r5_stop_outer_seal_file_sha256", "m1134r6_outer_seal_file_sha256",
            "m1136r6_outer_seal_file_sha256", "mapped_cases", "random_initialization",
            "status"} and attempt["dc_attempts"] == 1 and attempt["mapped_cases"] == 1 and
            attempt["random_initialization"] is False and
            attempt["status"] == "M1133R6_ATTEMPT_CONSUMED_AFTER_M1134R6_M1136R6" and
            attempt["engine_sha256"] == R6_ENGINE_SHA and
            attempt["contract_sha256"] == CONTRACT_SHA,
            "attempt schema/one-shot identity")
    require(failure == {
        "m1112r3_retry": False, "m1122r4_retry": False,
        "m1129r5_retry": False, "m1133r6_retry": False,
        "message": "shadow_service_result_count_q_reg_22_: inverter input is not canonical rst_core",
        "phase": "MAPPED_RESET_PROVENANCE_337",
        "status": "FAILED_DIAGNOSTIC_DO_NOT_CITE",
    }, "failure phase/message/no-retry")
    contract = strict_json(CONTRACT)
    require(contract["future_namespaces"]["maximum_attempts_after_all_hammers"] == 1 and
            contract["future_namespaces"]["automatic_retry"] is False and
            contract["authorization"]["automatic_retry"] is False,
            "contract no automatic retry")

    for path, expected in ((NETLIST, NETLIST_SHA), (AREA, AREA_SHA), (QOR, QOR_SHA),
                           (COMPILE, COMPILE_SHA), (CHECK_DESIGN, CHECK_DESIGN_SHA),
                           (CHECK_TIMING, CHECK_TIMING_SHA), (TERMINAL, TERMINAL_SHA),
                           (SELECTOR, SELECTOR_SHA)):
        regular(path, expected)
    terminal = TERMINAL.read_text(encoding="utf-8")
    compile_text = COMPILE.read_text(encoding="utf-8")
    qor = QOR.read_text(encoding="utf-8")
    area = AREA.read_text(encoding="utf-8")
    selector = strict_json(SELECTOR)
    require("status=PASS_M519_R8_SETUP_AREA_DC_TCL_TERMINAL" in terminal and
            "compile_ultra_count=1" in terminal and "TIM-209=0" in terminal and
            "OPT-150=0" in terminal and "hold_not_closed_at_dc=true" in terminal and
            "compile_ultra_count=1" in compile_text and
            "incremental_compile_count=0" in compile_text and
            "Cell Count" in qor and "153512" in qor and "125686.889144" in qor and
            "Number of cells:" in area and "153512" in area and
            selector["status"] == "PASS_M1129R5_EXACT_DC_SELECTOR_RUNTIME_CAPTURE" and
            selector["ppid"] == 1172090,
            "DC terminal/report/runtime identity")
    require(not (FAILURE / "mapped_vcs").exists(), "mapped VCS must not run after structural stop")

    library = CELL_LIB.read_text(encoding="utf-8", errors="strict")
    require(re.search(r"module BUFFD1BWP35P140 \(I, Z\);.*?buf \(Z, I\);.*?endmodule",
                      library, re.S) is not None and
            re.search(r"module CKND0BWP35P140 \(I, ZN\);.*?not \(ZN, I\);.*?endmodule",
                      library, re.S) is not None,
            "standard-cell buffer/inverter truth semantics")
    provenance = reset_provenance()
    result = {
        "schema": "m1140r6_m1133r6_c2_failure_quarantine_hammer_r1_v1",
        "status": "PASS_M1140R6_M1133R6_FAILURE_AUDIT__CHECKER_FALSE_NEGATIVE__AUTHOR_ADDITIVE_STRUCTURAL_CHECKER_REPAIR_SOURCE_ONLY",
        "checks_passed": checks,
        "attempt": {"exactly_one": True, "dc_attempts": 1,
                    "outer_seal_file_sha256": ATTEMPT_ID[2]},
        "failure": {"exactly_one": True, "phase": failure["phase"],
                    "message": failure["message"], "result_absent": True,
                    "mapped_vcs_not_run": True,
                    "outer_seal_file_sha256": FAILURE_ID[2],
                    "automatic_retry": False},
        "dc_evidence": {"netlist_sha256": NETLIST_SHA, "area_sha256": AREA_SHA,
                        "qor_sha256": QOR_SHA, "compile_receipt_sha256": COMPILE_SHA,
                        "selector_runtime_sha256": SELECTOR_SHA,
                        "compile_ultra_count": 1, "leaf_cells": 153512,
                        "cell_area_um2": 125686.889144,
                        "setup_wns_ns": 0.0, "hold_not_closed_at_dc": True,
                        "paper_citable": False},
        "reset_provenance": provenance,
        "root_cause": {
            "classification": "STRUCTURAL_CHECKER_FALSE_NEGATIVE",
            "actual_reset_provenance_break": False,
            "checker_overstrict_rule": "requires clear-driving inverter input to be exactly rst_core",
            "legal_synthesis_rewrite": "non-inverting BUFFD1 reset-tree branch before CKND0",
            "required_repair": "trace a bounded single-driver chain of proven polarity-preserving buffers and exactly one inversion back to rst_core; continue rejecting gates, constants, cycles, multiple drivers, wrong parity and reconvergence"
        },
        "identity": {"attempt_outer_seal_file_sha256": ATTEMPT_ID[2],
                     "failure_outer_seal_file_sha256": FAILURE_ID[2],
                     "r6_engine_sha256": R6_ENGINE_SHA,
                     "base_checker_engine_sha256": BASE_ENGINE_SHA,
                     "cell_library_sha256": CELL_LIB_SHA,
                     "docs359_sha256": DOCS359_SHA},
        "authorization": {"additive_structural_checker_repair_source_only": True,
                          "retry": False, "launch": False, "dc": False,
                          "vcs": False, "mapped_vcs": False,
                          "automatic_retry": False},
        "claim_boundary": {"read_only_failure_and_netlist_audit": True,
                           "mapped_functionality": False,
                           "area_timing_power_energy": False,
                           "cycles_speedup": False, "paper_citable": False},
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
