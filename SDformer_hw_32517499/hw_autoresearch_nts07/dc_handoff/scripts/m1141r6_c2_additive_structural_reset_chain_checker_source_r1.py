#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1141R6 additive reset-chain checker; static/read-only source only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M1140 = HW / "reviews/m1140r6_m1133r6_c2_failure_quarantine_hammer_r1_20260830"
M1140_ID = (
    "6737bddab995c56f9dc550e6f03349048a625f36a2e41305ed67972774aab9d3",
    "76bd85d75c8250bfb228c492a2fccc65473590dd73a918d460726d40bd6ac2a9",
    "f36895c57ea46eda4e492a201e6ae7b0dc0b979736fa50dead3a91e240073fae",
)
NETLIST = HW / ("results/m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_20260830."
                "failed_or_incomplete.1172090.quarantine/dc/netlist/"
                "m1129r5_c2_k1_async_observation_shadow_wrapper_mapped.v")
NETLIST_SHA = "362e855cd3b4391d31dc7a08e5388d9545f289c81d291c512d25294a8539cbc4"
CELL_LIB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/"
                "TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/"
                "tcbn28hpcplusbwp35p140.v")
CELL_LIB_SHA = "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a"
SUBJECT = HW / "dc_handoff/scripts/m1133r6_c2_authority_schema_repair_engine_source_r1.py"
SUBJECT_SHA = "1f8a190d7d1c8b7804e7302c8b6a38c30a49df466b6394a82e8f0cf4cec2ee40"
CONTRACT = HW / "contracts/m1141r6_c2_additive_structural_reset_chain_checker_source_contract_r1_20260830.json"
CONTRACT_ID = (
    "60577bc578ca1c9aaa8de5b446f712fc416738a2979e4dc4b86e7ba9b1bf5b37",
    "188ae927bc8ef40085e0e550a29d4892d94647717624487a2d9a4fdb34fa9196",
    "3f4c7cc217fdac94ca061883314bb1b2160a0ee3436ff27f3321a3f70e7a4479",
)
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CANONICAL_ROOT = "rst_core"
MAX_UNARY_CELLS = 8
SHADOW_REGISTER_BITS = 337
BUFFERS = {"BUFFD1BWP35P140": ("I", "Z")}
INVERTERS = {"CKND0BWP35P140": ("I", "ZN")}
OUTPUT_PINS = ("Z", "ZN", "Q", "QN")
CONSTANT_RE = re.compile(r"(?:[01]|\d+'[bBoOdDhH][0-9a-fA-FxXzZ_]+)")


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and
            sha256(path) == expected, "identity drift: " + str(path))


def strict_json(path: Path) -> Any:
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def verify_double(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, identity[0]); verify_regular(side, identity[1])
    verify_regular(outer, identity[2])
    require(side.read_text(encoding="utf-8").split() == [identity[0], path.name] and
            outer.read_text(encoding="utf-8").split() == [identity[1], side.name],
            "double-seal content drift")


def verify_m1140() -> dict[str, Any]:
    review = M1140 / "review.json"; manifest = M1140 / "SHA256SUMS"
    outer = M1140 / "SHA256SUMS.seal.sha256"
    verify_regular(review, M1140_ID[0]); verify_regular(manifest, M1140_ID[1])
    verify_regular(outer, M1140_ID[2])
    require(outer.read_text(encoding="utf-8").split() == [M1140_ID[1], "SHA256SUMS"],
            "M1140 sole authorization outer content")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*"); rel = Path(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and
                name not in expected and name == rel.as_posix() and
                not rel.is_absolute() and ".." not in rel.parts,
                "M1140 safe manifest member")
        expected[name] = digest
    actual = set()
    for member in M1140.rglob("*"):
        name = member.relative_to(M1140).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "M1140 sealed symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "M1140 sealed special member")
    require(actual == set(expected), "M1140 exact member census")
    for name, digest in expected.items():
        verify_regular(M1140 / name, digest)
    value = strict_json(review)
    require(value["status"] ==
            "PASS_M1140R6_M1133R6_FAILURE_AUDIT__CHECKER_FALSE_NEGATIVE__AUTHOR_ADDITIVE_STRUCTURAL_CHECKER_REPAIR_SOURCE_ONLY" and
            value["authorization"]["additive_structural_checker_repair_source_authoring"] is True and
            value["authorization"]["launch"] is False and
            value["authorization"]["dc"] is False and
            value["authorization"]["mapped_vcs"] is False,
            "M1140 authorization drift")
    return value


def prove_foundry_unary_semantics() -> dict[str, Any]:
    verify_regular(CELL_LIB, CELL_LIB_SHA)
    library = CELL_LIB.read_text(encoding="utf-8", errors="strict")
    proven = {}
    for cell, (input_pin, output_pin) in {**BUFFERS, **INVERTERS}.items():
        matches = re.findall(
            rf"(?ms)^module\s+{re.escape(cell)}\s*\(\s*{input_pin}\s*,\s*{output_pin}\s*\)\s*;(.*?)^endmodule",
            library)
        require(len(matches) == 1, "foundry unary module cardinality: " + cell)
        primitive = "buf" if cell in BUFFERS else "not"
        calls = re.findall(r"\b(buf|not|and|nand|or|nor|xor|xnor)\s*\(\s*([^;]+?)\s*\)\s*;",
                           matches[0])
        require(calls == [(primitive, f"{output_pin}, {input_pin}")],
                "foundry unary truth semantics drift: " + cell)
        proven[cell] = {"input_pin": input_pin, "output_pin": output_pin,
                        "primitive": primitive}
    return {"library_sha256": CELL_LIB_SHA, "proven_cells": proven}


def source_preflight() -> dict[str, Any]:
    authorization = verify_m1140()
    verify_regular(NETLIST, NETLIST_SHA); verify_regular(SUBJECT, SUBJECT_SHA)
    verify_double(CONTRACT, CONTRACT_ID); verify_regular(DOCS359, DOCS359_SHA)
    semantics = prove_foundry_unary_semantics()
    return {
        "status": "PASS_M1141R6_SOURCE_PREFLIGHT__STATIC_READ_ONLY_NO_LAUNCH",
        "sole_authorization_outer_seal_file_sha256": M1140_ID[2],
        "m1140_checker_false_negative":
            authorization["findings"]["checker_false_negative"],
        "foundry_semantics": semantics,
        "subject_modified": False,
        "failure_directory_modified": False,
        "launch_dc_vcs_mapped_vcs": False,
    }


def parse_instances(text: str) -> list[tuple[str, str, dict[str, str]]]:
    instances = []
    names = set()
    pattern = re.compile(r"(?ms)^\s*(\w+)\s+(\\?[^\s(]+)\s*\((.*?)\)\s*;")
    for match in pattern.finditer(text):
        pins = {}
        for pin, net in re.findall(r"\.(\w+)\s*\(\s*([^()\s,]+)\s*\)", match.group(3)):
            require(pin not in pins, "duplicate mapped pin: " + match.group(2))
            pins[pin] = net
        if not pins:
            continue
        require(match.group(2) not in names, "duplicate mapped instance name")
        names.add(match.group(2)); instances.append((match.group(1), match.group(2), pins))
    return instances


def driver_index(instances: list[tuple[str, str, dict[str, str]]]
                 ) -> dict[str, list[tuple[str, str, dict[str, str]]]]:
    drivers: dict[str, list[tuple[str, str, dict[str, str]]]] = {}
    for cell, name, pins in instances:
        for output_pin in OUTPUT_PINS:
            if output_pin in pins:
                drivers.setdefault(pins[output_pin], []).append((cell, name, pins))
    return drivers


def _trace_unary(net: str,
                 drivers: Mapping[str, list[tuple[str, str, dict[str, str]]]],
                 seen_nets: frozenset[str], seen_instances: frozenset[str],
                 depth: int) -> tuple[tuple[dict[str, Any], ...], int]:
    """Recursively trace one driver/input per level to the canonical root."""
    if net == CANONICAL_ROOT:
        return (), 0
    require(CONSTANT_RE.fullmatch(net) is None, "constant reset-chain source: " + net)
    require(depth < MAX_UNARY_CELLS, "reset-chain depth exceeds bounded maximum")
    require(net not in seen_nets, "reset-chain cycle: " + net)
    source = drivers.get(net, [])
    require(len(source) == 1, "reset-chain driver cardinality is not one: " + net)
    cell, instance, pins = source[0]
    require(instance not in seen_instances, "reset-chain cycle/reconvergence instance")
    if cell in BUFFERS:
        input_pin, output_pin = BUFFERS[cell]; polarity = "preserve"; inversion = 0
    elif cell in INVERTERS:
        input_pin, output_pin = INVERTERS[cell]; polarity = "invert"; inversion = 1
    else:
        raise Failure("unknown/non-unary/reconvergent reset driver cell: " + cell)
    require(set(pins) == {input_pin, output_pin} and pins[output_pin] == net,
            "unary foundry cell pin/output schema drift: " + cell)
    upstream = pins[input_pin]
    tail, tail_inversions = _trace_unary(
        upstream, drivers, seen_nets | frozenset((net,)),
        seen_instances | frozenset((instance,)), depth + 1)
    step = {"cell": cell, "instance": instance, "input": upstream,
            "output": net, "polarity": polarity}
    return (step,) + tail, inversion + tail_inversions


def structural_reset_chain_gate_text(text: str, expected_shadow_bits: int) -> dict[str, Any]:
    require(type(text) is str and type(expected_shadow_bits) is int and
            expected_shadow_bits > 0, "exact text/positive shadow census required")
    instances = parse_instances(text); drivers = driver_index(instances)
    shadow = [(cell, name, pins) for cell, name, pins in instances
              if re.search(r"shadow_\w+_q_reg", name)]
    require(len(shadow) == expected_shadow_bits,
            f"mapped shadow register census {len(shadow)} != {expected_shadow_bits}")
    paths: dict[str, tuple[dict[str, Any], ...]] = {}
    reset_net_register_counts: dict[str, int] = {}
    cell_types = set()
    inverter_count_by_register = []
    for cell, name, pins in shadow:
        cell_types.add(cell)
        clear_pins = [pin for pin in ("CDN", "CN") if pin in pins]
        require(len(clear_pins) == 1, name + ": exactly one active-low clear pin required")
        for set_pin in (pin for pin in ("SDN", "SN") if pin in pins):
            require(pins[set_pin] in {"1'b1", "1'h1", "1"},
                    name + ": async set must be inactive high")
        clear_net = pins[clear_pins[0]]
        path, inversions = _trace_unary(clear_net, drivers, frozenset(), frozenset(), 0)
        require(inversions == 1, name + ": clear path must contain exactly one inverter")
        require(path and path[-1]["input"] == CANONICAL_ROOT,
                name + ": reset chain does not terminate at rst_core")
        if clear_net in paths:
            require(paths[clear_net] == path, "shared reset path instability")
        else:
            paths[clear_net] = path
        reset_net_register_counts[clear_net] = reset_net_register_counts.get(clear_net, 0) + 1
        inverter_count_by_register.append(inversions)
    direct = sum(count for net, count in reset_net_register_counts.items()
                 if len(paths[net]) == 1)
    buffered = expected_shadow_bits - direct
    require(all(value == 1 for value in inverter_count_by_register),
            "per-register inverter conservation")
    return {
        "schema": "m1141r6_structural_reset_chain_gate_result_v1",
        "status": "PASS_SINGLE_DRIVER_UNARY_CHAIN__EXACTLY_ONE_INVERTER_TO_RST_CORE",
        "shadow_register_bits": expected_shadow_bits,
        "resettable_cell_types": sorted(cell_types),
        "active_low_clear_nets": len(paths),
        "direct_inverter_registers": direct,
        "buffered_then_inverter_registers": buffered,
        "maximum_chain_cells": max(len(path) for path in paths.values()),
        "all_paths_single_driver_unary": True,
        "all_paths_exactly_one_inverter": True,
        "all_paths_end_rst_core": True,
        "foundry_proven_buffers": sorted(BUFFERS),
        "foundry_proven_inverters": sorted(INVERTERS),
        "reset_net_register_counts": dict(sorted(reset_net_register_counts.items())),
        "paths": {net: list(path) for net, path in sorted(paths.items())},
    }


def frozen_quarantine_static_oracle() -> dict[str, Any]:
    verify_regular(NETLIST, NETLIST_SHA)
    result = structural_reset_chain_gate_text(
        NETLIST.read_text(encoding="utf-8", errors="strict"), SHADOW_REGISTER_BITS)
    require(result["active_low_clear_nets"] == 12 and
            result["direct_inverter_registers"] == 75 and
            result["buffered_then_inverter_registers"] == 262 and
            result["maximum_chain_cells"] == 2,
            "frozen quarantine reset-chain oracle drift")
    return result


def _bounded_netlist(buffered: bool) -> str:
    cells = (["BUFFD1BWP35P140 UBUF (.I(rst_core), .Z(nbuf));",
              "CKND0BWP35P140 UINV (.I(nbuf), .ZN(nclear));"] if buffered else
             ["CKND0BWP35P140 UINV (.I(rst_core), .ZN(nclear));"])
    return "\n".join([
        "module bounded(input rst_core, input d, input clk, output q);",
        "DFCNQD1BWP35P140 shadow_bounded_q_reg (.D(d), .CP(clk), .CDN(nclear), .Q(q));",
        *cells, "endmodule",
    ])


def source_small_oracle() -> dict[str, Any]:
    preflight = source_preflight()
    direct = structural_reset_chain_gate_text(_bounded_netlist(False), 1)
    buffered = structural_reset_chain_gate_text(_bounded_netlist(True), 1)
    frozen = frozen_quarantine_static_oracle()
    return {
        "schema": "m1141r6_additive_structural_checker_small_oracle_v1",
        "status": "PASS_M1141R6_BOUNDED_AND_FROZEN_STATIC_ORACLE__NO_LAUNCH_NO_EDA",
        "preflight": preflight,
        "bounded": {
            "direct_chain_cells": direct["maximum_chain_cells"],
            "buffered_chain_cells": buffered["maximum_chain_cells"],
            "both_exactly_one_inverter": True,
        },
        "frozen_quarantine": {
            "netlist_sha256": NETLIST_SHA,
            "shadow_register_bits": frozen["shadow_register_bits"],
            "active_low_clear_nets": frozen["active_low_clear_nets"],
            "direct_inverter_registers": frozen["direct_inverter_registers"],
            "buffered_then_inverter_registers":
                frozen["buffered_then_inverter_registers"],
            "maximum_chain_cells": frozen["maximum_chain_cells"],
            "read_only": True,
        },
        "subject_modified": False,
        "failure_directory_modified": False,
        "mapped_vcs_dc_launch_retry": False,
        "diagnostic_area_setup_promoted": False,
        "paper_citable": False,
    }


def main() -> None:
    require(sys.argv[1:] in ([], ["--self-test"]), "static self-test only")
    print(json.dumps(source_small_oracle(), indent=2, sort_keys=True,
                     allow_nan=False))


if __name__ == "__main__":
    main()
