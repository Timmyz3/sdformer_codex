#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1141R6 author check: bounded/static only, no launch, VCS, or DC."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat
import sys
from typing import Any, Callable

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "dc_handoff/scripts/m1141r6_c2_additive_structural_reset_chain_checker_source_r1.py"
CONTRACT = HW / "contracts/m1141r6_c2_additive_structural_reset_chain_checker_source_contract_r1_20260830.json"
NETLIST = HW / ("results/m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_20260830."
                "failed_or_incomplete.1172090.quarantine/dc/netlist/"
                "m1129r5_c2_k1_async_observation_shadow_wrapper_mapped.v")
FAILURE = NETLIST.parents[2]
SUBJECT = HW / "dc_handoff/scripts/m1133r6_c2_authority_schema_repair_engine_source_r1.py"
CELL_LIB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/"
                "TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/"
                "tcbn28hpcplusbwp35p140.v")
M1140_OUTER = HW / ("reviews/m1140r6_m1133r6_c2_failure_quarantine_hammer_r1_20260830/"
                    "SHA256SUMS.seal.sha256")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "source": "86ccd46fdaffcad77444ca105bde1593394dd7643febba1f6a45680bf515965e",
    "contract": "60577bc578ca1c9aaa8de5b446f712fc416738a2979e4dc4b86e7ba9b1bf5b37",
    "contract_side": "188ae927bc8ef40085e0e550a29d4892d94647717624487a2d9a4fdb34fa9196",
    "contract_outer": "3f4c7cc217fdac94ca061883314bb1b2160a0ee3436ff27f3321a3f70e7a4479",
    "netlist": "362e855cd3b4391d31dc7a08e5388d9545f289c81d291c512d25294a8539cbc4",
    "subject": "1f8a190d7d1c8b7804e7302c8b6a38c30a49df466b6394a82e8f0cf4cec2ee40",
    "cell_lib": "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
    "m1140_outer": "f36895c57ea46eda4e492a201e6ae7b0dc0b979736fa50dead3a91e240073fae",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
checks = 0
attacks: dict[str, str] = {}


class CheckFailure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise CheckFailure(message)


def rejected(label: str, action: Callable[[], Any], contains: str | None = None) -> None:
    try:
        action()
    except Exception as error:
        if contains is not None:
            require(contains in str(error), label + " wrong rejection: " + str(error))
        attacks[label] = type(error).__name__ + ": " + str(error)
        return
    raise CheckFailure("attack accepted: " + label)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_regular(path: Path, expected: str) -> None:
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and
            sha(path) == expected, "identity drift: " + str(path))


def strict_json(path: Path) -> Any:
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          CheckFailure("nonfinite JSON: " + token)))


def verify_frozen() -> None:
    for path, expected in ((SOURCE, EXPECTED["source"]),
                           (CONTRACT, EXPECTED["contract"]),
                           (Path(str(CONTRACT) + ".sha256"), EXPECTED["contract_side"]),
                           (Path(str(CONTRACT) + ".sha256.seal.sha256"),
                            EXPECTED["contract_outer"]),
                           (NETLIST, EXPECTED["netlist"]),
                           (SUBJECT, EXPECTED["subject"]),
                           (CELL_LIB, EXPECTED["cell_lib"]),
                           (M1140_OUTER, EXPECTED["m1140_outer"]),
                           (DOCS359, EXPECTED["docs359"])):
        verify_regular(path, expected)
    side = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    require(side.read_text(encoding="utf-8").split() ==
            [EXPECTED["contract"], CONTRACT.name] and
            outer.read_text(encoding="utf-8").split() ==
            [EXPECTED["contract_side"], side.name], "contract double seal")


def tree_signature(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        rel = path.relative_to(directory).as_posix()
        mode = path.lstat().st_mode
        require(not stat.S_ISLNK(mode), "failure tree symlink")
        if stat.S_ISREG(mode):
            digest.update(b"F\0" + rel.encode() + b"\0" + bytes.fromhex(sha(path)))
        else:
            require(stat.S_ISDIR(mode), "failure tree special member")
            digest.update(b"D\0" + rel.encode() + b"\0")
    return digest.hexdigest()


def load_subject():
    spec = importlib.util.spec_from_file_location("m1141r6_author_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "subject module spec")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def static_checks() -> dict[str, Any]:
    text = SOURCE.read_text(encoding="utf-8"); tree = ast.parse(text)
    contract = strict_json(CONTRACT)
    recursive = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "_trace_unary")
    recursive_calls = [node for node in ast.walk(recursive) if isinstance(node, ast.Call) and
                       isinstance(node.func, ast.Name) and node.func.id == "_trace_unary"]
    require(len(recursive_calls) == 1, "exact recursive unary predecessor call")
    imports = {alias.name for node in ast.walk(tree)
               if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    require("subprocess" not in imports and "socket" not in imports and
            "os" not in imports, "launch-capable import entered checker")
    require("compile_ultra" not in text and "dc_shell" not in text and
            "simv" not in text and "subprocess" not in text,
            "EDA/launch token entered checker")
    require(contract["sole_authorization"]["outer_seal_file_sha256"] ==
            EXPECTED["m1140_outer"] and
            contract["recursive_chain_contract"]["single_driver_each_non_root_net"] is True and
            contract["recursive_chain_contract"]["allowed_inverter_count"] == 1 and
            contract["recursive_chain_contract"]["multi_input_reconvergence_allowed"] is False and
            contract["authorization"]["different_author_hammer_only_next"] is True and
            contract["forbidden"]["promote_diagnostic_area_or_setup"] is True,
            "contract authorization/chain drift")
    return {"recursive_calls": 1, "launch_capable_imports": 0,
            "eda_launch_tokens": 0, "maximum_unary_cells": 8,
            "sole_authorization_outer": EXPECTED["m1140_outer"]}


def independent_foundry_check() -> dict[str, str]:
    library = CELL_LIB.read_text(encoding="utf-8", errors="strict")
    expectations = {
        "BUFFD1BWP35P140": ("I", "Z", "buf"),
        "CKND0BWP35P140": ("I", "ZN", "not"),
    }
    proven = {}
    for cell, (input_pin, output_pin, primitive) in expectations.items():
        bodies = re.findall(
            rf"(?ms)^module\s+{cell}\s*\(\s*{input_pin}\s*,\s*{output_pin}\s*\)\s*;(.*?)^endmodule",
            library)
        require(len(bodies) == 1, "independent foundry module cardinality")
        calls = re.findall(r"\b(buf|not|and|nand|or|nor|xor|xnor)\s*\(\s*([^;]+?)\s*\)\s*;",
                           bodies[0])
        require(calls == [(primitive, f"{output_pin}, {input_pin}")],
                "independent foundry truth primitive")
        proven[cell] = f"{primitive}({output_pin},{input_pin})"
    return proven


def shell(clear_net: str, cells: list[str], set_net: str | None = None) -> str:
    set_pin = "" if set_net is None else f", .SDN({set_net})"
    return "\n".join([
        "module mutation(input rst_core, input other_root, input d, input clk, output q);",
        f"DFCNQD1BWP35P140 shadow_mutation_q_reg (.D(d), .CP(clk), .CDN({clear_net}){set_pin}, .Q(q));",
        *cells, "endmodule",
    ])


def bounded_mutations(module) -> None:
    gate = lambda text: module.structural_reset_chain_gate_text(text, 1)
    direct = shell("nclear", [
        "CKND0BWP35P140 UINV (.I(rst_core), .ZN(nclear));"])
    buffered = shell("nclear", [
        "BUFFD1BWP35P140 UBUF (.I(rst_core), .Z(nbuf));",
        "CKND0BWP35P140 UINV (.I(nbuf), .ZN(nclear));"])
    require(gate(direct)["maximum_chain_cells"] == 1 and
            gate(buffered)["maximum_chain_cells"] == 2,
            "positive direct/buffered bounded chains")
    rejected("unknown_gate", lambda: gate(shell("nclear", [
        "XBUF_UNKNOWN U0 (.I(rst_core), .Z(nclear));"])), "unknown")
    rejected("constant_source", lambda: gate(shell("1'b0", [])), "constant")
    rejected("cycle", lambda: gate(shell("nclear", [
        "CKND0BWP35P140 UI (.I(n1), .ZN(nclear));",
        "BUFFD1BWP35P140 UB1 (.I(n2), .Z(n1));",
        "BUFFD1BWP35P140 UB2 (.I(n1), .Z(n2));"])), "cycle")
    rejected("multiple_drivers", lambda: gate(shell("nclear", [
        "CKND0BWP35P140 UI0 (.I(rst_core), .ZN(nclear));",
        "CKND0BWP35P140 UI1 (.I(rst_core), .ZN(nclear));"])), "cardinality")
    rejected("multi_input_reconvergence", lambda: gate(shell("nclear", [
        "AN2D1BWP35P140 UA (.A1(rst_core), .A2(rst_core), .Z(nclear));"])),
        "reconvergent")
    rejected("wrong_polarity_direct_root", lambda: gate(shell("rst_core", [])),
             "exactly one inverter")
    rejected("zero_inverter", lambda: gate(shell("nclear", [
        "BUFFD1BWP35P140 UB (.I(rst_core), .Z(nclear));"])), "exactly one inverter")
    rejected("two_inverters", lambda: gate(shell("nclear", [
        "CKND0BWP35P140 UI0 (.I(rst_core), .ZN(n1));",
        "CKND0BWP35P140 UI1 (.I(n1), .ZN(nclear));"])), "exactly one inverter")
    rejected("wrong_foundry_pin_schema", lambda: gate(shell("nclear", [
        "BUFFD1BWP35P140 UB (.I(rst_core), .Z(nclear), .X(other_root));"])),
        "pin/output schema")
    rejected("wrong_root", lambda: gate(shell("nclear", [
        "CKND0BWP35P140 UI (.I(other_root), .ZN(nclear));"])), "cardinality")
    depth_cells = [f"BUFFD1BWP35P140 UB{i} (.I(n{i}), .Z(n{i+1}));"
                   for i in range(9)]
    rejected("depth_exceeded", lambda: gate(shell("n9", depth_cells)), "depth")
    rejected("active_set_not_inactive", lambda: gate(shell("nclear", [
        "CKND0BWP35P140 UI (.I(rst_core), .ZN(nclear));"], "rst_core")),
        "inactive high")
    rejected("duplicate_instance", lambda: gate(shell("nclear", [
        "BUFFD1BWP35P140 U0 (.I(rst_core), .Z(n1));",
        "CKND0BWP35P140 U0 (.I(n1), .ZN(nclear));"])), "duplicate")


def main() -> None:
    verify_frozen()
    frozen_before = {
        "source": sha(SOURCE), "contract": sha(CONTRACT), "netlist": sha(NETLIST),
        "subject": sha(SUBJECT), "failure_tree": tree_signature(FAILURE),
        "docs359": sha(DOCS359),
    }
    module = load_subject()
    static = static_checks(); foundry = independent_foundry_check()
    bounded_mutations(module)
    oracle = module.source_small_oracle()
    frozen = module.frozen_quarantine_static_oracle()
    require(oracle["mapped_vcs_dc_launch_retry"] is False and
            oracle["diagnostic_area_setup_promoted"] is False and
            frozen["shadow_register_bits"] == 337 and
            frozen["active_low_clear_nets"] == 12 and
            frozen["direct_inverter_registers"] == 75 and
            frozen["buffered_then_inverter_registers"] == 262 and
            frozen["maximum_chain_cells"] == 2,
            "frozen static oracle/claim boundary")
    verify_frozen()
    frozen_after = {
        "source": sha(SOURCE), "contract": sha(CONTRACT), "netlist": sha(NETLIST),
        "subject": sha(SUBJECT), "failure_tree": tree_signature(FAILURE),
        "docs359": sha(DOCS359),
    }
    require(frozen_before == frozen_after, "source/subject/failure identity changed")
    result = {
        "schema": "m1141r6_additive_structural_checker_author_mechanical_r1_v1",
        "status": "PASS_M1141R6_ADDITIVE_STRUCTURAL_CHECKER_AUTHOR__BOUNDED_STATIC_ONLY_NO_EDA",
        "checks": checks,
        "attacks_rejected": attacks,
        "static": static,
        "foundry_semantics_independent": foundry,
        "frozen_quarantine_oracle": {
            "netlist_sha256": EXPECTED["netlist"], "shadow_register_bits": 337,
            "active_low_clear_nets": 12, "direct_inverter_registers": 75,
            "buffered_then_inverter_registers": 262, "maximum_chain_cells": 2,
            "all_paths_single_driver_unary": True,
            "all_paths_exactly_one_inverter_to_rst_core": True,
        },
        "identity_before_after_equal": True,
        "execution": {"mapped_vcs": False, "vcs": False, "dc": False,
                      "launch": False, "retry": False},
        "diagnostic_area_setup_promoted": False,
        "authorization": {"different_author_hammer_only_next": True},
        "source_sha256": EXPECTED["source"],
        "contract_identity": [EXPECTED["contract"], EXPECTED["contract_side"],
                              EXPECTED["contract_outer"]],
        "m1140_outer_seal_file_sha256": EXPECTED["m1140_outer"],
        "docs359_sha256": EXPECTED["docs359"],
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
