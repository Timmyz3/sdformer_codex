#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1142R6: independent, static hammer of the M1141R6 reset-chain checker."""
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
AUTHOR = HW / "reviews/m1141r6_c2_additive_structural_reset_chain_checker_author_receipt_r1_20260830"
NETLIST = HW / ("results/m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_20260830."
                "failed_or_incomplete.1172090.quarantine/dc/netlist/"
                "m1129r5_c2_k1_async_observation_shadow_wrapper_mapped.v")
FAILURE = NETLIST.parents[2]
SUBJECT = HW / "dc_handoff/scripts/m1133r6_c2_authority_schema_repair_engine_source_r1.py"
M1140_OUTER = HW / ("reviews/m1140r6_m1133r6_c2_failure_quarantine_hammer_r1_20260830/"
                    "SHA256SUMS.seal.sha256")
CELL_LIB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/"
                "TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/"
                "tcbn28hpcplusbwp35p140.v")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "source": "86ccd46fdaffcad77444ca105bde1593394dd7643febba1f6a45680bf515965e",
    "contract": "60577bc578ca1c9aaa8de5b446f712fc416738a2979e4dc4b86e7ba9b1bf5b37",
    "contract_side": "188ae927bc8ef40085e0e550a29d4892d94647717624487a2d9a4fdb34fa9196",
    "contract_outer": "3f4c7cc217fdac94ca061883314bb1b2160a0ee3436ff27f3321a3f70e7a4479",
    "author_review": "e1cf31a3e6aef9a57582ec414788b7ef22992bcb1a42902ac1aaa5df7743e7c1",
    "author_manifest": "b9d42229914cf11f62f0e76f91378f75b111eda87b055b1d48d1222490923aa6",
    "author_outer": "f47aa96a21b736607c55341569555ce59ad906fd9746125b14035557a3346e97",
    "netlist": "362e855cd3b4391d31dc7a08e5388d9545f289c81d291c512d25294a8539cbc4",
    "subject": "1f8a190d7d1c8b7804e7302c8b6a38c30a49df466b6394a82e8f0cf4cec2ee40",
    "cell_lib": "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
    "m1140_outer": "f36895c57ea46eda4e492a201e6ae7b0dc0b979736fa50dead3a91e240073fae",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
checks = 0
attacks: dict[str, str] = {}


class HammerFailure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise HammerFailure(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha(path) == expected,
            "identity drift: " + str(path))


def strict_json(path: Path) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          HammerFailure("nonfinite JSON: " + token)))


def exact_manifest(directory: Path, manifest_sha: str, outer_sha: str) -> None:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular(manifest, manifest_sha); regular(outer, outer_sha)
    require(outer.read_text(encoding="utf-8").split() == [manifest_sha, "SHA256SUMS"],
            "author outer seal content")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        rel = Path(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and
                name not in expected and name == rel.as_posix() and not rel.is_absolute() and
                ".." not in rel.parts, "unsafe author manifest")
        expected[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        rel = member.relative_to(directory).as_posix()
        if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "author receipt symlink")
        if stat.S_ISREG(mode):
            actual.add(rel)
        else:
            require(stat.S_ISDIR(mode), "author receipt special member")
    require(actual == set(expected), "author receipt exact member census")
    for name, digest in expected.items():
        regular(directory / name, digest)


def tree_signature(directory: Path) -> str:
    digest = hashlib.sha256()
    for member in sorted(directory.rglob("*")):
        rel = member.relative_to(directory).as_posix(); mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "failure tree symlink")
        if stat.S_ISREG(mode):
            digest.update(b"F\0" + rel.encode() + b"\0" + bytes.fromhex(sha(member)))
        else:
            require(stat.S_ISDIR(mode), "failure tree special member")
            digest.update(b"D\0" + rel.encode() + b"\0")
    return digest.hexdigest()


def verify_inputs() -> dict[str, str]:
    pairs = ((SOURCE, "source"), (CONTRACT, "contract"),
             (Path(str(CONTRACT) + ".sha256"), "contract_side"),
             (Path(str(CONTRACT) + ".sha256.seal.sha256"), "contract_outer"),
             (AUTHOR / "review.json", "author_review"), (NETLIST, "netlist"),
             (SUBJECT, "subject"), (CELL_LIB, "cell_lib"),
             (M1140_OUTER, "m1140_outer"), (DOCS359, "docs359"))
    for path, key in pairs:
        regular(path, EXPECTED[key])
    exact_manifest(AUTHOR, EXPECTED["author_manifest"], EXPECTED["author_outer"])
    require(Path(str(CONTRACT) + ".sha256").read_text(encoding="utf-8").split() ==
            [EXPECTED["contract"], CONTRACT.name], "contract inner seal content")
    require(Path(str(CONTRACT) + ".sha256.seal.sha256").read_text(encoding="utf-8").split() ==
            [EXPECTED["contract_side"], CONTRACT.name + ".sha256"],
            "contract outer seal content")
    author = strict_json(AUTHOR / "review.json")
    require(author["status"] ==
            "PASS_M1141R6_ADDITIVE_STRUCTURAL_CHECKER_AUTHOR__BOUNDED_STATIC_ONLY_NO_EDA" and
            author["authorization"]["different_author_hammer_only_next"] is True and
            all(author["authorization"][key] is False
                for key in ("mapped_vcs", "vcs", "dc", "launch", "retry")),
            "author authorization drift")
    return {"source": sha(SOURCE), "subject": sha(SUBJECT),
            "netlist": sha(NETLIST), "failure_tree": tree_signature(FAILURE),
            "docs359": sha(DOCS359), "author_outer": sha(AUTHOR / "SHA256SUMS.seal.sha256")}


def foundry_truth() -> dict[str, dict[str, str]]:
    library = CELL_LIB.read_text(encoding="utf-8", errors="strict")
    answer = {}
    for cell, inp, out, primitive in (("BUFFD1BWP35P140", "I", "Z", "buf"),
                                       ("CKND0BWP35P140", "I", "ZN", "not")):
        bodies = re.findall(rf"(?ms)^module\s+{cell}\s*\(\s*{inp}\s*,\s*{out}\s*\)\s*;(.*?)^endmodule",
                            library)
        require(len(bodies) == 1, "foundry module cardinality: " + cell)
        logic = re.findall(r"\b(buf|not|and|nand|or|nor|xor|xnor)\s*\(\s*([^;]+?)\s*\)\s*;",
                           bodies[0])
        require(logic == [(primitive, f"{out}, {inp}")], "foundry truth drift: " + cell)
        answer[cell] = {"input": inp, "output": out, "primitive": primitive}
    return answer


def parse_netlist(text: str) -> list[tuple[str, str, dict[str, str]]]:
    answer = []; names = set()
    for match in re.finditer(r"(?ms)^\s*(\w+)\s+(\\?[^\s(]+)\s*\((.*?)\)\s*;", text):
        pins = {}
        for pin, net in re.findall(r"\.(\w+)\s*\(\s*([^()\s,]+)\s*\)", match.group(3)):
            require(pin not in pins, "independent duplicate pin")
            pins[pin] = net
        if not pins:
            continue
        require(match.group(2) not in names, "independent duplicate instance")
        names.add(match.group(2)); answer.append((match.group(1), match.group(2), pins))
    return answer


def independent_census(text: str) -> dict[str, Any]:
    instances = parse_netlist(text)
    drivers: dict[str, list[tuple[str, str, dict[str, str], str]]] = {}
    for cell, name, pins in instances:
        for out in ("Z", "ZN", "Q", "QN"):
            if out in pins:
                drivers.setdefault(pins[out], []).append((cell, name, pins, out))
    shadows = [(cell, name, pins) for cell, name, pins in instances
               if re.search(r"shadow_\w+_q_reg", name)]
    require(len(shadows) == 337, "independent shadow census")
    paths: dict[str, tuple[tuple[str, str, str, str], ...]] = {}
    fanout: dict[str, int] = {}
    for _, name, pins in shadows:
        clear = [pin for pin in ("CDN", "CN") if pin in pins]
        require(len(clear) == 1, "independent clear-pin cardinality: " + name)
        for set_pin in ("SDN", "SN"):
            if set_pin in pins:
                require(pins[set_pin] in {"1", "1'b1", "1'h1"},
                        "independent active async set: " + name)
        origin = pins[clear[0]]; net = origin; seen_nets = set(); seen_cells = set(); chain = []
        inversions = 0
        while net != "rst_core":
            require(not re.fullmatch(r"(?:[01]|\d+'[bBoOdDhH][0-9a-fA-FxXzZ_]+)", net),
                    "independent constant source")
            require(len(chain) < 8 and net not in seen_nets,
                    "independent cycle/depth failure")
            source = drivers.get(net, [])
            require(len(source) == 1, "independent single-driver failure: " + net)
            cell, instance, cell_pins, out = source[0]
            require(instance not in seen_cells, "independent repeated instance")
            if cell == "BUFFD1BWP35P140":
                require(set(cell_pins) == {"I", "Z"} and out == "Z", "independent buffer schema")
                inp = "I"
            elif cell == "CKND0BWP35P140":
                require(set(cell_pins) == {"I", "ZN"} and out == "ZN", "independent inverter schema")
                inp = "I"; inversions += 1
            else:
                raise HammerFailure("independent unknown/reconvergent cell: " + cell)
            upstream = cell_pins[inp]
            chain.append((cell, instance, upstream, net)); seen_nets.add(net); seen_cells.add(instance)
            net = upstream
        require(inversions == 1 and chain, "independent inverter/root invariant")
        frozen_chain = tuple(chain)
        require(origin not in paths or paths[origin] == frozen_chain,
                "independent shared path instability")
        paths[origin] = frozen_chain; fanout[origin] = fanout.get(origin, 0) + 1
    direct = sum(fanout[net] for net, path in paths.items() if len(path) == 1)
    buffered = sum(fanout[net] for net, path in paths.items() if len(path) == 2)
    require(len(paths) == 12 and direct == 75 and buffered == 262 and
            direct + buffered == 337 and max(map(len, paths.values())) == 2,
            "independent frozen oracle mismatch")
    return {"shadow_register_bits": 337, "active_low_clear_nets": len(paths),
            "direct_inverter_registers": direct,
            "buffered_then_inverter_registers": buffered,
            "maximum_chain_cells": max(map(len, paths.values())),
            "reset_net_register_counts": dict(sorted(fanout.items()))}


def load_source():
    spec = importlib.util.spec_from_file_location("m1142_hammer_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "source module spec")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def shell(clear: str, cells: list[str], set_net: str | None = None) -> str:
    set_pin = "" if set_net is None else f", .SDN({set_net})"
    return "\n".join(["module mutation(input rst_core,input other_root,input d,input clk,output q);",
        f"DFCNQD1BWP35P140 shadow_mutation_q_reg (.D(d),.CP(clk),.CDN({clear}){set_pin},.Q(q));",
        *cells, "endmodule"])


def rejected(label: str, action: Callable[[], Any]) -> None:
    try:
        action()
    except Exception as error:
        attacks[label] = type(error).__name__ + ": " + str(error)
        return
    raise HammerFailure("attack accepted: " + label)


def attacks_check(module) -> None:
    gate = lambda text: module.structural_reset_chain_gate_text(text, 1)
    require(gate(shell("n", ["CKND0BWP35P140 I (.I(rst_core),.ZN(n));"]))["maximum_chain_cells"] == 1,
            "positive direct chain")
    require(gate(shell("n", ["BUFFD1BWP35P140 B (.I(rst_core),.Z(b));",
                             "CKND0BWP35P140 I (.I(b),.ZN(n));"]))["maximum_chain_cells"] == 2,
            "positive buffered chain")
    vectors = {
        "unknown_gate": shell("n", ["XBUF_UNKNOWN X (.I(rst_core),.Z(n));"]),
        "constant_source": shell("1'b0", []),
        "cycle": shell("n", ["CKND0BWP35P140 I (.I(a),.ZN(n));", "BUFFD1BWP35P140 A (.I(b),.Z(a));", "BUFFD1BWP35P140 B (.I(a),.Z(b));"]),
        "multi_driver": shell("n", ["CKND0BWP35P140 I0 (.I(rst_core),.ZN(n));", "CKND0BWP35P140 I1 (.I(rst_core),.ZN(n));"]),
        "reconvergence": shell("n", ["AN2D1BWP35P140 A (.A1(rst_core),.A2(other_root),.Z(n));"]),
        "wrong_polarity": shell("rst_core", []),
        "zero_inverter": shell("n", ["BUFFD1BWP35P140 B (.I(rst_core),.Z(n));"]),
        "two_inverters": shell("n", ["CKND0BWP35P140 I0 (.I(rst_core),.ZN(a));", "CKND0BWP35P140 I1 (.I(a),.ZN(n));"]),
        "wrong_pins": shell("n", ["BUFFD1BWP35P140 B (.I(rst_core),.Z(n),.X(other_root));"]),
        "wrong_root": shell("n", ["CKND0BWP35P140 I (.I(other_root),.ZN(n));"]),
        "depth": shell("n9", [f"BUFFD1BWP35P140 B{i} (.I(n{i}),.Z(n{i+1}));" for i in range(9)]),
        "active_set": shell("n", ["CKND0BWP35P140 I (.I(rst_core),.ZN(n));"], "rst_core"),
        "duplicate_instance": shell("n", ["BUFFD1BWP35P140 X (.I(rst_core),.Z(a));", "CKND0BWP35P140 X (.I(a),.ZN(n));"]),
    }
    for label, vector in vectors.items():
        rejected(label, lambda vector=vector: gate(vector))
    require(set(attacks) == set(vectors), "attack class census")


def static_boundary() -> None:
    text = SOURCE.read_text(encoding="utf-8"); tree = ast.parse(text)
    imports = {alias.name for node in ast.walk(tree)
               if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    require(not ({"subprocess", "os", "socket"} & imports), "launch-capable import")
    require(not any(token in text for token in ("dc_shell", "compile_ultra", "simv", "subprocess")),
            "EDA token in source")


def main() -> None:
    before = verify_inputs(); semantics = foundry_truth(); static_boundary()
    independent = independent_census(NETLIST.read_text(encoding="utf-8", errors="strict"))
    module = load_source(); attacks_check(module)
    subject = module.frozen_quarantine_static_oracle()
    for key in ("shadow_register_bits", "active_low_clear_nets", "direct_inverter_registers",
                "buffered_then_inverter_registers", "maximum_chain_cells",
                "reset_net_register_counts"):
        require(subject[key] == independent[key], "independent/subject mismatch: " + key)
    after = verify_inputs(); require(before == after, "frozen identities changed")
    result = {
        "schema": "m1142r6_m1141r6_structural_reset_chain_checker_hammer_r1_v1",
        "status": "PASS_M1142R6_INDEPENDENT_STRUCTURAL_RESET_CHAIN_HAMMER__AUTHOR_ADDITIVE_FROZEN_NETLIST_MAPPED_VCS_SUCCESSOR_SOURCE_ONLY",
        "checks": checks, "mutation_classes_rejected": len(attacks),
        "attacks_rejected": attacks, "foundry_truth_independent": semantics,
        "independent_frozen_quarantine_census": independent,
        "identity_before_after_equal": True,
        "authorization": {
            "additive_frozen_netlist_mapped_vcs_successor_source_authoring": True,
            "mapped_vcs_execution": False, "vcs_execution": False, "dc_retry": False,
            "eda_launch": False, "modify_subject_or_failure": False,
        },
        "claim_boundary": {"static_checker_hammer_only": True, "mapped_functionality": False,
                           "area_timing_power_energy": False, "cycles_speedup": False,
                           "paper_citable": False, "paper_ppa_ready": False},
        "source_sha256": EXPECTED["source"], "author_outer_seal_file_sha256": EXPECTED["author_outer"],
        "frozen_netlist_sha256": EXPECTED["netlist"], "subject_sha256": EXPECTED["subject"],
        "m1140_outer_seal_file_sha256": EXPECTED["m1140_outer"],
        "docs359_sha256": EXPECTED["docs359"],
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
