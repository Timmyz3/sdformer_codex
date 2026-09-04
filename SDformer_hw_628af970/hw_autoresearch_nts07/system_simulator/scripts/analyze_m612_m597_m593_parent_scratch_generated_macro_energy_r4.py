#!/usr/libexec/platform-python3.6
"""M612 path-hardened publication adapter over the exact M606/M597 model."""

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import stat


REPO = Path(os.path.abspath(__file__)).parents[3]
CORE_REL = "hw_autoresearch_nts07/system_simulator/scripts/analyze_m606_m597_m593_parent_scratch_generated_macro_energy_r3.py"
CORE_SHA = "69d5c2c521b84aee589b28531574d95ec621dfdeeaf35d517cc0bb386e87782d"
CONTRACT_REL = "hw_autoresearch_nts07/contracts/m597_m593_m528_parent_scratch_generated_macro_energy_source_contract_r2_20260828.json"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lexical_absolute(path):
    return Path(os.path.abspath(os.fspath(path)))


def plain_chain_before_resolve(path, directory=False):
    """Reject every lexical symlink component before any realpath/resolve call."""
    lexical = lexical_absolute(path)
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current = current / part
        require(os.path.lexists(str(current)), "missing lexical path: " + str(current))
        mode = os.lstat(str(current)).st_mode
        require(not stat.S_ISLNK(mode), "symlink lexical path: " + str(current))
        final = current == lexical
        require(stat.S_ISDIR(mode) if (not final or directory) else stat.S_ISREG(mode),
                "wrong lexical path type: " + str(current))
    require(Path(os.path.realpath(str(lexical))) == lexical,
            "lexical/real path drift after lstat walk")
    return lexical


core_path = REPO / CORE_REL
plain_chain_before_resolve(core_path)
require(sha(core_path) == CORE_SHA, "M606 adapter core SHA drift")
spec = importlib.util.spec_from_file_location("m612_exact_m606_adapter", str(core_path))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def patched_plain_parent(path):
    plain_chain_before_resolve(path, directory=True)


core.plain_parent = patched_plain_parent


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-contract", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    upstream, contract_path = core.load_upstream()
    supplied_contract = plain_chain_before_resolve(args.source_contract)
    require(supplied_contract == contract_path, "noncanonical contract")
    contract = upstream.validate_contract(supplied_contract)
    if args.self_test:
        require(args.output_dir is None, "self-test names output")
        upstream.self_test(contract)
        print("PASS_M612_PATH_HARDENED_NOREPLACE_ADAPTER_STATIC_SELF_TEST")
        return
    require(args.output_dir is not None, "production requires output")
    output = lexical_absolute(args.output_dir)
    plain_chain_before_resolve(output.parent, directory=True)
    require(not os.path.lexists(str(output)), "output coordinate exists")
    identity, parsed = upstream.verify_frozen_inputs(contract)
    result = upstream.build_result(contract, identity, parsed)
    core.publish_noreplace(upstream, output, result)


if __name__ == "__main__":
    main()
