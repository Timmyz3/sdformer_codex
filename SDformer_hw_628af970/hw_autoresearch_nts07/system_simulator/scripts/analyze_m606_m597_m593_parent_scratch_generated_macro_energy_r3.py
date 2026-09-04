#!/usr/libexec/platform-python3.6
"""M606 no-replace publication adapter for the frozen M597 r2 energy model.

The numerical/business model remains byte-identical in the sealed M597 module.
This adapter replaces only its caller-visible publication edge with an exact
lexists/lstat policy and renameat2(RENAME_NOREPLACE).
"""

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat


REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_REL = "hw_autoresearch_nts07/system_simulator/scripts/analyze_m597_m593_m528_parent_scratch_generated_macro_energy_r2.py"
UPSTREAM_SHA256 = "6896c8a406dc3274926e6c7d958136aca47b9df9afa3522d6c2539a142ea9cf9"
CONTRACT_REL = "hw_autoresearch_nts07/contracts/m597_m593_m528_parent_scratch_generated_macro_energy_source_contract_r2_20260828.json"
CONTRACT_SHA256 = "90399b6c932e28f6eac38f3408af0374b23beb369e1fd4e57e3b98d92d28b1bf"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plain_file(path):
    path = Path(path)
    require(os.path.lexists(str(path)), "missing path: " + str(path))
    mode = os.lstat(str(path)).st_mode
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode),
            "not a plain file: " + str(path))


def plain_parent(path):
    path = Path(path).resolve()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        require(os.path.lexists(str(current)), "missing parent: " + str(current))
        mode = os.lstat(str(current)).st_mode
        require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode),
                "nonplain parent: " + str(current))


def rename_noreplace(source, target):
    libc = ctypes.CDLL(None, use_errno=True)
    require(hasattr(libc, "renameat2"), "renameat2 unavailable")
    function = libc.renameat2
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                         ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    rc = function(-100, os.fsencode(str(source)), -100,
                  os.fsencode(str(target)), 1)
    if rc != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def load_upstream():
    upstream = REPO_ROOT / UPSTREAM_REL
    contract = REPO_ROOT / CONTRACT_REL
    plain_file(upstream)
    plain_file(contract)
    require(sha256(upstream) == UPSTREAM_SHA256, "M597 upstream SHA drift")
    require(sha256(contract) == CONTRACT_SHA256, "M597 contract SHA drift")
    spec = importlib.util.spec_from_file_location("m606_frozen_m597", str(upstream))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.CONTRACT_SHA256 == CONTRACT_SHA256,
            "M597 module/contract binding drift")
    return module, contract


def write_exclusive(path, payload):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def publish_noreplace(upstream, output, result):
    output = Path(output)
    require(output.is_absolute(), "output must be absolute")
    plain_parent(output.parent)
    require(not os.path.lexists(str(output)), "output coordinate exists")
    staging = output.parent / ("." + output.name + ".m606_staging_" + str(os.getpid()))
    require(not os.path.lexists(str(staging)), "adapter staging exists")
    os.mkdir(str(staging), 0o700)
    json_name = "m597_m593_m528_parent_scratch_generated_macro_energy_result_r2.json"
    csv_name = "m597_parent_scratch_energy_rows_r2.csv"
    complete_name = "RUN_COMPLETE.txt"
    write_exclusive(staging / json_name,
                    json.dumps(result, indent=2, sort_keys=True,
                               allow_nan=False) + "\n")
    import csv
    rows = result["rows"]
    with (staging / csv_name).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    write_exclusive(staging / complete_name,
                    "PASS_M597_R2_ANALYZER_OUTPUT_PENDING_INDEPENDENT_RESULT_HAMMER\n")
    names = [complete_name, csv_name, json_name]
    write_exclusive(staging / "SHA256SUMS", "".join(
        "%s  %s\n" % (sha256(staging / name), name) for name in names))
    write_exclusive(staging / "SHA256SUMS.seal.sha256",
                    "%s  SHA256SUMS\n" % sha256(staging / "SHA256SUMS"))
    require(not os.path.lexists(str(output)), "output appeared before publish")
    rename_noreplace(staging, output)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-contract", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    upstream, contract_path = load_upstream()
    require(args.source_contract.resolve() == contract_path.resolve(),
            "noncanonical contract")
    contract = upstream.validate_contract(args.source_contract)
    if args.self_test:
        require(args.output_dir is None, "self-test names output")
        upstream.self_test(contract)
        print("PASS_M606_NOREPLACE_ADAPTER_STATIC_SELF_TEST")
        return
    require(args.output_dir is not None, "production requires output")
    identity, parsed = upstream.verify_frozen_inputs(contract)
    result = upstream.build_result(contract, identity, parsed)
    publish_noreplace(upstream, args.output_dir, result)


if __name__ == "__main__":
    main()
