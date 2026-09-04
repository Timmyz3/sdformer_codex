#!/usr/bin/env python3
"""M672 r3 path-identity repair for the exact decoder polyphase mapper.

M670-r2 proved the K3/S2/P1/OP1 phase/tap/K arithmetic, but an independent
review found that its public helpers validated a file below ``trusted_root``
and then passed the caller's original path to NumPy.  A relative path could
therefore validate one file and consume another file in the current working
directory.  R2 also checked only the final trusted-root directory rather than
every ancestor component.

This module leaves the frozen r2 implementation byte-for-byte unchanged.  It
validates the complete trusted-root chain, obtains the exact regular-file path
accepted by r2, and passes that absolute path to every data-consuming r2
helper.  The polyphase arithmetic is deliberately not copied or changed.
"""

from __future__ import print_function

import argparse
import json
import os
import stat
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import map_m670_decoder_convtranspose_polyphase_workload_r2 as R2  # noqa: E402


M514_SLOT_ORDER = R2.M514_SLOT_ORDER
M514_PHASE_ORDER = R2.M514_PHASE_ORDER
M514_PHASE_TAPS = R2.M514_PHASE_TAPS
EXPECTED_BIT_ORDER = R2.EXPECTED_BIT_ORDER
EXPECTED_FLAT_ORDER = R2.EXPECTED_FLAT_ORDER
EXPECTED_K_ORDER = R2.EXPECTED_K_ORDER
EXPECTED_SPEC = R2.EXPECTED_SPEC

require = R2.require
strict_integer = R2.strict_integer
checked_positive_product = R2.checked_positive_product
sha256 = R2.sha256
strict_json = R2.strict_json
safe_member = R2.safe_member
validate_convtranspose_spec = R2.validate_convtranspose_spec
validate_input_shape = R2.validate_input_shape
expected_packed_bytes = R2.expected_packed_bytes
phase_bank = R2.phase_bank
build_phase_plan = R2.build_phase_plan
phase_weight_matrix = R2.phase_weight_matrix


def trusted_root_all_components(path):
    """Return an absolute root only if every lexical component is a real dir."""
    require(isinstance(path, (str, os.PathLike)),
            "trusted root must be string-like")
    raw = os.fspath(path)
    require(isinstance(raw, str) and raw.startswith("/") and raw != "/",
            "trusted root must be a non-root absolute path")
    require("//" not in raw,
            "trusted root must not contain empty lexical components")
    lexical = raw.split("/")[1:]
    require(lexical and all(part not in ("", ".", "..")
                            for part in lexical),
            "trusted root contains a forbidden lexical component")

    current = Path("/")
    root_stat = os.lstat(str(current))
    require(stat.S_ISDIR(root_stat.st_mode) and
            not stat.S_ISLNK(root_stat.st_mode),
            "filesystem anchor must be a real directory")
    for part in lexical:
        current = current / part
        try:
            observed = os.lstat(str(current))
        except OSError as error:
            raise RuntimeError("trusted-root component lstat failed: " +
                               str(error))
        require(stat.S_ISDIR(observed.st_mode) and
                not stat.S_ISLNK(observed.st_mode),
                "trusted-root chain contains a symlink or non-directory")
    resolved = current.resolve(strict=True)
    require(resolved == current,
            "trusted-root lexical and resolved identities differ")
    return current


def _validated_absolute_bitpack(path, shape,
                                bit_order=EXPECTED_BIT_ORDER,
                                flat_order=EXPECTED_FLAT_ORDER,
                                k_order=EXPECTED_K_ORDER,
                                trusted_root=None):
    root = trusted_root_all_components(trusted_root)
    identity = R2.validate_bitpack(
        path, shape, bit_order=bit_order, flat_order=flat_order,
        k_order=k_order, trusted_root=root)
    canonical = Path(identity["path"])
    require(canonical.is_absolute(),
            "validated bitpack path must be absolute")
    # Re-run the component/leaf check immediately before handing the exact
    # accepted name to NumPy.  This does not claim to solve hostile concurrent
    # filesystem mutation; the frozen package is immutable during evaluation.
    accepted = R2.trusted_regular_file(root, canonical, "bitpack consume")
    require(accepted == canonical,
            "validated and consumed bitpack identities differ")
    return identity, canonical, root


def validate_bitpack(path, shape, bit_order=EXPECTED_BIT_ORDER,
                     flat_order=EXPECTED_FLAT_ORDER,
                     k_order=EXPECTED_K_ORDER, trusted_root=None):
    identity, _canonical, _root = _validated_absolute_bitpack(
        path, shape, bit_order, flat_order, k_order, trusted_root)
    return identity


def trusted_regular_file(trusted_root, path, label):
    root = trusted_root_all_components(trusted_root)
    return R2.trusted_regular_file(root, path, label)


def iter_polyphase_tiles(bitpack_path, shape, tile_m=256,
                         phases=M514_PHASE_ORDER,
                         bit_order=EXPECTED_BIT_ORDER,
                         flat_order=EXPECTED_FLAT_ORDER,
                         k_order=EXPECTED_K_ORDER,
                         kernel_size=(3, 3), stride=(2, 2),
                         padding=(1, 1), output_padding=(1, 1),
                         dilation=(1, 1), groups=1, trusted_root=None):
    _identity, canonical, root = _validated_absolute_bitpack(
        bitpack_path, shape, bit_order, flat_order, k_order, trusted_root)
    for tile in R2.iter_polyphase_tiles(
            canonical, shape, tile_m=tile_m, phases=phases,
            bit_order=bit_order, flat_order=flat_order, k_order=k_order,
            kernel_size=kernel_size, stride=stride, padding=padding,
            output_padding=output_padding, dilation=dilation, groups=groups,
            trusted_root=root):
        yield tile


def materialize_polyphase(bitpack_path, shape, tile_m=256, **kwargs):
    trusted_root = kwargs.pop("trusted_root", None)
    bit_order = kwargs.get("bit_order", EXPECTED_BIT_ORDER)
    flat_order = kwargs.get("flat_order", EXPECTED_FLAT_ORDER)
    k_order = kwargs.get("k_order", EXPECTED_K_ORDER)
    _identity, canonical, root = _validated_absolute_bitpack(
        bitpack_path, shape, bit_order, flat_order, k_order, trusted_root)
    return R2.materialize_polyphase(
        canonical, shape, tile_m=tile_m, trusted_root=root, **kwargs)


def reconstruct_convtranspose(bitpack_path, shape, weight, tile_m=256,
                              **kwargs):
    trusted_root = kwargs.pop("trusted_root", None)
    bit_order = kwargs.get("bit_order", EXPECTED_BIT_ORDER)
    flat_order = kwargs.get("flat_order", EXPECTED_FLAT_ORDER)
    k_order = kwargs.get("k_order", EXPECTED_K_ORDER)
    _identity, canonical, root = _validated_absolute_bitpack(
        bitpack_path, shape, bit_order, flat_order, k_order, trusted_root)
    return R2.reconstruct_convtranspose(
        canonical, shape, weight, tile_m=tile_m, trusted_root=root, **kwargs)


def workload_accounting(bitpack_path, shape, output_channels=1,
                        tile_m=256, trusted_root=None):
    _identity, canonical, root = _validated_absolute_bitpack(
        bitpack_path, shape, trusted_root=trusted_root)
    return R2.workload_accounting(
        canonical, shape, output_channels=output_channels,
        tile_m=tile_m, trusted_root=root)


def m660_bitpack_records(manifest_path, trusted_root):
    root = trusted_root_all_components(trusted_root)
    records = R2.m660_bitpack_records(manifest_path, root)
    for record in records:
        path = Path(record["path"])
        accepted = R2.trusted_regular_file(root, path, "M660 payload consume")
        require(path.is_absolute() and accepted == path,
                "M660 validated/consumed payload identity drift")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitpack", required=True)
    parser.add_argument("--shape", required=True,
                        help="comma-separated T,1,C,H,W")
    parser.add_argument("--output-channels", type=int, required=True)
    parser.add_argument("--tile-m", type=int, default=256)
    parser.add_argument("--trusted-root", required=True)
    args = parser.parse_args(argv)
    shape = tuple(int(item) for item in args.shape.split(","))
    result = {
        "schema": "m672_decoder_polyphase_workload_summary_r3_v1",
        "status": "PASS_EXACT_CPU_MAPPING_INPUT_ONLY__PATH_IDENTITY_R3",
        "input": validate_bitpack(
            args.bitpack, shape, trusted_root=args.trusted_root),
        "mapping": {
            "phase_order": list(M514_PHASE_ORDER),
            "phase_taps": {str(bank): [list(tap) for tap in
                                        M514_PHASE_TAPS[bank]]
                           for bank in M514_PHASE_ORDER},
            "k_order": EXPECTED_K_ORDER,
            "convtranspose_spec": {key: (list(value)
                if isinstance(value, tuple) else value)
                for key, value in EXPECTED_SPEC.items()},
        },
        "accounting": workload_accounting(
            args.bitpack, shape, args.output_channels, args.tile_m,
            trusted_root=args.trusted_root),
        "repair": {
            "validated_absolute_path_is_consumed": True,
            "trusted_root_all_components_lstat": True,
            "m670_r2_arithmetic_reused_without_modification": True,
        },
        "claim_boundary": {
            "exact_input_mapping": True, "cycles": False,
            "speedup": False, "rtl": False, "eda": False,
            "paper_headline": False,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
