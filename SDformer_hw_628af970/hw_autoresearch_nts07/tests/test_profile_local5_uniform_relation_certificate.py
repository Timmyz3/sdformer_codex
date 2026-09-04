from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "profile_local5_uniform_relation_certificate",
    ROOT / "scripts/profile_local5_uniform_relation_certificate.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_uniform_gate_codebook() -> None:
    assert [MODULE.expected_uniform_gate(degree) for degree in range(1, 6)] == [
        128,
        64,
        32,
        32,
        16,
    ]


def test_live_ring_capacity_is_plane_local() -> None:
    flags = [False] * MODULE.TOKENS
    # Four exceptions in three adjacent rows of plane 0.
    flags[0] = True
    flags[MODULE.WIDTH] = True
    flags[2 * MODULE.WIDTH] = True
    flags[2 * MODULE.WIDTH + 1] = True
    # A separate exception in plane 1 must not join the first live ring.
    flags[MODULE.PLANE_TOKENS] = True
    assert MODULE.max_ring_exceptions(flags) == 4


def test_sparse_storage_accounts_for_five_read_ports() -> None:
    model = MODULE.sparse_role_bits(8)
    assert model["baseline_gate_valid_bits"] == 5 * 45 * 10
    assert model["implicit_mode_bits"] == 5 * 45
    assert model["exception_cam_bits"] == 5 * 8 * (6 + 9)
    assert model["candidate_gate_valid_bits"] == 825
