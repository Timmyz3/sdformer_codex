#!/usr/bin/env python3
"""Pure storage oracle for the M785 global-vector decoder model.

This oracle deliberately does not implement the M722 line-buffer schedule.
It describes the distinct M777/M785 object: Acc24 output vectors resident in a
fixed 221184-byte partition and addressed by deterministic global-vector
stripes, with direct-key external backing when capacity is exceeded.
"""

import hashlib
import json
import math
from typing import Dict, Mapping, Sequence, Tuple


SCHEMA = "m785_decoder_global_vector_storage_oracle_v1"
PSUM_VECTOR_BYTES = 96 * 3
DEFAULT_PSUM_BYTES = 221184


class StorageOracleError(RuntimeError):
    """Invalid storage geometry or conservation failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StorageOracleError(message)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_geometry(
    geometry: Sequence[object],
) -> Tuple[str, int, int, int, int, int, int, int]:
    require(len(geometry) in (6, 8), "storage geometry must have 6 or 8 fields")
    if len(geometry) == 6:
        cin, cout, hin, win, hout, wout = (int(value) for value in geometry)
        name = "UNNAMED"
        blocks = math.ceil(cout / 96)
    else:
        name = str(geometry[0])
        cin, cout, hin, win, hout, wout, blocks = (
            int(value) for value in geometry[1:]
        )
    require(min(cin, cout, hin, win, hout, wout, blocks) > 0,
            "storage geometry must be positive")
    require(blocks == math.ceil(cout / 96),
            "output block count does not match 96 lanes")
    return name, cin, cout, hin, win, hout, wout, blocks


def plan(geometry: Sequence[object],
         psum_bytes: int = DEFAULT_PSUM_BYTES) -> Dict[str, object]:
    """Return a deterministic, independently recomputable storage plan."""
    name, cin, cout, hin, win, hout, wout, blocks = _normalize_geometry(
        geometry)
    psum_bytes = int(psum_bytes)
    require(psum_bytes > 0 and psum_bytes % PSUM_VECTOR_BYTES == 0,
            "psum partition must contain integral Acc24 vectors")
    capacity_vectors = psum_bytes // PSUM_VECTOR_BYTES
    total_vectors = hout * wout * blocks
    stripes = [
        [lo, min(total_vectors, lo + capacity_vectors)]
        for lo in range(0, total_vectors, capacity_vectors)
    ]
    require(stripes and stripes[0][0] == 0 and
            stripes[-1][1] == total_vectors,
            "global-vector stripes do not cover the output")
    require(all(hi > lo and hi - lo <= capacity_vectors
                for lo, hi in stripes),
            "global-vector stripe exceeds physical capacity")
    for left, right in zip(stripes, stripes[1:]):
        require(left[1] == right[0], "global-vector stripes have a gap")

    overflow_vectors = max(0, total_vectors - capacity_vectors)
    # Direct-key backing uses base + vector_key * 288.  Once overflow exists,
    # any global vector may become a dirty victim under the deterministic LRU,
    # so the reserved address span covers the complete logical population.
    backing_span = total_vectors * PSUM_VECTOR_BYTES if overflow_vectors else 0
    value: Dict[str, object] = {
        "schema": SCHEMA,
        "model": "GLOBAL_VECTOR_LRU_DIRTY_BACKING",
        "module": name,
        "geometry": {
            "cin": cin, "cout": cout, "hin": hin, "win": win,
            "hout": hout, "wout": wout, "output_blocks": blocks,
        },
        "accumulator": "Acc24",
        "lanes": 96,
        "vector_bytes": PSUM_VECTOR_BYTES,
        "psum_partition_bytes": psum_bytes,
        "resident_capacity_vectors": capacity_vectors,
        "total_vectors": total_vectors,
        "stripe_count": len(stripes),
        "stripes": stripes,
        "resident_payload_bytes_at_full_capacity": min(
            total_vectors, capacity_vectors) * PSUM_VECTOR_BYTES,
        "capacity_overflow_vectors": overflow_vectors,
        "offchip_backing_address_span_bytes": backing_span,
        "offchip_dynamic_read_write_bytes": "TRACE_DEPENDENT__COUNT_IN_LEDGER",
        "m722_line_buffer_storage_equivalent": False,
    }
    value["plan_sha256"] = canonical_sha256(value)
    return value


def validate_plan(value: Mapping[str, object],
                  geometry: Sequence[object],
                  psum_bytes: int = DEFAULT_PSUM_BYTES) -> Dict[str, object]:
    """Reject any injected plan field, including stripes and off-chip span."""
    expected = plan(geometry, psum_bytes)
    require(dict(value) == expected, "M785 independent storage oracle mismatch")
    return expected

