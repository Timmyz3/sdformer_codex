#!/usr/bin/env python3
"""Small standalone proof that Linux ru_maxrss does not fall with current RSS."""
from __future__ import print_function

import gc
import json
import mmap
import os
import resource
import time


ALLOCATION_BYTES = 64 * 1024 * 1024


def current_rss_kib():
    with open("/proc/self/status", "r") as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    raise RuntimeError("VmRSS unavailable")


def sample():
    return {"current_rss_kib": current_rss_kib(),
            "ru_maxrss_kib": int(resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss)}


def main():
    before = sample()
    value = mmap.mmap(-1, ALLOCATION_BYTES)
    page = os.sysconf("SC_PAGE_SIZE")
    for offset in range(0, ALLOCATION_BYTES, page):
        value[offset:offset + 1] = b"x"
    during = sample()
    value.close()
    del value
    gc.collect()
    time.sleep(0.05)
    after = sample()
    result = {
        "schema": "m1571_linux_ru_maxrss_semantics_probe_r1_v1",
        "allocation_bytes": ALLOCATION_BYTES,
        "before": before,
        "during": during,
        "after_close": after,
        "checks": {
            "current_rss_fell_after_close":
                after["current_rss_kib"] < during["current_rss_kib"],
            "ru_maxrss_retained_high_water_after_close":
                after["ru_maxrss_kib"] == during["ru_maxrss_kib"],
            "ru_maxrss_is_not_current_rss":
                after["ru_maxrss_kib"] > after["current_rss_kib"],
        },
        "claim_boundary": {"m1570_pilot_rerun": False,
                           "canonical_payload_opened": False,
                           "allocation_is_probe_only": True},
    }
    if not all(result["checks"].values()):
        raise RuntimeError("ru_maxrss semantics probe failed")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
