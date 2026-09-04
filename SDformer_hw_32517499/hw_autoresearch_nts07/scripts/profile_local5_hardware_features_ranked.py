#!/usr/bin/env python3
"""Ranked-checkpoint receipt adapter for the frozen Local5 profiler."""

from __future__ import annotations

import profile_local5_hardware_features as profiler
from local5_release_receipt import validate_release_receipt


profiler.validate_release_receipt = validate_release_receipt


if __name__ == "__main__":
    raise SystemExit(profiler.main())
