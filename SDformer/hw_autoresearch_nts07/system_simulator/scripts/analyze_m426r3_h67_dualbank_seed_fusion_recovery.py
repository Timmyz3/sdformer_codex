#!/usr/bin/env python3
"""Decode-only recovery for M426's pre-phase raw-row call-site bug.

The frozen M426 main passes a raw fixed-width phase byte block plus the
frozen nibble LUT at the point where decoded words are required.  This wrapper
routes those two frozen arguments through the already-frozen ``decode_words``
and ``phase_population`` functions.  No replay, cycle, resource, threshold,
input, or output equation is changed.
"""

import importlib.util
from pathlib import Path


ORIGINAL = Path(__file__).resolve().with_name(
    "analyze_m426_h67_dualbank_seed_fusion.py")
SPEC = importlib.util.spec_from_file_location("m426_frozen_r1", str(ORIGINAL))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ORIGINAL_DECODE_WORDS = MODULE.decode_words
ORIGINAL_PHASE_POPULATION = MODULE.phase_population


def decode_then_phase_population(raw_block, nibble_lut):
    return ORIGINAL_PHASE_POPULATION(
        ORIGINAL_DECODE_WORDS(raw_block, nibble_lut))


MODULE.phase_population = decode_then_phase_population
raise SystemExit(MODULE.main())
