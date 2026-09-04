#!/usr/bin/env python3
"""Compatibility-only recovery for M426 r1's pre-phase call-site typo.

M426 r1 passed an already-decoded nibble table to ``phase_population`` even
though that function consumes only decoded words.  The r1 run therefore
failed before phase zero and exposed no candidate cycle result.  This wrapper
loads the exact frozen r1 analyzer and accepts/ignores only that redundant
second argument; every replay equation, input, gate, and output remains in the
frozen analyzer.
"""

import importlib.util
from pathlib import Path


ORIGINAL = Path(__file__).resolve().with_name(
    "analyze_m426_h67_dualbank_seed_fusion.py")
SPEC = importlib.util.spec_from_file_location("m426_frozen_r1", str(ORIGINAL))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ORIGINAL_PHASE_POPULATION = MODULE.phase_population


def phase_population_compat(words, _redundant_nibble):
    return ORIGINAL_PHASE_POPULATION(words)


MODULE.phase_population = phase_population_compat
raise SystemExit(MODULE.main())
