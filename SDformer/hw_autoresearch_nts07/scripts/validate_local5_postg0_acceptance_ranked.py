#!/usr/bin/env python3
"""Ranked-checkpoint receipt adapter for frozen Local5 post-G0 acceptance."""

from __future__ import annotations

import analyze_ds_flm_descriptor_manifest as analyzer
import validate_local5_postg0_acceptance as acceptance
from analyze_ds_flm_descriptor_manifest_ranked import (
    validate_ranked_identity_receipt,
)


analyzer.validate_release_receipt = validate_ranked_identity_receipt
acceptance.analyze = analyzer.analyze


if __name__ == "__main__":
    raise SystemExit(acceptance.main())
