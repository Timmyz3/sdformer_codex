#!/usr/bin/env python3
"""Bind the frozen post-score reporter to the ranked-checkpoint runner."""

from __future__ import annotations

from pathlib import Path

import report_local5_active_projection_postg0_rtl as reporter


reporter.RUNNER_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "sim_new_arch/run_local5_active_projection_postg0_checks_ranked.sh"
)


if __name__ == "__main__":
    raise SystemExit(reporter.main())
