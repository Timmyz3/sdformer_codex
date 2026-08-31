# M1547 S2 CCBS retained decoder fast-kill

Status: **PASS_LOCAL_SCREEN__REQUEST_INCREMENTAL_FC_PATCH_CAPTURE_ONLY__NO_RTL_AEE_OR_PERFORMANCE**.

This CPU-only screen binds ep34 checkpoint `4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48` and 36 sealed M1521 decoder calls (three fixed samples per DSEC sequence, all four layers). Passing block configurations: `16x16, 32x16`. Decoder is only a local binary screen: no FC/patch coverage, AEE, cycles, speedup, traffic, energy or RTL is admitted. Epsilon and debt values are unitless local diagnostics, not an AEE budget.
