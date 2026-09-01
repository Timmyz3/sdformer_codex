# M1776 C2 mapped-energy failure diagnosis

## Root event

The Python wrapper repair worked: both the M1768 wrapper attempt and the underlying M1753 campaign attempt were consumed. M1753 completed one K1 compile and entered K1 case-0 simulation. At 27 ns, after one source packet cover but before any endpoint, result/commit, or done cover, the unchanged M1684 four-state monitor stopped simulation because the aggregate mapped fault vector contained X/Z. No SAIF was closed and no PTPX run occurred. The later checker traceback reports the absent SAIF and is a downstream consequence, not another Python-path failure.

The compile also reported three K1 half-adder instances with an omitted `CO` output. Their `S` outputs were connected. K8 statically contains seven instances with the same omitted-output pattern, so this warning is not K1-unique and does not by itself explain an unknown public fault signal. The existing log does not identify which member of `{protocol_error, numeric_overflow, stale_response_seen, endpoint_fault[7:0]}` was unknown. Treating the warning as the root cause would therefore overclaim the evidence.

## First-principles decision

The paper's fair energy comparison is K8 against equal-bandwidth K1x8. Single K1 is a diagnostic area/timing axis, not the fair throughput-matched baseline. M1661 already preserves that diagnostic: 124,546.967176 µm², setup met with minimum reported slack 0.0011 ns, hold not closed, and no energy claim.

M1777 should therefore remove K1 from the energy campaign while retaining both the K1 DC row and this failed M1753 receipt. This is not result cherry-picking: no K1 energy value was produced, and K1 was never the primary fair denominator. The successor must keep the full X/Z monitor unchanged. If K8 or K1x8 reproduces the fault, it must stop immediately and repair/localize the affected primary-axis fault/reset/connection cone. It must never suppress the assertion, ignore X, or initialize the design into a pass.

## Minimal successor

Use only K8 and K1x8, five frozen cases each, at the same 3 ns clock and public-port workload. Budget exactly two fresh compiles, ten simulations, ten SAIF files, and ten PTPX runs. All ten checked SAIF files must exist before any PTPX invocation. Partial axes and automatic retry remain forbidden. A different author must review the source as M1778, followed by an exact M1779 release.
