# M234 independent hammer review

Score: **84/100**. Severity: **P0=0, P1=6, P2=3**.

The checkpoint-bound integer DSE is numerically reproducible. Strict source
SHA checks pass; the independent model reloads all 220,800 coefficient pairs,
recomputes the mean/variance/invstd/alpha/offset relations, and obtains exact
agreement for the selected 64-entry LUT, segmented address, exponent
normalization, signed RNE, zero-rail counts, reported error metrics, and all
1,024 exported vector payloads. The selected maximum captured-interval,
coefficient-only affine bound is `0.0017281948510117218`.

This admission stops at integer coefficient DSE. Five dependent multiplies can
be serialized onto one scalar multiplier, but first-result latency 16 and II16
are targets without RTL/VCS/DC. Sum/sumsq/population division and the moment
finalizer are outside the module. The endpoint result is exact for an affine
coefficient delta on each captured interval, but does not include activation
quantization, runtime affine rounding/saturation, ATLIF thresholds/events, BN2
residual behavior, cross-sequence data, or valid825.

Two corrections are needed before RTL. First, the production vector selector
sorts a merged priority set and then truncates it. The resulting source-index
range ends at 172,724 of 220,799 and drops six high-index extrema that the
script explicitly tried to retain. Second, 16 LUT entries plus two Newton
steps independently reaches a smaller maximum bound (`0.0015724991042134207`)
with a 304-bit LUT versus 1,216 bits, at eight rather than five multiplies per
pair. Because both may fit a nominal 16-cycle scalar schedule, 64+1 is not yet
a proved Pareto point; matched RTL/DC/SAIF must choose it.

The next honest milestone is an exact integer ready/valid coefficient engine,
a corrected stratified/extrema-preserving vector set, and matched Synopsys
comparison of 16-entry/two-Newton versus 64-entry/one-Newton. Moment finalization
must remain a separate unresolved boundary.
