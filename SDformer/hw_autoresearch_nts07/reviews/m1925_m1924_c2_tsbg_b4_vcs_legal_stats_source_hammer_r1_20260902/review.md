# M1925 — M1924 TSBG VCS-legal statistics-process source hammer

## Verdict

**PASS, 99/100; P0/P1/P2 = 0/0/0.** This is a source-only review. It authorizes only additive authoring of a fresh M1926 runner source; it does not authorize a license query, attempt, VCS/simv, DC/PT, result, or paper claim.

## Exact object delta

The unified diff from frozen M1880 TB SHA `07f638...ed2d5` to M1924 TB SHA `df99e8...64f1` contains one hunk. The only semantic edit is:

```systemverilog
-    always_ff @(posedge clk_core) begin
+    always @(posedge clk_core) begin
```

The other three added lines are an explanatory comment. The event control and entire scoreboard/statistics body are unchanged. Adapter, DUT RTL, and SVA remain byte-identical at `cd2640...0156`, `8524f6...9a05`, and `e5519a...58c2`.

The new filelist replaces only the TB path among the four source objects. It also adds an RTL include search path; none of the four reviewed sources has an include directive, so this directive is semantically inert for this compile object set.

## M1922 failure closure

The sealed M1922 compile log contains ten `Error-[ICPD]` diagnostics before the default error limit. Each is caused by a statistic variable written by the line-497 `always_ff` process and also initialized or phase-cleared by the directed `initial` process. The failure receipt and consumed-attempt receipt both verify through their inner and outer SHA seals; M1922 is `FAILED_OR_INCOMPLETE_DO_NOT_CITE`, exit 255, retry false.

IEEE 1800 gives `always_ff` an exclusive procedural-writer restriction. Re-expressing this testbench-only statistics process as an ordinary posedge `always` removes that restriction while retaining nonblocking clocked updates. This statically closes the observed ICPD root cause. A fresh VCS compile is still required to establish tool execution; this review did not run it.

## Non-weakening result

- TB fatal checks remain 31/31 and the directed PASS token remains unique at 1/1.
- Arithmetic, duplicate-commit, work-conservation, LRU4, exact bridge corner, independent-bank reorder/stall, and local 1.15x checks are unchanged.
- Retired-identity replay, bogus stale response, two reset recoveries, and post-reset legal service are unchanged.
- The frozen SVA still has 24 assert properties and 11 cover properties.
- `docs/359` remains SHA256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Next boundary

Author a fresh M1926 runner source that exact-SHA pins the M1924 TB and filelist, binds the sealed M1922 failure and consumed-attempt evidence, and uses fresh namespaces. A different-author runner review and separate launch authority are required before any EDA. M1925 is not a launch release.
