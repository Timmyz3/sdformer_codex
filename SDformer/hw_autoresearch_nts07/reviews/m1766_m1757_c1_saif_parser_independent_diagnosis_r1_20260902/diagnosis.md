# M1766 independent M1757 SAIF diagnosis

## Verdict

M1757 produced a real mapped VCS functional PASS and a syntactically valid
11,100,253-byte VCS SAIF, but the SAIF is **not admissible for PTPX**.  The old
checker first fails for a parser reason: it treats the two legal VCS `/** */`
header comments as atoms and therefore never reaches `(SAIFILE ...)`.  A strict
lexer which skips only C block comments outside quoted strings consumes all
2,706,931 tokens, finds one `SAIFILE` root, one top/DUT hierarchy and the exact
756 ns window.

That grammar fix exposes a second, substantive failure.  Of 117,690 `TX`
forms, 37,550 are nonzero.  All 1,152 parent-scratch `read_data` bits carry
`TX=500` ns, and unknown activity reaches 36,398 directly listed DUT nets.
The scratch address, enable, write-enable and clock signals are fully known;
the public request/commit controls and all checked public payload vectors are
also fully known.  Therefore the public scoreboard PASS is genuine, but it
does not make the internal switching activity safe for power annotation.

## Why the window is contaminated

The current testbench stops and enables activity immediately before accepting
prep row 0.  The foundry SRAM model starts with unknown Q.  During this first
task the public protocol eventually writes and reads legal parent rows, so the
architectural outputs pass, but the SAIF still includes the initial unknown-Q
interval and its combinational propagation.

## Minimal valid successor

Parser-only repair is useful only to classify this file as `TX_NONZERO_REJECT`;
it cannot authorize a PTPX run.  The preferred successor is a fresh public-port
campaign:

1. After reset, execute the same complete 64-row masks as a warmup task at
   epoch 5944 with activity disabled.
2. Wait for the matching `task_done`.  The RTL supports a later epoch, frees
   the execution bank on drain, clears per-task counters on the next execution,
   and deliberately retains the global parent SRAM contents.  Every accepted
   1152-bit scratch write fans out to all nine 128x128 SRAM slices.
3. Execute the same workload at epoch 5945 and enable SAIF only for this second
   task.  Reusing the same masks guarantees every measured parent address has
   been exercised through the public datapath during warmup.
4. Require public scoreboard PASS, `TX=0` for every DUT activity form, and
   100% annotation of the intended mapped nets before PTPX is allowed.

No `force`, `+initreg`, TX-ignore rule, timing-check suppression or private
state initialization is acceptable.  Treating unknown macro Q as harmless
under a valid bit is not admitted: PTPX sees the mapped internal cone, not only
the architectural observation point.

## Bound limits

M1766 is read-only and executed no EDA or license query.  It records one
consumed M1757 attempt, one compile, one simulation, one generated but rejected
SAIF, zero PTPX runs and no canonical M1757 result.  The private build remains
unsealed/do-not-cite; M1766 binds its current log/SAIF hashes for forensics but
does not promote that directory into a result.
