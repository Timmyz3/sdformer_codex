# M1120C independent hammer — M1119C/M1116C full-storage semantic gap

Verdict: **confirm STOP for storage RTL, wrapper, filelist, Tcl, VCS, DC and all EDA. Only a source-only semantic mapping may proceed.**

The audit independently binds the requested M1119C review outer `8ba6da0e...` and STOP-contract outer `5d53a7af...`, then re-derives the ledger from the sealed M1116C, M1102, M1000 and exact M935 RTL identities.

The arithmetic is exact:

- known parent + psum + weight: `18,432 + 122,880 + 49,152 = 190,464 B` across 93 known macros;
- six metadata/reserve proxies: `1,152 + 2,304 + 16,384 + 1,152 + 1,152 + 2,304 = 24,448 B`;
- total: `190,464 + 24,448 = 214,912 B`, leaving `30,848 B` under 240 KiB.

Capacity arithmetic is not semantic closure. M935 contains exactly two registered `1152-bit` response payload slots, totaling `288 B`. Thus `16,384−288 = 16,096 B` of the FIFO/control reserve still lacks a sealed object/field inventory, owner, depth/width, port and concurrency contract, clock/reset behavior, lifetime/conflict graph and live consumers. The remaining small metadata proxies have the same owner/port/lifetime problem and do not by themselves specify whole 2,048-byte foundry macros.

The hammer passed 196 checks and rejected all 21 mutations, including erased residuals, invented response dimensions, false semantic completion, dummy-capacity legalization, RTL/EDA escalation, live extras and symlinked seals. No source or RTL was changed and no tool was run.

The minimum legal next artifact is a source-only six-row semantic mapping. Each row must bind evidence and specify ledger bytes, concrete logical fields, owner and live producer/consumer, depth/width/count, clock/reset, read/write ports and simultaneous access, lifetime/conflicts, physical mapping class, justified padding and no-double-count proof. The FIFO row must separately identify the existing 288-byte payload and evidence every bit attributed to the remaining 16,096 bytes. All rows must conserve exactly 24,448 bytes and the whole coordinate exactly 214,912 bytes, with no dummy capacity or silent coordinate change. A different author must hammer that mapping before any RTL is written.

`docs/359_DATE终局冻结_20260813.md` remains unchanged at SHA256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
