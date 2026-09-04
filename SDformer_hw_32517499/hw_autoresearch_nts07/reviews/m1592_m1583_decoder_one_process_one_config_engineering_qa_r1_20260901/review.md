# M1592｜M1583 decoder source engineering QA

Verdict: **PASS source engineering QA. A later independent-process runner may now be authored, but no actual execution is authorized.**

Both author unit suites passed 6/6 under CPython 3.10.18 and 3.6.8. A separate in-memory hammer then passed 61/61 checks under each runtime. It never called the captured M1573 actual entry and never opened the decoder payload.

The one-shot closure contains the exact clean-import `M1573.fresh_worker_entry` function object. Replacing the mutable module attribute after construction did not change the captured cell. With an in-memory bound-entry witness, a second admitted configuration in the same process was rejected before another call. A forbidden product configuration was also rejected before the witness, while leaving the token available for one valid configuration. If the first admitted call raised, the token was already consumed and a second configuration could not enter.

The returned-result gate rejected removal of all 21 required fields and mutations to configuration, resource identity, positive cycle/request counts, request/kind conservation, address/commit/payload digests, payload extent, pilot scope, streaming mode, and RSS evidence. `gate_calls=0` failed. Baseline and maximum current/peak RSS values equal to the 8 GiB boundary all failed; the gate requires strict `< 8388608 KiB` and monotonic maxima.

This does not make M1583 itself a process launcher. The next artifact must be source-only and must execute a fresh interpreter for each of the three admitted configurations while pinning the exact M1583 identity. That runner needs a separate independent review before any actual payload execution. No cycle, request, traffic, energy, speedup, RTL, EDA, or paper result is created here.
