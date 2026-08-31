# M865 — M861 decoder streaming/event-sweep fresh source hammer

Verdict: **fixed-path FAIL, 92/100, P1=1.** M861's Python 3.10 semantics and bounded scalability checks are strong, but the requested dual-runtime gate cannot pass on this host: the native system Python 3.6 cannot import the exact source because `dataclasses` is absent; after transparent review-only shims, the real-prefix oracle is still blocked by absent `torch`, and the exact test suite itself uses the Python >=3.7 `subprocess.run(text=...)` API. Therefore this review does **not** authorize the full first-row diagnostic.

## What passed

- Exact M861 analyzer `f72ed3b8...`, tests `cd9cb5ac...`, contract `5ca88752...`, frozen M785 `7fbd72d2...`, frozen M768 `92606976...`, and `docs/359` `dedde7ce...` all match. The M857 failure authority, M861 author handoff, and M862 request manifests and outer seals independently verify.
- Python 3.10 ran the exact unmodified pytest population: 14/14 PASS. The exact source-candidate validator and self-test also pass. Both Python 3.6 and 3.10 can byte-compile the files; byte-compilation does not satisfy runtime import/execution.
- An independent 512-request DAG matched all 11 frozen fields and all 512 produced-token readiness entries. A separate endpoint oracle matched 256 shuffled trials / 23,687 cycles. A brute-force interval-union oracle matched 10,240 intervals including 7,423 out-of-order insertions.
- A different hand-authored E/D/I/R population made all six priority classes nonzero and preserved `active > dependency/inflight > weight > psum > memory > compute`. Independent attacks confirm weight 1R1W parallelism, psum 1RW serialization, outstanding=1 saturation, and same-cycle return-slot reuse.
- Aggregate mode consumes a one-shot iterable once and exposes neither `scheduled_requests` nor `compressed_schedule`; the scheduler object retains neither detail attribute. The exact 100K real-prefix state contained 100K token-ready entries, 24 port-calendar entries, 44 outstanding returns, and only 1,440 merged priority intervals, with no scheduled/compressed detail population.

## Bounded scale only

Synthetic 1K/10K/100K completed in 0.020/0.151/1.485 s with isolated maximum RSS 31,064/31,796/39,372 KiB. Exact M854 first-row prefixes of 1K/10K/100K completed in 1.943/2.225/4.378 s; isolated RSS was about 0.88/0.88/0.88 GiB for the streaming-only runs, showing the pinned payload/oracle load dominates the bounded baseline. The 100K real prefix plus a separate 1K old/new miter also passed, at 5.59 s and 952,480 KiB. These values are diagnostics only, never production cycles or speedup.

No full 38,672,612-request row, 4,800-row population, production result, VCS, DC/EDA, license, GPU, remote, or training action was run.

## Blocking finding and successor

M865-P1-1 is an authority/runtime mismatch, not an M861 scheduling-semantics defect. `/usr/bin/env python3` resolves to Python 3.6.8 here, while the proven execution environment is `/opt/anaconda3/envs/pytorch310/bin/python3.10`. Native 3.6 stops at missing `dataclasses`; the review-only shim intentionally does not hide the later missing `torch` and Python-3.7-only test API. Thus the requested Python 3.6/3.10 14/14 condition is false.

The minimum successor is additive: leave the exact M861 source unchanged, create a new nonproduction diagnostic identity/request that explicitly declares Python 3.10-only, pins the absolute interpreter and its SHA/version, forbids the ambient `python3`/shebang path, and repeats a fresh source hammer. Only a fresh 100/100, P0/P1/P2=0 review of that identity may authorize one full-first-row aggregate diagnostic. No production release or performance citation follows from this review.
