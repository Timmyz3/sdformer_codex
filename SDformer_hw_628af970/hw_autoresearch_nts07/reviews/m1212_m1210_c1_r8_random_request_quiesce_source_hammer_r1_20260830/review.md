# M1212 independent hammer: M1210/R8 random request quiesce

Status: **PASS_SOURCE_HAMMER__SUCCESSOR_RELEASE_AUTHORING_ONLY__NO_VCS_NO_EDA**.

Score: **99/100**.

The R8 repair is correctly placed.  Each random legal transaction first observes exactly one weight request fire and, only when `first=1`, exactly one psum request fire.  At the following falling edge it drives both request-ready inputs low and retires the dedicated request window.  Only after that boundary may either response-valid branch execute.  The original core-ready response stall remains active.

The dedicated counters are non-vacuous: each increments only on its real valid/ready handshake while the explicit random window is live.  Both the immediate pre-response oracle and the terminal post-accept oracle require weight count 1 and psum count equal to `first`.

The independent validator rejected 12 mutations.  They include removing or delaying either ready quiesce alone, removing or delaying both, retiring the window early, weakening either oracle, ungating a counter, removing core-ready backpressure, and dropping normal frozen-M935 completion.  The author checker independently rejected 11 of these, including all six mandatory ready removal/delay mutations.

Every non-random task is semantically identical to the clean R6 corpus.  The prior R7 transitive service call-closure and own-fault/peer-clean/composed-clean oracles were independently rerun and remain closed.  The 16 assertions, 6 covers, 7 protocol attacks, 2 service attacks, 24 random legal transactions, II=2, and normal frozen-M935 row/task are preserved.

This is a source-only verdict.  It authorizes a fresh successor release package, not a VCS result.  No VCS, simv, EDA, license, GPU, or network action occurred; no timing, cycle, PPA, energy, system-speedup, headline, or paper-citable claim is admitted.
