# M1210 C1 R8 random-request quiesce author receipt

Status: source-only PASS; a fresh different-author hammer is mandatory before any release or VCS run.

M1207 already proved compile/elaboration/link, then exposed a testbench race at random transaction 1. The forced request tuple remained asserted while both request-ready inputs stayed high. Response completion cleared `request_active`, allowing the same tuple to handshake again.

R8 changes only the testbench. It quiesces both request-ready inputs after the intended request handshakes and before response driving. Dedicated windowed counters enforce exactly one weight handshake and `first` psum handshakes per random transaction. Core-ready backpressure remains intact. Frozen RTL, 16 assertions, 6 covers, 7 protocol attacks, 2 service attacks, 24 random transactions, normal row/task completion, and II=2 are preserved.

The source checker passed 96 checks and rejected 6 mutations, including removal of the ready quiesce. No VCS, simv, EDA, license, GPU, or network action occurred.
