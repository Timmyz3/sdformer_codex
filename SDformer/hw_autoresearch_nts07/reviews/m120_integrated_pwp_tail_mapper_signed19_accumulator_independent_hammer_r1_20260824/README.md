# M120 integrated mapper-to-signed19 accumulator independent hammer

Verdict: the frozen positive directed scope passes, but full integrated
accepted-transaction exact-once remains open.  A legal-shaped pair of
consecutive events to the same accumulator address is accepted twice at the
service interface, then M118 rejects the second mapped update as an RMW hazard.
The independent commercial VCS run observes two accepted events and one write.

The positive campaign independently checks 768 weight-load tokens and reads,
1024 accepted events, 1024 mapped updates, 1024 writes, 98,304 mapper lanes,
6,144 commits and 589,824 commit lanes across two lazy-cleared windows.  It
also checks INT8 -128/+127 polarity endpoints, load2 tail bypass, event II=1,
lane read/write overlap and commit backpressure.

Negative campaigns check malformed load beat/key, premature window end with an
older accepted update draining, the consecutive same-address loss, separated
retry without deduplication, and reset after acceptance without recovery.

Reproduce the immutable independent commercial VCS run from the hardware root:

```bash
reviews/m120_integrated_pwp_tail_mapper_signed19_accumulator_independent_hammer_r1_20260824/run_vcs_m120_independent_hammer.sh
```

The runner refuses to overwrite `vcs_run_r1`; use a copy with a new run name
for a fresh execution.  Re-run the strict audit with:

```bash
python3 reviews/m120_integrated_pwp_tail_mapper_signed19_accumulator_independent_hammer_r1_20260824/audit_m120_integrated_independent_hammer.py
```

Safe claim: exact-SHA commercial VCS validates reset-free directed M120
conservation only for the tested traffic without consecutive same-address
events.  Foundry SRAM, macro-inclusive PPA, scheduled cycle ratio, physical
speedup, system speedup and headline admission remain false.  The M109
2.53546204172554 ratio remains a projection.
