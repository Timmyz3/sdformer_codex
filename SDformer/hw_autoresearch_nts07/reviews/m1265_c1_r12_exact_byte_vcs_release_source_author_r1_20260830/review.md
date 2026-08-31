# M1265 C1/R12 exact-byte one-shot VCS release source

Verdict: **source GO, execution NO-GO until a fresh different-author M1266
hammer**. This milestone ran no VCS, `simv`, EDA, GPU, or remote command.

M1265 deliberately ends the general SystemVerilog checker-hardening loop. The
launch admits only the exact R12 TB SHA `e13d630...d302`, exact M528/M935/M1162,
SVA, foundry model, VCS binary, Python binary, and frozen docs/359 identities.
Any byte change to the runner, filelist, TB, or technical corpus fails closed.

The runner additionally exposes fixed human-auditable anchors: 24 boundary-only
random cases, explicit integrated-normal enter/call/complete, and exact hashes
for the three frozen normal-M935 tasks. This does not promote boundary traffic
to integrated M935 evidence: only those three frozen tasks carry that evidence.

After M1266 independently binds the runner, contracts and its own sealed review,
the runner may consume one fresh attempt, execute one compile and one simulation,
and never retry automatically. Any failure is moved to a recursively manifested
and outer-sealed quarantine. Success remains functional boundary-only VCS, not
timing, cycle, PPA, power, energy, system-speedup, or paper-citable evidence.
