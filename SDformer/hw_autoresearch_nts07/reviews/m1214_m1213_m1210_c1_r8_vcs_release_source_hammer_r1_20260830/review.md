# M1214 independent release hammer

Verdict: **GO, 99/100, P0=0, P1=0, P2=0**.

This was a read-only source/release review. No VCS, simv, EDA, network, GPU, or capture process was launched.

The exact M1213 runner, source contract, release, checker, author receipt, M1210 authority, M1212 source hammer, RTL/TB/SVA/filelist, foundry UNIT_DELAY model, VCS binary, Python binary, and docs/359 identities are closed. Three recursive authority seals and two contract sidecars verify completely.

The future runner verifies the fresh M1214 recursive seal and the independently supplied review, manifest, and outer-seal-file SHA256 values before semantic review, namespace checks, collision checks, and persistent attempt creation. Only after those gates does it invoke exactly one VCS compile and one 1800-second bounded simv. There is no automatic retry. Any post-attempt failure is isolated in a recursively sealed quarantine.

Six independent semantic mutations were rejected: bad status, runner drift, forbidden self-manifest identity, two compile authorization, P1 issue, and sub-threshold score.

Authorization is narrowly limited to one M1213 functional UNIT_DELAY VCS compile and one bounded simv. Timing, cycles, speedup, PPA, energy, system speedup, and paper-citable claims remain false pending the result hammer.
