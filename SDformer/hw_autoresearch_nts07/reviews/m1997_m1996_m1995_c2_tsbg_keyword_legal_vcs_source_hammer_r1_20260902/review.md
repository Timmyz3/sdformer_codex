# M1997 independent source and VCS-runner review

## Verdict

**PASS, 100/100.** The frozen M1995 source is the exact additive repair modeled
by the sealed M1995 failure review. Relative to frozen M1880, exactly 16
standalone `context` identifier tokens on 12 code lines are renamed to `ctx`.
Replacing those 16 tokens back reconstructs M1880 byte-for-byte. There are no
comment, module, port, parameter, state, schedule, arithmetic, or protocol
changes. The old M1880 source and docs/359 identities remain unchanged.

The filelist contains M1995 exactly once and excludes M1880. It otherwise pins
the admitted M803 adapter, M1880 SVA, and M1984 testbench identities.

## M1998 one-shot audit

The exact M1998 runner is authorized for one bounded directed VCS attempt:

- one license query, one VCS compile, one `simv` run, zero other EDA runs;
- no automatic retry and a fresh result/attempt/work/lock namespace;
- exact source, filelist-via-review, prerequisite seals, runner, review, and
  docs/359 gates;
- `-assert svaext` at compile and `global_finish_maxfail=1` at runtime;
- 180-second external timeout with a 10-second TERM-to-KILL interval;
- one exact complete PASS line, all ten begin/complete phase pairs, 52/52 load
  begin/complete records, zero load timeout, and explicit compile/runtime error
  rejection;
- sealed attempt consumption, sealed quarantine on failure, and sealed atomic
  publication on success.

Eight mutations weakening identity, prerequisites, timeout, SVA maxfail, exact
PASS parsing, no-retry, quarantine, or the one-query census were independently
rejected. No EDA tool or license query was launched by this reviewer.

This review authorizes only the exact VCS attempt. It does not admit M1995
function before an independent result hammer, does not authorize DC, and makes
no same-area, exact-cycle-speedup, system-speedup, power, or PPA-ready claim.
