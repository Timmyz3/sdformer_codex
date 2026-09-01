# M1771 additive correction to the M1766 warmup recommendation

M1766's failure disposition remains correct, but its proposed **single
sequential warmup** is insufficient.  This additive overlay changes only that
successor recommendation; no M1766 file is modified.

## Evidence

The mapped SAIF contains 3,584 `directory_q` bits and 2,048 `mask_q` bits.
Their two flattened halves have sharply different unknown-time signatures:

- logical bank1 maps to `directory_q[0:1791]` and `mask_q[0:1023]`; every bit
  has `TX=756 ns`, the complete measurement window;
- logical bank0 maps to `directory_q[1792:3583]` and `mask_q[1024:2047]`;
  every bit has `TX=1..190 ns`, matching progressive first-task loading.

The netlist names bind the upper flattened halves to `*_q_reg_0__*`, while the
lower halves bind to `*_q_reg_1__*`.  The RTL free-bank loop selects bank0 for
the first task.  When that task drains, bank0 becomes free again; therefore a
second task loaded only after completion also selects bank0 and leaves bank1
unknown.  A single sequential warmup cannot make all mapped DUT activity known.

## Correct public-port warmup

M1772 should use three strictly increasing epochs and the same 64 masks:

1. start epoch 5943 in bank0;
2. while bank0 execution remains occupied, apply legal backpressure only on
   the public `psum_write_ready` and `row_complete_ready` sinks and load epoch
   5944 into bank1 through the public prep port;
3. restore the public sinks and wait for both matching `task_done` events;
4. start epoch 5945 and enable SAIF only for this third task.

This is supported by the RTL bank lifecycle and by M948's existing two-bank,
increasing-epoch overlap verification.  It also warms the persistent global
parent scratch; each accepted 1,152-bit write fans out across all nine SRAM
slices.  No `force`, hierarchy, `+initreg`, or TX masking is permitted.

The successor still passes only if the third task has a public scoreboard PASS,
every DUT SAIF `TX` is zero, and PrimeTime reports 100% annotation of the
intended mapped nets.  M1771 executes no EDA and authorizes none; the future
chain is M1772 source, M1773 independent review, and M1774 release.
