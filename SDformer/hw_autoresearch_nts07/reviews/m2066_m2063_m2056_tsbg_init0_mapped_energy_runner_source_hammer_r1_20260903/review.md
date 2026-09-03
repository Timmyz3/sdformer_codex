# M2066 M2063 init0 mapped-energy source hammer

## Verdict

**PASS, 99/100, P0/P1/P2 = 0/0/2.** This review ran no VCS, EDA,
license query, or GPU task. It authorizes one fresh, no-retry, P1-serial M2063
attempt and requires an independent result hammer before any mapped function,
power, or energy claim.

## Additive-clone proof

After reverting only the module/PASS identity and the three reset-release
phase changes, including the one-shot first-load alignment shim, the M2063 base
TB is byte-identical to frozen M2051. Both hashes are
`64805bdedb7c80d5c6141bc36e59ef61234507b40942e69ccbf4a30ac2383436`.
There are zero `force` and zero `release` statements. No workload payload,
scoreboard, ledger, attack, or recovery was removed.

The initial negedge release consumes its one-shot alignment flag immediately;
the first descriptor is therefore presented at that inactive edge and all
later descriptors keep the original inactive-edge task. With 192 descriptors,
the sampled preload endpoint remains `2*192-1 = 383` cycles. The wrapper also
fails unless `full_execute_start_cycle==383`.

## Mapped activity and power protocol

Each axis retains its corresponding M2029 netlist and SDC, global slot 42,
20,292/7,569 execute-cycle denominators, two wrapper stops, and a third UCLI
run through retired replay, stale attack, both reset recoveries, legal service,
all ledgers, and the exact M2051-equivalent PASS.

Qualifiers, faults, busy and counters are checked unconditionally. Payloads are
checked only under their owning valid and bank-valid. SAIF must have exact
duration, positive toggles and TX=0; PTPX must report 100% nets, 100% fully
annotated leaves, zero blackboxes/macros, successful `check_power`, and unique
nonnegative power fields.

## Initialization boundary

Following the VCS V-2023.12-SP1 two-phase use model, each of the two compile
commands contains exactly one `+vcs+initreg+random` to enable initialization
instrumentation, and each simv command contains exactly one
`+vcs+initreg+0` to select deterministic zero. No `UNIT_DELAY` define or SDF is
present. Ten negative command mutations were exercised without EDA; all ten
were rejected, including missing, wrong-phase, duplicate and mixed initreg
options, `UNIT_DELAY`, and extra/duplicate runtime plusargs. This is deterministic
handling of the M2061 zero-delay four-state reset X-pessimism. It is not a
silicon power-on model, reset implementation, timing repair, delayed gate
simulation, or mapped-equivalence proof.

## Authorization

The exact-pinned runner may execute once with one `lmstat`, two compiles, two
simulations, two SAIF files and two PTPX runs, serially and with no retry. Any
post-authority failure consumes the attempt. A successful output remains a
single positional G48 component result with directed weights and logic-only
prelayout power; external weight SRAM and system energy remain excluded.

`docs/359` remains unchanged at `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
