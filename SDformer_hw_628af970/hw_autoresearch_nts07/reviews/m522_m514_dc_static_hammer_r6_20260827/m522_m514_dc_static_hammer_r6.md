# M522/M514 logic-only DC static hammer r6

Verdict: **STATIC GO for exactly one positive execution of runner `1329b1656dff4580a227ab3f5143f4ccc843632536a25e15e2942680dd2d8d5d`.** Score 99/100; P0=0, P1=0, P2=0. This review did not run DC, VCS, PT, Formality, or any open-source EDA.

## The r6 launcher repair is exact

The only positive EDA launch is line 583: the literal launcher pathname `/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell` is invoked as `dc_shell -f <frozen Tcl>`. There is one such launch, no execution line for the resolved `/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell`, and zero `-shell` tokens in the runner.

The installed launcher is a symlink whose raw text is exactly `snps_shell`; it resolves to the frozen regular target with SHA256 `23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2`. Static inspection of that frozen wrapper proves why the basename matters: `script_name` starts empty at line 11, is captured from the invoked symlink basename at line 33, and the `dc_shell` arm at lines 191-200 constructs `common_shell_exec -shell dc_shell`. Direct invocation of the resolved regular file cannot enter that arm and reaches the unsupported default at lines 398-400. The r6 runner preserves `snps_shell` only for `readlink -f`, SHA checking, receipt identity, and collision checking.

## Identity, authorization, and one-shot order

`bash -n`, strict contract/upstream JSON parsing, and compilation of all six embedded Python blocks pass. The original r5 16 frozen inputs are byte-for-byte unchanged; the three root-cause-review inputs are added, and all 19/19 r6 frozen SHA values match. The historical VCS root has 94 sealed regular members and exactly the two permitted symlinks. The three existing review roots are zero-symlink and double-sealed. The runner requires this r6 review root as the fifth zero-symlink double-sealed input and self-checks schema `m522_m514_dc_static_hammer_r6`, exact GO status, P0=0, execution authorization, and the exact runner SHA.

The persistent r4 attempt directory is consumed with atomic `mkdir` only after exact identities, sealed authorizations, the process and resource gates, and the isolated wrong-self-SHA negative preflight. It is immediately before the sole positive launch. No trap or success path deletes, moves, or quarantines the attempt. Thus an interruption after `mkdir`, a DC failure, or a success all consume the authorization; a second invocation fails closed on the attempt or canonical guard.

The isolated wrong-self-SHA runner path returned the required rc=10 and created no r4 canonical, staging, attempt, or quarantine state. No positive runner path was exercised.

## Failure isolation and output closure

All new mutable identities are r4: canonical output, staging pattern, persistent attempt, quarantine, and receipt. The old r3 quarantine remains present and agrees with the sealed failure review: 14 regular files, two directories including the root, zero symlinks, inventory SHA `6a4002b2586b5a223b6775b30b150b178eb64c12b1fd4f08522ba2075c7a415c`, and one-line DC-log SHA `db8e7da6d428906db65cc813663d2345dfeb5a5cacd5ab04e2f80c439af56f39`.

The incomplete-run trap operates only on this invocation's staging or post-move canonical directory. Its scanner uses no-follow stat, records and rechecks raw link text before unlink, creates the inventory and failure marker exclusively with `O_NOFOLLOW` where supported, requires zero symlinks before the collision-guarded move, and rechecks zero symlinks plus inventory schema/status/count after quarantine. It never touches the one-shot attempt.

On success, the runner reparses finite receipt schema `m522_m514_c2d_logic_only_dc_receipt_v4`, exact topology schema `m522_exact_output_topology_v2`, launcher invoked/raw/resolved identities, five sealed-input-root inventories, TIM-209/OPT-150=0, five clean constraint classes, and zero symlinks. Staging is double-sealed before the atomic rename and canonical is fully reverified before completion.

## Authorization boundary

This review authorizes exactly one standalone M514 logic-only 3 ns Synopsys DC/STA attempt. A failure consumes the authorization. A successful run still requires a separate receipt-blind hammer before citing additive decoder-support logic area or timing. It does not admit decoder cycles, system speedup, energy, SRAM, Formality, paper-ready PPA, or a DATE headline.

`docs/359_DATE终局冻结_20260813.md` was not modified and remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
