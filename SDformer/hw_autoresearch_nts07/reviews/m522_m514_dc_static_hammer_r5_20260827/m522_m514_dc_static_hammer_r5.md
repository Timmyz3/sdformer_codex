# M522/M514 logic-only DC static hammer r5

Verdict: **STATIC GO for exactly one execution of runner `375e7602106e46d13520e3e1301254c61e489002030004bac751d7f5fb921a88`.** Score 98/100, P0=0. No EDA tool was run during this review.

## What the r5 repair closes

The r4 blocking gap is closed. The EXIT trap is installed after the unique r3 staging directory is created and before any of the four sealed-input-root verifiers. On every incomplete exit it selects only this invocation's r3 staging directory or the post-move r3 canonical directory. Before quarantine publication, the embedded Python walks with `os.scandir` and `entry.stat(follow_symlinks=False)`, records each root-relative path and raw `os.readlink()` text, rechecks both link type and raw text immediately before unlink, then requires a second scan to contain zero links. It creates both the inventory and failure marker with `O_CREAT|O_EXCL` and `O_NOFOLLOW` where available; the inventory is finite JSON (`allow_nan=false`). The shell rejects any occupied quarantine name. After the move, a second no-follow walk requires zero links and rereads the inventory schema, status, exit code, and count. A sanitizer or post-move assertion failure cannot reach canonical PASS.

The unchanged embedded sealed-root verifier still admits exactly the two historical VCS link tuples and no others. It excludes only root-relative `SHA256SUMS` and `SHA256SUMS.seal.sha256` from regular-file topology, so a nested file with either basename remains a normal member and an unsealed one fails. The four actual sealed roots and all 16 frozen contract inputs match; the historical VCS root has 94 sealed regular members and exactly two allowed links, while all review roots are zero-link and double-sealed.

All three requested r5 P1 hardenings are present. The collision gate names `dc_shell`, `dc_shell-t`, directly invoked `snps_shell`, `fm_shell`, and `pt_shell`. Receipt construction uniquely reparses `TIM-209=` and `OPT-150=` from `precompile_loop_gate.rpt` and requires 0/0; it separately rereads `constraint_violators.rpt` and requires five clean constraint results. The Tcl derives the precompile loop gate from `precompile_build.rpt`, `check_design_precompile.rpt`, and `check_timing_precompile.rpt` before flattening or compile.

## Isolation and mutation evidence

`bash -n`, strict contract JSON parsing, and compilation of all six embedded Python blocks pass. A wrong self-SHA exits 10 before staging, canonical, quarantine, resource admission, or EDA.

The unchanged sealed-root verifier was replayed in 17 isolated cases. The real VCS root passes only `historical_vcs_exact2`; the zero-link review root passes only `zero_symlink`. A third link, link-path drift, raw-text drift with the same resolved target, target drift, out-of-root target, dangling target, directory target, unsealed target, target-SHA drift, target replacement by a symlink, an unsafe manifest member, and unsealed nested manifest/seal basenames all fail as expected.

The exact r5 quarantine bodies were replayed in 11 isolated cases. Zero-link, relative-file link, relative-directory link, dangling link, absolute/out-of-root link, nested link, and preoccupied inventory symlink trees sanitize to zero links with a regular finite inventory. A preoccupied regular inventory fails closed. A raw-link-text mutation between scan and unlink fails before unlinking the changed link. An occupied quarantine name fails the literal collision guard. A link injected after the move fails the post-move assertion.

## Frozen implementation and authorization boundary

The top is `m514_c2_convtranspose_k3s2_polyphase_address_mapper`; the filelist contains only its frozen RTL. The runner pins the resolved Synopsys executable, both TSMC28 libraries, RTL, filelist, SDC, Tcl, contract, VCS receipt, upstream reviews and their double seals, and `docs/359`. It uses `-define SYNTHESIS`, a 3.000 ns clock, 0.200 ns setup uncertainty, final 0.090 ns hold uncertainty, 0.250 ns input/output delays, 0.010 pF output load, `ZeroWireload`, and an explicitly ideal `clk_core`. Successful staging must contain mapped Verilog/SDC/DDC/SVF, finite receipt schema `m522_m514_c2d_logic_only_dc_receipt_v3`, topology schema `m522_exact_output_topology_v2`, zero links, and a double seal before atomic rename; canonical is then fully reverified before completion.

This review authorizes **one and only one** positive invocation with the exact runner SHA above. A failed positive invocation consumes this authorization and requires a new independent failure review before any retry. Even a successful run admits only standalone 3 ns pre-macro additive decoder-support logic area/timing after a separate receipt-blind hammer. It does not admit decoder cycles, system speedup, energy, SRAM, Formality, paper-ready PPA, or a DATE headline.

`docs/359_DATE终局冻结_20260813.md` was not modified and remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
