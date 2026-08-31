# M1080 additive C2 one-shot release source receipt

Verdict: **PASS 100/100; freeze M1080 and request a different-author M1081 hammer. No EDA launch is authorized.**

The only functional repair relative to M1070 is a whitelist validator for the frozen DC launcher: `lstat(dc_shell)` must be a symlink, `readlink` must be exactly `snps_shell`, resolution must be exactly `/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell`, and that regular nonsymlink payload must hash to `23a4101c...`. The generic `expect_sha` policy remains strict and was not relaxed.

Static and negative self-tests rejected regular-file substitution, wrong link text, wrong resolved target, wrong payload and a dangling link. The exact production sidecar validator also rejected basename-only, suffix, traversal and extra-token attacks.

After milestone namespace normalization, `release_chain_gate`, `run_flow` and `quarantine_failure` remain byte-identical to M1070. Atomic attempt-before-EDA, fresh ARCH_MODE=0 DC, five anchors `259/737/3153/7569/14`, no initreg, failure quarantine and atomic publication are preserved. M1071 STOP outer `812a1543...` and authorization=false are pinned; an independently sealed M1081 PASS remains mandatory.

A nonlaunching invocation passed all static identities including the repaired DC launcher and stopped at the absent M1080 caller pin before attempt creation. No result, attempt, quarantine or EDA was produced. `docs/359` remains `dedde7ce...`.
