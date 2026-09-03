# M2036 independent M2037 successor-runner source review

## Verdict

**PASS, 97/100; P0/P1/P2 = 0/0/1.** Exactly one fresh M2037
successor run is authorized: one VCS compile followed by one `simv` execution,
with no automatic retry. The reviewed runner SHA is
`9ecfea0331368385421c2b7bfbf84d00fe9bf6f4d793f8fc07bfa2b25fc047b3`.
No old M2033 artifact may be reused or salvaged.

The runner retains the clean exact-bash entry, `env -i` tool allowlists,
same-UID lock and dual collision scans, immutable attempt-before-tool order,
900-second compile timeout, 180-second simulation timeout, exact terminal
cardinality, negative error gate, failure quarantine, and narrow claim
boundary. It pins and verifies the M2032 source review, M2034 old-runner
authority, M2035 failure classification, all frozen technical inputs and the
future M2036 review/release.

## Independent M2035 recheck

The old M2033 canonical result remains absent, its attempt is double sealed,
and exactly one old quarantine exists with `FAILED_DO_NOT_CITE`. Independent
filesystem inspection—not merely trust in M2035—finds exactly one symlink:

```text
csrc/_2362104_archive_1.so -> .//../simv.daidir//_2362104_archive_1.so
```

It resolves to the matching regular in-tree `simv.daidir` member. Its size is
573,944 bytes and SHA-256 is
`6e63b0e29cf867d67d6eb68fbfd434cbed4b26a6bbf6176d3a20ec22995924c8`,
matching M2035. Thus the allowed successor difference is a packaging repair,
not reuse of prior functional output.

## Symlink and publication repair

After all functional gates, M2037 enumerates all symlinks and requires exactly
one `csrc/_<digits>_archive_1.so`. It records the raw target, requires resolution
to the same-named regular non-symlink file under the private
`simv.daidir`, records target size and SHA, unlinks exactly that selected link,
and rejects any remaining symlink. The generated removal record is itself
SHA-bound by the result receipt.

The repaired terminal sequence is now complete:

1. double-seal the private stage;
2. read back both stage seals;
3. publish with `mv -T -n`;
4. require the stage to disappear and canonical result to be a real directory;
5. read back the canonical double seal before disabling failure handling.

Independent source checking passes 29 static predicates and 32/32 semantic
mutations, including stage-seal removal, no-clobber removal, missing
post-publish assertions, missing canonical readback, overly broad symlink
unlink, dirty tool environments, expanded execution budget, and forbidden
claim promotion. `bash -n` also passes. This reviewer launched no EDA, GPU, or
license utility.

## Residual boundary

P2 only: the runner records the literal raw symlink target but accepts any
spelling that resolves to the exact expected in-stage regular target. This
does not allow an outside-tree target; the independent result hammer should
still require the observed VCS raw form and exact canonical topology.

A future PASS remains one ep34 64-row real **mask** tile with deterministic
synthetic signed12 lane values and zero prior psum. It cannot turn the M1590
`1.694510x` CPU model into RTL cycles and admits no same-area speedup, timing,
power, energy, full-network/system, or headline claim. docs/359 is unchanged.
