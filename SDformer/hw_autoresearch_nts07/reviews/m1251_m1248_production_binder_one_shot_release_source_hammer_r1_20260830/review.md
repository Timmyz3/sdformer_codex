# M1251 independent hammer of M1248

Verdict: **BLOCK**. The exact baseline suite passes 10/10, but controlled mutations expose four acceptance gaps. M1251 does not authorize production now or later; a source successor and a fresh different-author hammer are required before the one allowed binder execution.

## What passed

- Exact M1241 source/test/contract pins and the recursive M1245 double seal match.
- The preflight enumerates four profiles, four checkpoints, two distinct configs, and one new-run manifest.
- Output, attempt, and log are fresh sibling namespaces; attempt creation uses `O_EXCL`.
- Attempt races are stopped. Output/log races fail closed after attempt consumption, and the attempt remains consumed.
- A child failure is not retried. Receipt member population, manifest, outer seal, schema, status, and four required denial fields are checked.

## Release-blocking findings

1. **F1, HIGH — executable TOCTOU.** M1241 is hashed before attempt consumption and later reopened by pathname for execution. The hammer changed it after attempt consumption and the wrapper did not bind the executed bytes back to the preflight identity.
2. **F2, HIGH — artifact TOCTOU.** The eleven candidate inputs are presence-checked only. Their pre-attempt identities are not compared with `candidate_population`/manifest identities in the child receipt.
3. **F3, HIGH — open claim boundary.** A double-sealed receipt with `power_or_energy=true`, `paper_metric=true`, and `hardware_replay_complete=true` is accepted because only four fields are inspected.
4. **F4, MEDIUM — selection pair not bound.** `legacy_ep29` plus epoch `34` is accepted because candidate and epoch membership are checked independently.

Self-generated SHA manifests are integrity checks, not authorization roots. Therefore exact schema/status and double seals do not compensate for F1–F4.

## Required successor behavior

- Snapshot SHA256/size/path identity for all eleven real inputs before consuming the attempt and compare the complete child receipt against that snapshot.
- Bind the executed child and its imported dependencies to the exact preflight bytes through an immutable execution image or equivalent descriptor-rooted mechanism.
- Require the exact M1241 claim-boundary key/value map; reject unknown affirmative claims.
- Require one of the four exact candidate/epoch pairs and recompute the selected row from the complete four-row metrics table.
- Preserve existing `O_EXCL`, no-retry, failed-attempt retention, docs/359, and fresh-output rules.

Even after a successor passes, execution is only conditionally eligible once all four strict valid825 profiles exist. This review accessed no remote host and started no GPU, training, capture, VCS, or EDA process.

