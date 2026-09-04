# M1231 independent hammer of M1228 cross-run binder

Verdict: **NO-GO for production binder release**. No checkpoint is selected by this review.

The good part is substantial. The original 15 tests pass, the production policy contains exactly `legacy_ep29`, `resume_ep30`, `resume_ep32`, and `resume_ep34` across two distinct run directories and two distinct configuration identities, and the legacy/new configuration SHAs are pinned. All four candidates are mandatory. The 825-sample type gate, four typed zero load-audit fields, exact 105 ATLIF/12 attention counts, complete artifact identity dictionary, finite-AEE rejection, lower-epoch tie break, and all nine E0–E8 conditional rebind targets work. Thirteen additional independent malformed-input attacks were rejected.

Two source defects still block a production release.

First, `validate_profile` parses `spike_profile.json` and later calls `stable_identity` in a separate read. A controlled replacement between those operations was accepted: AEE 1.0 drove the selection, but the recorded profile SHA belonged to a file containing AEE 9.0. Thus the selected checkpoint can cite a profile digest that does not contain the metric used to select it. The successor must read one immutable byte snapshot, hash those bytes, parse those same bytes, and also reject stat drift.

Second, finite is insufficient for an error metric. A candidate with AEE -1.0 was admitted and selected. AEE and the other error metrics must be finite and nonnegative.

This hammer did not access either production run, either production configuration, any real checkpoint/profile, a remote host, GPU, valid825, EDA, or hardware replay. It did not modify the M1228 source, tests, or contract. The author must produce a successor and a fresh different-author hammer before any one-shot production binder release.
