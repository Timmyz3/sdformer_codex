# M1171 corrected remote checkpoint-binder launcher author receipt

Status: `PASS_M1171_REMOTE_BINDER_LAUNCHER_SOURCE_AND_MOCK_TESTS__FRESH_DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_REMOTE_RUN`

The failed predecessor named an absent `/opt/anaconda3/...` interpreter and stopped before Python execution. M1171 is additive: it binds the observed remote `/opt/conda/envs/sdformerflow/bin/python`, exact Python 3.10.20, all three sealed binder source SHAs, and a new M1171 output/attempt namespace.

Nine temporary-fixture tests pass. The launcher performs every read-only preflight before consuming its attempt; once consumed, it invokes exactly one child and never retries. It accepts a normal conda interpreter symlink only when its resolved target is an executable regular file. A successful child is still rejected unless the exact four payloads, manifest, outer seal, hashes, and M1167 terminal token all validate.

This is source readiness only. A fresh different-author hammer must pass before any remote invocation; the eventual sealed selection result also requires a different-author result hammer before E1-E8 hardware rebind.
