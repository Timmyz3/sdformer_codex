# M1173 / M1171 remote binder launcher independent hammer

Verdict: **PASS, 99/100**.  The exact M1171 launcher source is authorized for
one transfer and one zero-argument remote invocation.  This is not checkpoint
selection authority and does not authorize hardware rebind.

The hammer independently pinned the launcher, author test, contract, M1163/R2/R3
source chain, configuration contract and protected `docs/359` identity.  It
verified the raw interpreter path `/opt/conda/envs/sdformerflow/bin/python`,
exact Python `3.10.20`, and the rule that a conda symlink must resolve to an
executable regular file.

Twenty-eight independent temporary-fixture tests and nine rerun author tests
passed with zero failures/errors.  Attacks covered wrong interpreter/version,
all three source drifts, config/docs/ranking substitution, epoch alias/missing
population, profile and directory symlinks, preexisting output/attempt names,
child failure/no-retry, unsealed or extra output, duplicate stdout, and wrong
terminal token.  The attempt is consumed after all preflight checks and before
the sole child; the child has a fixed clean environment and no shell.

Authorization is deliberately narrow:

1. Transfer exactly the launcher whose SHA256 is
   `ec3483ec484e3e61c7bb27530682b597837e375c2649403f9b27617b4b54c695`
   to its repository-relative scripts path exactly once.
2. From `/root/private_data/work/sdformer_codex/SDformer`, invoke that source
   with the pinned interpreter and zero arguments exactly once.
3. Do not retry after attempt consumption.  Independently hammer the resulting
   sealed M1167 output before selecting a checkpoint or starting E0–E8 rebind.

No remote host, checkpoint, GPU, VCS or EDA was accessed by this review.
`docs/359` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
