# M460R3 sealed launch bundle

Status: **REMOTE NO-GO** until an independent reviewer approves the exact outer-seal-file SHA below and the sealed remote preflight passes. This bundle has not contacted the remote host, touched a GPU, launched capture, or trained a model.

Frozen candidate external trust-root literal:

```text
fcddf4f03ecbf552083ec5dbdf0fd2bac7bf0299eb3fa3155a158d6b52e34ad6
```

It is the SHA256 of `M460R3_LAUNCH_SHA256SUMS.outer.seal.sha256`. The outer seal binds launch-manifest SHA `499b8ab4d00af49b36f49c780f9f67b2861ca8997b3e9b27e76bf3cc79d8f6b0`; the manifest binds contract SHA `7662a07de28bc129562c5377387810d967e312a98ef912ff54cd1d19e4395514` and every launch/runtime leaf named there. The contract intentionally does not embed either later seal SHA, so the trust root is non-circular.

After independent approval and after rsyncing only the eight new files listed in the contract, the canonical remote **preflight/no-launch** command is:

```bash
ssh -p 10037 root@ssh.sd5ai.scnet.cn -- \
  /usr/bin/env M460R3_EXPECTED_OUTER_SEAL_SHA256=fcddf4f03ecbf552083ec5dbdf0fd2bac7bf0299eb3fa3155a158d6b52e34ad6 \
  /bin/bash /root/private_data/work/sdformer_codex/SDformer/hw_autoresearch_nts07/system_handoff/run_m460r3_h67_g8_ffn_token_residual_s10_sealed_20260826.sh \
  --preflight-no-launch
```

The runner first authenticates the outer-seal file against the literal, then checks the outer seal and launch manifest, and only afterward starts Python preflight. It verifies exact remote host contract, repo, interpreter, Git commit/tree, 21 critical files, checkpoint SHA, and all 30 frozen S10 DSEC inputs. The existing checkpoint and dataset must not be transferred or replaced by M460R3.

Capture remains a separate explicit operation requiring both `--capture` and `M460R3_EXPLICIT_CAPTURE=1`. Do not run it until the independent seal review and remote no-launch preflight are both recorded as PASS.

The post-capture G8 gate is readiness-only: M159 accounts 205,384,111 / 620,302,905 = 33.1103% FFN cycles. Even ideal whole-FFN skipping needs 39.3940% skip for 1.15x, 50.3368% for 1.20x, and 69.6971% for 1.30x, before predictor/BN/residual/memory overhead. An acceptable-tau S10 oracle below 39.3940% is a fast-kill; exceeding it permits only train-only predictor precompute plus valid825, not an executable-cycle or speedup claim.
