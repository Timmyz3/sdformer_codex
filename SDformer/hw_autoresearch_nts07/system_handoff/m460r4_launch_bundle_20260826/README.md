# M460R4 code/data/environment trust split

M460R4 closes the three P1 findings from the M460R3 independent hammer without authorizing capture. The executable code root is the clean `c1531749` linked worktree; the original dirty tree is an immutable-data-only root and is never placed on `sys.path`.

Launch roots:

- Contract SHA: `9bdbf4d33d52feda9e52142c032bc53f0bbd08bea19ea345b17b0f0edad170fa`
- Launch manifest SHA: `2da885993a72f881baf66a9792f88764f9357bf91d8d002a7afd9f4d7ac70f76`
- External launch outer-seal-file SHA: `4a9d8effe78878774c910284d256537fc258290a015c6c174af1850acd72e604`

The exact sealed command used for the read-only inventory/preflight/idle run was:

```bash
ssh -p 10037 root@ssh.sd5ai.scnet.cn -- \
  /usr/bin/env M460R4_EXPECTED_OUTER_SEAL_SHA256=4a9d8effe78878774c910284d256537fc258290a015c6c174af1850acd72e604 \
  /bin/bash /root/private_data/work/sdformer_codex/SDformer_m460r4_c153/SDformer/hw_autoresearch_nts07/system_handoff/run_m460r4_sealed_preflight_no_capture_20260826.sh \
  --preflight-no-capture
```

The runner has no capture mode. It authenticates the external seal before Python or `nvidia-smi`, launches Python with `-I`, sets `PYTHONNOUSERSITE=1`, unsets `PYTHONPATH`, builds a 7-package/29-import inventory, re-collects it inside preflight and requires exact equality, verifies the clean Git roots and immutable data assets, scans both worktrees for critical untracked import shadows, and records four idle snapshots.

Remote result roots:

- Result inner manifest SHA: `7b6ad2e0d536687aa6fd04460f04bdd438075ac40ee49dcab653d2b72fea7d91`
- Result outer-seal-file SHA: `317d36436de1d02b94597f4ad1946ff13924da2c03dedc67921d614b0549acd5`
- Package/build inventory SHA: `1e846dbb1ac6bbef3d262d0381829f342c6959e74e9b96460f3803a270b1ea5d`
- Preflight receipt SHA: `79ca51381d5a23398b0f854f0e7de2f65ba5ff44fc1db44caf6c50355a64418d`
- Idle receipt SHA: `fefd88b3b22ab99d2d6f0c28f3cb4b3ac99e95c0867f02b21621dfa6ccdd9207`

Decision: **GO independent hammer of the exact R4 roots and remote preflight receipt; NO-GO GPU capture/training.**
