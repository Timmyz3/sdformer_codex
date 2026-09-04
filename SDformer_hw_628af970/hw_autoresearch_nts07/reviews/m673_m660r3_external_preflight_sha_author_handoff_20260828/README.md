# M673｜M660-r3 external preflight SHA roots author handoff

Status: `AUTHOR_REPAIR_ONLY__FRESH_HAMMER_REQUIRED__NO_GPU_COMMAND`.

M672 independently closed all M660-r2 numeric/capture repairs except one P1:
the one-shot runner semantically verified a double-sealed CPU preflight but did
not bind the exact receipt and outer-seal identities frozen by the fresh
review.  M660-r3 keeps the byte-identical M660-r2 producer and adds two caller
supplied, fresh-review SHA roots to a new runner.  Both are compared after
nested seal verification and before the attempt directory is created.

## Frozen candidate

- producer: `53b91b9ec8be00e60a5e029c63c392f5fe5e4773de92b440c6d4561dc1ab0116`
- runner: `8fc347dc3ba8f8dba601a34938e1f5788c0c3c2153c3da9e0dbb09b7ecffdf55`
- contract: `4acdfef539cdb03c26a3eeb9944842f94601e316676745bc36a1836f77705195`
- repair tests: `aa43c821fde7cbb675fe57e379033919477de739c6f17266d1aff15099caec85`
- CPU preflight receipt: `e773b5538ea39586b99a56c80f221df4f0e6e689fefc5648ecb6f413eb05f11b`
- CPU preflight outer-seal file: `97c565a9a458c7d8b793f0dbe9afb52a7a78566edc614706da2d935ec0bf5880`
- M672 NO-GO review outer-seal file: `fa8049662366734c43f353e3ab67dbe8fb3e124edae2f62d8a29f06895e89a22`

The runner is executable.  A wrong reviewed receipt SHA returned exit 41
before attempt creation; canonical output and attempt remain absent.  The
combined M660-r2/M665/M673 author regression passed 44/44.  The new CPU exact
load passed with missing/unexpected 0/0, no forward and no GPU.

No GPU, one-shot, performance simulator, RTL or EDA is authorized by this
handoff.  `docs/359` remains `dedde7ce...`.
