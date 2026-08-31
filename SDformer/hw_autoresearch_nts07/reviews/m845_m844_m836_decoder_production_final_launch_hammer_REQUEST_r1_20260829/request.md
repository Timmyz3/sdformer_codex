# M845 fresh final-launch hammer request for M844/M836

Independently hammer the exact M844 true release and frozen M836 production chain. This is a source-and-release review only. The reviewer must not invoke the runner or create the one-shot attempt, production result, failure quarantine, log, stage, or partial artifact.

The exact release SHA is `32ada02b95c3b845d604cf3d902cda105bddd386dd5499e7f73ad8cfb40f445e`; driver/runner/contract/candidate remain `4ffb51ed...` / `be666f23...` / `4479f537...` / `bcdaa576...`.

A PASS is valid only at 100/100 with P0/P1/P2 = 0/0/0 and status `PASS100_M836_FINAL_LAUNCH__AUTHORIZE_EXACTLY_ONE_PRODUCTION_REPLAY`. It may authorize the root caller, not the reviewer, to run one exact production replay.

Recheck the exact FD/inode/bytes/nonce publication protocol, rollback no-clobber behavior, and empty canonical attempt/result/failure population. Recheck the frozen 160-row, T10, three-configuration, 96-lane, 240-KiB, Acc24, 3-ns, 192-B/cycle meaning. D1 remains charged/nonheadline; only K8 versus equal-service K1x8 may headline.

The source-only unit tests assert that the future release is absent. Run those tests only in a hermetic temporary mirror with the release omitted; validate the actual sealed release separately through the exact M836 release preflight. Do not move or rewrite the canonical release.

Even after a final PASS, a raw production result is noncitable until a separate fresh result hammer passes. `docs/359` must remain exact.
