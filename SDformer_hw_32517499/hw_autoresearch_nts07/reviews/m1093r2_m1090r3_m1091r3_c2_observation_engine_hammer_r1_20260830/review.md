# M1093r2 independent fixed-history engine hammer

Verdict: **GO to author one different-author zero-argument launcher only; no launch, EDA or attempt.** All 82 receipt-blind static and bounded checks pass.

The frozen identities match exactly: source receipt outer `8bc6f725...edd9e`, engine `41b78990...af04`, contract outer `d2e5d49d...82c0d`, and release outer `fc6bb488...c853`. All 21 project-source pins, seven regular external pins, and the exact `dc_shell -> snps_shell` exception verify. The observation wrapper/TB preserve 22 fanout-only probes, 22 first-X fatal checks, 16/32-cycle ingress limits, the 128-cycle stage window and diagnostic-only/no-SAIF/no-initreg boundaries.

The repaired historical validator is correctly narrow. The real M1080 quarantine verifies with exactly one manifest-listed VCS symlink. Synthetic attacks reject non-M1080 paths, directory/path escape, external target, followed-byte drift, `../` manifest paths and directory symlinks. Retargeting to a different internal regular file with identical bytes is intentionally accepted by the frozen manifest's followed-byte semantics; no live input uses this policy. Live source/tool/library/model symlinks and byte changes remain rejected by direct `lstat` plus exact SHA.

Bounded direct invocations with no argv, extra argv and forged legacy expected-hash environment variables all fail before attempt. The exact authorized argv completes the valid fixed-history/source/tool preflight and stops only because the fixed launcher/receipt do not yet exist. M1091r3 result and attempt remain absent.

Authorization is limited to creating the fixed-path zero-argument launcher and double-sealed launch receipt. The launcher must hardcode engine SHA `41b7899083152f8099acac759109a8eb22c381cb6a17506ae85e6666656daf04` and this M1093r2 outer seal, sanitize the execution environment, accept no arguments or caller-selected authority, and remain unexecuted until M1096r2 independently hammers it. Root must then externally pin the exact launcher SHA and M1096r2 review/manifest/outer tuple before the single launch.
