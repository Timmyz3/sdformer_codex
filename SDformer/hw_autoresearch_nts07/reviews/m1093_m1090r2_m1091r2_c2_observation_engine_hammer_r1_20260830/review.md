# M1093 receipt-blind M1090r2/M1091r2 engine hammer

Verdict: **STOP**, `P0/P1/P2 = 1/1/0`. No launcher authoring and no EDA/attempt are authorized.

## What passed

The receipt outer identity is exactly `e1e904c18a17cd3ce4c1154c0d262aecd65826e4afaff1960e5945ee169d6216`; the engine is exactly `51e8af72a1c48ca556249ccf7abcaf1ef8e0700265c3705c24bf37f44405cd77`. All 21 project-source pins and seven regular external tool/library/model pins match. `dc_shell` is the one explicit `snps_shell` symlink exception and its regular target is hash-pinned. The engine accepts only `--authorized-launch`, has no caller-selected expected-hash environment variable, fixes the future DAG paths, and places launch/static/lock/process/resource/license gates before attempt consumption. The future workload remains one DC, one mapped compile and one 128-cycle mapped simulation, with no SAIF or initreg and diagnostic-only output.

The M1090r2 wrapper still has exactly 22 fanout-only observation outputs. The TB retains 22 first-X fatal checks, 16/32-cycle header/raw limits, the 128-cycle trace window, per-cycle stage printing and the 1000 ns watchdog. Three bounded direct-engine processes (no argv, extra argv, authorized argv plus forged legacy expected-hash environment variables) all stopped before attempt; no EDA tool was launched.

## P0 — frozen quarantine is rejected by the engine itself

`static_gate()` calls `verify_flat(M1080_FAILURE, 2e3367...)`. `verify_flat()` applies `verify_regular()` to every historical quarantine manifest member, and `verify_regular()` rejects every direct symlink. The exact sealed M1080 quarantine contains one legitimate VCS-generated internal symlink:

```text
mapped_vcs/csrc/_2931510_archive_1.so
```

Its followed bytes match the manifest digest, and the quarantine manifest/outer seal are intact. Nevertheless, the frozen M1091r2 engine deterministically raises `non-regular or direct symlink rejected` before reaching `verify_launch_authority()` or attempt consumption. Consequently a valid future launcher/M1096 chain could never launch this engine.

Minimum repair: author an additive engine revision. For the immutable historical M1080 quarantine only, validate its exact existing manifest/outer semantics and allow manifest-listed VCS-internal symlinks while checking the followed-byte digest; alternatively bind only the exact independently sealed M1088 failure-audit summary. Preserve direct `lstat` symlink rejection for all live project sources, tools, libraries, models and new inputs. Re-hammer the repaired engine before writing a launcher.

## Finite future trust boundary

M1093 does not require a filesystem artifact to predict and hardcode its own future seal. After engine repair and a later GO, root must use the independent M1096 message as the external execution trust root: verify the exact launcher SHA plus the exact M1096 review/manifest/outer triple, then invoke only that zero-argument launcher. M1096 must also verify that the launcher hardcodes the repaired engine SHA and M1093 outer and launches the engine with a sanitized environment. A fully co-resigned internal DAG is rejected by this external tuple. This is an execution requirement, not the present P0.

`docs/359` remains unchanged at `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
