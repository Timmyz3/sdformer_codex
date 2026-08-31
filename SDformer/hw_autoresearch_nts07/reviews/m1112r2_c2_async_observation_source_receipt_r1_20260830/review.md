# M1112r2 C2 reset-provenance/live-seal source receipt

Status: `M1112R2_RESET_PROVENANCE_LIVE_SEAL_SOURCE_FROZEN__M1114R2_INDEPENDENT_HAMMER_REQUIRED__NO_EDA`

M1112r2 closes only the two P0 findings from the M1113 STOP. The 13-counter, 337-bit async observation bank and 22-signal atomic unknown bitmap remain unchanged under a fresh additive preprocessed module/testbench identity; no observation feeds the functional C2 cone.

The mapped structural gate now parses every shadow sequential instance and traces its active-low `CDN`/`CN` net to exactly one allowed single-input TSMC28 inverter. That inverter must take the canonical active-high `rst_core` directly and drive the clear net directly. Constants, direct wrong-polarity connections, unrelated reset nets, multiple drivers, buffers/data gates, multi-level or reconvergent logic, set-only reset-to-one cells, and any census other than 337 are rejected.

Every live primary, sidecar, manifest, outer seal, source, executable, library, model, and input is required to be an `lstat` regular non-symlink. Live sealed directories require safe unique manifest paths and exact equality between listed and actual regular members, so omitted or extra files fail. The Synopsys regular `snps_shell` target is invoked directly rather than through `dc_shell`. The only followed-byte symlink exception is the exact sealed historical M1080 quarantine, whose manifest and outer remain regular and whose sole linked target must stay inside the sealed directory.

Author static/mutation testing passed 70 checks and rejected 13/13 adversarial mutations, including M1113's fake-reset, sidecar-symlink, manifest-symlink, and unlisted-extra attacks. No VCS, DC, simv, launcher, result, or attempt was created. This is not mapped functionality, PPA, activity, performance, or paper evidence. Only a different-author M1114r2 hammer is permitted next.
