# M1163 final-checkpoint selection/rebind binder — source receipt

Status: `PASS_M1163_SOURCE_AND_MUTATION_TESTS__INDEPENDENT_SOURCE_HAMMER_REQUIRED__WAIT_VALID825`

The source-only binder is ready. It has **not** selected a checkpoint: the existing remote standard-valid825 job is still the authority and all five epochs must finish first. This receipt did not access the remote run, hash or copy a checkpoint, start or interrupt a GPU process, capture a profile, replay hardware, launch EDA, or modify `docs/359`.

The production selector is deliberately narrow: exactly epochs 9/14/19/24/29; every profile must contain 825 samples; artifact identity must exactly equal the current config and checkpoint SHA/size/mtime; all four missing/unexpected audit counts must be zero; module counts must be exactly 105 ATLIF and 12 attention; and the generated ranking must declare `ranking_mode=aee`. It independently recomputes the minimum exact AEE and uses the lower epoch only as a deterministic tie break.

Its sealed output freezes the config and all five checkpoint/profile identities, a five-checkpoint accuracy/activity table, the selected checkpoint SHA/size/mtime, and E0-E8 invalidation/rebind targets. The spike energy field is explicitly labeled an activity proxy, not hardware energy.

Thirteen controlled tests passed. They include successful output sealing plus fail-closed mutations for incomplete populations, 824 samples, identity drift, each missing/unexpected axis, module-count drift, candidate ranking, wrong/incomplete ranking order, duplicate/non-finite JSON, config/checkpoint drift, symlinks, and overwrite attempts.

Next gate: a different author must first hammer this exact source and contract. Once that passes and the existing valid825 process exits successfully with all five profiles, execute the exact command frozen in the contract once. A second different-author hammer of the resulting small binder receipt is mandatory before any E1 deployment evaluation or E2-E8 hardware rebind.
