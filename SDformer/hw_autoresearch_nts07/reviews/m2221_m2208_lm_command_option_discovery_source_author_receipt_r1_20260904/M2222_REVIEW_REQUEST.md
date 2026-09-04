# M2222 independent source review request

Independently hammer the new M2221 source identity. M2221 does not authorize LM execution.

Required review focus:

- prove there is exactly one future `lm_shell -no_init -f` startup and no alternate-shell guessing;
- prove all HOME/CWD/TMP/XDG/cache/output paths are fresh and isolated with no setup files;
- prove `generate_frame_from_mw`, create/open/save library, Milkyway launch, NDM/NLIB writes, and P&R are impossible in this discovery source;
- prove the only allowed session mutation is a conditional `lib.setting.milkyway_exec` set/readback after runtime registration is observed;
- mutate every command/option state, the set/readback gate, side-effect evidence, censuses, inventory, and output manifest;
- verify the fixed tool, LM documentation, and 1051-member Milkyway manifest identities;
- verify M2207 remains consumed and unauthorized for retry;
- independently verify `docs/359` remains at SHA-256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

Only an exhaustive double-sealed result with status
`PASS_M2222_M2221_SOURCE_HAMMER__M2223_ONE_SHOT_AUTHORIZED`, score at least 95,
and P0/P1/P2 = 0/0/0 may authorize the single M2223 discovery attempt. Any other result keeps M2223 unauthorized.
