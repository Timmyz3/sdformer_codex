# M1312 canonical-path compatibility successor

## Verdict

**PASS. Use M1312—not M1309—as the frozen M1233/M1249 selection-hammer entry.**

M1309 correctly binds the staged archive, but its `result_path` names the local
incoming directory. Frozen M1233 requires exact equality between that field and
the launch contract's canonical selection entry. M1312 changes exactly one of
the twelve `selection_authority` fields:

- old: `hw_autoresearch_nts07/system_handoff/incoming/.../hw_autoresearch_nts07/results/...`
- new: `hw_autoresearch_nts07/results/m1257_motion_cross_run_final_checkpoint_selection_r5_20260830`

The other eleven fields are byte-for-byte equal to M1309. The M1237 schema,
status, independence and authorization are unchanged.

This rewrite is evidence-backed, not an alias assumption. Through the existing
SSH master, read-only `find/stat/sha256sum/sed` checks confirmed that the remote
canonical directory has the exact seven-member population and every member SHA
equals staged bytes. The mode-0400 attempt and log also have identical SHA and
content. No remote command executed Python or mutated state.

Authorization remains limited to hardware-rebind **release authoring**.
Production capture, hardware execution, speedup and energy remain unauthorized;
E2--E8 still require ep34-bound replay or recapture.
