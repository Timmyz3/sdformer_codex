# M848/C2 R23 source-author handoff

M848 is ready for one fresh independent **source** hammer. It is not a VCS result and authorizes no release or launch.

The hardware and test boundary is unchanged from M837 R22. The extracted `compile_and_run` function is byte-identical (`b6f6753b...`), and the complete attack/equal-bandwidth invocation plus PASS-gate block is byte-identical (`261d47f0...`). Exact cycles remain `51/53,131/133,486/499,1231/1246,14/14`.

The only functional delta is artifact publication. VCS work remains private and may contain normal tool-generated symlinks. Fifteen exact evidence/control files are reopened with `O_NOFOLLOW`, required to be regular, checked for device/inode/size/SHA stability before and after copying, copied into a new private result stage, checked for zero symlinks/extras, double-sealed, and atomically published with no replacement.

The consumed M837 attempt is bound through the sealed M846 PASS100 failure classification. It is not reused, renamed, salvaged, or post-sealed.

Five Python 3.6 tests passed. The exact source dry run stopped with rc 86 before license/VCS and created no formal M848 attempt/result/quarantine. `docs/359` remains unchanged.
