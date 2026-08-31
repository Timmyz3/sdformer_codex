# M467R2 r1 failed pre-pass

This directory is retained as negative evidence.  The first exact-SHA run
deadlocked after a descriptor-write stall because the top consumed the M414
result before retaining `result_last`.  It produced no `RUN_COMPLETE.txt` and
admits no result.  The RTL was repaired by explicitly retaining the last flag;
the fresh r1b directory is the only candidate admission evidence.
