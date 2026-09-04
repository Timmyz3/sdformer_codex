# M1486 different-author blind hammer — M1485 nested M1233 compatibility

Verdict: **PASS**, 100/100, P0=0, P1=0.

The native M1485 source self-check passes, and the frozen M1475/M1480 plus
M1485 author tests pass 43/43. The independent campaign passes 24 checks and
rejects 65/65 mutations with zero false negatives.

The hammer confirms the exact runtime failure and repair boundary. M1319 first
checks the selected configuration through `exact_extended_identity`, then its
distinct frozen-M1233 module checks it again through `exact_identity`. M1485
targets both runtime-chain objects, only for the exact `selected configuration`
label. Checkpoint and profile still call the original frozen verifier.

The compatibility remains content-exact and type-exact. Missing or extra keys,
equal-valued float substitutions, frozen-value drift, wrong or relative paths,
missing files, symlinks, nonregular files, size/SHA drift, and a hash-time stat
race all fail closed. Preinstalled hooks and re-entry are rejected. Both hooks
are unconditionally restored after normal return, an exception, and detected
inner tampering.

This review authorizes only M1487 release authoring. It used and authorizes no
SSH, remote preflight, GPU query, launch, capture, attempt consumption,
controller operation, retry, EDA, performance claim, or paper headline.
