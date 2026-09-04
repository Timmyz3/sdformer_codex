# M1259 independent hammer of the M1257 binder successor

Verdict: **PASS for the source release successor, not for execution now**.  M1257 closes all four M1255 findings while preserving the sealed-child and one-shot controls.  Its future one-shot may be authorized only after all four real strict-valid825 artifacts exist and fresh output/attempt/log namespaces are confirmed.

The exact M1257 author suite passed 14/14, the M1253 predecessor regression passed 13/13, and this independent suite passed 12/12.  All tests use temporary fixtures.  No production binder, remote host, GPU, valid825, VCS, DC, PTPX or other EDA action ran.

## Closure evidence

- **F1 — full `st_mode` binding:** mode is published and exactly compared for the new manifest, every candidate checkpoint/config/profile, the selected projection, and the selected sidecar.  Coordinated missing-mode, bool-mode, changed-mode and extra-mode attacks were resealed and rejected.
- **F2 — closed result root:** the exact eight-key result root rejects both false and positive claim keys and an alias key after valid receipt resealing.
- **F3 — closed nested schemas/types:** candidate rows, identity/profile/activity/metric maps reject nested false and positive extras.  Candidate epoch bool confusion is rejected.
- **F4 — source-pinned E0-E8:** all nine rows are exact and ordered.  The hammer changed each E0 through E8 target in both the result and sidecar together; every coordinated splice was rejected.  Swapped rows, a dropped row, and a false extra row key were also rejected.

## Controls preserved

Preparation still snapshots eleven inputs and creates exactly three execution memfds.  Every memfd has WRITE/GROW/SHRINK/SEAL locks, and the child receives exactly those three descriptors through `pass_fds`.  Candidate pair/order, minimum finite nonnegative AEE, lowest-epoch tie break, and selected projection are independently recomputed.  The attempt is created with `O_EXCL` before the child; a failed child consumes the attempt and log, and a second call cannot retry.

## Authority boundary

M1259 accepts M1257 as the future release successor.  It does **not** confirm that the four production valid825 artifacts exist and does **not** authorize execution now.  One future production execution is conditional on all four real strict-valid825 inputs plus fresh result/attempt/log namespaces.  No checkpoint is selected, no hardware rebind is authorized, and no speed, power, energy, or paper metric follows from this review.

`docs/359` remains frozen at `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
