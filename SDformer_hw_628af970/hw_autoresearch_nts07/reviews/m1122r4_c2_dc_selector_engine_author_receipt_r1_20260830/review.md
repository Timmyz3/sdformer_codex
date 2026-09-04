# M1122r4 C2 DC-selector engine source author receipt

Status: `PASS_M1122R4_DC_SELECTOR_ENGINE_SOURCE_AUTHOR_RECEIPT__M1123R4_REQUIRED__NO_EDA`

## Verdict

The additive M1122r4 engine source is ready only for a different-author M1123r4 engine hammer. This receipt does not authorize launcher authoring, an attempt, DC, mapped VCS, or any performance claim.

## Repair and preserved boundary

The only functional difference from M1112r3 is the DC invocation trust path. The engine must invoke the exact `/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell -f <pinned Tcl>` selector. Static checks require that path to be a symlink whose raw target is `snps_shell`, whose resolved wrapper has SHA256 `23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2`. Runtime checks must observe the same PID exec into `common_shell_exec`, verify SHA256 `bf91e6abfb9e2523c3c4884844117c629bef9dd83e2959934029a409118aa391`, and match the complete `-shell dc_shell -r ... -f ...` argv.

The RTL, TB, filelist, SDC, Tcl, libraries, mapped-VCS requirement, and the 13-counter / 337-bit / 22-predicate / 128-cycle observation contract remain rebound exactly. M1112r3 is permanently no-retry under the M1121 failure audit; M1122r4 uses a fresh one-shot namespace.

## Evidence

The author self-test passed 233 checks and rejected 26 directed mutations, including selector type/target drift, wrapper/backend hash bypass, backend argv drift, old namespace reuse, future-hash circularity, false performance admission, live sealed extras, and symlinked sealed evidence.

No EDA tool or production engine path was executed. `docs/359` remains SHA256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
