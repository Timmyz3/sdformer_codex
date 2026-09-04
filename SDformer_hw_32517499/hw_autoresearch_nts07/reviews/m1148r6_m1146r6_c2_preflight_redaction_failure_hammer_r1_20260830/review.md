# M1148R6 independent M1146R6 pre-attempt redaction hammer

Verdict: **PASS; M1146R6 rejects a valid license route before consuming an attempt because successful `lmstat` output legitimately echoes the selected route. This is a preflight false negative.**

Independent real probing returned `lmstat` rc=0 and confirmed that its raw stdout contains the selected SNPSLMD route. No raw bytes or route value are returned, logged, serialized, or sealed; only rc, byte count, route-presence boolean, and public route identity (variable, presence, length, SHA-256) are recorded. M1146 `source_preflight(True)` fails with the exact safe diagnostic `lmstat output echoed secret route`. Before and after probing, attempt, result, work, and failure counts are all zero; VCS/DC counts are zero.

Raw tool output may transiently exist in memory. The safety boundary is that it must never be returned or persisted. The minimal repair should discard raw output and decide only from return code; if diagnostics are retained, redact before any persistence. M1146 correctly omits HOME, and every successor must continue to forbid HOME reuse or override.

Only additive preflight-redaction repair **source authoring** is authorized. No attempt, launch, VCS, DC, retry, mapped-functionality claim, or paper claim is authorized.
