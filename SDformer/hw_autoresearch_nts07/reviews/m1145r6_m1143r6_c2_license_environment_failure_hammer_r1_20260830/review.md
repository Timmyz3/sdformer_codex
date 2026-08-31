# M1145R6 independent M1143R6 license-environment failure hammer

Verdict: **PASS; the sole M1143R6 execution stopped at VCS license acquisition because its clean child environment omitted both Synopsys license route variables. This is a launcher-environment failure before RTL/netlist compilation, not evidence of a VCS or mapped-netlist semantic defect.**

Exactly one sealed attempt and one sealed quarantine exist. The canonical result and unquarantined work are absent; automatic retry and DC rerun are false. The quarantine contains only `compile.log` plus failure/seal metadata: there is no `simv` and no `case0.log`, hence simulation invocations are zero. The attempt field reserving one case0 attempt must not be interpreted as an executed simulation.

The current shell has both `SNPSLMD_LICENSE_FILE` and `LM_LICENSE_FILE`; this review records only presence, byte length, and SHA-256, never route values. Frozen M1129 checks SNPSLMD first, falls back to LM, runs `lmstat`, and copies the caller environment. M1143 instead builds a five-key environment that contains neither route. It also sets `HOME=/tmp`, which violates the current runtime constraint against repurposing HOME.

Only additive successor **source authoring** is authorized. The successor must preflight and hash-bind the selected route (SNPSLMD first, LM fallback), insert that selected key/value into the child environment without logging or sealing the value, and omit HOME entirely. It must retain the frozen netlist, cell model, memory model, TB, command, no-SDF contract, and no-retry discipline. No direct retry, VCS/DC launch, mapped-functionality claim, or paper claim is authorized.
