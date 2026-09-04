# M2204 independent M2203 native-RTL SAIF result hammer

## Verdict

**PASS, 99/100, P0/P1/P2 = 0/0/0.** M2203 establishes a fresh native
RTL SAIF acquisition preflight for the ordinary axis. It authorizes only
source authoring and independent review for a future matched ordinary/TSBG
SAIF/PTPX chain. It does not directly authorize VCS, DC, PTPX, ICC2, or any
other EDA run.

## Independently reproduced evidence

- The result has exactly one consumed attempt, one license query, one VCS
  compile, one `simv` run, and two raw SAIF files. There was no retry, second
  axis, DC, PTPX, ICC2, or GPU run.
- The unique runtime window is 20,292 cycles and independently yields 149
  rows, 1,278 issues, 29,472 products, 24 commits, 1,788 bundles, and 14,304
  scalar weight reads, with an exact arithmetic scoreboard.
- An independent streaming S-expression parser—not the production parser—saw
  exactly 93,971 activity records inside the single `dut_ordinary` subtree
  and zero outside it in each SAIF.
- Diagnostic prehistory has duration 1167.01 ns. All T0/T1/TX/TC fields are
  integers; every T0+T1+TX sum is exactly floor(1167.01)=1167. The uniform
  residual is 0.01 tick, strictly below one tick. Its 45 nonzero TX records
  are accepted only in this diagnostic file, which is never annotatable.
- Measurement duration is exactly 60,876 ns. Every one of 93,971 records
  conserves duration exactly, TX is zero everywhere, 76,264 records toggle,
  and all eight bridge/commit/memory valid/accept activity classes are nonzero.
- Raw file sidecars, outer seals, the 16-member result seal, one-member attempt
  seal, M2202 source review, M2188 failure lineage, and the old M2187 failure
  package all pass exhaustive verification. Both M2203 SAIF hashes differ from
  M2187, so the failed raw files were not reused. The only M2203 filesystem
  identities are the canonical result and consumed attempt; no work, stage,
  lock, failure, or retry identity remains.

## Claim boundary

This admits native RTL SAIF acquisition and one measurement-SAIF candidate
only. It is not mapped-netlist activity, power, energy, PPA, component or
system speedup, paper evidence, or paper-ready PPA. The diagnostic prehistory
must never be annotated.

The permitted next action is a new matched ordinary/TSBG SAIF/PTPX **source
package**, followed by an independent source hammer. This review itself ran
no VCS, simulator, license, EDA, GPU, or Git action and did not modify sources.
