# M2140 independent M2139 failure hammer

## Verdict

**The failure audit passes 100/100; review P0/P1/P2 = 0/0/0.  M2139 itself is
a consumed, noncitable failure.**  Its campaign findings are P0/P1/P2 =
1/1/0: the raw ordinary SAIF is inadmissible, and the preceding static source
review did not exercise native-SAIF late-enable state acquisition.

This review invoked no license query, VCS, `simv`, DC, PT/PTPX, ICC2, or GPU.
It authorizes source authoring only.  It does not authorize a retry or any EDA
execution.

## Chain and execution disposition

The one-member M2139 attempt and nine-member failure quarantine are exhaustive,
symlink-free, and double-sealed.  The M2137 contract and M2138 review identities
match exactly; M2138 scored 100 with P0/P1/P2 = 0/0/0 and authorized at most
1 license query, 1 compile, 2 serial simulations, 2 admitted SAIF files, and no
DC/PTPX or retry.

M2139 consumed 1 license query, 1 successful VCS compile, and the first ordinary
simulation.  That simulation wrote one raw SAIF, but the parser rejected it
before `saif_files` could be incremented.  TSBG never ran.  There is no canonical
result and no launch lock.  The ordinary functional path itself is intact:
20,292 cycles, 14,304 scalar reads, the frozen completion ledger, an exact
60,876-ns settled-negedge window, and no functional fatal.

## Exact TX fingerprint

The raw SAIF has the expected 93,971 records and zero conservation failures.
Only 223 records have nonzero TX, summing to 78,955 ns:

| family | records | exact TX pattern | TX sum (ns) |
|---|---:|---|---:|
| `row_live_q` | 191 | every `[0..3][0..47]` except `[0][0]`, each 397 ns | 75,827 |
| `cache_valid_q` | 3 | `[1..3]` = 397/817/1219 ns | 2,433 |
| `slot_valid_q` | 7 | `[1..7]` = 1/34/64/97/127/160/190 ns | 673 |
| `bridge_overflow` | 15 | `[1..15]`, each 1 ns | 15 |
| `rsp_shape_legal` | 7 | `[1..7]`, each 1 ns | 7 |

The parser is correct to reject these records.  The old mixed-edge duration bug
is not present, and changing the duration gate or conservation tolerance cannot
repair this artifact.

## Root cause

The evidence strongly supports a **late-enable native-SAIF observer-state gap**,
not an RTL functional failure:

1. M2018 explicitly resets every `row_live_q` and `cache_valid_q`; M803 resets
   every `slot_valid_q`.  The testbench applies five reset clocks and executes
   the complete 383-cycle preload before the activity boundary.
2. Public controls and valid-qualified payloads remain known at every settled
   sample, and the full functional ledger passes.
3. TX is confined to unpacked-array/internal-validity families.  The exact
   all-indices-except-current-index pattern and finite prefix durations show
   values becoming visible to the observer at their first later element update.
4. The UCLI sequence is `scope; run-to-first-stop; power -enable; run`.  Thus
   the native activity monitor did not observe reset or preload, which had
   already established those internal states.

This is high-confidence diagnosis, not yet causal proof.  M2125 checked public
signals, not the five internal families, at the first boundary.  Therefore this
artifact alone cannot formally exclude a validity-masked internal X.  That
residual uncertainty is exactly why there is no admitted SAIF or power claim.

## What M2125/M2126 missed

M2125's selfcheck invoked no EDA.  M2126 tested the parser with a synthetic
93,971-record file whose TX fields were zero by construction.  That proved the
fail-closed parser and mutation gates, but did not reproduce native VCS activity
acquisition after a late `power -enable`.  M2138 changed only the option-aware
SDF/UNIT_DELAY path guard and inherited this acquisition protocol unchanged.

## Minimum additive successor

Do not edit or retry M2139.  Under a fresh source identity and an independent
source hammer, first run one ordinary-axis acquisition preflight:

1. Establish the same DUT scope and issue `power -enable` **before** the first
   run, so the monitor observes reset and preload.
2. At the first settled-negedge stop, issue `power -reset` to clear the activity
   counters while retaining observed state; then run the exact 20,292-cycle
   window, disable, and report.
3. Add a nonintrusive boundary census for every element in the five TX families.
   It may observe only: no force, deposit, mask, initialization, or RTL edit.
4. Keep the same workload, ledgers, 3-ns clock, 60,876-ns duration, 93,971-record
   count, conservation, critical-activity, and all-record `TX=0` gates.

Do not assume that `power -reset` while monitoring has never been enabled is
enough; the small preflight must test the enable-before-reset/preload sequence.
Do not relax the parser, rewrite TX into T0/T1, or drop MDA/SV/internal records.
Only after this preflight passes may a fresh two-axis RTL campaign be authored;
mapped SAIF/PTPX remains a separate independently reviewed gate.

## Paper boundary

M2139 supplies no citable SAIF, RTL activity comparison, power, energy, mapped
activity, component/system speedup, or paper-ready PPA.  It also does **not**
show that TSBG costs more energy: the TSBG axis never ran.  Protected docs/359
remains unchanged at `dedde7ce...`.
