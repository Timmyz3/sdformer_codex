# M2138 independent M2137 source hammer

## Verdict

**PASS, 100/100; P0/P1/P2 = 0/0/0.**  This review authorizes exactly one
fresh M2139 diagnostic attempt under the budget in `review.json`.  The review
invoked no license query, VCS, `simv`, DC, PT, PTPX, ICC2, or GPU work.

The authorization is not a result.  Even a passing M2139 remains VCS-only RTL
SAIF diagnostics and requires a separate exhaustive M2140 result hammer.  It
cannot be cited as mapped activity, power, energy, component/system speedup,
or paper-ready PPA.

## Source identity and predecessor disposition

- M2137 runner SHA is
  `a1a72dcdfbbf0f1f0cbae52424b1dac08b023edd612223236f9c2fb77e7445d4`;
  contract SHA is
  `42d2394942f25e80a28b6b448ad966715366dc3d71ea60e5cf1899b07b89b2cd`.
- The 12-entry contract inventory is exhaustive for the M2137 delta and every
  entry is a regular, non-symlink file at its exact SHA.  The contract,
  M2137 selfcheck, M2126 source hammer, and M2128 failure hammer seals all
  verify.  `docs/359` remains at `dedde7ce...`.
- M2127 is still a consumed failure: its attempt directory exists, its
  canonical result and launch lock do not, retry is forbidden, and no M2127
  VCS/simulation/SAIF/power or paper claim exists.
- M2139 result, attempt, and lock namespaces were all absent during this
  review.

## Independent guard mutations

The harmless positive test put `/SDformer/` in all three dynamic pathname
roles: `-Mdir`, the `-f` operand, and `-o`.  It passed.  The exact current
filelist plus six active source files also passed the content scan.

All required negative classes failed closed:

1. Three explicit SDF option forms were rejected, including lower/upper-case
   `-sdf*` and `+sdf*` variants.
2. Three `+define+UNIT_DELAY` forms were rejected, including combined, valued,
   and case variants.
3. Four active-input mutations were rejected: source and filelist content,
   each with `$sdf_annotate` and with `UNIT_DELAY`.

Therefore the M2127 false positive is specifically repaired without weakening
the actual timing-contamination boundary.

## Inheritance and launch topology

The only semantic delta from frozen M2125 is the option-aware timing guard.
M2137 imports and exact-pins the M2125 runner and reuses its RTL, TB, parser,
filelist, and two UCLI scripts byte-for-byte.  The following gates remain on
the sole production path:

- workload slot 42; compile `+vcs+initreg+random`; per-axis runtime
  `+vcs+initreg+0`;
- one ordinary then one TSBG axis in one serial loop;
- phase-matched settled-negedge measurement windows;
- exact completion/cycle/read/product/commit ledgers;
- exactly 93,971 DUT-only SAIF records per axis, every TX exactly zero;
- per-record T0+T1+TX conservation and nonzero activity in every critical
  request/response/bridge/commit valid/accept cone.

Control-flow inspection establishes: M2138 review validation occurs first,
M2139 freshness second, collision screening third, then lock and consumed
attempt creation.  Only afterward may the single license query occur.  The
new timing guard runs before the VCS compile counter and launch.  There is one
shared compile and one strictly serial two-axis simulation loop.  No caller
can select another workload or axis, there is no reuse path, and the exception
path seals a quarantine with `automatic_retry=false`.

## Exact authorization

M2139 may consume once: one license query, one VCS compile, two serial `simv`
runs, and two SAIF files.  DC and PTPX counts are zero; PT/ICC2/GPU work is not
present.  There is no automatic retry and no reuse of old artifacts.

Any identity drift, freshness failure, mutation regression, execution failure,
or mismatch against the inherited functional/activity gates consumes or
blocks this authorization fail-closed.  Do not start M2139 from any other
runner or retry M2127.
