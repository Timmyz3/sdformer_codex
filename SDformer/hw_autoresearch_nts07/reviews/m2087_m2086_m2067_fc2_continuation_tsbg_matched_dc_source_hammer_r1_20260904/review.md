# M2087 independent source hammer: M2067 matched ordinary/TSBG DC

Date: 2026-09-04 16:20 CST  
Status: **PASS_M2087_SOURCE_AUTHORIZED_ONCE_PENDING_M2085**  
Score: **97/100**; P0/P1/P2 = **0/0/3**

## Scope and prohibition

This is a static, independent review of the M2086 contract and M2088 one-shot
runner.  It launched no Synopsys tool, license query, simulator, or GPU job.  A
negative test invoked the Python runner without authority variables; it exited
before owner/attempt creation and left no M2088 run namespace.

## Bound source identity

- runner: `0933de895ad10972c8b8e4556a8c26d0a7b9dec4bd26dc1fcf60f89833e3db34`
- contract: `6302946283b4f3e1dc59e7a8eff92741d8de05ee77c6dc81f1abc7a8d44bae88`
- filelist: `f5f661eb98e011c9e5f9922bf298eb91083e014869e714fdf1c1d8971d1b490d`
- DC Tcl: `c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe`
- SDC: `808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5`
- M803 / M2018 / M2067 RTL: `cd264021...` / `96fb3557...` /
  `75502745...`; the filelist contains exactly these three dependencies.
- protected docs/359: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

The runner also pins the Python, `dc_shell`, `lmutil`, slow library, and fast
library binaries/data.  All source and tool pins matched the files inspected.

## Findings

### 1. Owner, attempt, result, and failure race: pass

The nonblocking owner lock is acquired before `fresh_namespaces()` and before
the attempt directory is created.  The shared same-UID EDA queue is then held
across both axes.  Only the winner records an owner PID, random nonce, and
runner hash.  Failure publication additionally requires that exact owner
triple, refuses an existing result/failure, and uses no-replace publication.
Therefore a lock loser or pre-authority invocation cannot consume or publish
another process's attempt or failure.  A successful result also uses
no-replace publication.  The consumed attempt and `automatic_retry=false`
make the two-compile identity genuinely one-shot.

### 2. R9/M2085 authority chain: pass, execution-time precondition

Execution requires externally supplied exact hashes for the exhaustively
sealed R9 result and for the sealed M2085 review.  The runner re-hashes every
manifest member, rejects links and non-exhaustive directories, binds M2085's
reviewed `result.json`, manifest, and outer-seal hashes back to the live R9
directory, requires 960 workloads, and requires M2085's explicit
`m2088_two_axis_dc=true`.  It also rechecks all authority after taking the
locks and before each axis.  R9/M2085 had not yet been published at this source
review cut; therefore this review authorizes only the bound source and does
not certify or pre-approve an R9 result.  The runner cannot proceed without
that later sealed chain.

### 3. Matched elaboration and RTL dependencies: pass

The two iterations are exactly `ordinary, SCHEDULE_MODE=0` and `tsbg_b4,
SCHEDULE_MODE=1`.  `clean_env()` passes the value through
`ELAB_PARAMETERS=SCHEDULE_MODE=<0|1>`; the pinned Tcl converts this to the DC
elaboration parameter.  The M2067 wrapper propagates it directly to M2018.
M2018 permits only modes 0/1 and changes scheduler ordering inside the same
module; M803 and the retained Acc24 continuation logic are common.  The top
port declaration is parameter-independent, and the result parser also
requires equal public-port counts.

### 4. Same physical flow and one compile per axis: pass

Both axes share the same top, filelist, SSG target/FFG minimum libraries,
3.000 ns SDC, input/output assumptions, ideal pre-CTS clock, and ZeroWireload.
Only the schedule mode, output path, and an unused provenance label differ.
The pinned Tcl contains one `compile_ultra`, no incremental compile, no hold
optimization, and no false-path/case-analysis timing exception.  Each axis is
launched once; timeout or nonzero exit publishes failure and forbids retry.
The parser now requires terminal, flow, compile, timing, area, netlist/DDC/SVF,
port, and electrical-constraint artifacts.  It rejects setup/hold reports
without parseable slack, max-cap/transition/fanout violations, TIM-209/OPT-150,
and every non-whitelisted error/fatal line.  The sole allowed bootstrap error
is exact-position and exact-SHA matched.

### 5. Result arithmetic and claim boundary: pass

The result computes `R9 cycle ratio / mapped logic-area ratio`, records both
axis setup and diagnostic hold WNS, and applies the 2% area-tax and 1.15x
throughput-per-logic-area gates without rewriting a failing point.  It remains
`PENDING_INDEPENDENT_RESULT_HAMMER_DO_NOT_CITE`.  The contract/result explicitly
deny macro/chip area, hold closure, power, energy, real checkpoint weights,
full-FC wall time, system speedup, paper-ready PPA, and paper admission.

## Residual P2 observations

1. R9/M2085 were pending at review time; the hard execution gate, not this
   review, must establish their final identities.
2. Public-port equivalence is recorded as count equality rather than a
   name/direction digest.  The static top declaration is common and independent
   of `SCHEDULE_MODE`, so this is not a fairness defect for these sources.
3. A SIGKILL or host loss can leave a consumed attempt without a sealed failure;
   it cannot create a success or authorize a retry, but would require a new
   identity and forensic note.

## Authorization

Exactly one M2088 execution of this source identity is authorized, only after
the runner independently verifies the sealed R9 result, sealed M2085 result
review, this sealed M2087 review, resource gates, and absence of same-UID EDA.
Automatic retry is not authorized.  Any source/hash change, precondition
failure, namespace residue, or failed axis requires a new reviewed identity.

