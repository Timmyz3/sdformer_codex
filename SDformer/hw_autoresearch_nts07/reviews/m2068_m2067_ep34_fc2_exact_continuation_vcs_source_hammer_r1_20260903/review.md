# M2068 M2067 FC2 exact-continuation VCS source hammer

## Verdict

**FAIL, 78/100, P0/P1/P2 = 1/2/3. No VCS authorization.**

The architectural core is coherent, but the reviewed R1 one-shot cannot be
released. Its filelist/cwd combination exposes the sole no-retry attempt to a
deterministic compile failure; its failure path does not preserve evidence;
and its only negative alias attack covers G96, not G192.

During this review the author began an R2 correction in the same working
paths. The current files therefore no longer match the R1 hashes in
`input_manifest.sha256`. This review is a sealed rejection of that observed R1
identity, not an approval of the moving R2. R2 requires its own frozen identity
and M2070 independent source hammer.

## What is already sound

### Frozen population

The fixture contains 960 unique `(sample, layer, token-role)` workloads:

- 40 samples and four DSEC sequences, 240 workloads per sequence;
- eight FC2 layers, 120 workloads per layer;
- six G96 layers / 720 workloads and two G192 layers / 240 workloads;
- G96 uses four output tiles and bases `0,48`; G192 uses eight output tiles
  and bases `0,48,96,144`;
- 115,200 final commits and 1,843,200 per-axis integer checks.

The fixture has 212,468 nonzero source codes and no negative source code. The
weights remain directed INT8, not checkpoint weights. Thus a future PASS is a
real ep34 activity/sign component workload with directed weights, not a full
FC wall-time or system result.

### Global addressing and mutation sensitivity

The wrapper statically enforces G96/G192, 2/4 chunks and
`global_group_base = 48*chunk_index`. It exports:

`source_channel = global_group_base*16 + local_source_channel`

and

`weight_row_index = output_tile*source_group_count + global_group`.

The frozen maxima are channel 3071 and row 1535, so the 12-bit fields suffice.
An independent alias mutation that reuses chunk-zero weights is detected by
560/720 G96 and 183/240 G192 workloads, producing 623,168 and 408,576 wrong
committed values respectively under the M2067 directed-weight definition.

### Acc24 continuation and commit visibility

The retained array is exactly 4 contexts x 6 slices x 16 lanes x signed
Acc24. W_RESET_ASSERT resets only the inner physical engine, not this retained
state. Non-final `inner_commit_accept` events accumulate into the array;
external `commit_valid` and `bundle_done_valid` are gated by the final chunk.
The final view sums retained and current inner Acc24 through 25 bits and checks
overflow. The analytic magnitude bound is `192*16*128 = 393216 < 2^23`.

### Logical-cycle boundary

The counter excludes passive W_HEADER wait, including peer synchronization
after an axis finishes a chunk. It counts the accepted header and every
W_RESET_ASSERT, W_RESET_RELEASE, W_LOAD and W_RUN cycle. The first header sets
the count to one and later headers increment it. Both axes have the same
physical G48 shape, descriptor stream, reset states and final commit count;
their scheduling difference remains inside W_RUN.

These are good reasons to repair and re-review the source. They are not an
excuse to launch the reviewed R1.

## Blocking findings

### P0-1: filelist cannot resolve from the compile cwd

All seven R1 filelist entries begin with `hw_autoresearch_nts07/`. The runner
invokes VCS with cwd set to a newly created private directory below
`hw_autoresearch_nts07/results`. Under `-f` working-directory resolution those
entries become `WORK/hw_autoresearch_nts07/...`, where none exists. The one
allowed compile therefore fails after consuming the attempt.

The successor must use absolute filelist entries or otherwise prove path
resolution independently of WORK. A negative mutation replacing one absolute
entry with an R1-style relative entry should fail before attempt creation.

### P1-1: no-retry failure evidence is incomplete

R1 creates and seals ATTEMPT before lmstat, which is correct. But its exception
handler publishes FAILURE only if WORK exists. An lmstat failure therefore
leaves a consumed attempt with no failure quarantine. Later failures seal only
a generic error JSON: no phase/count state, compile log, completed slot logs or
raw-tree fingerprint survives in the quarantine.

The successor must publish, after every post-attempt failure, a no-replace
double-sealed quarantine carrying phase, command, license/compile/sim counts,
current/completed slot, the relevant regular logs and a complete file/symlink
fingerprint. Automatic retry remains forbidden.

### P1-2: G192 negative alias attack is absent

R1 runs one attack per simulation: G96 chunk1 with base0 instead of base48.
There is no G192 chunk2/chunk3 mutation for bases96/144. Real G192 workloads
provide strong positive address and oracle coverage, but positive coverage is
not a fail-closed negative attack. The successor needs at least one G192
wrong-base header after a legal prefix, with independent per-axis rejection
and reset recovery.

## P2 findings

1. The parser counts unique row/chunk records but does not require the exact
   output-tile Cartesian set or bound each tile ID.
2. Directed response weights do not vary with output tile. The separately
   checked row-index port covers the address equation, but arithmetic values
   cannot independently expose output-tile aliasing.
3. The pending result JSON lacks direct runner/parser/contract/review/attempt
   identity fields; a result hammer would have to reconstruct that chain.

## Release decision

No M2067 R1 command or SHA authority pins are provided. No VCS, EDA, GPU or
license query was launched by this review. `docs/359` remains unchanged at
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
