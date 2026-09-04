# M2198 independent M2197 source-hammer request

Review only the additive M2197 commit-tag validation repair. Do not run
`lmutil`, VCS, `simv`, any EDA tool, or a GPU job, and do not create any M2199
attempt, result, work, or lock state.

First verify the exhaustive seals and confirm that M2193 RTL/SVA/TB, M803,
M2018, the M2194 failure package, and docs/359 remain unchanged. Then review
the new M2197 SVA and TB against the single P1 finding:

- `commit_tag` must be an SVA input and must be stable with context, slice,
  terminal, and the Acc24 payload throughout commit backpressure;
- ordinary and TSBG must each have a four-entry golden-tag array;
- tags must equal `24'h530000 + bundle_index*16 + context`, so all four
  context tags differ and all three bundle ranges differ;
- each of 72 accepted commits per mode must explicitly match its next golden
  context and slice, the golden tag indexed by that context, terminal iff
  slice five, and all sixteen golden Acc24 lanes;
- all ten validation mutations, including deleted tag comparisons, deleted
  tag stall protection, and context-free golden-tag mappings, must fail;
- the parser must require 72 identity checks per mode; the future runner must
  retain exactly one license query, one VCS compile, one simv run, no other
  EDA, and no retry.

Authorize M2199 only at score >=95 and P0/P1/P2=0/0/0 with exact status
`PASS_M2198_M2197_SOURCE_HAMMER__M2199_ONE_SHOT_VCS_AUTHORIZED`. M2199 then
has the single-run budget above. M2200 must independently hammer the raw
result before any directed-VCS verification claim. No performance, PPA,
energy, or paper claim is authorized by this source chain.

