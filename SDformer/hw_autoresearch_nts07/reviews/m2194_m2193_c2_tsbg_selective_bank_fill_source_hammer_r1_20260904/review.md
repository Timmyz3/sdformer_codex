# M2194 independent M2193 selective-bank source hammer

## Verdict

**FAIL, 92/100, P0/P1/P2 = 0/1/0. M2195 is not authorized.** The
selective-bank mechanism is structurally sound, but the sealed directed
verification contract is not complete: `commit_tag_o` and `commit_tag_t` are
wired yet never compared with a golden tag, and `commit_tag` is absent from
the SVA interface and commit-stall stability assertion.

## What passed

- The M2184 CPU quick-kill and M2193 author package have exhaustive double
  seals. All 11 frozen source and parent identities match; M2018, M803, and
  docs/359 are unchanged.
- Both half masks are the exact OR of the four loaded B4 contexts. A hit
  requires coverage of both unions; a partial hit issues only
  `needed & ~valid`, charges `popcount(request_mask)`, accepts the exact same
  response mask, writes only returned banks, and publishes bank validity only
  after slice 5.
- The unchanged M803 accepts arbitrary nonzero masks with source-count equality,
  independently backpressures banks, and assembles responses by slot/bank so
  bank return order is unrestricted.
- Ordinary and TSBG instantiate the same side wrapper, public interface,
  cache geometry, bank memory, backpressure functions, Acc24 datapath, and B4
  input. Their only parameter difference is static `SCHEDULE_MODE`.
- Private context Acc24, sign, product, context, tag storage, slice, and terminal
  state are present. The TB contains cold refill, missing-only partial refill,
  eviction, response reorder, request/bridge/commit stalls, zero descriptors,
  positive and negative sources, INT8 `-128`, exact Acc24, and exact request
  masks.
- The official source suite reproduces 12/12 semantic-mutation rejections,
  one parser control, 5/5 parser-mutation rejections, and three lexical
  SystemVerilog balance checks. No M2195 result, attempt, work, or lock exists.
  No VCS, simulator, EDA, license, or GPU action was run by this review.

## Blocking finding

The contract requires the directed test/SVA to verify
destination/tag/slice/terminal commit identity. Context and slice select the
golden Acc24 scoreboard and terminal has two assertions, but tag has neither:

- the TB never compares `commit_tag_o` or `commit_tag_t` to
  `24'h530000 + bundle_index*16 + commit_context`; and
- the SVA has no `commit_tag` input, so `ap_commit_header_hold` cannot ensure
  tag stability while `commit_valid && !commit_ready`.

Consequently a wrong or stall-unstable destination tag can still emit the
unique PASS token. Source inspection suggests the current RTL assignment is
correct; that does not satisfy the sealed directed-verification requirement.

## Required repair

Use a fresh additive source identity. Add golden tag comparisons for both
modes, add `commit_tag` to the SVA port and commit-header hold property, and
add static mutations that remove or corrupt each check. Reseal and request a
new independent source hammer. Keep M803, M2018, and docs/359 unchanged.

M2195 remains unauthorized; its execution budget is zero.
