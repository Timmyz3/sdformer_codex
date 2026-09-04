# Independent source-only static hammer request: M529 / M528 DW1RW

Review mode must be read-only and source-only. Do not run VCS, iverilog, Verilator, DC, Formality, PT/PTPX, CPU production analysis, or GPU work. Do not modify any author source, contract, result, review, or `docs/359_DATE终局冻结_20260813.md`.

Freeze these identities first:

- author handoff JSON: `86dd591948fc7d09850110f280e2f884185e9368d2a0508a66b7a2ae4f119d5a`
- source contract: `03cbeabdf67d36f6489172e6839824b5b38435adf050ddea9ddff0a8a7a89f70`
- top: `c6a8258892c2b15ae5e8a181554c02ed62efb7d3318dcaac3b10c87df96df070`
- macro adapter: `8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783`
- macro binding plan: `db4075cb9d34323dcc8c9bb04e575104acb9cb97a819b7f0750ce4a2d3976983`
- SVA: `43f78d8ae2b6243becf3846a8dfa15698577c9f0fc9c2e5ee96fdc90afce1605`
- TB: `c0dbabe24b24beaacbc5ef5601f4578291f221166e2d2af29e84ccdcf89d1b33`
- runner: `cedb757c2bf9c08929bd9ddf9f5eed4069fa3488b29aff43ed46796a61f651ed`
- docs/359: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

The review must fail closed with P0/P1 on any of the following:

1. Placeholder/incomplete matcher, directory, ownership, live/written map, stable scanner, lookahead, queue, arithmetic, completion, or counter path.
2. Drift from M504 subset, equal-later exclusion, maximum-population/lowest-ID tie, stable population/row ordering, one-lookahead deadline rule, or no-consume-credit rule.
3. Any dead row losing psum/row completion, any live row losing its write, stale same-address RAW, read-before-written, response identity/order loss, reservation above two, or non-atomic overflow/format handling.
4. A signed12 non-synthetic source not being constrained to sign-extended INT8, or more than 16 source beats per row.
5. Combined PVRF, single-use store elision, concurrent 1R1W behavior, second lookahead, second architecture, hidden fallback, decoder/full-network scope, or synthesized parent register array.
6. Anything other than nine coherent `TS1N28HPCPHVTB128X128M4S` instances, any use of rows 64..127, any missing/overlapping slice, or divergent control/address.
7. VCS runner failure to verify/bind the private foundry slow `.v`; DC plan compiling that `.v` instead of linking the `.db`; Formality plan lacking nine matched macro blackboxes/cutpoints.
8. TB/SVA missing directed, reproducible random, backpressure, ping-pong, exact/partial parent, dead/live, forward, dual enqueue, queue-full, row 0/63, all-slice, wrong-parent, read-before-write, dirty-reserved, overflow, and stale-epoch intent.
9. Any run receipt, measured RTL speedup, PPA, energy, full-network, system-speedup, or paper-headline claim in the author package.

Expected independent output directory: `reviews/m529_m528_dead_write_only_1rw_source_static_hammer_r1_20260827`. A passing review must use schema `m529_m528_dead_write_only_1rw_source_static_hammer_v1`, status `PASS_M529_M528_DW1RW_SOURCE_STATIC_HAMMER`, `p0_count=0`, `p1_count=0`, bind the exact source-contract SHA above under `identity.source_contract_sha256`, and be double-sealed. Static PASS alone does not authorize VCS; root must create a separate double-sealed launch admission.

