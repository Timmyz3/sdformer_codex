# M1682 independent M1681 source review

Status: `FAIL_M1682_M1681_DECODER_D0_EXECUTION_CLOSURE_SOURCE__NO_M1683_RELEASE__SUCCESSOR_REDUCER_TOPOLOGY_REPAIR_REQUIRED`.

Most of the M1681 execution closure is sound. The private path is statically reachable from exact future authority through a fresh namespace and immutable attempt to the canonical record, opened-FD/SHA-bound timestep snapshot, actual shard scheduler, sealed work tree and atomic result rename. All 8,700 shards map to 34,800 unique result/attempt/work/failure paths. Request aggregation is conserved; the earlier apparent duplicated request line was a display artifact and is not a finding.

The source rejects zero or negative cycles/requests, boolean requests, negative or inconsistent kind/byte ledgers, bad address/commit/chain/final-state digests, unsealed extra result files and result pycache. The reducer iterates all 8,700 ordinals and builds an ordered manifest chain.

One P1 remains. `verify_sealed_shard()` accepts a correctly sealed result and SHA-bound attempt without checking that the sibling work and failure namespaces are absent. The reducer calls this verifier directly, while `resume_state()` contains the stricter topology check but is not called. A synthetic `{result:true, attempt:true, work:false, failure:true}` shard was therefore accepted by the reducer path and rejected by resume. That ambiguity is incompatible with claiming `COMPLETE_8700_SEALED_SHARDS`.

The repair is small: enforce exact sibling topology for every shard inside `verify_sealed_shard` or the reducer and preserve this attack as a regression. Explicitly checking that the attempt is a regular 0400 file should be added at the same time. Until the successor passes a fresh review, M1683 release authoring and all shard execution remain unauthorized.
