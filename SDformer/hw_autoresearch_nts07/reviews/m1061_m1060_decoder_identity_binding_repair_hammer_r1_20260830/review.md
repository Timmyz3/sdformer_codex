# M1061 independent receipt-blind hammer: STOP M1062

Verdict: `FAIL_M1061_M1060_BOOL_INTEGER_EXACT_SCHEMA_ESCAPE__STOP_M1062` (88/100, one fail-closed P1).

M1060 repairs the M1053 P0. The hammer rejected a wholly forged attempt/root/manifest/packed-SHA receipt, nonexistent selected paths, selected-record relabel plus rehash, raw-layer relabeling, and an attacker-refreshed double seal at publish. The frozen M699 root manifest, payload manifest and outer seal match exactly. D0/D2/D3 are each double-anchored by an identical manifest record and sealed-member-list path/SHA. Assemble and publish both rebuild the canonical attempt/runner/contract context and revalidate payload, raw and result.

The release still stops on exact-schema discipline. `validate_payload_receipt` uses whole-dictionary equality without explicit boolean type checks. Python therefore accepts integer `1` for `payload_members_verified` and `post_attempt`, and integer `0` for `paper_citable`. These values would also pass the assemble/publish path. This is not a SHA/path rebinding or a `paper_citable=true` escape, but it contradicts the recursive exact-JSON-schema contract and the requested bool attack replay. Add explicit boolean checks before equality and rehammer.

No real M1062 attempt, payload member, decoder cycle, EDA, GPU or remote job was executed. Pre-attempt `calls/*` access remained 0 open / 0 stat / 0 hash. A synthetic post-attempt identity failure retained the attempt and quarantined work. `docs/359` remains at `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
