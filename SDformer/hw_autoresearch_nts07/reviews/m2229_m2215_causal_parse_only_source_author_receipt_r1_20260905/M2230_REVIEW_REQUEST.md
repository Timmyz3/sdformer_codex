# M2230 independent source review request

Please review the minimal additive M2229 parse-only successor. The author previously performed M2216's independent failure diagnosis; use another reviewer for this source milestone.

Source: `system_simulator/scripts/run_m2231_m2229_causal_parse_only_successor.py`.
Contract: `contracts/m2229_m2215_causal_parse_only_source_contract_r1_20260905.json`.
Tests: `tests/test_m2229_causal_parse_only_successor.py`.

The compiler banner now belongs to `vcs_compile.log`, the simulator banner to `simv.log`. Every existing directed ledger constraint remains checked. Exact SVA covers and narrow disposition of the 26 TB `context` warnings plus one known platform warning are also checked. Unknown warnings, malformed logs, duplicate tokens, changed counts or failed source authority are rejected. The script uses only the Python standard library and launches no process or EDA/license tool.

The raw quarantine/attempt seals, M2214/M2216 reviews, all M2213 sources and parser runtime are locked by the M2229 contract. Raw files remain read-only. Only a fresh M2231 attempt/result can be written after M2230 passes. A successful receipt is still pending M2232; raw directed cycles do not become population/mapped/energy claims.

Author checks passed: static identity validation; six test methods containing 27 rejected log mutations and three rejected source-authority cases. No M2231 attempt/result exists; source tests do not create production receipts.

Suggested read-only commands from `hw_autoresearch_nts07`:

```sh
/opt/anaconda3/bin/python3.12 -B system_simulator/scripts/run_m2231_m2229_causal_parse_only_successor.py --static
/opt/anaconda3/bin/python3.12 -B -m unittest discover -s tests -p 'test_m2229_causal_parse_only_successor.py' -v
```

If passed, write an exhaustive double-sealed review at `reviews/m2230_m2229_causal_parse_only_source_hammer_r1_20260905` with the schema fields required by the contract: status, score, p0/p1 severity, `identity.source_contract_sha256`, `identity.parser_runner_sha256`, and the exact authorization dictionary. Do not run the production parser as part of source review.

After independent PASS, root can execute exactly once:

```sh
/opt/anaconda3/bin/python3.12 -B system_simulator/scripts/run_m2231_m2229_causal_parse_only_successor.py --execute --contract-sha256 a11fcae7cae626cf29c4ebf5ccce712244be2b8174ab9519ad4b0cf9c7fbdeb2 --source-review-sha256 REVIEW_JSON_SHA256
```

The original M2215 consumed identity stays failed; M2231 reports new CPU postprocessing of those unchanged raw logs. No VCS rerun is needed or authorized.
