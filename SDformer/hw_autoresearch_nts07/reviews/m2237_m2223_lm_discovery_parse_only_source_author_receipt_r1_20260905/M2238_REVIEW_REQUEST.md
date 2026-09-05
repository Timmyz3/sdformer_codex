# M2238 source review: CPU-only LM discovery recovery

Review the additive `check_m2237_m2223_lm_discovery_parse_only.py`, its `run_m2239_m2237_lm_discovery_parse_only.py` runner, and the M2237 contract. M2237/M2238/M2239/M2240 had no artifact-name collisions at source start.

The checker is a directly reviewable copy of frozen M2223. `diff -u` shows the two validation changes: fatal markers are anchored to actual runtime lines; recorded cwd/home and execution paths must equal the authenticated PID-3569314 staging identity while physical files are checked at the exact sealed quarantine. All command/option state, set/readback, return-code, no-side-effect, output-manifest, census and repository-snapshot checks remain intact. Output writes moved into the separately gated runner; the checker itself only returns a dictionary.

M2224 authorized this recovery. Its review, M2222 source review, raw quarantine and old consumed attempt are exhaustively double-sealed and hash pinned. The old source contract and all six original source files are pinned as well. The new runner imports exactly one custom checker, also hash pinned. It contains no subprocess or EDA call.

Five test methods passed, including 19 negative cases: runtime fatal (plain and indented), duplicate/missing runtime markers, wrong cwd/execution mapping, wrong quarantine, nonzero return code, invalid option state, false exact readback, side-effect marker, changed census, snapshot and manifest. No M2239 attempt/result was created.

Read-only checks from `hw_autoresearch_nts07`:

```sh
/opt/anaconda3/bin/python3.12 -B system_simulator/scripts/run_m2239_m2237_lm_discovery_parse_only.py --static
/opt/anaconda3/bin/python3.12 -B -m unittest discover -s tests -p 'test_m2237_lm_discovery_parse_only.py' -v
diff -u system_simulator/scripts/check_m2223_lm_command_option_discovery.py system_simulator/scripts/check_m2237_m2223_lm_discovery_parse_only.py
```

If passed, create an exhaustive double-sealed review at `reviews/m2238_m2237_lm_discovery_parse_only_source_hammer_r1_20260905`. The required review JSON fields are specified in the M2237 contract. In particular, the runner expects exactly three `identity` keys: `source_contract_sha256`, `checker_sha256`, `runner_sha256`; the execution budget is one CPU parse, zero LM/license/EDA/GPU, no automatic retry. Source review itself should not consume M2239.

After PASS, root may execute once:

```sh
/opt/anaconda3/bin/python3.12 -B system_simulator/scripts/run_m2239_m2237_lm_discovery_parse_only.py --execute --contract-sha256 4dafe1df8e987d44b0fe74e71efd79a881cf7b7be98e89c8f29a7467d250928e --source-review-sha256 REVIEW_JSON_SHA256
```

Keep M2223 failed. The new receipt is command/option discovery evidence pending M2240 result review. It says nothing about successful conversion, library compatibility, NDM creation, placement/routing or paper PPA. M2224's before/after-census limitation also remains; a parse-only receipt cannot add continuous process observation.
