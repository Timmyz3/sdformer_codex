# R26 implementation handoff

Ready for independent source/CPU review and root-controlled GPU launch. No new GPU/model inference, training, GT or quality was performed here. Six CPU checks passed with actual original MS_Spiking_Mlp/PSN at small channel widths and a78-BN fixture. They do not establish CUDA `_int_mm` support or optical-flow quality.

`bitplane_linear.py` implements the new full-space integer FC graph. `run_precision.py` reuses only the frozen R24 `source_preflight` and existing original-checkpoint loader/source/BN contracts. It never invokes the spatial compact primitive. Source/core/test/protocol are bound by `IMPLEMENTATION_FREEZE_V1.json`; independently accepted source bytes must be included in root's executable freeze. Do not mutate them after review.

Input schema: `flow_r26_window_precision_inputs_v1`, `BN_mode=ALL78_NATIVE_CURRENT`. Required assets: `primitive`, `data_helper`, `independent_admission`, `runtime`, `membership`, `h8`, `regions`, `checkpoint`, plus `old_root`. Each asset is `{path,sha256}`. Data helper is the frozen R24 runner, SHA4677fe4c1fb7f296ffea62887381ada7bcab46b8e641e14cbd8f31de86abde66. No `base_primitive` is consumed. Independent admission schema: `flow_r26_precision_independent_admission_v1`, statusADMIT, permitted_plans contains precision42, matching runner/core SHA. Root source-freeze schema: `flow_r26_window_precision_source_freeze_v1`, files list of actual path/SHA including runner, primitive and data helper.

Remote Python remains `/data/flow_a/envs/v2/bin/python`. CPU preflight does not create the run directory; it can create the preflight receipt's parent. Root substitutes actual frozen paths/SHA in:

```sh
/data/flow_a/envs/v2/bin/python -B /data/flow_a/code/research_20261017/window_precision_round26/implementation/run_precision.py \
  --inputs INPUTS --inputs-sha256 INPUTS_SHA \
  --freeze EXECUTION_FREEZE --freeze-sha256 FREEZE_SHA \
  --plan precision42 --wall-minutes 20 --output FRESH_RUN \
  --preflight-output FRESH_PREFLIGHT_JSON
```

Execution uses the same arguments with `--execute` (preflight-output may be omitted). It takes the existing common GPU0 nonblocking lock before CUDA initialization, then runs four declared actual integer-operator shape tests and the bounded42-forward plan. Backend failures stop without float fallback. Each exception writes an incomplete prefix result; no automatic next experiment. A complete result schema is `flow_r26_window_precision_v1`, graph `r26_W271482_all78current_s2b2_fullspatial_weightplanes_H8`, status `WINDOW_PRECISION42_COMPLETE`.

Artifacts: `results.json` with42 records,10 exact checks, full source/BN/state/work metadata;42 NPZ with314 raw arrays; `quantization.npz` with4 q8/scale arrays and a hash-bound `quantization.json`. Repeated forwards are explicitly not quality endpoints. Target logical rows, original Linear module calls, true integer dense contractions and bitplane work remain distinct. Extra CPU quantization, int8 plane caching, binary validation, hashing/capture/serialization and four backend tests are not free production hardware work.

CPU repeat (new receipt path):

```sh
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B test_precision_cpu.py NEW_RECEIPT.json
```

The earlier R21/R24 failures and R25 stop stay in force. A passing new execution gate is only permission for independent quality reduction; it is not flow-specific novelty or a measured cycle/energy improvement.
