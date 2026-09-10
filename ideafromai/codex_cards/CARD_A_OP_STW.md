# Codex Card A — C1* OP-STW predictor

**Paste this entire card into Codex.**

**Server tree (ismd-nemo):** `/home/zhumd/work/sdformer_c1c2star_grokbot/`  
**Box mirror:** `/workspace/sdformer_c1c2star_grokbot/`

## Hard rules
1. Work **only** under `/home/zhumd/work/sdformer_c1c2star_grokbot/` (ismd-nemo) or the box mirror `/workspace/sdformer_c1c2star_grokbot/` (or the copy you are given).  
2. **Do not modify, move, or overwrite** any files under `SDformer/hw_autoresearch_nts07/` or other existing Codex hardware. Reading those files is OK.  
3. Every new file must start with: `// GROKBOT NEW FILE -- iscas_ssh`  
4. Do not claim Motion-XOR as AEE algorithm novelty; optional signature input is address-key only.

## Goal
Implement a working **OP-STW** (optical-flow predictive spike-tile wake) module and a tiny directed TB.

## Files to fill (already skeletoned)
- `rtl_c1star/c1s_op_stw_predictor.sv`
- `tb_c1star/tb_c1s_op_stw_predictor.sv` (create)
- Update `rtl_c1star/c1s_top.sv` only if needed to instantiate predictor (keep top optional/thin)

## Spec
### Ports (suggested; you may refine but document in README)
```systemverilog
module c1s_op_stw_predictor #(
  parameter int N_TILE = 64,
  parameter int FLOW_W = 8,
  parameter int EVT_W  = 8,
  parameter logic signed [FLOW_W-1:0] TH_W = 8'sd2,
  parameter logic [EVT_W-1:0] TH_E = 8'd1
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic signed [FLOW_W-1:0]     flow_cur  [N_TILE],
  input  logic signed [FLOW_W-1:0]     flow_prev [N_TILE],
  input  logic        [EVT_W-1:0]      event_cnt [N_TILE],
  output logic        [N_TILE-1:0]     wake_bitmap,
  output logic                         wake_valid
);
```
### Behavior
- When `valid_i`: for each tile i  
  `wake[i] = (abs(flow_cur[i]-flow_prev[i]) > TH_W) || (event_cnt[i] > TH_E)`  
- `wake_valid` pulses/holds as you document (prefer registered 1-cycle after valid).  
- Reset clears wake.

### TB accept criteria
- Case1: all deltas 0, events 0 → wake all 0  
- Case2: one tile large delta → only that bit 1  
- Case3: one tile high event_cnt → that bit 1  
- Self-checking `$display` PASS/FAIL; finish with `$finish` and non-zero `$fatal` on fail  
- Runnable with iverilog or verilator

### Out of scope
PRRC, OGEC, exact capture, any C2*, any old-tree edits.

## Done means
1. Module + TB committed under isolated tree only  
2. `README` snippet in `rtl_c1star/README_GROKBOT.md` describing ports  
3. Simulation transcript shows PASS for 3 cases  
