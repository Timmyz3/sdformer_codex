# Codex Card B — C2* HBG-RP packetizer

**Paste this entire card into Codex.**

**Server tree (ismd-nemo):** `/home/zhumd/work/sdformer_c1c2star_grokbot/`  
**Box mirror:** `/workspace/sdformer_c1c2star_grokbot/`

## Hard rules
1. Work **only** under `/home/zhumd/work/sdformer_c1c2star_grokbot/` (ismd-nemo) or the box mirror `/workspace/sdformer_c1c2star_grokbot/`.  
2. **Do not modify** any existing Codex hardware under `hw_autoresearch_nts07/`. Read-only OK.  
3. Header every new file: `// GROKBOT NEW FILE -- iscas_ssh`  
4. Follow `docs/ATLIF_contract_r1_grokbot.md` defaults (int8 amp, eps=1, **not** absorbable into W).

## Goal
Implement **HBG-RP**: hybrid binary-gate + real payload packetizer that turns ATLIF amplitude into `{g,p}` and a clock-enable style gate.

## Files
- `rtl_c2star/c2s_hbg_rp_packetizer.sv` (fill skeleton)
- `tb_c2star/tb_c2s_hbg_rp_packetizer.sv` (create)

## Spec
```systemverilog
module c2s_hbg_rp_packetizer #(
  parameter int AMP_W = 8,
  parameter logic signed [AMP_W-1:0] EPS = 8'sd1
) (
  input  logic                      clk,
  input  logic                      rst_n,
  input  logic                      amp_valid,
  input  logic signed [AMP_W-1:0]   amp,
  output logic                      g,           // gate
  output logic signed [AMP_W-1:0]   p,           // payload
  output logic                      gp_valid,
  output logic                      pe_clk_en    // == g when valid path active; 0 if !amp_valid
);
```
### Behavior
- `amp_abs = (amp < 0) ? -amp : amp` (careful with most-negative)  
- `g = amp_valid && (amp_abs > EPS)`  
- `p = g ? amp : '0`  
- `pe_clk_en = g`  
- `gp_valid` tracks registered handshake (document)

### TB cases
1. amp=0 → g=0,p=0  
2. amp=±1 with EPS=1 → g=0 (strict `>`)  
3. amp=±2 → g=1, p=amp  
4. amp_valid=0 → g=0, pe_clk_en=0  

### Out of scope
ADP-MAC, ARM-Acc, MFBD, SP-Gate, old TSBG edits.

## Done means
PASS self-checking TB; README ports updated in `rtl_c2star/README_GROKBOT.md`.
