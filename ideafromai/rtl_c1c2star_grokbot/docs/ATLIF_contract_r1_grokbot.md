# ATLIF output contract r1 (Grok Bot defaults)

**Status:** DEFAULT DRAFT — change anytime; C2* HBG-RP depends on this.  
**Author:** Grok Bot (`iscas_ssh`) 2026-09-05  
**Scope:** NEW co-design only. Does not modify any existing algo/RTL files.

## Defaults (assumed until you override)

| Field | Default | Notes |
|---|---|---|
| `amp` format | `logic signed [7:0]` (int8) | Q-format optional later |
| `eps` | `8'sd1` absolute | gate if `abs(amp) > eps` |
| Absorb into next W? | **NO** | payloads stay per-event; this makes HBG-RP meaningful |
| Gate `g` | hard threshold | `g = (amp_abs > eps)` |
| Payload `p` | `amp` truncated/sat to 8b when `g=1`, else 0 | |
| Soft gate? | no | keep HW simple for P1 |

## Packet
```text
typedef struct packed {
  logic        g;       // event gate / clock enable
  logic signed [7:0] p; // real-valued payload
} atlif_gp_t;
```

## Claim discipline
- If later you decide amps **are** absorbable into W, downgrade HBG-RP claim and say so in paper.
- Canonical NeurIPS'25 AT-LIF `{0,θ}` is **not** identical to this contract; document the difference.

## Override log
| Date | Who | Change |
|---|---|---|
| 2026-09-05 | grokbot | initial defaults |
