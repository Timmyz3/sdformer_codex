`ifndef NTS07_PKG_VH
`define NTS07_PKG_VH

// ============================================================
// NTS-11bc / NTS-11bd Unified H60 Hardware Constants (DATE 2027)
// All encoder stages (S0-S3, 12 blocks) use H60 attention.
// Dual-mode ATLIF: ternary for Q/K/downsample, binary for all others.
// ============================================================

// --- Geometry ---
`define NTS07_HEAD_DIM           32     // Fixed head dimension across all stages
`define NTS07_MAX_TOKENS         98     // 2x7x7 windows = 98 tokens per window
`define NTS07_MAX_HEADS          24     // S3 uses 24 heads
`define NTS07_TIMESTEPS          10     // T=10 time bins

// --- Fixed-point widths ---
`define NTS07_ACT_W              16     // Activation (INT16 / Q0.15)
`define NTS07_WGT_W              8      // Weight (INT8)
`define NTS07_ACC_W              24     // MAC accumulator width
`define NTS07_SCORE_W            8      // Fused TX+SC score (Q4.3 signed, range -8..+7)
`define NTS07_GATE_W             8      // Shiftmax gate (Q0.7 unsigned, 0..255/256)
`define NTS07_THRESH_W           16     // Neuron threshold width
`define NTS07_MEMBRANE_W         18     // Membrane potential width (ATLIF u)

// --- H60 fusion parameters (checkpoint-frozen, Q0.8 fixed point) ---
// mu: SC weight, default ~0.05 → 13/256 ≈ 0.0508
`define NTS07_MU_Q8_DEFAULT      8'd13
// alpha0: same-zero bonus (TX) ~0.02 → 5/256 ≈ 0.0195
`define NTS07_ALPHA0_Q8_DEFAULT  8'd5
// beta: opposite sign penalty ~0.25 → 64/256 = 0.25
`define NTS07_BETA_Q8_DEFAULT    8'd64
// gamma: single-active penalty ~0.15 → 38/256 ≈ 0.148
`define NTS07_GAMMA_Q8_DEFAULT   8'd38

// --- ATLIF ternary spike encoding ---
// 2-bit packed: {sign_bit, fire_bit}
//   2'b00 = SILENT  (no spike)
//   2'b01 = POS     (+threshold)
//   2'b10 = NEG     (-threshold)  (ternary_en=1 only)
`define TERN_SILENT  2'b00
`define TERN_POS     2'b01
`define TERN_NEG     2'b10

// --- Neuron mode (per-layer descriptor bit) ---
`define ATLIF_MODE_BINARY   1'b0   // 1-bit output {0, +thresh}
`define ATLIF_MODE_TERNARY  1'b1   // 2-bit output {-thresh, 0, +thresh}

// --- Engine IDs ---
// NTS-11bc+: attention ALWAYS routes to ENG_H60 (no Legacy)
`define ENG_SPARSE_MAC  2'd0
`define ENG_H60         2'd1
`define ENG_RESERVED    2'd2   // Legacy QKFormer removed, do not synthesize
`define ENG_DENSE_MAC   2'd3

// --- Leak factor for ATLIF membrane (Q0.7) ---
// leak = 1 - 1/tau, default tau≈16 → leak≈15/16 = 240/256 Q0.8
`define NTS07_LEAK_Q8       8'd240
`define NTS07_RESET_VAL     18'd0

// --- Popcount tree parameters ---
`define NTS07_POPCOUNT_W    6      // ceil(log2(32)) = 6 bits for 32-dim popcount

// --- Shiftmax LUT depth ---
// Scores are Q4.3, offset by row_max (which is ≤ +7), so shifted ∈ [-15, 0]
// 2^x for x = 0..-15 needs 16 entries Q1.7 (0..255)
`define NTS07_SHIFT_LUT_DEPTH 16
`define NTS07_SHIFT_LUT_ADDR_W 4

// --- SRAM tile sizes (for documentation; actual RTL uses parameterized widths) ---
`define NTS07_QK_TILE_W     64     // 32ch × 2bit = 64 bits per token packed
`define NTS07_KVAL_TILE_W   512    // 32ch × 16bit = 512 bits per token for K_orig/V

`endif
