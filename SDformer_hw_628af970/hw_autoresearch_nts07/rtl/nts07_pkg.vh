`ifndef NTS07_PKG_VH
`define NTS07_PKG_VH

// NTS-07b H60 hardware constants
`define NTS07_HEAD_DIM        32
`define NTS07_MAX_TOKENS      98
`define NTS07_SCORE_W         6
`define NTS07_GATE_W          8
`define NTS07_MU_Q8_DEFAULT   8'd13   // 13/256 ~= 0.0508

// ATLIF spike encoding: 2'b00 silent, 2'b01 positive, 2'b10 negative
`define TERN_SILENT  2'b00
`define TERN_POS     2'b01
`define TERN_NEG     2'b10

// Per-layer neuron_mode in descriptor bit[127] (APB shadow: NEURON_MODE_LUT)
`define ATLIF_MODE_BINARY  1'b0
`define ATLIF_MODE_TERNARY 1'b1

// Engine IDs for controller routing (NTS-11bc+: attn always ENG_H60 on stages 0..3)
`define ENG_SPARSE_MAC  2'd0
`define ENG_H60         2'd1
`define ENG_LEGACY_QK   2'd2   // deprecated — not synthesized in unified-H60 flow
`define ENG_DENSE_MAC   2'd3

`endif