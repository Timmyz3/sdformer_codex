`ifndef UNIBIN_H60_PKG_VH
`define UNIBIN_H60_PKG_VH

`define UBIN_HEAD_DIM        32
`define UBIN_MAX_TOKENS      162
`define UBIN_SCORE_W         16
`define UBIN_GATE_W          8
`define UBIN_DATA_W          8
`define UBIN_SCORE_FRAC      7
`define UBIN_ALPHA0_Q8       5
`define UBIN_MU_Q8_DEFAULT   8'd16   // 1/16 in Q0.8, matching deploy quantization.

`endif
