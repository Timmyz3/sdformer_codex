`include "nts07_pkg.vh"

// Unified ATLIF-PSN inference encoder: one comparator tree, ternary_en selects output width.
// Software: ATLIFTernaryPSN(output_mode="ternary"|"binary") — same threshold update, different surrogate.
// ternary_en=1 → 2-bit {-1,0,+1}; ternary_en=0 → binary {0,+1} (neg axis forced silent).
module atlif_unified_encode_unit #(
    parameter integer DATA_W   = 16,
    parameter integer THRESH_W = 16
)(
    input  wire                         ternary_en,
    input  wire signed [DATA_W-1:0]     activation,
    input  wire signed [THRESH_W-1:0]   pos_thresh,
    input  wire signed [THRESH_W-1:0]   neg_thresh,
    output wire [1:0]                   spike_out,      // packed ternary lane for H60 / DMA
    output wire                         binary_out      // LSB for Sparse MAC (ternary_en=0)
);
    wire pos_fire = (activation >= pos_thresh);
    wire neg_fire = ternary_en & (activation <= neg_thresh);

    assign spike_out  = pos_fire ? `TERN_POS :
                        neg_fire ? `TERN_NEG : `TERN_SILENT;
    assign binary_out = pos_fire;
endmodule