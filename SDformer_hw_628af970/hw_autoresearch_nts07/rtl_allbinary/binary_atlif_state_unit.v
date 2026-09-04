`include "unibin_h60_pkg.vh"

module binary_atlif_state_unit #(
    parameter integer DATA_W = 16,
    parameter integer THRESH_W = 16,
    parameter integer LEAK_SHIFT_W = 4
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         clear,
    input  wire                         enable,
    input  wire signed [DATA_W-1:0]     input_current,
    input  wire signed [THRESH_W-1:0]   threshold,
    input  wire [LEAK_SHIFT_W-1:0]      leak_shift,
    input  wire                         soft_reset_en,
    output reg                          event_out,
    output reg signed [DATA_W-1:0]      mem_state,
    output wire signed [DATA_W-1:0]     mem_candidate
);
    wire signed [DATA_W-1:0] leak_term;
    wire signed [DATA_W-1:0] leaked_mem;
    wire signed [DATA_W-1:0] threshold_ext;
    wire                     fire;
    wire signed [DATA_W-1:0] soft_reset_mem;
    wire signed [DATA_W-1:0] hard_reset_mem;
    wire signed [DATA_W-1:0] next_mem;

    assign leak_term = (leak_shift == {LEAK_SHIFT_W{1'b0}}) ? {DATA_W{1'b0}} : (mem_state >>> leak_shift);
    assign leaked_mem = mem_state - leak_term;
    assign mem_candidate = leaked_mem + input_current;
    assign fire = (mem_candidate >= threshold);
    assign threshold_ext = {{(DATA_W-THRESH_W){threshold[THRESH_W-1]}}, threshold};
    assign soft_reset_mem = mem_candidate - threshold_ext;
    assign hard_reset_mem = {DATA_W{1'b0}};
    assign next_mem = fire ? (soft_reset_en ? soft_reset_mem : hard_reset_mem) : mem_candidate;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_state <= {DATA_W{1'b0}};
            event_out <= 1'b0;
        end else if (clear) begin
            mem_state <= {DATA_W{1'b0}};
            event_out <= 1'b0;
        end else if (enable) begin
            mem_state <= next_mem;
            event_out <= fire;
        end else begin
            event_out <= 1'b0;
        end
    end
endmodule
