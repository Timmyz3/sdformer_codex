`timescale 1ns/1ps
`default_nettype none

module hitflow_fanout_event_buffer #(
    parameter int DATA_W = 32,
    parameter int TAG_W  = 48
) (
    input  logic              clk_core,
    input  logic              rst_core,
    input  logic              in_valid,
    output logic              in_ready,
    input  logic [DATA_W-1:0] in_data,
    input  logic [TAG_W-1:0]  in_tag,
    output logic              out_q_valid,
    input  logic              out_q_ready,
    output logic [DATA_W-1:0] out_q_data,
    output logic [TAG_W-1:0]  out_q_tag,
    output logic              out_k_valid,
    input  logic              out_k_ready,
    output logic [DATA_W-1:0] out_k_data,
    output logic [TAG_W-1:0]  out_k_tag
);

    logic              valid_q;
    logic              pending_q_q;
    logic              pending_k_q;
    logic [DATA_W-1:0] data_q;
    logic [TAG_W-1:0]  tag_q;
    logic              q_fire;
    logic              k_fire;
    logic              entry_done;

    assign out_q_valid = valid_q & pending_q_q;
    assign out_k_valid = valid_q & pending_k_q;
    assign out_q_data = data_q;
    assign out_k_data = data_q;
    assign out_q_tag = tag_q;
    assign out_k_tag = tag_q;
    assign q_fire = out_q_valid & out_q_ready;
    assign k_fire = out_k_valid & out_k_ready;
    assign entry_done = valid_q & (~pending_q_q | q_fire) &
                        (~pending_k_q | k_fire);
    assign in_ready = ~valid_q | entry_done;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            valid_q     <= 1'b0;
            pending_q_q <= 1'b0;
            pending_k_q <= 1'b0;
            data_q      <= '0;
            tag_q       <= '0;
        end else begin
            if (in_valid & in_ready) begin
                valid_q     <= 1'b1;
                pending_q_q <= 1'b1;
                pending_k_q <= 1'b1;
                data_q      <= in_data;
                tag_q       <= in_tag;
            end else if (valid_q) begin
                if (q_fire) begin
                    pending_q_q <= 1'b0;
                end
                if (k_fire) begin
                    pending_k_q <= 1'b0;
                end
                if (entry_done) begin
                    valid_q <= 1'b0;
                end
            end
        end
    end

endmodule

`default_nettype wire
