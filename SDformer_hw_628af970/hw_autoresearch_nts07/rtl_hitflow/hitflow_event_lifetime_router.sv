`timescale 1ns/1ps
`default_nettype none

module hitflow_event_lifetime_router #(
    parameter int DATA_W    = 32,
    parameter int TAG_W     = 48,
    parameter int COUNTER_W = 64
) (
    input  logic                  clk_core,
    input  logic                  rst_core,
    input  logic                  in_valid,
    output logic                  in_ready,
    input  logic [1:0]            in_route,
    input  logic [1:0]            in_pair_slot,
    input  logic [DATA_W-1:0]     in_data,
    input  logic [TAG_W-1:0]      in_tag,
    output logic                  single_valid,
    input  logic                  single_ready,
    output logic [DATA_W-1:0]     single_data,
    output logic [TAG_W-1:0]      single_tag,
    output logic                  fanout_q_valid,
    input  logic                  fanout_q_ready,
    output logic [DATA_W-1:0]     fanout_q_data,
    output logic [TAG_W-1:0]      fanout_q_tag,
    output logic                  fanout_k_valid,
    input  logic                  fanout_k_ready,
    output logic [DATA_W-1:0]     fanout_k_data,
    output logic [TAG_W-1:0]      fanout_k_tag,
    output logic                  pair_valid,
    input  logic                  pair_ready,
    output logic [(4*DATA_W)-1:0] pair_data,
    output logic [TAG_W-1:0]      pair_tag,
    output logic                  pair_tag_mismatch,
    output logic                  pair_duplicate_slot,
    output logic                  route_unsupported,
    output logic [COUNTER_W-1:0]  count_accepted,
    output logic [COUNTER_W-1:0]  count_single_forwarded,
    output logic [COUNTER_W-1:0]  count_fanout_q,
    output logic [COUNTER_W-1:0]  count_fanout_k,
    output logic [COUNTER_W-1:0]  count_pair_issued
);

    localparam logic [1:0] ROUTE_SINGLE = 2'd0;
    localparam logic [1:0] ROUTE_FANOUT = 2'd1;
    localparam logic [1:0] ROUTE_PAIR   = 2'd2;

    logic single_in_ready;
    logic fanout_in_ready;
    logic pair_in_ready;
    logic route_supported;

    always_comb begin
        in_ready = 1'b0;
        route_supported = 1'b1;
        unique case (in_route)
            ROUTE_SINGLE: in_ready = single_in_ready;
            ROUTE_FANOUT: in_ready = fanout_in_ready;
            ROUTE_PAIR:   in_ready = pair_in_ready;
            default: begin
                in_ready = 1'b0;
                route_supported = 1'b0;
            end
        endcase
    end

    assign route_unsupported = in_valid & ~route_supported;

    hitflow_single_event_buffer #(
        .DATA_W(DATA_W),
        .TAG_W(TAG_W)
    ) u_single_buffer (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(in_valid & route_supported & (in_route == ROUTE_SINGLE)),
        .in_ready(single_in_ready),
        .in_data(in_data),
        .in_tag(in_tag),
        .out_valid(single_valid),
        .out_ready(single_ready),
        .out_data(single_data),
        .out_tag(single_tag)
    );

    hitflow_fanout_event_buffer #(
        .DATA_W(DATA_W),
        .TAG_W(TAG_W)
    ) u_fanout_buffer (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(in_valid & route_supported & (in_route == ROUTE_FANOUT)),
        .in_ready(fanout_in_ready),
        .in_data(in_data),
        .in_tag(in_tag),
        .out_q_valid(fanout_q_valid),
        .out_q_ready(fanout_q_ready),
        .out_q_data(fanout_q_data),
        .out_q_tag(fanout_q_tag),
        .out_k_valid(fanout_k_valid),
        .out_k_ready(fanout_k_ready),
        .out_k_data(fanout_k_data),
        .out_k_tag(fanout_k_tag)
    );

    hitflow_qk_pair_assembler #(
        .DATA_W(DATA_W),
        .TAG_W(TAG_W)
    ) u_pair_assembler (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(in_valid & route_supported & (in_route == ROUTE_PAIR)),
        .in_ready(pair_in_ready),
        .in_slot(in_pair_slot),
        .in_data(in_data),
        .in_tag(in_tag),
        .out_valid(pair_valid),
        .out_ready(pair_ready),
        .out_pair(pair_data),
        .out_tag(pair_tag),
        .tag_mismatch(pair_tag_mismatch),
        .duplicate_slot(pair_duplicate_slot)
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            count_accepted         <= '0;
            count_single_forwarded <= '0;
            count_fanout_q         <= '0;
            count_fanout_k         <= '0;
            count_pair_issued      <= '0;
        end else begin
            if (in_valid & in_ready) begin
                count_accepted <= count_accepted + 1'b1;
            end
            if (single_valid & single_ready) begin
                count_single_forwarded <= count_single_forwarded + 1'b1;
            end
            if (fanout_q_valid & fanout_q_ready) begin
                count_fanout_q <= count_fanout_q + 1'b1;
            end
            if (fanout_k_valid & fanout_k_ready) begin
                count_fanout_k <= count_fanout_k + 1'b1;
            end
            if (pair_valid & pair_ready) begin
                count_pair_issued <= count_pair_issued + 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
