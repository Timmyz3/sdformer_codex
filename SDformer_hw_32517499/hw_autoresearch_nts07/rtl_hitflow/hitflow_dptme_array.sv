`timescale 1ns/1ps
`default_nettype none

module hitflow_dptme_array #(
    parameter int LANES       = 32,
    parameter int SLOTS       = 10,
    parameter int PACK_GROUPS = 5,
    parameter int X_W         = 8,
    parameter int W_W         = 8,
    parameter int ACC_W       = 24,
    parameter int TAG_W       = 48
) (
    input  logic                              clk_core,
    input  logic                              rst_core,
    input  logic                              step_valid,
    output logic                              step_ready,
    input  logic                              mode_t2,
    input  logic                              step_first,
    input  logic                              step_last,
    input  logic [PACK_GROUPS-1:0]            group_valid,
    input  logic [(PACK_GROUPS*LANES*X_W)-1:0] x_groups,
    input  logic [(SLOTS*W_W)-1:0]            weight_slots,
    input  logic [(SLOTS*ACC_W)-1:0]          bias_slots,
    input  logic [(SLOTS*ACC_W)-1:0]          threshold_slots,
    input  logic [TAG_W-1:0]                  step_tag,
    output logic                              out_valid,
    input  logic                              out_ready,
    output logic [(SLOTS*LANES)-1:0]          out_events,
    output logic [(SLOTS*LANES*ACC_W)-1:0]    out_hidden,
    output logic [SLOTS-1:0]                  out_slot_valid,
    output logic [TAG_W-1:0]                  out_tag,
    output logic                              protocol_error
);

    localparam LINT_SLOTS = SLOTS;
    localparam LINT_LANES = LANES;

    logic                           busy_q;
    logic                           mode_q;
    logic [3:0]                     steps_seen_q;
    logic [PACK_GROUPS-1:0]         group_valid_q;
    logic [TAG_W-1:0]               tag_q;
    logic                           out_valid_q;
    logic [(SLOTS*LANES)-1:0]       events_q;
    logic [SLOTS-1:0]               slot_valid_q;
    logic                           command_matches;
    logic                           length_matches;
    logic                           protocol_ok;
    logic                           step_fire;

    assign command_matches = (mode_t2 == mode_q) &&
                             (group_valid == group_valid_q) &&
                             (step_tag == tag_q);
    assign length_matches = mode_q ?
                            (step_last ? (steps_seen_q == 4'd1) : (steps_seen_q < 4'd1)) :
                            (step_last ? (steps_seen_q == 4'd9) : (steps_seen_q < 4'd9));
    assign protocol_ok = step_first ? (~busy_q & ~step_last) :
                                      (busy_q & command_matches & length_matches);
    assign step_ready = (~out_valid_q | out_ready) & protocol_ok;
    assign step_fire = step_valid & step_ready;
    assign protocol_error = step_valid & ~protocol_ok;
    assign out_valid = out_valid_q;
    assign out_events = events_q;
    assign out_slot_valid = slot_valid_q;
    assign out_tag = tag_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            busy_q         <= 1'b0;
            mode_q         <= 1'b0;
            steps_seen_q   <= '0;
            group_valid_q  <= '0;
            tag_q          <= '0;
            out_valid_q    <= 1'b0;
            slot_valid_q   <= '0;
        end else begin
            if (out_valid_q & out_ready) begin
                out_valid_q <= 1'b0;
            end
            if (step_fire) begin
                if (step_first) begin
                    mode_q        <= mode_t2;
                    group_valid_q <= group_valid;
                    tag_q         <= step_tag;
                    steps_seen_q  <= 4'd1;
                end
                if (step_last) begin
                    busy_q      <= 1'b0;
                    steps_seen_q <= '0;
                    out_valid_q <= 1'b1;
                    if (mode_t2) begin
                        for (int slot = 32'd0; slot < LINT_SLOTS; slot = slot + 32'd1) begin
                            if (slot < (2 * PACK_GROUPS)) begin
                                slot_valid_q[slot] <= group_valid[slot / 32'd2];
                            end else begin
                                slot_valid_q[slot] <= 1'b0;
                            end
                        end
                    end else begin
                        slot_valid_q <= '1;
                    end
                end else begin
                    busy_q <= 1'b1;
                    if (~step_first) begin
                        steps_seen_q <= steps_seen_q + 1'b1;
                    end
                end
            end
        end
    end

    for (genvar slot = 32'd0; slot < LINT_SLOTS; slot = slot + 32'd1) begin : g_slot
        localparam int GROUP_INDEX = slot / 32'd2;
        logic signed [W_W-1:0] weight;
        logic signed [ACC_W-1:0] bias;
        logic signed [ACC_W-1:0] threshold_q;
        logic slot_active;

        assign weight = weight_slots[(slot*W_W) +: W_W];
        assign bias = bias_slots[(slot*ACC_W) +: ACC_W];
        assign slot_active = ~mode_t2 |
                             ((slot < (2 * PACK_GROUPS)) && group_valid[GROUP_INDEX]);

        always_ff @(posedge clk_core) begin
            if (rst_core) begin
                threshold_q <= '0;
            end else if (step_fire & step_first) begin
                threshold_q <= threshold_slots[(slot*ACC_W) +: ACC_W];
            end
        end

        for (genvar lane = 32'd0; lane < LINT_LANES; lane = lane + 32'd1) begin : g_lane
            localparam int EVENT_INDEX = slot * LANES + lane;
            localparam int T10_X_INDEX = lane * X_W;
            localparam int T2_X_INDEX = (GROUP_INDEX * LANES + lane) * X_W;
            logic signed [X_W-1:0] x_value;
            logic signed [(X_W+W_W)-1:0] product;
            logic signed [ACC_W-1:0] product_ext;
            logic signed [ACC_W-1:0] accum_q;
            logic signed [ACC_W-1:0] accum_next;

            assign x_value = mode_t2 ? x_groups[T2_X_INDEX +: X_W] :
                                       x_groups[T10_X_INDEX +: X_W];
            assign product = x_value * weight;
            assign product_ext = {{(ACC_W-X_W-W_W){product[X_W+W_W-1]}}, product};
            assign accum_next = (step_first ? bias : accum_q) + product_ext;
            assign out_hidden[(EVENT_INDEX*ACC_W) +: ACC_W] = accum_q;

            always_ff @(posedge clk_core) begin
                if (rst_core) begin
                    accum_q <= '0;
                    events_q[EVENT_INDEX] <= 1'b0;
                end else if (step_fire) begin
                    if (slot_active) begin
                        accum_q <= accum_next;
                        if (step_last) begin
                            events_q[EVENT_INDEX] <= (accum_next >= threshold_q);
                        end
                    end else if (step_last) begin
                        accum_q <= '0;
                        events_q[EVENT_INDEX] <= 1'b0;
                    end
                end
            end
        end
    end

endmodule

`default_nettype wire
