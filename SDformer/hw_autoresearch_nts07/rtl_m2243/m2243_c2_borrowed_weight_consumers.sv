`timescale 1ns/1ps
`default_nettype none

// Consumer-lifetime extension, not a new sparse arithmetic engine.
// The existing M803 response slot owns the 128-byte weight beat throughout.
// This block stores only metadata and pending consumers: it must NOT copy
// rsp_weight into a second payload array. Each context has independent signs.
// The caller supplies metadata belonging to the selected adapter response.
// Conventional eager-fork acknowledgement is prior art; the experimental
// question is whether removing the extra row-cache copy saves mapped energy.
module m2243_c2_borrowed_weight_consumers (
    input logic clk_core, rst_core,
    input logic meta_valid,
    output logic meta_ready,
    input logic [15:0] meta_epoch,
    input logic [31:0] meta_generation,
    input logic [23:0] meta_tag,
    input logic [5:0] meta_group,
    input logic meta_half,
    input logic [2:0] meta_slice,
    input logic [7:0] meta_active [0:3],
    input logic [7:0] meta_negative [0:3],
    input logic rsp_valid,
    output logic rsp_ready,
    input logic [15:0] rsp_epoch,
    input logic [31:0] rsp_generation,
    input logic [23:0] rsp_tag,
    input logic [7:0] rsp_bank_valid,
    input logic signed [7:0] rsp_weight [0:7][0:15],
    output logic bridge_valid,
    input logic bridge_ready,
    output logic [1:0] bridge_context,
    output logic [5:0] bridge_group,
    output logic bridge_half,
    output logic [2:0] bridge_slice,
    output logic [23:0] bridge_tag,
    output logic [7:0] bridge_bank_valid,
    output logic signed [8:0] bridge_effective_weight [0:7][0:15],
    output logic beat_done,
    output logic protocol_error
);
    logic busy_q, fault_q;
    logic [3:0] pending_q, chosen_onehot;
    logic [15:0] epoch_q;
    logic [31:0] generation_q;
    logic [23:0] tag_q;
    logic [5:0] group_q;
    logic half_q;
    logic [2:0] slice_q;
    logic [7:0] active_q [0:3], negative_q [0:3];
    logic [7:0] required_banks;
    logic identity_ok, bank_coverage_ok, last_consumer;

    assign meta_ready = !busy_q && !fault_q;
    assign bridge_group = group_q;
    assign bridge_half = half_q;
    assign bridge_slice = slice_q;
    assign bridge_tag = tag_q;
    assign identity_ok = rsp_epoch == epoch_q
        && rsp_generation == generation_q && rsp_tag == tag_q;
    assign bank_coverage_ok = (required_banks & ~rsp_bank_valid) == 0;
    assign protocol_error = fault_q;

    always_comb begin
        bridge_context = 0;
        chosen_onehot = 0;
        required_banks = 0;
        for (int c = 0; c < 4; c++) begin
            if (pending_q[c]) required_banks |= active_q[c];
            if (pending_q[c] && chosen_onehot == 0) begin
                bridge_context = 2'(c);
                chosen_onehot[c] = 1;
            end
        end
        last_consumer = (pending_q & ~chosen_onehot) == 0;
        bridge_valid = busy_q && !fault_q && rsp_valid && identity_ok
            && bank_coverage_ok && pending_q != 0;
        bridge_bank_valid = active_q[bridge_context];
        for (int b = 0; b < 8; b++) begin
            for (int l = 0; l < 16; l++) begin
                bridge_effective_weight[b][l] = 0;
                if (bridge_bank_valid[b]) begin
                    // Widen BEFORE negating: -(-128) is +128, not -128.
                    if (negative_q[bridge_context][b])
                        bridge_effective_weight[b][l] =
                            -$signed({rsp_weight[b][l][7], rsp_weight[b][l]});
                    else
                        bridge_effective_weight[b][l] =
                            $signed({rsp_weight[b][l][7], rsp_weight[b][l]});
                end
            end
        end
        // Empty beats retire without a fake compute event. Otherwise the
        // adapter may release its payload only with the last accepted update.
        rsp_ready = busy_q && !fault_q && identity_ok && bank_coverage_ok
            && (pending_q == 0 || (last_consumer && bridge_ready));
        beat_done = rsp_valid && rsp_ready;
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            busy_q <= 0;
            fault_q <= 0;
            pending_q <= 0;
            epoch_q <= 0;
            generation_q <= 0;
            tag_q <= 0;
            group_q <= 0;
            half_q <= 0;
            slice_q <= 0;
            for (int c = 0; c < 4; c++) begin
                active_q[c] <= 0;
                negative_q[c] <= 0;
            end
        end else begin
            if (meta_valid && meta_ready) begin
                busy_q <= 1;
                epoch_q <= meta_epoch;
                generation_q <= meta_generation;
                tag_q <= meta_tag;
                group_q <= meta_group;
                half_q <= meta_half;
                slice_q <= meta_slice;
                for (int c = 0; c < 4; c++) begin
                    pending_q[c] <= meta_active[c] != 0;
                    active_q[c] <= meta_active[c];
                    negative_q[c] <= meta_negative[c];
                end
            end
            if (busy_q && rsp_valid && (!identity_ok || !bank_coverage_ok))
                fault_q <= 1;
            if (bridge_valid && bridge_ready)
                pending_q <= pending_q & ~chosen_onehot;
            if (beat_done) begin
                busy_q <= 0;
                pending_q <= 0;
            end
        end
    end
endmodule
`default_nettype wire
