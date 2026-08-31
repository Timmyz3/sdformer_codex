`timescale 1ns/1ps
`default_nettype none

// Testbench-only drop-in replacement for the frozen M349 bank model.
// It preserves clean four-state behavior while preventing invalid/X request
// payload from indexing bank state.  Every stored payload and handshake-hold
// register is explicitly reset.  The module name intentionally matches the
// M979 testbench dependency; the M1332 exact filelists exclude the old model.
module m349_fc2_scalar_bank_memory_model #(
    parameter int BANK_ID = 0,
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int EPOCH_BITS = 16,
    parameter int GENERATION_BITS = 32,
    parameter int SLICE_LANES = 16,
    parameter int LATENCY = 4
) (
    input logic clk_core, input logic rst_core, input logic enable,
    input logic request_allow, input logic newest_first,
    input logic spurious_valid, input logic mem_req_valid,
    output logic mem_req_ready,
    input logic [EPOCH_BITS-1:0] mem_req_epoch,
    input logic [2:0] mem_req_slot,
    input logic [GENERATION_BITS-1:0] mem_req_generation,
    input logic [TAG_BITS-1:0] mem_req_tag,
    input logic [2:0] mem_req_output_block,
    input logic [2:0] mem_req_slice,
    input logic [CHANNEL_BITS-1:0] mem_req_source_channel,
    input logic mem_req_accept,
    output logic mem_rsp_valid, input logic mem_rsp_ready,
    output logic [EPOCH_BITS-1:0] mem_rsp_epoch,
    output logic [2:0] mem_rsp_slot,
    output logic [GENERATION_BITS-1:0] mem_rsp_generation,
    output logic [TAG_BITS-1:0] mem_rsp_tag,
    output logic signed [7:0] mem_rsp_weight [0:SLICE_LANES-1],
    input logic mem_rsp_accept,
    output logic [31:0] request_count, output logic [31:0] response_count,
    output logic [3:0] pending_count, output logic live_slot_reuse_error
);
    integer cycle_q;
    logic pending_q [0:7];
    integer due_q [0:7];
    logic [EPOCH_BITS-1:0] epoch_q [0:7];
    logic [GENERATION_BITS-1:0] generation_q [0:7];
    logic [TAG_BITS-1:0] tag_q [0:7];
    logic [2:0] block_q [0:7], slice_q [0:7];
    logic [CHANNEL_BITS-1:0] channel_q [0:7];
    logic held_valid_q;
    logic [2:0] held_slot_q;
    logic endpoint_protocol_fault_q;
    logic request_payload_known;
    integer candidate_slot, selected_slot;

    function automatic integer signed weight_value(
        input integer lane, input integer channel,
        input integer block, input integer slice);
        integer value;
        begin
            value = (channel*3 + BANK_ID*5 + block*7
                + slice*11 + lane*13) % 31;
            return value - 15;
        end
    endfunction

    always_comb begin
        request_payload_known = !$isunknown({mem_req_epoch, mem_req_slot,
            mem_req_generation, mem_req_tag, mem_req_output_block,
            mem_req_slice, mem_req_source_channel});
        candidate_slot = -1;
        for (int slot = 0; slot < 8; slot++) begin
            if (pending_q[slot] && due_q[slot] <= cycle_q) begin
                if (candidate_slot < 0)
                    candidate_slot = slot;
                else if (newest_first
                        && generation_q[slot] > generation_q[candidate_slot])
                    candidate_slot = slot;
                else if (!newest_first
                        && generation_q[slot] < generation_q[candidate_slot])
                    candidate_slot = slot;
            end
        end
        selected_slot = held_valid_q ? held_slot_q : candidate_slot;

        // Known idle ready keeps a legal producer from waiting on ready before
        // raising valid, without indexing pending_q using an invalid payload.
        mem_req_ready = 1'b0;
        if (!rst_core && enable && request_allow) begin
            if (mem_req_valid === 1'b0)
                mem_req_ready = 1'b1;
            else if (mem_req_valid === 1'b1 && request_payload_known)
                mem_req_ready = !pending_q[mem_req_slot]
                    || (mem_rsp_accept === 1'b1 && !spurious_valid
                        && selected_slot == mem_req_slot);
        end

        mem_rsp_valid = 1'b0;
        mem_rsp_epoch = '0;
        mem_rsp_slot = '0;
        mem_rsp_generation = '0;
        mem_rsp_tag = '0;
        for (int lane = 0; lane < SLICE_LANES; lane++)
            mem_rsp_weight[lane] = '0;
        if (!rst_core && enable && (spurious_valid || selected_slot >= 0)) begin
            mem_rsp_valid = 1'b1;
            if (spurious_valid) begin
                mem_rsp_tag = {TAG_BITS{1'b1}};
            end else begin
                mem_rsp_epoch = epoch_q[selected_slot];
                mem_rsp_slot = selected_slot[2:0];
                mem_rsp_generation = generation_q[selected_slot];
                mem_rsp_tag = tag_q[selected_slot];
                for (int lane = 0; lane < SLICE_LANES; lane++)
                    mem_rsp_weight[lane] = weight_value(lane,
                        channel_q[selected_slot], block_q[selected_slot],
                        slice_q[selected_slot]);
            end
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_q <= 0;
            held_valid_q <= 1'b0;
            held_slot_q <= '0;
            endpoint_protocol_fault_q <= 1'b0;
            request_count <= '0;
            response_count <= '0;
            pending_count <= '0;
            live_slot_reuse_error <= 1'b0;
            for (int slot = 0; slot < 8; slot++) begin
                pending_q[slot] <= 1'b0;
                due_q[slot] <= 0;
                epoch_q[slot] <= '0;
                generation_q[slot] <= '0;
                tag_q[slot] <= '0;
                block_q[slot] <= '0;
                slice_q[slot] <= '0;
                channel_q[slot] <= '0;
            end
        end else begin
            cycle_q <= cycle_q + 1;
            if (mem_req_valid !== 1'b0 && mem_req_valid !== 1'b1)
                endpoint_protocol_fault_q <= 1'b1;
            if (mem_req_valid === 1'b1 && !request_payload_known)
                endpoint_protocol_fault_q <= 1'b1;
            if (mem_req_accept !== 1'b0 && mem_req_accept !== 1'b1)
                endpoint_protocol_fault_q <= 1'b1;
            if (mem_rsp_accept !== 1'b0 && mem_rsp_accept !== 1'b1)
                endpoint_protocol_fault_q <= 1'b1;
            if (mem_rsp_valid && $isunknown({mem_rsp_ready, mem_rsp_accept}))
                endpoint_protocol_fault_q <= 1'b1;

            if (!held_valid_q && !spurious_valid && candidate_slot >= 0
                    && mem_rsp_valid && mem_rsp_ready === 1'b0) begin
                held_valid_q <= 1'b1;
                held_slot_q <= candidate_slot[2:0];
            end
            if (held_valid_q && mem_rsp_accept === 1'b1)
                held_valid_q <= 1'b0;

            if (mem_rsp_accept === 1'b1 && !spurious_valid
                    && selected_slot >= 0) begin
                pending_q[selected_slot] <= 1'b0;
                response_count <= response_count + 1'b1;
            end
            if (mem_req_accept === 1'b1) begin
                if (!request_payload_known) begin
                    endpoint_protocol_fault_q <= 1'b1;
                end else begin
                    if (pending_q[mem_req_slot]
                            && !(mem_rsp_accept === 1'b1 && !spurious_valid
                                && selected_slot == mem_req_slot))
                        live_slot_reuse_error <= 1'b1;
                    pending_q[mem_req_slot] <= 1'b1;
                    due_q[mem_req_slot] <= cycle_q + LATENCY;
                    epoch_q[mem_req_slot] <= mem_req_epoch;
                    generation_q[mem_req_slot] <= mem_req_generation;
                    tag_q[mem_req_slot] <= mem_req_tag;
                    block_q[mem_req_slot] <= mem_req_output_block;
                    slice_q[mem_req_slot] <= mem_req_slice;
                    channel_q[mem_req_slot] <= mem_req_source_channel;
                    request_count <= request_count + 1'b1;
                end
            end
            case ({mem_req_accept === 1'b1,
                    mem_rsp_accept === 1'b1 && !spurious_valid
                        && selected_slot >= 0})
                2'b10: pending_count <= pending_count + 1'b1;
                2'b01: pending_count <= pending_count - 1'b1;
                default: pending_count <= pending_count;
            endcase
        end
    end
endmodule

`default_nettype wire
