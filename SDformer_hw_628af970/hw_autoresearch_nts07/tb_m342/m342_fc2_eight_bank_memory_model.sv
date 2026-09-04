`timescale 1ns/1ps
`default_nettype none

// Reusable logical eight-bank, 128-bit-per-bank memory used by both M342
// variants.  It preserves request identity, permits eight live slots, returns
// fixed-latency out-of-order responses, and computes the same deterministic
// signed INT8 word for K8 and K1 requests.
module m342_fc2_eight_bank_memory_model #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int EPOCH_BITS = 16,
    parameter int GENERATION_BITS = 32,
    parameter int SLICE_LANES = 16,
    parameter int LATENCY = 4
) (
    input logic clk_core, input logic rst_core,
    input logic enable, input logic stall_enable, input logic newest_first,
    input logic spurious_valid,
    input logic mem_req_valid, output logic mem_req_ready,
    input logic [EPOCH_BITS-1:0] mem_req_epoch,
    input logic [2:0] mem_req_slot,
    input logic [GENERATION_BITS-1:0] mem_req_generation,
    input logic [TAG_BITS-1:0] mem_req_tag,
    input logic [2:0] mem_req_output_block,
    input logic [2:0] mem_req_slice,
    input logic [3:0] mem_req_source_count,
    input logic [7:0] mem_req_bank_valid,
    input logic [CHANNEL_BITS-1:0] mem_req_source_channel [0:7],
    input logic mem_req_accept,
    output logic mem_rsp_valid, input logic mem_rsp_ready,
    output logic [EPOCH_BITS-1:0] mem_rsp_epoch,
    output logic [2:0] mem_rsp_slot,
    output logic [GENERATION_BITS-1:0] mem_rsp_generation,
    output logic [TAG_BITS-1:0] mem_rsp_tag,
    output logic [7:0] mem_rsp_bank_valid,
    output logic signed [7:0] mem_rsp_weight [0:7][0:SLICE_LANES-1],
    input logic mem_rsp_accept,
    output logic [31:0] request_count,
    output logic [31:0] response_count,
    output logic [31:0] active_bank_read_count,
    output logic [3:0] pending_count,
    output logic live_slot_reuse_error
);
    integer cycle_q;
    logic pending_q [0:7];
    integer due_q [0:7];
    logic [EPOCH_BITS-1:0] epoch_q [0:7];
    logic [GENERATION_BITS-1:0] generation_q [0:7];
    logic [TAG_BITS-1:0] tag_q [0:7];
    logic [2:0] block_q [0:7], slice_q [0:7];
    logic [3:0] source_count_q [0:7];
    logic [7:0] bank_valid_q [0:7];
    logic [CHANNEL_BITS-1:0] channel_q [0:7][0:7];
    logic held_valid_q;
    logic [2:0] held_slot_q;
    integer candidate_slot, selected_slot;

    function automatic integer signed weight_value(
        input integer bank, input integer lane, input integer channel,
        input integer block, input integer slice);
        integer value;
        begin
            value = (channel*3 + bank*5 + block*7
                + slice*11 + lane*13) % 31;
            return value - 15;
        end
    endfunction

    function automatic logic [3:0] popcount8(input logic [7:0] value);
        logic [3:0] count;
        begin
            count = 0;
            for (int bit_index = 0; bit_index < 8; bit_index++)
                count = count + value[bit_index];
            return count;
        end
    endfunction

    always_comb begin
        candidate_slot = -1;
        for (int slot = 0; slot < 8; slot++) begin
            if (pending_q[slot] && due_q[slot] <= cycle_q) begin
                if (candidate_slot < 0)
                    candidate_slot = slot;
                else if (newest_first
                        && generation_q[slot]
                            > generation_q[candidate_slot])
                    candidate_slot = slot;
                else if (!newest_first
                        && generation_q[slot]
                            < generation_q[candidate_slot])
                    candidate_slot = slot;
            end
        end
        selected_slot = held_valid_q ? held_slot_q : candidate_slot;

        mem_rsp_valid = enable && (spurious_valid || selected_slot >= 0);
        mem_rsp_epoch = 0;
        mem_rsp_slot = 0;
        mem_rsp_generation = 0;
        mem_rsp_tag = 0;
        mem_rsp_bank_valid = 8'b0000_0001;
        for (int bank = 0; bank < 8; bank++) begin
            for (int lane = 0; lane < SLICE_LANES; lane++)
                mem_rsp_weight[bank][lane] = 0;
        end
        if (spurious_valid) begin
            mem_rsp_tag = {TAG_BITS{1'b1}};
        end else if (selected_slot >= 0) begin
            mem_rsp_epoch = epoch_q[selected_slot];
            mem_rsp_slot = selected_slot[2:0];
            mem_rsp_generation = generation_q[selected_slot];
            mem_rsp_tag = tag_q[selected_slot];
            mem_rsp_bank_valid = bank_valid_q[selected_slot];
            for (int bank = 0; bank < 8; bank++) begin
                for (int lane = 0; lane < SLICE_LANES; lane++) begin
                    if (bank_valid_q[selected_slot][bank])
                        mem_rsp_weight[bank][lane] = weight_value(
                            bank, lane, channel_q[selected_slot][bank],
                            block_q[selected_slot], slice_q[selected_slot]);
                end
            end
        end

        mem_req_ready = enable
            && (!stall_enable || cycle_q % 7 != 2)
            && (!pending_q[mem_req_slot]
                || (mem_rsp_accept && !spurious_valid
                    && selected_slot == mem_req_slot));
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_q <= 0;
            held_valid_q <= 0;
            held_slot_q <= 0;
            request_count <= 0;
            response_count <= 0;
            active_bank_read_count <= 0;
            pending_count <= 0;
            live_slot_reuse_error <= 0;
            for (int slot = 0; slot < 8; slot++) begin
                pending_q[slot] <= 0;
                due_q[slot] <= 0;
                epoch_q[slot] <= 0;
                generation_q[slot] <= 0;
                tag_q[slot] <= 0;
                block_q[slot] <= 0;
                slice_q[slot] <= 0;
                source_count_q[slot] <= 0;
                bank_valid_q[slot] <= 0;
                for (int bank = 0; bank < 8; bank++)
                    channel_q[slot][bank] <= 0;
            end
        end else begin
            cycle_q <= cycle_q + 1;
            if (!held_valid_q && !spurious_valid && candidate_slot >= 0
                    && mem_rsp_valid && !mem_rsp_ready) begin
                held_valid_q <= 1;
                held_slot_q <= candidate_slot[2:0];
            end
            if (held_valid_q && mem_rsp_accept) held_valid_q <= 0;

            if (mem_rsp_accept && !spurious_valid && selected_slot >= 0) begin
                pending_q[selected_slot] <= 0;
                response_count <= response_count + 1;
            end
            if (mem_req_accept) begin
                if (pending_q[mem_req_slot]
                        && !(mem_rsp_accept && !spurious_valid
                            && selected_slot == mem_req_slot))
                    live_slot_reuse_error <= 1;
                pending_q[mem_req_slot] <= 1;
                due_q[mem_req_slot] <= cycle_q + LATENCY;
                epoch_q[mem_req_slot] <= mem_req_epoch;
                generation_q[mem_req_slot] <= mem_req_generation;
                tag_q[mem_req_slot] <= mem_req_tag;
                block_q[mem_req_slot] <= mem_req_output_block;
                slice_q[mem_req_slot] <= mem_req_slice;
                source_count_q[mem_req_slot] <= mem_req_source_count;
                bank_valid_q[mem_req_slot] <= mem_req_bank_valid;
                for (int bank = 0; bank < 8; bank++)
                    channel_q[mem_req_slot][bank]
                        <= mem_req_source_channel[bank];
                request_count <= request_count + 1;
                active_bank_read_count <= active_bank_read_count
                    + popcount8(mem_req_bank_valid);
            end
            case ({mem_req_accept,
                    mem_rsp_accept && !spurious_valid && selected_slot >= 0})
                2'b10: pending_count <= pending_count + 1'b1;
                2'b01: pending_count <= pending_count - 1'b1;
                default: pending_count <= pending_count;
            endcase
        end
    end
endmodule

`default_nettype wire
