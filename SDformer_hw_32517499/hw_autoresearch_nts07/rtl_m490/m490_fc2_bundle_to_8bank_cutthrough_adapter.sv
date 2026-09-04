`timescale 1ns/1ps
`default_nettype none

// M490 converts M218's atomic multi-bank request/response transaction into
// eight independent single-bank SRAM endpoints.  A one-entry fall-through
// request distributor tolerates unequal bank ready signals without duplicating
// an already accepted bank request.  Eight slot-indexed response assemblies
// restore the exact bank mask.  A guarded cut-through path returns a bundle
// on the same edge that its final bank beats arrive.  If M218 stalls, the
// exact visible payload is captured in the slot store and remains stable.
//
// This adapter is deliberately arithmetic-free.  It exists to put the shared
// M218 K8 service and the eight-service M349 baseline behind the same physical
// bank interface.  The external SRAM latency may be variable and responses may
// return out of order across banks and slots.
module m490_fc2_bundle_to_8bank_cutthrough_adapter #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int EPOCH_BITS = 16,
    parameter int GENERATION_BITS = 32,
    parameter int OUTSTANDING = 8,
    parameter int SLICE_LANES = 16
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         core_req_valid,
    output logic                         core_req_ready,
    input  logic [EPOCH_BITS-1:0]        core_req_epoch,
    input  logic [2:0]                   core_req_slot,
    input  logic [GENERATION_BITS-1:0]   core_req_generation,
    input  logic [TAG_BITS-1:0]          core_req_tag,
    input  logic [2:0]                   core_req_output_block,
    input  logic [2:0]                   core_req_slice,
    input  logic [3:0]                   core_req_source_count,
    input  logic [7:0]                   core_req_bank_valid,
    input  logic [CHANNEL_BITS-1:0]      core_req_source_channel [0:7],
    output logic                         core_req_accept,

    output logic [7:0]                   bank_req_valid,
    input  logic [7:0]                   bank_req_ready,
    output logic [EPOCH_BITS-1:0]        bank_req_epoch [0:7],
    output logic [2:0]                   bank_req_slot [0:7],
    output logic [GENERATION_BITS-1:0]   bank_req_generation [0:7],
    output logic [TAG_BITS-1:0]          bank_req_tag [0:7],
    output logic [2:0]                   bank_req_output_block [0:7],
    output logic [2:0]                   bank_req_slice [0:7],
    output logic [CHANNEL_BITS-1:0]      bank_req_source_channel [0:7],
    output logic [7:0]                   bank_req_accept,

    input  logic [7:0]                   bank_rsp_valid,
    output logic [7:0]                   bank_rsp_ready,
    input  logic [EPOCH_BITS-1:0]        bank_rsp_epoch [0:7],
    input  logic [2:0]                   bank_rsp_slot [0:7],
    input  logic [GENERATION_BITS-1:0]   bank_rsp_generation [0:7],
    input  logic [TAG_BITS-1:0]          bank_rsp_tag [0:7],
    input  logic signed [7:0]            bank_rsp_weight
                                                   [0:7][0:SLICE_LANES-1],
    output logic [7:0]                   bank_rsp_accept,

    output logic                         core_rsp_valid,
    input  logic                         core_rsp_ready,
    output logic [EPOCH_BITS-1:0]        core_rsp_epoch,
    output logic [2:0]                   core_rsp_slot,
    output logic [GENERATION_BITS-1:0]   core_rsp_generation,
    output logic [TAG_BITS-1:0]          core_rsp_tag,
    output logic [7:0]                   core_rsp_bank_valid,
    output logic signed [7:0]            core_rsp_weight
                                                   [0:7][0:SLICE_LANES-1],
    output logic                         core_rsp_accept,

    output logic                         protocol_error,
    output logic                         stale_response_seen,
    output logic                         busy,
    output logic [3:0]                   debug_live_slots,
    output logic [31:0]                  debug_bundle_request_count,
    output logic [31:0]                  debug_bank_request_count,
    output logic [31:0]                  debug_bank_response_count,
    output logic [31:0]                  debug_bundle_response_count
);
    localparam bit PARAMETERS_LEGAL = OUTSTANDING == 8
        && SLICE_LANES == 16;

    logic fault_q, stale_q;

    logic [7:0] pending_mask_q;
    logic [EPOCH_BITS-1:0] pending_epoch_q;
    logic [2:0] pending_slot_q;
    logic [GENERATION_BITS-1:0] pending_generation_q;
    logic [TAG_BITS-1:0] pending_tag_q;
    logic [2:0] pending_output_block_q, pending_slice_q;
    logic [CHANNEL_BITS-1:0] pending_source_channel_q [0:7];

    logic slot_valid_q [0:OUTSTANDING-1];
    logic [EPOCH_BITS-1:0] slot_epoch_q [0:OUTSTANDING-1];
    logic [GENERATION_BITS-1:0] slot_generation_q
                                                  [0:OUTSTANDING-1];
    logic [TAG_BITS-1:0] slot_tag_q [0:OUTSTANDING-1];
    logic [7:0] slot_expected_q [0:OUTSTANDING-1];
    logic [7:0] slot_arrived_q [0:OUTSTANDING-1];
    logic signed [7:0] slot_weight_q
                              [0:OUTSTANDING-1][0:7][0:SLICE_LANES-1];

    logic req_payload_legal, req_shape_legal, req_slot_open;
    logic illegal_request;
    logic rsp_shape_legal [0:7];
    logic illegal_response;
    logic complete_found;
    logic [2:0] complete_slot;
    logic complete_cutthrough;
    logic [7:0] incoming_mask [0:OUTSTANDING-1];
    logic rsp_hold_valid_q;
    logic [2:0] rsp_hold_slot_q;
    logic [3:0] req_mask_count;

    logic [31:0] bundle_request_count_q, bank_request_count_q;
    logic [31:0] bank_response_count_q, bundle_response_count_q;

    function automatic logic [3:0] popcount8(input logic [7:0] value);
        logic [3:0] count;
        begin
            count = 0;
            for (int index = 0; index < 8; index++)
                count = count + value[index];
            return count;
        end
    endfunction

    generate
        if (!PARAMETERS_LEGAL) begin : g_illegal_parameters
            initial $fatal(1, "M490 supports only OUTSTANDING8/SLICE16");
        end
    endgenerate

    always_comb begin : completion_select
        for (int slot = 0; slot < OUTSTANDING; slot++) begin
            incoming_mask[slot] = 0;
            for (int bank = 0; bank < 8; bank++) begin
                if (bank_rsp_valid[bank] && rsp_shape_legal[bank]
                        && bank_rsp_slot[bank] == slot[2:0])
                    incoming_mask[slot][bank] = 1;
            end
        end

        complete_found = rsp_hold_valid_q;
        complete_slot = rsp_hold_slot_q;
        complete_cutthrough = 0;
        if (!rsp_hold_valid_q) begin
            for (int slot = 0; slot < OUTSTANDING; slot++) begin
                if (!complete_found && slot_valid_q[slot]
                        && slot_expected_q[slot] != 0
                        && slot_arrived_q[slot] == slot_expected_q[slot]) begin
                    complete_found = 1;
                    complete_slot = slot[2:0];
                end
            end
            for (int slot = 0; slot < OUTSTANDING; slot++) begin
                if (!complete_found && slot_valid_q[slot]
                        && slot_expected_q[slot] != 0
                        && incoming_mask[slot] != 0
                        && (slot_arrived_q[slot] | incoming_mask[slot])
                            == slot_expected_q[slot]) begin
                    complete_found = 1;
                    complete_slot = slot[2:0];
                    complete_cutthrough = 1;
                end
            end
        end
    end

    always_comb begin : request_analysis
        req_mask_count = popcount8(core_req_bank_valid);
        // Slot availability is flow control, not request legality.  Keeping it
        // out of illegal_request removes the former combinational cycle through
        // core_rsp_accept while allowing a producer to hold valid until the
        // slot becomes free on the following edge.
        // Request legality no longer depends on slot availability, so this
        // retirement bypass cannot feed protocol_error and has no Boolean
        // fixed-point loop.  It removes the exact-slot reuse bubble.
        req_slot_open = !slot_valid_q[core_req_slot]
            || (core_rsp_accept && complete_slot == core_req_slot);
        req_payload_legal = PARAMETERS_LEGAL
            && core_req_slot < OUTSTANDING
            && core_req_output_block < 8 && core_req_slice < 6
            && core_req_source_count >= 1 && core_req_source_count <= 8
            && core_req_source_count == req_mask_count
            && core_req_bank_valid != 0;
        for (int bank = 0; bank < 8; bank++) begin
            if (core_req_bank_valid[bank]
                    && core_req_source_channel[bank][2:0] != bank[2:0])
                req_payload_legal = 0;
        end
        req_shape_legal = req_payload_legal && req_slot_open;
        illegal_request = core_req_valid && !req_payload_legal;
    end

    always_comb begin : response_analysis
        illegal_response = 0;
        for (int bank = 0; bank < 8; bank++) begin
            rsp_shape_legal[bank] = bank_rsp_slot[bank] < OUTSTANDING;
            if (bank_rsp_slot[bank] < OUTSTANDING) begin
                rsp_shape_legal[bank] = rsp_shape_legal[bank]
                    && slot_valid_q[bank_rsp_slot[bank]]
                    && slot_expected_q[bank_rsp_slot[bank]][bank]
                    && !slot_arrived_q[bank_rsp_slot[bank]][bank]
                    && bank_rsp_epoch[bank]
                        == slot_epoch_q[bank_rsp_slot[bank]]
                    && bank_rsp_generation[bank]
                        == slot_generation_q[bank_rsp_slot[bank]]
                    && bank_rsp_tag[bank]
                        == slot_tag_q[bank_rsp_slot[bank]];
            end
            if (bank_rsp_valid[bank] && !rsp_shape_legal[bank])
                illegal_response = 1;
        end
    end

    assign protocol_error = fault_q || illegal_request || illegal_response;
    assign stale_response_seen = stale_q || illegal_response;

    always_comb begin : interfaces
        logic use_pending;
        use_pending = pending_mask_q != 0;

        core_rsp_valid = complete_found && !protocol_error;
        core_rsp_accept = core_rsp_valid && core_rsp_ready;
        core_rsp_epoch = complete_found ? slot_epoch_q[complete_slot] : 0;
        core_rsp_slot = complete_slot;
        core_rsp_generation = complete_found
            ? slot_generation_q[complete_slot] : 0;
        core_rsp_tag = complete_found ? slot_tag_q[complete_slot] : 0;
        core_rsp_bank_valid = complete_found
            ? slot_expected_q[complete_slot] : 0;
        for (int bank = 0; bank < 8; bank++) begin
            for (int lane = 0; lane < SLICE_LANES; lane++) begin
                core_rsp_weight[bank][lane] = 0;
                if (complete_found) begin
                    core_rsp_weight[bank][lane]
                        = slot_weight_q[complete_slot][bank][lane];
                    if (complete_cutthrough
                            && incoming_mask[complete_slot][bank])
                        core_rsp_weight[bank][lane]
                            = bank_rsp_weight[bank][lane];
                end
            end
        end

        core_req_ready = !protocol_error && !use_pending
            && (!core_req_valid || req_shape_legal);
        core_req_accept = core_req_valid && core_req_ready;

        for (int bank = 0; bank < 8; bank++) begin
            bank_req_valid[bank] = !protocol_error
                && (use_pending ? pending_mask_q[bank]
                    : (core_req_valid && req_shape_legal
                       && core_req_bank_valid[bank]));
            bank_req_epoch[bank] = use_pending
                ? pending_epoch_q : core_req_epoch;
            bank_req_slot[bank] = use_pending
                ? pending_slot_q : core_req_slot;
            bank_req_generation[bank] = use_pending
                ? pending_generation_q : core_req_generation;
            bank_req_tag[bank] = use_pending
                ? pending_tag_q : core_req_tag;
            bank_req_output_block[bank] = use_pending
                ? pending_output_block_q : core_req_output_block;
            bank_req_slice[bank] = use_pending
                ? pending_slice_q : core_req_slice;
            bank_req_source_channel[bank] = use_pending
                ? pending_source_channel_q[bank]
                : core_req_source_channel[bank];
            bank_req_accept[bank] = bank_req_valid[bank]
                && bank_req_ready[bank];

            bank_rsp_ready[bank] = !protocol_error
                && (!bank_rsp_valid[bank] || rsp_shape_legal[bank]);
            bank_rsp_accept[bank] = bank_rsp_valid[bank]
                && bank_rsp_ready[bank];
        end
    end

    always_comb begin : debug_view
        debug_live_slots = 0;
        for (int slot = 0; slot < OUTSTANDING; slot++)
            debug_live_slots = debug_live_slots + slot_valid_q[slot];
        busy = pending_mask_q != 0 || debug_live_slots != 0;
        debug_bundle_request_count = bundle_request_count_q;
        debug_bank_request_count = bank_request_count_q;
        debug_bank_response_count = bank_response_count_q;
        debug_bundle_response_count = bundle_response_count_q;
    end

    always_ff @(posedge clk_core) begin : state
        if (rst_core) begin
            fault_q <= 0;
            stale_q <= 0;
            rsp_hold_valid_q <= 0;
            rsp_hold_slot_q <= 0;
            pending_mask_q <= 0;
            pending_epoch_q <= 0;
            pending_slot_q <= 0;
            pending_generation_q <= 0;
            pending_tag_q <= 0;
            pending_output_block_q <= 0;
            pending_slice_q <= 0;
            bundle_request_count_q <= 0;
            bank_request_count_q <= 0;
            bank_response_count_q <= 0;
            bundle_response_count_q <= 0;
            for (int bank = 0; bank < 8; bank++)
                pending_source_channel_q[bank] <= 0;
            for (int slot = 0; slot < OUTSTANDING; slot++) begin
                slot_valid_q[slot] <= 0;
                slot_epoch_q[slot] <= 0;
                slot_generation_q[slot] <= 0;
                slot_tag_q[slot] <= 0;
                slot_expected_q[slot] <= 0;
                slot_arrived_q[slot] <= 0;
                for (int bank = 0; bank < 8; bank++) begin
                    for (int lane = 0; lane < SLICE_LANES; lane++)
                        slot_weight_q[slot][bank][lane] <= 0;
                end
            end
        end else begin
            if (illegal_request || illegal_response) begin
                fault_q <= 1;
                if (illegal_response) stale_q <= 1;
            end

            if (!protocol_error) begin
                if (core_rsp_valid && !core_rsp_ready
                        && !rsp_hold_valid_q) begin
                    rsp_hold_valid_q <= 1;
                    rsp_hold_slot_q <= complete_slot;
                end else if (core_rsp_accept) begin
                    rsp_hold_valid_q <= 0;
                end

                if (pending_mask_q != 0) begin
                    pending_mask_q <= pending_mask_q & ~bank_req_accept;
                end else if (core_req_accept) begin
                    pending_mask_q <= core_req_bank_valid
                        & ~bank_req_accept;
                    pending_epoch_q <= core_req_epoch;
                    pending_slot_q <= core_req_slot;
                    pending_generation_q <= core_req_generation;
                    pending_tag_q <= core_req_tag;
                    pending_output_block_q <= core_req_output_block;
                    pending_slice_q <= core_req_slice;
                    for (int bank = 0; bank < 8; bank++)
                        pending_source_channel_q[bank]
                            <= core_req_source_channel[bank];
                end

                if (core_rsp_accept) begin
                    slot_valid_q[complete_slot] <= 0;
                    slot_expected_q[complete_slot] <= 0;
                    slot_arrived_q[complete_slot] <= 0;
                    bundle_response_count_q <= bundle_response_count_q + 1'b1;
                end

                if (core_req_accept) begin
                    slot_valid_q[core_req_slot] <= 1;
                    slot_epoch_q[core_req_slot] <= core_req_epoch;
                    slot_generation_q[core_req_slot]
                        <= core_req_generation;
                    slot_tag_q[core_req_slot] <= core_req_tag;
                    slot_expected_q[core_req_slot]
                        <= core_req_bank_valid;
                    slot_arrived_q[core_req_slot] <= 0;
                    bundle_request_count_q <= bundle_request_count_q + 1'b1;
                end

                for (int bank = 0; bank < 8; bank++) begin
                    // A cut-through response accepted by the core is already
                    // consumed on this edge.  Do not let its final bank beats
                    // overwrite the cleared/reused slot's new arrived state.
                    if (bank_rsp_accept[bank]
                            && !(core_rsp_accept
                                && bank_rsp_slot[bank] == complete_slot)) begin
                        slot_arrived_q[bank_rsp_slot[bank]][bank] <= 1;
                        for (int lane = 0; lane < SLICE_LANES; lane++)
                            slot_weight_q[bank_rsp_slot[bank]][bank][lane]
                                <= bank_rsp_weight[bank][lane];
                    end
                end

                bank_request_count_q <= bank_request_count_q
                    + popcount8(bank_req_accept);
                bank_response_count_q <= bank_response_count_q
                    + popcount8(bank_rsp_accept);
            end
        end
    end
endmodule

`default_nettype wire
