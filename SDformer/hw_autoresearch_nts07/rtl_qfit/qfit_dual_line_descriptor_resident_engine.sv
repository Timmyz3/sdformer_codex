`timescale 1ns/1ps
`default_nettype none

// M4 Local/Motion engine with row descriptors resident across output-lane
// tiles.  A batch contains up to CONTEXTS rows with identical immutable weight
// geometry.  All source chunks are loaded once, then replayed for every lane
// tile while Acc32 remains on chip across chunks.
//
// The response datapath compacts ISSUE_WIDTH bank words into REDUCE_SLOTS slots
// per context before reduction.  The request scheduler enforces that capacity,
// avoiding the CONTEXTS x ISSUE_WIDTH x OUT_LANES add-tree replication exposed
// by the M3 synthesis experiment.
module qfit_dual_line_descriptor_resident_engine #(
    parameter int TILE_BITS = 256,
    parameter int MAX_CHUNKS = 12,
    parameter int MAX_LANE_TILES = 32,
    parameter int ISSUE_WIDTH = 16,
    parameter int CONTEXTS = 4,
    parameter int REDUCE_SLOTS = 4,
    parameter int OUT_LANES = 96,
    parameter int TAG_W = 32,
    parameter int OBJECT_W = 64,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter int INDEX_W = (TILE_BITS <= 1) ? 1 : $clog2(TILE_BITS),
    parameter int BANK_BITS = (ISSUE_WIDTH <= 1) ? 0 : $clog2(ISSUE_WIDTH),
    parameter int BANK_ADDR_W = INDEX_W - BANK_BITS,
    parameter int CTX_W = (CONTEXTS <= 1) ? 1 : $clog2(CONTEXTS),
    parameter int CTX_COUNT_W = $clog2(CONTEXTS + 1),
    parameter int CHUNK_W = (MAX_CHUNKS <= 1) ? 1 : $clog2(MAX_CHUNKS),
    parameter int CHUNK_COUNT_W = $clog2(MAX_CHUNKS + 1),
    parameter int LANE_TILE_W = (MAX_LANE_TILES <= 1) ? 1 : $clog2(MAX_LANE_TILES),
    parameter int LANE_COUNT_W = $clog2(MAX_LANE_TILES + 1),
    parameter int SLOT_W = (REDUCE_SLOTS <= 1) ? 1 : $clog2(REDUCE_SLOTS),
    parameter int SLOT_COUNT_W = $clog2(REDUCE_SLOTS + 1),
    parameter int SOURCE_COUNT_W = $clog2(MAX_CHUNKS*TILE_BITS + 1)
) (
    input  logic                                      clk_core,
    input  logic                                      rst_core,

    input  logic                                      descriptor_valid,
    output logic                                      descriptor_ready,
    input  logic                                      descriptor_row_first,
    input  logic                                      descriptor_row_last,
    input  logic                                      descriptor_batch_last,
    input  logic [TAG_W-1:0]                          descriptor_tag,
    input  logic [OBJECT_W-1:0]                       descriptor_object_tag,
    input  logic [CHUNK_W-1:0]                        descriptor_chunk_index,
    input  logic [CHUNK_COUNT_W-1:0]                  descriptor_chunk_count,
    input  logic [LANE_COUNT_W-1:0]                   descriptor_lane_tile_count,
    input  logic                                      descriptor_use_motion,
    input  logic [TILE_BITS-1:0]                      descriptor_source_bits,
    input  logic [TILE_BITS-1:0]                      descriptor_negative_bits,

    output logic                                      weight_request_valid,
    input  logic                                      weight_request_ready,
    output logic [OBJECT_W-1:0]                       weight_request_object_tag,
    output logic [CHUNK_W-1:0]                        weight_request_chunk_index,
    output logic [LANE_TILE_W-1:0]                    weight_request_lane_tile,
    output logic [ISSUE_WIDTH-1:0]                    weight_request_bank_valid,
    output logic [ISSUE_WIDTH*BANK_ADDR_W-1:0]        weight_request_bank_addr,
    output logic [ISSUE_WIDTH*CTX_W-1:0]              weight_request_bank_context,
    output logic [ISSUE_WIDTH*SLOT_W-1:0]             weight_request_bank_slot,
    output logic [ISSUE_WIDTH-1:0]                    weight_request_bank_negative,

    input  logic                                      weight_response_valid,
    output logic                                      weight_response_ready,
    input  logic [ISSUE_WIDTH-1:0]                    weight_response_bank_valid,
    input  logic [ISSUE_WIDTH*OUT_LANES*W_W-1:0]     weight_response_data,

    output logic                                      output_valid,
    input  logic                                      output_ready,
    output logic [TAG_W-1:0]                          output_tag,
    output logic [OBJECT_W-1:0]                       output_object_tag,
    output logic [LANE_TILE_W-1:0]                    output_lane_tile,
    output logic                                      output_use_motion,
    output logic [SOURCE_COUNT_W-1:0]                 output_source_count,
    output logic [OUT_LANES*ACC_W-1:0]                output_acc,

    output logic [2:0]                                controller_state,
    output logic [CTX_COUNT_W-1:0]                    resident_contexts,
    output logic                                      protocol_error
);
    typedef enum logic [2:0] {
        ST_LOAD   = 3'd0,
        ST_PREP   = 3'd1,
        ST_ISSUE  = 3'd2,
        ST_OUTPUT = 3'd3
    } state_t;

    state_t state_q;
    logic faulted_q;
    logic in_row_q;
    logic [CTX_COUNT_W-1:0] loaded_contexts_q;
    logic [CTX_COUNT_W-1:0] batch_contexts_q;
    logic [CTX_W-1:0] load_context_q;
    logic [CHUNK_W-1:0] expected_chunk_q;
    logic [OBJECT_W-1:0] object_tag_q;
    logic [CHUNK_COUNT_W-1:0] chunk_count_q;
    logic [LANE_COUNT_W-1:0] lane_tile_count_q;
    logic [CHUNK_W-1:0] current_chunk_q;
    logic [LANE_TILE_W-1:0] current_lane_tile_q;
    logic [CTX_W-1:0] output_context_q;

    logic [TAG_W-1:0] tag_q [0:CONTEXTS-1];
    logic use_motion_q [0:CONTEXTS-1];
    logic [TILE_BITS-1:0] source_descriptor_q [0:CONTEXTS-1][0:MAX_CHUNKS-1];
    logic [TILE_BITS-1:0] negative_descriptor_q [0:CONTEXTS-1][0:MAX_CHUNKS-1];
    logic [TILE_BITS-1:0] remaining_q [0:CONTEXTS-1];
    logic [TILE_BITS-1:0] negative_q [0:CONTEXTS-1];
    logic [SOURCE_COUNT_W-1:0] source_count_q [0:CONTEXTS-1];
    logic signed [ACC_W-1:0] acc_q [0:CONTEXTS-1][0:OUT_LANES-1];

    logic pending_q;
    logic pending_chunk_last_q;
    logic [ISSUE_WIDTH-1:0] pending_bank_valid_q;
    logic [ISSUE_WIDTH*CTX_W-1:0] pending_bank_context_q;
    logic [ISSUE_WIDTH*SLOT_W-1:0] pending_bank_slot_q;
    logic [ISSUE_WIDTH-1:0] pending_bank_negative_q;

    logic descriptor_fire;
    logic request_fire;
    logic response_fire;
    logic output_fire;
    logic selected_valid;
    logic response_contract_valid;
    logic can_issue_request;
    logic protocol_violation;
    logic batch_remaining_empty;
    logic request_chunk_last;

    logic [TILE_BITS-1:0] selection_mask [0:CONTEXTS-1];
    logic [TILE_BITS-1:0] remaining_after_request [0:CONTEXTS-1];
    logic [SLOT_COUNT_W-1:0] request_context_sources [0:CONTEXTS-1];
    logic [SLOT_COUNT_W-1:0] context_slot_count [0:CONTEXTS-1];
    logic signed [ACC_W-1:0] slot_weight [0:CONTEXTS-1][0:REDUCE_SLOTS-1][0:OUT_LANES-1];
    logic signed [ACC_W-1:0] response_sum [0:CONTEXTS-1][0:OUT_LANES-1];

    function automatic logic bank_has_source(
        input logic [TILE_BITS-1:0] value,
        input integer bank
    );
        logic found;
        begin
            found = 1'b0;
            for (int source = bank; source < TILE_BITS; source = source + ISSUE_WIDTH)
                found = found | value[source];
            bank_has_source = found;
        end
    endfunction

    function automatic logic [BANK_ADDR_W-1:0] first_bank_address(
        input logic [TILE_BITS-1:0] value,
        input integer bank
    );
        logic found;
        logic [BANK_ADDR_W-1:0] address;
        begin
            found = 1'b0;
            address = '0;
            for (int source = bank; source < TILE_BITS; source = source + ISSUE_WIDTH) begin
                if (!found && value[source]) begin
                    found = 1'b1;
                    address = BANK_ADDR_W'(source >> BANK_BITS);
                end
            end
            first_bank_address = address;
        end
    endfunction

    function automatic logic signed [ACC_W-1:0] extend_weight(
        input logic [W_W-1:0] value
    );
        extend_weight = {{(ACC_W-W_W){value[W_W-1]}}, value};
    endfunction

    initial begin
        if (ACC_W < W_W)
            $error("ACC_W must be at least W_W");
        if (TILE_BITS % ISSUE_WIDTH != 0)
            $error("TILE_BITS must be divisible by ISSUE_WIDTH");
        if (ISSUE_WIDTH < 1 || (ISSUE_WIDTH & (ISSUE_WIDTH - 1)) != 0)
            $error("ISSUE_WIDTH must be a positive power of two");
        if (CONTEXTS < 1 || REDUCE_SLOTS < 1 || REDUCE_SLOTS > ISSUE_WIDTH)
            $error("invalid context/reducer geometry");
        if (MAX_CHUNKS < 1 || MAX_LANE_TILES < 1)
            $error("descriptor geometry must be positive");
    end

    assign descriptor_ready = state_q == ST_LOAD
        && loaded_contexts_q < CTX_COUNT_W'(CONTEXTS) && !faulted_q;
    assign descriptor_fire = descriptor_valid && descriptor_ready;
    assign controller_state = state_q;
    assign resident_contexts = batch_contexts_q;

    always_comb begin
        for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
            selection_mask[ctx] = '0;
            context_slot_count[ctx] = '0;
        end
        weight_request_bank_valid = '0;
        weight_request_bank_addr = '0;
        weight_request_bank_context = '0;
        weight_request_bank_slot = '0;
        weight_request_bank_negative = '0;
        for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin
            logic context_found;
            context_found = 1'b0;
            for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
                if (!context_found && CTX_COUNT_W'(ctx) < batch_contexts_q
                        && context_slot_count[ctx] < SLOT_COUNT_W'(REDUCE_SLOTS)
                        && bank_has_source(remaining_q[ctx], bank)) begin
                    logic [BANK_ADDR_W-1:0] address;
                    integer source;
                    context_found = 1'b1;
                    address = first_bank_address(remaining_q[ctx], bank);
                    source = $unsigned(address) * ISSUE_WIDTH + bank;
                    weight_request_bank_valid[bank] = 1'b1;
                    weight_request_bank_addr[bank*BANK_ADDR_W +: BANK_ADDR_W] = address;
                    weight_request_bank_context[bank*CTX_W +: CTX_W] = CTX_W'(ctx);
                    weight_request_bank_slot[bank*SLOT_W +: SLOT_W]
                        = SLOT_W'(context_slot_count[ctx]);
                    weight_request_bank_negative[bank] = negative_q[ctx][source];
                    selection_mask[ctx][source] = 1'b1;
                    context_slot_count[ctx] = context_slot_count[ctx] + SLOT_COUNT_W'(1);
                end
            end
        end
        selected_valid = |weight_request_bank_valid;
        batch_remaining_empty = 1'b1;
        for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
            remaining_after_request[ctx] = remaining_q[ctx] & ~selection_mask[ctx];
            request_context_sources[ctx] = context_slot_count[ctx];
            if (CTX_COUNT_W'(ctx) < batch_contexts_q
                    && remaining_after_request[ctx] != '0)
                batch_remaining_empty = 1'b0;
        end
        request_chunk_last = selected_valid && batch_remaining_empty;
    end

    assign response_contract_valid = weight_response_bank_valid == pending_bank_valid_q;
    assign weight_response_ready = pending_q;
    assign response_fire = weight_response_valid && weight_response_ready;
    assign can_issue_request = !pending_q
        || (weight_response_valid && response_contract_valid);
    assign weight_request_valid = state_q == ST_ISSUE && selected_valid
        && can_issue_request && !faulted_q;
    assign weight_request_object_tag = object_tag_q;
    assign weight_request_chunk_index = current_chunk_q;
    assign weight_request_lane_tile = current_lane_tile_q;
    assign request_fire = weight_request_valid && weight_request_ready;

    always_comb begin
        for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
            for (int slot = 0; slot < REDUCE_SLOTS; slot = slot + 1) begin
                for (int lane = 0; lane < OUT_LANES; lane = lane + 1) begin
                    slot_weight[ctx][slot][lane] = '0;
                    for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin
                        if (pending_bank_valid_q[bank]
                                && pending_bank_context_q[bank*CTX_W +: CTX_W] == CTX_W'(ctx)
                                && pending_bank_slot_q[bank*SLOT_W +: SLOT_W] == SLOT_W'(slot)) begin
                            if (pending_bank_negative_q[bank])
                                slot_weight[ctx][slot][lane] = -extend_weight(
                                    weight_response_data[(bank*OUT_LANES + lane)*W_W +: W_W]
                                );
                            else
                                slot_weight[ctx][slot][lane] = extend_weight(
                                    weight_response_data[(bank*OUT_LANES + lane)*W_W +: W_W]
                                );
                        end
                    end
                end
            end
            for (int lane = 0; lane < OUT_LANES; lane = lane + 1) begin
                response_sum[ctx][lane] = '0;
                for (int slot = 0; slot < REDUCE_SLOTS; slot = slot + 1)
                    response_sum[ctx][lane] = response_sum[ctx][lane]
                        + slot_weight[ctx][slot][lane];
            end
        end
    end

    assign output_valid = state_q == ST_OUTPUT && !faulted_q;
    assign output_fire = output_valid && output_ready;
    assign output_tag = tag_q[output_context_q];
    assign output_object_tag = object_tag_q;
    assign output_lane_tile = current_lane_tile_q;
    assign output_use_motion = use_motion_q[output_context_q];
    assign output_source_count = source_count_q[output_context_q];
    always_comb begin
        output_acc = '0;
        for (int lane = 0; lane < OUT_LANES; lane = lane + 1)
            output_acc[lane*ACC_W +: ACC_W] = acc_q[output_context_q][lane];
    end

    assign protocol_error = faulted_q;
    assign protocol_violation =
        (descriptor_fire && ((descriptor_negative_bits & ~descriptor_source_bits) != '0))
        || (descriptor_fire && descriptor_chunk_count == '0)
        || (descriptor_fire && descriptor_chunk_count > CHUNK_COUNT_W'(MAX_CHUNKS))
        || (descriptor_fire && descriptor_lane_tile_count == '0)
        || (descriptor_fire && descriptor_lane_tile_count > LANE_COUNT_W'(MAX_LANE_TILES))
        || (descriptor_fire && descriptor_row_first == in_row_q)
        || (descriptor_fire && descriptor_chunk_index != expected_chunk_q)
        || (descriptor_fire && descriptor_row_first && descriptor_chunk_index != '0)
        || (descriptor_fire && !descriptor_row_first
            && descriptor_chunk_index == '0)
        || (descriptor_fire && descriptor_row_last
            && descriptor_chunk_index != CHUNK_W'(descriptor_chunk_count - 1'b1))
        || (descriptor_fire && !descriptor_row_last
            && descriptor_chunk_index == CHUNK_W'(descriptor_chunk_count - 1'b1))
        || (descriptor_fire && descriptor_batch_last && !descriptor_row_last)
        || (descriptor_fire && descriptor_row_last && !descriptor_batch_last
            && loaded_contexts_q == CTX_COUNT_W'(CONTEXTS-1))
        || (descriptor_fire && (in_row_q || loaded_contexts_q != '0)
            && (descriptor_object_tag != object_tag_q
                || descriptor_chunk_count != chunk_count_q
                || descriptor_lane_tile_count != lane_tile_count_q))
        || (descriptor_fire && in_row_q
            && (descriptor_tag != tag_q[load_context_q]
                || descriptor_use_motion != use_motion_q[load_context_q]))
        || (weight_response_valid && !pending_q)
        || (weight_response_valid && pending_q && !response_contract_valid);

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_LOAD;
            faulted_q <= 1'b0;
            in_row_q <= 1'b0;
            loaded_contexts_q <= '0;
            batch_contexts_q <= '0;
            load_context_q <= '0;
            expected_chunk_q <= '0;
            object_tag_q <= '0;
            chunk_count_q <= '0;
            lane_tile_count_q <= '0;
            current_chunk_q <= '0;
            current_lane_tile_q <= '0;
            output_context_q <= '0;
            pending_q <= 1'b0;
            pending_chunk_last_q <= 1'b0;
            pending_bank_valid_q <= '0;
            pending_bank_context_q <= '0;
            pending_bank_slot_q <= '0;
            pending_bank_negative_q <= '0;
            for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
                tag_q[ctx] <= '0;
                use_motion_q[ctx] <= 1'b0;
                remaining_q[ctx] <= '0;
                negative_q[ctx] <= '0;
                source_count_q[ctx] <= '0;
                for (int chunk = 0; chunk < MAX_CHUNKS; chunk = chunk + 1) begin
                    source_descriptor_q[ctx][chunk] <= '0;
                    negative_descriptor_q[ctx][chunk] <= '0;
                end
                for (int lane = 0; lane < OUT_LANES; lane = lane + 1)
                    acc_q[ctx][lane] <= '0;
            end
        end else if (protocol_violation) begin
            state_q <= ST_LOAD;
            pending_q <= 1'b0;
            faulted_q <= 1'b1;
        end else begin
            if (descriptor_fire) begin
                source_descriptor_q[load_context_q][descriptor_chunk_index]
                    <= descriptor_source_bits;
                negative_descriptor_q[load_context_q][descriptor_chunk_index]
                    <= descriptor_negative_bits;
                if (descriptor_row_first) begin
                    tag_q[load_context_q] <= descriptor_tag;
                    use_motion_q[load_context_q] <= descriptor_use_motion;
                    if (loaded_contexts_q == '0) begin
                        object_tag_q <= descriptor_object_tag;
                        chunk_count_q <= descriptor_chunk_count;
                        lane_tile_count_q <= descriptor_lane_tile_count;
                    end
                end
                in_row_q <= !descriptor_row_last;
                if (descriptor_row_last)
                    expected_chunk_q <= '0;
                else
                    expected_chunk_q <= descriptor_chunk_index + CHUNK_W'(1);
                if (descriptor_row_last) begin
                    loaded_contexts_q <= loaded_contexts_q + CTX_COUNT_W'(1);
                    if (!descriptor_batch_last)
                        load_context_q <= load_context_q + CTX_W'(1);
                end
                if (descriptor_batch_last) begin
                    batch_contexts_q <= loaded_contexts_q + CTX_COUNT_W'(1);
                    current_chunk_q <= '0;
                    current_lane_tile_q <= '0;
                    output_context_q <= '0;
                    state_q <= ST_PREP;
                end
            end

            if (state_q == ST_PREP) begin
                for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
                    if (CTX_COUNT_W'(ctx) < batch_contexts_q) begin
                        remaining_q[ctx] <= source_descriptor_q[ctx][current_chunk_q];
                        negative_q[ctx] <= negative_descriptor_q[ctx][current_chunk_q];
                        if (current_chunk_q == '0) begin
                            source_count_q[ctx] <= '0;
                            for (int lane = 0; lane < OUT_LANES; lane = lane + 1)
                                acc_q[ctx][lane] <= '0;
                        end
                    end
                end
                state_q <= ST_ISSUE;
            end

            if (request_fire) begin
                pending_q <= 1'b1;
                pending_chunk_last_q <= request_chunk_last;
                pending_bank_valid_q <= weight_request_bank_valid;
                pending_bank_context_q <= weight_request_bank_context;
                pending_bank_slot_q <= weight_request_bank_slot;
                pending_bank_negative_q <= weight_request_bank_negative;
                for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
                    remaining_q[ctx] <= remaining_after_request[ctx];
                    source_count_q[ctx] <= source_count_q[ctx]
                        + SOURCE_COUNT_W'(request_context_sources[ctx]);
                end
            end

            if (state_q == ST_ISSUE && !selected_valid && !pending_q) begin
                // Zero-source chunks still consume their explicit DRAIN/control
                // cycle before the next resident descriptor is selected.
                if (current_chunk_q + CHUNK_W'(1) < chunk_count_q) begin
                    current_chunk_q <= current_chunk_q + CHUNK_W'(1);
                    state_q <= ST_PREP;
                end else begin
                    output_context_q <= '0;
                    state_q <= ST_OUTPUT;
                end
            end

            if (response_fire) begin
                for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
                    for (int lane = 0; lane < OUT_LANES; lane = lane + 1)
                        acc_q[ctx][lane] <= acc_q[ctx][lane] + response_sum[ctx][lane];
                end
                if (!request_fire) begin
                    pending_q <= 1'b0;
                    pending_bank_valid_q <= '0;
                end
                if (pending_chunk_last_q) begin
                    if (current_chunk_q + CHUNK_W'(1) < chunk_count_q) begin
                        current_chunk_q <= current_chunk_q + CHUNK_W'(1);
                        state_q <= ST_PREP;
                    end else begin
                        output_context_q <= '0;
                        state_q <= ST_OUTPUT;
                    end
                end
            end

            if (output_fire) begin
                if (CTX_COUNT_W'(output_context_q) + CTX_COUNT_W'(1)
                        < batch_contexts_q) begin
                    output_context_q <= output_context_q + CTX_W'(1);
                end else if (LANE_COUNT_W'(current_lane_tile_q) + LANE_COUNT_W'(1)
                        < lane_tile_count_q) begin
                    current_lane_tile_q <= current_lane_tile_q + LANE_TILE_W'(1);
                    current_chunk_q <= '0;
                    output_context_q <= '0;
                    state_q <= ST_PREP;
                end else begin
                    state_q <= ST_LOAD;
                    in_row_q <= 1'b0;
                    loaded_contexts_q <= '0;
                    batch_contexts_q <= '0;
                    load_context_q <= '0;
                    expected_chunk_q <= '0;
                    current_chunk_q <= '0;
                    current_lane_tile_q <= '0;
                    output_context_q <= '0;
                end
            end
        end
    end
endmodule

`default_nettype wire
