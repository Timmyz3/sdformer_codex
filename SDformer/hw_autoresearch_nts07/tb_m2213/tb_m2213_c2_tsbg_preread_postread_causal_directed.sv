`timescale 1ns/1ps
`default_nettype none

module m2213_directed_bank_memory #(
    parameter int BANK_ID = 0,
    parameter int LANES = 16
) (
    input logic clk_core, rst_core, req_valid,
    output logic req_ready,
    input logic [15:0] req_epoch,
    input logic [2:0] req_slot,
    input logic [31:0] req_generation,
    input logic [23:0] req_tag,
    input logic [2:0] req_output_block, req_slice,
    input logic [11:0] req_source_channel,
    input logic req_accept,
    output logic rsp_valid,
    input logic rsp_ready,
    output logic [15:0] rsp_epoch,
    output logic [2:0] rsp_slot,
    output logic [31:0] rsp_generation,
    output logic [23:0] rsp_tag,
    output logic signed [7:0] rsp_weight [0:LANES-1],
    input logic rsp_accept,
    output logic [31:0] request_count, response_count, request_stall_count
);
    logic pending_q;
    logic [3:0] delay_q;
    logic [31:0] cycle_q;
    logic [15:0] epoch_q;
    logic [2:0] slot_q, slice_q;
    logic [31:0] generation_q;
    logic [23:0] tag_q;
    logic [11:0] channel_q;

    function automatic integer weight_value(
        input integer channel, slice, lane);
        integer raw;
        begin
            if (channel == 0 && slice == 0 && lane == 0)
                weight_value = -128;
            else begin
                raw = (channel * 7 + slice * 5 + lane * 3) % 31;
                weight_value = raw - 15;
            end
        end
    endfunction

    assign req_ready = !pending_q && ((cycle_q + BANK_ID * 3 + 1) % 7 != 0);
    assign rsp_valid = pending_q && delay_q == 0;
    assign rsp_epoch = epoch_q;
    assign rsp_slot = slot_q;
    assign rsp_generation = generation_q;
    assign rsp_tag = tag_q;
    always_comb begin
        for (int lane = 0; lane < LANES; lane++)
            rsp_weight[lane] = 8'(weight_value(
                int'(channel_q), int'(slice_q), lane));
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            pending_q <= 0;
            delay_q <= 0;
            cycle_q <= 0;
            epoch_q <= 0;
            slot_q <= 0;
            slice_q <= 0;
            generation_q <= 0;
            tag_q <= 0;
            channel_q <= 0;
            request_count <= 0;
            response_count <= 0;
            request_stall_count <= 0;
        end else begin
            cycle_q <= cycle_q + 1'b1;
            if (req_valid && !req_ready)
                request_stall_count <= request_stall_count + 1'b1;
            if (req_accept) begin
                if (pending_q)
                    $fatal(1, "M2213 bank accepted over live request");
                if (req_source_channel[2:0] != BANK_ID[2:0])
                    $fatal(1, "M2213 bank/channel mismatch");
                pending_q <= 1;
                delay_q <= 8 - BANK_ID;
                epoch_q <= req_epoch;
                slot_q <= req_slot;
                slice_q <= req_slice;
                generation_q <= req_generation;
                tag_q <= req_tag;
                channel_q <= req_source_channel;
                request_count <= request_count + 1'b1;
            end else if (pending_q && delay_q != 0) begin
                delay_q <= delay_q - 1'b1;
            end
            if (rsp_accept) begin
                if (!pending_q)
                    $fatal(1, "M2213 bank accepted response without request");
                pending_q <= 0;
                response_count <= response_count + 1'b1;
            end
        end
    end
endmodule

interface m2213_side_if;
    logic load_ready, load_accept;
    logic [7:0] mem_req_valid, mem_req_ready, mem_req_accept;
    logic [15:0] mem_req_epoch [0:7];
    logic [2:0] mem_req_slot [0:7];
    logic [31:0] mem_req_generation [0:7];
    logic [23:0] mem_req_tag [0:7];
    logic [2:0] mem_req_output_block [0:7], mem_req_slice [0:7];
    logic [11:0] mem_req_source_channel [0:7];
    logic [7:0] mem_rsp_valid, mem_rsp_ready, mem_rsp_accept;
    logic [15:0] mem_rsp_epoch [0:7];
    logic [2:0] mem_rsp_slot [0:7];
    logic [31:0] mem_rsp_generation [0:7];
    logic [23:0] mem_rsp_tag [0:7];
    logic signed [7:0] mem_rsp_weight [0:7][0:15];
    logic bridge_valid, bridge_ready, bridge_accept;
    logic [2:0] bridge_context;
    logic [5:0] bridge_group;
    logic bridge_half;
    logic [2:0] bridge_slice;
    logic [7:0] bridge_bank_valid;
    logic [11:0] bridge_source_channel [0:7];
    logic signed [1:0] bridge_source_value [0:7];
    logic signed [8:0] bridge_effective_weight [0:7][0:15];
    logic commit_valid, commit_ready, commit_terminal, commit_accept;
    logic [2:0] commit_context, commit_slice;
    logic [23:0] commit_tag;
    logic signed [23:0] commit_accumulator [0:15];
    logic bundle_done_valid, bundle_done_ready;
    logic protocol_error, stale_response_seen, numeric_overflow, busy;
    logic [31:0] cycle_count, row_access_count, cache_hit_count;
    logic [31:0] cache_miss_count, cache_eviction_count;
    logic [31:0] weight_bundle_beat_count, scalar_bank_request_count;
    logic [31:0] scalar_bank_response_count, issue_count, product_count;
    logic [31:0] commit_count;
    logic [31:0] postread_row_count, postread_bundle_request_count;
    logic [31:0] postread_bundle_response_count, postread_bank_request_count;
    logic [31:0] postread_bank_response_count, postread_identity_accept_count;
    logic [31:0] memory_request_count [0:7];
    logic [31:0] memory_response_count [0:7];
    logic [31:0] memory_stall_count [0:7];
endinterface

module tb_m2213_c2_tsbg_preread_postread_causal_directed;
    localparam int BUNDLE = 4;
    localparam int GROUPS = 6;
    localparam int SLICES = 6;
    localparam int LANES = 16;
    localparam int ROWS = BUNDLE * GROUPS;
    localparam int GROUP_MAJOR_HITS = (BUNDLE - 1) * GROUPS;
    localparam int BUNDLES_PER_ROW = 2 * SLICES;
    localparam int BANKS_PER_ROW = BUNDLES_PER_ROW * 8;
    localparam int EXPECTED_ISSUES = ROWS * BUNDLES_PER_ROW;
    localparam int EXPECTED_PRODUCTS = EXPECTED_ISSUES * LANES;
    localparam int EXPECTED_COMMITS = BUNDLE * SLICES;

    logic clk_core = 0;
    logic rst_core = 1;
    logic load_valid_o, load_valid_l, load_valid_p;
    logic [2:0] load_context;
    logic [23:0] load_tag;
    logic [5:0] load_group;
    logic [15:0] load_source_active, load_source_sign;
    logic load_last;
    integer tb_cycle;
    integer signed expected [0:BUNDLE-1][0:SLICES-1][0:LANES-1];
    logic observed_o [0:BUNDLE-1][0:SLICES-1];
    logic observed_l [0:BUNDLE-1][0:SLICES-1];
    logic observed_p [0:BUNDLE-1][0:SLICES-1];
    integer mismatch_o, mismatch_l, mismatch_p;
    logic done_o, done_l, done_p;
    m2213_side_if ordinary();
    m2213_side_if postread();
    m2213_side_if preread();

    always #1.5 clk_core = ~clk_core;
    always @(posedge clk_core) begin
        if (rst_core) tb_cycle <= 0;
        else tb_cycle <= tb_cycle + 1;
    end

`define CONNECT_FROZEN(inst, side, mode, load_v) \
    m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend #( \
        .SCHEDULE_MODE(mode), .SOURCE_GROUPS(GROUPS)) inst ( \
        .clk_core(clk_core), .rst_core(rst_core), \
        .load_valid(load_v), .load_ready(side.load_ready), \
        .load_context(load_context), .load_tag(load_tag), \
        .load_group(load_group), .load_source_active(load_source_active), \
        .load_source_sign(load_source_sign), .load_last(load_last), \
        .load_accept(side.load_accept), \
        .mem_req_valid(side.mem_req_valid), .mem_req_ready(side.mem_req_ready), \
        .mem_req_epoch(side.mem_req_epoch), .mem_req_slot(side.mem_req_slot), \
        .mem_req_generation(side.mem_req_generation), \
        .mem_req_tag(side.mem_req_tag), \
        .mem_req_output_block(side.mem_req_output_block), \
        .mem_req_slice(side.mem_req_slice), \
        .mem_req_source_channel(side.mem_req_source_channel), \
        .mem_req_accept(side.mem_req_accept), \
        .mem_rsp_valid(side.mem_rsp_valid), .mem_rsp_ready(side.mem_rsp_ready), \
        .mem_rsp_epoch(side.mem_rsp_epoch), .mem_rsp_slot(side.mem_rsp_slot), \
        .mem_rsp_generation(side.mem_rsp_generation), \
        .mem_rsp_tag(side.mem_rsp_tag), .mem_rsp_weight(side.mem_rsp_weight), \
        .mem_rsp_accept(side.mem_rsp_accept), \
        .bridge_valid(side.bridge_valid), .bridge_ready(side.bridge_ready), \
        .bridge_context(side.bridge_context), .bridge_group(side.bridge_group), \
        .bridge_half(side.bridge_half), .bridge_slice(side.bridge_slice), \
        .bridge_bank_valid(side.bridge_bank_valid), \
        .bridge_source_channel(side.bridge_source_channel), \
        .bridge_source_value(side.bridge_source_value), \
        .bridge_effective_weight(side.bridge_effective_weight), \
        .bridge_accept(side.bridge_accept), \
        .commit_valid(side.commit_valid), .commit_ready(side.commit_ready), \
        .commit_context(side.commit_context), .commit_tag(side.commit_tag), \
        .commit_slice(side.commit_slice), \
        .commit_accumulator(side.commit_accumulator), \
        .commit_terminal(side.commit_terminal), \
        .commit_accept(side.commit_accept), \
        .bundle_done_valid(side.bundle_done_valid), \
        .bundle_done_ready(side.bundle_done_ready), \
        .protocol_error(side.protocol_error), \
        .stale_response_seen(side.stale_response_seen), \
        .numeric_overflow(side.numeric_overflow), .busy(side.busy), \
        .debug_cycle_count(side.cycle_count), \
        .debug_row_access_count(side.row_access_count), \
        .debug_cache_hit_count(side.cache_hit_count), \
        .debug_cache_miss_count(side.cache_miss_count), \
        .debug_cache_eviction_count(side.cache_eviction_count), \
        .debug_weight_bundle_beat_count(side.weight_bundle_beat_count), \
        .debug_scalar_bank_request_count(side.scalar_bank_request_count), \
        .debug_scalar_bank_response_count(side.scalar_bank_response_count), \
        .debug_issue_count(side.issue_count), \
        .debug_signed_product_count(side.product_count), \
        .debug_commit_count(side.commit_count))

    `CONNECT_FROZEN(dut_ordinary, ordinary, 0, load_valid_o);
    `CONNECT_FROZEN(dut_preread, preread, 1, load_valid_p);
`undef CONNECT_FROZEN

    m2213_c2_tsbg_b4_postread_causal_frontend #(
        .SOURCE_GROUPS(GROUPS)
    ) dut_postread (
        .clk_core(clk_core), .rst_core(rst_core),
        .load_valid(load_valid_l), .load_ready(postread.load_ready),
        .load_context(load_context), .load_tag(load_tag),
        .load_group(load_group), .load_source_active(load_source_active),
        .load_source_sign(load_source_sign), .load_last(load_last),
        .load_accept(postread.load_accept),
        .mem_req_valid(postread.mem_req_valid),
        .mem_req_ready(postread.mem_req_ready),
        .mem_req_epoch(postread.mem_req_epoch),
        .mem_req_slot(postread.mem_req_slot),
        .mem_req_generation(postread.mem_req_generation),
        .mem_req_tag(postread.mem_req_tag),
        .mem_req_output_block(postread.mem_req_output_block),
        .mem_req_slice(postread.mem_req_slice),
        .mem_req_source_channel(postread.mem_req_source_channel),
        .mem_req_accept(postread.mem_req_accept),
        .mem_rsp_valid(postread.mem_rsp_valid),
        .mem_rsp_ready(postread.mem_rsp_ready),
        .mem_rsp_epoch(postread.mem_rsp_epoch),
        .mem_rsp_slot(postread.mem_rsp_slot),
        .mem_rsp_generation(postread.mem_rsp_generation),
        .mem_rsp_tag(postread.mem_rsp_tag),
        .mem_rsp_weight(postread.mem_rsp_weight),
        .mem_rsp_accept(postread.mem_rsp_accept),
        .bridge_valid(postread.bridge_valid),
        .bridge_ready(postread.bridge_ready),
        .bridge_context(postread.bridge_context),
        .bridge_group(postread.bridge_group), .bridge_half(postread.bridge_half),
        .bridge_slice(postread.bridge_slice),
        .bridge_bank_valid(postread.bridge_bank_valid),
        .bridge_source_channel(postread.bridge_source_channel),
        .bridge_source_value(postread.bridge_source_value),
        .bridge_effective_weight(postread.bridge_effective_weight),
        .bridge_accept(postread.bridge_accept),
        .commit_valid(postread.commit_valid),
        .commit_ready(postread.commit_ready),
        .commit_context(postread.commit_context),
        .commit_tag(postread.commit_tag),
        .commit_slice(postread.commit_slice),
        .commit_accumulator(postread.commit_accumulator),
        .commit_terminal(postread.commit_terminal),
        .commit_accept(postread.commit_accept),
        .bundle_done_valid(postread.bundle_done_valid),
        .bundle_done_ready(postread.bundle_done_ready),
        .protocol_error(postread.protocol_error),
        .stale_response_seen(postread.stale_response_seen),
        .numeric_overflow(postread.numeric_overflow), .busy(postread.busy),
        .debug_cycle_count(postread.cycle_count),
        .debug_row_access_count(postread.row_access_count),
        .debug_cache_hit_count(postread.cache_hit_count),
        .debug_cache_miss_count(postread.cache_miss_count),
        .debug_cache_eviction_count(postread.cache_eviction_count),
        .debug_weight_bundle_beat_count(postread.weight_bundle_beat_count),
        .debug_scalar_bank_request_count(postread.scalar_bank_request_count),
        .debug_scalar_bank_response_count(postread.scalar_bank_response_count),
        .debug_issue_count(postread.issue_count),
        .debug_signed_product_count(postread.product_count),
        .debug_commit_count(postread.commit_count),
        .debug_postread_row_count(postread.postread_row_count),
        .debug_postread_bundle_request_count(
            postread.postread_bundle_request_count),
        .debug_postread_bundle_response_count(
            postread.postread_bundle_response_count),
        .debug_postread_bank_request_count(
            postread.postread_bank_request_count),
        .debug_postread_bank_response_count(
            postread.postread_bank_response_count),
        .debug_postread_identity_accept_count(
            postread.postread_identity_accept_count));

    assign ordinary.postread_row_count = 0;
    assign ordinary.postread_bundle_request_count = 0;
    assign ordinary.postread_bundle_response_count = 0;
    assign ordinary.postread_bank_request_count = 0;
    assign ordinary.postread_bank_response_count = 0;
    assign ordinary.postread_identity_accept_count = 0;
    assign preread.postread_row_count = 0;
    assign preread.postread_bundle_request_count = 0;
    assign preread.postread_bundle_response_count = 0;
    assign preread.postread_bank_request_count = 0;
    assign preread.postread_bank_response_count = 0;
    assign preread.postread_identity_accept_count = 0;

    for (genvar bank = 0; bank < 8; bank++) begin : g_memory
`define CONNECT_MEMORY(inst, side) \
        m2213_directed_bank_memory #(.BANK_ID(bank)) inst ( \
            .clk_core(clk_core), .rst_core(rst_core), \
            .req_valid(side.mem_req_valid[bank]), \
            .req_ready(side.mem_req_ready[bank]), \
            .req_epoch(side.mem_req_epoch[bank]), \
            .req_slot(side.mem_req_slot[bank]), \
            .req_generation(side.mem_req_generation[bank]), \
            .req_tag(side.mem_req_tag[bank]), \
            .req_output_block(side.mem_req_output_block[bank]), \
            .req_slice(side.mem_req_slice[bank]), \
            .req_source_channel(side.mem_req_source_channel[bank]), \
            .req_accept(side.mem_req_accept[bank]), \
            .rsp_valid(side.mem_rsp_valid[bank]), \
            .rsp_ready(side.mem_rsp_ready[bank]), \
            .rsp_epoch(side.mem_rsp_epoch[bank]), \
            .rsp_slot(side.mem_rsp_slot[bank]), \
            .rsp_generation(side.mem_rsp_generation[bank]), \
            .rsp_tag(side.mem_rsp_tag[bank]), \
            .rsp_weight(side.mem_rsp_weight[bank]), \
            .rsp_accept(side.mem_rsp_accept[bank]), \
            .request_count(side.memory_request_count[bank]), \
            .response_count(side.memory_response_count[bank]), \
            .request_stall_count(side.memory_stall_count[bank]))
        `CONNECT_MEMORY(memory_ordinary, ordinary);
        `CONNECT_MEMORY(memory_postread, postread);
        `CONNECT_MEMORY(memory_preread, preread);
`undef CONNECT_MEMORY
    end

    m2213_c2_tsbg_postread_causal_assertions sva_postread (
        .clk_core(clk_core), .rst_core(rst_core),
        .mem_req_valid(postread.mem_req_valid),
        .mem_req_ready(postread.mem_req_ready),
        .mem_req_accept(postread.mem_req_accept),
        .mem_rsp_valid(postread.mem_rsp_valid),
        .mem_rsp_ready(postread.mem_rsp_ready),
        .mem_rsp_accept(postread.mem_rsp_accept),
        .bridge_valid(postread.bridge_valid),
        .bridge_ready(postread.bridge_ready),
        .bridge_accept(postread.bridge_accept),
        .commit_valid(postread.commit_valid),
        .commit_ready(postread.commit_ready),
        .commit_accept(postread.commit_accept),
        .commit_context(postread.commit_context),
        .commit_slice(postread.commit_slice),
        .commit_tag(postread.commit_tag),
        .commit_terminal(postread.commit_terminal),
        .commit_accumulator(postread.commit_accumulator),
        .bundle_done_valid(postread.bundle_done_valid),
        .protocol_error(postread.protocol_error),
        .stale_response_seen(postread.stale_response_seen),
        .numeric_overflow(postread.numeric_overflow),
        .debug_postread_row_count(postread.postread_row_count),
        .debug_postread_bundle_request_count(
            postread.postread_bundle_request_count),
        .debug_postread_bundle_response_count(
            postread.postread_bundle_response_count),
        .debug_postread_bank_request_count(
            postread.postread_bank_request_count),
        .debug_postread_bank_response_count(
            postread.postread_bank_response_count),
        .debug_postread_identity_accept_count(
            postread.postread_identity_accept_count));

    function automatic integer directed_weight(
        input integer channel, slice, lane);
        integer raw;
        begin
            if (channel == 0 && slice == 0 && lane == 0)
                directed_weight = -128;
            else begin
                raw = (channel * 7 + slice * 5 + lane * 3) % 31;
                directed_weight = raw - 15;
            end
        end
    endfunction

    task automatic prepare_descriptor(input integer context, group_index);
        integer lower_source, upper_source, lower_value, upper_value;
        begin
            load_source_active = 0;
            load_source_sign = 0;
            lower_source = (context + group_index * 3) % 8;
            upper_source = 8 + ((context * 3 + group_index) % 8);
            lower_value = ((context + group_index) % 2) ? -1 : 1;
            upper_value = -lower_value;
            load_source_active[lower_source] = 1;
            load_source_active[upper_source] = 1;
            load_source_sign[lower_source] = lower_value < 0;
            load_source_sign[upper_source] = upper_value < 0;
            for (int slice = 0; slice < SLICES; slice++)
                for (int lane = 0; lane < LANES; lane++) begin
                    expected[context][slice][lane] += lower_value
                        * directed_weight(group_index * 16 + lower_source,
                                          slice, lane);
                    expected[context][slice][lane] += upper_value
                        * directed_weight(group_index * 16 + upper_source,
                                          slice, lane);
                end
        end
    endtask

    task automatic load_descriptor_to_three;
        logic accepted_o, accepted_l, accepted_p;
        integer waited;
        begin
            accepted_o = 0;
            accepted_l = 0;
            accepted_p = 0;
            @(negedge clk_core);
            load_valid_o = 1;
            load_valid_l = 1;
            load_valid_p = 1;
            for (waited = 0; waited < 10000
                    && !(accepted_o && accepted_l && accepted_p);
                 waited = waited + 1) begin
                @(posedge clk_core);
                if (ordinary.load_accept) begin
                    accepted_o = 1;
                    load_valid_o <= 0;
                end
                if (postread.load_accept) begin
                    accepted_l = 1;
                    load_valid_l <= 0;
                end
                if (preread.load_accept) begin
                    accepted_p = 1;
                    load_valid_p <= 0;
                end
            end
            if (!(accepted_o && accepted_l && accepted_p))
                $fatal(1, "M2213 descriptor load timeout ctx=%0d group=%0d",
                       load_context, load_group);
            @(negedge clk_core);
            load_valid_o = 0;
            load_valid_l = 0;
            load_valid_p = 0;
        end
    endtask

    task automatic check_commit(input integer axis);
        integer ctx, slice;
        begin
            if (axis == 0) begin
                ctx = ordinary.commit_context;
                slice = ordinary.commit_slice;
                if (observed_o[ctx][slice])
                    $fatal(1, "M2213 ordinary duplicate commit");
                observed_o[ctx][slice] = 1;
                if (ordinary.commit_tag !== 24'hA30000 + ctx
                        || ordinary.commit_terminal !== (slice == 5))
                    $fatal(1, "M2213 ordinary commit identity mismatch");
                for (int lane = 0; lane < LANES; lane++)
                    if (ordinary.commit_accumulator[lane]
                            !== expected[ctx][slice][lane])
                        mismatch_o = mismatch_o + 1;
            end else if (axis == 1) begin
                ctx = postread.commit_context;
                slice = postread.commit_slice;
                if (observed_l[ctx][slice])
                    $fatal(1, "M2213 post-read duplicate commit");
                observed_l[ctx][slice] = 1;
                if (postread.commit_tag !== 24'hA30000 + ctx
                        || postread.commit_terminal !== (slice == 5))
                    $fatal(1, "M2213 post-read commit identity mismatch");
                for (int lane = 0; lane < LANES; lane++)
                    if (postread.commit_accumulator[lane]
                            !== expected[ctx][slice][lane])
                        mismatch_l = mismatch_l + 1;
            end else begin
                ctx = preread.commit_context;
                slice = preread.commit_slice;
                if (observed_p[ctx][slice])
                    $fatal(1, "M2213 pre-read duplicate commit");
                observed_p[ctx][slice] = 1;
                if (preread.commit_tag !== 24'hA30000 + ctx
                        || preread.commit_terminal !== (slice == 5))
                    $fatal(1, "M2213 pre-read commit identity mismatch");
                for (int lane = 0; lane < LANES; lane++)
                    if (preread.commit_accumulator[lane]
                            !== expected[ctx][slice][lane])
                        mismatch_p = mismatch_p + 1;
            end
        end
    endtask

    always_comb begin
        ordinary.bridge_ready = tb_cycle % 11 != 3;
        postread.bridge_ready = tb_cycle % 11 != 3;
        preread.bridge_ready = tb_cycle % 11 != 3;
        ordinary.commit_ready = tb_cycle % 13 != 5;
        postread.commit_ready = tb_cycle % 13 != 5;
        preread.commit_ready = tb_cycle % 13 != 5;
        ordinary.bundle_done_ready = 1;
        postread.bundle_done_ready = 1;
        preread.bundle_done_ready = 1;
    end

    always @(posedge clk_core) begin
        if (!rst_core) begin
            if (ordinary.commit_accept) check_commit(0);
            if (postread.commit_accept) check_commit(1);
            if (preread.commit_accept) check_commit(2);
            if (ordinary.bundle_done_valid) done_o <= 1;
            if (postread.bundle_done_valid) done_l <= 1;
            if (preread.bundle_done_valid) done_p <= 1;
        end
    end

    initial begin
        tb_cycle = 0;
        load_valid_o = 0;
        load_valid_l = 0;
        load_valid_p = 0;
        load_context = 0;
        load_tag = 0;
        load_group = 0;
        load_source_active = 0;
        load_source_sign = 0;
        load_last = 0;
        mismatch_o = 0;
        mismatch_l = 0;
        mismatch_p = 0;
        done_o = 0;
        done_l = 0;
        done_p = 0;
        for (int context = 0; context < BUNDLE; context++)
            for (int slice = 0; slice < SLICES; slice++) begin
                observed_o[context][slice] = 0;
                observed_l[context][slice] = 0;
                observed_p[context][slice] = 0;
                for (int lane = 0; lane < LANES; lane++)
                    expected[context][slice][lane] = 0;
            end
        repeat (6) @(posedge clk_core);
        rst_core = 0;

        for (int context = 0; context < BUNDLE; context++) begin
            for (int group = 0; group < GROUPS; group++) begin
                prepare_descriptor(context, group);
                load_context = 3'(context);
                load_tag = 24'hA30000 + context;
                load_group = 6'(group);
                load_last = group == GROUPS - 1;
                load_descriptor_to_three();
            end
        end

        fork : completion_watchdog
            begin wait (done_o && done_l && done_p); end
            begin
                repeat (200000) @(posedge clk_core);
                $fatal(1, "M2213 whole-workload timeout");
            end
        join_any
        disable completion_watchdog;
        repeat (4) @(posedge clk_core);

        if (ordinary.protocol_error || postread.protocol_error
                || preread.protocol_error || ordinary.stale_response_seen
                || postread.stale_response_seen || preread.stale_response_seen
                || ordinary.numeric_overflow || postread.numeric_overflow
                || preread.numeric_overflow)
            $fatal(1, "M2213 protocol or numeric fault");
        if (mismatch_o != 0 || mismatch_l != 0 || mismatch_p != 0)
            $fatal(1, "M2213 golden mismatch o=%0d post=%0d pre=%0d",
                   mismatch_o, mismatch_l, mismatch_p);
        for (int context = 0; context < BUNDLE; context++)
            for (int slice = 0; slice < SLICES; slice++)
                if (!observed_o[context][slice]
                        || !observed_l[context][slice]
                        || !observed_p[context][slice])
                    $fatal(1, "M2213 missing commit ctx=%0d slice=%0d",
                           context, slice);

        if (ordinary.row_access_count != ROWS
                || postread.row_access_count != ROWS
                || preread.row_access_count != ROWS)
            $fatal(1, "M2213 row count mismatch");
        if (ordinary.cache_hit_count != 0
                || ordinary.cache_miss_count != ROWS)
            $fatal(1, "M2213 ordinary LRU4 premise mismatch");
        if (postread.cache_hit_count != GROUP_MAJOR_HITS
                || preread.cache_hit_count != GROUP_MAJOR_HITS
                || postread.cache_miss_count != GROUPS
                || preread.cache_miss_count != GROUPS)
            $fatal(1, "M2213 group-major LRU4 premise mismatch");
        if (ordinary.scalar_bank_request_count != ROWS * BANKS_PER_ROW
                || postread.scalar_bank_request_count != ROWS * BANKS_PER_ROW
                || preread.scalar_bank_request_count != GROUPS * BANKS_PER_ROW)
            $fatal(1, "M2213 SRAM request ledger mismatch");
        if (postread.postread_row_count != GROUP_MAJOR_HITS
                || postread.postread_bundle_request_count
                    != GROUP_MAJOR_HITS * BUNDLES_PER_ROW
                || postread.postread_bundle_response_count
                    != GROUP_MAJOR_HITS * BUNDLES_PER_ROW
                || postread.postread_bank_request_count
                    != GROUP_MAJOR_HITS * BANKS_PER_ROW
                || postread.postread_bank_response_count
                    != GROUP_MAJOR_HITS * BANKS_PER_ROW
                || postread.postread_identity_accept_count
                    != GROUP_MAJOR_HITS * BUNDLES_PER_ROW)
            $fatal(1, "M2213 post-read causal ledger mismatch");
        if (postread.scalar_bank_request_count
                - preread.scalar_bank_request_count
                != postread.postread_bank_request_count)
            $fatal(1, "M2213 pre-read suppression identity mismatch");
        if (ordinary.issue_count != EXPECTED_ISSUES
                || postread.issue_count != EXPECTED_ISSUES
                || preread.issue_count != EXPECTED_ISSUES
                || ordinary.product_count != EXPECTED_PRODUCTS
                || postread.product_count != EXPECTED_PRODUCTS
                || preread.product_count != EXPECTED_PRODUCTS
                || ordinary.commit_count != EXPECTED_COMMITS
                || postread.commit_count != EXPECTED_COMMITS
                || preread.commit_count != EXPECTED_COMMITS)
            $fatal(1, "M2213 issue/product/commit mismatch");
        for (int bank = 0; bank < 8; bank++) begin
            if (ordinary.memory_request_count[bank]
                    != ordinary.scalar_bank_request_count / 8
                    || postread.memory_request_count[bank]
                    != postread.scalar_bank_request_count / 8
                    || preread.memory_request_count[bank]
                    != preread.scalar_bank_request_count / 8
                    || ordinary.memory_response_count[bank]
                    != ordinary.memory_request_count[bank]
                    || postread.memory_response_count[bank]
                    != postread.memory_request_count[bank]
                    || preread.memory_response_count[bank]
                    != preread.memory_request_count[bank])
                $fatal(1, "M2213 physical bank ledger mismatch bank=%0d", bank);
        end

        $display("M2213_COVER rows=%0d hits_post=%0d hits_pre=%0d real_postread_rows=%0d postread_bundle_req=%0d postread_bundle_rsp=%0d postread_bank_req=%0d postread_bank_rsp=%0d identity_rsp=%0d commits_each=%0d products_each=%0d golden_mismatches=%0d",
                 ROWS, postread.cache_hit_count, preread.cache_hit_count,
                 postread.postread_row_count,
                 postread.postread_bundle_request_count,
                 postread.postread_bundle_response_count,
                 postread.postread_bank_request_count,
                 postread.postread_bank_response_count,
                 postread.postread_identity_accept_count,
                 postread.commit_count, postread.product_count,
                 mismatch_o + mismatch_l + mismatch_p);
        $display("RAW_PASS_M2215_M2213_PREREAD_POSTREAD_CAUSAL_DIRECTED ordinary_reads=%0d postread_reads=%0d preread_reads=%0d suppressed_reads=%0d ordinary_cycles=%0d postread_cycles=%0d preread_cycles=%0d",
                 ordinary.scalar_bank_request_count,
                 postread.scalar_bank_request_count,
                 preread.scalar_bank_request_count,
                 postread.scalar_bank_request_count
                    - preread.scalar_bank_request_count,
                 ordinary.cycle_count, postread.cycle_count,
                 preread.cycle_count);
        $finish;
    end
endmodule

`default_nettype wire
