`timescale 1ns/1ps
`default_nettype none

// M2149 is a genuinely single-axis native-SAIF causal preflight.  Unlike the
// failed M2142 source, this file neither imports nor instantiates a parent
// dual-axis testbench.  It directly instantiates exactly one M2018 frontend at
// SCHEDULE_MODE=0 and contains its own frozen fixture loader, memory service,
// arithmetic scoreboard, completion ledger, and SAIF boundary controls.
//
// The first stop is observational: the VCS native activity observer has
// already watched reset and all 383 preload cycles.  The 228-element census
// neither forces nor deposits DUT state.  UCLI then resets activity history
// and measures the exact 20,292-cycle execution window.
module m2149_directed_scalar_bank_memory #(
    parameter int BANK_ID = 0,
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int EPOCH_BITS = 16,
    parameter int GENERATION_BITS = 32,
    parameter int LANES = 16
) (
    input  logic clk_core,
    input  logic rst_core,
    input  logic req_valid,
    output logic req_ready,
    input  logic [EPOCH_BITS-1:0] req_epoch,
    input  logic [2:0] req_slot,
    input  logic [GENERATION_BITS-1:0] req_generation,
    input  logic [TAG_BITS-1:0] req_tag,
    input  logic [2:0] req_output_block,
    input  logic [2:0] req_slice,
    input  logic [CHANNEL_BITS-1:0] req_source_channel,
    input  logic req_accept,
    output logic rsp_valid,
    input  logic rsp_ready,
    output logic [EPOCH_BITS-1:0] rsp_epoch,
    output logic [2:0] rsp_slot,
    output logic [GENERATION_BITS-1:0] rsp_generation,
    output logic [TAG_BITS-1:0] rsp_tag,
    output logic signed [7:0] rsp_weight [0:LANES-1],
    input  logic rsp_accept,
    output logic [31:0] request_count,
    output logic [31:0] response_count,
    output logic [31:0] request_stall_count
);
    logic pending_q;
    logic [3:0] delay_q;
    logic [31:0] cycle_q;
    logic [EPOCH_BITS-1:0] epoch_q;
    logic [2:0] slot_q, block_q, slice_q;
    logic [GENERATION_BITS-1:0] generation_q;
    logic [TAG_BITS-1:0] tag_q;
    logic [CHANNEL_BITS-1:0] channel_q;
    integer group_index, half_index, raw_weight;

    assign req_ready = !pending_q && ((cycle_q + BANK_ID * 2) % 7 != 0);
    assign rsp_valid = pending_q && delay_q == 0;
    assign rsp_epoch = epoch_q;
    assign rsp_slot = slot_q;
    assign rsp_generation = generation_q;
    assign rsp_tag = tag_q;

    always_comb begin
        group_index = int'(channel_q) / 16;
        half_index = (int'(channel_q) / 8) % 2;
        for (int lane = 0; lane < LANES; lane++) begin
            raw_weight = (group_index * 17 + half_index * 11
                          + int'(slice_q) * 7 + BANK_ID * 5
                          + lane * 3) % 255 - 127;
            if (group_index == 0 && half_index == 0 && slice_q == 0
                    && BANK_ID == 0 && lane == 0)
                raw_weight = -128;
            rsp_weight[lane] = raw_weight;
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            pending_q <= 0;
            delay_q <= 0;
            cycle_q <= 0;
            epoch_q <= 0;
            slot_q <= 0;
            generation_q <= 0;
            tag_q <= 0;
            block_q <= 0;
            slice_q <= 0;
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
                    $fatal(1, "M2149 memory overwrote live request");
                if (req_source_channel[2:0] != BANK_ID[2:0])
                    $fatal(1, "M2149 source-channel bank mismatch");
                pending_q <= 1;
                delay_q <= 8 - BANK_ID;
                epoch_q <= req_epoch;
                slot_q <= req_slot;
                generation_q <= req_generation;
                tag_q <= req_tag;
                block_q <= req_output_block;
                slice_q <= req_slice;
                channel_q <= req_source_channel;
                request_count <= request_count + 1'b1;
            end else if (pending_q && delay_q != 0) begin
                delay_q <= delay_q - 1'b1;
            end
            if (rsp_accept) begin
                pending_q <= 0;
                response_count <= response_count + 1'b1;
            end
        end
    end
endmodule

interface m2149_ordinary_side_if #(
    parameter int TAG_BITS=24, CHANNEL_BITS=12, EPOCH_BITS=16,
    parameter int GENERATION_BITS=32, LANES=16
);
    logic load_ready, load_accept;
    logic [7:0] mem_req_valid, mem_req_ready, mem_req_accept;
    logic [EPOCH_BITS-1:0] mem_req_epoch [0:7];
    logic [2:0] mem_req_slot [0:7];
    logic [GENERATION_BITS-1:0] mem_req_generation [0:7];
    logic [TAG_BITS-1:0] mem_req_tag [0:7];
    logic [2:0] mem_req_output_block [0:7], mem_req_slice [0:7];
    logic [CHANNEL_BITS-1:0] mem_req_source_channel [0:7];
    logic [7:0] mem_rsp_valid, mem_rsp_ready, mem_rsp_accept;
    logic [EPOCH_BITS-1:0] mem_rsp_epoch [0:7];
    logic [2:0] mem_rsp_slot [0:7];
    logic [GENERATION_BITS-1:0] mem_rsp_generation [0:7];
    logic [TAG_BITS-1:0] mem_rsp_tag [0:7];
    logic signed [7:0] mem_rsp_weight [0:7][0:LANES-1];
    logic bridge_valid, bridge_ready, bridge_accept;
    logic [2:0] bridge_context;
    logic [5:0] bridge_group;
    logic bridge_half;
    logic [2:0] bridge_slice;
    logic [7:0] bridge_bank_valid;
    logic [CHANNEL_BITS-1:0] bridge_source_channel [0:7];
    logic signed [1:0] bridge_source_value [0:7];
    logic signed [8:0] bridge_effective_weight [0:7][0:LANES-1];
    logic commit_valid, commit_ready, commit_terminal, commit_accept;
    logic [2:0] commit_context, commit_slice;
    logic [TAG_BITS-1:0] commit_tag;
    logic signed [23:0] commit_accumulator [0:LANES-1];
    logic bundle_done_valid, bundle_done_ready;
    logic protocol_error, stale_response_seen, numeric_overflow, busy;
    logic [31:0] cycle_count, row_access_count, cache_hit_count;
    logic [31:0] cache_miss_count, cache_eviction_count;
    logic [31:0] weight_bundle_beat_count, scalar_bank_request_count;
    logic [31:0] scalar_bank_response_count, issue_count, product_count;
    logic [31:0] commit_count;
    logic [31:0] memory_request_count [0:7];
    logic [31:0] memory_response_count [0:7];
    logic [31:0] memory_stall_count [0:7];
endinterface

module tb_m2149_m2018_ordinary_single_axis_native_saif_preflight;
    localparam int BUNDLE = 4;
    localparam int GROUPS = 48;
    localparam int SLICES = 6;
    localparam int LANES = 16;
    localparam int FROZEN_WORKLOAD_SLOT = 42;
    localparam int FROZEN_PRELOAD_CYCLES = 383;
    localparam int FROZEN_ROWS = 149;
    localparam int FROZEN_ISSUES = 1278;
    localparam int FROZEN_PRODUCTS = 29472;
    localparam int FROZEN_COMMITS = 24;
    localparam int FROZEN_BUNDLES = 1788;
    localparam int FROZEN_SCALAR_READS = 14304;
    localparam int FROZEN_CYCLES = 20292;
    localparam int FROZEN_INTERNAL_ELEMENTS = 228;
    localparam realtime FROZEN_CLOCK_PERIOD_NS = 3.0;
    localparam string FIXTURE = "/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.memh";
    localparam string STATS = "/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920_stats.memh";

    logic clk_core = 0;
    logic rst_core;
    logic load_valid;
    logic [2:0] load_context;
    logic [23:0] load_tag;
    logic [5:0] load_group;
    logic [15:0] load_source_active, load_source_sign;
    logic load_last;
    integer tb_cycle, done_cycle, execute_start_cycle;
    integer terminal_count, reorder_count, last_response_bank;
    integer independent_stall_count;
    integer workload_slot, sample_id, layer_id, is_fc2, token_start;
    integer real_source_groups;
    integer expected_rows, expected_issues, expected_products;
    integer expected_misses, expected_hits, expected_evictions;
    integer expected_bundles;
    integer signed expected [0:BUNDLE-1][0:SLICES-1][0:LANES-1];
    logic observed [0:BUNDLE-1][0:SLICES-1];
    logic [31:0] fixture_word [0:368639];
    logic [447:0] stats_word [0:1919];
    logic measurement_window_active = 1'b0;
    realtime measurement_begin_time;

    m2149_ordinary_side_if ordinary();

    always #1.5 clk_core = ~clk_core;

    initial begin : whole_test_watchdog
        repeat (100000) @(posedge clk_core);
        $fatal(1, "M2149 whole-test watchdog expired");
    end

    always @(posedge clk_core) begin
        if (rst_core)
            tb_cycle <= 0;
        else
            tb_cycle <= tb_cycle + 1;
    end

    // Exactly one frontend is instantiated, directly, at the ordinary mode.
    m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend #(
        .SCHEDULE_MODE(0), .SOURCE_GROUPS(GROUPS)
    ) dut_ordinary (
        .clk_core(clk_core), .rst_core(rst_core),
        .load_valid(load_valid), .load_ready(ordinary.load_ready),
        .load_context(load_context), .load_tag(load_tag),
        .load_group(load_group), .load_source_active(load_source_active),
        .load_source_sign(load_source_sign), .load_last(load_last),
        .load_accept(ordinary.load_accept),
        .mem_req_valid(ordinary.mem_req_valid),
        .mem_req_ready(ordinary.mem_req_ready),
        .mem_req_epoch(ordinary.mem_req_epoch),
        .mem_req_slot(ordinary.mem_req_slot),
        .mem_req_generation(ordinary.mem_req_generation),
        .mem_req_tag(ordinary.mem_req_tag),
        .mem_req_output_block(ordinary.mem_req_output_block),
        .mem_req_slice(ordinary.mem_req_slice),
        .mem_req_source_channel(ordinary.mem_req_source_channel),
        .mem_req_accept(ordinary.mem_req_accept),
        .mem_rsp_valid(ordinary.mem_rsp_valid),
        .mem_rsp_ready(ordinary.mem_rsp_ready),
        .mem_rsp_epoch(ordinary.mem_rsp_epoch),
        .mem_rsp_slot(ordinary.mem_rsp_slot),
        .mem_rsp_generation(ordinary.mem_rsp_generation),
        .mem_rsp_tag(ordinary.mem_rsp_tag),
        .mem_rsp_weight(ordinary.mem_rsp_weight),
        .mem_rsp_accept(ordinary.mem_rsp_accept),
        .bridge_valid(ordinary.bridge_valid),
        .bridge_ready(ordinary.bridge_ready),
        .bridge_context(ordinary.bridge_context),
        .bridge_group(ordinary.bridge_group),
        .bridge_half(ordinary.bridge_half),
        .bridge_slice(ordinary.bridge_slice),
        .bridge_bank_valid(ordinary.bridge_bank_valid),
        .bridge_source_channel(ordinary.bridge_source_channel),
        .bridge_source_value(ordinary.bridge_source_value),
        .bridge_effective_weight(ordinary.bridge_effective_weight),
        .bridge_accept(ordinary.bridge_accept),
        .commit_valid(ordinary.commit_valid),
        .commit_ready(ordinary.commit_ready),
        .commit_context(ordinary.commit_context),
        .commit_tag(ordinary.commit_tag),
        .commit_slice(ordinary.commit_slice),
        .commit_accumulator(ordinary.commit_accumulator),
        .commit_terminal(ordinary.commit_terminal),
        .commit_accept(ordinary.commit_accept),
        .bundle_done_valid(ordinary.bundle_done_valid),
        .bundle_done_ready(ordinary.bundle_done_ready),
        .protocol_error(ordinary.protocol_error),
        .stale_response_seen(ordinary.stale_response_seen),
        .numeric_overflow(ordinary.numeric_overflow), .busy(ordinary.busy),
        .debug_cycle_count(ordinary.cycle_count),
        .debug_row_access_count(ordinary.row_access_count),
        .debug_cache_hit_count(ordinary.cache_hit_count),
        .debug_cache_miss_count(ordinary.cache_miss_count),
        .debug_cache_eviction_count(ordinary.cache_eviction_count),
        .debug_weight_bundle_beat_count(ordinary.weight_bundle_beat_count),
        .debug_scalar_bank_request_count(ordinary.scalar_bank_request_count),
        .debug_scalar_bank_response_count(ordinary.scalar_bank_response_count),
        .debug_issue_count(ordinary.issue_count),
        .debug_signed_product_count(ordinary.product_count),
        .debug_commit_count(ordinary.commit_count)
    );

    m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions #(
        .SOURCE_GROUPS(GROUPS)
    ) sva_ordinary (
        .clk_core(clk_core), .rst_core(rst_core),
        .load_valid(load_valid), .load_ready(ordinary.load_ready),
        .load_accept(ordinary.load_accept), .load_context(load_context),
        .mem_req_valid(ordinary.mem_req_valid),
        .mem_req_ready(ordinary.mem_req_ready),
        .mem_req_epoch(ordinary.mem_req_epoch),
        .mem_req_slot(ordinary.mem_req_slot),
        .mem_req_generation(ordinary.mem_req_generation),
        .mem_req_tag(ordinary.mem_req_tag),
        .mem_req_output_block(ordinary.mem_req_output_block),
        .mem_req_slice(ordinary.mem_req_slice),
        .mem_req_source_channel(ordinary.mem_req_source_channel),
        .mem_req_accept(ordinary.mem_req_accept),
        .mem_rsp_valid(ordinary.mem_rsp_valid),
        .mem_rsp_ready(ordinary.mem_rsp_ready),
        .mem_rsp_epoch(ordinary.mem_rsp_epoch),
        .mem_rsp_slot(ordinary.mem_rsp_slot),
        .mem_rsp_generation(ordinary.mem_rsp_generation),
        .mem_rsp_tag(ordinary.mem_rsp_tag),
        .mem_rsp_weight(ordinary.mem_rsp_weight),
        .mem_rsp_accept(ordinary.mem_rsp_accept),
        .bridge_valid(ordinary.bridge_valid),
        .bridge_ready(ordinary.bridge_ready),
        .bridge_context(ordinary.bridge_context),
        .bridge_group(ordinary.bridge_group),
        .bridge_half(ordinary.bridge_half),
        .bridge_slice(ordinary.bridge_slice),
        .bridge_bank_valid(ordinary.bridge_bank_valid),
        .bridge_source_channel(ordinary.bridge_source_channel),
        .bridge_source_value(ordinary.bridge_source_value),
        .bridge_effective_weight(ordinary.bridge_effective_weight),
        .bridge_accept(ordinary.bridge_accept),
        .commit_valid(ordinary.commit_valid),
        .commit_ready(ordinary.commit_ready),
        .commit_context(ordinary.commit_context),
        .commit_tag(ordinary.commit_tag),
        .commit_slice(ordinary.commit_slice),
        .commit_accumulator(ordinary.commit_accumulator),
        .commit_terminal(ordinary.commit_terminal),
        .commit_accept(ordinary.commit_accept),
        .protocol_error(ordinary.protocol_error),
        .stale_response_seen(ordinary.stale_response_seen),
        .numeric_overflow(ordinary.numeric_overflow),
        .debug_cache_eviction_count(ordinary.cache_eviction_count),
        .debug_weight_bundle_beat_count(ordinary.weight_bundle_beat_count)
    );

    for (genvar bank = 0; bank < 8; bank++) begin : g_memory
        m2149_directed_scalar_bank_memory #(.BANK_ID(bank)) memory (
            .clk_core(clk_core), .rst_core(rst_core),
            .req_valid(ordinary.mem_req_valid[bank]),
            .req_ready(ordinary.mem_req_ready[bank]),
            .req_epoch(ordinary.mem_req_epoch[bank]),
            .req_slot(ordinary.mem_req_slot[bank]),
            .req_generation(ordinary.mem_req_generation[bank]),
            .req_tag(ordinary.mem_req_tag[bank]),
            .req_output_block(ordinary.mem_req_output_block[bank]),
            .req_slice(ordinary.mem_req_slice[bank]),
            .req_source_channel(ordinary.mem_req_source_channel[bank]),
            .req_accept(ordinary.mem_req_accept[bank]),
            .rsp_valid(ordinary.mem_rsp_valid[bank]),
            .rsp_ready(ordinary.mem_rsp_ready[bank]),
            .rsp_epoch(ordinary.mem_rsp_epoch[bank]),
            .rsp_slot(ordinary.mem_rsp_slot[bank]),
            .rsp_generation(ordinary.mem_rsp_generation[bank]),
            .rsp_tag(ordinary.mem_rsp_tag[bank]),
            .rsp_weight(ordinary.mem_rsp_weight[bank]),
            .rsp_accept(ordinary.mem_rsp_accept[bank]),
            .request_count(ordinary.memory_request_count[bank]),
            .response_count(ordinary.memory_response_count[bank]),
            .request_stall_count(ordinary.memory_stall_count[bank])
        );
    end

    function automatic integer directed_weight(
        input integer group_index, half_index, output_slice, bank, lane);
        integer value;
        begin
            value = (group_index * 17 + half_index * 11
                     + output_slice * 7 + bank * 5 + lane * 3) % 255 - 127;
            if (group_index == 0 && half_index == 0 && output_slice == 0
                    && bank == 0 && lane == 0)
                value = -128;
            return value;
        end
    endfunction

    task automatic prepare_real_descriptor(input integer ctx, group_index);
        integer source, value, fixture_index;
        begin
            fixture_index = FROZEN_WORKLOAD_SLOT * BUNDLE * GROUPS
                + ctx * GROUPS + group_index;
            load_source_active = fixture_word[fixture_index][15:0];
            load_source_sign = fixture_word[fixture_index][31:16]
                & load_source_active;
            for (int output_slice = 0; output_slice < SLICES; output_slice++)
                for (int lane = 0; lane < LANES; lane++)
                    for (source = 0; source < 16; source++)
                        if (load_source_active[source]) begin
                            value = load_source_sign[source] ? -1 : 1;
                            expected[ctx][output_slice][lane] += value
                                * directed_weight(group_index, source / 8,
                                    output_slice, source % 8, lane);
                        end
        end
    endtask

    task automatic load_current_descriptor;
        integer wait_cycles;
        logic accepted;
        begin
            accepted = 0;
            @(negedge clk_core);
            load_valid = 1;
            for (wait_cycles = 0; wait_cycles < 10000 && !accepted;
                 wait_cycles++) begin
                @(posedge clk_core);
                if (ordinary.load_accept) begin
                    accepted = 1;
                    load_valid <= 0;
                end
            end
            if (!accepted)
                $fatal(1, "M2149 ordinary descriptor load timeout");
            @(negedge clk_core);
            load_valid = 0;
        end
    endtask

    task automatic load_frozen_workload;
        begin
            for (int ctx = 0; ctx < BUNDLE; ctx++)
                for (int group_index = 0; group_index < GROUPS;
                     group_index++) begin
                    prepare_real_descriptor(ctx, group_index);
                    load_context = ctx;
                    load_tag = 24'h340000 + ctx;
                    load_group = group_index;
                    load_last = group_index == GROUPS - 1;
                    load_current_descriptor();
                end
        end
    endtask

    task automatic check_axis_selection;
        begin
            if (!$test$plusargs("M2149_AXIS_ORDINARY"))
                $fatal(1, "M2149 requires ordinary-axis plusarg");
            if ($test$plusargs("M2142_AXIS_ORDINARY")
                    || $test$plusargs("M2125_AXIS_ORDINARY"))
                $fatal(1, "M2149 rejects predecessor plusargs");
        end
    endtask

    task automatic check_frozen_identity;
        begin
            if (workload_slot != FROZEN_WORKLOAD_SLOT
                    || sample_id != 0 || layer_id != 28 || is_fc2 != 0
                    || token_start != 0 || real_source_groups != 48
                    || expected_rows != FROZEN_ROWS
                    || expected_issues != FROZEN_ISSUES
                    || expected_products != FROZEN_PRODUCTS
                    || expected_misses != 149 || expected_hits != 0
                    || expected_evictions != 145
                    || expected_bundles != FROZEN_BUNDLES)
                $fatal(1, "M2149 frozen workload identity drift");
            if (execute_start_cycle != FROZEN_PRELOAD_CYCLES)
                $fatal(1, "M2149 preload denominator drift");
        end
    endtask

    task automatic census_internal_knownness;
        integer row_live_known, row_live_one;
        integer cache_valid_known, cache_valid_one;
        integer slot_valid_known, slot_valid_one;
        integer bridge_overflow_known, bridge_overflow_one;
        integer rsp_shape_legal_known, rsp_shape_legal_one;
        integer total_known;
        begin
            row_live_known = 0; row_live_one = 0;
            cache_valid_known = 0; cache_valid_one = 0;
            slot_valid_known = 0; slot_valid_one = 0;
            bridge_overflow_known = 0; bridge_overflow_one = 0;
            rsp_shape_legal_known = 0; rsp_shape_legal_one = 0;
            for (int ctx = 0; ctx < 4; ctx++)
                for (int group = 0; group < 48; group++)
                    if (!$isunknown(dut_ordinary.row_live_q[ctx][group])) begin
                        row_live_known++;
                        row_live_one += dut_ordinary.row_live_q[ctx][group];
                    end
            for (int entry = 0; entry < 4; entry++)
                if (!$isunknown(dut_ordinary.cache_valid_q[entry])) begin
                    cache_valid_known++;
                    cache_valid_one += dut_ordinary.cache_valid_q[entry];
                end
            for (int slot = 0; slot < 8; slot++)
                if (!$isunknown(dut_ordinary.adapter.slot_valid_q[slot])) begin
                    slot_valid_known++;
                    slot_valid_one += dut_ordinary.adapter.slot_valid_q[slot];
                end
            for (int lane = 0; lane < 16; lane++)
                if (!$isunknown(dut_ordinary.bridge_overflow[lane])) begin
                    bridge_overflow_known++;
                    bridge_overflow_one += dut_ordinary.bridge_overflow[lane];
                end
            for (int bank = 0; bank < 8; bank++)
                if (!$isunknown(dut_ordinary.adapter.rsp_shape_legal[bank])) begin
                    rsp_shape_legal_known++;
                    rsp_shape_legal_one +=
                        dut_ordinary.adapter.rsp_shape_legal[bank];
                end
            total_known = row_live_known + cache_valid_known
                + slot_valid_known + bridge_overflow_known
                + rsp_shape_legal_known;
            $display("M2149_INTERNAL_KNOWNNESS_CENSUS phase=pre_power_reset row_live=%0d/192 row_live_one=%0d cache_valid=%0d/4 cache_valid_one=%0d slot_valid=%0d/8 slot_valid_one=%0d bridge_overflow=%0d/16 bridge_overflow_one=%0d rsp_shape_legal=%0d/8 rsp_shape_legal_one=%0d total=%0d/228 observe_only=1 force=0 deposit=0 mask=0 rtl_edit=0",
                row_live_known, row_live_one,
                cache_valid_known, cache_valid_one,
                slot_valid_known, slot_valid_one,
                bridge_overflow_known, bridge_overflow_one,
                rsp_shape_legal_known, rsp_shape_legal_one, total_known);
            if (row_live_known != 192 || cache_valid_known != 4
                    || slot_valid_known != 8
                    || bridge_overflow_known != 16
                    || rsp_shape_legal_known != 8
                    || total_known != FROZEN_INTERNAL_ELEMENTS)
                $fatal(1, "M2149 first-boundary knownness census failed");
        end
    endtask

    task automatic check_public_known;
        begin
            if ($isunknown({clk_core, rst_core, load_context, load_tag,
                    load_group, load_source_active, load_source_sign,
                    load_last, load_valid, ordinary.load_ready,
                    ordinary.load_accept, ordinary.mem_req_valid,
                    ordinary.mem_req_ready, ordinary.mem_req_accept,
                    ordinary.mem_rsp_valid, ordinary.mem_rsp_ready,
                    ordinary.mem_rsp_accept, ordinary.bridge_valid,
                    ordinary.bridge_ready, ordinary.bridge_accept,
                    ordinary.commit_valid, ordinary.commit_ready,
                    ordinary.commit_accept, ordinary.bundle_done_valid,
                    ordinary.bundle_done_ready, ordinary.protocol_error,
                    ordinary.stale_response_seen, ordinary.numeric_overflow,
                    ordinary.busy, ordinary.cycle_count,
                    ordinary.row_access_count, ordinary.cache_hit_count,
                    ordinary.cache_miss_count, ordinary.cache_eviction_count,
                    ordinary.weight_bundle_beat_count,
                    ordinary.scalar_bank_request_count,
                    ordinary.scalar_bank_response_count, ordinary.issue_count,
                    ordinary.product_count, ordinary.commit_count}))
                $fatal(1, "M2149 ordinary public X/Z");
            if (ordinary.protocol_error || ordinary.stale_response_seen
                    || ordinary.numeric_overflow)
                $fatal(1, "M2149 ordinary fault");
            for (int bank = 0; bank < 8; bank++) begin
                if (ordinary.mem_req_valid[bank]
                        && $isunknown({ordinary.mem_req_epoch[bank],
                            ordinary.mem_req_slot[bank],
                            ordinary.mem_req_generation[bank],
                            ordinary.mem_req_tag[bank],
                            ordinary.mem_req_output_block[bank],
                            ordinary.mem_req_slice[bank],
                            ordinary.mem_req_source_channel[bank]}))
                    $fatal(1, "M2149 request payload X/Z");
                if (ordinary.mem_rsp_valid[bank]
                        && $isunknown({ordinary.mem_rsp_epoch[bank],
                            ordinary.mem_rsp_slot[bank],
                            ordinary.mem_rsp_generation[bank],
                            ordinary.mem_rsp_tag[bank]}))
                    $fatal(1, "M2149 response payload X/Z");
                if (ordinary.bridge_valid
                        && ordinary.bridge_bank_valid[bank]
                        && $isunknown({
                            ordinary.bridge_source_channel[bank],
                            ordinary.bridge_source_value[bank]}))
                    $fatal(1, "M2149 bridge payload X/Z");
                for (int lane = 0; lane < 16; lane++) begin
                    if (ordinary.mem_rsp_valid[bank]
                            && $isunknown(ordinary.mem_rsp_weight[bank][lane]))
                        $fatal(1, "M2149 response weight X/Z");
                    if (ordinary.bridge_valid
                            && ordinary.bridge_bank_valid[bank]
                            && $isunknown(ordinary
                                .bridge_effective_weight[bank][lane]))
                        $fatal(1, "M2149 effective weight X/Z");
                end
            end
            if (ordinary.bridge_valid
                    && $isunknown({ordinary.bridge_bank_valid,
                        ordinary.bridge_context, ordinary.bridge_group,
                        ordinary.bridge_half, ordinary.bridge_slice}))
                $fatal(1, "M2149 bridge header X/Z");
            if (ordinary.commit_valid
                    && $isunknown({ordinary.commit_context,
                        ordinary.commit_tag, ordinary.commit_slice,
                        ordinary.commit_terminal}))
                $fatal(1, "M2149 commit payload X/Z");
            for (int lane = 0; lane < 16; lane++)
                if (ordinary.commit_valid
                        && $isunknown(ordinary.commit_accumulator[lane]))
                    $fatal(1, "M2149 commit accumulator X/Z");
        end
    endtask

    task automatic check_completion;
        integer measured_cycles;
        begin
            measured_cycles = done_cycle - execute_start_cycle;
            if (measured_cycles != FROZEN_CYCLES
                    || ordinary.row_access_count != FROZEN_ROWS
                    || ordinary.issue_count != FROZEN_ISSUES
                    || ordinary.product_count != FROZEN_PRODUCTS
                    || ordinary.commit_count != FROZEN_COMMITS
                    || ordinary.cache_miss_count != 149
                    || ordinary.cache_hit_count != 0
                    || ordinary.cache_eviction_count != 145
                    || ordinary.weight_bundle_beat_count != FROZEN_BUNDLES
                    || ordinary.scalar_bank_request_count
                        != FROZEN_SCALAR_READS
                    || ordinary.scalar_bank_response_count
                        != FROZEN_SCALAR_READS
                    || terminal_count != 4
                    || reorder_count == 0 || independent_stall_count == 0)
                $fatal(1, "M2149 ordinary completion ledger drift");
            for (int ctx = 0; ctx < BUNDLE; ctx++)
                for (int output_slice = 0; output_slice < SLICES;
                     output_slice++)
                    if (!observed[ctx][output_slice])
                        $fatal(1, "M2149 missing scoreboard commit");
        end
    endtask

    always_comb begin
        ordinary.bridge_ready = (tb_cycle % 11 != 3);
        ordinary.commit_ready = (tb_cycle % 13 != 5);
        ordinary.bundle_done_ready = 1;
    end

    always @(posedge clk_core) begin
        if (!rst_core) begin
            for (int bank = 0; bank < 8; bank++) begin
                if (ordinary.mem_rsp_accept[bank]) begin
                    if (last_response_bank >= 0 && bank < last_response_bank)
                        reorder_count <= reorder_count + 1;
                    last_response_bank <= bank;
                end
                if (ordinary.mem_req_valid[bank]
                        && !ordinary.mem_req_ready[bank])
                    independent_stall_count <= independent_stall_count + 1;
            end
            if (ordinary.commit_accept) begin
                if (observed[ordinary.commit_context][ordinary.commit_slice])
                    $fatal(1, "M2149 duplicate commit");
                observed[ordinary.commit_context][ordinary.commit_slice] <= 1;
                for (int lane = 0; lane < LANES; lane++)
                    if (ordinary.commit_accumulator[lane] !==
                            expected[ordinary.commit_context]
                                    [ordinary.commit_slice][lane])
                        $fatal(1, "M2149 arithmetic scoreboard mismatch");
                if (ordinary.commit_terminal)
                    terminal_count <= terminal_count + 1;
            end
            if (ordinary.bundle_done_valid && done_cycle < 0)
                done_cycle <= tb_cycle;
        end
    end

    always @(negedge clk_core) begin : settled_window_monitor
        if (measurement_window_active) begin
            #0.01;
            check_public_known();
        end
    end

    initial begin : ordinary_single_axis_preflight
        check_axis_selection();
        tb_cycle = 0;
        done_cycle = -1;
        execute_start_cycle = -1;
        terminal_count = 0;
        reorder_count = 0;
        last_response_bank = -1;
        independent_stall_count = 0;
        workload_slot = FROZEN_WORKLOAD_SLOT;
        rst_core = 1;
        load_valid = 0;
        load_context = 0;
        load_tag = 0;
        load_group = 0;
        load_source_active = 0;
        load_source_sign = 0;
        load_last = 0;
        for (int ctx = 0; ctx < BUNDLE; ctx++)
            for (int output_slice = 0; output_slice < SLICES;
                 output_slice++) begin
                observed[ctx][output_slice] = 0;
                for (int lane = 0; lane < LANES; lane++)
                    expected[ctx][output_slice][lane] = 0;
            end
        $readmemh(FIXTURE, fixture_word);
        $readmemh(STATS, stats_word);
        sample_id = stats_word[workload_slot][31:0];
        layer_id = stats_word[workload_slot][63:32];
        is_fc2 = stats_word[workload_slot][95:64];
        token_start = stats_word[workload_slot][127:96];
        real_source_groups = stats_word[workload_slot][159:128];
        expected_rows = stats_word[workload_slot][191:160];
        expected_issues = stats_word[workload_slot][223:192];
        expected_products = stats_word[workload_slot][255:224];
        expected_misses = stats_word[workload_slot][287:256];
        expected_hits = stats_word[workload_slot][319:288];
        expected_evictions = stats_word[workload_slot][351:320];
        expected_bundles = expected_misses * 2 * SLICES;
        repeat (5) @(posedge clk_core);
        rst_core = 0;
        load_frozen_workload();
        execute_start_cycle = tb_cycle;
        @(negedge clk_core);
        #0.01;
        census_internal_knownness();
        check_frozen_identity();
        check_public_known();
        measurement_begin_time = $realtime;
        measurement_window_active = 1'b1;
        $display("M2149_RTL_SAIF_WINDOW_BEGIN sampling=settled_negedge global_slot=42 sample=0 layer=28 is_fc2=0 token_start=0 source_groups=48 preload_cycles=383 time_ns=%0.2f next_ucli_action=power_reset",
            measurement_begin_time);
        $stop;
        wait (done_cycle >= 0);
        @(negedge clk_core);
        #0.01;
        check_completion();
        check_public_known();
        measurement_window_active = 1'b0;
        if (($realtime - measurement_begin_time)
                != FROZEN_CYCLES * FROZEN_CLOCK_PERIOD_NS)
            $fatal(1, "M2149 ordinary physical window duration drift");
        $display("M2149_RTL_SAIF_WINDOW_END axis=ordinary_lru4 sampling=settled_negedge measurement_cycles=20292 rows=149 issues=1278 products=29472 commits=24 bundles=1788 scalar_weight_reads=14304 duration_ns=%0.2f",
            $realtime - measurement_begin_time);
        $display("PASS_M2149_ORDINARY_SINGLE_AXIS_NATIVE_SAIF_PREFLIGHT ledger_exact=1 arithmetic_scoreboard_exact=1 internal_census_exact=1 enable_before_reset_preload=1 power_reset_at_first_stop=1 frontends=1 schedule_mode=0 second_axis=0 initreg_diagnostic_only=1 paper_citable=0");
        $stop;
    end
endmodule

`default_nettype wire
