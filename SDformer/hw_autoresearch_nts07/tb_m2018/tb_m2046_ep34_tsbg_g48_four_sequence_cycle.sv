`timescale 1ns/1ps
`default_nettype none

module m1880_directed_scalar_bank_memory #(
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
    input  logic inject_stale,
    input  logic inject_replay,
    input  logic [EPOCH_BITS-1:0] replay_epoch,
    input  logic [2:0] replay_slot,
    input  logic [GENERATION_BITS-1:0] replay_generation,
    input  logic [TAG_BITS-1:0] replay_tag,
    input  logic signed [7:0] replay_weight [0:LANES-1],
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
    assign rsp_valid = inject_stale || inject_replay
        || (pending_q && delay_q == 0);
    assign rsp_epoch = inject_stale ? 16'hdead
        : (inject_replay ? replay_epoch : epoch_q);
    assign rsp_slot = inject_stale ? 3'd7
        : (inject_replay ? replay_slot : slot_q);
    assign rsp_generation = inject_stale ? 32'hbad0_1794
        : (inject_replay ? replay_generation : generation_q);
    assign rsp_tag = inject_stale ? 24'hbad194
        : (inject_replay ? replay_tag : tag_q);

    always_comb begin
        group_index = int'(channel_q) / 16;
        half_index = (int'(channel_q) / 8) % 2;
        for (int lane = 0; lane < LANES; lane++) begin
            raw_weight = (group_index * 17 + half_index * 11
                          + int'(slice_q) * 7 + BANK_ID * 5
                          + lane * 3) % 255 - 127;
            // A legal directed corner that proves -(-128)=+128 in nine bits.
            if (group_index == 0 && half_index == 0 && slice_q == 0
                    && BANK_ID == 0 && lane == 0)
                raw_weight = -128;
            rsp_weight[lane] = inject_stale ? 8'sh5a
                : (inject_replay ? replay_weight[lane] : raw_weight);
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
                if (pending_q) $fatal(1, "M1880 bank overwrote live request");
                if (req_source_channel[2:0] != BANK_ID[2:0])
                    $fatal(1, "M1880 source-channel bank mismatch");
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
            if (rsp_accept && !inject_stale && !inject_replay) begin
                pending_q <= 0;
                response_count <= response_count + 1'b1;
            end
        end
    end
endmodule

interface m1880_side_if #(
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
    logic [7:0] inject_stale;
    logic [7:0] inject_replay;
    logic [EPOCH_BITS-1:0] replay_epoch [0:7];
    logic [2:0] replay_slot [0:7];
    logic [GENERATION_BITS-1:0] replay_generation [0:7];
    logic [TAG_BITS-1:0] replay_tag [0:7];
    logic signed [7:0] replay_weight [0:7][0:LANES-1];
    logic [31:0] memory_request_count [0:7];
    logic [31:0] memory_response_count [0:7];
    logic [31:0] memory_stall_count [0:7];
endinterface

module tb_m2046_ep34_tsbg_g48_four_sequence_cycle;
    localparam int BUNDLE=4, GROUPS=48, SLICES=6, LANES=16;
    localparam string FIXTURE = "/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/tb_m2018/fixtures/m2046_ep34_tsbg_g48_s4.memh";
    localparam int EXPECTED_COMMITS=BUNDLE*SLICES;
    localparam int PRODUCTION_ACC24_ABS_BOUND=48*16*128;
    localparam int DIRECTED_ACC24_ABS_BOUND=GROUPS*16*128;

    logic clk_core=0, rst_core;
    logic load_valid_base, load_valid_tsbg;
    logic [2:0] load_context;
    logic [23:0] load_tag;
    logic [5:0] load_group;
    logic [15:0] load_source_active, load_source_sign;
    logic load_last;
    integer tb_cycle, base_done_cycle, tsbg_done_cycle;
    integer terminal_base, terminal_tsbg;
    integer reorder_base, reorder_tsbg;
    integer last_response_bank_base, last_response_bank_tsbg;
    integer independent_stall_base, independent_stall_tsbg;
    integer stale_attack_count, retired_identity_replay_count;
    integer replay_accept_count, reset_recovery_count;
    integer post_reset_legal_service_count;
    integer full_base_done_cycle, full_tsbg_done_cycle;
    integer full_execute_start_cycle, full_base_exec_cycles;
    integer full_tsbg_exec_cycles;
    integer sample_slot, sample_id;
    integer expected_rows, expected_issues, expected_products;
    integer expected_base_misses, expected_base_hits, expected_base_evictions;
    integer expected_tsbg_misses, expected_tsbg_hits, expected_tsbg_evictions;
    integer expected_base_bundles, expected_tsbg_bundles;
    logic [31:0] fixture_word [0:767];
    integer saw_exact_neg128;
    logic saved_rsp_valid;
    logic [15:0] saved_rsp_epoch;
    logic [2:0] saved_rsp_slot;
    logic [31:0] saved_rsp_generation;
    logic [23:0] saved_rsp_tag;
    logic signed [7:0] saved_rsp_weight [0:LANES-1];
    integer signed expected [0:BUNDLE-1][0:SLICES-1][0:LANES-1];
    logic observed_base [0:BUNDLE-1][0:SLICES-1];
    logic observed_tsbg [0:BUNDLE-1][0:SLICES-1];
    string m1970_phase;

    m1880_side_if base();
    m1880_side_if tsbg();

    initial begin : directed_parameter_proof
        if (GROUPS < 1 || GROUPS > 48
                || PRODUCTION_ACC24_ABS_BOUND != 98304
                || DIRECTED_ACC24_ABS_BOUND != 98304
                || DIRECTED_ACC24_ABS_BOUND > PRODUCTION_ACC24_ABS_BOUND
                || PRODUCTION_ACC24_ABS_BOUND >= (1 << 23))
            $fatal(1, "M1880 directed/production Acc24 bound mismatch");
    end

    always #1.5 clk_core = ~clk_core;
    // M1970: this starts before any task-level wait and therefore bounds the
    // complete test, including descriptor loading and reset recovery.
    initial begin : m1970_whole_test_watchdog
        repeat (100000) @(posedge clk_core);
        $fatal(1, "M1970 whole-test watchdog expired");
    end

    // M1942: tb_cycle is initialized by the directed initial process, so IEEE
    // 1800 forbids always_ff ownership here.  This remains a clocked TB-only
    // counter; no DUT, interface, protocol, or scoreboard semantics change.
    always @(posedge clk_core) begin
        if (rst_core) tb_cycle <= 0;
        else tb_cycle <= tb_cycle + 1;
    end

`define CONNECT_M1880(inst, side, mode, load_v) \
    m1880_c2_tsbg_b4_real_channel_signed_frontend #( \
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

    `CONNECT_M1880(dut_base, base, 0, load_valid_base);
    `CONNECT_M1880(dut_tsbg, tsbg, 1, load_valid_tsbg);
`undef CONNECT_M1880

`define CONNECT_SVA(inst, side, load_v) \
    m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions #( \
        .SOURCE_GROUPS(GROUPS)) inst ( \
        .clk_core(clk_core), .rst_core(rst_core), \
        .load_valid(load_v), .load_ready(side.load_ready), \
        .load_accept(side.load_accept), .load_context(load_context), \
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
        .protocol_error(side.protocol_error), \
        .stale_response_seen(side.stale_response_seen), \
        .numeric_overflow(side.numeric_overflow), \
        .debug_cache_eviction_count(side.cache_eviction_count), \
        .debug_weight_bundle_beat_count(side.weight_bundle_beat_count))

    `CONNECT_SVA(sva_base, base, load_valid_base);
    `CONNECT_SVA(sva_tsbg, tsbg, load_valid_tsbg);
`undef CONNECT_SVA

    for (genvar bank = 0; bank < 8; bank++) begin : g_memory
        m1880_directed_scalar_bank_memory #(.BANK_ID(bank)) mem_base (
            .clk_core(clk_core), .rst_core(rst_core),
            .req_valid(base.mem_req_valid[bank]),
            .req_ready(base.mem_req_ready[bank]),
            .req_epoch(base.mem_req_epoch[bank]),
            .req_slot(base.mem_req_slot[bank]),
            .req_generation(base.mem_req_generation[bank]),
            .req_tag(base.mem_req_tag[bank]),
            .req_output_block(base.mem_req_output_block[bank]),
            .req_slice(base.mem_req_slice[bank]),
            .req_source_channel(base.mem_req_source_channel[bank]),
            .req_accept(base.mem_req_accept[bank]),
            .rsp_valid(base.mem_rsp_valid[bank]),
            .rsp_ready(base.mem_rsp_ready[bank]),
            .rsp_epoch(base.mem_rsp_epoch[bank]),
            .rsp_slot(base.mem_rsp_slot[bank]),
            .rsp_generation(base.mem_rsp_generation[bank]),
            .rsp_tag(base.mem_rsp_tag[bank]),
            .rsp_weight(base.mem_rsp_weight[bank]),
            .rsp_accept(base.mem_rsp_accept[bank]),
            .inject_stale(base.inject_stale[bank]),
            .inject_replay(base.inject_replay[bank]),
            .replay_epoch(base.replay_epoch[bank]),
            .replay_slot(base.replay_slot[bank]),
            .replay_generation(base.replay_generation[bank]),
            .replay_tag(base.replay_tag[bank]),
            .replay_weight(base.replay_weight[bank]),
            .request_count(base.memory_request_count[bank]),
            .response_count(base.memory_response_count[bank]),
            .request_stall_count(base.memory_stall_count[bank]));
        m1880_directed_scalar_bank_memory #(.BANK_ID(bank)) mem_tsbg (
            .clk_core(clk_core), .rst_core(rst_core),
            .req_valid(tsbg.mem_req_valid[bank]),
            .req_ready(tsbg.mem_req_ready[bank]),
            .req_epoch(tsbg.mem_req_epoch[bank]),
            .req_slot(tsbg.mem_req_slot[bank]),
            .req_generation(tsbg.mem_req_generation[bank]),
            .req_tag(tsbg.mem_req_tag[bank]),
            .req_output_block(tsbg.mem_req_output_block[bank]),
            .req_slice(tsbg.mem_req_slice[bank]),
            .req_source_channel(tsbg.mem_req_source_channel[bank]),
            .req_accept(tsbg.mem_req_accept[bank]),
            .rsp_valid(tsbg.mem_rsp_valid[bank]),
            .rsp_ready(tsbg.mem_rsp_ready[bank]),
            .rsp_epoch(tsbg.mem_rsp_epoch[bank]),
            .rsp_slot(tsbg.mem_rsp_slot[bank]),
            .rsp_generation(tsbg.mem_rsp_generation[bank]),
            .rsp_tag(tsbg.mem_rsp_tag[bank]),
            .rsp_weight(tsbg.mem_rsp_weight[bank]),
            .rsp_accept(tsbg.mem_rsp_accept[bank]),
            .inject_stale(tsbg.inject_stale[bank]),
            .inject_replay(tsbg.inject_replay[bank]),
            .replay_epoch(tsbg.replay_epoch[bank]),
            .replay_slot(tsbg.replay_slot[bank]),
            .replay_generation(tsbg.replay_generation[bank]),
            .replay_tag(tsbg.replay_tag[bank]),
            .replay_weight(tsbg.replay_weight[bank]),
            .request_count(tsbg.memory_request_count[bank]),
            .response_count(tsbg.memory_response_count[bank]),
            .request_stall_count(tsbg.memory_stall_count[bank]));
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

    task automatic prepare_synthetic_descriptor(input integer ctx, group_index);
        integer source0, source1, value0, value1;
        begin
            load_source_active = 0;
            load_source_sign = 0;
            source0 = (ctx + group_index) % 8;
            source1 = 8 + ((ctx * 3 + group_index) % 8);
            value0 = ((ctx + group_index) % 2 == 0) ? -1 : 1;
            value1 = -value0;
            load_source_active[source0] = 1;
            load_source_active[source1] = 1;
            load_source_sign[source0] = value0 < 0;
            load_source_sign[source1] = value1 < 0;
            for (int output_slice = 0; output_slice < SLICES; output_slice++)
                for (int lane = 0; lane < LANES; lane++) begin
                    expected[ctx][output_slice][lane] +=
                        value0 * directed_weight(group_index, 0,
                            output_slice, source0, lane);
                    expected[ctx][output_slice][lane] +=
                        value1 * directed_weight(group_index, 1,
                            output_slice, source1 - 8, lane);
                end
        end
    endtask

    task automatic prepare_real_descriptor(input integer ctx, group_index);
        integer source, value;
        integer fixture_index;
        begin
            fixture_index = sample_slot * BUNDLE * GROUPS
                + ctx * GROUPS + group_index;
            load_source_active = fixture_word[fixture_index][15:0];
            load_source_sign = fixture_word[fixture_index][31:16]
                & load_source_active;
            for (int output_slice = 0; output_slice < SLICES; output_slice++)
                for (int lane = 0; lane < LANES; lane++)
                    for (source = 0; source < 16; source = source + 1)
                        if (load_source_active[source]) begin
                            value = load_source_sign[source] ? -1 : 1;
                            expected[ctx][output_slice][lane] += value
                                * directed_weight(group_index, source / 8,
                                    output_slice, source % 8, lane);
                        end
        end
    endtask

    task automatic clear_scoreboard_and_phase_counters;
        begin
            base_done_cycle = -1;
            tsbg_done_cycle = -1;
            terminal_base = 0;
            terminal_tsbg = 0;
            saw_exact_neg128 = 0;
            for (int ctx = 0; ctx < BUNDLE; ctx++)
                for (int output_slice = 0; output_slice < SLICES;
                     output_slice++) begin
                    observed_base[ctx][output_slice] = 0;
                    observed_tsbg[ctx][output_slice] = 0;
                    for (int lane = 0; lane < LANES; lane++)
                        expected[ctx][output_slice][lane] = 0;
                end
        end
    endtask

    // M1970: baseline and TSBG use different legal schedules, so their ready
    // pulses are not required to coincide.  Hold one immutable descriptor for
    // both sides, then retire each side's valid independently after its own
    // acceptance.  The bounded loop covers the load phase that M1956 left
    // outside its workload-completion watchdog.
    task automatic load_current_descriptor_to_both;
        integer load_wait_cycles;
        logic base_seen, tsbg_seen;
        begin
            base_seen = 0;
            tsbg_seen = 0;
            // Drive only on the inactive edge so the shared payload and each
            // valid are stable before either DUT samples the next posedge.
            @(negedge clk_core);
            load_valid_base = 1;
            load_valid_tsbg = 1;
            $display("M1970_LOAD_BEGIN context=%0d group=%0d last=%0d cycle=%0d",
                load_context, load_group, load_last, tb_cycle);
            for (load_wait_cycles = 0;
                 load_wait_cycles < 10000 && !(base_seen && tsbg_seen);
                 load_wait_cycles = load_wait_cycles + 1) begin
                @(posedge clk_core);
                if (base.load_accept) begin
                    base_seen = 1;
                    load_valid_base <= 0;
                end
                if (tsbg.load_accept) begin
                    tsbg_seen = 1;
                    load_valid_tsbg <= 0;
                end
            end
            if (!(base_seen && tsbg_seen)) begin
                $display("M1970_LOAD_TIMEOUT phase=%s context=%0d group=%0d last=%0d cycle=%0d base_valid=%0d tsbg_valid=%0d base_accept=%0d tsbg_accept=%0d base_seen=%0d tsbg_seen=%0d base_pending=%0d tsbg_pending=%0d base_ready=%0d tsbg_ready=%0d base_busy=%0d tsbg_busy=%0d base_fault=%0d tsbg_fault=%0d",
                    m1970_phase, load_context, load_group, load_last, tb_cycle,
                    load_valid_base, load_valid_tsbg,
                    base.load_accept, tsbg.load_accept,
                    base_seen, tsbg_seen, !base_seen, !tsbg_seen,
                    base.load_ready, tsbg.load_ready, base.busy, tsbg.busy,
                    base.protocol_error, tsbg.protocol_error);
                $fatal(1, "M1970 load timeout phase=%s context=%0d group=%0d",
                    m1970_phase, load_context, load_group);
            end
            @(negedge clk_core);
            load_valid_base = 0;
            load_valid_tsbg = 0;
            $display("M1970_LOAD_COMPLETE context=%0d group=%0d wait_cycles=%0d cycle=%0d",
                load_context, load_group, load_wait_cycles, tb_cycle);
        end
    endtask

    // The frontend owns a fixed B4 bundle, so its minimum complete legal
    // recovery workload is one live source group for each of the four token
    // identities.  It necessarily exercises request, reordered response,
    // typed bridge, Acc24 commit and one terminal per token.
    task automatic load_minimal_legal_workload;
        begin
            for (int ctx = 0; ctx < BUNDLE; ctx++) begin
                prepare_synthetic_descriptor(ctx, 0);
                load_context = ctx;
                load_tag = 24'h940000 + ctx;
                load_group = 0;
                load_last = 1;
                load_current_descriptor_to_both();
            end
        end
    endtask

    task automatic load_workload;
        begin
            for (int ctx = 0; ctx < BUNDLE; ctx++) begin
                for (int group_index = 0; group_index < GROUPS; group_index++) begin
                    prepare_real_descriptor(ctx, group_index);
                    load_context = ctx;
                    load_tag = 24'h340000 + ctx;
                    load_group = group_index;
                    load_last = group_index == GROUPS - 1;
                    load_current_descriptor_to_both();
                end
            end
        end
    endtask

    always_comb begin
        base.bridge_ready = (tb_cycle % 11 != 3);
        tsbg.bridge_ready = (tb_cycle % 11 != 3);
        base.commit_ready = (tb_cycle % 13 != 5);
        tsbg.commit_ready = (tb_cycle % 13 != 5);
        base.bundle_done_ready = 1;
        tsbg.bundle_done_ready = 1;
    end

    // M1924: this is a testbench statistics/scoreboard process.  It shares
    // initialization ownership with the directed initial block, so it must be
    // an ordinary clocked process rather than always_ff under IEEE 1800.
    always @(posedge clk_core) begin
        if (!rst_core) begin
            if (tsbg.mem_rsp_accept[3] && tsbg.inject_replay[3])
                replay_accept_count <= replay_accept_count + 1;
            if (tsbg.mem_rsp_accept[3] && !tsbg.inject_replay[3]
                    && !tsbg.inject_stale[3] && !saved_rsp_valid) begin
                saved_rsp_valid <= 1;
                saved_rsp_epoch <= tsbg.mem_rsp_epoch[3];
                saved_rsp_slot <= tsbg.mem_rsp_slot[3];
                saved_rsp_generation <= tsbg.mem_rsp_generation[3];
                saved_rsp_tag <= tsbg.mem_rsp_tag[3];
                for (int lane = 0; lane < LANES; lane++)
                    saved_rsp_weight[lane] <= tsbg.mem_rsp_weight[3][lane];
            end
            for (int bank = 0; bank < 8; bank++) begin
                if (base.mem_rsp_accept[bank]) begin
                    if (last_response_bank_base >= 0
                            && bank < last_response_bank_base)
                        reorder_base <= reorder_base + 1;
                    last_response_bank_base <= bank;
                end
                if (tsbg.mem_rsp_accept[bank]) begin
                    if (last_response_bank_tsbg >= 0
                            && bank < last_response_bank_tsbg)
                        reorder_tsbg <= reorder_tsbg + 1;
                    last_response_bank_tsbg <= bank;
                end
                if (base.mem_req_valid[bank] && !base.mem_req_ready[bank])
                    independent_stall_base <= independent_stall_base + 1;
                if (tsbg.mem_req_valid[bank] && !tsbg.mem_req_ready[bank])
                    independent_stall_tsbg <= independent_stall_tsbg + 1;
            end
            if (base.bridge_accept && base.bridge_source_value[0] == -2'sd1
                    && base.bridge_effective_weight[0][0] == 9'sd128)
                saw_exact_neg128 <= saw_exact_neg128 + 1;
            if (base.commit_accept) begin
                if (observed_base[base.commit_context][base.commit_slice])
                    $fatal(1, "M1880 duplicate baseline commit");
                observed_base[base.commit_context][base.commit_slice] <= 1;
                for (int lane = 0; lane < LANES; lane++)
                    if (base.commit_accumulator[lane] !==
                            expected[base.commit_context][base.commit_slice][lane])
                        $fatal(1, "M1880 baseline arithmetic mismatch");
                if (base.commit_terminal) terminal_base <= terminal_base + 1;
            end
            if (tsbg.commit_accept) begin
                if (observed_tsbg[tsbg.commit_context][tsbg.commit_slice])
                    $fatal(1, "M1880 duplicate TSBG commit");
                observed_tsbg[tsbg.commit_context][tsbg.commit_slice] <= 1;
                for (int lane = 0; lane < LANES; lane++)
                    if (tsbg.commit_accumulator[lane] !==
                            expected[tsbg.commit_context][tsbg.commit_slice][lane])
                        $fatal(1, "M1880 TSBG arithmetic mismatch");
                if (tsbg.commit_terminal) terminal_tsbg <= terminal_tsbg + 1;
            end
            if (base.bundle_done_valid && base_done_cycle < 0)
                base_done_cycle <= tb_cycle;
            if (tsbg.bundle_done_valid && tsbg_done_cycle < 0)
                tsbg_done_cycle <= tb_cycle;
        end
    end

    initial begin
        m1970_phase = "reset";
        tb_cycle = 0;
        $display("M1970_PHASE reset_begin cycle=%0d", tb_cycle);
        rst_core = 1;
        load_valid_base = 0;
        load_valid_tsbg = 0;
        load_context = 0;
        load_tag = 0;
        load_group = 0;
        load_source_active = 0;
        load_source_sign = 0;
        load_last = 0;
        base.inject_stale = 0;
        tsbg.inject_stale = 0;
        base.inject_replay = 0;
        tsbg.inject_replay = 0;
        for (int bank = 0; bank < 8; bank++) begin
            base.replay_epoch[bank] = 0;
            base.replay_slot[bank] = 0;
            base.replay_generation[bank] = 0;
            base.replay_tag[bank] = 0;
            tsbg.replay_epoch[bank] = 0;
            tsbg.replay_slot[bank] = 0;
            tsbg.replay_generation[bank] = 0;
            tsbg.replay_tag[bank] = 0;
            for (int lane = 0; lane < LANES; lane++) begin
                base.replay_weight[bank][lane] = 0;
                tsbg.replay_weight[bank][lane] = 0;
            end
        end
        reorder_base = 0;
        reorder_tsbg = 0;
        last_response_bank_base = -1;
        last_response_bank_tsbg = -1;
        independent_stall_base = 0;
        independent_stall_tsbg = 0;
        stale_attack_count = 0;
        retired_identity_replay_count = 0;
        replay_accept_count = 0;
        reset_recovery_count = 0;
        post_reset_legal_service_count = 0;
        full_base_done_cycle = -1;
        full_tsbg_done_cycle = -1;
        full_execute_start_cycle = -1;
        full_base_exec_cycles = -1;
        full_tsbg_exec_cycles = -1;
        sample_slot = 0;
        if (!$value$plusargs("SAMPLE_SLOT=%d", sample_slot))
            sample_slot = 0;
        if (sample_slot < 0 || sample_slot > 3)
            $fatal(1, "M2046 SAMPLE_SLOT outside 0..3");
        case (sample_slot)
            0: begin
                sample_id=0; expected_rows=149; expected_issues=1278;
                expected_products=29472; expected_base_misses=149;
                expected_base_hits=0; expected_base_evictions=145;
                expected_tsbg_misses=48; expected_tsbg_hits=101;
                expected_tsbg_evictions=44;
            end
            1: begin
                sample_id=10; expected_rows=159; expected_issues=1410;
                expected_products=31680; expected_base_misses=159;
                expected_base_hits=0; expected_base_evictions=155;
                expected_tsbg_misses=47; expected_tsbg_hits=112;
                expected_tsbg_evictions=43;
            end
            2: begin
                sample_id=20; expected_rows=174; expected_issues=1668;
                expected_products=42240; expected_base_misses=174;
                expected_base_hits=0; expected_base_evictions=170;
                expected_tsbg_misses=48; expected_tsbg_hits=126;
                expected_tsbg_evictions=44;
            end
            default: begin
                sample_id=30; expected_rows=153; expected_issues=1296;
                expected_products=28416; expected_base_misses=153;
                expected_base_hits=0; expected_base_evictions=149;
                expected_tsbg_misses=48; expected_tsbg_hits=105;
                expected_tsbg_evictions=44;
            end
        endcase
        expected_base_bundles = expected_base_misses * 2 * SLICES;
        expected_tsbg_bundles = expected_tsbg_misses * 2 * SLICES;
        $readmemh(FIXTURE, fixture_word);
        saved_rsp_valid = 0;
        saved_rsp_epoch = 0;
        saved_rsp_slot = 0;
        saved_rsp_generation = 0;
        saved_rsp_tag = 0;
        for (int lane = 0; lane < LANES; lane++)
            saved_rsp_weight[lane] = 0;
        clear_scoreboard_and_phase_counters();
        repeat (5) @(posedge clk_core);
        rst_core = 0;
        $display("M1970_PHASE reset_complete cycle=%0d", tb_cycle);
        m1970_phase = "full_load";
        $display("M1970_PHASE full_load_begin cycle=%0d", tb_cycle);
        load_workload();
        $display("M1970_PHASE full_load_complete cycle=%0d", tb_cycle);
        m1970_phase = "full_execute";
        $display("M1970_PHASE full_execute_begin cycle=%0d", tb_cycle);
        full_execute_start_cycle = tb_cycle;
        fork : m1970_full_completion_wait
            begin wait (base.bundle_done_valid); end
            begin wait (tsbg.bundle_done_valid); end
            begin repeat (300000) @(posedge clk_core);
                $fatal(1, "M1880 directed timeout"); end
        join_any
        disable m1970_full_completion_wait;
        wait (base_done_cycle >= 0 && tsbg_done_cycle >= 0);
        @(posedge clk_core);
        $display("M1970_PHASE full_execute_complete cycle=%0d", tb_cycle);

        if (base.protocol_error || tsbg.protocol_error
                || base.numeric_overflow || tsbg.numeric_overflow)
            $fatal(1, "M1880 clean workload faulted");
        if (base.row_access_count != expected_rows
                || tsbg.row_access_count != expected_rows)
            $fatal(1, "M1880 row ledger mismatch");
        if (base.issue_count != expected_issues
                || tsbg.issue_count != expected_issues
                || base.product_count != expected_products
                || tsbg.product_count != expected_products
                || base.commit_count != EXPECTED_COMMITS
                || tsbg.commit_count != EXPECTED_COMMITS)
            $fatal(1, "M1880 work conservation mismatch");
        if (base.weight_bundle_beat_count != expected_base_bundles
                || tsbg.weight_bundle_beat_count != expected_tsbg_bundles)
            $fatal(1, "M1880 aggregate bundle-beat expectation mismatch");
        if (base.scalar_bank_response_count != expected_base_bundles * 8
                || tsbg.scalar_bank_response_count != expected_tsbg_bundles * 8)
            $fatal(1, "M1880 scalar response ledger mismatch");
        if (base.cache_miss_count != expected_base_misses
                || tsbg.cache_miss_count != expected_tsbg_misses
                || base.cache_hit_count != expected_base_hits
                || tsbg.cache_hit_count != expected_tsbg_hits)
            $fatal(1, "M1880 LRU4 ledger mismatch");
        if (base.cache_eviction_count != expected_base_evictions
                || tsbg.cache_eviction_count != expected_tsbg_evictions)
            $fatal(1, "M1880 exact LRU4 eviction ledger mismatch");
        if (terminal_base != 4 || terminal_tsbg != 4)
            $fatal(1, "M1880 terminal count mismatch");
        if (reorder_base == 0 || reorder_tsbg == 0
                || independent_stall_base == 0 || independent_stall_tsbg == 0)
            $fatal(1, "M1880 independent-bank skew/reorder not covered");
        // The frozen ep34 FC1 fixture contains only {0,+1} sources.  Exercise
        // the signed -128 bridge corner in the independent recovery phase
        // below instead of requiring an impossible cover from this workload.
        if (!saved_rsp_valid)
            $fatal(1, "M1880 accepted response identity was not captured");
        full_base_done_cycle = base_done_cycle;
        full_tsbg_done_cycle = tsbg_done_cycle;
        full_base_exec_cycles = full_base_done_cycle - full_execute_start_cycle;
        full_tsbg_exec_cycles = full_tsbg_done_cycle - full_execute_start_cycle;

        // Replay the exact epoch/slot/generation/tag and payload accepted from
        // bank 3.  The bundle and its slot are now retired, so the same-bank
        // replay must not be accepted and must set sticky protocol/stale state.
        m1970_phase = "retired_replay";
        $display("M1970_PHASE retired_replay_begin cycle=%0d", tb_cycle);
        tsbg.replay_epoch[3] = saved_rsp_epoch;
        tsbg.replay_slot[3] = saved_rsp_slot;
        tsbg.replay_generation[3] = saved_rsp_generation;
        tsbg.replay_tag[3] = saved_rsp_tag;
        for (int lane = 0; lane < LANES; lane++)
            tsbg.replay_weight[3][lane] = saved_rsp_weight[lane];
        tsbg.inject_replay[3] = 1;
        retired_identity_replay_count = retired_identity_replay_count + 1;
        @(posedge clk_core);
        if (tsbg.mem_rsp_accept[3])
            $fatal(1, "M1880 retired legal identity replay was accepted");
        tsbg.inject_replay[3] = 0;
        repeat (2) @(posedge clk_core);
        if (!tsbg.protocol_error || !tsbg.stale_response_seen
                || replay_accept_count != 0)
            $fatal(1, "M1880 retired legal identity replay did not fail closed");
        $display("M1970_PHASE retired_replay_complete cycle=%0d", tb_cycle);

        m1970_phase = "replay_reset_recovery";
        $display("M1970_PHASE replay_reset_recovery_begin cycle=%0d", tb_cycle);
        rst_core = 1;
        repeat (3) @(posedge clk_core);
        rst_core = 0;
        repeat (2) @(posedge clk_core);
        if (tsbg.protocol_error || tsbg.stale_response_seen)
            $fatal(1, "M1880 first reset did not clear replay fault");
        reset_recovery_count = reset_recovery_count + 1;
        $display("M1970_PHASE replay_reset_recovery_complete cycle=%0d", tb_cycle);

        // Preserve the original bogus mismatched/stale attack as a separate
        // attack class; it is not counted as the retired-identity replay.
        m1970_phase = "stale_attack";
        $display("M1970_PHASE stale_attack_begin cycle=%0d", tb_cycle);
        tsbg.inject_stale[3] = 1;
        stale_attack_count = stale_attack_count + 1;
        @(posedge clk_core);
        if (tsbg.mem_rsp_accept[3])
            $fatal(1, "M1880 bogus stale response was accepted");
        tsbg.inject_stale[3] = 0;
        repeat (2) @(posedge clk_core);
        if (!tsbg.protocol_error || !tsbg.stale_response_seen)
            $fatal(1, "M1880 bogus stale response did not fail closed");
        $display("M1970_PHASE stale_attack_complete cycle=%0d", tb_cycle);

        // Three reset clocks satisfy the >=1-cycle recovery cover.  Do not
        // stop at flag clearing: run the minimum complete legal B4 service.
        m1970_phase = "stale_reset_recovery";
        $display("M1970_PHASE stale_reset_recovery_begin cycle=%0d", tb_cycle);
        rst_core = 1;
        repeat (3) @(posedge clk_core);
        rst_core = 0;
        repeat (2) @(posedge clk_core);
        if (base.protocol_error || tsbg.protocol_error
                || base.stale_response_seen || tsbg.stale_response_seen)
            $fatal(1, "M1880 second reset did not clear protocol state");
        reset_recovery_count = reset_recovery_count + 1;
        $display("M1970_PHASE stale_reset_recovery_complete cycle=%0d", tb_cycle);
        clear_scoreboard_and_phase_counters();
        m1970_phase = "recovery_load";
        $display("M1970_PHASE recovery_load_begin cycle=%0d", tb_cycle);
        load_minimal_legal_workload();
        $display("M1970_PHASE recovery_load_complete cycle=%0d", tb_cycle);
        m1970_phase = "recovery_execute";
        $display("M1970_PHASE recovery_execute_begin cycle=%0d", tb_cycle);
        fork : m1970_recovery_completion_wait
            begin wait (base.bundle_done_valid); end
            begin wait (tsbg.bundle_done_valid); end
            begin repeat (300000) @(posedge clk_core);
                $fatal(1, "M1880 post-reset legal-service timeout"); end
        join_any
        disable m1970_recovery_completion_wait;
        wait (base_done_cycle >= 0 && tsbg_done_cycle >= 0);
        @(posedge clk_core);
        $display("M1970_PHASE recovery_execute_complete cycle=%0d", tb_cycle);
        if (base.protocol_error || tsbg.protocol_error
                || base.numeric_overflow || tsbg.numeric_overflow)
            $fatal(1, "M1880 post-reset legal service faulted");
        if (base.row_access_count != 4 || tsbg.row_access_count != 4
                || base.issue_count != 48 || tsbg.issue_count != 48
                || base.product_count != 768 || tsbg.product_count != 768
                || base.commit_count != 24 || tsbg.commit_count != 24
                || base.weight_bundle_beat_count != 12
                || tsbg.weight_bundle_beat_count != 12
                || base.scalar_bank_response_count != 96
                || tsbg.scalar_bank_response_count != 96
                || terminal_base != 4 || terminal_tsbg != 4)
            $fatal(1, "M1880 post-reset legal-service ledger mismatch");
        if (base.cache_miss_count != 1 || tsbg.cache_miss_count != 1
                || base.cache_hit_count != 3 || tsbg.cache_hit_count != 3
                || saw_exact_neg128 == 0)
            $fatal(1, "M1880 post-reset bridge/cache coverage mismatch");
        post_reset_legal_service_count = post_reset_legal_service_count + 1;

        m1970_phase = "final_checks";
        $display("M1970_PHASE final_checks_begin cycle=%0d", tb_cycle);
        if (full_tsbg_exec_cycles <= 0 || full_base_exec_cycles <= 0
                || full_base_exec_cycles * 1.0 / full_tsbg_exec_cycles < 1.15)
            $fatal(1, "M1880 directed local cycle gate below 1.15x");
        if (stale_attack_count != 1 || retired_identity_replay_count != 1
                || replay_accept_count != 0 || reset_recovery_count != 2
                || post_reset_legal_service_count != 1)
            $fatal(1, "M1880 protocol attack ledger mismatch");
        $display("M1970_PHASE final_checks_complete cycle=%0d", tb_cycle);

        // M1984: one format string is mandatory.  Multiple comma-separated
        // string arguments are data operands to $display and corrupt the
        // machine receipt even when all preceding functional checks pass.
        $display("PASS_M2046_EP34_TSBG_G48_CYCLE sample_slot=%0d sample_id=%0d layer=28 rows=%0d issues=%0d products=%0d commits=%0d base_cycles=%0d tsbg_cycles=%0d bundles_base=%0d bundles_tsbg=%0d scalar_base=%0d scalar_tsbg=%0d stale=%0d retired_replay=%0d replay_accept=%0d reset=%0d recovery=%0d system_speedup=false",
            sample_slot, sample_id, expected_rows, expected_issues,
            expected_products, EXPECTED_COMMITS,
            full_base_exec_cycles, full_tsbg_exec_cycles,
            expected_base_bundles, expected_tsbg_bundles,
            expected_base_bundles*8, expected_tsbg_bundles*8,
            stale_attack_count, retired_identity_replay_count,
            replay_accept_count, reset_recovery_count,
            post_reset_legal_service_count);
        $finish;
    end
endmodule

`default_nettype wire
