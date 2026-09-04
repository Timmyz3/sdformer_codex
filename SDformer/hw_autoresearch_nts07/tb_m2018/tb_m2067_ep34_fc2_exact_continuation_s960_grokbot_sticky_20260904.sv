// =============================================================================
// GROKBOT NEW FILE -- do NOT confuse with the original TB.
// Written by: Grok Bot (agent iscas_ssh) for Timothee Z / Timmyz3
// Date: 2026-09-04 (Asia/Shanghai)
// Purpose: M2067 FC2 exact-continuation TB handshake repair (sticky accept
//          catchers + settle-before-valid). Wrapper/RTL unchanged.
// Original (untouched): tb_m2067_ep34_fc2_exact_continuation_s960.sv
//   server sha256: df9022bad55d9c46a95f48a88ba79987aad60e397d821e6ac2afb01a9ef1b3d0
// Identity plan: intended for M2073 r4 source / M2074 hammer (new paths only).
// Quarantine DO NOT RETRY: ..._vcs_r1_20260903
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

// The directed scalar-bank memory is byte-frozen in the M2051 source and is
// compiled alongside this top.  M2067 changes only the request channel from a
// local G48 channel to the wrapper's explicitly translated logical channel.

// Preserve the frozen M1880 timing/backpressure model but make the returned
// directed arithmetic depend on the wrapper's translated global weight-row
// address.  This makes an output-tile row alias independently visible in the
// integer oracle instead of relying only on a sideband address assertion.
module m2067_row_aware_directed_scalar_bank_memory #(
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
    input  logic [11:0] req_weight_row_index,
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
    logic [11:0] row_q;
    logic [CHANNEL_BITS-1:0] channel_q;
    logic signed [7:0] legacy_weight [0:LANES-1];

    m1880_directed_scalar_bank_memory #(
        .BANK_ID(BANK_ID), .TAG_BITS(TAG_BITS),
        .CHANNEL_BITS(CHANNEL_BITS), .EPOCH_BITS(EPOCH_BITS),
        .GENERATION_BITS(GENERATION_BITS), .LANES(LANES)
    ) timing_model (
        .clk_core(clk_core), .rst_core(rst_core),
        .req_valid(req_valid), .req_ready(req_ready),
        .req_epoch(req_epoch), .req_slot(req_slot),
        .req_generation(req_generation), .req_tag(req_tag),
        .req_output_block(req_output_block), .req_slice(req_slice),
        .req_source_channel(req_source_channel), .req_accept(req_accept),
        .rsp_valid(rsp_valid), .rsp_ready(rsp_ready),
        .rsp_epoch(rsp_epoch), .rsp_slot(rsp_slot),
        .rsp_generation(rsp_generation), .rsp_tag(rsp_tag),
        .rsp_weight(legacy_weight), .rsp_accept(rsp_accept),
        .inject_stale(inject_stale), .inject_replay(inject_replay),
        .replay_epoch(replay_epoch), .replay_slot(replay_slot),
        .replay_generation(replay_generation), .replay_tag(replay_tag),
        .replay_weight(replay_weight), .request_count(request_count),
        .response_count(response_count),
        .request_stall_count(request_stall_count)
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            row_q <= 0;
            channel_q <= 0;
        end else if (req_accept) begin
            row_q <= req_weight_row_index;
            channel_q <= req_source_channel;
        end
    end

    always_comb begin : row_aware_weight
        integer natural_weight, row_delta, adjusted_weight;
        row_delta = int'(row_q) - int'(channel_q >> 4);
        for (int lane = 0; lane < LANES; lane++) begin
            natural_weight = int'(legacy_weight[lane]);
            // The legacy model reserves -128 as one explicit signed corner;
            // its underlying modulo value is -127.  Retain the corner only
            // for global row zero, then translate every other row exactly.
            if (natural_weight == -128 && row_q == 0)
                adjusted_weight = -128;
            else begin
                if (natural_weight == -128) natural_weight = -127;
                adjusted_weight = ((natural_weight + 127
                                    + row_delta * 17) % 255) - 127;
            end
            rsp_weight[lane] = (inject_stale || inject_replay)
                ? legacy_weight[lane] : adjusted_weight;
        end
    end
endmodule

interface m2067_side_if #(
    parameter int TAG_BITS=24, CHANNEL_BITS=12, EPOCH_BITS=16,
    parameter int GENERATION_BITS=32, LANES=16
);
    logic chunk_ready, chunk_accept;
    logic load_ready, load_accept;
    logic [7:0] mem_req_valid, mem_req_ready, mem_req_accept;
    logic [EPOCH_BITS-1:0] mem_req_epoch [0:7];
    logic [2:0] mem_req_slot [0:7];
    logic [GENERATION_BITS-1:0] mem_req_generation [0:7];
    logic [TAG_BITS-1:0] mem_req_tag [0:7];
    logic [2:0] mem_req_output_block [0:7], mem_req_slice [0:7];
    logic [CHANNEL_BITS-1:0] mem_req_source_channel [0:7];
    logic [7:0] mem_req_global_group [0:7];
    logic [11:0] mem_req_weight_row_index [0:7];
    logic [7:0] mem_rsp_valid, mem_rsp_ready, mem_rsp_accept;
    logic [EPOCH_BITS-1:0] mem_rsp_epoch [0:7];
    logic [2:0] mem_rsp_slot [0:7];
    logic [GENERATION_BITS-1:0] mem_rsp_generation [0:7];
    logic [TAG_BITS-1:0] mem_rsp_tag [0:7];
    logic signed [7:0] mem_rsp_weight [0:7][0:LANES-1];
    logic bridge_valid, bridge_ready, bridge_accept;
    logic [2:0] bridge_context, bridge_slice;
    logic [7:0] bridge_global_group, bridge_bank_valid;
    logic bridge_half;
    logic [CHANNEL_BITS-1:0] bridge_source_channel [0:7];
    logic signed [1:0] bridge_source_value [0:7];
    logic signed [8:0] bridge_effective_weight [0:7][0:LANES-1];
    logic commit_valid, commit_ready, commit_terminal, commit_accept;
    logic [2:0] commit_context, commit_slice;
    logic [TAG_BITS-1:0] commit_tag;
    logic signed [23:0] commit_accumulator [0:LANES-1];
    logic bundle_done_valid, bundle_done_ready;
    logic protocol_error, stale_response_seen, numeric_overflow, busy;
    logic [31:0] logical_cycle_count, descriptor_preload_cycles;
    logic [31:0] continuation_cycles, chunk_count;
    logic [31:0] intermediate_chunk_count, final_chunk_count;
    logic [31:0] final_commit_count, row_access_count;
    logic [31:0] cache_hit_count, cache_miss_count, cache_eviction_count;
    logic [31:0] weight_bundle_beat_count, scalar_bank_request_count;
    logic [31:0] scalar_bank_response_count, issue_count, product_count;
    logic [31:0] alias_reject_count;
    logic [7:0] debug_global_group_base;
    logic [1:0] debug_chunk_index;
    logic debug_chunk_first, debug_chunk_intermediate, debug_chunk_final;
    logic debug_retained_acc_valid;
    logic [31:0] memory_request_count [0:7];
    logic [31:0] memory_response_count [0:7];
    logic [31:0] memory_stall_count [0:7];
endinterface

module tb_m2067_ep34_fc2_exact_continuation_s960;
    localparam int BUNDLE=4, PHYSICAL_GROUPS=48, MAX_GROUPS=192;
    localparam int SOURCES=16, SLICES=6, LANES=16, WORKLOADS=960;
    localparam int FIXTURE_WORDS=WORKLOADS*BUNDLE*MAX_GROUPS;
    localparam int EXPECTED_COMMITS_PER_TILE=BUNDLE*SLICES;
    localparam int ACC24_ABS_BOUND=MAX_GROUPS*SOURCES*128;
    localparam string FIXTURE = "/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/tb_m2018/fixtures/m2067_ep34_fc2_exact_continuation_s960.memh";
    localparam string STATS = "/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/tb_m2018/fixtures/m2067_ep34_fc2_exact_continuation_s960_stats.memh";

    logic clk_core=0, rst_core;
    logic chunk_valid_base, chunk_valid_tsbg;
    logic [23:0] logical_tag;
    logic [7:0] source_group_count, global_group_base;
    logic [2:0] output_tile_id, chunk_count;
    logic [1:0] chunk_index;
    logic chunk_first, chunk_intermediate, chunk_final;
    logic load_valid_base, load_valid_tsbg;
    logic [2:0] load_context;
    logic [5:0] load_group;
    logic [15:0] load_source_active, load_source_sign;
    logic load_last;
    logic [31:0] fixture_word [0:FIXTURE_WORDS-1];
    logic [383:0] stats_word [0:WORKLOADS-1];
    integer workload_slot, sample_id, layer_id, token_start;
    integer real_source_groups, output_tiles, chunks;
    integer token_role_id, sequence_id, expected_commits;
    integer expected_integer_checks, expected_nonzero_codes;
    integer expected_negative_codes;
    integer tb_cycle, current_output_tile;
    integer signed expected [0:BUNDLE-1][0:SLICES-1][0:LANES-1];
    logic observed_base [0:BUNDLE-1][0:SLICES-1];
    logic observed_tsbg [0:BUNDLE-1][0:SLICES-1];
    integer tile_commits_base, tile_commits_tsbg;
    integer tile_terminals_base, tile_terminals_tsbg;
    integer workload_commits_base, workload_commits_tsbg;
    integer workload_checks_base, workload_checks_tsbg;
    integer total_base_cycles, total_tsbg_cycles;
    integer row_chunk_records, address_checks_base, address_checks_tsbg;
    integer alias_attacks, alias_rejects_base, alias_rejects_tsbg;
    integer alias_attacks_g96, alias_attacks_g192;
    string global_group_bases_text;
    logic base_done_seen, tsbg_done_seen;
    logic signed [7:0] zero_replay_weight [0:LANES-1];

    m2067_side_if base();
    m2067_side_if tsbg();

    // Sticky posedge catchers: chunk_accept/load_accept are one-cycle combo
    // pulses cleared when the wrapper leaves W_HEADER/W_LOAD.  Sampling them
    // from a task after the DUT NBA region can miss the pulse (VCS observed
    // M2067 legal header timeout at 3016500 ps with ready already low).
    logic header_accept_base_sticky, header_accept_tsbg_sticky;
    logic load_accept_base_sticky, load_accept_tsbg_sticky;
    logic header_accept_clear=0, load_accept_clear=0;
    always_ff @(posedge clk_core) begin
        if (rst_core || header_accept_clear) begin
            header_accept_base_sticky <= 0;
            header_accept_tsbg_sticky <= 0;
        end else begin
            if (chunk_valid_base && base.chunk_accept)
                header_accept_base_sticky <= 1;
            if (chunk_valid_tsbg && tsbg.chunk_accept)
                header_accept_tsbg_sticky <= 1;
        end
        if (rst_core || load_accept_clear) begin
            load_accept_base_sticky <= 0;
            load_accept_tsbg_sticky <= 0;
        end else begin
            if (load_valid_base && base.load_accept)
                load_accept_base_sticky <= 1;
            if (load_valid_tsbg && tsbg.load_accept)
                load_accept_tsbg_sticky <= 1;
        end
    end

    initial begin : parameter_proof
        if (ACC24_ABS_BOUND != 393216 || ACC24_ABS_BOUND >= (1 << 23))
            $fatal(1, "M2067 Acc24 analytical bound drift");
    end
    always #1.5 clk_core = ~clk_core;
    always @(posedge clk_core) begin
        if (rst_core) tb_cycle <= 0;
        else tb_cycle <= tb_cycle + 1;
    end
    initial begin : whole_test_watchdog
        repeat (5000000) @(posedge clk_core);
        $fatal(1, "M2067 whole-test watchdog expired");
    end

`define CONNECT_M2067(inst, side, mode, chunk_v, load_v) \
    m2067_fc2_exact_continuation_wrapper #(.SCHEDULE_MODE(mode)) inst ( \
        .clk_core(clk_core), .rst_core(rst_core), \
        .chunk_valid(chunk_v), .chunk_ready(side.chunk_ready), \
        .logical_tag(logical_tag), .source_group_count(source_group_count), \
        .output_tile_id(output_tile_id), .chunk_index(chunk_index), \
        .chunk_count(chunk_count), .global_group_base(global_group_base), \
        .chunk_first(chunk_first), .chunk_intermediate(chunk_intermediate), \
        .chunk_final(chunk_final), .chunk_accept(side.chunk_accept), \
        .load_valid(load_v), .load_ready(side.load_ready), \
        .load_context(load_context), .load_group(load_group), \
        .load_source_active(load_source_active), \
        .load_source_sign(load_source_sign), .load_last(load_last), \
        .load_accept(side.load_accept), \
        .mem_req_valid(side.mem_req_valid), .mem_req_ready(side.mem_req_ready), \
        .mem_req_epoch(side.mem_req_epoch), .mem_req_slot(side.mem_req_slot), \
        .mem_req_generation(side.mem_req_generation), \
        .mem_req_tag(side.mem_req_tag), \
        .mem_req_output_block(side.mem_req_output_block), \
        .mem_req_slice(side.mem_req_slice), \
        .mem_req_source_channel(side.mem_req_source_channel), \
        .mem_req_global_group(side.mem_req_global_group), \
        .mem_req_weight_row_index(side.mem_req_weight_row_index), \
        .mem_req_accept(side.mem_req_accept), \
        .mem_rsp_valid(side.mem_rsp_valid), .mem_rsp_ready(side.mem_rsp_ready), \
        .mem_rsp_epoch(side.mem_rsp_epoch), .mem_rsp_slot(side.mem_rsp_slot), \
        .mem_rsp_generation(side.mem_rsp_generation), \
        .mem_rsp_tag(side.mem_rsp_tag), .mem_rsp_weight(side.mem_rsp_weight), \
        .mem_rsp_accept(side.mem_rsp_accept), \
        .bridge_valid(side.bridge_valid), .bridge_ready(side.bridge_ready), \
        .bridge_context(side.bridge_context), \
        .bridge_global_group(side.bridge_global_group), \
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
        .debug_logical_cycle_count(side.logical_cycle_count), \
        .debug_descriptor_preload_cycles(side.descriptor_preload_cycles), \
        .debug_continuation_cycles(side.continuation_cycles), \
        .debug_chunk_count(side.chunk_count), \
        .debug_intermediate_chunk_count(side.intermediate_chunk_count), \
        .debug_final_chunk_count(side.final_chunk_count), \
        .debug_final_commit_count(side.final_commit_count), \
        .debug_row_access_count(side.row_access_count), \
        .debug_cache_hit_count(side.cache_hit_count), \
        .debug_cache_miss_count(side.cache_miss_count), \
        .debug_cache_eviction_count(side.cache_eviction_count), \
        .debug_weight_bundle_beat_count(side.weight_bundle_beat_count), \
        .debug_scalar_bank_request_count(side.scalar_bank_request_count), \
        .debug_scalar_bank_response_count(side.scalar_bank_response_count), \
        .debug_issue_count(side.issue_count), \
        .debug_signed_product_count(side.product_count), \
        .debug_alias_reject_count(side.alias_reject_count), \
        .debug_global_group_base(side.debug_global_group_base), \
        .debug_chunk_index(side.debug_chunk_index), \
        .debug_chunk_first(side.debug_chunk_first), \
        .debug_chunk_intermediate(side.debug_chunk_intermediate), \
        .debug_chunk_final(side.debug_chunk_final), \
        .debug_retained_acc_valid(side.debug_retained_acc_valid))

    `CONNECT_M2067(dut_base, base, 0, chunk_valid_base, load_valid_base);
    `CONNECT_M2067(dut_tsbg, tsbg, 1, chunk_valid_tsbg, load_valid_tsbg);

    for (genvar bank=0; bank<8; bank++) begin : memories
        m2067_row_aware_directed_scalar_bank_memory #(.BANK_ID(bank)) mem_base (
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
            .req_weight_row_index(base.mem_req_weight_row_index[bank]),
            .req_accept(base.mem_req_accept[bank]),
            .rsp_valid(base.mem_rsp_valid[bank]),
            .rsp_ready(base.mem_rsp_ready[bank]),
            .rsp_epoch(base.mem_rsp_epoch[bank]),
            .rsp_slot(base.mem_rsp_slot[bank]),
            .rsp_generation(base.mem_rsp_generation[bank]),
            .rsp_tag(base.mem_rsp_tag[bank]),
            .rsp_weight(base.mem_rsp_weight[bank]),
            .rsp_accept(base.mem_rsp_accept[bank]),
            .inject_stale(1'b0), .inject_replay(1'b0),
            .replay_epoch('0), .replay_slot('0), .replay_generation('0),
            .replay_tag('0), .replay_weight(zero_replay_weight),
            .request_count(base.memory_request_count[bank]),
            .response_count(base.memory_response_count[bank]),
            .request_stall_count(base.memory_stall_count[bank]));
        m2067_row_aware_directed_scalar_bank_memory #(.BANK_ID(bank)) mem_tsbg (
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
            .req_weight_row_index(tsbg.mem_req_weight_row_index[bank]),
            .req_accept(tsbg.mem_req_accept[bank]),
            .rsp_valid(tsbg.mem_rsp_valid[bank]),
            .rsp_ready(tsbg.mem_rsp_ready[bank]),
            .rsp_epoch(tsbg.mem_rsp_epoch[bank]),
            .rsp_slot(tsbg.mem_rsp_slot[bank]),
            .rsp_generation(tsbg.mem_rsp_generation[bank]),
            .rsp_tag(tsbg.mem_rsp_tag[bank]),
            .rsp_weight(tsbg.mem_rsp_weight[bank]),
            .rsp_accept(tsbg.mem_rsp_accept[bank]),
            .inject_stale(1'b0), .inject_replay(1'b0),
            .replay_epoch('0), .replay_slot('0), .replay_generation('0),
            .replay_tag('0), .replay_weight(zero_replay_weight),
            .request_count(tsbg.memory_request_count[bank]),
            .response_count(tsbg.memory_response_count[bank]),
            .request_stall_count(tsbg.memory_stall_count[bank]));
    end

    function automatic integer directed_weight(
        input integer group_index, half_index, output_tile, logical_groups,
        output_slice, bank, lane);
        integer value;
        begin
            value = ((output_tile * logical_groups + group_index) * 17
                     + half_index * 11
                     + output_slice * 7 + bank * 5 + lane * 3) % 255 - 127;
            if (output_tile == 0 && group_index == 0 && half_index == 0
                    && output_slice == 0 && bank == 0 && lane == 0)
                value = -128;
            return value;
        end
    endfunction

    task automatic initialize_drives;
        begin
            chunk_valid_base=0; chunk_valid_tsbg=0; logical_tag=0;
            source_group_count=0; output_tile_id=0; chunk_index=0;
            chunk_count=0; global_group_base=0; chunk_first=0;
            chunk_intermediate=0; chunk_final=0;
            load_valid_base=0; load_valid_tsbg=0; load_context=0;
            load_group=0; load_source_active=0; load_source_sign=0;
            load_last=0;
            header_accept_clear=0; load_accept_clear=0;
        end
    endtask

    task automatic reset_both;
        begin
            @(negedge clk_core); rst_core=1;
            repeat (4) @(posedge clk_core);
            @(negedge clk_core); rst_core=0;
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic send_header_both(
        input integer groups_i, tile_i, index_i, count_i, base_i,
        input logic first_i, intermediate_i, final_i);
        integer wait_cycles;
        begin
            // Clear sticky catchers, wait for ready, then settle header fields
            // with valid low for one full negedge→negedge before asserting valid.
            // Accept is observed via always_ff sticky bits, not a post-NBA
            // combinatorial sample of the one-cycle chunk_accept pulse.
            @(negedge clk_core);
            header_accept_clear=1;
            chunk_valid_base=0; chunk_valid_tsbg=0;
            @(posedge clk_core);
            @(negedge clk_core); header_accept_clear=0;
            while (!(base.chunk_ready && tsbg.chunk_ready)) @(negedge clk_core);
            logical_tag=24'h670000 + workload_slot*8 + tile_i;
            source_group_count=groups_i; output_tile_id=tile_i;
            chunk_index=index_i; chunk_count=count_i;
            global_group_base=base_i; chunk_first=first_i;
            chunk_intermediate=intermediate_i; chunk_final=final_i;
            @(negedge clk_core);
            if (!(base.chunk_ready && tsbg.chunk_ready))
                $fatal(1, "M2067 header ready lost while fields settled");
            chunk_valid_base=1; chunk_valid_tsbg=1;
            for (wait_cycles=0; wait_cycles<1000
                    && !(header_accept_base_sticky && header_accept_tsbg_sticky);
                 wait_cycles=wait_cycles+1) begin
                @(posedge clk_core);
                if (header_accept_base_sticky) chunk_valid_base<=0;
                if (header_accept_tsbg_sticky) chunk_valid_tsbg<=0;
                if (base.protocol_error || tsbg.protocol_error) begin
                    $display("M2067_HEADER_FAULT groups=%0d tile=%0d idx=%0d count=%0d base=%0d first=%0d inter=%0d final=%0d ready_b=%0b ready_t=%0b accept_b=%0b accept_t=%0b sticky_b=%0b sticky_t=%0b proto_b=%0b proto_t=%0b alias_b=%0d alias_t=%0d",
                        groups_i, tile_i, index_i, count_i, base_i,
                        first_i, intermediate_i, final_i,
                        base.chunk_ready, tsbg.chunk_ready,
                        base.chunk_accept, tsbg.chunk_accept,
                        header_accept_base_sticky, header_accept_tsbg_sticky,
                        base.protocol_error, tsbg.protocol_error,
                        base.alias_reject_count, tsbg.alias_reject_count);
                    $fatal(1, "M2067 legal header rejected into FAULT");
                end
            end
            if (!(header_accept_base_sticky && header_accept_tsbg_sticky)) begin
                $display("M2067_HEADER_TIMEOUT groups=%0d tile=%0d idx=%0d count=%0d base=%0d first=%0d inter=%0d final=%0d ready_b=%0b ready_t=%0b accept_b=%0b accept_t=%0b sticky_b=%0b sticky_t=%0b proto_b=%0b proto_t=%0b dbg_base_b=%0d dbg_idx_b=%0d dbg_first_b=%0b",
                    groups_i, tile_i, index_i, count_i, base_i,
                    first_i, intermediate_i, final_i,
                    base.chunk_ready, tsbg.chunk_ready,
                    base.chunk_accept, tsbg.chunk_accept,
                    header_accept_base_sticky, header_accept_tsbg_sticky,
                    base.protocol_error, tsbg.protocol_error,
                    base.debug_global_group_base, base.debug_chunk_index,
                    base.debug_chunk_first);
                $fatal(1, "M2067 legal header timeout/reject");
            end
            @(negedge clk_core); chunk_valid_base=0; chunk_valid_tsbg=0;
        end
    endtask

    task automatic send_invalid_alias_header_both(
        input integer groups_i, index_i, count_i, bad_base_i,
        input logic first_i, intermediate_i, final_i);
        integer before_base, before_tsbg;
        begin
            while (!(base.chunk_ready && tsbg.chunk_ready)) @(negedge clk_core);
            before_base=base.alias_reject_count;
            before_tsbg=tsbg.alias_reject_count;
            logical_tag=24'h66aa55; source_group_count=groups_i;
            output_tile_id=0; chunk_index=index_i; chunk_count=count_i;
            global_group_base=bad_base_i; chunk_first=first_i;
            chunk_intermediate=intermediate_i; chunk_final=final_i;
            // Settle illegal fields with valid low so FAULT sees stable legality.
            @(negedge clk_core);
            chunk_valid_base=1; chunk_valid_tsbg=1;
            @(posedge clk_core);
            if (base.chunk_accept || tsbg.chunk_accept)
                $fatal(1, "M2067 aliased global_group_base accepted");
            @(negedge clk_core); chunk_valid_base=0; chunk_valid_tsbg=0;
            repeat (2) @(posedge clk_core);
            if (!base.protocol_error || !tsbg.protocol_error
                    || base.alias_reject_count != before_base+1
                    || tsbg.alias_reject_count != before_tsbg+1)
                $fatal(1, "M2067 aliased global_group_base did not fail closed");
            alias_attacks=alias_attacks+1;
            if (groups_i==96) alias_attacks_g96=alias_attacks_g96+1;
            if (groups_i==192) alias_attacks_g192=alias_attacks_g192+1;
            alias_rejects_base=alias_rejects_base+1;
            alias_rejects_tsbg=alias_rejects_tsbg+1;
        end
    endtask

    task automatic load_descriptor_both;
        integer wait_cycles;
        begin
            @(negedge clk_core);
            load_accept_clear=1;
            load_valid_base=0; load_valid_tsbg=0;
            @(posedge clk_core);
            @(negedge clk_core); load_accept_clear=0;
            while (!(base.load_ready && tsbg.load_ready)) @(negedge clk_core);
            load_valid_base=1; load_valid_tsbg=1;
            for (wait_cycles=0; wait_cycles<10000
                    && !(load_accept_base_sticky && load_accept_tsbg_sticky);
                 wait_cycles=wait_cycles+1) begin
                @(posedge clk_core);
                if (load_accept_base_sticky) load_valid_base<=0;
                if (load_accept_tsbg_sticky) load_valid_tsbg<=0;
            end
            if (!(load_accept_base_sticky && load_accept_tsbg_sticky))
                $fatal(1, "M2067 load timeout");
            @(negedge clk_core); load_valid_base=0; load_valid_tsbg=0;
        end
    endtask

    task automatic load_chunk(input integer group_base_i, input logic zero_fill);
        integer fixture_index;
        begin
            for (int context=0; context<BUNDLE; context++)
                for (int local_group=0; local_group<PHYSICAL_GROUPS;
                     local_group++) begin
                    load_context=context; load_group=local_group;
                    load_last=(context==BUNDLE-1
                               && local_group==PHYSICAL_GROUPS-1);
                    if (zero_fill) begin
                        load_source_active=0; load_source_sign=0;
                    end else begin
                        fixture_index=workload_slot*BUNDLE*MAX_GROUPS
                            + context*MAX_GROUPS+group_base_i+local_group;
                        load_source_active=fixture_word[fixture_index][15:0];
                        load_source_sign=fixture_word[fixture_index][31:16]
                            & load_source_active;
                    end
                    load_descriptor_both();
                end
        end
    endtask

    task automatic compute_expected_tile;
        integer fixture_index, source, value;
        begin
            for (int context=0; context<BUNDLE; context++)
                for (int slice=0; slice<SLICES; slice++) begin
                    observed_base[context][slice]=0;
                    observed_tsbg[context][slice]=0;
                    for (int lane=0; lane<LANES; lane++) begin
                        expected[context][slice][lane]=0;
                        for (int group=0; group<real_source_groups; group++) begin
                            fixture_index=workload_slot*BUNDLE*MAX_GROUPS
                                + context*MAX_GROUPS+group;
                            for (source=0; source<SOURCES; source=source+1)
                                if (fixture_word[fixture_index][source]) begin
                                    value=fixture_word[fixture_index][16+source]
                                          ? -1 : 1;
                                    expected[context][slice][lane] += value
                                        * directed_weight(
                                            group,source/8,
                                            current_output_tile,
                                            real_source_groups,slice,
                                            source%8,lane);
                                end
                        end
                        if (expected[context][slice][lane] >= (1<<23)
                                || expected[context][slice][lane] < -(1<<23))
                            $fatal(1, "M2067 oracle Acc24 overflow");
                    end
                end
            tile_commits_base=0; tile_commits_tsbg=0;
            tile_terminals_base=0; tile_terminals_tsbg=0;
            base_done_seen=0; tsbg_done_seen=0;
        end
    endtask

    task automatic run_alias_attack(
        input integer groups_i, attack_index_i, bad_base_i);
        integer chunk_total;
        begin
            chunk_total=groups_i/PHYSICAL_GROUPS;
            for (int ci=0; ci<attack_index_i; ci++) begin
                send_header_both(groups_i,0,ci,chunk_total,
                    ci*PHYSICAL_GROUPS,ci==0,
                    ci>0&&ci<chunk_total-1,ci==chunk_total-1);
                load_chunk(ci*PHYSICAL_GROUPS,1);
                while (!(base.chunk_ready && tsbg.chunk_ready))
                    @(posedge clk_core);
            end
            send_invalid_alias_header_both(groups_i,attack_index_i,
                chunk_total,bad_base_i,attack_index_i==0,
                attack_index_i>0&&attack_index_i<chunk_total-1,
                attack_index_i==chunk_total-1);
            reset_both();
            if (base.protocol_error || tsbg.protocol_error)
                $fatal(1, "M2067 reset did not clear alias fault");
        end
    endtask

    task automatic run_output_tile(input integer tile_i);
        integer intermediate_expected;
        begin
            current_output_tile=tile_i;
            compute_expected_tile();
            for (int ci=0; ci<chunks; ci++) begin
                $display("M2067_ROW_CHUNK workload_slot=%0d sample_id=%0d layer_id=%0d token_start=%0d output_tile=%0d source_groups=%0d chunk_index=%0d chunk_count=%0d global_group_base=%0d first=%0d intermediate=%0d final=%0d",
                    workload_slot,sample_id,layer_id,token_start,tile_i,
                    real_source_groups,ci,chunks,ci*PHYSICAL_GROUPS,
                    ci==0,ci>0&&ci<chunks-1,ci==chunks-1);
                row_chunk_records=row_chunk_records+1;
                send_header_both(real_source_groups,tile_i,ci,chunks,
                    ci*PHYSICAL_GROUPS,ci==0,ci>0&&ci<chunks-1,
                    ci==chunks-1);
                load_chunk(ci*PHYSICAL_GROUPS,0);
                if (ci != chunks-1)
                    while (!(base.chunk_ready && tsbg.chunk_ready))
                        @(posedge clk_core);
                else begin
                    while (!(base_done_seen && tsbg_done_seen))
                        @(posedge clk_core);
                    while (!(base.chunk_ready && tsbg.chunk_ready))
                        @(posedge clk_core);
                end
            end
            for (int context=0; context<BUNDLE; context++)
                for (int slice=0; slice<SLICES; slice++)
                    if (!observed_base[context][slice]
                            || !observed_tsbg[context][slice])
                        $fatal(1, "M2067 missing final commit");
            intermediate_expected=(chunks==4)?2:0;
            if (tile_commits_base!=EXPECTED_COMMITS_PER_TILE
                    || tile_commits_tsbg!=EXPECTED_COMMITS_PER_TILE
                    || tile_terminals_base!=BUNDLE
                    || tile_terminals_tsbg!=BUNDLE
                    || base.chunk_count!=chunks || tsbg.chunk_count!=chunks
                    || base.intermediate_chunk_count!=intermediate_expected
                    || tsbg.intermediate_chunk_count!=intermediate_expected
                    || base.final_chunk_count!=1 || tsbg.final_chunk_count!=1
                    || base.final_commit_count!=EXPECTED_COMMITS_PER_TILE
                    || tsbg.final_commit_count!=EXPECTED_COMMITS_PER_TILE
                    || base.continuation_cycles!=2*(chunks-1)
                    || tsbg.continuation_cycles!=2*(chunks-1)
                    || base.descriptor_preload_cycles
                       !=tsbg.descriptor_preload_cycles
                    || base.row_access_count!=tsbg.row_access_count
                    || base.issue_count!=tsbg.issue_count
                    || base.product_count!=tsbg.product_count)
                $fatal(1, "M2067 continuation/fair-fee/work ledger mismatch");
            if (base.numeric_overflow || tsbg.numeric_overflow
                    || base.protocol_error || tsbg.protocol_error)
                $fatal(1, "M2067 clean logical output faulted");
            total_base_cycles += base.logical_cycle_count;
            total_tsbg_cycles += tsbg.logical_cycle_count;
        end
    endtask

    always_comb begin
        base.bridge_ready=(tb_cycle%11!=3);
        tsbg.bridge_ready=(tb_cycle%11!=3);
        base.commit_ready=(tb_cycle%13!=5);
        tsbg.commit_ready=(tb_cycle%13!=5);
        base.bundle_done_ready=1;
        tsbg.bundle_done_ready=1;
    end

    always @(posedge clk_core) begin
        if (!rst_core) begin
            for (int bank=0; bank<8; bank++) begin
                if (base.mem_req_accept[bank]) begin
                    if (base.mem_req_global_group[bank]
                            != (base.mem_req_source_channel[bank]>>4)
                            || base.mem_req_weight_row_index[bank]
                            != current_output_tile*real_source_groups
                               +(base.mem_req_source_channel[bank]>>4)
                            || (base.mem_req_source_channel[bank]>>4)
                               >= real_source_groups)
                        $fatal(1, "M2067 ordinary global address alias");
                    address_checks_base <= address_checks_base+1;
                end
                if (tsbg.mem_req_accept[bank]) begin
                    if (tsbg.mem_req_global_group[bank]
                            != (tsbg.mem_req_source_channel[bank]>>4)
                            || tsbg.mem_req_weight_row_index[bank]
                            != current_output_tile*real_source_groups
                               +(tsbg.mem_req_source_channel[bank]>>4)
                            || (tsbg.mem_req_source_channel[bank]>>4)
                               >= real_source_groups)
                        $fatal(1, "M2067 TSBG global address alias");
                    address_checks_tsbg <= address_checks_tsbg+1;
                end
            end
            if (base.commit_accept) begin
                if (observed_base[base.commit_context][base.commit_slice])
                    $fatal(1, "M2067 duplicate ordinary final commit");
                observed_base[base.commit_context][base.commit_slice] <= 1;
                tile_commits_base <= tile_commits_base+1;
                workload_commits_base <= workload_commits_base+1;
                workload_checks_base <= workload_checks_base+LANES;
                if (base.commit_terminal)
                    tile_terminals_base <= tile_terminals_base+1;
                for (int lane=0; lane<LANES; lane++) begin
                    if (base.commit_accumulator[lane] !==
                            expected[base.commit_context][base.commit_slice][lane])
                        $fatal(1, "M2067 ordinary exact oracle mismatch");
                end
            end
            if (tsbg.commit_accept) begin
                if (observed_tsbg[tsbg.commit_context][tsbg.commit_slice])
                    $fatal(1, "M2067 duplicate TSBG final commit");
                observed_tsbg[tsbg.commit_context][tsbg.commit_slice] <= 1;
                tile_commits_tsbg <= tile_commits_tsbg+1;
                workload_commits_tsbg <= workload_commits_tsbg+1;
                workload_checks_tsbg <= workload_checks_tsbg+LANES;
                if (tsbg.commit_terminal)
                    tile_terminals_tsbg <= tile_terminals_tsbg+1;
                for (int lane=0; lane<LANES; lane++) begin
                    if (tsbg.commit_accumulator[lane] !==
                            expected[tsbg.commit_context][tsbg.commit_slice][lane])
                        $fatal(1, "M2067 TSBG exact oracle mismatch");
                end
            end
            if (base.bundle_done_valid) base_done_seen <= 1;
            if (tsbg.bundle_done_valid) tsbg_done_seen <= 1;
        end
    end

    initial begin
        tb_cycle=0; rst_core=1; initialize_drives();
        current_output_tile=0; alias_attacks=0;
        alias_attacks_g96=0; alias_attacks_g192=0;
        alias_rejects_base=0; alias_rejects_tsbg=0;
        row_chunk_records=0; address_checks_base=0; address_checks_tsbg=0;
        workload_commits_base=0; workload_commits_tsbg=0;
        workload_checks_base=0; workload_checks_tsbg=0;
        total_base_cycles=0; total_tsbg_cycles=0;
        for (int lane=0; lane<LANES; lane++) zero_replay_weight[lane]=0;
        workload_slot=0;
        if (!$value$plusargs("WORKLOAD_SLOT=%d",workload_slot)) workload_slot=0;
        if (workload_slot<0 || workload_slot>=WORKLOADS)
            $fatal(1,"M2067 WORKLOAD_SLOT outside 0..959");
        $readmemh(FIXTURE,fixture_word);
        $readmemh(STATS,stats_word);
        sample_id=stats_word[workload_slot][31:0];
        layer_id=stats_word[workload_slot][63:32];
        token_start=stats_word[workload_slot][95:64];
        real_source_groups=stats_word[workload_slot][127:96];
        output_tiles=stats_word[workload_slot][159:128];
        chunks=stats_word[workload_slot][191:160];
        token_role_id=stats_word[workload_slot][223:192];
        sequence_id=stats_word[workload_slot][255:224];
        expected_commits=stats_word[workload_slot][287:256];
        expected_integer_checks=stats_word[workload_slot][319:288];
        expected_nonzero_codes=stats_word[workload_slot][351:320];
        expected_negative_codes=stats_word[workload_slot][383:352];
        if (!((real_source_groups==96&&chunks==2)
                || (real_source_groups==192&&chunks==4))
                || expected_commits!=output_tiles*EXPECTED_COMMITS_PER_TILE
                || expected_integer_checks!=expected_commits*LANES)
            $fatal(1,"M2067 workload metadata drift");
        repeat (4) @(posedge clk_core);
        @(negedge clk_core); rst_core=0;
        repeat (2) @(posedge clk_core);
        run_alias_attack(96,1,0);
        // After correct G192 chunks 0/48, attack chunk index 2 with base 48
        // rather than its required global base 96.  This specifically proves
        // that local G48 numbering cannot alias a later logical FC2 row.
        run_alias_attack(192,2,48);
        for (int tile=0; tile<output_tiles; tile++) run_output_tile(tile);
        if (workload_commits_base!=expected_commits
                || workload_commits_tsbg!=expected_commits
                || workload_checks_base!=expected_integer_checks
                || workload_checks_tsbg!=expected_integer_checks
                || row_chunk_records!=output_tiles*chunks
                || address_checks_base<=0 || address_checks_tsbg<=0
                || alias_attacks!=2 || alias_attacks_g96!=1
                || alias_attacks_g192!=1 || alias_rejects_base!=2
                || alias_rejects_tsbg!=2 || total_base_cycles<=0
                || total_tsbg_cycles<=0)
            $fatal(1,"M2067 final workload ledger mismatch");
        if (real_source_groups==96)
            global_group_bases_text="0,48";
        else
            global_group_bases_text="0,48,96,144";
        $display("PASS_M2067_EP34_FC2_EXACT_CONTINUATION workload_slot=%0d sample_id=%0d layer_id=%0d token_start=%0d token_role_id=%0d sequence_id=%0d source_groups=%0d physical_groups=48 output_tiles=%0d chunks=%0d global_group_bases=%s commits=%0d integer_checks=%0d oracle_mismatches=0 overflow=0 row_chunk_records=%0d address_checks_base=%0d address_checks_tsbg=%0d alias_attacks=2 alias_attacks_g96=1 alias_attacks_g192=1 alias_rejects_base=2 alias_rejects_tsbg=2 base_cycles=%0d tsbg_cycles=%0d ordinary_tsbg_same_fixed_fees=true real_ep34_sources=true directed_weights=true rtl_speedup_claimed=false system_speedup=false paper_admitted=false",
            workload_slot,sample_id,layer_id,token_start,token_role_id,
            sequence_id,real_source_groups,output_tiles,chunks,
            global_group_bases_text,
            expected_commits,expected_integer_checks,row_chunk_records,
            address_checks_base,address_checks_tsbg,total_base_cycles,
            total_tsbg_cycles);
        $finish;
    end
endmodule

`default_nettype wire
