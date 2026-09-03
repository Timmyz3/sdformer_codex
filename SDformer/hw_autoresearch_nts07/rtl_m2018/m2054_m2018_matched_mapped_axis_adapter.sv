`timescale 1ns/1ps
`default_nettype none

// Simulation-only adapter for a matched M2018 ordinary/TSBG gate-activity
// experiment.  Exactly one schedule mode is mapped in each build; the other
// side of the existing dual-DUT scoreboard remains the frozen RTL.  This lets
// the mature M2051 arithmetic/protocol checks validate the mapped axis while
// SAIF is collected only below g_mapped.mapped_implementation.
`ifdef M2054_AXIS_ORDINARY
  `define M2054_MAPPED_MODE 0
  `define M2054_MAPPED_TOP m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_SCHEDULE_MODE0
`elsif M2054_AXIS_TSBG
  `define M2054_MAPPED_MODE 1
  `define M2054_MAPPED_TOP m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_SCHEDULE_MODE1
`else
  `error "M2054 requires exactly one mapped-axis define"
`endif

module m1880_c2_tsbg_b4_real_channel_signed_frontend #(
    parameter int SCHEDULE_MODE = 1,
    parameter int BUNDLE = 4,
    parameter int SOURCE_GROUPS = 48,
    parameter int SOURCES_PER_GROUP = 16,
    parameter int OUTPUT_SLICES = 6,
    parameter int CACHE_ROWS = 4,
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int EPOCH_BITS = 16,
    parameter int GENERATION_BITS = 32,
    parameter int LANES = 16
) (
    input  logic clk_core, input logic rst_core,
    input  logic load_valid, output logic load_ready,
    input  logic [2:0] load_context,
    input  logic [TAG_BITS-1:0] load_tag,
    input  logic [5:0] load_group,
    input  logic [SOURCES_PER_GROUP-1:0] load_source_active,
    input  logic [SOURCES_PER_GROUP-1:0] load_source_sign,
    input  logic load_last, output logic load_accept,
    output logic [7:0] mem_req_valid, input logic [7:0] mem_req_ready,
    output logic [EPOCH_BITS-1:0] mem_req_epoch [0:7],
    output logic [2:0] mem_req_slot [0:7],
    output logic [GENERATION_BITS-1:0] mem_req_generation [0:7],
    output logic [TAG_BITS-1:0] mem_req_tag [0:7],
    output logic [2:0] mem_req_output_block [0:7],
    output logic [2:0] mem_req_slice [0:7],
    output logic [CHANNEL_BITS-1:0] mem_req_source_channel [0:7],
    output logic [7:0] mem_req_accept,
    input logic [7:0] mem_rsp_valid, output logic [7:0] mem_rsp_ready,
    input logic [EPOCH_BITS-1:0] mem_rsp_epoch [0:7],
    input logic [2:0] mem_rsp_slot [0:7],
    input logic [GENERATION_BITS-1:0] mem_rsp_generation [0:7],
    input logic [TAG_BITS-1:0] mem_rsp_tag [0:7],
    input logic signed [7:0] mem_rsp_weight [0:7][0:LANES-1],
    output logic [7:0] mem_rsp_accept,
    output logic bridge_valid, input logic bridge_ready,
    output logic [2:0] bridge_context,
    output logic [5:0] bridge_group, output logic bridge_half,
    output logic [2:0] bridge_slice,
    output logic [7:0] bridge_bank_valid,
    output logic [CHANNEL_BITS-1:0] bridge_source_channel [0:7],
    output logic signed [1:0] bridge_source_value [0:7],
    output logic signed [8:0] bridge_effective_weight [0:7][0:LANES-1],
    output logic bridge_accept,
    output logic commit_valid, input logic commit_ready,
    output logic [2:0] commit_context,
    output logic [TAG_BITS-1:0] commit_tag,
    output logic [2:0] commit_slice,
    output logic signed [23:0] commit_accumulator [0:LANES-1],
    output logic commit_terminal, output logic commit_accept,
    output logic bundle_done_valid, input logic bundle_done_ready,
    output logic protocol_error, output logic stale_response_seen,
    output logic numeric_overflow, output logic busy,
    output logic [31:0] debug_cycle_count,
    output logic [31:0] debug_row_access_count,
    output logic [31:0] debug_cache_hit_count,
    output logic [31:0] debug_cache_miss_count,
    output logic [31:0] debug_cache_eviction_count,
    output logic [31:0] debug_weight_bundle_beat_count,
    output logic [31:0] debug_scalar_bank_request_count,
    output logic [31:0] debug_scalar_bank_response_count,
    output logic [31:0] debug_issue_count,
    output logic [31:0] debug_signed_product_count,
    output logic [31:0] debug_commit_count
);
    generate
        if (SCHEDULE_MODE == `M2054_MAPPED_MODE) begin : g_mapped
            logic [127:0] mem_req_epoch_p, mem_rsp_epoch_p;
            logic [23:0] mem_req_slot_p, mem_rsp_slot_p;
            logic [255:0] mem_req_generation_p, mem_rsp_generation_p;
            logic [191:0] mem_req_tag_p, mem_rsp_tag_p;
            logic [23:0] mem_req_output_block_p, mem_req_slice_p;
            logic [95:0] mem_req_source_channel_p;
            logic [1023:0] mem_rsp_weight_p;
            logic [95:0] bridge_source_channel_p;
            logic [15:0] bridge_source_value_p;
            logic [1151:0] bridge_effective_weight_p;
            logic [383:0] commit_accumulator_p;

            for (genvar bank = 0; bank < 8; bank++) begin : g_bank_pack
                assign mem_req_epoch[bank] = mem_req_epoch_p[bank*16 +: 16];
                assign mem_req_slot[bank] = mem_req_slot_p[bank*3 +: 3];
                assign mem_req_generation[bank] =
                    mem_req_generation_p[bank*32 +: 32];
                assign mem_req_tag[bank] = mem_req_tag_p[bank*24 +: 24];
                assign mem_req_output_block[bank] =
                    mem_req_output_block_p[bank*3 +: 3];
                assign mem_req_slice[bank] = mem_req_slice_p[bank*3 +: 3];
                assign mem_req_source_channel[bank] =
                    mem_req_source_channel_p[bank*12 +: 12];
                assign mem_rsp_epoch_p[bank*16 +: 16] = mem_rsp_epoch[bank];
                assign mem_rsp_slot_p[bank*3 +: 3] = mem_rsp_slot[bank];
                assign mem_rsp_generation_p[bank*32 +: 32] =
                    mem_rsp_generation[bank];
                assign mem_rsp_tag_p[bank*24 +: 24] = mem_rsp_tag[bank];
                assign bridge_source_channel[bank] =
                    bridge_source_channel_p[bank*12 +: 12];
                assign bridge_source_value[bank] =
                    bridge_source_value_p[bank*2 +: 2];
                for (genvar lane = 0; lane < 16; lane++) begin : g_lane_pack
                    localparam int WEIGHT_INDEX = (bank*16 + lane)*8;
                    localparam int EFFECTIVE_INDEX = (bank*16 + lane)*9;
                    assign mem_rsp_weight_p[WEIGHT_INDEX +: 8] =
                        mem_rsp_weight[bank][lane];
                    assign bridge_effective_weight[bank][lane] =
                        bridge_effective_weight_p[EFFECTIVE_INDEX +: 9];
                end
            end
            for (genvar lane = 0; lane < 16; lane++) begin : g_acc_pack
                assign commit_accumulator[lane] =
                    commit_accumulator_p[lane*24 +: 24];
            end

            `M2054_MAPPED_TOP mapped_implementation (
                .clk_core(clk_core), .rst_core(rst_core),
                .load_valid(load_valid), .load_ready(load_ready),
                .load_context(load_context), .load_tag(load_tag),
                .load_group(load_group), .load_source_active(load_source_active),
                .load_source_sign(load_source_sign), .load_last(load_last),
                .load_accept(load_accept), .mem_req_valid(mem_req_valid),
                .mem_req_ready(mem_req_ready), .mem_req_epoch(mem_req_epoch_p),
                .mem_req_slot(mem_req_slot_p),
                .mem_req_generation(mem_req_generation_p),
                .mem_req_tag(mem_req_tag_p),
                .mem_req_output_block(mem_req_output_block_p),
                .mem_req_slice(mem_req_slice_p),
                .mem_req_source_channel(mem_req_source_channel_p),
                .mem_req_accept(mem_req_accept), .mem_rsp_valid(mem_rsp_valid),
                .mem_rsp_ready(mem_rsp_ready), .mem_rsp_epoch(mem_rsp_epoch_p),
                .mem_rsp_slot(mem_rsp_slot_p),
                .mem_rsp_generation(mem_rsp_generation_p),
                .mem_rsp_tag(mem_rsp_tag_p), .mem_rsp_weight(mem_rsp_weight_p),
                .mem_rsp_accept(mem_rsp_accept), .bridge_valid(bridge_valid),
                .bridge_ready(bridge_ready), .bridge_context(bridge_context),
                .bridge_group(bridge_group), .bridge_half(bridge_half),
                .bridge_slice(bridge_slice),
                .bridge_bank_valid(bridge_bank_valid),
                .bridge_source_channel(bridge_source_channel_p),
                .bridge_source_value(bridge_source_value_p),
                .bridge_effective_weight(bridge_effective_weight_p),
                .bridge_accept(bridge_accept), .commit_valid(commit_valid),
                .commit_ready(commit_ready), .commit_context(commit_context),
                .commit_tag(commit_tag), .commit_slice(commit_slice),
                .commit_accumulator(commit_accumulator_p),
                .commit_terminal(commit_terminal), .commit_accept(commit_accept),
                .bundle_done_valid(bundle_done_valid),
                .bundle_done_ready(bundle_done_ready),
                .protocol_error(protocol_error),
                .stale_response_seen(stale_response_seen),
                .numeric_overflow(numeric_overflow), .busy(busy),
                .debug_cycle_count(debug_cycle_count),
                .debug_row_access_count(debug_row_access_count),
                .debug_cache_hit_count(debug_cache_hit_count),
                .debug_cache_miss_count(debug_cache_miss_count),
                .debug_cache_eviction_count(debug_cache_eviction_count),
                .debug_weight_bundle_beat_count(debug_weight_bundle_beat_count),
                .debug_scalar_bank_request_count(debug_scalar_bank_request_count),
                .debug_scalar_bank_response_count(debug_scalar_bank_response_count),
                .debug_issue_count(debug_issue_count),
                .debug_signed_product_count(debug_signed_product_count),
                .debug_commit_count(debug_commit_count));
        end else begin : g_rtl
            m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend #(
                .SCHEDULE_MODE(SCHEDULE_MODE), .BUNDLE(BUNDLE),
                .SOURCE_GROUPS(SOURCE_GROUPS),
                .SOURCES_PER_GROUP(SOURCES_PER_GROUP),
                .OUTPUT_SLICES(OUTPUT_SLICES), .CACHE_ROWS(CACHE_ROWS),
                .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
                .EPOCH_BITS(EPOCH_BITS), .GENERATION_BITS(GENERATION_BITS),
                .LANES(LANES)) rtl_implementation (.*);
        end
    endgenerate
endmodule

`undef M2054_MAPPED_MODE
`undef M2054_MAPPED_TOP
`default_nettype wire
