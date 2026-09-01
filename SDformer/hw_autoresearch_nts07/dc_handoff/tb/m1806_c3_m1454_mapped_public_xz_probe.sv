`timescale 1ns/1ps
`default_nettype none

module m1806_c3_m1454_mapped_public_xz_probe #(
    parameter integer TAG_W = 48
) (
    input logic clk_core,
    input logic rst_core,
    input logic config_ready, config_accept, raw_ready, raw_accept,
    input logic result_valid, result_accept,
    input logic [TAG_W-1:0] result_tag,
    input logic [2:0] result_beat,
    input logic [47:0] result_valid_bits, result_data,
    input logic release_ready, release_accept,
    input logic tile_done_valid,
    input logic [TAG_W-1:0] tile_done_tag,
    input logic context_retire_valid,
    input logic [31:0] context_retire_cycles,
    input logic config_loaded, protocol_error, busy,
    input logic stage1_issue, stage2_issue, product_push, product_replace,
    input logic fifo_push, fifo_pop,
    input logic [4:0] result_fifo_occupancy,
    input logic [1:0] raw_bank_occupancy, intermediate_bank_occupancy,
    input logic [31:0] debug_config_beats, debug_raw_beats,
    input logic [31:0] debug_tiles_loaded, debug_stage1_issues,
    input logic [31:0] debug_stage1_done, debug_stage2_issues,
    input logic [31:0] debug_stage2_done, debug_product_pushes,
    input logic [31:0] debug_result_departures,
    input logic [31:0] debug_product_replacements,
    input logic [31:0] debug_context_cycles
);
    integer cycle_count;
    initial cycle_count = 0;

`define M1806_XZ_CHECK(signal_name) \
    if ($isunknown(signal_name)) \
        $display("M1806_FIRST_X field=%s value=%b cycle=%0d time_ps=%0t", \
            `"signal_name`", signal_name, cycle_count, $time)

    always @(posedge clk_core) begin
        cycle_count = cycle_count + 1;
        if (!rst_core) begin
            #0.1;
            `M1806_XZ_CHECK(config_ready);
            `M1806_XZ_CHECK(config_accept);
            `M1806_XZ_CHECK(raw_ready);
            `M1806_XZ_CHECK(raw_accept);
            `M1806_XZ_CHECK(result_valid);
            `M1806_XZ_CHECK(result_accept);
            `M1806_XZ_CHECK(result_tag);
            `M1806_XZ_CHECK(result_beat);
            `M1806_XZ_CHECK(result_valid_bits);
            `M1806_XZ_CHECK(result_data);
            `M1806_XZ_CHECK(release_ready);
            `M1806_XZ_CHECK(release_accept);
            `M1806_XZ_CHECK(tile_done_valid);
            `M1806_XZ_CHECK(tile_done_tag);
            `M1806_XZ_CHECK(context_retire_valid);
            `M1806_XZ_CHECK(context_retire_cycles);
            `M1806_XZ_CHECK(config_loaded);
            `M1806_XZ_CHECK(protocol_error);
            `M1806_XZ_CHECK(busy);
            `M1806_XZ_CHECK(stage1_issue);
            `M1806_XZ_CHECK(stage2_issue);
            `M1806_XZ_CHECK(product_push);
            `M1806_XZ_CHECK(product_replace);
            `M1806_XZ_CHECK(fifo_push);
            `M1806_XZ_CHECK(fifo_pop);
            `M1806_XZ_CHECK(result_fifo_occupancy);
            `M1806_XZ_CHECK(raw_bank_occupancy);
            `M1806_XZ_CHECK(intermediate_bank_occupancy);
            `M1806_XZ_CHECK(debug_config_beats);
            `M1806_XZ_CHECK(debug_raw_beats);
            `M1806_XZ_CHECK(debug_tiles_loaded);
            `M1806_XZ_CHECK(debug_stage1_issues);
            `M1806_XZ_CHECK(debug_stage1_done);
            `M1806_XZ_CHECK(debug_stage2_issues);
            `M1806_XZ_CHECK(debug_stage2_done);
            `M1806_XZ_CHECK(debug_product_pushes);
            `M1806_XZ_CHECK(debug_result_departures);
            `M1806_XZ_CHECK(debug_product_replacements);
            `M1806_XZ_CHECK(debug_context_cycles);
        end
    end
`undef M1806_XZ_CHECK
endmodule

bind tb_m1790_c3_m1454_fixed_t10_mapped_energy
    m1806_c3_m1454_mapped_public_xz_probe m1806_probe (.*);

`default_nettype wire
