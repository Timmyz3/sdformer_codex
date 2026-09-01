`timescale 1ns/1ps
`default_nettype none

// Additive M1808 correction to the sealed M1790 public-port workload.
//
// This monitor observes only signals already wired between the M1790
// testbench and the public ports of m518_matched_fixed_t10_atlif.  It neither
// reads nor drives private DUT hierarchy.  The accepted final raw beat of each
// tile is checked against the independently fixed directed tag sequence, then
// enqueued.  Every tile_done_tag must retire the queue head in order.  Thus a
// stale, duplicated, missing, reordered, or wrong tile-done tag cannot inherit
// the predecessor PASS token.
module m1808_c3_tile_done_tag_scoreboard #(
    parameter integer TAG_W = 48
) (
    input logic clk_core,
    input logic rst_core,
    input logic raw_accept,
    input logic raw_last,
    input logic [TAG_W-1:0] raw_tag,
    input logic raw_valid,
    input logic raw_ready,
    input logic result_valid,
    input logic result_ready,
    input logic tile_done_valid,
    input logic [TAG_W-1:0] tile_done_tag
);
    localparam integer EXPECTED_TOTAL_TAGS = 9;
    localparam integer EXPECTED_MEASURED_TAGS = 8;

    logic [TAG_W-1:0] expected_tile_done_tag [0:15];
    logic sampled_raw_accept, sampled_raw_last;
    logic sampled_tile_done_valid;
    logic [TAG_W-1:0] sampled_raw_tag, sampled_tile_done_tag;
    integer expected_write, expected_read, tag_mismatches;
    integer raw_stall_cycles, result_stall_cycles;
    logic pass_printed;

    function automatic logic [TAG_W-1:0] directed_tag(input integer index);
        begin
            if (index == 0)
                directed_tag = 48'h1790_0000_0000;
            else
                directed_tag = 48'h1790_1000_0000 + (index - 1);
        end
    endfunction

    initial begin
        expected_write = 0;
        expected_read = 0;
        tag_mismatches = 0;
        raw_stall_cycles = 0;
        result_stall_cycles = 0;
        pass_printed = 1'b0;
    end

    always @(posedge clk_core) begin
        sampled_raw_accept = raw_accept;
        sampled_raw_last = raw_last;
        sampled_raw_tag = raw_tag;
        sampled_tile_done_valid = tile_done_valid;
        sampled_tile_done_tag = tile_done_tag;
        #0.2;
        if (rst_core) begin
            expected_write = 0;
            expected_read = 0;
            tag_mismatches = 0;
            raw_stall_cycles = 0;
            result_stall_cycles = 0;
            pass_printed = 1'b0;
        end else begin
            if (raw_valid && !raw_ready)
                raw_stall_cycles = raw_stall_cycles + 1;
            if (result_valid && !result_ready)
                result_stall_cycles = result_stall_cycles + 1;

            if (sampled_raw_accept && sampled_raw_last) begin
                if (expected_write >= EXPECTED_TOTAL_TAGS)
                    $fatal(1, "M1808 extra accepted tile tag index=%0d",
                        expected_write);
                if ($isunknown(sampled_raw_tag))
                    $fatal(1, "M1808 accepted tile tag contains X/Z");
                if (sampled_raw_tag !== directed_tag(expected_write)) begin
                    tag_mismatches = tag_mismatches + 1;
                    $fatal(1,
                        "M1808 directed input tag mismatch index=%0d got=%h want=%h",
                        expected_write, sampled_raw_tag,
                        directed_tag(expected_write));
                end
                expected_tile_done_tag[expected_write] =
                    directed_tag(expected_write);
                expected_write = expected_write + 1;
            end

            if (sampled_tile_done_valid) begin
                if ($isunknown(sampled_tile_done_tag))
                    $fatal(1, "M1808 tile-done tag contains X/Z");
                if (expected_read >= expected_write)
                    $fatal(1, "M1808 unexpected tile-done tag index=%0d",
                        expected_read);
                if (sampled_tile_done_tag !==
                        expected_tile_done_tag[expected_read]) begin
                    tag_mismatches = tag_mismatches + 1;
                    $fatal(1,
                        "M1808 tile-done tag mismatch index=%0d got=%h want=%h",
                        expected_read, sampled_tile_done_tag,
                        expected_tile_done_tag[expected_read]);
                end
                expected_read = expected_read + 1;
                if (expected_read == EXPECTED_TOTAL_TAGS) begin
                    if (expected_write != EXPECTED_TOTAL_TAGS
                            || tag_mismatches != 0
                            || raw_stall_cycles == 0
                            || result_stall_cycles == 0)
                        $fatal(1,
                            "M1808 tag coverage/conservation failure write=%0d read=%0d mismatch=%0d stalls=%0d/%0d",
                            expected_write, expected_read, tag_mismatches,
                            raw_stall_cycles, result_stall_cycles);
                    if (pass_printed)
                        $fatal(1, "M1808 duplicate tag scoreboard PASS");
                    pass_printed = 1'b1;
                    $display("M1808_TILE_DONE_TAG_CHECK total=%0d warmup=1 measured=%0d mismatches=%0d raw_stall=%0d result_stall=%0d",
                        EXPECTED_TOTAL_TAGS, EXPECTED_MEASURED_TAGS,
                        tag_mismatches, raw_stall_cycles,
                        result_stall_cycles);
                    $display("PASS_M1808_C3_ORDERED_TILE_DONE_TAG_SCOREBOARD");
                end
            end
        end
    end
endmodule

bind tb_m1808_c3_m1454_fixed_t10_mapped_energy
    m1808_c3_tile_done_tag_scoreboard m1808_tile_done_tag_scoreboard (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .raw_accept(raw_accept),
        .raw_last(raw_last),
        .raw_tag(raw_tag),
        .raw_valid(raw_valid),
        .raw_ready(raw_ready),
        .result_valid(result_valid),
        .result_ready(result_ready),
        .tile_done_valid(tile_done_valid),
        .tile_done_tag(tile_done_tag)
    );

`default_nettype wire


