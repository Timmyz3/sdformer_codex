`timescale 1ns/1ps
`default_nettype none

module tb_m190_fc2_k7_single_hole_elision_accumulator;
    localparam int LANES = 96;
    logic clk_core;
    logic rst_core;
    logic issue_valid;
    logic issue_ready;
    logic [23:0] issue_tag;
    logic issue_last;
    logic [7:0] issue_bank_valid;
    logic signed [7:0] issue_weight_bank [0:7][0:LANES-1];
    logic signed [23:0] issue_accumulator [0:LANES-1];
    logic issue_accept;
    logic result_valid;
    logic result_ready;
    logic [23:0] result_tag;
    logic result_last;
    logic [3:0] result_source_count;
    logic [7:0] result_bank_mask;
    logic signed [23:0] result_accumulator [0:LANES-1];
    logic result_accept;
    logic [8*LANES-1:0] accepted_weight_bank_active_mask;
    logic [7*LANES-1:0] accepted_elided_lane_active_mask;
    logic [2:0] accepted_hole_bank;
    logic protocol_error;
    logic numeric_overflow;
    logic busy;

    logic signed [23:0] expected_accumulator [0:LANES-1];
    logic [23:0] expected_tag;
    logic expected_last;
    logic [7:0] expected_mask;
    integer legal_mask_checks;
    integer numeric_lane_checks;
    integer stall_hold_checks;
    integer same_cycle_replace_checks;

    m190_fc2_k7_single_hole_elision_accumulator dut (.*);
    bind m190_fc2_k7_single_hole_elision_accumulator
        m190_fc2_k7_single_hole_elision_accumulator_assertions sva (.*);

    initial clk_core = 1'b0;
    always #1.5 clk_core = ~clk_core;

    function automatic integer popcount8(input logic [7:0] mask);
        begin
            popcount8 = 0;
            for (int bank = 0; bank < 8; bank++)
                popcount8 = popcount8 + mask[bank];
        end
    endfunction

    function automatic integer lowest_hole(input logic [7:0] mask);
        begin
            lowest_hole = 7;
            for (int bank = 7; bank >= 0; bank--) begin
                if (!mask[bank])
                    lowest_hole = bank;
            end
        end
    endfunction

    task automatic load_vector(
        input logic [7:0] mask,
        input integer tag_value
    );
        integer total;
        begin
            expected_mask = mask;
            expected_tag = tag_value;
            expected_last = (tag_value % 13) == 0;
            issue_bank_valid = mask;
            issue_tag = expected_tag;
            issue_last = expected_last;
            for (int bank = 0; bank < 8; bank++) begin
                for (int lane = 0; lane < LANES; lane++)
                    issue_weight_bank[bank][lane]
                        = ((bank*31 + lane*17 + tag_value*13) % 255) - 127;
            end
            for (int lane = 0; lane < LANES; lane++) begin
                issue_accumulator[lane]
                    = ((lane*1009 + tag_value*23) % 100001) - 50000;
                total = $signed(issue_accumulator[lane]);
                for (int bank = 0; bank < 8; bank++) begin
                    if (mask[bank])
                        total = total + $signed(issue_weight_bank[bank][lane]);
                end
                expected_accumulator[lane] = total;
            end
        end
    endtask

    task automatic check_accept_mapping(input logic [7:0] mask);
        integer hole;
        integer mapped_bank;
        begin
            hole = lowest_hole(mask);
            if (!issue_accept || accepted_hole_bank !== hole[2:0])
                $fatal(1, "M190 accept/hole mismatch mask=%02x got=%0d expected=%0d",
                    mask, accepted_hole_bank, hole);
            for (int bank = 0; bank < 8; bank++) begin
                for (int lane = 0; lane < LANES; lane++) begin
                    if (accepted_weight_bank_active_mask[(bank*LANES)+lane]
                            !== mask[bank])
                        $fatal(1, "M190 bank activity mismatch");
                end
            end
            for (int slot = 0; slot < 7; slot++) begin
                mapped_bank = slot < hole ? slot : slot + 1;
                for (int lane = 0; lane < LANES; lane++) begin
                    if (accepted_elided_lane_active_mask[(slot*LANES)+lane]
                            !== mask[mapped_bank])
                        $fatal(1, "M190 elided activity mismatch mask=%02x slot=%0d",
                            mask, slot);
                end
            end
        end
    endtask

    task automatic check_result;
        begin
            if (!result_valid || result_tag !== expected_tag
                    || result_last !== expected_last
                    || result_bank_mask !== expected_mask
                    || result_source_count !== popcount8(expected_mask))
                $fatal(1, "M190 result header mismatch mask=%02x", expected_mask);
            for (int lane = 0; lane < LANES; lane++) begin
                if (result_accumulator[lane] !== expected_accumulator[lane])
                    $fatal(1, "M190 numeric mismatch mask=%02x lane=%0d got=%0d expected=%0d",
                        expected_mask, lane, result_accumulator[lane],
                        expected_accumulator[lane]);
                numeric_lane_checks = numeric_lane_checks + 1;
            end
        end
    endtask

    task automatic send_one(
        input logic [7:0] mask,
        input integer tag_value
    );
        begin
            @(negedge clk_core);
            load_vector(mask, tag_value);
            issue_valid = 1'b1;
            @(posedge clk_core);
            check_accept_mapping(mask);
            @(negedge clk_core);
            issue_valid = 1'b0;
            check_result();
            @(posedge clk_core);
            if (!result_accept)
                $fatal(1, "M190 result did not drain");
            legal_mask_checks = legal_mask_checks + 1;
        end
    endtask

    initial begin
        rst_core = 1'b1;
        issue_valid = 1'b0;
        issue_tag = '0;
        issue_last = 1'b0;
        issue_bank_valid = '0;
        result_ready = 1'b1;
        legal_mask_checks = 0;
        numeric_lane_checks = 0;
        stall_hold_checks = 0;
        same_cycle_replace_checks = 0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        // Exhaust every legal structural mask and all lowest-hole positions.
        for (int mask = 1; mask <= 254; mask++)
            send_one(mask[7:0], 24'h700000 + mask);

        // Hold one result stable for two cycles of downstream backpressure.
        @(negedge clk_core);
        result_ready = 1'b0;
        load_vector(8'h81, 24'h710081);
        issue_valid = 1'b1;
        @(posedge clk_core);
        check_accept_mapping(8'h81);
        @(negedge clk_core);
        issue_valid = 1'b0;
        check_result();
        repeat (2) begin
            @(posedge clk_core);
            if (!result_valid || result_accept)
                $fatal(1, "M190 stalled result disappeared");
            stall_hold_checks = stall_hold_checks + 1;
        end
        @(negedge clk_core);
        check_result();
        result_ready = 1'b1;
        @(posedge clk_core);

        // Consecutive issues prove II=1 and same-cycle result replacement.
        @(negedge clk_core);
        load_vector(8'h7f, 24'h720001);
        issue_valid = 1'b1;
        @(posedge clk_core);
        check_accept_mapping(8'h7f);
        @(negedge clk_core);
        check_result();
        load_vector(8'hfe, 24'h720002);
        @(posedge clk_core);
        if (!(result_accept && issue_accept))
            $fatal(1, "M190 same-cycle replacement missing");
        check_accept_mapping(8'hfe);
        same_cycle_replace_checks = same_cycle_replace_checks + 1;
        @(negedge clk_core);
        issue_valid = 1'b0;
        check_result();
        @(posedge clk_core);

        // Legal overflow remains drainable; illegal full/empty masks fail shut.
        @(negedge clk_core);
        issue_tag = 24'hfff900;
        issue_last = 1'b1;
        issue_bank_valid = 8'hfe;
        for (int bank = 0; bank < 8; bank++) begin
            for (int lane = 0; lane < LANES; lane++)
                issue_weight_bank[bank][lane] = 8'sd127;
        end
        for (int lane = 0; lane < LANES; lane++)
            issue_accumulator[lane] = 24'sh7ffff0;
        issue_valid = 1'b1;
        @(posedge clk_core);
        check_accept_mapping(8'hfe);
        @(negedge clk_core);
        issue_bank_valid = 8'hff;
        @(posedge clk_core);
        @(negedge clk_core);
        issue_bank_valid = 8'h00;
        @(posedge clk_core);
        @(negedge clk_core);
        issue_valid = 1'b0;
        repeat (2) @(posedge clk_core);

        if (!numeric_overflow || !protocol_error || issue_ready)
            $fatal(1, "M190 numeric/protocol fail-close missing");
        if (legal_mask_checks != 254
                || numeric_lane_checks != 24768
                || stall_hold_checks != 2
                || same_cycle_replace_checks != 1)
            $fatal(1, "M190 population drift masks=%0d lanes=%0d holds=%0d replace=%0d",
                legal_mask_checks, numeric_lane_checks,
                stall_hold_checks, same_cycle_replace_checks);

        $display("PASS M190 FC2 K7 single-hole-elision accumulator VCS legal_masks_exhausted=254 numeric_lane_checks=24768 lowest_hole_positions=8 stall_hold_checks=2 same_cycle_replace_checks=1 overflow_attacks=1 full_mask_attacks=1 empty_mask_attacks=1 output_lanes=96 structural_weight_banks=8 elided_weight_lanes=7 adjacent_choices_per_lane=2 stable_prefix_compaction=false multipliers=0 sn2_threshold_frozen_one_required=true weight_sram_response=false full_fc2=false bn2=false residual=false physical_speedup=false system_speedup=false headline=false");
        $finish;
    end

    initial begin
        #500000;
        $fatal(1, "M190 watchdog timeout");
    end
endmodule

`default_nettype wire
