`timescale 1ns/1ps
`default_nettype none

module tb_m194_fc2_same_token_pair_bank_head_selector;
    localparam int TAG_BITS = 24;
    localparam int CHANNEL_BITS = 12;
    localparam int COUNT_BITS = 8;
    localparam int MAX_EXPECTED = 7000;

    logic clk_core, rst_core;
    logic pair_valid, pair_ready, pair_accept;
    logic [1:0] window_valid;
    logic [TAG_BITS-1:0] window_token_tag [0:1];
    logic [COUNT_BITS-1:0] window_bank_count [0:1][0:7];
    logic [CHANNEL_BITS-1:0] window_head_channel [0:1][0:7];
    logic issue_valid, issue_ready, issue_pair_last, issue_accept;
    logic [TAG_BITS-1:0] issue_token_tag;
    logic [3:0] issue_source_count;
    logic [7:0] issue_bank_valid, issue_selected_window;
    logic [CHANNEL_BITS-1:0] issue_source_channel [0:7];
    logic protocol_error, busy;

    logic [TAG_BITS-1:0] expected_tag [0:MAX_EXPECTED-1];
    logic [3:0] expected_count [0:MAX_EXPECTED-1];
    logic [7:0] expected_valid [0:MAX_EXPECTED-1];
    logic [7:0] expected_window [0:MAX_EXPECTED-1];
    logic expected_last [0:MAX_EXPECTED-1];
    logic [CHANNEL_BITS-1:0] expected_channel
        [0:MAX_EXPECTED-1][0:7];
    integer expected_write, expected_read;
    integer accepted_pairs, accepted_issues, bank_selection_checks;
    integer stall_hold_checks, same_cycle_replace_checks;
    integer final_accepted_pairs, final_accepted_issues;
    integer final_bank_selection_checks, final_same_cycle_replace_checks;
    integer cross_token_attacks, empty_pair_attacks;
    integer invalid_window_attacks, bad_channel_attacks;
    logic scoreboard_enabled;

    m194_fc2_same_token_pair_bank_head_selector dut (.*);
    bind m194_fc2_same_token_pair_bank_head_selector
        m194_fc2_same_token_pair_bank_head_selector_assertions sva (.*);

    initial clk_core = 1'b0;
    always #1.5 clk_core = ~clk_core;

    function automatic integer popcount8(input logic [7:0] value);
        integer count;
        begin
            count = 0;
            for (int bank = 0; bank < 8; bank++)
                count += value[bank];
            return count;
        end
    endfunction

    task automatic clear_inputs;
        begin
            pair_valid = 1'b0;
            window_valid = '0;
            for (int window = 0; window < 2; window++) begin
                window_token_tag[window] = '0;
                for (int bank = 0; bank < 8; bank++) begin
                    window_bank_count[window][bank] = '0;
                    window_head_channel[window][bank] = '0;
                end
            end
        end
    endtask

    task automatic apply_reset;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            pair_valid = 1'b0;
            issue_ready = 1'b1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
        end
    endtask

    task automatic drive_legal(
            input integer ordinal,
            input logic [1:0] valid_value,
            input integer mode);
        integer total_sources;
        begin
            @(negedge clk_core);
            window_valid = valid_value;
            window_token_tag[0] = 24'h510000 + ordinal;
            window_token_tag[1] = window_token_tag[0];
            total_sources = 0;
            for (int window = 0; window < 2; window++) begin
                for (int bank = 0; bank < 8; bank++) begin
                    if (!valid_value[window]) begin
                        window_bank_count[window][bank] = 0;
                    end else if (mode == 1) begin
                        window_bank_count[window][bank]
                            = (window == 0) ? 1 : 0;
                    end else if (mode == 2) begin
                        window_bank_count[window][bank]
                            = (window == 1 && (bank % 2) == 0) ? 1 : 0;
                    end else begin
                        window_bank_count[window][bank]
                            = $urandom_range(0, 7);
                    end
                    window_head_channel[window][bank]
                        = {$urandom_range(0, 511), bank[2:0]};
                    total_sources += window_bank_count[window][bank];
                end
            end
            if (total_sources == 0) begin
                if (valid_value[0])
                    window_bank_count[0][ordinal % 8] = 1;
                else
                    window_bank_count[1][ordinal % 8] = 1;
            end
            pair_valid = 1'b1;
            do @(posedge clk_core); while (!pair_accept);
            @(negedge clk_core);
            pair_valid = 1'b0;
        end
    endtask

    task automatic drain_scoreboard;
        begin
            while (expected_read != expected_write)
                @(posedge clk_core);
            @(negedge clk_core);
        end
    endtask

    // Independent reference is captured directly from each accepted pair.
    always @(posedge clk_core) begin
        logic [7:0] valid_ref;
        logic [7:0] selected_ref;
        logic last_ref;
        integer merged;
        if (rst_core) begin
            expected_write = 0;
            expected_read = 0;
            accepted_pairs = 0;
            accepted_issues = 0;
            bank_selection_checks = 0;
            same_cycle_replace_checks = 0;
        end else if (scoreboard_enabled) begin
            if (pair_accept) begin
                if (expected_write >= MAX_EXPECTED)
                    $fatal(1, "M194 expected queue overflow");
                valid_ref = '0;
                selected_ref = '0;
                last_ref = 1'b1;
                expected_tag[expected_write] = window_valid[0]
                    ? window_token_tag[0] : window_token_tag[1];
                for (int bank = 0; bank < 8; bank++) begin
                    merged = (window_valid[0]
                            ? window_bank_count[0][bank] : 0)
                        + (window_valid[1]
                            ? window_bank_count[1][bank] : 0);
                    valid_ref[bank] = merged != 0;
                    if (window_valid[0]
                            && window_bank_count[0][bank] != 0) begin
                        selected_ref[bank] = 1'b0;
                        expected_channel[expected_write][bank]
                            = window_head_channel[0][bank];
                    end else if (window_valid[1]
                            && window_bank_count[1][bank] != 0) begin
                        selected_ref[bank] = 1'b1;
                        expected_channel[expected_write][bank]
                            = window_head_channel[1][bank];
                    end else begin
                        expected_channel[expected_write][bank] = '0;
                    end
                    if (merged > 1)
                        last_ref = 1'b0;
                end
                expected_valid[expected_write] = valid_ref;
                expected_window[expected_write] = selected_ref;
                expected_count[expected_write] = popcount8(valid_ref);
                expected_last[expected_write] = last_ref;
                expected_write = expected_write + 1;
                accepted_pairs = accepted_pairs + 1;
            end
            if (issue_accept) begin
                if (expected_read >= expected_write)
                    $fatal(1, "M194 issue without expected pair");
                if (issue_token_tag !== expected_tag[expected_read]
                        || issue_source_count !== expected_count[expected_read]
                        || issue_bank_valid !== expected_valid[expected_read]
                        || issue_selected_window
                            !== expected_window[expected_read]
                        || issue_pair_last !== expected_last[expected_read])
                    $fatal(1, "M194 issue header mismatch index=%0d",
                        expected_read);
                for (int bank = 0; bank < 8; bank++) begin
                    if (issue_source_channel[bank]
                            !== expected_channel[expected_read][bank])
                        $fatal(1,
                            "M194 channel mismatch index=%0d bank=%0d",
                            expected_read, bank);
                    bank_selection_checks = bank_selection_checks + 1;
                end
                expected_read = expected_read + 1;
                accepted_issues = accepted_issues + 1;
            end
            if (pair_accept && issue_accept)
                same_cycle_replace_checks = same_cycle_replace_checks + 1;
        end
    end

    initial begin : stimulus
        rst_core = 1'b1;
        issue_ready = 1'b1;
        scoreboard_enabled = 1'b0;
        stall_hold_checks = 0;
        cross_token_attacks = 0;
        empty_pair_attacks = 0;
        invalid_window_attacks = 0;
        bad_channel_attacks = 0;
        final_accepted_pairs = 0;
        final_accepted_issues = 0;
        final_bank_selection_checks = 0;
        final_same_cycle_replace_checks = 0;
        clear_inputs();
        apply_reset();
        scoreboard_enabled = 1'b1;

        // Directed coverage: each window alone, fallthrough, all/partial banks,
        // last/not-last and a held output.
        drive_legal(0, 2'b01, 1);
        drive_legal(1, 2'b10, 2);
        drive_legal(2, 2'b11, 1);
        @(negedge clk_core);
        issue_ready = 1'b0;
        drive_legal(3, 2'b11, 0);
        repeat (3) begin
            @(posedge clk_core);
            if (issue_valid)
                stall_hold_checks = stall_hold_checks + 1;
        end
        @(negedge clk_core);
        issue_ready = 1'b1;

        for (int test = 0; test < 5000; test++) begin
            logic [1:0] valid_value;
            valid_value = $urandom_range(1, 3);
            if ((test % 37) == 0) begin
                @(negedge clk_core);
                issue_ready = 1'b0;
                fork
                    begin
                        repeat ($urandom_range(1, 3)) @(posedge clk_core);
                        @(negedge clk_core);
                        issue_ready = 1'b1;
                    end
                join_none
            end
            drive_legal(test + 10, valid_value, 0);
        end
        issue_ready = 1'b1;
        drain_scoreboard();
        if (accepted_pairs != 5004 || accepted_issues != accepted_pairs)
            $fatal(1, "M194 accepted population drift pairs=%0d issues=%0d",
                accepted_pairs, accepted_issues);
        if (bank_selection_checks != accepted_issues * 8)
            $fatal(1, "M194 bank selection check drift");
        if (stall_hold_checks < 2 || same_cycle_replace_checks < 1)
            $fatal(1, "M194 elastic coverage missing");
        final_accepted_pairs = accepted_pairs;
        final_accepted_issues = accepted_issues;
        final_bank_selection_checks = bank_selection_checks;
        final_same_cycle_replace_checks = same_cycle_replace_checks;

        // Each sticky fail-closed attack is isolated by reset.
        scoreboard_enabled = 1'b0;
        apply_reset();
        @(negedge clk_core);
        window_valid = 2'b11;
        window_token_tag[0] = 24'h1;
        window_token_tag[1] = 24'h2;
        window_bank_count[0][0] = 1;
        window_bank_count[1][1] = 1;
        window_head_channel[0][0] = 12'h000;
        window_head_channel[1][1] = 12'h001;
        pair_valid = 1'b1;
        @(posedge clk_core);
        if (!protocol_error || pair_accept)
            $fatal(1, "M194 cross-token attack not rejected");
        cross_token_attacks = 1;

        apply_reset(); clear_inputs();
        @(negedge clk_core);
        pair_valid = 1'b1;
        @(posedge clk_core);
        if (!protocol_error || pair_accept)
            $fatal(1, "M194 empty-pair attack not rejected");
        empty_pair_attacks = 1;

        apply_reset(); clear_inputs();
        @(negedge clk_core);
        window_valid = 2'b01;
        window_token_tag[0] = 24'h3;
        window_bank_count[1][2] = 1;
        window_bank_count[0][0] = 1;
        window_head_channel[0][0] = 12'h000;
        pair_valid = 1'b1;
        @(posedge clk_core);
        if (!protocol_error || pair_accept)
            $fatal(1, "M194 invalid-window count attack not rejected");
        invalid_window_attacks = 1;

        apply_reset(); clear_inputs();
        @(negedge clk_core);
        window_valid = 2'b01;
        window_token_tag[0] = 24'h4;
        window_bank_count[0][5] = 1;
        window_head_channel[0][5] = 12'h003;
        pair_valid = 1'b1;
        @(posedge clk_core);
        if (!protocol_error || pair_accept)
            $fatal(1, "M194 bad-channel attack not rejected");
            bad_channel_attacks = 1;

        $display("PASS M194 FC2 same-token pair bank-head selector VCS legal_pairs=5004 bank_selection_checks=%0d stalls=%0d same_cycle_replace=%0d cross_token_attacks=%0d empty_pair_attacks=%0d invalid_window_attacks=%0d bad_channel_attacks=%0d physical_banks=8 windows=2 extra_acc24_contexts=0 queue_storage=false sram_response=false complete_fc2=false physical_speedup=false system_speedup=false headline=false",
            final_bank_selection_checks, stall_hold_checks,
            final_same_cycle_replace_checks, cross_token_attacks,
            empty_pair_attacks, invalid_window_attacks,
            bad_channel_attacks);
        $finish;
    end

    initial begin : watchdog
        repeat (300000) @(posedge clk_core);
        $fatal(1, "M194 watchdog timeout");
    end
endmodule

`default_nettype wire
