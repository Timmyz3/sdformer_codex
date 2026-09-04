`timescale 1ns/1ps
`default_nettype none

module tb_m405r3_q32_elastic_integration_repair;
    localparam int TAG_BITS = 24;
    logic clk_core, reset_n;
    logic config_valid, config_ready, config_accept;
    logic [1:0] config_beat_index;
    logic config_commit;
    logic [TAG_BITS-1:0] config_tag;
    logic [255:0] config_data;
    logic phase_release_valid, phase_release_ready, phase_release_accept;
    logic row_valid, row_ready, row_accept;
    logic [11:0] row_id;
    logic [15:0] row_original;
    logic row_last;
    logic result_valid, result_ready, result_accept;
    logic [TAG_BITS-1:0] result_tag;
    logic [11:0] result_row_id;
    logic [15:0] result_original;
    logic [4:0] result_center_id, result_distance;
    logic result_use_pwp, result_last;
    logic pwp_low_valid, pwp_low_ready, pwp_low_accept;
    logic [TAG_BITS-1:0] pwp_low_tag;
    logic pwp_low_tile;
    logic [4:0] pwp_low_center_id;
    logic [2:0] pwp_low_output_block;
    logic [767:0] pwp_low_data;
    logic pwp_high_valid, pwp_high_ready, pwp_high_accept;
    logic [TAG_BITS-1:0] pwp_high_tag;
    logic pwp_high_tile;
    logic [4:0] pwp_high_center_id;
    logic [2:0] pwp_high_output_block;
    logic [511:0] pwp_high_data;
    logic contribution_valid, contribution_ready, contribution_accept;
    logic [TAG_BITS-1:0] contribution_tag;
    logic contribution_tile;
    logic [4:0] contribution_center_id;
    logic [2:0] contribution_output_block;
    logic contribution_narrow, contribution_part_high, contribution_last;
    logic [1151:0] contribution_data;
    logic protocol_error, busy;

    integer legal_replay_after_last, legal_releases;
    integer global_fault_attacks, post_fault_accepts;

    m405_q32_elastic_selected_slice #(
        .TAG_BITS(TAG_BITS), .ROWS_PER_PHASE(1)
    ) dut (.*);
    m405_q32_elastic_selected_slice_assertions shell_sva (.*);

    always #1.5 clk_core = ~clk_core;

    task automatic clear_inputs;
        begin
            config_valid = 0;
            config_beat_index = 0;
            config_commit = 0;
            config_tag = 0;
            config_data = 0;
            phase_release_valid = 0;
            row_valid = 0;
            row_id = 0;
            row_original = 0;
            row_last = 0;
            result_ready = 0;
            pwp_low_valid = 0;
            pwp_low_tag = 0;
            pwp_low_tile = 0;
            pwp_low_center_id = 0;
            pwp_low_output_block = 0;
            pwp_low_data = 0;
            pwp_high_valid = 0;
            pwp_high_tag = 0;
            pwp_high_tile = 0;
            pwp_high_center_id = 0;
            pwp_high_output_block = 0;
            pwp_high_data = 0;
            contribution_ready = 1;
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            reset_n = 0;
            clear_inputs();
            repeat (4) @(posedge clk_core);
            @(negedge clk_core); reset_n = 1;
        end
    endtask

    task automatic drive_config(input logic [23:0] tag_value);
        begin
            for (int beat = 0; beat < 3; beat++) begin
                @(negedge clk_core);
                config_valid = 1;
                config_beat_index = beat[1:0];
                config_commit = beat == 2;
                config_tag = tag_value;
                // All-zero centers and all-narrow bitmap.
                config_data = beat == 2 ? {256{1'b1}} : '0;
                do @(posedge clk_core); while (!config_accept
                                               && !protocol_error);
            end
            @(negedge clk_core); config_valid = 0;
        end
    endtask

    task automatic drive_last_zero_row;
        begin
            if (clk_core !== 0) @(negedge clk_core);
            row_valid = 1;
            row_id = 0;
            row_original = 0;
            row_last = 1;
            do @(posedge clk_core); while (!row_accept && !protocol_error);
            @(negedge clk_core); row_valid = 0;
            wait (result_valid || protocol_error);
        end
    endtask

    initial begin
        clk_core = 0;
        reset_n = 0;
        clear_inputs();
        legal_replay_after_last = 0;
        legal_releases = 0;
        global_fault_attacks = 0;
        post_fault_accepts = 0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core); reset_n = 1;

        // Positive lifetime test: row matching is complete, but config must
        // remain live until replay and explicit phase release are complete.
        drive_config(24'h407001);
        drive_last_zero_row();
        if (!dut.configuration_live_w || phase_release_ready)
            $fatal(1,"M405R3 config not retained or premature release ready");
        @(negedge clk_core);
        pwp_low_valid = 1;
        pwp_low_tag = 24'h407001;
        pwp_low_center_id = 0;
        pwp_low_output_block = 0;
        pwp_low_data = '0;
        do @(posedge clk_core); while (!pwp_low_accept && !protocol_error);
        @(negedge clk_core); pwp_low_valid = 0;
        wait (contribution_accept || protocol_error);
        if (protocol_error || contribution_tag != 24'h407001
                || !contribution_narrow || !contribution_last
                || contribution_data != 0)
            $fatal(1,"M405R3 legal replay after last row failed");
        legal_replay_after_last++;
        @(negedge clk_core); result_ready = 1;
        do @(posedge clk_core); while (!result_accept && !protocol_error);
        @(negedge clk_core); result_ready = 0;
        if (!phase_release_ready || !dut.configuration_live_w)
            $fatal(1,"M405R3 legal release boundary absent");
        phase_release_valid = 1;
        @(posedge clk_core);
        if (!phase_release_accept)
            $fatal(1,"M405R3 legal release not accepted");
        @(negedge clk_core); phase_release_valid = 0;
        @(posedge clk_core);
        if (dut.configuration_live_w || busy || protocol_error)
            $fatal(1,"M405R3 release did not quiesce cleanly");
        legal_releases++;

        // Negative test: wrong-tag PWP is rejected combinationally and then
        // latched globally. A pending matcher result may never leak or retire.
        reset_dut();
        drive_config(24'h407002);
        drive_last_zero_row();
        @(negedge clk_core);
        pwp_low_valid = 1;
        pwp_low_tag = 24'hbad002;
        #1;
        if (!protocol_error || pwp_low_ready || pwp_low_accept
                || result_valid || result_accept || config_ready
                || row_ready || pwp_high_ready || contribution_valid)
            $fatal(1,"M405R3 same-cycle global fail-closed failure");
        @(posedge clk_core);
        @(negedge clk_core);
        pwp_low_valid = 0;
        result_ready = 1;
        repeat (4) begin
            @(posedge clk_core);
            if (config_ready || config_accept || phase_release_ready
                    || phase_release_accept || row_ready || row_accept
                    || result_valid || result_accept || pwp_low_ready
                    || pwp_low_accept || pwp_high_ready || pwp_high_accept
                    || contribution_valid || contribution_accept)
                post_fault_accepts++;
        end
        if (!protocol_error || post_fault_accepts != 0)
            $fatal(1,"M405R3 sticky global quiescence failure accepts=%0d",
                   post_fault_accepts);
        global_fault_attacks++;

        // Negative test: releasing while row matching is still active is an
        // immediate global protocol violation and cannot reach the matcher.
        reset_dut();
        drive_config(24'h407003);
        @(negedge clk_core); phase_release_valid = 1;
        #1;
        if (!protocol_error || phase_release_ready || phase_release_accept
                || row_ready || result_valid || pwp_low_ready)
            $fatal(1,"M405R3 early release not fail-closed");
        @(posedge clk_core);
        global_fault_attacks++;

        if (legal_replay_after_last != 1 || legal_releases != 1
                || global_fault_attacks != 2 || post_fault_accepts != 0)
            $fatal(1,"M405R3 coverage mismatch replay=%0d release=%0d attacks=%0d post=%0d",
                   legal_replay_after_last,legal_releases,
                   global_fault_attacks,post_fault_accepts);
        $display("PASS M405R3 integration config_live_through_replay=1 legal_replay_after_last=1 legal_phase_release=1 global_fault_attacks=2 post_fault_accepts=0 accept_equations=true sticky_global_quiescence=true system_speedup=false headline=false");
        $finish;
    end

    initial begin
        #50000;
        $fatal(1,"M405R3 integration watchdog");
    end
endmodule

`default_nettype wire
