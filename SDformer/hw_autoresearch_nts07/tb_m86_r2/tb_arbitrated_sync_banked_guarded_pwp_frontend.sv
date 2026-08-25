`timescale 1ns/1ps
`default_nettype none

module tb_arbitrated_sync_banked_guarded_pwp_frontend;
    logic clk_core, rst_core;
    logic payload_load_valid, payload_load_ready, payload_load_accept;
    logic [9:0] payload_load_row;
    logic [255:0] payload_load_words;
    logic phase_load_valid, phase_load_ready, phase_loaded, metadata_error;
    logic [591:0] phase_metadata;
    logic descriptor_valid, descriptor_ready, descriptor_accept;
    logic [3:0] descriptor_pattern;
    logic [2:0] descriptor_block;
    logic [31:0] descriptor_tag;
    logic output_valid, output_ready, output_escape, output_accept;
    logic [31:0] output_tag;
    logic [3:0] output_width;
    logic [1151:0] output_values;
    logic protocol_error, busy, bank_read_issue, bank_response_enqueue;
    logic [2:0] bank_read_beat, response_fifo_level;
    logic payload_selected, descriptor_selected;
    integer payload_accepts, descriptor_accepts, outputs;
    integer bank_issues, bank_responses;

    arbitrated_sync_banked_guarded_pwp_frontend dut (.*);
    arbitrated_sync_banked_guarded_pwp_frontend_assertions m86_r2_sva (.*);

    always #1.5 clk_core = ~clk_core;
    initial begin
        #50000;
        $fatal(1, "M86-R2 watchdog timeout");
    end

    always @(posedge clk_core) begin
        if (!rst_core) begin
            if (payload_load_accept) payload_accepts++;
            if (descriptor_accept) descriptor_accepts++;
            if (bank_read_issue) bank_issues++;
            if (bank_response_enqueue) bank_responses++;
            if (output_accept) begin
                if (output_tag != 32'h8602 || output_width != 8
                        || output_escape || output_values != '0)
                    $fatal(1, "M86-R2 output mismatch");
                outputs++;
            end
        end
    end

    task automatic load_row(input integer row_value);
        begin
            @(negedge clk_core);
            payload_load_valid = 1;
            payload_load_row = row_value;
            payload_load_words = '0;
            do @(posedge clk_core); while (!payload_load_ready);
            #0.1 payload_load_valid = 0;
        end
    endtask

    initial begin
        integer cursor;
        clk_core = 0;
        rst_core = 1;
        payload_load_valid = 0;
        payload_load_row = 0;
        payload_load_words = 0;
        phase_load_valid = 0;
        phase_metadata = 0;
        descriptor_valid = 0;
        descriptor_pattern = 0;
        descriptor_block = 0;
        descriptor_tag = 0;
        output_ready = 1;
        payload_accepts = 0;
        descriptor_accepts = 0;
        outputs = 0;
        bank_issues = 0;
        bank_responses = 0;
        repeat (4) @(posedge clk_core);
        @(negedge clk_core); rst_core = 0;

        for (int row = 0; row < 460; row++) load_row(row);
        phase_metadata = '0;
        cursor = 0;
        for (int pattern = 0; pattern < 16; pattern++) begin
            phase_metadata[384+pattern*13 +: 13] = cursor;
            cursor += 8*24;
        end
        @(negedge clk_core); phase_load_valid = 1;
        do @(posedge clk_core); while (!phase_load_ready);
        #0.1 phase_load_valid = 0;
        #0.9;
        if (!phase_loaded || metadata_error || protocol_error)
            $fatal(1, "M86-R2 legal phase did not load");

        // The independent review's exact deadlock trigger.  With a committed
        // phase, descriptor traffic has explicit priority and must progress.
        @(negedge clk_core);
        payload_load_valid = 1;
        payload_load_row = 0;
        descriptor_valid = 1;
        descriptor_pattern = 0;
        descriptor_block = 0;
        descriptor_tag = 32'h8602;
        @(posedge clk_core); #1;
        if (descriptor_accepts != 1 || payload_accepts != 460
                || protocol_error)
            $fatal(1, "M86-R2 loaded contention priority failed");
        descriptor_valid = 0;
        payload_load_valid = 0;
        while (busy || outputs != 1) @(posedge clk_core);

        // A lone loader request starts the next phase image and clears commit.
        load_row(0);
        #1;
        if (phase_loaded || protocol_error)
            $fatal(1, "M86-R2 next-phase loader did not clear commit");

        // Before commit, loader traffic has priority over an early descriptor.
        @(negedge clk_core);
        payload_load_valid = 1;
        payload_load_row = 1;
        descriptor_valid = 1;
        descriptor_tag = 32'hdead_beef;
        @(posedge clk_core); #1;
        if (payload_accepts != 462 || descriptor_accepts != 1
                || protocol_error)
            $fatal(1, "M86-R2 unloaded contention priority failed");
        payload_load_valid = 0;
        descriptor_valid = 0;
        @(posedge clk_core); #1;

        if (outputs != 1 || bank_issues != 3 || bank_responses != 3)
            $fatal(1, "M86-R2 conservation mismatch out=%0d issue=%0d response=%0d",
                   outputs, bank_issues, bank_responses);
        $display("PASS M86-R2 arbitration loaded_descriptor_wins=1 unloaded_loader_wins=1 payload_accepts=462 descriptor_accepts=1 outputs=1 bank_issues=3 bank_responses=3 silent_deadlock=0");
        $finish;
    end
endmodule

`default_nettype wire
