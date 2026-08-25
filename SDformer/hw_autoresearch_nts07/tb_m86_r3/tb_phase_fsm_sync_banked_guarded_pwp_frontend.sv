`timescale 1ns/1ps
`default_nettype none

module tb_phase_fsm_sync_banked_guarded_pwp_frontend;
    logic clk_core, rst_core;
    logic payload_load_valid, payload_load_ready, payload_load_accept;
    logic [9:0] payload_load_row;
    logic [255:0] payload_load_words;
    logic phase_load_valid, phase_load_ready, phase_load_accept;
    logic [591:0] phase_metadata;
    logic phase_loaded, metadata_error;
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
    logic payload_selected, phase_selected, descriptor_selected;
    logic [2:0] fsm_state;
    logic [8:0] accepted_rows, accepted_descriptors;
    integer payload_accepts, phase_accepts, descriptor_accepts;
    integer output_count, issue_count, response_count;
    integer expected_output_tag, bounded_loader_wait;
    integer metadata_cursor;

    phase_fsm_sync_banked_guarded_pwp_frontend dut (.*);
    phase_fsm_sync_banked_guarded_pwp_frontend_assertions m86_r3_sva (.*);

    always #1.5 clk_core = ~clk_core;
    initial begin
        #200000;
        $fatal(1, "M86-R3 watchdog timeout state=%0d rows=%0d descriptors=%0d outputs=%0d",
               fsm_state, accepted_rows, descriptor_accepts, output_count);
    end

    always @(posedge clk_core) begin
        if (!rst_core) begin
            if (payload_load_accept) payload_accepts++;
            if (phase_load_accept) phase_accepts++;
            if (descriptor_accept) descriptor_accepts++;
            if (bank_read_issue) issue_count++;
            if (bank_response_enqueue) response_count++;
            if (output_accept) begin
                if (output_tag != expected_output_tag || output_width != 8
                        || output_escape || output_values !== '0)
                    $fatal(1, "M86-R3 output mismatch got_tag=%0d expected_tag=%0d width=%0d escape=%0d",
                           output_tag, expected_output_tag,
                           output_width, output_escape);
                expected_output_tag++;
                output_count++;
            end
            if (protocol_error || metadata_error)
                $fatal(1, "M86-R3 unexpected legal-path fault state=%0d", fsm_state);
        end
    end

    task automatic drive_descriptor(input integer tag_value);
        begin
            @(negedge clk_core);
            descriptor_tag = tag_value;
            descriptor_pattern = tag_value >> 3;
            descriptor_block = tag_value & 7;
            descriptor_valid = 1'b1;
            do @(posedge clk_core); while (!descriptor_ready);
            #0.1 descriptor_valid = 1'b0;
        end
    endtask

    initial begin
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
        phase_accepts = 0;
        descriptor_accepts = 0;
        output_count = 0;
        issue_count = 0;
        response_count = 0;
        expected_output_tag = 0;
        bounded_loader_wait = 0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core); rst_core = 0;

        // Hold all three channels valid through the entire LOAD/COMMIT
        // boundary.  R2 deadlocked here after only row0; R3 must load every
        // row, then commit, then accept the early descriptor in state order.
        payload_load_words = '0;
        phase_metadata = '0;
        metadata_cursor = 0;
        for (int pattern = 0; pattern < 16; pattern++) begin
            phase_metadata[384+pattern*13 +: 13] = metadata_cursor;
            metadata_cursor += 8*24;
        end
        descriptor_pattern = 0;
        descriptor_block = 0;
        descriptor_tag = 0;
        for (int row = 0; row < 460; row++) begin
            @(negedge clk_core);
            payload_load_row = row;
            if (row == 0) begin
                payload_load_valid = 1;
                phase_load_valid = 1;
                descriptor_valid = 1;
            end
            do @(posedge clk_core); while (!payload_load_ready);
            #0.1;
            if (payload_accepts != row + 1 || phase_accepts != 0
                    || descriptor_accepts != 0)
                $fatal(1, "M86-R3 LOAD arbitration failed row=%0d", row);
        end
        do @(posedge clk_core); while (!phase_load_ready);
        #0.1;
        if (payload_accepts != 460 || phase_accepts != 1
                || descriptor_accepts != 0)
            $fatal(1, "M86-R3 COMMIT arbitration failed");
        do @(posedge clk_core); while (!descriptor_ready);
        #0.1;
        if (payload_accepts != 460 || phase_accepts != 1
                || descriptor_accepts != 1)
            $fatal(1, "M86-R3 EXECUTE arbitration failed");
        @(negedge clk_core);
        payload_load_valid = 0;
        phase_load_valid = 0;
        descriptor_valid = 0;

        for (int tag = 1; tag < 128; tag++) drive_descriptor(tag);
        if (descriptor_accepts != 128 || accepted_descriptors != 128
                || fsm_state != 3)
            $fatal(1, "M86-R3 phase bound failed accepts=%0d debug=%0d state=%0d",
                   descriptor_accepts, accepted_descriptors, fsm_state);

        // A queued next-phase loader cannot starve behind more descriptor or
        // commit traffic.  DRAIN advertises busy; LOAD then grants the row.
        @(negedge clk_core);
        payload_load_valid = 1;
        payload_load_row = 0;
        phase_load_valid = 1;
        descriptor_valid = 1;
        descriptor_tag = 32'hdeaddead;
        while (!payload_load_accept) begin
            @(posedge clk_core);
            bounded_loader_wait++;
            if (!busy && fsm_state != 0)
                $fatal(1, "M86-R3 silent post-phase state");
            if (bounded_loader_wait > 32)
                $fatal(1, "M86-R3 next loader starvation");
        end
        #0.1;
        payload_load_valid = 0;
        phase_load_valid = 0;
        descriptor_valid = 0;
        while (output_count != 128 || response_count != issue_count)
            @(posedge clk_core);
        @(posedge clk_core); #1;
        if (payload_accepts != 461 || phase_accepts != 1
                || descriptor_accepts != 128 || output_count != 128
                || issue_count != 384 || response_count != 384
                || bounded_loader_wait == 0 || protocol_error)
            $fatal(1, "M86-R3 conservation mismatch payload=%0d phase=%0d desc=%0d out=%0d issue=%0d response=%0d wait=%0d",
                   payload_accepts, phase_accepts, descriptor_accepts,
                   output_count, issue_count, response_count,
                   bounded_loader_wait);
        $display("PASS M86-R3 phase-fsm triple_contention=3 payload_accepts=461 phase_accepts=1 descriptor_accepts=128 outputs=128 bank_issues=384 bank_responses=384 bounded_loader_wait=%0d silent_deadlock=0", bounded_loader_wait);
        $finish;
    end
endmodule

`default_nettype wire
