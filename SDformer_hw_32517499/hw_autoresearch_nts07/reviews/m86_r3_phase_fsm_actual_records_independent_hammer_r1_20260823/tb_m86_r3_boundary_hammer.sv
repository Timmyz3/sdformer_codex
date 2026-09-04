`timescale 1ns/1ps
`default_nettype none

// Independent state-boundary, error, backpressure, and reset attack bench.
// Production SVA is intentionally not instantiated because OOB/duplicate
// cases deliberately enter sticky fault states.
module tb_m86_r3_boundary_hammer;
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

    integer payload_accepts, phase_accepts, descriptor_accepts, outputs;
    integer bank_issues, bank_responses, prev_issue;
    integer onehot_checks, exclusive_checks, early_commit_wait_cycles;
    integer late_commit_wait_cycles, drain_stall_cycles, held_loader_wait;
    integer fault_classes, reset_classes;

    phase_fsm_sync_banked_guarded_pwp_frontend dut (.*);

    always #1.5 clk_core = ~clk_core;
    initial begin
        #300000;
        $fatal(1, "M86-R3 independent boundary watchdog state=%0d rows=%0d desc=%0d",
               fsm_state, accepted_rows, accepted_descriptors);
    end

    task automatic build_metadata(input logic poison);
        integer cursor;
        begin
            phase_metadata = '0;
            cursor = 0;
            for (int pattern = 0; pattern < 16; pattern++) begin
                phase_metadata[384 + pattern*13 +: 13] = cursor[12:0];
                cursor = cursor + 8*24;
            end
            if (poison)
                phase_metadata[0 +: 3] = 3'd5;
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1;
            payload_load_valid = 0;
            phase_load_valid = 0;
            descriptor_valid = 0;
            output_ready = 1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 0;
            prev_issue = 0;
            #1;
            if (fsm_state != 0 || accepted_rows != 0
                    || accepted_descriptors != 0 || busy || protocol_error
                    || metadata_error || phase_loaded || output_valid)
                $fatal(1, "M86-R3 hammer reset did not return empty LOAD");
            reset_classes++;
        end
    endtask

    task automatic drive_row(input integer row_value);
        begin
            @(negedge clk_core);
            payload_load_row = row_value[9:0];
            payload_load_words = '0;
            payload_load_valid = 1;
            do @(posedge clk_core); while (!payload_load_ready);
            #0.1 payload_load_valid = 0;
        end
    endtask

    task automatic load_rows(input integer first, input integer last_exclusive);
        begin
            for (int row = first; row < last_exclusive; row++)
                drive_row(row);
        end
    endtask

    task automatic commit_metadata(input logic poison);
        begin
            build_metadata(poison);
            @(negedge clk_core);
            phase_load_valid = 1;
            do @(posedge clk_core); while (!phase_load_ready);
            #0.1 phase_load_valid = 0;
            #1;
        end
    endtask

    task automatic drive_descriptor(input integer tag_value);
        begin
            @(negedge clk_core);
            descriptor_pattern = tag_value[6:3];
            descriptor_block = tag_value[2:0];
            descriptor_tag = tag_value;
            descriptor_valid = 1;
            do @(posedge clk_core); while (!descriptor_ready);
            #0.1 descriptor_valid = 0;
        end
    endtask

    always @(posedge clk_core) begin
        if (rst_core) begin
            prev_issue = 0;
        end else begin
            if (!$onehot0({payload_selected, phase_selected,
                           descriptor_selected}))
                $fatal(1, "M86-R3 hammer non-onehot selection");
            onehot_checks++;
            if (!$onehot0({payload_load_accept, phase_load_accept,
                           descriptor_accept}))
                $fatal(1, "M86-R3 hammer nonexclusive accept");
            exclusive_checks++;
            if (bank_response_enqueue !== prev_issue[0])
                $fatal(1, "M86-R3 hammer response != prior issue");
            prev_issue = bank_read_issue;
            if (payload_load_accept) payload_accepts++;
            if (phase_load_accept) phase_accepts++;
            if (descriptor_accept) descriptor_accepts++;
            if (bank_read_issue) bank_issues++;
            if (bank_response_enqueue) bank_responses++;
            if (output_accept) begin
                if (output_width != 8 || output_escape
                        || output_values !== '0)
                    $fatal(1, "M86-R3 hammer nonzero/legal output mismatch");
                outputs++;
            end
        end
    end

    initial begin
        integer p0, c0, d0, o0, i0, r0;
        logic [31:0] held_tag;
        logic [1151:0] held_values;
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
        outputs = 0;
        bank_issues = 0;
        bank_responses = 0;
        prev_issue = 0;
        onehot_checks = 0;
        exclusive_checks = 0;
        early_commit_wait_cycles = 0;
        late_commit_wait_cycles = 0;
        drain_stall_cycles = 0;
        held_loader_wait = 0;
        fault_classes = 0;
        reset_classes = 0;
        repeat (4) @(posedge clk_core);
        @(negedge clk_core); rst_core = 0;
        build_metadata(0);

        // Early commit/descriptor without payload is blocked in empty LOAD.
        @(negedge clk_core);
        phase_load_valid = 1;
        descriptor_valid = 1;
        repeat (3) begin
            @(posedge clk_core); #1;
            if (payload_selected || phase_selected || descriptor_selected
                    || payload_load_ready || phase_load_ready
                    || descriptor_ready || payload_load_accept
                    || phase_load_accept || descriptor_accept || busy
                    || protocol_error)
                $fatal(1, "M86-R3 hammer early request behavior drift");
            early_commit_wait_cycles++;
        end

        // Hold all three valids across the exact 459/460/461-row boundary.
        payload_load_valid = 1;
        payload_load_words = '0;
        for (int row = 0; row < 459; row++) begin
            @(negedge clk_core); payload_load_row = row;
            @(posedge clk_core); #1;
            if (!payload_load_accept || phase_load_accept
                    || descriptor_accept || fsm_state != 0
                    || accepted_rows != row + 1)
                $fatal(1, "M86-R3 hammer 459-row boundary failed row=%0d", row);
        end
        if (accepted_rows != 459 || fsm_state != 0
                || phase_accepts != 0 || descriptor_accepts != 0)
            $fatal(1, "M86-R3 hammer phase advanced at 459 rows");

        @(negedge clk_core); payload_load_row = 459;
        @(posedge clk_core); #1;
        if (payload_accepts != 460 || accepted_rows != 460 || fsm_state != 1
                || phase_accepts != 0 || descriptor_accepts != 0)
            $fatal(1, "M86-R3 hammer 460th row did not enter COMMIT");

        // Present a would-be 461st/OOB row while early phase and descriptor
        // remain held.  COMMIT must accept only metadata and mask that row.
        @(negedge clk_core); payload_load_row = 10'd460;
        @(posedge clk_core); #1;
        if (payload_accepts != 460 || phase_accepts != 1
                || descriptor_accepts != 0 || fsm_state != 2
                || accepted_rows != 0 || accepted_descriptors != 0
                || protocol_error)
            $fatal(1, "M86-R3 hammer 461st-row/commit arbitration failed");

        @(posedge clk_core); #1;
        if (payload_accepts != 460 || phase_accepts != 1
                || descriptor_accepts != 1
                || accepted_descriptors != 1 || fsm_state != 2)
            $fatal(1, "M86-R3 hammer EXECUTE triple priority failed");
        @(negedge clk_core);
        payload_load_valid = 0;
        phase_load_valid = 0;
        descriptor_valid = 0;

        // Reach exactly 127 descriptors and retire their outputs.
        for (int tag = 1; tag < 127; tag++)
            drive_descriptor(tag);
        while (outputs < 127 || bank_responses != bank_issues)
            @(posedge clk_core);
        #1;
        if (accepted_descriptors != 127 || fsm_state != 2
                || descriptor_accepts != 127)
            $fatal(1, "M86-R3 hammer 127-descriptor boundary failed");

        // Stall only the 128th output in DRAIN.  A held 129th descriptor,
        // phase request, and next loader must all remain blocked until drain.
        output_ready = 0;
        drive_descriptor(127);
        if (accepted_descriptors != 128 || fsm_state != 3
                || descriptor_accepts != 128)
            $fatal(1, "M86-R3 hammer 128th descriptor did not enter DRAIN");
        p0 = payload_accepts;
        c0 = phase_accepts;
        d0 = descriptor_accepts;
        @(negedge clk_core);
        payload_load_valid = 1;
        payload_load_row = 0;
        phase_load_valid = 1;
        descriptor_valid = 1;
        descriptor_tag = 32'h129;
        wait (output_valid);
        held_tag = output_tag;
        held_values = output_values;
        repeat (6) begin
            @(posedge clk_core); #1;
            if (fsm_state != 3 || !busy || payload_load_ready
                    || phase_load_ready || descriptor_ready
                    || payload_load_accept || phase_load_accept
                    || descriptor_accept || output_tag != held_tag
                    || output_values != held_values)
                $fatal(1, "M86-R3 hammer DRAIN/backpressure instability");
            drain_stall_cycles++;
        end
        @(negedge clk_core); output_ready = 1;
        while (payload_accepts == p0) begin
            @(posedge clk_core);
            held_loader_wait++;
            if (held_loader_wait > 20)
                $fatal(1, "M86-R3 hammer held loader starved after drain");
        end
        #0.1;
        payload_load_valid = 0;
        phase_load_valid = 0;
        descriptor_valid = 0;
        if (payload_accepts != p0 + 1 || phase_accepts != c0
                || descriptor_accepts != d0 || outputs != 128
                || fsm_state != 0 || accepted_rows != 1
                || protocol_error)
            $fatal(1, "M86-R3 hammer drain-to-load conservation failed");

        // Reset a partially loaded next phase and require no stale state.
        reset_dut();

        // OOB row in LOAD is not accepted; R1 fault must propagate to FAULT.
        p0 = payload_accepts;
        @(negedge clk_core);
        payload_load_valid = 1;
        payload_load_row = 10'd460;
        phase_load_valid = 1;
        descriptor_valid = 1;
        repeat (2) @(posedge clk_core);
        #1;
        if (payload_accepts != p0 || !protocol_error || !busy
                || fsm_state != 4 || accepted_rows != 0)
            $fatal(1, "M86-R3 hammer OOB did not enter FAULT");
        fault_classes++;
        @(negedge clk_core);
        payload_load_valid = 0;
        phase_load_valid = 0;
        descriptor_valid = 0;
        reset_dut();

        // Duplicate row is accepted once more at the R1 handshake, but both
        // the wrapper bitmap and R1 sticky checker must enter FAULT.
        drive_row(0);
        p0 = payload_accepts;
        @(negedge clk_core);
        payload_load_valid = 1;
        payload_load_row = 0;
        @(posedge clk_core); #1;
        if (payload_accepts != p0 + 1 || !protocol_error
                || fsm_state != 4 || !busy)
            $fatal(1, "M86-R3 hammer duplicate did not enter FAULT");
        fault_classes++;
        @(negedge clk_core); payload_load_valid = 0;
        reset_dut();

        // Late metadata: after 460 rows COMMIT is busy and blocks payload and
        // descriptor until phase_valid arrives.  Poison then propagates.
        load_rows(0, 460);
        if (fsm_state != 1 || accepted_rows != 460 || !busy)
            $fatal(1, "M86-R3 hammer did not reach late COMMIT");
        p0 = payload_accepts;
        c0 = phase_accepts;
        d0 = descriptor_accepts;
        @(negedge clk_core);
        payload_load_valid = 1;
        payload_load_row = 0;
        descriptor_valid = 1;
        repeat (3) begin
            @(posedge clk_core); #1;
            if (fsm_state != 1 || !busy || payload_load_accept
                    || phase_load_accept || descriptor_accept
                    || payload_accepts != p0 || descriptor_accepts != d0
                    || protocol_error)
                $fatal(1, "M86-R3 hammer late COMMIT wait failed");
            late_commit_wait_cycles++;
        end
        build_metadata(1);
        phase_load_valid = 1;
        @(posedge clk_core); #1;
        if (phase_accepts != c0 + 1 || !metadata_error || !protocol_error)
            $fatal(1, "M86-R3 hammer poison commit did not propagate");
        @(posedge clk_core); #1;
        if (fsm_state != 4 || !busy || descriptor_accept)
            $fatal(1, "M86-R3 hammer poison did not settle in FAULT");
        fault_classes++;
        @(negedge clk_core);
        payload_load_valid = 0;
        phase_load_valid = 0;
        descriptor_valid = 0;
        reset_dut();

        // Build another legal phase and reset specifically while the 128th
        // output is held in DRAIN; no output/response/state may survive reset.
        load_rows(0, 460);
        commit_metadata(0);
        if (fsm_state != 2 || protocol_error)
            $fatal(1, "M86-R3 hammer reset phase commit failed");
        d0 = descriptor_accepts;
        o0 = outputs;
        i0 = bank_issues;
        r0 = bank_responses;
        descriptor_valid = 1;
        descriptor_pattern = 0;
        descriptor_block = 0;
        descriptor_tag = 32'h8603;
        do @(posedge clk_core); while (descriptor_accepts < d0 + 128);
        @(negedge clk_core);
        descriptor_valid = 0;
        output_ready = 0;
        wait (fsm_state == 3 && output_valid);
        if (accepted_descriptors != 128 || !busy)
            $fatal(1, "M86-R3 hammer did not reach reset-in-DRAIN point");
        reset_dut();
        repeat (3) begin
            @(posedge clk_core); #1;
            if (output_valid || bank_response_enqueue || busy
                    || fsm_state != 0 || accepted_rows != 0
                    || accepted_descriptors != 0 || protocol_error)
                $fatal(1, "M86-R3 hammer ghost state after DRAIN reset");
        end
        if (descriptor_accepts != d0 + 128
                || bank_issues < i0 || bank_responses < r0)
            $fatal(1, "M86-R3 hammer reset counters regressed");

        if (early_commit_wait_cycles != 3
                || late_commit_wait_cycles != 3
                || drain_stall_cycles != 6 || held_loader_wait == 0
                || fault_classes != 3 || reset_classes != 5
                || onehot_checks == 0 || exclusive_checks == 0
                || bank_responses > bank_issues)
            $fatal(1, "M86-R3 hammer coverage mismatch early=%0d late=%0d drain=%0d held=%0d faults=%0d resets=%0d onehot=%0d exclusive=%0d issue=%0d response=%0d",
                   early_commit_wait_cycles, late_commit_wait_cycles,
                   drain_stall_cycles, held_loader_wait, fault_classes,
                   reset_classes, onehot_checks, exclusive_checks,
                   bank_issues, bank_responses);
        $display("PASS M86-R3 independent boundary triple_states=3 rows_459_460_461=3 descriptors_127_128_129=3 early_commit_wait=3 late_commit_wait=3 drain_stall=6 held_loader_wait=%0d fault_classes=3 reset_classes=5 repeated_descriptor_accepts=128 onehot_checks=%0d issue=%0d response=%0d",
                 held_loader_wait, onehot_checks, bank_issues,
                 bank_responses);
        $finish;
    end
endmodule

`default_nettype wire
