`timescale 1ns/1ps
`default_nettype none

// Independent arbitration/phase/error attack bench.  Production SVA is not
// instantiated because two tests deliberately enter sticky protocol-error
// states that the production contention properties do not exempt.
module tb_m86_r2_hammer;
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
    integer bank_issues, bank_responses, prev_issue;
    integer phase_payload_deadlock_cycles, loaded_payload_denied_cycles;
    integer onehot_checks, exclusive_accept_checks, error_propagation_checks;

    arbitrated_sync_banked_guarded_pwp_frontend dut (.*);

    always #1.5 clk_core = ~clk_core;
    initial begin
        #200000;
        $fatal(1, "M86-R2 independent hammer watchdog");
    end

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

    task automatic build_legal_metadata;
        integer cursor;
        begin
            phase_metadata = '0;
            cursor = 0;
            for (int pattern = 0; pattern < 16; pattern++) begin
                phase_metadata[384 + pattern*13 +: 13] = cursor[12:0];
                cursor = cursor + 8*24;
            end
        end
    endtask

    task automatic commit_phase(input logic poison);
        begin
            build_legal_metadata();
            if (poison)
                phase_metadata[0 +: 3] = 3'd5;
            @(negedge clk_core);
            phase_load_valid = 1;
            do @(posedge clk_core); while (!phase_load_ready);
            #0.1 phase_load_valid = 0;
            #1;
        end
    endtask

    always @(posedge clk_core) begin
        if (rst_core) begin
            prev_issue = 0;
        end else begin
            if (payload_selected && descriptor_selected)
                $fatal(1, "M86-R2 hammer non-onehot selection");
            onehot_checks++;
            if (payload_load_accept && descriptor_accept)
                $fatal(1, "M86-R2 hammer dual accept");
            exclusive_accept_checks++;
            if (bank_response_enqueue !== prev_issue[0])
                $fatal(1, "M86-R2 hammer response latency/conservation mismatch");
            prev_issue = bank_read_issue;
            if (payload_load_accept) payload_accepts++;
            if (descriptor_accept) descriptor_accepts++;
            if (bank_read_issue) bank_issues++;
            if (bank_response_enqueue) bank_responses++;
            if (phase_loaded && payload_load_valid && descriptor_valid
                    && !payload_load_accept)
                loaded_payload_denied_cycles++;
            if (output_accept) begin
                if (output_tag != 32'h8602 || output_width != 8
                        || output_escape || output_values != '0)
                    $fatal(1, "M86-R2 hammer output mismatch");
                outputs++;
            end
        end
    end

    initial begin
        integer payload_before;
        integer descriptor_before;
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
        descriptor_tag = 32'h8602;
        output_ready = 1;
        payload_accepts = 0;
        descriptor_accepts = 0;
        outputs = 0;
        bank_issues = 0;
        bank_responses = 0;
        prev_issue = 0;
        phase_payload_deadlock_cycles = 0;
        loaded_payload_denied_cycles = 0;
        onehot_checks = 0;
        exclusive_accept_checks = 0;
        error_propagation_checks = 0;
        repeat (4) @(posedge clk_core);
        @(negedge clk_core); rst_core = 0;

        // Exact unloaded half of the R1 trigger: loader must win immediately.
        @(negedge clk_core);
        payload_load_valid = 1;
        payload_load_row = 0;
        descriptor_valid = 1;
        @(posedge clk_core); #1;
        if (!payload_load_accept || descriptor_accept
                || !payload_selected || descriptor_selected
                || protocol_error || busy)
            $fatal(1, "M86-R2 hammer unloaded contention repair failed");
        @(negedge clk_core);
        payload_load_valid = 0;
        descriptor_valid = 0;

        // The third ready/valid channel is not arbitrated.  An early phase
        // commit request and the next payload row silently block each other;
        // include an early descriptor to exercise all three inputs.
        @(negedge clk_core);
        payload_load_valid = 1;
        payload_load_row = 1;
        phase_load_valid = 1;
        descriptor_valid = 1;
        repeat (4) begin
            @(posedge clk_core); #1;
            if (!payload_selected || descriptor_selected
                    || payload_load_ready || descriptor_ready
                    || phase_load_ready || payload_load_accept
                    || descriptor_accept || protocol_error || busy)
                $fatal(1, "M86-R2 hammer three-channel deadlock behavior drift");
            phase_payload_deadlock_cycles++;
        end
        // Releasing only phase_valid must allow the selected payload to move.
        @(negedge clk_core); phase_load_valid = 0;
        @(posedge clk_core); #1;
        if (!payload_load_accept || descriptor_accept || protocol_error)
            $fatal(1, "M86-R2 hammer phase/payload deadlock did not recover");
        @(negedge clk_core);
        payload_load_valid = 0;
        descriptor_valid = 0;

        load_rows(2, 460);
        commit_phase(0);
        if (!phase_loaded || metadata_error || protocol_error)
            $fatal(1, "M86-R2 hammer legal first phase failed");

        // Exact loaded half of the R1 trigger plus bounded starvation.  Keep
        // the losing payload valid while eight descriptors traverse busy,
        // last-beat overlap, and output retirement.  It may move only after
        // descriptor_valid drops and the pipeline drains.
        payload_before = payload_accepts;
        descriptor_before = descriptor_accepts;
        @(negedge clk_core);
        payload_load_valid = 1;
        payload_load_row = 0;
        descriptor_valid = 1;
        descriptor_pattern = 0;
        descriptor_block = 0;
        descriptor_tag = 32'h8602;
        do @(posedge clk_core); while (descriptor_accepts < descriptor_before + 8);
        #1;
        if (payload_accepts != payload_before || protocol_error)
            $fatal(1, "M86-R2 hammer loaded loser moved or faulted");
        @(negedge clk_core); descriptor_valid = 0;
        do @(posedge clk_core); while (payload_accepts == payload_before);
        #1;
        if (payload_accepts != payload_before + 1 || phase_loaded
                || protocol_error)
            $fatal(1, "M86-R2 hammer bounded starvation did not recover");
        @(negedge clk_core); payload_load_valid = 0;
        while (busy || outputs < 8) @(posedge clk_core);
        if (descriptor_accepts != descriptor_before + 8
                || bank_issues != 24 || bank_responses != 24)
            $fatal(1, "M86-R2 hammer loaded conservation mismatch");

        // Complete the next image (row zero was just accepted) and commit.
        load_rows(1, 460);
        commit_phase(0);
        if (!phase_loaded || protocol_error)
            $fatal(1, "M86-R2 hammer legal second phase failed");

        // Loaded priority masks an invalid loader request while a descriptor
        // wins.  Once descriptor_valid drops, the OOB request propagates into
        // R1 even while the accepted descriptor is busy, producing a sticky
        // protocol fault and fail-stop busy state.
        descriptor_before = descriptor_accepts;
        @(negedge clk_core);
        payload_load_valid = 1;
        payload_load_row = 10'd460;
        descriptor_valid = 1;
        @(posedge clk_core); #1;
        if (descriptor_accepts != descriptor_before + 1
                || payload_load_accept || protocol_error
                || !descriptor_selected || payload_selected)
            $fatal(1, "M86-R2 hammer loaded error masking drift");
        @(negedge clk_core); descriptor_valid = 0;
        @(posedge clk_core); #1;
        if (!protocol_error || payload_load_accept || !busy)
            $fatal(1, "M86-R2 hammer delayed OOB error did not propagate fail-stop");
        error_propagation_checks++;
        // Allow the one in-flight bank response to arrive before reset.
        repeat (2) @(posedge clk_core);
        @(negedge clk_core); payload_load_valid = 0;

        // In the unloaded state, the same OOB request wins immediately over a
        // descriptor and propagates a sticky error without an acceptance.
        reset_dut();
        payload_before = payload_accepts;
        descriptor_before = descriptor_accepts;
        @(negedge clk_core);
        payload_load_valid = 1;
        payload_load_row = 10'd460;
        descriptor_valid = 1;
        @(posedge clk_core); #1;
        if (!protocol_error || payload_load_accept || descriptor_accept
                || !payload_selected || descriptor_selected
                || busy || payload_accepts != payload_before
                || descriptor_accepts != descriptor_before)
            $fatal(1, "M86-R2 hammer unloaded OOB propagation failed");
        error_propagation_checks++;
        @(negedge clk_core);
        payload_load_valid = 0;
        descriptor_valid = 0;

        // Duplicate-row attack under unloaded contention must be accepted by
        // the selected loader, detected by R1, and propagated outward.
        reset_dut();
        drive_row(0);
        payload_before = payload_accepts;
        @(negedge clk_core);
        payload_load_valid = 1;
        payload_load_row = 0;
        descriptor_valid = 1;
        @(posedge clk_core); #1;
        if (!payload_load_accept || descriptor_accept || !protocol_error
                || payload_accepts != payload_before + 1)
            $fatal(1, "M86-R2 hammer duplicate propagation failed");
        error_propagation_checks++;
        @(negedge clk_core);
        payload_load_valid = 0;
        descriptor_valid = 0;

        // Poisoned metadata remains visible and blocks both request accepts.
        reset_dut();
        load_rows(0, 460);
        commit_phase(1);
        if (!phase_loaded || !metadata_error || !protocol_error)
            $fatal(1, "M86-R2 hammer metadata error did not propagate");
        payload_before = payload_accepts;
        descriptor_before = descriptor_accepts;
        @(negedge clk_core);
        payload_load_valid = 1;
        payload_load_row = 0;
        descriptor_valid = 1;
        @(posedge clk_core); #1;
        if (!descriptor_selected || payload_selected
                || payload_load_accept || descriptor_accept
                || payload_accepts != payload_before
                || descriptor_accepts != descriptor_before
                || !metadata_error || !protocol_error)
            $fatal(1, "M86-R2 hammer poisoned contention did not fail closed");
        error_propagation_checks++;
        @(negedge clk_core);
        payload_load_valid = 0;
        descriptor_valid = 0;

        if (phase_payload_deadlock_cycles != 4
                || loaded_payload_denied_cycles < 8
                || error_propagation_checks != 4
                || onehot_checks == 0 || exclusive_accept_checks == 0
                || bank_issues != bank_responses || outputs != 8)
            $fatal(1, "M86-R2 hammer coverage mismatch phase_dead=%0d denied=%0d errors=%0d onehot=%0d exclusive=%0d issue=%0d response=%0d outputs=%0d",
                   phase_payload_deadlock_cycles,
                   loaded_payload_denied_cycles,
                   error_propagation_checks, onehot_checks,
                   exclusive_accept_checks, bank_issues,
                   bank_responses, outputs);
        $display("PASS M86-R2 independent hammer exact_r1_trigger_closed=2 phase_payload_silent_deadlock_cycles=4 bounded_descriptor_priority_accepts=8 losing_payload_recovery=1 error_propagation_classes=4 onehot_checks=%0d bank_issues=%0d bank_responses=%0d outputs=8",
                 onehot_checks, bank_issues, bank_responses);
        $finish;
    end
endmodule

`default_nettype wire
