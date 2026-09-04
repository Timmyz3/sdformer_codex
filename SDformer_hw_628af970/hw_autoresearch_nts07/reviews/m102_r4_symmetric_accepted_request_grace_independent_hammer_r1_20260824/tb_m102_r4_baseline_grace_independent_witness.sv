`timescale 1ns/1ps
`default_nettype none

module tb_m102_r4_baseline_grace_independent_witness;
    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    logic lookup_valid = 1'b0;
    logic lookup_ready;
    logic [3:0] lookup_source = '0;
    logic [2:0] lookup_block = '0;
    logic [1:0] lookup_beat = '0;
    logic [31:0] lookup_tag = '0;
    logic [255:0] bank_words = {8{32'h807f_01ff}};
    logic [79:0] bank_row_addresses;
    logic output_valid;
    logic output_ready = 1'b0;
    logic [31:0] output_tag;
    logic [3:0] output_width;
    logic output_escape;
    logic [1151:0] output_values;
    logic output_accept;
    logic protocol_error;
    logic busy;

    logic grace_hold_event = 1'b0;
    logic glitch_return_event = 1'b0;
    logic mutation_event = 1'b0;
    logic phase_reload_event = 1'b0;
    logic reset_recovery_event = 1'b0;
    integer input_accept_count = 0;
    integer output_accept_count = 0;
    integer mutation_count = 0;
    integer reset_recovery_count = 0;

    wire request_valid = lookup_valid;
    wire request_ready = lookup_ready;
    wire accepted_grace_match = dut.accepted_grace_match;
    wire request_semantically_valid = dut.request_semantically_valid;
    wire request_violation = dut.request_violation;
    wire request_fault = dut.request_fault_q;
    wire m82_beat_accept = dut.m82_beat_accept;
    wire m82_output_valid = dut.m82_output_valid;
    wire m82_output_accept = dut.m82_stream.output_accept;
    wire phase_load_ready = 1'b0;

    m102_bit_sparse_weight_stream dut (.*);
    m102_r4_independent_grace_assertions witness_sva (.*);

    always #1.5 clk_core = ~clk_core;
    always @(posedge clk_core) begin
        if (dut.m82_beat_accept)
            input_accept_count = input_accept_count + 1;
        if (output_accept)
            output_accept_count = output_accept_count + 1;
    end

    task automatic clear_events;
        begin
            grace_hold_event = 1'b0;
            glitch_return_event = 1'b0;
            mutation_event = 1'b0;
            phase_reload_event = 1'b0;
            reset_recovery_event = 1'b0;
        end
    endtask

    task automatic reset_clean;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            lookup_valid = 1'b0;
            output_ready = 1'b0;
            clear_events();
            repeat (2) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
        end
    endtask

    task automatic drive_three_beats(input logic [31:0] tag_value);
        begin
            for (int beat = 0; beat < 3; beat++) begin
                @(negedge clk_core);
                lookup_valid = 1'b1;
                lookup_source = 4'd2;
                lookup_block = 3'd3;
                lookup_beat = beat[1:0];
                lookup_tag = tag_value;
                do @(posedge clk_core); while (!lookup_ready);
            end
        end
    endtask

    task automatic reset_after_fault;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            lookup_valid = 1'b0;
            output_ready = 1'b1;
            clear_events();
            @(posedge clk_core);
            #0.1;
            if (protocol_error || dut.request_fault_q
                    || dut.m82_output_valid || output_valid || output_accept)
                $fatal(1, "baseline reset edge failed to clear sticky state");
            @(negedge clk_core);
            rst_core = 1'b0;
            reset_recovery_event = 1'b1;
            @(posedge clk_core);
            #0.1;
            if (protocol_error || dut.request_fault_q
                    || dut.m82_output_valid || output_valid || output_accept)
                $fatal(1, "baseline post-reset recovery failed");
            reset_recovery_count = reset_recovery_count + 1;
            @(negedge clk_core);
            reset_recovery_event = 1'b0;
        end
    endtask

    task automatic attack_identity_field(input integer field_id);
        integer accepts_before;
        begin
            reset_clean();
            accepts_before = input_accept_count;
            output_ready = 1'b0;
            drive_three_beats(32'hb400_1000 + field_id);
            #0.1;
            if (!dut.accepted_grace_match || protocol_error
                    || !dut.m82_output_valid || !output_valid
                    || input_accept_count != accepts_before + 3)
                $fatal(1, "baseline mutation%0d setup failed", field_id);

            @(negedge clk_core);
            case (field_id)
                0: lookup_source = lookup_source + 1'b1;
                1: lookup_block = lookup_block + 1'b1;
                2: lookup_beat = 2'd1;
                3: lookup_tag = lookup_tag + 1'b1;
                default: $fatal(1, "unknown baseline mutation field=%0d", field_id);
            endcase
            mutation_event = 1'b1;
            output_ready = 1'b1;
            #0.1;
            if (!dut.request_violation || !protocol_error || lookup_ready
                    || output_valid || output_accept
                    || !dut.m82_output_valid
                    || dut.m82_stream.output_accept)
                $fatal(1, "baseline mutation%0d pre-edge quarantine failed", field_id);
            @(posedge clk_core);
            #0.1;
            if (!dut.request_fault_q || !protocol_error
                    || output_valid || output_accept
                    || !dut.m82_output_valid
                    || input_accept_count != accepts_before + 3)
                $fatal(1, "baseline mutation%0d fault edge failed", field_id);
            @(negedge clk_core);
            mutation_event = 1'b0;
            lookup_valid = 1'b0;
            repeat (2) begin
                @(posedge clk_core);
                #0.1;
                if (!dut.request_fault_q || !protocol_error
                        || output_valid || output_accept
                        || !dut.m82_output_valid)
                    $fatal(1, "baseline mutation%0d fault not sticky", field_id);
            end
            mutation_count = mutation_count + 1;
            reset_after_fault();
        end
    endtask

    initial begin
        integer accepts_before;
        integer outputs_before;

        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        // Exact final request is kept asserted across a complete additional
        // active edge.  It is grace, not a fourth beat, and its result remains
        // externally visible while output_ready is low.
        accepts_before = input_accept_count;
        outputs_before = output_accept_count;
        output_ready = 1'b0;
        drive_three_beats(32'hb400_0001);
        #0.1;
        if (!dut.accepted_grace_match || protocol_error
                || !output_valid || output_tag != 32'hb400_0001
                || output_width != 8 || output_escape
                || input_accept_count != accepts_before + 3)
            $fatal(1, "baseline exact-grace setup failed");
        @(negedge clk_core);
        grace_hold_event = 1'b1;
        @(posedge clk_core);
        #0.1;
        if (protocol_error || lookup_ready || dut.m82_beat_accept
                || !output_valid || output_tag != 32'hb400_0001
                || input_accept_count != accepts_before + 3)
            $fatal(1, "baseline full-cycle grace caused fault/double accept");
        @(negedge clk_core);
        grace_hold_event = 1'b0;

        // No active sampling edge observes this low pulse.  Returning high
        // with the exact identity must retain the existing grace token.
        #0.2 lookup_valid = 1'b0;
        #0.4 lookup_valid = 1'b1;
        #0.2 glitch_return_event = 1'b1;
        if (!dut.accepted_grace_match || protocol_error)
            $fatal(1, "baseline between-edge low/high was misclassified");
        @(posedge clk_core);
        #0.1;
        if (!dut.accepted_grace_match || protocol_error
                || dut.m82_beat_accept
                || input_accept_count != accepts_before + 3)
            $fatal(1, "baseline glitch return caused fault/double accept");
        @(negedge clk_core);
        glitch_return_event = 1'b0;

        // The held request remains grace while the old result is consumed.
        output_ready = 1'b1;
        #0.1;
        if (!output_valid || !output_accept
                || output_tag != 32'hb400_0001 || protocol_error)
            $fatal(1, "baseline grace result was not visible/acceptable");
        @(posedge clk_core);
        #0.1;
        if (output_valid || protocol_error || dut.m82_beat_accept
                || input_accept_count != accepts_before + 3
                || output_accept_count != outputs_before + 1)
            $fatal(1, "baseline result retirement/double-accept mismatch");
        @(negedge clk_core);
        lookup_valid = 1'b0;
        @(posedge clk_core);
        #0.1;
        if (dut.accepted_grace_q)
            $fatal(1, "baseline grace did not clear on sampled valid-low");

        // Every baseline request-identity field is changed independently.
        for (int field_id = 0; field_id < 4; field_id++)
            attack_identity_field(field_id);

        if (mutation_count != 4 || reset_recovery_count != 4
                || protocol_error)
            $fatal(1, "baseline witness counters mutations=%0d resets=%0d fault=%0d",
                   mutation_count, reset_recovery_count, protocol_error);
        $display("PASS M102_R4_INDEPENDENT_BASELINE grace_full_edges=1 glitch_low_high=1 identity_mutations=4 sticky_checks=8 reset_recoveries=4 no_double_accept=1 result_visible=1");
        $finish;
    end

    initial begin
        #200000;
        $fatal(1, "M102 r4 baseline independent witness watchdog");
    end
endmodule

`default_nettype wire
