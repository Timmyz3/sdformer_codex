`timescale 1ns/1ps
`default_nettype none

module tb_m102_r4_candidate_grace_independent_witness;
    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    logic phase_load_valid = 1'b0;
    logic phase_load_ready;
    logic [591:0] phase_metadata = '0;
    logic phase_loaded;
    logic metadata_error;
    logic service_valid = 1'b0;
    logic service_ready;
    logic [1:0] service_kind = '0;
    logic [3:0] service_pattern = '0;
    logic [3:0] service_source = '0;
    logic [2:0] service_block = '0;
    logic [2:0] service_beat = '0;
    logic service_negate = 1'b0;
    logic [31:0] service_tag = '0;
    logic [255:0] bank_words = {8{32'h807f_01ff}};
    logic [79:0] bank_row_addresses;
    logic bank_select_pwp;
    logic output_valid;
    logic output_ready = 1'b0;
    logic [31:0] output_tag;
    logic [1:0] output_kind;
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
    integer phase_reload_block_count = 0;

    wire request_valid = service_valid;
    wire request_ready = service_ready;
    wire accepted_grace_match = dut.accepted_grace_match;
    wire request_semantically_valid = dut.request_semantically_valid;
    wire request_violation = dut.request_violation;
    wire request_fault = dut.request_fault_q;
    wire m82_beat_accept = dut.m82_beat_accept;
    wire m82_output_valid = dut.m82_output_valid;
    wire m82_output_accept = dut.m82_output_accept;

    m102_combined_candidate_service_top dut (.*);
    m102_r4_independent_grace_assertions witness_sva (.*);

    always #1.5 clk_core = ~clk_core;
    always @(posedge clk_core) begin
        if (dut.m82_beat_accept)
            input_accept_count = input_accept_count + 1;
        if (output_accept)
            output_accept_count = output_accept_count + 1;
    end

    function automatic integer words_for_code(input integer code);
        case (code)
            0: words_for_code = 24;
            1: words_for_code = 27;
            2: words_for_code = 30;
            3: words_for_code = 33;
            default: words_for_code = 0;
        endcase
    endfunction

    task automatic build_legal_metadata;
        integer cursor;
        integer code;
        begin
            phase_metadata = '0;
            cursor = 0;
            for (int pattern = 0; pattern < 16; pattern++) begin
                phase_metadata[384+pattern*13 +: 13] = cursor[12:0];
                for (int block = 0; block < 8; block++) begin
                    code = (pattern*8 + block) % 5;
                    phase_metadata[(pattern*8+block)*3 +: 3] = code[2:0];
                    cursor += words_for_code(code);
                end
            end
            if (cursor <= 0 || cursor > 3680)
                $fatal(1, "candidate metadata terminal=%0d", cursor);
        end
    endtask

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
            phase_load_valid = 1'b0;
            service_valid = 1'b0;
            output_ready = 1'b0;
            clear_events();
            repeat (2) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
        end
    endtask

    task automatic load_legal_phase;
        integer waited;
        begin
            build_legal_metadata();
            @(negedge clk_core);
            #0.1;
            if (!phase_load_ready)
                $fatal(1, "candidate legal phase load not ready");
            phase_load_valid = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            phase_load_valid = 1'b0;
            waited = 0;
            while (!phase_loaded) begin
                @(posedge clk_core);
                #0.1;
                waited = waited + 1;
                if (waited > 128)
                    $fatal(1, "candidate metadata parser timeout");
            end
            if (waited != 128 || metadata_error || protocol_error)
                $fatal(1, "candidate metadata admission cycles=%0d metadata_error=%0d fault=%0d",
                       waited, metadata_error, protocol_error);
        end
    endtask

    task automatic drive_correction_three_beats(input logic [31:0] tag_value);
        begin
            for (int beat = 0; beat < 3; beat++) begin
                @(negedge clk_core);
                service_valid = 1'b1;
                service_kind = 2'd1;
                service_pattern = 4'd0;
                service_source = 4'd2;
                service_block = 3'd3;
                service_beat = beat[2:0];
                service_negate = 1'b0;
                service_tag = tag_value;
                do @(posedge clk_core); while (!service_ready);
            end
        end
    endtask

    task automatic reset_after_fault;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            phase_load_valid = 1'b0;
            service_valid = 1'b0;
            output_ready = 1'b1;
            clear_events();
            @(posedge clk_core);
            #0.1;
            if (protocol_error || dut.request_fault_q
                    || dut.m82_output_valid || output_valid || output_accept)
                $fatal(1, "candidate reset edge failed to clear sticky state");
            @(negedge clk_core);
            rst_core = 1'b0;
            reset_recovery_event = 1'b1;
            @(posedge clk_core);
            #0.1;
            if (protocol_error || dut.request_fault_q
                    || dut.m82_output_valid || output_valid || output_accept)
                $fatal(1, "candidate post-reset recovery failed");
            reset_recovery_count = reset_recovery_count + 1;
            @(negedge clk_core);
            reset_recovery_event = 1'b0;
        end
    endtask

    task automatic attack_identity_field(input integer field_id);
        integer accepts_before;
        begin
            reset_clean();
            load_legal_phase();
            accepts_before = input_accept_count;
            output_ready = 1'b0;
            drive_correction_three_beats(32'hc400_1000 + field_id);
            #0.1;
            if (!dut.accepted_grace_match || protocol_error
                    || !dut.m82_output_valid || !output_valid
                    || input_accept_count != accepts_before + 3)
                $fatal(1, "candidate mutation%0d setup failed", field_id);

            @(negedge clk_core);
            case (field_id)
                0: service_kind = 2'd3;
                1: service_pattern = service_pattern + 1'b1;
                2: service_source = service_source + 1'b1;
                3: service_block = service_block + 1'b1;
                4: service_beat = 3'd1;
                5: service_negate = 1'b1;
                6: service_tag = service_tag + 1'b1;
                default: $fatal(1, "unknown candidate mutation field=%0d", field_id);
            endcase
            mutation_event = 1'b1;
            output_ready = 1'b1;
            #0.1;
            if (!dut.request_violation || !protocol_error || service_ready
                    || phase_load_ready || output_valid || output_accept
                    || !dut.m82_output_valid || dut.m82_output_accept)
                $fatal(1, "candidate mutation%0d pre-edge quarantine failed", field_id);
            @(posedge clk_core);
            #0.1;
            if (!dut.request_fault_q || !protocol_error
                    || output_valid || output_accept
                    || !dut.m82_output_valid
                    || input_accept_count != accepts_before + 3)
                $fatal(1, "candidate mutation%0d fault edge failed", field_id);
            @(negedge clk_core);
            mutation_event = 1'b0;
            service_valid = 1'b0;

            // The first mutation also holds phase_load_valid across two
            // registered-fault edges.  It is not a recovery protocol.
            if (field_id == 0) begin
                phase_load_valid = 1'b1;
                phase_reload_event = 1'b1;
                repeat (2) begin
                    #0.1;
                    if (phase_load_ready || !dut.request_fault_q
                            || !protocol_error || !dut.m82_output_valid)
                        $fatal(1, "candidate faulted phase reload escaped");
                    @(posedge clk_core);
                    #0.1;
                    if (phase_load_ready || !dut.request_fault_q
                            || !protocol_error || !dut.m82_output_valid)
                        $fatal(1, "candidate phase reload changed sticky state");
                    @(negedge clk_core);
                end
                phase_load_valid = 1'b0;
                phase_reload_event = 1'b0;
                phase_reload_block_count = phase_reload_block_count + 1;
            end else begin
                repeat (2) begin
                    @(posedge clk_core);
                    #0.1;
                    if (!dut.request_fault_q || !protocol_error
                            || output_valid || output_accept
                            || !dut.m82_output_valid)
                        $fatal(1, "candidate mutation%0d fault not sticky", field_id);
                end
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
        load_legal_phase();

        // Hold an exact accepted final request for a complete extra edge.
        accepts_before = input_accept_count;
        outputs_before = output_accept_count;
        output_ready = 1'b0;
        drive_correction_three_beats(32'hc400_0001);
        #0.1;
        if (!dut.accepted_grace_match || protocol_error
                || !output_valid || output_tag != 32'hc400_0001
                || output_kind != 2'd1 || output_width != 8
                || output_escape || input_accept_count != accepts_before + 3)
            $fatal(1, "candidate exact-grace setup failed");
        @(negedge clk_core);
        grace_hold_event = 1'b1;
        @(posedge clk_core);
        #0.1;
        if (protocol_error || service_ready || dut.m82_beat_accept
                || !output_valid || output_tag != 32'hc400_0001
                || input_accept_count != accepts_before + 3)
            $fatal(1, "candidate full-cycle grace caused fault/double accept");
        @(negedge clk_core);
        grace_hold_event = 1'b0;

        // A low-high pulse strictly between active sampling edges cannot
        // revoke and recreate a synchronous request.
        #0.2 service_valid = 1'b0;
        #0.4 service_valid = 1'b1;
        #0.2 glitch_return_event = 1'b1;
        if (!dut.accepted_grace_match || protocol_error)
            $fatal(1, "candidate between-edge low/high was misclassified");
        @(posedge clk_core);
        #0.1;
        if (!dut.accepted_grace_match || protocol_error
                || dut.m82_beat_accept
                || input_accept_count != accepts_before + 3)
            $fatal(1, "candidate glitch return caused fault/double accept");
        @(negedge clk_core);
        glitch_return_event = 1'b0;

        output_ready = 1'b1;
        #0.1;
        if (!output_valid || !output_accept
                || output_tag != 32'hc400_0001 || protocol_error)
            $fatal(1, "candidate grace result was not visible/acceptable");
        @(posedge clk_core);
        #0.1;
        if (output_valid || protocol_error || dut.m82_beat_accept
                || input_accept_count != accepts_before + 3
                || output_accept_count != outputs_before + 1)
            $fatal(1, "candidate result retirement/double-accept mismatch");
        @(negedge clk_core);
        service_valid = 1'b0;
        @(posedge clk_core);
        #0.1;
        if (dut.accepted_grace_q)
            $fatal(1, "candidate grace did not clear on sampled valid-low");

        // Every candidate request-identity field is changed independently.
        for (int field_id = 0; field_id < 7; field_id++)
            attack_identity_field(field_id);

        if (mutation_count != 7 || reset_recovery_count != 7
                || phase_reload_block_count != 1 || protocol_error)
            $fatal(1, "candidate witness counters mutations=%0d resets=%0d reload=%0d fault=%0d",
                   mutation_count, reset_recovery_count,
                   phase_reload_block_count, protocol_error);
        $display("PASS M102_R4_INDEPENDENT_CANDIDATE grace_full_edges=1 glitch_low_high=1 identity_mutations=7 sticky_checks=14 phase_reload_blocks=1 reset_recoveries=7 no_double_accept=1 result_visible=1");
        $finish;
    end

    initial begin
        #500000;
        $fatal(1, "M102 r4 candidate independent witness watchdog");
    end
endmodule

`default_nettype wire
