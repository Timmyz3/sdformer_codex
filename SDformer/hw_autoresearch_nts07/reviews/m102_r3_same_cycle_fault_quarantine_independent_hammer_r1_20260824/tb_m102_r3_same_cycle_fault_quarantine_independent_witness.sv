`timescale 1ns/1ps
`default_nettype none

// Independent Synopsys witness for the two r2 P0 counterexamples.  This TB is
// intentionally separate from the production directed suite and reaches the
// M82 buffered-result state through the public service interface.
module tb_m102_r3_same_cycle_fault_quarantine_independent_witness;
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
    logic [255:0] bank_words = '0;
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

    logic [1151:0] held_m82_values;
    logic [31:0] held_m82_tag;

    m102_combined_candidate_service_top dut (.*);
    always #1.5 clk_core = ~clk_core;

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
                $fatal(1, "independent metadata builder terminal=%0d", cursor);
        end
    endtask

    task automatic load_legal_phase;
        integer wait_cycles;
        begin
            build_legal_metadata();
            @(negedge clk_core);
            if (!phase_load_ready)
                $fatal(1, "initial legal phase load was not ready");
            phase_load_valid = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            phase_load_valid = 1'b0;
            wait_cycles = 0;
            while (!phase_loaded) begin
                @(posedge clk_core);
                #0.1;
                wait_cycles++;
                if (wait_cycles > 128)
                    $fatal(1, "metadata parser timeout");
            end
            if (wait_cycles != 128 || metadata_error || protocol_error)
                $fatal(1, "metadata admission mismatch cycles=%0d metadata_error=%0d protocol_error=%0d",
                       wait_cycles, metadata_error, protocol_error);
        end
    endtask

    task automatic produce_stalled_correction;
        begin
            output_ready = 1'b0;
            for (int beat = 0; beat < 3; beat++) begin
                @(negedge clk_core);
                service_valid = 1'b1;
                service_kind = 2'd1;
                service_pattern = 4'd0;
                service_source = 4'd2;
                service_block = 3'd3;
                service_beat = beat[2:0];
                service_negate = 1'b0;
                service_tag = 32'hf017_3001;
                bank_words = {8{32'h807f_01ff}};
                do @(posedge clk_core); while (!service_ready);
            end
            @(negedge clk_core);
            service_valid = 1'b0;
            #0.1;
            if (!output_valid || output_accept || protocol_error
                    || !dut.m82_output_valid || dut.m82_output_accept)
                $fatal(1, "failed to build clean stalled M82 output valid=%0d accept=%0d fault=%0d m82_valid=%0d m82_accept=%0d",
                       output_valid, output_accept, protocol_error,
                       dut.m82_output_valid, dut.m82_output_accept);
            held_m82_values = dut.m82_output_values;
            held_m82_tag = dut.m82_output_tag;
        end
    endtask

    initial begin
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        load_legal_phase();
        produce_stalled_correction();

        // The r2 counterexample ordering: an illegal request is presented in
        // the same combinational window in which output_ready is released.
        service_valid = 1'b1;
        service_kind = 2'd3;
        service_beat = 3'd0;
        service_tag = 32'hf017_3002;
        output_ready = 1'b1;
        #0.1;
        if (!dut.request_violation || !protocol_error || service_ready
                || phase_load_ready || output_valid || output_accept
                || !dut.m82_output_valid || dut.m82_output_accept
                || dut.m82_output_values !== held_m82_values
                || dut.m82_output_tag !== held_m82_tag)
            $fatal(1, "pre-edge quarantine failed violation=%0d fault=%0d service_ready=%0d phase_ready=%0d output_valid=%0d output_accept=%0d m82_valid=%0d m82_accept=%0d",
                   dut.request_violation, protocol_error, service_ready,
                   phase_load_ready, output_valid, output_accept,
                   dut.m82_output_valid, dut.m82_output_accept);
        $display("WITNESS M102_R3_PREEDGE_QUARANTINE output_valid=0 output_accept=0 protocol_error=1 m82_valid=1 m82_accept=0");

        @(posedge clk_core);
        #0.1;
        if (!dut.request_fault_q || !protocol_error || output_valid
                || output_accept || !dut.m82_output_valid
                || dut.m82_output_accept
                || dut.m82_output_values !== held_m82_values
                || dut.m82_output_tag !== held_m82_tag)
            $fatal(1, "faulting edge retired/corrupted buffered result request_fault=%0d fault=%0d output_valid=%0d accept=%0d m82_valid=%0d m82_accept=%0d",
                   dut.request_fault_q, protocol_error, output_valid,
                   output_accept, dut.m82_output_valid,
                   dut.m82_output_accept);
        $display("WITNESS M102_R3_POSTEDGE_RETENTION request_fault=1 m82_valid=1 old_output_retired=0");

        // Phase reload is held for multiple faulted edges.  It must neither
        // be accepted nor clear/flush any registered or M82 state.
        @(negedge clk_core);
        service_valid = 1'b0;
        phase_load_valid = 1'b1;
        repeat (3) begin
            #0.1;
            if (phase_load_ready || !protocol_error || !dut.request_fault_q
                    || dut.parse_active_q || output_valid || output_accept
                    || !dut.m82_output_valid
                    || dut.m82_output_values !== held_m82_values
                    || dut.m82_output_tag !== held_m82_tag)
                $fatal(1, "faulted phase reload escaped ready=%0d fault=%0d request_fault=%0d parse=%0d m82_valid=%0d",
                       phase_load_ready, protocol_error,
                       dut.request_fault_q, dut.parse_active_q,
                       dut.m82_output_valid);
            @(posedge clk_core);
            #0.1;
            if (phase_load_ready || !protocol_error || !dut.request_fault_q
                    || dut.parse_active_q || output_valid || output_accept
                    || !dut.m82_output_valid
                    || dut.m82_output_values !== held_m82_values
                    || dut.m82_output_tag !== held_m82_tag)
                $fatal(1, "phase reload changed sticky fault on edge");
            @(negedge clk_core);
        end
        $display("WITNESS M102_R3_PHASE_RELOAD_BLOCKED edges=3 request_fault=1 m82_valid=1");

        // Keep phase_load_valid asserted: a synchronous reset must be the
        // event that clears both the top fault and the quarantined M82 result.
        rst_core = 1'b1;
        @(posedge clk_core);
        #0.1;
        if (protocol_error || dut.request_fault_q || dut.m82_output_valid
                || output_valid || output_accept || busy || !phase_load_ready)
            $fatal(1, "reset did not recover fault/output fault=%0d request_fault=%0d m82_valid=%0d busy=%0d phase_ready=%0d",
                   protocol_error, dut.request_fault_q,
                   dut.m82_output_valid, busy, phase_load_ready);
        @(negedge clk_core);
        rst_core = 1'b0;
        #0.1;
        if (!phase_load_ready || protocol_error)
            $fatal(1, "post-reset phase-load admission missing");
        @(posedge clk_core);
        #0.1;
        if (!dut.parse_active_q || dut.request_fault_q || protocol_error)
            $fatal(1, "post-reset phase load did not start parser");
        $display("WITNESS_CONFIRMED M102_R3 reset_only_recovery=1 post_reset_phase_load=1");
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "M102 r3 independent witness watchdog");
    end
endmodule

`default_nettype wire
