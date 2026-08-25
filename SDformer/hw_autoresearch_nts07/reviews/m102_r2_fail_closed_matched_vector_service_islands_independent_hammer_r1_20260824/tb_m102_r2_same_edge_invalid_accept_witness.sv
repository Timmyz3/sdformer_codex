`timescale 1ns/1ps
`default_nettype none

// Independent hammer witness.  This does not replace the sealed directed TB.
// It checks whether an already-stalled result can retire on the same cycle that
// a new, semantically invalid request is presented.
module tb_m102_r2_same_edge_invalid_accept_witness;
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
        end
    endtask

    initial begin
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        build_legal_metadata();
        phase_load_valid = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        phase_load_valid = 1'b0;
        while (!phase_loaded) @(posedge clk_core);

        // Produce one valid correction result and keep it stalled.
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
            service_tag = 32'hf017_1001;
            bank_words = {8{32'h0102_0304}};
            do @(posedge clk_core); while (!service_ready);
        end
        @(negedge clk_core);
        service_valid = 1'b0;
        #0.1;
        if (!output_valid || !dut.m82_output_valid || output_accept || protocol_error)
            $fatal(1, "witness setup failed");

        // Present an invalid kind and release the old output in the same cycle.
        service_valid = 1'b1;
        service_kind = 2'd3;
        service_beat = 3'd0;
        service_tag = 32'hf017_1002;
        output_ready = 1'b1;
        #0.1;
        if (!output_accept || protocol_error || service_ready)
            $fatal(1, "expected same-edge acceptance window was absent");
        $display("WITNESS M102_R2_SAME_EDGE_INVALID_ACCEPT preedge output_accept=%0d protocol_error=%0d semantic_valid=%0d m82_valid=%0d",
                 output_accept, protocol_error, dut.request_semantically_valid,
                 dut.m82_output_valid);
        @(posedge clk_core);
        #0.1;
        if (!protocol_error || dut.m82_output_valid)
            $fatal(1, "witness did not retire old output while latching fault");
        $display("WITNESS_CONFIRMED M102_R2 old output retired on invalid-request edge");

        // request_fault_q is also not reset-only sticky: once the bypass has
        // drained M82, a new phase load is admitted and clears the fault.
        @(negedge clk_core);
        service_valid = 1'b0;
        phase_load_valid = 1'b1;
        #0.1;
        if (!phase_load_ready || !protocol_error)
            $fatal(1, "expected phase-load recovery window was absent");
        @(posedge clk_core);
        #0.1;
        if (protocol_error || !dut.parse_active_q || dut.request_fault_q)
            $fatal(1, "expected non-reset fault clear was absent");
        $display("WITNESS_CONFIRMED M102_R2 request fault cleared by phase load without reset");
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "witness watchdog");
    end
endmodule

`default_nettype wire
