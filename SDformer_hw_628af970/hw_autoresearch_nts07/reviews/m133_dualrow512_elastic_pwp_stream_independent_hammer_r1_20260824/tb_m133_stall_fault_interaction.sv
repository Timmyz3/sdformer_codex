`timescale 1ns/1ps
`default_nettype none

// Independent adversarial cross-property test.  It deliberately overlaps a
// previously sampled output stall with an invalid input request.  The RTL's
// same-cycle quarantine is expected to hide output_valid immediately; the
// frozen production SVA's unconditional stall-stability consequent is expected
// to fail on the next clock.  This is a verification-contract diagnostic, not
// a proposed production implementation.
module tb_m133_stall_fault_interaction;
    localparam int LANES = 96;
    localparam int OUT_W = 12;
    localparam int OUT_BITS = LANES * OUT_W;

    logic clk_core, rst_core;
    logic beat_valid, beat_ready, beat_start, beat_last;
    logic [3:0] beat_width;
    logic [31:0] beat_tag;
    logic [511:0] beat_data;
    logic beat_accept;
    logic output_valid, output_ready;
    logic [31:0] output_tag;
    logic [3:0] output_width;
    logic output_escape;
    logic [OUT_BITS-1:0] output_values;
    logic output_accept, protocol_error, collecting, busy;

    m133_dualrow512_elastic_pwp_stream dut (.*);
    m133_dualrow512_elastic_pwp_stream_assertions frozen_sva (.*);

    initial clk_core = 0;
    always #1.5 clk_core = ~clk_core;

    task automatic drive_beat(
        input logic start_value,
        input logic last_value,
        input logic [3:0] width_value,
        input logic [31:0] tag_value
    );
        @(negedge clk_core);
        beat_valid = 1;
        beat_start = start_value;
        beat_last = last_value;
        beat_width = width_value;
        beat_tag = tag_value;
        beat_data = 0;
        @(posedge clk_core);
        if (!beat_accept)
            $fatal(1, "legal setup beat was not accepted");
    endtask

    initial begin
        rst_core = 1;
        beat_valid = 0;
        beat_start = 0;
        beat_last = 0;
        beat_width = 0;
        beat_tag = 0;
        beat_data = 0;
        output_ready = 0;

        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 0;

        drive_beat(1, 0, 8, 32'h133);
        drive_beat(0, 1, 0, 0);

        // Clear input and sample one full cycle of legal output stall, which
        // starts ap_output_stable_under_stall's overlapped implication.
        @(negedge clk_core);
        beat_valid = 0;
        beat_start = 0;
        beat_last = 0;
        beat_width = 0;
        beat_tag = 0;
        #1ps;
        if (!output_valid || output_ready)
            $fatal(1, "failed to establish legal output stall");
        @(posedge clk_core);

        // Before the consequent sample, present an invalid continuation while
        // the output remains stalled.  The RTL intentionally quarantines it.
        @(negedge clk_core);
        beat_valid = 1;
        beat_start = 0;
        beat_last = 0;
        beat_width = 9;
        beat_tag = 32'hbad;
        beat_data = 0;
        #1ps;
        if (!protocol_error || beat_ready || beat_accept || output_valid
                || output_accept)
            $fatal(1, "same-cycle quarantine behavior changed");
        @(posedge clk_core);
        #1ps;
        if (!protocol_error || output_valid)
            $fatal(1, "sticky quarantine behavior changed");

        // Remove the bad request.  faulted_q must keep all activity isolated
        // for multiple cycles, rather than merely mirroring request_violation.
        @(negedge clk_core);
        beat_valid = 0;
        beat_start = 0;
        beat_last = 0;
        beat_width = 0;
        beat_tag = 0;
        repeat (3) begin
            @(posedge clk_core);
            #1ps;
            if (!protocol_error || beat_ready || beat_accept || output_valid
                    || output_accept)
                $fatal(1, "protocol fault was not sticky and quarantined");
        end

        $display("PASS M133 independent cross-property stimulus reached expected quarantine and 3-cycle sticky fault; frozen stall SVA must report one failure");
        $finish;
    end

    initial begin
        #10000;
        $fatal(1, "independent M133 interaction timeout");
    end
endmodule

`default_nettype wire
