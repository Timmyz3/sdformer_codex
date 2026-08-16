`timescale 1ns/1ps
`default_nettype none

module tb_qfit_source_multicast_term_builder;
    localparam int HEAD_DIM = 8;
    localparam int GATE_W = 9;
    localparam int SOURCE_ID_W = 8;
    localparam int Y_W = 4;
    localparam int X_W = 4;
    localparam int LANE_W = $clog2(HEAD_DIM);

    logic clk_core;
    logic rst_core;
    logic descriptor_valid;
    logic descriptor_ready;
    logic [SOURCE_ID_W-1:0] descriptor_source_id;
    logic [Y_W-1:0] descriptor_y;
    logic [X_W-1:0] descriptor_x;
    logic [HEAD_DIM-1:0] descriptor_k;
    logic [5*GATE_W-1:0] descriptor_incoming_gates;
    logic [4:0] descriptor_valid_mask;
    logic term_valid;
    logic term_ready;
    logic [SOURCE_ID_W-1:0] term_source_id;
    logic [Y_W-1:0] term_source_y;
    logic [X_W-1:0] term_source_x;
    logic [LANE_W-1:0] term_lane;
    logic [GATE_W-1:0] term_gate;
    logic [4:0] term_destination_mask;
    logic term_last;
    logic [31:0] perf_descriptors;
    logic [31:0] perf_terms;
    logic [31:0] perf_destination_updates;

    int expected_terms;
    int observed_terms;
    int expected_updates;
    int observed_updates;
    int cycle_count;
    logic [4:0] seen_lane_gate [0:HEAD_DIM-1][0:2];

    qfit_source_multicast_term_builder #(
        .HEAD_DIM(HEAD_DIM),
        .GATE_W(GATE_W),
        .SOURCE_ID_W(SOURCE_ID_W),
        .Y_W(Y_W),
        .X_W(X_W)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (cycle_count > 500)
                $fatal(
                    1,
                    "timeout observed=%0d perf=%0d valid=%b ready=%b desc_ready=%b state=%0d lane=%0d gate_index=%0d remaining=%0d",
                    observed_terms,
                    perf_terms,
                    term_valid,
                    term_ready,
                    descriptor_ready,
                    dut.state_q,
                    dut.selected_lane,
                    dut.gate_index_q,
                    dut.terms_remaining_q
                );
        end
    end

    function automatic int popcount5(input logic [4:0] value);
        int count;
        count = 0;
        for (int i = 0; i < 5; i = i + 1)
            count = count + value[i];
        return count;
    endfunction

    task automatic send_descriptor(
        input logic [SOURCE_ID_W-1:0] sid,
        input logic [Y_W-1:0] y,
        input logic [X_W-1:0] x,
        input logic [HEAD_DIM-1:0] k,
        input logic [4:0] valid_mask,
        input int g0,
        input int g1,
        input int g2,
        input int g3,
        input int g4
    );
        int gate_values [0:4];
        gate_values[0] = g0;
        gate_values[1] = g1;
        gate_values[2] = g2;
        gate_values[3] = g3;
        gate_values[4] = g4;
        @(negedge clk_core);
        descriptor_source_id = sid;
        descriptor_y = y;
        descriptor_x = x;
        descriptor_k = k;
        descriptor_valid_mask = valid_mask;
        for (int role = 0; role < 5; role = role + 1)
            descriptor_incoming_gates[
                role*GATE_W +: GATE_W
            ] = GATE_W'(gate_values[role]);
        while (!descriptor_ready)
            @(negedge clk_core);
        descriptor_valid = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        descriptor_valid = 1'b0;
    endtask

    always_ff @(posedge clk_core) begin
        if (!rst_core && term_valid && term_ready) begin
            logic [4:0] expected_mask;
            int gate_slot;
            expected_mask = '0;
            gate_slot = -1;
            case (term_gate)
                9'd7: begin
                    expected_mask = 5'b00101;
                    gate_slot = 0;
                end
                9'd11: begin
                    expected_mask = 5'b01010;
                    gate_slot = 1;
                end
                9'd256: begin
                    expected_mask = 5'b10000;
                    gate_slot = 2;
                end
                default: $fatal(1, "unexpected gate %0d", term_gate);
            endcase
            if (
                term_source_id != 8'd23
                || term_source_y != 4'd3
                || term_source_x != 4'd4
            )
                $fatal(1, "source metadata mismatch");
            if (!descriptor_k[term_lane])
                $fatal(1, "term emitted for inactive K lane %0d", term_lane);
            if (term_destination_mask != expected_mask)
                $fatal(
                    1,
                    "mask mismatch gate=%0d got=%b exp=%b",
                    term_gate,
                    term_destination_mask,
                    expected_mask
                );
            if (seen_lane_gate[term_lane][gate_slot] != 0)
                $fatal(1, "duplicate lane/gate term");
            seen_lane_gate[term_lane][gate_slot] <= expected_mask;
            observed_terms <= observed_terms + 1;
            observed_updates <=
                observed_updates + popcount5(term_destination_mask);
            if (
                term_last
                != (observed_terms + 1 == expected_terms)
            )
                $fatal(1, "term_last mismatch");
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        descriptor_valid = 1'b0;
        descriptor_source_id = '0;
        descriptor_y = '0;
        descriptor_x = '0;
        descriptor_k = '0;
        descriptor_incoming_gates = '0;
        descriptor_valid_mask = '0;
        term_ready = 1'b0;
        expected_terms = 12;
        expected_updates = 20;
        observed_terms = 0;
        observed_updates = 0;
        cycle_count = 0;
        for (int lane = 0; lane < HEAD_DIM; lane = lane + 1)
            for (int gate = 0; gate < 3; gate = gate + 1)
                seen_lane_gate[lane][gate] = '0;
        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;

        // Four active lanes and three unique nonzero gates. Gate 7 and 11
        // each multicast to two destinations; 256 verifies the ninth bit.
        send_descriptor(
            8'd23,
            4'd3,
            4'd4,
            8'b10100101,
            5'b11111,
            7,
            11,
            7,
            11,
            256
        );
        repeat (3) begin
            @(negedge clk_core);
            term_ready = 1'b0;
        end
        term_ready = 1'b1;
        wait (observed_terms == expected_terms);
        repeat (3) @(negedge clk_core);
        if (observed_updates != expected_updates)
            $fatal(
                1,
                "update count mismatch got=%0d exp=%0d",
                observed_updates,
                expected_updates
            );
        if (
            perf_descriptors != 1
            || perf_terms != expected_terms
            || perf_destination_updates != expected_updates
        )
            $fatal(1, "performance counters mismatch");

        // A zero-K descriptor must produce no term and release immediately.
        send_descriptor(
            8'd24,
            4'd3,
            4'd5,
            '0,
            5'b00001,
            9,
            0,
            0,
            0,
            0
        );
        repeat (3) @(negedge clk_core);
        if (!descriptor_ready || term_valid)
            $fatal(1, "zero-K descriptor did not retire");
        if (perf_descriptors != 2 || perf_terms != expected_terms)
            $fatal(1, "zero-K counters mismatch");
        $display(
            "PASS qfit_source_multicast_term_builder terms=%0d updates=%0d",
            observed_terms,
            observed_updates
        );
        $finish;
    end
endmodule

`default_nettype wire
