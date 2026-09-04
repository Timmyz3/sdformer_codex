`timescale 1ns/1ps
`default_nettype none

module tb_qfit_ds_flm_lane_miter;
    localparam int HEAD_DIM = 32;
    localparam int GATE_W = 9;
    localparam int SOURCE_ID_W = 9;
    localparam int Y_W = 4;
    localparam int X_W = 4;
    localparam int LANE_W = $clog2(HEAD_DIM);
    localparam int RANDOM_DESCRIPTORS = 100;

    logic clk_core;
    logic rst_core;
    logic descriptor_valid;
    logic [SOURCE_ID_W-1:0] descriptor_source_id;
    logic [Y_W-1:0] descriptor_y;
    logic [X_W-1:0] descriptor_x;
    logic [HEAD_DIM-1:0] descriptor_k;
    logic [5*GATE_W-1:0] descriptor_incoming_gates;
    logic [4:0] descriptor_valid_mask;
    logic term_ready;

    logic baseline_descriptor_ready;
    logic baseline_term_valid;
    logic [SOURCE_ID_W-1:0] baseline_term_source_id;
    logic [Y_W-1:0] baseline_term_source_y;
    logic [X_W-1:0] baseline_term_source_x;
    logic [LANE_W-1:0] baseline_term_lane;
    logic [GATE_W-1:0] baseline_term_gate;
    logic [4:0] baseline_term_destination_mask;
    logic baseline_term_last;
    logic [31:0] baseline_perf_descriptors;
    logic [31:0] baseline_perf_terms;
    logic [31:0] baseline_perf_updates;

    logic ds_descriptor_ready;
    logic ds_term_valid;
    logic [SOURCE_ID_W-1:0] ds_term_source_id;
    logic [Y_W-1:0] ds_term_source_y;
    logic [X_W-1:0] ds_term_source_x;
    logic [LANE_W-1:0] ds_term_lane;
    logic [GATE_W-1:0] ds_term_gate;
    logic [4:0] ds_term_destination_mask;
    logic ds_term_last;
    logic [31:0] ds_perf_descriptors;
    logic [31:0] ds_perf_terms;
    logic [31:0] ds_perf_updates;

    int cycle_count;
    int accepted_descriptors;
    int compared_terms;
    int input_stall_cycles;
    int output_stall_cycles;

    qfit_source_multicast_term_builder #(
        .HEAD_DIM(HEAD_DIM),
        .GATE_W(GATE_W),
        .SOURCE_ID_W(SOURCE_ID_W),
        .Y_W(Y_W),
        .X_W(X_W)
    ) baseline (
        .clk_core,
        .rst_core,
        .descriptor_valid,
        .descriptor_ready(baseline_descriptor_ready),
        .descriptor_source_id,
        .descriptor_y,
        .descriptor_x,
        .descriptor_k,
        .descriptor_incoming_gates,
        .descriptor_valid_mask,
        .term_valid(baseline_term_valid),
        .term_ready,
        .term_source_id(baseline_term_source_id),
        .term_source_y(baseline_term_source_y),
        .term_source_x(baseline_term_source_x),
        .term_lane(baseline_term_lane),
        .term_gate(baseline_term_gate),
        .term_destination_mask(baseline_term_destination_mask),
        .term_last(baseline_term_last),
        .perf_descriptors(baseline_perf_descriptors),
        .perf_terms(baseline_perf_terms),
        .perf_destination_updates(baseline_perf_updates)
    );

    qfit_ds_flm_materializer #(
        .HEAD_DIM(HEAD_DIM),
        .GATE_W(GATE_W),
        .SOURCE_ID_W(SOURCE_ID_W),
        .Y_W(Y_W),
        .X_W(X_W)
    ) candidate (
        .clk_core,
        .rst_core,
        .descriptor_valid,
        .descriptor_ready(ds_descriptor_ready),
        .descriptor_mode(1'b0),
        .descriptor_source_id,
        .descriptor_y,
        .descriptor_x,
        .descriptor_k,
        .descriptor_incoming_gates,
        .descriptor_valid_mask,
        .term_valid(ds_term_valid),
        .term_ready,
        .term_source_id(ds_term_source_id),
        .term_source_y(ds_term_source_y),
        .term_source_x(ds_term_source_x),
        .term_lane(ds_term_lane),
        .term_gate(ds_term_gate),
        .term_destination_mask(ds_term_destination_mask),
        .term_last(ds_term_last),
        .perf_descriptors(ds_perf_descriptors),
        .perf_terms(ds_perf_terms),
        .perf_destination_updates(ds_perf_updates)
    );

    always #5 clk_core = ~clk_core;

    task automatic send_descriptor(
        input int index,
        input logic [HEAD_DIM-1:0] k,
        input logic [4:0] valid_mask,
        input int g0,
        input int g1,
        input int g2,
        input int g3,
        input int g4
    );
        int gates [0:4];
        gates[0] = g0;
        gates[1] = g1;
        gates[2] = g2;
        gates[3] = g3;
        gates[4] = g4;
        @(negedge clk_core);
        descriptor_source_id = SOURCE_ID_W'(index);
        descriptor_y = Y_W'(index >> 4);
        descriptor_x = X_W'(index);
        descriptor_k = k;
        descriptor_valid_mask = valid_mask;
        for (int role = 0; role < 5; role = role + 1)
            descriptor_incoming_gates[
                role*GATE_W +: GATE_W
            ] = GATE_W'(gates[role]);
        descriptor_valid = 1'b1;
        while (!(baseline_descriptor_ready && ds_descriptor_ready)) begin
            input_stall_cycles = input_stall_cycles + 1;
            @(negedge clk_core);
        end
        @(posedge clk_core);
        accepted_descriptors = accepted_descriptors + 1;
        @(negedge clk_core);
        descriptor_valid = 1'b0;
    endtask

    always @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count = 0;
        end else begin
            cycle_count = cycle_count + 1;
            if (cycle_count > 50000)
                $fatal(1, "miter timeout");
            if (baseline_descriptor_ready !== ds_descriptor_ready)
                $fatal(1, "descriptor ready mismatch");
            if (baseline_term_valid !== ds_term_valid)
                $fatal(1, "term valid mismatch");
            if (baseline_term_valid) begin
                if (
                    baseline_term_source_id !== ds_term_source_id
                    || baseline_term_source_y !== ds_term_source_y
                    || baseline_term_source_x !== ds_term_source_x
                    || baseline_term_lane !== ds_term_lane
                    || baseline_term_gate !== ds_term_gate
                    || baseline_term_destination_mask
                        !== ds_term_destination_mask
                    || baseline_term_last !== ds_term_last
                )
                    $fatal(1, "lane-major payload mismatch");
                if (!term_ready)
                    output_stall_cycles = output_stall_cycles + 1;
                else
                    compared_terms = compared_terms + 1;
            end
        end
    end

    always @(negedge clk_core) begin
        if (rst_core)
            term_ready = 1'b0;
        else
            term_ready = ($urandom_range(0, 4) != 0);
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
        cycle_count = 0;
        accepted_descriptors = 0;
        compared_terms = 0;
        input_stall_cycles = 0;
        output_stall_cycles = 0;
        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;

        send_descriptor(
            0, 32'hffff_ffff, 5'b11111, 7, 11, 7, 0, 256
        );
        send_descriptor(1, '0, 5'b11111, 1, 2, 3, 4, 5);
        send_descriptor(2, 32'h8000_0001, 5'b00000, 1, 2, 3, 4, 5);

        for (int index = 0; index < RANDOM_DESCRIPTORS; index = index + 1)
            send_descriptor(
                index + 3,
                $urandom,
                5'($urandom_range(0, 31)),
                $urandom_range(0, 511),
                $urandom_range(0, 511),
                (index % 3 == 0)
                    ? $urandom_range(0, 511)
                    : (index + 1),
                (index % 4 == 0) ? 0 : (index + 3),
                (index % 5 == 0) ? (index + 1) : (index + 5)
            );

        wait (
            baseline_descriptor_ready
            && ds_descriptor_ready
            && !baseline_term_valid
            && !ds_term_valid
        );
        repeat (3) @(negedge clk_core);
        if (
            baseline_perf_descriptors != ds_perf_descriptors
            || baseline_perf_terms != ds_perf_terms
            || baseline_perf_updates != ds_perf_updates
        )
            $fatal(1, "performance counter mismatch");
        if (baseline_perf_descriptors != accepted_descriptors)
            $fatal(1, "accepted descriptor count mismatch");
        if (baseline_perf_terms != compared_terms)
            $fatal(1, "accepted term count mismatch");
        if (input_stall_cycles == 0 || output_stall_cycles == 0)
            $fatal(1, "miter did not exercise both backpressure directions");
        $display(
            "PASS qfit DS-FLM lane miter descriptors=%0d terms=%0d input_stalls=%0d output_stalls=%0d",
            accepted_descriptors,
            compared_terms,
            input_stall_cycles,
            output_stall_cycles
        );
        $finish;
    end
endmodule

`default_nettype wire
