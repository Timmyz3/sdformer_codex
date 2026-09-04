`timescale 1ns/1ps
`default_nettype none

module tb_qfit_ds_flm_materializer #(
    parameter int HEAD_DIM = 8
);
    localparam int GATE_W = 9;
    localparam int SOURCE_ID_W = 8;
    localparam int Y_W = 4;
    localparam int X_W = 4;
    localparam int LANE_W = $clog2(HEAD_DIM);
    localparam int MAX_TERMS = HEAD_DIM * 5;

    logic clk_core;
    logic rst_core;
    logic descriptor_valid;
    logic descriptor_ready;
    logic descriptor_mode;
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

    int exp_lane [0:MAX_TERMS-1];
    int exp_gate [0:MAX_TERMS-1];
    logic [4:0] exp_mask [0:MAX_TERMS-1];
    int canonical_gate [0:4];
    logic [4:0] canonical_mask [0:4];
    logic canonical_seen [0:HEAD_DIM-1][0:4];
    logic [HEAD_DIM-1:0] canonical_k;
    int canonical_unique_count;
    int expected_count;
    int observed_count;
    int descriptor_last_count;
    int expected_total_terms;
    int expected_total_updates;
    int cycle_count;
    int completed_descriptors;
    int stall_cycles;
    int mode_toggle_cycles;
    logic [SOURCE_ID_W-1:0] expected_sid;
    logic [Y_W-1:0] expected_y;
    logic [X_W-1:0] expected_x;

    qfit_ds_flm_materializer #(
        .HEAD_DIM(HEAD_DIM),
        .GATE_W(GATE_W),
        .SOURCE_ID_W(SOURCE_ID_W),
        .Y_W(Y_W),
        .X_W(X_W)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    function automatic int popcount5(input logic [4:0] value);
        int count;
        count = 0;
        for (int i = 0; i < 5; i = i + 1)
            count = count + int'(value[i]);
        return count;
    endfunction

    task automatic build_expected(
        input logic mode,
        input logic [HEAD_DIM-1:0] k,
        input logic [4:0] valid_mask,
        input int g0,
        input int g1,
        input int g2,
        input int g3,
        input int g4
    );
        int role_gate [0:4];
        int found_slot;

        role_gate[0] = g0;
        role_gate[1] = g1;
        role_gate[2] = g2;
        role_gate[3] = g3;
        role_gate[4] = g4;
        expected_count = 0;
        observed_count = 0;
        descriptor_last_count = 0;
        canonical_k = k;
        canonical_unique_count = 0;
        for (int lane = 0; lane < HEAD_DIM; lane = lane + 1)
            for (int slot = 0; slot < 5; slot = slot + 1)
                canonical_seen[lane][slot] = 1'b0;
        for (int slot = 0; slot < 5; slot = slot + 1) begin
            canonical_gate[slot] = 0;
            canonical_mask[slot] = '0;
        end

        for (int role = 0; role < 5; role = role + 1) begin
            if (valid_mask[role] && role_gate[role] != 0) begin
                found_slot = -1;
                for (
                    int slot = 0;
                    slot < canonical_unique_count;
                    slot = slot + 1
                ) begin
                    if (canonical_gate[slot] == role_gate[role])
                        found_slot = slot;
                end
                if (found_slot >= 0) begin
                    canonical_mask[found_slot][role] = 1'b1;
                end else begin
                    canonical_gate[canonical_unique_count] =
                        role_gate[role];
                    canonical_mask[canonical_unique_count][role] = 1'b1;
                    canonical_unique_count = canonical_unique_count + 1;
                end
            end
        end

        if (!mode) begin
            for (int lane = 0; lane < HEAD_DIM; lane = lane + 1) begin
                if (k[lane]) begin
                    for (
                        int slot = 0;
                        slot < canonical_unique_count;
                        slot = slot + 1
                    ) begin
                        exp_lane[expected_count] = lane;
                        exp_gate[expected_count] = canonical_gate[slot];
                        exp_mask[expected_count] = canonical_mask[slot];
                        expected_count = expected_count + 1;
                    end
                end
            end
        end else begin
            for (
                int slot = 0;
                slot < canonical_unique_count;
                slot = slot + 1
            ) begin
                for (int lane = 0; lane < HEAD_DIM; lane = lane + 1) begin
                    if (k[lane]) begin
                        exp_lane[expected_count] = lane;
                        exp_gate[expected_count] = canonical_gate[slot];
                        exp_mask[expected_count] = canonical_mask[slot];
                        expected_count = expected_count + 1;
                    end
                end
            end
        end
    endtask

    task automatic send_and_check(
        input logic mode,
        input logic mutate_mode,
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
        int this_expected;
        int this_updates;
        int expected_last_count;

        gate_values[0] = g0;
        gate_values[1] = g1;
        gate_values[2] = g2;
        gate_values[3] = g3;
        gate_values[4] = g4;
        build_expected(mode, k, valid_mask, g0, g1, g2, g3, g4);
        this_expected = expected_count;
        this_updates = 0;
        expected_last_count = (this_expected != 0) ? 1 : 0;
        for (int i = 0; i < expected_count; i = i + 1)
            this_updates = this_updates + popcount5(exp_mask[i]);

        @(negedge clk_core);
        while (!descriptor_ready)
            @(negedge clk_core);
        expected_sid = sid;
        expected_y = y;
        expected_x = x;
        descriptor_mode = mode;
        descriptor_source_id = sid;
        descriptor_y = y;
        descriptor_x = x;
        descriptor_k = k;
        descriptor_valid_mask = valid_mask;
        for (int role = 0; role < 5; role = role + 1)
            descriptor_incoming_gates[
                role*GATE_W +: GATE_W
            ] = GATE_W'(gate_values[role]);
        descriptor_valid = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        descriptor_valid = 1'b0;

        while (observed_count < this_expected) begin
            if (mutate_mode) begin
                descriptor_mode = ~descriptor_mode;
                mode_toggle_cycles = mode_toggle_cycles + 1;
            end
            @(negedge clk_core);
        end
        repeat (2) @(negedge clk_core);

        if (!descriptor_ready || term_valid)
            $fatal(1, "descriptor did not retire cleanly sid=%0d", sid);
        if (descriptor_last_count != expected_last_count)
            $fatal(
                1,
                "last count mismatch sid=%0d got=%0d exp=%0d",
                sid,
                descriptor_last_count,
                expected_last_count
            );
        for (int lane = 0; lane < HEAD_DIM; lane = lane + 1) begin
            for (
                int slot = 0;
                slot < canonical_unique_count;
                slot = slot + 1
            ) begin
                if (canonical_seen[lane][slot] != canonical_k[lane])
                    $fatal(
                        1,
                        "canonical multiset mismatch sid=%0d lane=%0d slot=%0d",
                        sid,
                        lane,
                        slot
                    );
            end
        end
        expected_total_terms = expected_total_terms + this_expected;
        expected_total_updates = expected_total_updates + this_updates;
        completed_descriptors = completed_descriptors + 1;
    endtask

    always @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count = 0;
        end else begin
            cycle_count = cycle_count + 1;
            if (cycle_count > 4000)
                $fatal(1, "timeout");
            if (term_valid && !term_ready)
                stall_cycles = stall_cycles + 1;
            if (term_valid && term_ready) begin
                int slot_match;
                if (observed_count >= expected_count)
                    $fatal(1, "unexpected extra term");
                if (
                    term_source_id !== expected_sid
                    || term_source_y !== expected_y
                    || term_source_x !== expected_x
                )
                    $fatal(1, "source metadata mismatch");
                if (
                    integer'(term_lane) != exp_lane[observed_count]
                    || integer'(term_gate) != exp_gate[observed_count]
                    || term_destination_mask
                        !== exp_mask[observed_count]
                )
                    $fatal(
                        1,
                        "ordered term mismatch idx=%0d lane=%0d/%0d gate=%0d/%0d mask=%b/%b",
                        observed_count,
                        term_lane,
                        exp_lane[observed_count],
                        term_gate,
                        exp_gate[observed_count],
                        term_destination_mask,
                        exp_mask[observed_count]
                    );
                slot_match = -1;
                for (
                    int slot = 0;
                    slot < canonical_unique_count;
                    slot = slot + 1
                ) begin
                    if (
                        canonical_gate[slot] == integer'(term_gate)
                        && canonical_mask[slot]
                            == term_destination_mask
                    )
                        slot_match = slot;
                end
                if (slot_match < 0)
                    $fatal(1, "term absent from canonical dictionary");
                if (canonical_seen[term_lane][slot_match])
                    $fatal(1, "duplicate canonical lane/gate pair");
                canonical_seen[term_lane][slot_match] = 1'b1;
                if (
                    term_last
                    != (observed_count + 1 == expected_count)
                )
                    $fatal(1, "term_last position mismatch");
                if (term_last)
                    descriptor_last_count = descriptor_last_count + 1;
                observed_count = observed_count + 1;
            end
        end
    end

    always @(negedge clk_core) begin
        if (rst_core)
            term_ready = 1'b0;
        else
            term_ready = ($urandom_range(0, 3) != 0);
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        descriptor_valid = 1'b0;
        descriptor_mode = 1'b0;
        descriptor_source_id = '0;
        descriptor_y = '0;
        descriptor_x = '0;
        descriptor_k = '0;
        descriptor_incoming_gates = '0;
        descriptor_valid_mask = '0;
        term_ready = 1'b0;
        expected_count = 0;
        observed_count = 0;
        descriptor_last_count = 0;
        expected_total_terms = 0;
        expected_total_updates = 0;
        cycle_count = 0;
        completed_descriptors = 0;
        stall_cycles = 0;
        mode_toggle_cycles = 0;
        expected_sid = '0;
        expected_y = '0;
        expected_x = '0;

        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;

        // Same descriptor proves both exact loop orders and first-role dedup.
        send_and_check(
            1'b0, 1'b0, 8'd23, 4'd3, 4'd4,
            8'b10100101, 5'b11111, 7, 11, 7, 11, 256
        );
        send_and_check(
            1'b1, 1'b0, 8'd24, 4'd3, 4'd5,
            8'b10100101, 5'b11111, 7, 11, 7, 11, 256
        );

        // Toggle descriptor_mode during emission; captured mode must win.
        send_and_check(
            1'b0, 1'b1, 8'd25, 4'd4, 4'd4,
            8'b01010110, 5'b11111, 19, 3, 19, 5, 3
        );
        send_and_check(
            1'b1, 1'b1, 8'd26, 4'd4, 4'd5,
            8'b01010110, 5'b11111, 19, 3, 19, 5, 3
        );

        // Zero-K, zero-gate, and invalid-role descriptors emit zero terms.
        send_and_check(
            1'b0, 1'b1, 8'd27, 4'd5, 4'd4,
            '0, 5'b11111, 1, 2, 3, 4, 5
        );
        send_and_check(
            1'b1, 1'b1, 8'd28, 4'd5, 4'd5,
            8'b11111111, 5'b11111, 0, 0, 0, 0, 0
        );
        send_and_check(
            1'b0, 1'b1, 8'd29, 4'd6, 4'd4,
            8'b11111111, 5'b00000, 1, 2, 3, 4, 5
        );

        // Five unique gates exercise the complete dictionary.
        send_and_check(
            1'b1, 1'b1, 8'd30, 4'd6, 4'd5,
            8'b10000001, 5'b11111, 2, 4, 8, 16, 32
        );

        // The parameterized 32-lane run exercises the maximum 160-term
        // descriptor in both loop orders.
        if (HEAD_DIM == 32) begin
            send_and_check(
                1'b0, 1'b1, 8'd31, 4'd7, 4'd4,
                '1, 5'b11111, 3, 5, 9, 17, 257
            );
            send_and_check(
                1'b1, 1'b1, 8'd32, 4'd7, 4'd5,
                '1, 5'b11111, 3, 5, 9, 17, 257
            );
        end

        if (
            perf_descriptors != completed_descriptors
            || perf_terms != expected_total_terms
            || perf_destination_updates != expected_total_updates
        )
            $fatal(
                1,
                "counter mismatch desc=%0d/%0d terms=%0d/%0d updates=%0d/%0d",
                perf_descriptors,
                completed_descriptors,
                perf_terms,
                expected_total_terms,
                perf_destination_updates,
                expected_total_updates
            );
        if (stall_cycles == 0)
            $fatal(1, "random backpressure did not create a stall");
        if (mode_toggle_cycles == 0)
            $fatal(1, "descriptor_mode did not toggle while busy");
        $display(
            "PASS qfit_ds_flm_materializer descriptors=%0d terms=%0d updates=%0d cycles=%0d stalls=%0d mode_toggles=%0d",
            completed_descriptors,
            expected_total_terms,
            expected_total_updates,
            cycle_count,
            stall_cycles,
            mode_toggle_cycles
        );
        $finish;
    end
endmodule

`default_nettype wire
