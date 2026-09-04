`timescale 1ns/1ps
`default_nettype none

module tb_qfit_adaptive_parent_selector_p256_sustained_r2;
    localparam int TESTS = 2048;
    localparam int FULL_PHASE_INPUTS = 128;
    localparam int TAG_W = 48;
    localparam int TILE_BITS = 256;
    localparam int COUNT_W = 9;

    logic clk_core, rst_core;
    logic in_valid, in_ready;
    logic [TAG_W-1:0] in_tag;
    logic [TILE_BITS-1:0] in_target_bits;
    logic [TILE_BITS-1:0] in_left_bits;
    logic [TILE_BITS-1:0] in_up_bits;
    logic [TILE_BITS-1:0] in_previous_bits;
    logic in_left_valid, in_up_valid, in_previous_valid;
    logic out_valid, out_ready;
    logic [TAG_W-1:0] out_tag;
    logic [1:0] out_parent_id;
    logic [TILE_BITS-1:0] out_add_bits, out_subtract_bits;
    logic [COUNT_W-1:0] out_source_count;

    logic [TAG_W-1:0] expected_tag [0:TESTS-1];
    logic [1:0] expected_parent [0:TESTS-1];
    logic [TILE_BITS-1:0] expected_add [0:TESTS-1];
    logic [TILE_BITS-1:0] expected_subtract [0:TESTS-1];
    integer expected_count [0:TESTS-1];

    logic [TAG_W-1:0] pending_tag;
    logic [1:0] pending_parent;
    logic [TILE_BITS-1:0] pending_add, pending_subtract;
    integer pending_count;
    logic pending_tie_case;

    integer writes, reads, mismatches;
    integer parent_hits [0:3];
    integer source256_outputs, tie_accepts;
    integer back_to_back_input_accepts;
    integer full_throughput_cycles, max_full_throughput_run;
    integer current_full_throughput_run, pipeline_full_push_pop_cycles;
    integer output_stall_cycles, random_output_stall_cycles;
    integer max_outstanding, sustained_valid_low_cycles;
    integer random_ready_decisions;
    logic previous_input_accept, last_input_accept;
    logic campaign_active, throughput_phase, random_backpressure_phase;
    logic forced_tie_input;
    logic [31:0] stimulus_rng, backpressure_rng;

    qfit_adaptive_parent_selector_p256 dut (.*);

    qfit_adaptive_parent_selector_p256_sustained_assertions_r2 r2_sva (
        .clk_core,
        .rst_core,
        .in_valid,
        .in_ready,
        .in_tag,
        .in_target_bits,
        .in_left_bits,
        .in_up_bits,
        .in_previous_bits,
        .in_left_valid,
        .in_up_valid,
        .in_previous_valid,
        .out_valid,
        .out_ready,
        .out_tag,
        .out_parent_id,
        .out_add_bits,
        .out_subtract_bits,
        .out_source_count,
        .s0_valid(dut.s0_valid_q),
        .s1_valid(dut.s1_valid_q),
        .throughput_phase,
        .random_backpressure_phase,
        .forced_tie_input
    );

    always #1.5 clk_core = ~clk_core;

    function automatic integer popcount_reference(
        input logic [TILE_BITS-1:0] value
    );
        integer result;
        begin
            result = 0;
            for (int bit_index = 0; bit_index < TILE_BITS; bit_index++)
                result = result + (value[bit_index] ? 1 : 0);
            popcount_reference = result;
        end
    endfunction

    function automatic logic [31:0] xorshift32(
        input logic [31:0] value
    );
        logic [31:0] next_value;
        begin
            next_value = value ^ (value << 13);
            next_value = next_value ^ (next_value >> 17);
            next_value = next_value ^ (next_value << 5);
            xorshift32 = next_value;
        end
    endfunction

    task automatic compute_pending_oracle;
        integer best_count, candidate_count;
        logic [1:0] best_parent;
        logic [TILE_BITS-1:0] best_bits;
        begin
            best_parent = 2'd0;
            best_bits = '0;
            best_count = popcount_reference(in_target_bits);
            if (in_left_valid) begin
                candidate_count = popcount_reference(
                    in_target_bits ^ in_left_bits);
                if (candidate_count < best_count) begin
                    best_count = candidate_count;
                    best_parent = 2'd1;
                    best_bits = in_left_bits;
                end
            end
            if (in_up_valid) begin
                candidate_count = popcount_reference(
                    in_target_bits ^ in_up_bits);
                if (candidate_count < best_count) begin
                    best_count = candidate_count;
                    best_parent = 2'd2;
                    best_bits = in_up_bits;
                end
            end
            if (in_previous_valid) begin
                candidate_count = popcount_reference(
                    in_target_bits ^ in_previous_bits);
                if (candidate_count < best_count) begin
                    best_count = candidate_count;
                    best_parent = 2'd3;
                    best_bits = in_previous_bits;
                end
            end
            pending_tag = in_tag;
            pending_parent = best_parent;
            pending_add = in_target_bits & ~best_bits;
            pending_subtract = best_bits & ~in_target_bits;
            pending_count = best_count;
        end
    endtask

    task automatic build_input(input integer test_index);
        begin
            for (int word = 0; word < 8; word++) begin
                stimulus_rng = xorshift32(stimulus_rng);
                in_target_bits[word*32 +: 32] = stimulus_rng;
                stimulus_rng = xorshift32(stimulus_rng);
                in_left_bits[word*32 +: 32] = stimulus_rng;
                stimulus_rng = xorshift32(stimulus_rng);
                in_up_bits[word*32 +: 32] = stimulus_rng;
                stimulus_rng = xorshift32(stimulus_rng);
                in_previous_bits[word*32 +: 32] = stimulus_rng;
            end
            in_left_valid = (test_index % 7) != 0;
            in_up_valid = (test_index % 11) != 0;
            in_previous_valid = (test_index % 5) != 0;
            forced_tie_input = 1'b0;
            pending_tie_case = 1'b0;
            case (test_index)
                0: begin
                    // Force the legal nine-bit maximum source count: 256.
                    in_target_bits = '1;
                    in_left_bits = '0;
                    in_up_bits = '0;
                    in_previous_bits = '0;
                    in_left_valid = 1'b0;
                    in_up_valid = 1'b0;
                    in_previous_valid = 1'b0;
                end
                1: begin
                    in_target_bits = '0;
                    in_left_bits = '1;
                    in_up_bits = '1;
                    in_previous_bits = '1;
                    in_left_valid = 1'b1;
                    in_up_valid = 1'b1;
                    in_previous_valid = 1'b1;
                end
                2: begin
                    in_target_bits = {8{32'ha5c3_691e}};
                    in_left_bits = in_target_bits;
                    in_up_bits = ~in_target_bits;
                    in_previous_bits = {8{32'h3c96_a55a}};
                    in_left_valid = 1'b1;
                    in_up_valid = 1'b1;
                    in_previous_valid = 1'b1;
                end
                3: begin
                    in_target_bits = {8{32'h5a3c_96e1}};
                    in_left_bits = ~in_target_bits;
                    in_up_bits = in_target_bits;
                    in_previous_bits = {8{32'hc369_5aa5}};
                    in_left_valid = 1'b1;
                    in_up_valid = 1'b1;
                    in_previous_valid = 1'b1;
                end
                4: begin
                    in_target_bits = {8{32'h0ff0_c33c}};
                    in_left_bits = ~in_target_bits;
                    in_up_bits = {8{32'hf00f_3cc3}};
                    in_previous_bits = in_target_bits;
                    in_left_valid = 1'b1;
                    in_up_valid = 1'b1;
                    in_previous_valid = 1'b1;
                end
                5: begin
                    // Four-way zero-distance tie must select zero.
                    in_target_bits = '0;
                    in_left_bits = '0;
                    in_up_bits = '0;
                    in_previous_bits = '0;
                    in_left_valid = 1'b1;
                    in_up_valid = 1'b1;
                    in_previous_valid = 1'b1;
                    forced_tie_input = 1'b1;
                    pending_tie_case = 1'b1;
                end
                6: begin
                    // Nonzero three-way tie must select left.
                    in_target_bits = {8{32'h9696_3cc3}};
                    in_left_bits = in_target_bits;
                    in_up_bits = in_target_bits;
                    in_previous_bits = in_target_bits;
                    in_left_valid = 1'b1;
                    in_up_valid = 1'b1;
                    in_previous_valid = 1'b1;
                    forced_tie_input = 1'b1;
                    pending_tie_case = 1'b1;
                end
                7: begin
                    // Up/previous tie with left disabled must select up.
                    in_target_bits = {8{32'h69c3_a55a}};
                    in_left_bits = ~in_target_bits;
                    in_up_bits = in_target_bits;
                    in_previous_bits = in_target_bits;
                    in_left_valid = 1'b0;
                    in_up_valid = 1'b1;
                    in_previous_valid = 1'b1;
                    forced_tie_input = 1'b1;
                    pending_tie_case = 1'b1;
                end
                default: begin end
            endcase
            in_tag = 48'h6402_0000_0000 + test_index;
            compute_pending_oracle();
        end
    endtask

    always @(negedge clk_core) begin
        if (rst_core) begin
            out_ready = 1'b0;
            throughput_phase = 1'b0;
            random_backpressure_phase = 1'b0;
        end else if (writes < FULL_PHASE_INPUTS) begin
            out_ready = 1'b1;
            throughput_phase = 1'b1;
            random_backpressure_phase = 1'b0;
        end else if (reads < TESTS) begin
            backpressure_rng = xorshift32(backpressure_rng);
            out_ready = backpressure_rng[0] || backpressure_rng[3];
            throughput_phase = 1'b0;
            random_backpressure_phase = 1'b1;
            random_ready_decisions = random_ready_decisions + 1;
        end else begin
            out_ready = 1'b1;
            throughput_phase = 1'b0;
            random_backpressure_phase = 1'b0;
        end
    end

    always @(posedge clk_core) begin : scoreboard_and_telemetry
        logic input_accept_now, output_accept_now, full_push_pop_now;
        integer outstanding_now;
        if (rst_core) begin
            last_input_accept = 1'b0;
            previous_input_accept = 1'b0;
            current_full_throughput_run = 0;
        end else begin
            input_accept_now = in_valid && in_ready;
            output_accept_now = out_valid && out_ready;
            full_push_pop_now = input_accept_now && output_accept_now
                && dut.s0_valid_q && dut.s1_valid_q;
            last_input_accept = input_accept_now;

            if (output_accept_now) begin
                if (reads >= writes) begin
                    mismatches = mismatches + 1;
                    $fatal(1, "M64-r2 output without prior accepted input");
                end
                if (out_tag !== expected_tag[reads]
                        || out_parent_id !== expected_parent[reads]
                        || out_add_bits !== expected_add[reads]
                        || out_subtract_bits !== expected_subtract[reads]
                        || out_source_count !== expected_count[reads][8:0]) begin
                    mismatches = mismatches + 1;
                    $fatal(1, "M64-r2 oracle mismatch index=%0d tag=%0h/%0h parent=%0d/%0d count=%0d/%0d",
                           reads, out_tag, expected_tag[reads], out_parent_id,
                           expected_parent[reads], out_source_count,
                           expected_count[reads]);
                end
                parent_hits[out_parent_id] = parent_hits[out_parent_id] + 1;
                if (out_source_count == 9'd256)
                    source256_outputs = source256_outputs + 1;
                reads = reads + 1;
            end

            if (input_accept_now) begin
                if (writes >= TESTS) begin
                    mismatches = mismatches + 1;
                    $fatal(1, "M64-r2 accepted too many inputs");
                end
                expected_tag[writes] = pending_tag;
                expected_parent[writes] = pending_parent;
                expected_add[writes] = pending_add;
                expected_subtract[writes] = pending_subtract;
                expected_count[writes] = pending_count;
                if (pending_tie_case)
                    tie_accepts = tie_accepts + 1;
                writes = writes + 1;
                if (previous_input_accept)
                    back_to_back_input_accepts =
                        back_to_back_input_accepts + 1;
            end
            previous_input_accept = input_accept_now;

            if (campaign_active && !in_valid)
                sustained_valid_low_cycles = sustained_valid_low_cycles + 1;
            if (out_valid && !out_ready) begin
                output_stall_cycles = output_stall_cycles + 1;
                if (random_backpressure_phase)
                    random_output_stall_cycles =
                        random_output_stall_cycles + 1;
            end
            if (full_push_pop_now) begin
                pipeline_full_push_pop_cycles =
                    pipeline_full_push_pop_cycles + 1;
                if (throughput_phase) begin
                    full_throughput_cycles = full_throughput_cycles + 1;
                    current_full_throughput_run =
                        current_full_throughput_run + 1;
                    if (current_full_throughput_run > max_full_throughput_run)
                        max_full_throughput_run =
                            current_full_throughput_run;
                end else begin
                    current_full_throughput_run = 0;
                end
            end else begin
                current_full_throughput_run = 0;
            end
            outstanding_now = writes - reads;
            if (outstanding_now > max_outstanding)
                max_outstanding = outstanding_now;
            if (outstanding_now < 0 || outstanding_now > 2) begin
                mismatches = mismatches + 1;
                $fatal(1, "M64-r2 outstanding bound failure value=%0d",
                       outstanding_now);
            end
        end
    end

    initial begin : campaign
        clk_core = 1'b0;
        rst_core = 1'b1;
        in_valid = 1'b0;
        in_tag = '0;
        in_target_bits = '0;
        in_left_bits = '0;
        in_up_bits = '0;
        in_previous_bits = '0;
        in_left_valid = 1'b0;
        in_up_valid = 1'b0;
        in_previous_valid = 1'b0;
        out_ready = 1'b0;
        writes = 0;
        reads = 0;
        mismatches = 0;
        source256_outputs = 0;
        tie_accepts = 0;
        back_to_back_input_accepts = 0;
        full_throughput_cycles = 0;
        max_full_throughput_run = 0;
        current_full_throughput_run = 0;
        pipeline_full_push_pop_cycles = 0;
        output_stall_cycles = 0;
        random_output_stall_cycles = 0;
        max_outstanding = 0;
        sustained_valid_low_cycles = 0;
        random_ready_decisions = 0;
        previous_input_accept = 1'b0;
        last_input_accept = 1'b0;
        campaign_active = 1'b0;
        throughput_phase = 1'b0;
        random_backpressure_phase = 1'b0;
        forced_tie_input = 1'b0;
        stimulus_rng = 32'h64a5_2026;
        backpressure_rng = 32'h64b2_5e11;
        for (int parent = 0; parent < 4; parent++)
            parent_hits[parent] = 0;

        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        build_input(0);
        in_valid = 1'b1;
        campaign_active = 1'b1;

        while (writes < TESTS) begin
            @(posedge clk_core); #1;
            @(negedge clk_core);
            if (last_input_accept) begin
                if (writes < TESTS)
                    build_input(writes);
                else begin
                    in_valid = 1'b0;
                    forced_tie_input = 1'b0;
                    campaign_active = 1'b0;
                end
            end
        end
        while (reads < TESTS)
            @(posedge clk_core);
        @(negedge clk_core);

        if (writes != TESTS || reads != TESTS || mismatches != 0
                || source256_outputs < 1 || tie_accepts != 3
                || back_to_back_input_accepts < 32
                || max_full_throughput_run < 32
                || pipeline_full_push_pop_cycles < 32
                || random_output_stall_cycles < 1
                || sustained_valid_low_cycles != 0
                || max_outstanding != 2
                || parent_hits[0] == 0 || parent_hits[1] == 0
                || parent_hits[2] == 0 || parent_hits[3] == 0)
            $fatal(1, "M64-r2 terminal gate failure");
        $display("PASS M64 R2 sustained tests=%0d inputs=%0d outputs=%0d b2b_accepts=%0d full_cycles=%0d max_full_run=%0d full_push_pop=%0d source256=%0d parent_hits=%0d,%0d,%0d,%0d ties=%0d random_stalls=%0d output_stalls=%0d max_outstanding=%0d valid_low=%0d mismatches=%0d",
                 TESTS, writes, reads, back_to_back_input_accepts,
                 full_throughput_cycles, max_full_throughput_run,
                 pipeline_full_push_pop_cycles, source256_outputs,
                 parent_hits[0], parent_hits[1], parent_hits[2],
                 parent_hits[3], tie_accepts, random_output_stall_cycles,
                 output_stall_cycles, max_outstanding,
                 sustained_valid_low_cycles, mismatches);
        $finish;
    end

    initial begin : timeout
        repeat (20000) @(posedge clk_core);
        $fatal(1, "M64-r2 timeout");
    end
endmodule

`default_nettype wire
