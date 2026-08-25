`timescale 1ns/1ps
`default_nettype none

module tb_qfit_adaptive_parent_selector_p256;
    localparam int TESTS = 4096;
    logic clk_core, rst_core;
    logic in_valid, in_ready;
    logic [47:0] in_tag;
    logic [255:0] in_target_bits, in_left_bits, in_up_bits, in_previous_bits;
    logic in_left_valid, in_up_valid, in_previous_valid;
    logic out_valid, out_ready;
    logic [47:0] out_tag;
    logic [1:0] out_parent_id;
    logic [255:0] out_add_bits, out_subtract_bits;
    logic [8:0] out_source_count;

    logic [47:0] exp_tag [0:TESTS-1];
    logic [1:0] exp_parent [0:TESTS-1];
    logic [255:0] exp_add [0:TESTS-1];
    logic [255:0] exp_sub [0:TESTS-1];
    integer exp_count [0:TESTS-1];
    integer writes, reads, cycle_count, stall_cycles;
    integer parent_hits [0:3];
    logic [31:0] rng_state;

    qfit_adaptive_parent_selector_p256 dut (.*);
    qfit_adaptive_parent_selector_p256_assertions sva (.*);

    always #1.5 clk_core = ~clk_core;

    function automatic integer pc256(input logic [255:0] value);
        pc256 = $countones(value);
    endfunction

    function automatic logic [31:0] xorshift32(input logic [31:0] value);
        logic [31:0] next_value;
        begin
            next_value = value ^ (value << 13);
            next_value = next_value ^ (next_value >> 17);
            next_value = next_value ^ (next_value << 5);
            xorshift32 = next_value;
        end
    endfunction

    task automatic drive_one(input integer test_index);
        logic [255:0] target, left, up, previous, parent_bits;
        logic left_valid, up_valid, previous_valid;
        integer best_count, candidate_count;
        logic [1:0] best_parent;
        begin
            for (int word = 0; word < 8; word++) begin
                rng_state = xorshift32(rng_state);
                target[word*32 +: 32] = rng_state;
                rng_state = xorshift32(rng_state);
                left[word*32 +: 32] = rng_state;
                rng_state = xorshift32(rng_state);
                up[word*32 +: 32] = rng_state;
                rng_state = xorshift32(rng_state);
                previous[word*32 +: 32] = rng_state;
            end
            left_valid = (test_index % 7) != 0;
            up_valid = (test_index % 11) != 0;
            previous_valid = (test_index % 5) != 0;
            // Four deterministic cases force every parent and exercise ties.
            if (test_index == 0) begin
                target = '0; left = '1; up = '1; previous = '1;
            end else if (test_index == 1) begin
                left_valid = 1; target = left;
            end else if (test_index == 2) begin
                up_valid = 1; target = up;
            end else if (test_index == 3) begin
                previous_valid = 1; target = previous;
            end else if (test_index == 4) begin
                left_valid = 1; up_valid = 1; previous_valid = 1;
                target = left; up = left; previous = left;
            end

            best_parent = 0;
            parent_bits = '0;
            best_count = pc256(target);
            if (left_valid) begin
                candidate_count = pc256(target ^ left);
                if (candidate_count < best_count) begin
                    best_count = candidate_count; best_parent = 1;
                    parent_bits = left;
                end
            end
            if (up_valid) begin
                candidate_count = pc256(target ^ up);
                if (candidate_count < best_count) begin
                    best_count = candidate_count; best_parent = 2;
                    parent_bits = up;
                end
            end
            if (previous_valid) begin
                candidate_count = pc256(target ^ previous);
                if (candidate_count < best_count) begin
                    best_count = candidate_count; best_parent = 3;
                    parent_bits = previous;
                end
            end
            exp_tag[writes] = test_index;
            exp_parent[writes] = best_parent;
            exp_add[writes] = target & ~parent_bits;
            exp_sub[writes] = parent_bits & ~target;
            exp_count[writes] = best_count;
            writes = writes + 1;

            @(negedge clk_core);
            in_tag = test_index;
            in_target_bits = target;
            in_left_bits = left;
            in_up_bits = up;
            in_previous_bits = previous;
            in_left_valid = left_valid;
            in_up_valid = up_valid;
            in_previous_valid = previous_valid;
            in_valid = 1;
            do @(posedge clk_core); while (!in_ready);
            @(negedge clk_core);
            in_valid = 0;
        end
    endtask

    always @(negedge clk_core) begin
        if (rst_core) begin
            out_ready = 0;
        end else begin
            out_ready = ((cycle_count % 13) != 0) && ((cycle_count % 17) != 0);
            if (!out_ready) stall_cycles = stall_cycles + 1;
        end
    end

    always @(posedge clk_core) begin
        if (!rst_core) begin
            cycle_count = cycle_count + 1;
            if (out_valid && out_ready) begin
                if (reads >= writes) $fatal(1, "M64 unexpected output");
                if (out_tag !== exp_tag[reads]
                        || out_parent_id !== exp_parent[reads]
                        || out_add_bits !== exp_add[reads]
                        || out_subtract_bits !== exp_sub[reads]
                        || out_source_count !== exp_count[reads][8:0])
                    $fatal(1, "M64 mismatch index=%0d tag=%0h/%0h parent=%0d/%0d count=%0d/%0d",
                           reads, out_tag, exp_tag[reads], out_parent_id,
                           exp_parent[reads], out_source_count, exp_count[reads]);
                parent_hits[out_parent_id] = parent_hits[out_parent_id] + 1;
                reads = reads + 1;
            end
        end
    end

    initial begin
        clk_core = 0; rst_core = 1;
        in_valid = 0; in_tag = 0; in_target_bits = 0;
        in_left_bits = 0; in_up_bits = 0; in_previous_bits = 0;
        in_left_valid = 0; in_up_valid = 0; in_previous_valid = 0;
        out_ready = 0;
        writes = 0; reads = 0; cycle_count = 0; stall_cycles = 0;
        rng_state = 32'h64a5_2026;
        for (int parent = 0; parent < 4; parent++) parent_hits[parent] = 0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core); rst_core = 0;
        for (int test_index = 0; test_index < TESTS; test_index++)
            drive_one(test_index);
        while (reads != TESTS) @(posedge clk_core);
        if (parent_hits[0] == 0 || parent_hits[1] == 0
                || parent_hits[2] == 0 || parent_hits[3] == 0)
            $fatal(1, "M64 parent cover missing %0d/%0d/%0d/%0d",
                   parent_hits[0], parent_hits[1], parent_hits[2], parent_hits[3]);
        $display("PASS M64 selector tests=%0d outputs=%0d parent_hits=%0d,%0d,%0d,%0d stalls=%0d",
                 TESTS, reads, parent_hits[0], parent_hits[1], parent_hits[2],
                 parent_hits[3], stall_cycles);
        $finish;
    end
endmodule

`default_nettype wire
