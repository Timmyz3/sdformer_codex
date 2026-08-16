`timescale 1ns/1ps
`default_nettype none

module tb_alpha_xnor_delta4;

    logic [3:0] lane_valid;
    logic [19:0] lane_ids;
    logic [31:0] q_old_bits;
    logic [31:0] k_old_bits;
    logic [31:0] q_new_bits;
    logic [31:0] k_new_bits;
    logic signed [9:0] delta_raw16;

    int signed expected;
    int checks;

    alpha_xnor_delta4 dut (.*);

    function automatic int lane_score(
        input logic q_bit,
        input logic k_bit
    );
        if (q_bit && k_bit) begin
            return 64;
        end
        if (!q_bit && !k_bit) begin
            return 1;
        end
        return 0;
    endfunction

    task automatic check_current;
        logic [4:0] lane;
        logic signed [9:0] expected_width;
        expected = 0;
        for (int way = 0; way < 4; way = way + 1) begin
            if (lane_valid[way]) begin
                lane = lane_ids[(way*5) +: 5];
                expected +=
                    lane_score(q_new_bits[lane], k_new_bits[lane])
                    - lane_score(q_old_bits[lane], k_old_bits[lane]);
            end
        end
        expected_width = 10'(expected);
        #1;
        if (delta_raw16 !== expected_width) begin
            $fatal(
                1,
                "delta mismatch expected=%0d actual=%0d",
                expected,
                $signed(delta_raw16)
            );
        end
        checks += 1;
    endtask

    initial begin
        lane_valid = '0;
        lane_ids = '0;
        q_old_bits = '0;
        k_old_bits = '0;
        q_new_bits = '0;
        k_new_bits = '0;
        checks = 0;

        // Exhaust all Qold/Kold/Qnew/Knew states on one selected lane.
        lane_valid = 4'b0001;
        lane_ids[4:0] = 5'd7;
        for (int pattern = 0; pattern < 16; pattern = pattern + 1) begin
            q_old_bits = '0;
            k_old_bits = '0;
            q_new_bits = '0;
            k_new_bits = '0;
            q_old_bits[7] = pattern[3];
            k_old_bits[7] = pattern[2];
            q_new_bits[7] = pattern[1];
            k_new_bits[7] = pattern[0];
            check_current();
        end

        // Local5 contract: Q is unchanged while Kself transitions to Kneighbor.
        for (int index = 0; index < 10000; index = index + 1) begin
            q_old_bits = $urandom;
            q_new_bits = q_old_bits;
            k_old_bits = $urandom;
            k_new_bits = $urandom;
            lane_valid = 4'b1111;
            lane_ids = {
                5'($urandom_range(24, 31)),
                5'($urandom_range(16, 23)),
                5'($urandom_range(8, 15)),
                5'($urandom_range(0, 7))
            };
            check_current();
        end

        // H67 contract: Q and K can both transition between temporal slices.
        for (int index = 0; index < 10000; index = index + 1) begin
            q_old_bits = $urandom;
            k_old_bits = $urandom;
            q_new_bits = $urandom;
            k_new_bits = $urandom;
            lane_valid = 4'($urandom_range(0, 15));
            lane_ids = {
                5'($urandom_range(24, 31)),
                5'($urandom_range(16, 23)),
                5'($urandom_range(8, 15)),
                5'($urandom_range(0, 7))
            };
            check_current();
        end

        $display(
            "PASS: alpha-XNOR delta4 checks=%0d Local5+H67",
            checks
        );
        $finish;
    end

endmodule

`default_nettype wire
