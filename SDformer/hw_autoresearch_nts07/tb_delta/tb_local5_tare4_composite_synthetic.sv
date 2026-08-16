`timescale 1ns/1ps
`default_nettype none

module tb_local5_tare4_composite_synthetic;

    localparam int VECTORS = 12000;

    logic clk_core;
    logic rst_core;
    logic in_valid;
    logic in_ready;
    logic [15:0] in_tag;
    logic [31:0] in_q_self;
    logic [31:0] in_k_self;
    logic [31:0] in_k_neighbor;
    logic out_valid;
    logic out_ready;
    logic [15:0] out_tag;
    logic out_mode_local5;
    logic [1:0] out_kind;
    logic [5:0] out_update_count;
    logic [12:0] out_raw16;
    logic [8:0] out_score_q7;

    logic [127:0] payload_mem [0:VECTORS-1];
    logic input_accepted_pulse;

    int accepted;
    int emitted;
    int zero_count;
    int sparse_count;
    int dense_count;
    int cycle_count;
    int first_accept_cycle;
    int last_emit_cycle;
    int random_stall;
    logic [31:0] stall_prng_state;

    local5_tare4_composite_top dut (.*);

    always #5 clk_core <= ~clk_core;

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

    function automatic int axnor_raw16(
        input logic [31:0] q_bits,
        input logic [31:0] k_bits
    );
        int result;
        result = 0;
        for (int lane = 0; lane < 32; lane = lane + 1) begin
            result += lane_score(q_bits[lane], k_bits[lane]);
        end
        return result;
    endfunction

    function automatic int popcount32(input logic [31:0] bits);
        int result;
        result = 0;
        for (int lane = 0; lane < 32; lane = lane + 1) begin
            result += int'(bits[lane]);
        end
        return result;
    endfunction

    function automatic int rne_div16(input int raw);
        int quotient;
        int remainder;
        quotient = raw / 16;
        remainder = raw % 16;
        if (remainder > 8 ||
            (remainder == 8 && ((quotient & 1) != 0))) begin
            quotient += 1;
        end
        return quotient;
    endfunction

    function automatic logic [31:0] prng_next(
        input logic [31:0] state
    );
        logic [31:0] value;
        value = state;
        value = value ^ (value << 13);
        value = value ^ (value >> 17);
        value = value ^ (value << 5);
        return value;
    endfunction

    task automatic check_output(input int index);
        logic [127:0] payload;
        logic [31:0] update_mask;
        int expected_count;
        int expected_kind;
        int direct_raw;
        int direct_q7;
        payload = payload_mem[index];
        update_mask =
            payload[95:64] ^ payload[31:0];
        expected_count = popcount32(update_mask);
        expected_kind =
            expected_count == 0 ? 0 : (expected_count <= 4 ? 1 : 2);
        if (payload[127:96] !== payload[63:32]) begin
            $fatal(1, "Local5 query changed within one stencil edge");
        end
        direct_raw = axnor_raw16(payload[127:96], payload[31:0]);
        direct_q7 = rne_div16(direct_raw);
        if (out_tag !== 16'(index) ||
            out_mode_local5 !== 1'b1 ||
            out_kind !== 2'(expected_kind) ||
            out_update_count !== 6'(expected_count) ||
            out_raw16 !== 13'(direct_raw) ||
            out_score_q7 !== 9'(direct_q7)) begin
            $fatal(
                1,
                "Local5 composite mismatch index=%0d kind=%0d/%0d count=%0d/%0d raw=%0d/%0d q7=%0d/%0d",
                index,
                out_kind,
                expected_kind,
                out_update_count,
                expected_count,
                out_raw16,
                direct_raw,
                out_score_q7,
                direct_q7
            );
        end
    endtask

    always @(posedge clk_core) begin
        if (rst_core) begin
            accepted <= 0;
            emitted <= 0;
            zero_count <= 0;
            sparse_count <= 0;
            dense_count <= 0;
            cycle_count <= 0;
            first_accept_cycle <= -1;
            last_emit_cycle <= -1;
            input_accepted_pulse <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            input_accepted_pulse <= 1'b0;
            if (in_valid && in_ready) begin
                if (first_accept_cycle < 0) begin
                    first_accept_cycle <= cycle_count;
                end
                accepted <= accepted + 1;
                input_accepted_pulse <= 1'b1;
            end
            if (out_valid && out_ready) begin
                if (emitted >= accepted) begin
                    $fatal(1, "Local5 output without accepted input");
                end
                check_output(emitted);
                emitted <= emitted + 1;
                last_emit_cycle <= cycle_count;
                case (out_kind)
                    2'd0: zero_count <= zero_count + 1;
                    2'd1: sparse_count <= sparse_count + 1;
                    2'd2: dense_count <= dense_count + 1;
                    default: $fatal(1, "illegal Local5 kind");
                endcase
            end
        end
    end

    initial begin
        int desired_count;
        int chosen;
        logic [4:0] lane;
        int source_index;
        logic holding;
        logic [31:0] rng_state;
        logic [31:0] q_anchor;
        logic [31:0] k_anchor;
        logic [31:0] k_target;
        logic [31:0] chosen_mask;

        rng_state = 32'h5a17c0de;
        for (int index = 0; index < VECTORS; index = index + 1) begin
            rng_state = prng_next(rng_state);
            q_anchor = rng_state;
            rng_state = prng_next(rng_state);
            k_anchor = rng_state;
            k_target = k_anchor;
            chosen_mask = 32'd0;
            if (index < 33) begin
                desired_count = index;
            end else begin
                rng_state = prng_next(rng_state);
                case (rng_state % 10)
                    0, 1, 2, 3, 4, 5:
                        desired_count = 0;
                    6, 7, 8: begin
                        rng_state = prng_next(rng_state);
                        desired_count = int'(rng_state % 4) + 1;
                    end
                    default: begin
                        rng_state = prng_next(rng_state);
                        desired_count = int'(rng_state % 28) + 5;
                    end
                endcase
            end
            chosen = 0;
            while (chosen < desired_count) begin
                rng_state = prng_next(rng_state);
                lane = rng_state[4:0];
                if (!chosen_mask[lane]) begin
                    chosen_mask[lane] = 1'b1;
                    k_target[lane] = ~k_target[lane];
                    chosen += 1;
                end
            end
            payload_mem[index] = {
                q_anchor,
                k_anchor,
                q_anchor,
                k_target
            };
        end

        clk_core = 1'b0;
        rst_core = 1'b1;
        in_valid = 1'b0;
        in_tag = '0;
        in_q_self = '0;
        in_k_self = '0;
        in_k_neighbor = '0;
        out_ready = 1'b0;
        source_index = 0;
        holding = 1'b0;
        random_stall = 0;
        stall_prng_state = 32'h734c91a5;
        if (!$value$plusargs("STALL=%d", random_stall)) begin
            random_stall = 0;
        end

        repeat (4) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        while (source_index < VECTORS || holding) begin
            @(negedge clk_core);
            if (holding && input_accepted_pulse) begin
                holding = 1'b0;
                source_index += 1;
            end
            if (!holding && source_index < VECTORS) begin
                in_tag = 16'(source_index);
                in_q_self = payload_mem[source_index][127:96];
                in_k_self = payload_mem[source_index][95:64];
                in_k_neighbor = payload_mem[source_index][31:0];
                holding = 1'b1;
            end
            in_valid = holding;
            if (random_stall == 0) begin
                out_ready = 1'b1;
            end else begin
                stall_prng_state = prng_next(stall_prng_state);
                out_ready = stall_prng_state[0];
            end
            if (cycle_count > 200000) begin
                $fatal(1, "Local5 source timeout");
            end
        end

        @(negedge clk_core);
        in_valid = 1'b0;
        out_ready = 1'b1;
        while (emitted < accepted) begin
            @(posedge clk_core);
            if (cycle_count > 210000) begin
                $fatal(1, "Local5 drain timeout");
            end
        end
        @(negedge clk_core);
        if (accepted != VECTORS ||
            emitted != VECTORS ||
            zero_count == 0 ||
            sparse_count == 0 ||
            dense_count == 0) begin
            $fatal(
                1,
                "Local5 totals mismatch accepted=%0d emitted=%0d kinds=%0d/%0d/%0d",
                accepted,
                emitted,
                zero_count,
                sparse_count,
                dense_count
            );
        end
        $display(
            "PASS: Local5 TARE4 composite stall=%0d vectors=%0d kinds=%0d/%0d/%0d active_cycles=%0d",
            random_stall,
            VECTORS,
            zero_count,
            sparse_count,
            dense_count,
            last_emit_cycle - first_accept_cycle + 1
        );
        $finish;
    end

endmodule

`default_nettype wire
