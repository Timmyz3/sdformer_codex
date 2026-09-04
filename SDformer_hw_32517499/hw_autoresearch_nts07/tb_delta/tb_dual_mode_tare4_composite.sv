`timescale 1ns/1ps
`default_nettype none

module tb_dual_mode_tare4_composite;

    localparam int VECTORS = 6000;

    logic clk_core;
    logic rst_core;
    logic in_valid;
    logic in_ready;
    logic [15:0] in_tag;
    logic in_mode_local5;
    logic [31:0] in_q_anchor;
    logic [31:0] in_k_anchor;
    logic [31:0] in_q_temporal_target;
    logic [31:0] in_k_target;
    logic out_valid;
    logic out_ready;
    logic [15:0] out_tag;
    logic out_mode_local5;
    logic [1:0] out_kind;
    logic [5:0] out_update_count;
    logic [12:0] out_raw16;
    logic [8:0] out_score_q7;

    logic [128:0] vector_mem [0:VECTORS-1];
    logic input_accepted_pulse;
    logic [31:0] stall_prng_state;

    int accepted;
    int emitted;
    int motion_count;
    int local5_count;
    int cycle_count;
    int first_accept_cycle;
    int last_emit_cycle;
    int random_stall;

    dual_mode_tare4_composite_top dut (.*);

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
        logic mode_local5;
        logic [127:0] payload;
        logic [31:0] q_target;
        logic [31:0] update_mask;
        int expected_count;
        int expected_kind;
        int bias_raw;
        int direct_raw;
        int direct_q7;
        mode_local5 = vector_mem[index][128];
        payload = vector_mem[index][127:0];
        q_target = mode_local5 ? payload[127:96] : payload[63:32];
        update_mask =
            (payload[127:96] ^ q_target) |
            (payload[95:64] ^ payload[31:0]);
        expected_count = popcount32(update_mask);
        expected_kind =
            expected_count == 0 ? 0 : (expected_count <= 4 ? 1 : 2);
        bias_raw =
            mode_local5
            ? 0
            : 16 * popcount32(payload[95:64] ^ payload[31:0]);
        direct_raw =
            axnor_raw16(q_target, payload[31:0]) + bias_raw;
        direct_q7 = rne_div16(direct_raw);
        if (out_tag !== 16'(index) ||
            out_mode_local5 !== mode_local5 ||
            out_kind !== 2'(expected_kind) ||
            out_update_count !== 6'(expected_count) ||
            out_raw16 !== 13'(direct_raw) ||
            out_score_q7 !== 9'(direct_q7)) begin
            $fatal(
                1,
                "dual-mode mismatch index=%0d mode=%0d kind=%0d/%0d count=%0d/%0d raw=%0d/%0d q7=%0d/%0d",
                index,
                mode_local5,
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
            motion_count <= 0;
            local5_count <= 0;
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
                if (in_mode_local5) begin
                    local5_count <= local5_count + 1;
                end else begin
                    motion_count <= motion_count + 1;
                end
            end
            if (out_valid && out_ready) begin
                if (emitted >= accepted) begin
                    $fatal(1, "dual-mode output without accepted input");
                end
                check_output(emitted);
                emitted <= emitted + 1;
                last_emit_cycle <= cycle_count;
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
        logic mode_local5;
        logic [31:0] q_anchor;
        logic [31:0] k_anchor;
        logic [31:0] q_target;
        logic [31:0] k_target;
        logic [31:0] chosen_mask;

        rng_state = 32'hc17a4e29;
        for (int index = 0; index < VECTORS; index = index + 1) begin
            mode_local5 = index[0];
            rng_state = prng_next(rng_state);
            q_anchor = rng_state;
            rng_state = prng_next(rng_state);
            k_anchor = rng_state;
            q_target = q_anchor;
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
                    if (mode_local5) begin
                        k_target[lane] = ~k_target[lane];
                    end else begin
                        rng_state = prng_next(rng_state);
                        if (rng_state[0]) begin
                            q_target[lane] = ~q_target[lane];
                        end else begin
                            k_target[lane] = ~k_target[lane];
                        end
                    end
                    chosen += 1;
                end
            end
            vector_mem[index] = {
                mode_local5,
                q_anchor,
                k_anchor,
                q_target,
                k_target
            };
        end

        clk_core = 1'b0;
        rst_core = 1'b1;
        in_valid = 1'b0;
        in_tag = '0;
        in_mode_local5 = 1'b0;
        in_q_anchor = '0;
        in_k_anchor = '0;
        in_q_temporal_target = '0;
        in_k_target = '0;
        out_ready = 1'b0;
        source_index = 0;
        holding = 1'b0;
        random_stall = 0;
        stall_prng_state = 32'h0d51ab73;
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
                in_mode_local5 = vector_mem[source_index][128];
                in_q_anchor = vector_mem[source_index][127:96];
                in_k_anchor = vector_mem[source_index][95:64];
                in_q_temporal_target =
                    vector_mem[source_index][63:32];
                in_k_target = vector_mem[source_index][31:0];
                holding = 1'b1;
            end
            in_valid = holding;
            if (random_stall == 0) begin
                out_ready = 1'b1;
            end else begin
                stall_prng_state = prng_next(stall_prng_state);
                out_ready = stall_prng_state[0];
            end
            if (cycle_count > 120000) begin
                $fatal(1, "dual-mode source timeout");
            end
        end

        @(negedge clk_core);
        in_valid = 1'b0;
        out_ready = 1'b1;
        while (emitted < accepted) begin
            @(posedge clk_core);
            if (cycle_count > 130000) begin
                $fatal(1, "dual-mode drain timeout");
            end
        end
        @(negedge clk_core);
        if (accepted != VECTORS ||
            emitted != VECTORS ||
            motion_count != VECTORS / 2 ||
            local5_count != VECTORS / 2) begin
            $fatal(
                1,
                "dual-mode totals mismatch accepted=%0d emitted=%0d modes=%0d/%0d",
                accepted,
                emitted,
                motion_count,
                local5_count
            );
        end
        $display(
            "PASS: dual-mode TARE4 stall=%0d vectors=%0d modes=%0d/%0d active_cycles=%0d",
            random_stall,
            VECTORS,
            motion_count,
            local5_count,
            last_emit_cycle - first_accept_cycle + 1
        );
        $finish;
    end

endmodule

`default_nettype wire
