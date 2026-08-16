`timescale 1ns/1ps
`default_nettype none

module tb_h67_tare4_composite_real_trace;

    localparam int VECTORS = 3645;

    logic clk_core;
    logic rst_core;
    logic in_valid;
    logic in_ready;
    logic [15:0] in_tag;
    logic [31:0] in_q0;
    logic [31:0] in_k0;
    logic [31:0] in_q1;
    logic [31:0] in_k1;
    logic out_valid;
    logic out_ready;
    logic [15:0] out_tag;
    logic out_mode_local5;
    logic [1:0] out_kind;
    logic [5:0] out_update_count;
    logic [12:0] out_raw16;
    logic [8:0] out_score_q7;

    logic [127:0] payload_mem [0:VECTORS-1];
    logic [17:0] expected_mem [0:VECTORS-1];
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

    h67_tare4_composite_top dut (.*);

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

    task automatic check_output(input int index);
        logic [127:0] payload;
        logic [1:0] expected_kind;
        logic [5:0] expected_count;
        logic [31:0] update_mask;
        int derived_count;
        int derived_kind;
        int motion;
        int direct_raw;
        int direct_q7;
        payload = payload_mem[index];
        expected_kind = expected_mem[index][17:16];
        expected_count = expected_mem[index][15:10];
        update_mask =
            (payload[127:96] ^ payload[63:32]) |
            (payload[95:64] ^ payload[31:0]);
        derived_count = popcount32(update_mask);
        derived_kind =
            derived_count == 0 ? 0 : (derived_count <= 4 ? 1 : 2);
        if (expected_count !== 6'(derived_count) ||
            expected_kind !== 2'(derived_kind)) begin
            $fatal(
                1,
                "trace metadata mismatch index=%0d kind=%0d/%0d count=%0d/%0d",
                index,
                expected_kind,
                derived_kind,
                expected_count,
                derived_count
            );
        end
        motion = popcount32(payload[95:64] ^ payload[31:0]);
        direct_raw =
            axnor_raw16(payload[63:32], payload[31:0])
            + 16 * motion;
        direct_q7 = rne_div16(direct_raw);
        if (out_tag !== 16'(index) ||
            out_mode_local5 !== 1'b0 ||
            out_kind !== expected_kind ||
            out_update_count !== expected_count ||
            out_raw16 !== 13'(direct_raw) ||
            out_score_q7 !== 9'(direct_q7)) begin
            $fatal(
                1,
                "composite mismatch index=%0d kind=%0d/%0d count=%0d/%0d raw=%0d/%0d q7=%0d/%0d",
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
                    $fatal(1, "composite output without accepted input");
                end
                check_output(emitted);
                emitted <= emitted + 1;
                last_emit_cycle <= cycle_count;
                case (out_kind)
                    2'd0: zero_count <= zero_count + 1;
                    2'd1: sparse_count <= sparse_count + 1;
                    2'd2: dense_count <= dense_count + 1;
                    default: $fatal(1, "illegal composite kind");
                endcase
            end
        end
    end

    initial begin
        int source_index;
        logic holding;

        clk_core = 1'b0;
        rst_core = 1'b1;
        in_valid = 1'b0;
        in_tag = '0;
        in_q0 = '0;
        in_k0 = '0;
        in_q1 = '0;
        in_k1 = '0;
        out_ready = 1'b0;
        source_index = 0;
        holding = 1'b0;
        random_stall = 0;
        if (!$value$plusargs("STALL=%d", random_stall)) begin
            random_stall = 0;
        end

        $readmemh(
            "results/tare4_h67_real_trace_20260726/payload128.mem",
            payload_mem
        );
        $readmemh(
            "results/tare4_h67_real_trace_20260726/expected18.mem",
            expected_mem
        );

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
                in_q0 = payload_mem[source_index][127:96];
                in_k0 = payload_mem[source_index][95:64];
                in_q1 = payload_mem[source_index][63:32];
                in_k1 = payload_mem[source_index][31:0];
                holding = 1'b1;
            end
            in_valid = holding;
            out_ready =
                random_stall == 0
                ? 1'b1
                : 1'($urandom_range(0, 1));
            if (cycle_count > 100000) begin
                $fatal(1, "composite source timeout");
            end
        end

        @(negedge clk_core);
        in_valid = 1'b0;
        out_ready = 1'b1;
        while (emitted < accepted) begin
            @(posedge clk_core);
            if (cycle_count > 110000) begin
                $fatal(1, "composite drain timeout");
            end
        end
        @(negedge clk_core);
        if (accepted != VECTORS ||
            emitted != VECTORS ||
            zero_count != 2168 ||
            sparse_count != 914 ||
            dense_count != 563) begin
            $fatal(
                1,
                "composite totals mismatch accepted=%0d emitted=%0d kinds=%0d/%0d/%0d",
                accepted,
                emitted,
                zero_count,
                sparse_count,
                dense_count
            );
        end
        $display(
            "PASS: H67 TARE4 composite stall=%0d vectors=%0d kinds=%0d/%0d/%0d active_cycles=%0d",
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
