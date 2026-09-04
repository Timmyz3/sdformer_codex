`timescale 1ns/1ps
`default_nettype none

module tb_delta_shared_real_trace;

    localparam int VECTORS = 3645;

    logic clk_core;
    logic rst_core;
    logic in_valid;
    logic in_ready;
    logic [15:0] in_tag;
    logic [31:0] in_delta_mask;
    logic [127:0] in_payload;
    logic out_valid;
    logic out_ready;
    logic [15:0] out_tag;
    logic [1:0] out_kind;
    logic [31:0] out_delta_mask;
    logic [127:0] out_payload;
    logic [5:0] out_count;
    logic [3:0] out_lane_valid;
    logic [19:0] out_lane_ids;
    logic signed [9:0] delta_raw16;

    logic [127:0] payload_mem [0:VECTORS-1];
    logic [17:0] expected_mem [0:VECTORS-1];

    wire [31:0] q0 = out_payload[127:96];
    wire [31:0] k0 = out_payload[95:64];
    wire [31:0] q1 = out_payload[63:32];
    wire [31:0] k1 = out_payload[31:0];

    int zero_count;
    int sparse_count;
    int dense_count;

    delta_bounded_classifier #(
        .TAG_W(16),
        .PAYLOAD_W(128)
    ) classifier (.*);

    alpha_xnor_delta4 delta4 (
        .lane_valid(out_lane_valid),
        .lane_ids(out_lane_ids),
        .q_old_bits(q0),
        .k_old_bits(k0),
        .q_new_bits(q1),
        .k_new_bits(k1),
        .delta_raw16(delta_raw16)
    );

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

    task automatic run_vector(input int index);
        logic [127:0] payload;
        logic [31:0] mask;
        logic [1:0] expected_kind;
        logic [5:0] expected_count;
        logic signed [9:0] expected_delta;
        int motion;
        int anchor_raw;
        int direct_raw;
        int delta_int;
        payload = payload_mem[index];
        mask = (
            payload[127:96] ^ payload[63:32]
        ) | (
            payload[95:64] ^ payload[31:0]
        );
        expected_kind = expected_mem[index][17:16];
        expected_count = expected_mem[index][15:10];
        expected_delta = expected_mem[index][9:0];

        @(negedge clk_core);
        in_valid = 1'b1;
        in_tag = 16'(index);
        in_delta_mask = mask;
        in_payload = payload;
        out_ready = 1'b0;
        do begin
            @(posedge clk_core);
        end while (!in_ready);
        @(negedge clk_core);
        in_valid = 1'b0;

        repeat (index % 4) begin
            @(posedge clk_core);
            @(negedge clk_core);
        end
        if (!out_valid ||
            out_tag !== 16'(index) ||
            out_delta_mask !== mask ||
            out_kind !== expected_kind ||
            out_count !== expected_count) begin
            $fatal(1, "real trace classifier mismatch index=%0d", index);
        end

        motion = popcount32(k0 ^ k1);
        anchor_raw = axnor_raw16(q0, k0) + 16 * motion;
        direct_raw = axnor_raw16(q1, k1) + 16 * motion;
        if (expected_kind == 2'd0) begin
            if (delta_raw16 !== 10'sd0 || anchor_raw != direct_raw) begin
                $fatal(1, "real trace zero-forward mismatch index=%0d", index);
            end
            zero_count += 1;
        end else if (expected_kind == 2'd1) begin
            delta_int = { {22{delta_raw16[9]}}, delta_raw16 };
            if (delta_raw16 !== expected_delta ||
                anchor_raw + delta_int != direct_raw) begin
                $fatal(1, "real trace sparse residual mismatch index=%0d", index);
            end
            sparse_count += 1;
        end else begin
            if (expected_kind !== 2'd2 || out_lane_valid !== 4'b0000) begin
                $fatal(1, "real trace dense replay mismatch index=%0d", index);
            end
            dense_count += 1;
        end

        out_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        out_ready = 1'b0;
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        in_valid = 1'b0;
        in_tag = '0;
        in_delta_mask = '0;
        in_payload = '0;
        out_ready = 1'b0;
        zero_count = 0;
        sparse_count = 0;
        dense_count = 0;

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

        for (int index = 0; index < VECTORS; index = index + 1) begin
            run_vector(index);
        end

        if (zero_count != 2168 ||
            sparse_count != 914 ||
            dense_count != 563) begin
            $fatal(
                1,
                "real trace kind totals mismatch zero=%0d sparse=%0d dense=%0d",
                zero_count,
                sparse_count,
                dense_count
            );
        end
        $display(
            "PASS: H67 real trace vectors=%0d zero=%0d sparse=%0d dense=%0d",
            VECTORS,
            zero_count,
            sparse_count,
            dense_count
        );
        $finish;
    end

endmodule

`default_nettype wire
