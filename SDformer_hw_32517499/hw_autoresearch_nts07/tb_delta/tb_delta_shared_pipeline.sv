`timescale 1ns/1ps
`default_nettype none

module tb_delta_shared_pipeline;

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

    wire [31:0] q_old_bits = out_payload[127:96];
    wire [31:0] k_old_bits = out_payload[95:64];
    wire [31:0] q_new_bits = out_payload[63:32];
    wire [31:0] k_new_bits = out_payload[31:0];

    int checks;

    delta_bounded_classifier #(
        .TAG_W(16),
        .PAYLOAD_W(128)
    ) classifier (.*);

    alpha_xnor_delta4 delta4 (
        .lane_valid(out_lane_valid),
        .lane_ids(out_lane_ids),
        .q_old_bits(q_old_bits),
        .k_old_bits(k_old_bits),
        .q_new_bits(q_new_bits),
        .k_new_bits(k_new_bits),
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

    task automatic run_case(
        input logic [31:0] q0,
        input logic [31:0] k0,
        input logic [31:0] q1,
        input logic [31:0] k1,
        input logic [15:0] tag
    );
        logic [31:0] mask;
        int count;
        int anchor_raw;
        int direct_raw;
        int residual_raw;
        int signed delta_int;
        mask = (q0 ^ q1) | (k0 ^ k1);
        count = $countones(mask);
        anchor_raw = axnor_raw16(q0, k0);
        direct_raw = axnor_raw16(q1, k1);

        @(negedge clk_core);
        in_valid = 1'b1;
        in_tag = tag;
        in_delta_mask = mask;
        in_payload = {q0, k0, q1, k1};
        out_ready = 1'b0;
        do begin
            @(posedge clk_core);
        end while (!in_ready);
        @(negedge clk_core);
        in_valid = 1'b0;

        if (!out_valid ||
            out_tag !== tag ||
            out_delta_mask !== mask ||
            out_count !== 6'(count)) begin
            $fatal(1, "shared pipeline classifier mismatch");
        end

        if (count == 0) begin
            if (out_kind !== 2'd0 ||
                delta_raw16 !== 10'sd0 ||
                anchor_raw != direct_raw) begin
                $fatal(1, "zero-forward contract mismatch");
            end
        end else if (count <= 4) begin
            delta_int = { {22{delta_raw16[9]}}, delta_raw16 };
            residual_raw = anchor_raw + delta_int;
            if (out_kind !== 2'd1 || residual_raw != direct_raw) begin
                $fatal(
                    1,
                    "sparse residual mismatch count=%0d residual=%0d direct=%0d",
                    count,
                    residual_raw,
                    direct_raw
                );
            end
        end else begin
            if (out_kind !== 2'd2 ||
                out_lane_valid !== 4'b0000 ||
                direct_raw < 0) begin
                $fatal(1, "dense replay contract mismatch");
            end
        end

        out_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        out_ready = 1'b0;
        checks += 1;
    endtask

    initial begin
        logic [31:0] q0;
        logic [31:0] k0;
        logic [31:0] q1;
        logic [31:0] k1;
        int tag;

        clk_core = 1'b0;
        rst_core = 1'b1;
        in_valid = 1'b0;
        in_tag = '0;
        in_delta_mask = '0;
        in_payload = '0;
        out_ready = 1'b0;
        checks = 0;
        tag = 0;

        repeat (4) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        for (int trial = 0; trial < 32; trial = trial + 1) begin
            for (int count = 0; count <= 32; count = count + 1) begin
                // H67: both Q and K may change.
                q0 = $urandom;
                k0 = $urandom;
                q1 = q0;
                k1 = k0;
                for (int lane = 0; lane < count; lane = lane + 1) begin
                    if ((lane & 1) == 0) begin
                        q1[lane] = ~q1[lane];
                    end else begin
                        k1[lane] = ~k1[lane];
                    end
                end
                run_case(q0, k0, q1, k1, 16'(tag));
                tag += 1;

                // Local5: Q is stationary; only Kself -> Kneighbor changes.
                q0 = $urandom;
                k0 = $urandom;
                q1 = q0;
                k1 = k0;
                for (int lane = 0; lane < count; lane = lane + 1) begin
                    k1[lane] = ~k1[lane];
                end
                run_case(q0, k0, q1, k1, 16'(tag));
                tag += 1;
            end
        end

        if (checks != 2112) begin
            $fatal(1, "shared pipeline check count mismatch");
        end
        $display(
            "PASS: shared classifier+delta4 H67/Local5 checks=%0d",
            checks
        );
        $finish;
    end

endmodule

`default_nettype wire
