`timescale 1ns/1ps
`default_nettype none

module tb_delta_bounded_classifier;

    localparam int LANES = 32;
    localparam int WAYS = 4;
    localparam int TAG_W = 16;
    localparam int PAYLOAD_W = 64;
    localparam int LANE_ID_W = 5;
    localparam int COUNT_W = 6;

    logic clk_core;
    logic rst_core;
    logic in_valid;
    logic in_ready;
    logic [TAG_W-1:0] in_tag;
    logic [LANES-1:0] in_delta_mask;
    logic [PAYLOAD_W-1:0] in_payload;
    logic out_valid;
    logic out_ready;
    logic [TAG_W-1:0] out_tag;
    logic [1:0] out_kind;
    logic [LANES-1:0] out_delta_mask;
    logic [PAYLOAD_W-1:0] out_payload;
    logic [COUNT_W-1:0] out_count;
    logic [WAYS-1:0] out_lane_valid;
    logic [(WAYS*LANE_ID_W)-1:0] out_lane_ids;

    int unsigned transaction_count;

    delta_bounded_classifier #(
        .TAG_W(TAG_W),
        .PAYLOAD_W(PAYLOAD_W)
    ) dut (.*);

    always #5 clk_core <= ~clk_core;

    function automatic int popcount32(input logic [31:0] value);
        int count;
        count = 0;
        for (int lane = 0; lane < 32; lane = lane + 1) begin
            count += int'(value[lane]);
        end
        return count;
    endfunction

    function automatic logic [31:0] low_bits(input int count);
        if (count == 32) begin
            return 32'hffff_ffff;
        end
        if (count == 0) begin
            return 32'h0000_0000;
        end
        return (32'h0000_0001 << count) - 32'h0000_0001;
    endfunction

    task automatic check_output(
        input logic [31:0] expected_mask,
        input logic [15:0] expected_tag,
        input logic [63:0] expected_payload
    );
        int expected_count;
        int selected;
        expected_count = popcount32(expected_mask);
        if (!out_valid) begin
            $fatal(1, "output transaction missing");
        end
        if (out_tag !== expected_tag ||
            out_payload !== expected_payload ||
            out_delta_mask !== expected_mask ||
            out_count !== COUNT_W'(expected_count)) begin
            $fatal(1, "payload/tag/mask/count mismatch");
        end
        if (expected_count == 0) begin
            if (out_kind !== 2'd0 || out_lane_valid !== '0) begin
                $fatal(1, "zero classification mismatch");
            end
        end else if (expected_count <= WAYS) begin
            if (out_kind !== 2'd1 ||
                $countones(out_lane_valid) != expected_count) begin
                $fatal(1, "sparse classification mismatch");
            end
            selected = 0;
            for (int lane = 0; lane < LANES; lane = lane + 1) begin
                if (expected_mask[lane]) begin
                    if (!out_lane_valid[selected] ||
                        out_lane_ids[(selected*LANE_ID_W) +: LANE_ID_W]
                            !== LANE_ID_W'(lane)) begin
                        $fatal(1, "lane extraction mismatch");
                    end
                    selected += 1;
                end
            end
        end else begin
            if (out_kind !== 2'd2 || out_lane_valid !== '0) begin
                $fatal(1, "dense fallback classification mismatch");
            end
        end
    endtask

    task automatic reset_during_stall;
        @(negedge clk_core);
        in_valid = 1'b1;
        in_tag = 16'hdead;
        in_delta_mask = 32'h0000_000f;
        in_payload = 64'hfeed_face_dead_beef;
        out_ready = 1'b0;
        @(posedge clk_core);
        @(negedge clk_core);
        in_valid = 1'b0;
        check_output(
            32'h0000_000f,
            16'hdead,
            64'hfeed_face_dead_beef
        );
        rst_core = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        if (out_valid) begin
            $fatal(1, "synchronous reset did not clear stalled output");
        end
        rst_core = 1'b0;
    endtask

    task automatic send_and_check(
        input logic [31:0] mask,
        input logic [15:0] tag,
        input logic [63:0] payload,
        input int stall_cycles
    );
        @(negedge clk_core);
        in_valid = 1'b1;
        in_tag = tag;
        in_delta_mask = mask;
        in_payload = payload;
        out_ready = 1'b0;
        do begin
            @(posedge clk_core);
        end while (!in_ready);
        @(negedge clk_core);
        in_valid = 1'b0;
        check_output(mask, tag, payload);
        repeat (stall_cycles) begin
            @(posedge clk_core);
            @(negedge clk_core);
            check_output(mask, tag, payload);
        end
        out_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        out_ready = 1'b0;
        transaction_count += 1;
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        in_valid = 1'b0;
        in_tag = '0;
        in_delta_mask = '0;
        in_payload = '0;
        out_ready = 1'b0;
        transaction_count = 0;

        repeat (4) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        send_and_check(32'h0000_0000, 16'h0000, 64'h1000, 3);
        send_and_check(32'h0000_0001, 16'h0001, 64'h1001, 0);
        send_and_check(32'h8000_0025, 16'h0002, 64'h1002, 2);
        send_and_check(32'h8000_8025, 16'h0003, 64'h1003, 1);
        send_and_check(32'hffff_ffff, 16'h0004, 64'h1004, 4);

        for (int lane = 0; lane < 32; lane = lane + 1) begin
            send_and_check(
                32'h0000_0001 << lane,
                TAG_W'(lane + 32'h0000_0100),
                64'h2000 + 64'(lane),
                lane % 3
            );
        end

        for (int count = 0; count <= 32; count = count + 1) begin
            send_and_check(
                low_bits(count),
                TAG_W'(count + 32'h0000_0200),
                64'h3000 + 64'(count),
                count % 4
            );
        end

        for (int index = 0; index < 256; index = index + 1) begin
            send_and_check(
                low_bits($urandom_range(0, 4))
                    << $urandom_range(0, 27),
                TAG_W'(index + 32'h0000_0300),
                {32'h4000_0000, 16'(index), 16'($urandom)},
                $urandom_range(0, 3)
            );
        end

        for (int index = 0; index < 2000; index = index + 1) begin
            send_and_check(
                $urandom,
                TAG_W'(index + 16),
                {$urandom, $urandom},
                $urandom_range(0, 5)
            );
        end

        if (transaction_count != 2326) begin
            $fatal(1, "transaction count mismatch");
        end
        reset_during_stall();
        $display(
            "PASS: delta bounded classifier transactions=%0d W=%0d",
            transaction_count,
            WAYS
        );
        $finish;
    end

endmodule

`default_nettype wire
