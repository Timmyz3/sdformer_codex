`timescale 1ns/1ps
`default_nettype none

module tb_delta_bounded_classifier_stream;

    localparam int TRANSACTIONS = 3000;
    localparam int DEPTH = 4096;

    logic clk_core;
    logic rst_core;
    logic in_valid;
    logic in_ready;
    logic [15:0] in_tag;
    logic [31:0] in_delta_mask;
    logic [63:0] in_payload;
    logic out_valid;
    logic out_ready;
    logic [15:0] out_tag;
    logic [1:0] out_kind;
    logic [31:0] out_delta_mask;
    logic [63:0] out_payload;
    logic [5:0] out_count;
    logic [3:0] out_lane_valid;
    logic [19:0] out_lane_ids;

    logic [15:0] expected_tag [0:DEPTH-1];
    logic [31:0] expected_mask [0:DEPTH-1];
    logic [63:0] expected_payload [0:DEPTH-1];
    int head;
    int tail;
    int accepted;
    int emitted;
    logic input_accepted_pulse;

    delta_bounded_classifier #(
        .TAG_W(16),
        .PAYLOAD_W(64)
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
        input logic [15:0] tag,
        input logic [31:0] mask,
        input logic [63:0] payload
    );
        int count;
        int selected;
        count = popcount32(mask);
        if (out_tag !== tag ||
            out_delta_mask !== mask ||
            out_payload !== payload ||
            out_count !== 6'(count)) begin
            $fatal(1, "stream scoreboard payload mismatch");
        end
        if (count == 0) begin
            if (out_kind !== 2'd0 || out_lane_valid !== 4'b0000) begin
                $fatal(1, "stream zero kind mismatch");
            end
        end else if (count <= 4) begin
            if (out_kind !== 2'd1 ||
                $countones(out_lane_valid) != count) begin
                $fatal(1, "stream sparse kind mismatch");
            end
            selected = 0;
            for (int lane = 0; lane < 32; lane = lane + 1) begin
                if (mask[lane]) begin
                    if (out_lane_ids[(selected*5) +: 5] !== 5'(lane)) begin
                        $fatal(1, "stream lane ID mismatch");
                    end
                    selected += 1;
                end
            end
        end else if (out_kind !== 2'd2 || out_lane_valid !== 4'b0000) begin
            $fatal(1, "stream dense kind mismatch");
        end
    endtask

    always @(posedge clk_core) begin
        if (rst_core) begin
            head <= 0;
            tail <= 0;
            accepted <= 0;
            emitted <= 0;
            input_accepted_pulse <= 1'b0;
        end else begin
            input_accepted_pulse <= 1'b0;
            if (out_valid && out_ready) begin
                if (head >= tail) begin
                    $fatal(1, "output without accepted input");
                end
                check_output(
                    expected_tag[head],
                    expected_mask[head],
                    expected_payload[head]
                );
                head <= head + 1;
                emitted <= emitted + 1;
            end
            if (in_valid && in_ready) begin
                if (tail >= DEPTH) begin
                    $fatal(1, "scoreboard overflow");
                end
                expected_tag[tail] <= in_tag;
                expected_mask[tail] <= in_delta_mask;
                expected_payload[tail] <= in_payload;
                tail <= tail + 1;
                accepted <= accepted + 1;
                input_accepted_pulse <= 1'b1;
            end
        end
    end

    initial begin
        int generated;
        int cycles;
        logic holding;
        logic [31:0] next_mask;

        clk_core = 1'b0;
        rst_core = 1'b1;
        in_valid = 1'b0;
        in_tag = '0;
        in_delta_mask = '0;
        in_payload = '0;
        out_ready = 1'b0;
        generated = 0;
        cycles = 0;
        holding = 1'b0;
        next_mask = '0;

        repeat (4) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        while (generated < TRANSACTIONS || holding) begin
            @(negedge clk_core);
            cycles += 1;
            if (holding && input_accepted_pulse) begin
                holding = 1'b0;
            end
            if (!holding &&
                generated < TRANSACTIONS &&
                $urandom_range(0, 3) != 0) begin
                case (generated % 8)
                    0: next_mask = 32'h0000_0000;
                    1: next_mask =
                        32'h0000_0001 << (generated % 32);
                    2: next_mask =
                        low_bits(2) << (generated % 30);
                    3: next_mask =
                        low_bits(4) << (generated % 28);
                    4: next_mask =
                        low_bits(5) << (generated % 27);
                    5: next_mask = 32'hffff_ffff;
                    default: next_mask = $urandom;
                endcase
                in_tag = 16'(generated);
                in_delta_mask = next_mask;
                in_payload = {32'(generated), $urandom};
                holding = 1'b1;
                generated += 1;
            end
            in_valid = holding;
            out_ready = 1'($urandom_range(0, 1));
            if (cycles > 100000) begin
                $fatal(1, "stream generation timeout");
            end
        end

        @(negedge clk_core);
        in_valid = 1'b0;
        out_ready = 1'b1;
        while (emitted < accepted) begin
            @(posedge clk_core);
            if (cycles > 110000) begin
                $fatal(1, "stream drain timeout");
            end
            cycles += 1;
        end
        @(negedge clk_core);
        if (accepted != TRANSACTIONS ||
            emitted != TRANSACTIONS ||
            head != tail) begin
            $fatal(
                1,
                "stream count mismatch accepted=%0d emitted=%0d head=%0d tail=%0d",
                accepted,
                emitted,
                head,
                tail
            );
        end
        $display(
            "PASS: classifier stream scoreboard accepted=%0d emitted=%0d cycles=%0d",
            accepted,
            emitted,
            cycles
        );
        $finish;
    end

endmodule

`default_nettype wire
