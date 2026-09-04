`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_hatf96_weight_coalescer;
    localparam int BANK_COUNT = 3;
    localparam int LANES = 32;
    logic clk_core = 1'b0, rst_core;
    logic req_valid, req_ready;
    logic [15:0] req_tag;
    logic [7:0] req_input_channel;
    logic [3:0] req_supertile;
    logic [2:0] bank_req_valid, bank_req_ready;
    logic [47:0] bank_req_tags;
    logic [23:0] bank_req_input_channels;
    logic [11:0] bank_req_output_tiles;
    logic [2:0] bank_rsp_valid, bank_rsp_ready;
    logic [47:0] bank_rsp_tags;
    logic [23:0] bank_rsp_input_channels;
    logic [11:0] bank_rsp_output_tiles;
    logic [(BANK_COUNT*LANES*8)-1:0] bank_rsp_weights;
    logic rsp_valid, rsp_ready;
    logic [15:0] rsp_tag;
    logic [7:0] rsp_input_channel;
    logic [3:0] rsp_supertile;
    logic [(BANK_COUNT*LANES*8)-1:0] rsp_weights;
    logic rsp_error, protocol_error;
    logic [31:0] count_requests, count_bank_requests;
    logic [31:0] count_bank_responses, count_response_stalls;

    /* verilator lint_off BLKSEQ */
    always #1 clk_core = ~clk_core;
    /* verilator lint_on BLKSEQ */

    gatestack_hatf96_weight_coalescer #(
        .BANK_COUNT(BANK_COUNT), .LANES_PER_BANK(LANES), .TAG_W(16),
        .INPUT_CH_W(8), .OUTPUT_TILE_W(4)
    ) dut (.*);

    initial begin
        rst_core = 1'b1;
        req_valid = 1'b0;
        req_tag = 16'h9630;
        req_input_channel = 8'd17;
        req_supertile = 4'd2;
        bank_req_ready = '0;
        bank_rsp_valid = '0;
        bank_rsp_tags = '0;
        bank_rsp_input_channels = '0;
        bank_rsp_output_tiles = '0;
        bank_rsp_weights = '0;
        rsp_ready = 1'b0;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        req_valid = 1'b1;
        do @(posedge clk_core); while (!req_ready);
        @(negedge clk_core);
        req_valid = 1'b0;

        for (int bank = 0; bank < BANK_COUNT; bank = bank + 1) begin
            bank_req_ready[bank] = 1'b1;
            if (!bank_req_valid[bank] ||
                bank_req_tags[(bank*16) +: 16] != 16'h9630 ||
                bank_req_input_channels[(bank*8) +: 8] != 8'd17 ||
                bank_req_output_tiles[(bank*4) +: 4] != 4'(6 + bank))
                $fatal(1, "bank request identity mismatch bank=%0d", bank);
            @(posedge clk_core);
            @(negedge clk_core);
            bank_req_ready[bank] = 1'b0;
        end
        if (count_bank_requests != 3)
            $fatal(1, "bank request count mismatch %0d", count_bank_requests);

        for (int bank = BANK_COUNT - 1; bank >= 0; bank = bank - 1) begin
            bank_rsp_tags[(bank*16) +: 16] = 16'h9630;
            bank_rsp_input_channels[(bank*8) +: 8] = 8'd17;
            bank_rsp_output_tiles[(bank*4) +: 4] = 4'(6 + bank);
            for (int lane = 0; lane < LANES; lane = lane + 1)
                bank_rsp_weights[(bank*LANES*8)+(lane*8) +: 8] =
                    8'(bank*32 + lane);
            bank_rsp_valid[bank] = 1'b1;
            do @(posedge clk_core); while (!bank_rsp_ready[bank]);
            @(negedge clk_core);
            bank_rsp_valid[bank] = 1'b0;
        end

        wait (rsp_valid);
        repeat (3) begin
            @(posedge clk_core);
            if (!rsp_valid || rsp_tag != 16'h9630 ||
                rsp_input_channel != 8'd17 || rsp_supertile != 4'd2 ||
                rsp_error)
                $fatal(1, "joined response unstable or identity mismatch");
        end
        @(negedge clk_core);
        for (int lane = 0; lane < BANK_COUNT*LANES; lane = lane + 1)
            if (rsp_weights[(lane*8) +: 8] != 8'(lane))
                $fatal(1, "joined lane mismatch lane=%0d value=%0d",
                       lane, rsp_weights[(lane*8) +: 8]);
        rsp_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        rsp_ready = 1'b0;
        if (protocol_error || count_requests != 1 ||
            count_bank_requests != 3 || count_bank_responses != 3 ||
            count_response_stalls < 3) begin
            $display("DIAG protocol_error=%0d requests=%0d bank_req=%0d bank_rsp=%0d stalls=%0d rsp_error=%0d",
                     protocol_error, count_requests, count_bank_requests,
                     count_bank_responses, count_response_stalls, rsp_error);
            $fatal(1, "coalescer counters/protocol mismatch");
        end

        req_tag = 16'h9631;
        req_input_channel = 8'd18;
        req_supertile = 4'd3;
        req_valid = 1'b1;
        do @(posedge clk_core); while (!req_ready);
        @(negedge clk_core);
        req_valid = 1'b0;
        bank_req_ready = '1;
        for (int bank = 0; bank < BANK_COUNT; bank = bank + 1) begin
            if (!bank_req_valid[bank] ||
                bank_req_tags[(bank*16) +: 16] != 16'h9631 ||
                bank_req_input_channels[(bank*8) +: 8] != 8'd18 ||
                bank_req_output_tiles[(bank*4) +: 4] != 4'(9 + bank)) begin
                $display("DIAG2 bank=%0d valid=%0d tag=%h input=%0d tile=%0d ready=%0d",
                         bank, bank_req_valid[bank],
                         bank_req_tags[(bank*16) +: 16],
                         bank_req_input_channels[(bank*8) +: 8],
                         bank_req_output_tiles[(bank*4) +: 4], req_ready);
                $fatal(1, "simultaneous bank request mismatch bank=%0d", bank);
            end
        end
        @(posedge clk_core);
        @(negedge clk_core);
        bank_req_ready = '0;

        for (int bank = 0; bank < BANK_COUNT; bank = bank + 1) begin
            bank_rsp_tags[(bank*16) +: 16] =
                bank == 1 ? 16'hdead : 16'h9631;
            bank_rsp_input_channels[(bank*8) +: 8] = 8'd18;
            bank_rsp_output_tiles[(bank*4) +: 4] = 4'(9 + bank);
            for (int lane = 0; lane < LANES; lane = lane + 1)
                bank_rsp_weights[(bank*LANES*8)+(lane*8) +: 8] =
                    8'(32'd100 + bank*32 + lane);
        end
        bank_rsp_valid = '1;
        @(posedge clk_core);
        @(negedge clk_core);
        bank_rsp_valid = '0;

        wait (rsp_valid);
        if (!rsp_error || !protocol_error || rsp_tag != 16'h9631 ||
            rsp_input_channel != 8'd18 || rsp_supertile != 4'd3)
            $fatal(1, "identity error was not atomically propagated");
        for (int lane = 0; lane < BANK_COUNT*LANES; lane = lane + 1)
            if (rsp_weights[(lane*8) +: 8] != 8'(32'd100 + lane))
                $fatal(1, "simultaneous joined lane mismatch lane=%0d", lane);
        rsp_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        rsp_ready = 1'b0;
        if (count_requests != 2 || count_bank_requests != 6 ||
            count_bank_responses != 6)
            $fatal(1, "simultaneous bank handshake count mismatch");
        $display("RESULT status=PASS banks=3 lanes=96 requests=1 bank_req=3 bank_rsp=3 stalls=%0d error=0",
                 count_response_stalls);
        $display("RESULT status=PASS simultaneous_bank_req_rsp=1 identity_error_propagated=1 requests=2 bank_req=6 bank_rsp=6");
        $finish;
    end
endmodule

`default_nettype wire
