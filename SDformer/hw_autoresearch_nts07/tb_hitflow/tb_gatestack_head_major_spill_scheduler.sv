`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_head_major_spill_scheduler;
    localparam int TOKENS = 6;
    localparam int HEADS = 3;
    localparam int TILES = 2;
    localparam int BANKS = 2;
    localparam int OUT_TILE = 4;
    logic clk_core, rst_core;
    logic start_valid, start_ready;
    logic [15:0] start_tag;
    logic [2:0] start_head_count;
    logic [2:0] start_output_tile_count;
    logic decode_req_valid, decode_req_ready;
    logic [15:0] decode_req_tag;
    logic [1:0] decode_req_head;
    logic decode_done_valid, decode_done_ready, decode_done_error;
    logic spill_read_valid, spill_read_ready;
    logic [1:0] spill_read_tile;
    logic [2:0] spill_read_token_base;
    logic [BANKS-1:0] spill_read_token_valid;
    logic spill_write_valid, spill_write_ready;
    logic [1:0] spill_write_tile;
    logic [2:0] spill_write_token_base;
    logic [BANKS-1:0] spill_write_token_valid;
    logic final_valid, final_ready;
    logic [1:0] final_tile;
    logic [2:0] final_token_base;
    logic [BANKS-1:0] final_token_valid;
    logic done_valid, done_ready;
    logic [15:0] done_tag;
    logic protocol_error;
    logic [31:0] count_decodes, count_spill_reads, count_spill_writes;
    logic [31:0] count_final_batches;
    logic [63:0] count_spill_value_bytes;
    integer cycles, decode_requests;
    logic done_accept_enable;

    gatestack_head_major_spill_scheduler #(
        .TOKENS(TOKENS), .MAX_HEADS(4), .MAX_OUTPUT_TILES(4),
        .BANKS(BANKS), .OUT_TILE(OUT_TILE), .TAG_W(16),
        .TOKEN_ID_W(3), .HEAD_W(2), .TILE_W(2)
    ) dut (.*);

    always #5 clk_core <= ~clk_core;
    always_comb begin
        decode_req_ready = (cycles % 3) != 1;
        spill_read_ready = (cycles % 5) != 2;
        spill_write_ready = (cycles % 4) != 1;
        final_ready = (cycles % 3) != 2;
        done_ready = done_accept_enable;
    end

    initial begin : decode_model
        decode_done_valid = 1'b0;
        decode_done_error = 1'b0;
        wait (!rst_core);
        forever begin
            do @(posedge clk_core); while (!(decode_req_valid && decode_req_ready));
            if (decode_req_tag != 16'h7711 ||
                32'(decode_req_head) != decode_requests)
                $fatal(1, "decode request order mismatch");
            decode_requests = decode_requests + 1;
            repeat (2) @(posedge clk_core);
            @(negedge clk_core);
            decode_done_valid = 1'b1;
            do @(posedge clk_core); while (!decode_done_ready);
            @(negedge clk_core);
            decode_done_valid = 1'b0;
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core)
            cycles <= 0;
        else begin
            cycles <= cycles + 1;
            if (spill_read_valid && spill_read_ready) begin
                if (spill_read_token_valid != 2'b11 ||
                    spill_read_token_base % 2 != 0 ||
                    32'(spill_read_tile) >= TILES)
                    $fatal(1, "spill read metadata mismatch");
            end
            if (spill_write_valid && spill_write_ready) begin
                if (spill_write_token_valid != 2'b11 ||
                    spill_write_token_base % 2 != 0 ||
                    32'(spill_write_tile) >= TILES)
                    $fatal(1, "spill write metadata mismatch");
            end
            if (final_valid && final_ready) begin
                if (final_token_valid != 2'b11 ||
                    final_token_base % 2 != 0 || 32'(final_tile) >= TILES)
                    $fatal(1, "final metadata mismatch");
            end
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        start_valid = 1'b0;
        start_tag = 16'h7711;
        start_head_count = 3'(HEADS);
        start_output_tile_count = 3'(TILES);
        done_accept_enable = 1'b0;
        decode_requests = 0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;
        @(negedge clk_core);
        start_valid = 1'b1;
        do @(posedge clk_core); while (!start_ready);
        @(negedge clk_core);
        start_valid = 1'b0;
        wait (done_valid);
        @(negedge clk_core);
        if (done_tag != 16'h7711 || protocol_error || decode_requests != HEADS ||
            count_decodes != HEADS || count_spill_reads != 12 ||
            count_spill_writes != 12 || count_final_batches != 6 ||
            count_spill_value_bytes != 768)
            $fatal(1, "head-major counts mismatch dec=%0d read=%0d write=%0d final=%0d bytes=%0d",
                   count_decodes, count_spill_reads, count_spill_writes,
                   count_final_batches, count_spill_value_bytes);
        done_accept_enable = 1'b1;
        @(posedge clk_core);
        $display("PASS: head-major spill scheduler decodes=%0d reads=%0d writes=%0d finals=%0d bytes=%0d cycles=%0d",
                 count_decodes, count_spill_reads, count_spill_writes,
                 count_final_batches, count_spill_value_bytes, cycles);
        $finish;
    end

    initial begin
        repeat (10000) @(posedge clk_core);
        $fatal(1, "head-major spill scheduler timeout");
    end
endmodule

`default_nettype wire
