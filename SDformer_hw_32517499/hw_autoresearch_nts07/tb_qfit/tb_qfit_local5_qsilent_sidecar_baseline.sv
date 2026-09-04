`timescale 1ns/1ps
`default_nettype none

module tb_qfit_local5_qsilent_sidecar_baseline;
    localparam int ROW_TOKENS = 15;
    logic clk = 1'b0;
    logic rst;
    logic wr_valid;
    logic [1:0] wr_row;
    logic [3:0] wr_x;
    logic [31:0] wr_k;
    logic rd_valid;
    logic rd_ready;
    logic [3:0] rd_x;
    logic stat_valid;
    logic stat_ready;
    logic [29:0] stat_pop;
    logic [4:0] stat_mask;
    logic score_valid;
    logic score_ready;
    logic [79:0] score_bus;
    logic [4:0] score_mask;
    integer checks;
    integer x_idx;
    integer row_idx;

    always #1 clk = ~clk;

    qfit_local5_qsilent_popcount_sidecar u_store (
        .clk_core(clk), .rst_core(rst),
        .write_valid(wr_valid), .write_ready(),
        .write_row_sel(wr_row), .write_x(wr_x), .write_k(wr_k),
        .write_token_valid(1'b1),
        .read_valid(rd_valid), .read_ready(rd_ready), .read_x(rd_x),
        .rsp_valid(stat_valid), .rsp_ready(stat_ready),
        .rsp_popcount(stat_pop), .rsp_valid_mask(stat_mask),
        .protocol_error()
    );

    qfit_local5_qsilent_sidecar_score_leaf u_score (
        .clk_core(clk), .rst_core(rst),
        .in_valid(stat_valid), .in_ready(stat_ready), .in_tag({12'd0, rd_x}),
        .in_popcount(stat_pop), .in_valid_mask(stat_mask),
        .out_valid(score_valid), .out_ready(score_ready), .out_tag(),
        .out_score_q7(score_bus), .out_valid_mask(score_mask)
    );

    function automatic logic [31:0] word_for(
        input integer row,
        input integer x
    );
        logic [31:0] value;
        begin
            value = 32'h9e37_79b9 ^ (32'(row) << 24) ^ (32'(x) * 32'h1021);
            word_for = value;
        end
    endfunction

    function automatic integer popcount(input logic [31:0] value);
        integer count;
        begin
            count = 0;
            for (int lane = 0; lane < 32; lane = lane + 1)
                count = count + value[lane];
            popcount = count;
        end
    endfunction

    function automatic integer qsilent_score(input logic [31:0] value);
        integer raw;
        integer quotient;
        integer remainder;
        begin
            raw = 32 - popcount(value);
            quotient = raw / 16;
            remainder = raw % 16;
            qsilent_score = quotient
                + ((remainder > 8) || ((remainder == 8) && (quotient & 1)));
        end
    endfunction

    task automatic check_x(input integer x);
        integer expected [0:4];
        logic [4:0] expected_mask;
        begin
            expected[0] = qsilent_score(word_for(1, x));
            expected[1] = qsilent_score(word_for(0, x));
            expected[2] = qsilent_score(word_for(2, x));
            expected[3] = (x + 1 < ROW_TOKENS)
                        ? qsilent_score(word_for(1, x + 1)) : -256;
            expected[4] = (x > 0)
                        ? qsilent_score(word_for(1, x - 1)) : -256;
            expected_mask = {x > 0, x + 1 < ROW_TOKENS, 1'b1, 1'b1, 1'b1};
            if (score_mask !== expected_mask)
                $fatal(1, "mask mismatch x=%0d exp=%b got=%b",
                    x, expected_mask, score_mask);
            for (int role = 0; role < 5; role = role + 1)
                if ($signed(score_bus[role*16 +: 16]) !== expected[role])
                    $fatal(1, "score mismatch x=%0d role=%0d exp=%0d got=%0d",
                        x, role, expected[role],
                        $signed(score_bus[role*16 +: 16]));
            checks = checks + 1;
        end
    endtask

    initial begin
        rst = 1'b1;
        wr_valid = 1'b0;
        wr_row = '0;
        wr_x = '0;
        wr_k = '0;
        rd_valid = 1'b0;
        rd_x = '0;
        score_ready = 1'b0;
        checks = 0;
        repeat (4) @(negedge clk);
        rst = 1'b0;

        for (row_idx = 0; row_idx < 3; row_idx = row_idx + 1) begin
            for (x_idx = 0; x_idx < ROW_TOKENS; x_idx = x_idx + 1) begin
                @(negedge clk);
                wr_valid = 1'b1;
                wr_row = 2'(row_idx);
                wr_x = 4'(x_idx);
                wr_k = word_for(row_idx, x_idx);
            end
        end
        @(negedge clk);
        wr_valid = 1'b0;

        for (x_idx = 0; x_idx < ROW_TOKENS; x_idx = x_idx + 1) begin
            rd_x = 4'(x_idx);
            rd_valid = 1'b1;
            score_ready = 1'b0;
            @(posedge clk);
            while (!rd_ready) @(posedge clk);
            @(negedge clk);
            rd_valid = 1'b0;
            repeat ((x_idx % 3) + 1) @(negedge clk);
            score_ready = 1'b1;
            @(posedge clk);
            while (!score_valid) @(posedge clk);
            check_x(x_idx);
            @(negedge clk);
            score_ready = 1'b0;
        end

        if (checks != ROW_TOKENS)
            $fatal(1, "check count mismatch %0d", checks);
        $display("SIDECAR_BASELINE rows=3 tokens=45 destinations=15 mismatch=0");
        $display("PASS tb_qfit_local5_qsilent_sidecar_baseline");
        $finish;
    end
endmodule

`default_nettype wire
