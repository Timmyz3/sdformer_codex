`timescale 1ns/1ps
`default_nettype none

module tb_ttx_gate_quant_q17;
    logic [15:0] exp_q8;
    logic [31:0] row_sum_q8;
    logic [7:0] n_tokens;
    logic preserve_mean;
    logic [8:0] gate_q17;

    integer fd;
    integer status;
    integer count;
    integer errors;
    integer expected;
    reg [1023:0] vector_path;

    ttx_gate_quant_q17 #(
        .TOKEN_W(8)
    ) dut (
        .exp_q8(exp_q8),
        .row_sum_q8(row_sum_q8),
        .n_tokens(n_tokens),
        .preserve_mean(preserve_mean),
        .gate_q17(gate_q17)
    );

    initial begin
        if (!$value$plusargs("VECTORS=%s", vector_path)) begin
            $fatal(1, "缺少 +VECTORS=<path>");
        end
        fd = $fopen(vector_path, "r");
        if (fd == 0) begin
            $fatal(1, "无法打开参考向量文件");
        end

        count = 0;
        errors = 0;
        while (!$feof(fd)) begin
            status = $fscanf(fd, "%h %h %h %h %h\n", exp_q8, row_sum_q8, n_tokens, preserve_mean, expected);
            if (status == 5) begin
                #1;
                if (gate_q17 !== expected[8:0]) begin
                    if (errors < 10) begin
                        $display("ERROR idx=%0d exp=%0d sum=%0d n=%0d preserve=%0d got=%0d expected=%0d",
                                 count, exp_q8, row_sum_q8, n_tokens, preserve_mean, gate_q17, expected);
                    end
                    errors = errors + 1;
                end
                count = count + 1;
            end
        end
        $fclose(fd);
        if (errors != 0) begin
            $fatal(1, "Q1.7 Gate对拍失败：%0d/%0d", errors, count);
        end
        $display("PASS: Q1.7 Gate独立整数参考对拍 %0d 组", count);
        $finish;
    end
endmodule

`default_nettype wire
