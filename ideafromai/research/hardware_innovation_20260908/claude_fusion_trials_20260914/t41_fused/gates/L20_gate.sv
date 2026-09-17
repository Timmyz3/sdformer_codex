// T41 生成文件（勿手改）：融合基线门 —— 生产者算完 A·Y 后只发 10 个判决位。
// A 来源：results/t21b_k4_gate_params.npz: L20_A（t21f k=4 微调，与 T25 k4 门核同源）。
// 与 C1 的差别：C1 把 Y 位平面串行送出、消费者逐平面维护区间证书；本模块把
// A_q·Y 一次算完并直接比较，线上只有 10 位。判决与证书递归终态恒等。
module t41_fused_gate (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         in_valid,
    input  wire [239:0] in_yv,        // 10 lane x 24b signed，切片 [24*c +: 24]
    input  wire [639:0] in_thr_row,   // 10 lane x 64b，与 T25/T15 同端口口径
    output reg  [9:0]   out_dec,
    output reg          out_valid
);

    wire signed [23:0] yv [0:9];
    wire signed [47:0] thrw [0:9];
    wire signed [47:0] V [0:9];

    genvar g;
    generate
        for (g = 0; g < 10; g = g + 1) begin : g_lane
            assign yv[g]   = $signed(in_yv[24*g +: 24]);
            assign thrw[g] = $signed(in_thr_row[64*g +: 48]);
        end
    endgenerate

    assign V[0] = (yv[0] * 48'sd1699) + (yv[6] * -48'sd1059) + (yv[7] * -48'sd1277) + (yv[8] * 48'sd1006);
    assign V[1] = (yv[3] * 48'sd1294) + (yv[4] * 48'sd1096) + (yv[7] * 48'sd1083) + (yv[8] * 48'sd1498);
    assign V[2] = (yv[0] * 48'sd1132) + (yv[1] * 48'sd1186) + (yv[3] * -48'sd1150) + (yv[7] * -48'sd1150);
    assign V[3] = (yv[0] * 48'sd873) + (yv[2] * 48'sd851) + (yv[5] * 48'sd1127) + (yv[9] * 48'sd1010);
    assign V[4] = (yv[1] * -48'sd900) + (yv[3] * 48'sd1040) + (yv[7] * 48'sd1255) + (yv[9] * 48'sd870);
    assign V[5] = (yv[0] * -48'sd1439) + (yv[3] * -48'sd1526) + (yv[4] * 48'sd631) + (yv[5] * -48'sd992);
    assign V[6] = (yv[1] * 48'sd1514) + (yv[2] * 48'sd1371) + (yv[5] * 48'sd1146) + (yv[9] * 48'sd1447);
    assign V[7] = (yv[3] * -48'sd434) + (yv[6] * -48'sd1183) + (yv[7] * 48'sd1252) + (yv[8] * 48'sd682);
    assign V[8] = (yv[1] * -48'sd1109) + (yv[3] * 48'sd1465) + (yv[4] * -48'sd1480) + (yv[7] * 48'sd1373);
    assign V[9] = (yv[2] * -48'sd903) + (yv[4] * -48'sd1293) + (yv[7] * -48'sd832) + (yv[9] * -48'sd993);

    integer t;
    always @(posedge clk) begin
        if (!rst_n) begin
            out_dec   <= 10'd0;
            out_valid <= 1'b0;
        end else begin
            out_valid <= in_valid;
            for (t = 0; t < 10; t = t + 1)
                out_dec[t] <= (V[t] >= thrw[t]);
        end
    end
endmodule
