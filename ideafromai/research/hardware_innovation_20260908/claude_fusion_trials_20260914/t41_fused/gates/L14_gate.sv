// T41 生成文件（勿手改）：融合基线门 —— 生产者算完 A·Y 后只发 10 个判决位。
// A 来源：results/t21b_k4_gate_params.npz: L14_A（t21f k=4 微调，与 T25 k4 门核同源）。
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

    assign V[0] = (yv[0] * -48'sd673) + (yv[4] * -48'sd810) + (yv[5] * -48'sd1249) + (yv[9] * 48'sd1370);
    assign V[1] = (yv[4] * -48'sd1266) + (yv[7] * 48'sd1371) + (yv[8] * -48'sd712) + (yv[9] * 48'sd891);
    assign V[2] = (yv[1] * 48'sd1426) + (yv[4] * 48'sd1075) + (yv[8] * 48'sd1311) + (yv[9] * -48'sd1069);
    assign V[3] = (yv[0] * -48'sd878) + (yv[4] * -48'sd1286) + (yv[7] * -48'sd694) + (yv[9] * 48'sd787);
    assign V[4] = (yv[0] * 48'sd865) + (yv[3] * -48'sd1107) + (yv[5] * -48'sd998) + (yv[6] * -48'sd1164);
    assign V[5] = (yv[0] * -48'sd1453) + (yv[3] * 48'sd1161) + (yv[5] * 48'sd1443) + (yv[7] * 48'sd1472);
    assign V[6] = (yv[0] * 48'sd1141) + (yv[1] * 48'sd1836) + (yv[2] * 48'sd1122) + (yv[4] * 48'sd1049);
    assign V[7] = (yv[0] * 48'sd879) + (yv[3] * 48'sd1361) + (yv[6] * 48'sd960) + (yv[9] * 48'sd851);
    assign V[8] = (yv[0] * 48'sd1318) + (yv[2] * -48'sd1065) + (yv[7] * 48'sd1030) + (yv[8] * 48'sd1176);
    assign V[9] = (yv[0] * -48'sd1700) + (yv[1] * -48'sd1494) + (yv[3] * 48'sd1208) + (yv[8] * -48'sd1139);

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
