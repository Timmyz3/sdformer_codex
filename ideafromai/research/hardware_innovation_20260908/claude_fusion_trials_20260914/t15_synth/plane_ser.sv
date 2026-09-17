// T15 生产者端 RTL：plane_ser —— 组块浮点平面串行器（自有代码）。
// 功能：锁存 10 个 signed24 词 Y[s]，sop 拍输出符号字+组指数，随后按消费者请求
// 逐平面移出（bit e-1 … bit 0，每平面 10bit，bit s = Y[s] 二补码表示的对应位）。
// 编码口径与 T5 激励生成器一致（t5_c1_cert_core_model.py）：
//   e = max_s bit_length(|Y[s]|)（符号偏离 OR 树优先编码：dev = Y ^ signext(Y[23])）；
//   平面 = Y 的二补码原始位（负数不是幅值位——供数侧低位 TC 位 + 符号字）。
// 与门核 cert_gate_bitl_synth 的供数协议对齐（T13b plane_ser 部署口径的 RTL 化）。
// 验证：由 stim_fx.txt 重建 Yq 喂入，输出逐行复现 stim_bf.txt（verify_plane.py）。
module plane_ser (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_sop,
    input  wire [239:0] in_ywords,   // {Y[9],…,Y[0]}，各 signed24
    output reg  [9:0]  out_sign,
    output reg  [4:0]  out_e,
    output reg         out_sop_valid,
    input  wire        in_plane_req,
    output reg         out_plane_valid,
    output reg  [9:0]  out_plane,
    output reg         out_done
);

    reg [23:0] y   [0:9];
    reg [4:0]  cur;
    reg        active;

    integer s;
    genvar gi;

    // e 的 OR 树用幅值（负数 m 的 XOR 偏离 = m−1，幂次幅值会差 1）：
    // mag[s] = Y[s]<0 ? −Y[s] : Y[s]（24b 取负对 −2^23 也正确）
    wire [23:0] magin [0:9];
    generate
        for (gi = 0; gi < 10; gi = gi + 1) begin : g_dev
            wire [23:0] yw = in_ywords[gi*24 +: 24];
            assign magin[gi] = yw[23] ? ((~yw) + 24'd1) : yw;
        end
    endgenerate
    wire [23:0] orin = magin[0] | magin[1] | magin[2] | magin[3] | magin[4]
                     | magin[5] | magin[6] | magin[7] | magin[8] | magin[9];
    // 最高偏离位 top（全零 → 0），e = orin==0 ? 0 : top+1
    wire [4:0] top =
        orin[23] ? 5'd23 : orin[22] ? 5'd22 : orin[21] ? 5'd21 :
        orin[20] ? 5'd20 : orin[19] ? 5'd19 : orin[18] ? 5'd18 :
        orin[17] ? 5'd17 : orin[16] ? 5'd16 : orin[15] ? 5'd15 :
        orin[14] ? 5'd14 : orin[13] ? 5'd13 : orin[12] ? 5'd12 :
        orin[11] ? 5'd11 : orin[10] ? 5'd10 : orin[9]  ? 5'd9  :
        orin[8]  ? 5'd8  : orin[7]  ? 5'd7  : orin[6]  ? 5'd6  :
        orin[5]  ? 5'd5  : orin[4]  ? 5'd4  : orin[3]  ? 5'd3  :
        orin[2]  ? 5'd2  : orin[1]  ? 5'd1  : 5'd0;
    wire [4:0] e_next = (orin == 24'd0) ? 5'd0 : (top + 5'd1);

    always @(posedge clk) begin
        if (!rst_n) begin
            out_sign <= 10'd0; out_e <= 5'd0; out_sop_valid <= 1'b0;
            out_plane_valid <= 1'b0; out_plane <= 10'd0; out_done <= 1'b0;
            cur <= 5'd0; active <= 1'b0;
            for (s = 0; s < 10; s = s + 1) y[s] <= 24'd0;
        end else begin
            out_sop_valid   <= 1'b0;
            out_plane_valid <= 1'b0;
            out_done        <= 1'b0;
            if (in_sop) begin
                for (s = 0; s < 10; s = s + 1)
                    y[s] <= in_ywords[s*24 +: 24];
                out_sign <= {in_ywords[9*24+23], in_ywords[8*24+23], in_ywords[7*24+23],
                             in_ywords[6*24+23], in_ywords[5*24+23], in_ywords[4*24+23],
                             in_ywords[3*24+23], in_ywords[2*24+23], in_ywords[1*24+23],
                             in_ywords[0*24+23]};
                out_e <= e_next;
                out_sop_valid <= 1'b1;
                // 平面 MSB-first：cur 从 e-1 递减到 0（e=0 无平面，直接 done）
                cur <= (e_next == 5'd0) ? 5'd0 : (e_next - 5'd1);
                out_done <= (e_next == 5'd0);
                active <= 1'b1;
            end else if (active) begin
                if (in_plane_req && cur < out_e) begin
                    out_plane <= {y[9][cur], y[8][cur], y[7][cur], y[6][cur],
                                  y[5][cur], y[4][cur], y[3][cur], y[2][cur],
                                  y[1][cur], y[0][cur]};
                    out_plane_valid <= 1'b1;
                    if (cur == 5'd0)
                        out_done <= 1'b1;
                    cur <= cur - 5'd1;
                end
            end
        end
    end
endmodule
