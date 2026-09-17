// T15 公平基线：FX 全精度门核（字串行，1 词/拍，10 路并行 16×24 MAC + 比较）。
// 同常数（a/pn 外置烘焙，thr 外置 640b 行）、同吞吐口径（1 判决组输入单元/拍）。
// 数据通路：第 s 拍输入 Y[s]（signed24），V[t] += A[t,s]·Y[s]，10 单元并行；
// 10 词齐后与 thr 比较输出。乘法器保留（FX 基线的诚实面积）。
module fx_gate_synth (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_sop,
    input  wire [11:0] in_h,
    input  wire        in_word_valid,
    input  wire signed [23:0] in_word,   // Y[s] signed24
    input  wire [639:0] in_thr_row,
    output reg  [9:0]  out_dec,
    output wire        out_valid
);

    reg signed [15:0] a_mem  [0:99];
    reg signed [47:0] pn_mem [0:19];

    reg [11:0] h_reg;
    reg [3:0]  s_cnt;
    reg signed [47:0] vacc [0:9];
    reg [63:0] tauw [0:9];
    reg running;

    integer t, s;
    wire signed [47:0] pprod [0:9];
    wire signed [47:0] vacc_n [0:9];
    wire signed [47:0] thrw [0:9];
    wire [9:0] dec_w;
    genvar gt;
    generate
        for (gt = 0; gt < 10; gt = gt + 1) begin : g_cmp
            assign thrw[gt]   = $signed(tauw[gt][47:0]);
            assign pprod[gt]  = $signed({{32{a_mem[gt*10+s_cnt][15]}}, a_mem[gt*10+s_cnt]})
                              * $signed({{24{in_word[23]}}, in_word});
            assign vacc_n[gt] = vacc[gt] + pprod[gt];
            assign dec_w[gt]  = tauw[gt][63] ?  (vacc_n[gt] >= thrw[gt])
                                            : ~(vacc_n[gt] >= thrw[gt]);
        end
    endgenerate
    assign out_valid = running && (s_cnt == 4'd9) && in_word_valid;

    always @(posedge clk) begin
        if (!rst_n) begin
            h_reg <= 12'd0; s_cnt <= 4'd0; running <= 1'b0;
            out_dec <= 10'd0;
            for (t = 0; t < 10; t = t + 1) begin
                vacc[t] <= 48'sd0; tauw[t] <= 64'd0;
            end
        end else if (in_sop) begin
            h_reg <= in_h;
            s_cnt <= 4'd0;
            running <= 1'b1;
            for (t = 0; t < 10; t = t + 1) begin
                vacc[t] <= 48'sd0;
                tauw[t] <= in_thr_row[t*64 +: 64];
            end
        end else if (in_word_valid && running) begin
            for (t = 0; t < 10; t = t + 1)
                vacc[t] <= vacc_n[t];
            if (s_cnt == 4'd9) begin
                out_dec <= dec_w;
                running <= 1'b0;
            end
            s_cnt <= s_cnt + 4'd1;
        end
    end

    initial begin
        $readmemh("t15_synth/s10_stage0/a.hex", a_mem);
        $readmemh("t15_synth/s10_stage0/pn.hex", pn_mem);
    end
endmodule
