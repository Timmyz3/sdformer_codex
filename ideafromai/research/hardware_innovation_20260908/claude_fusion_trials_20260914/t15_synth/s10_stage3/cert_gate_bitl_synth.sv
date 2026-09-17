// T15 综合版：cert_gate_bitl.sv 的面积模型变体（自有代码，派生自 T10）。
// 与功能版差异（仅计费口径，不改变数据通路）：
// 1. tau0..tau9 ROM（10×3072×64b）外置为 640b 行输入 thr_row（T13a：层级静态
//    常数 SRAM，stage3 ~2Mb 宏），sop 拍随 h 锁存；
// 2. a/pn/lut 常数烘焙（$readmemh 固定路径 + initial 常数计算）。
module cert_gate_bitl_synth (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_sop,
    input  wire [11:0] in_h,
    input  wire [9:0]  in_sign,
    input  wire [4:0]  in_e,
    input  wire        in_plane_valid,
    input  wire [9:0]  in_plane,
    input  wire [639:0] in_thr_row,   // 10×64b thr 行（外置 SRAM 口）
    output wire [9:0]  out_dec,
    output wire [9:0]  dbg_locked
);

    reg signed [15:0] a_mem  [0:99];
    reg signed [47:0] pn_mem [0:19];
    reg signed [19:0] lut_lo [0:319];
    reg signed [19:0] lut_hi [0:319];

    reg [11:0] h_reg;
    reg [4:0]  m_chk;
    reg signed [47:0] vtop  [0:9];
    reg signed [47:0] pmsk  [0:9];
    reg signed [47:0] nmsk  [0:9];
    reg [63:0] tauw   [0:9];
    reg [9:0]  locked;
    reg [9:0]  dec_raw;

    integer t, tt, cc, ss;

    function signed [47:0] lut10;
        input integer tt;
        input [9:0] w;
        begin
            lut10 = $signed({{28{lut_lo[tt*32+w[4:0]][19]}}, lut_lo[tt*32+w[4:0]]})
                  + $signed({{28{lut_hi[tt*32+w[9:5]][19]}}, lut_hi[tt*32+w[9:5]]});
        end
    endfunction

    wire [63:0]        tau_sel [0:9];
    wire signed [47:0] thrw   [0:9];
    wire signed [47:0] nv     [0:9];
    wire signed [47:0] vmin_n [0:9];
    wire signed [47:0] vmax_n [0:9];
    wire               lock_n [0:9];
    genvar gt;
    generate
        for (gt = 0; gt < 10; gt = gt + 1) begin : g_chk
            assign tau_sel[gt]    = tauw[gt];
            assign thrw[gt]       = $signed(tau_sel[gt][47:0]);
            assign nv[gt]         = (vtop[gt] << 1) + lut10(gt, in_plane);
            assign vmin_n[gt]     = (nv[gt] << m_chk) + nmsk[gt];
            assign vmax_n[gt]     = (nv[gt] << m_chk) + pmsk[gt];
            assign lock_n[gt]     = ~locked[gt]
                                  & ((vmin_n[gt] >= thrw[gt]) | (vmax_n[gt] < thrw[gt]));
        end
    endgenerate

    assign dbg_locked = locked;
    generate
        for (gt = 0; gt < 10; gt = gt + 1) begin : g_out
            assign out_dec[gt] = tauw[gt][63] ? dec_raw[gt] : ~dec_raw[gt];
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            h_reg <= 12'd0; m_chk <= 5'd0;
            locked <= 10'd0; dec_raw <= 10'd0;
            for (t = 0; t < 10; t = t + 1) begin
                vtop[t] <= 48'sd0; pmsk[t] <= 48'sd0; nmsk[t] <= 48'sd0;
                tauw[t] <= 64'd0;
            end
        end else if (in_sop) begin
            h_reg <= in_h;
            m_chk <= (in_e == 5'd0) ? 5'd0 : (in_e - 5'd1);
            locked <= 10'd0;
            dec_raw <= 10'd0;
            for (t = 0; t < 10; t = t + 1) begin
                tauw[t] <= in_thr_row[t*64 +: 64];
                vtop[t] <= -lut10(t, in_sign);
                pmsk[t] <= (pn_mem[t*2]   <<< ((in_e == 5'd0) ? 5'd0 : (in_e - 5'd1)))
                          - pn_mem[t*2];
                nmsk[t] <= (pn_mem[t*2+1] <<< ((in_e == 5'd0) ? 5'd0 : (in_e - 5'd1)))
                          - pn_mem[t*2+1];
            end
        end else if (in_plane_valid) begin
            for (t = 0; t < 10; t = t + 1) begin
                vtop[t] <= nv[t];
                if (m_chk != 5'd0) begin
                    pmsk[t] <= (pmsk[t] - pn_mem[t*2])   >>> 1;
                    nmsk[t] <= (nmsk[t] - pn_mem[t*2+1]) >>> 1;
                end
                if (lock_n[t]) begin
                    locked[t] <= 1'b1;
                    dec_raw[t] <= (vmin_n[t] >= thrw[t]);
                end
            end
            if (m_chk != 5'd0)
                m_chk <= m_chk - 5'd1;
        end
    end

    initial begin
        $readmemh("t15_synth/s10_stage3/a.hex", a_mem);
        $readmemh("t15_synth/s10_stage3/pn.hex", pn_mem);
        $readmemh("t15_synth/s10_stage3/lut_lo.hex", lut_lo);
        $readmemh("t15_synth/s10_stage3/lut_hi.hex", lut_hi);
    end
endmodule
