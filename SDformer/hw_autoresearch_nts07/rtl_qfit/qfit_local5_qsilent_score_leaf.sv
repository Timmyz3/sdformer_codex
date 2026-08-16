`timescale 1ns/1ps
`default_nettype none

// Exact query-silent / identical-K score bypass around the existing Local5
// score leaf. ENABLE_QSILENT=0 is a combinational pass-through.
//
// When Q==0 the AXNOR raw score collapses to 32-popcount(K). When Q!=0 and
// every valid neighborhood K word is identical, one raw16(Q,K) broadcasts
// to all valid roles. Both paths are bit-exact and skip residual XOR walking.
module qfit_local5_qsilent_score_leaf #(
    parameter bit ENABLE_QSILENT = 1'b1,
    parameter bit ENABLE_IDENTK = 1'b1,
    parameter bit ENABLE_OVERLAP = 1'b1,
    parameter bit ARCH_QFSA = 1'b1,
    parameter bit PIPE_COMPACTOR = 1'b0,
    parameter bit XBF_BANKED = 1'b0,
    parameter bit USE_THRESHOLD_ROUTE = 1'b0,
    parameter int ROUTE_THRESHOLD = 8,
    parameter bit USE_BANK_PRESSURE_ROUTE = 1'b0,
    parameter int BANK_PRESSURE_THRESHOLD = 2,
    parameter int TAG_W = 16,
    parameter int SCORE_W = 16,
    parameter int GATE_W = 9
) (
    input  logic                     clk_core,
    input  logic                     rst_core,
    input  logic                     in_valid,
    output logic                     in_ready,
    input  logic [TAG_W-1:0]         in_tag,
    input  logic [31:0]              in_q,
    input  logic [5*32-1:0]          in_k,
    input  logic [4:0]               in_valid_mask,
    output logic                     out_valid,
    input  logic                     out_ready,
    output logic [TAG_W-1:0]         out_tag,
    output logic [5*SCORE_W-1:0]     out_score_q7,
    output logic [5*GATE_W-1:0]      out_gate_q17,
    output logic [31:0]              out_k_self,
    output logic [4:0]               out_valid_mask,
    output logic [15:0]              perf_service_cycles,
    output logic [3:0]               perf_route_direct_mask,
    output logic [31:0]              perf_qsilent_rows,
    output logic [31:0]              perf_identk_rows,
    output logic [31:0]              perf_overlap_accepts
);

    typedef enum logic [1:0] {
        ST_IDLE = 2'd0,
        ST_OUT  = 2'd1
    } fast_state_t;

    logic base_in_valid;
    logic base_in_ready;
    logic base_out_valid;
    logic base_out_ready;
    logic [TAG_W-1:0] base_out_tag;
    logic [5*SCORE_W-1:0] base_out_score;
    logic [5*GATE_W-1:0] base_out_gate;
    logic [31:0] base_out_k_self;
    logic [4:0] base_out_mask;
    logic [15:0] base_service_cycles;
    logic [3:0] base_route_mask;

    logic take_qsilent;
    logic take_identk;
    logic take_fast;
    logic [31:0] ident_k_w;
    logic ident_k_valid;
    fast_state_t state_q;
    logic [TAG_W-1:0] tag_q;
    logic [31:0] k_self_q;
    logic [4:0] valid_q;
    logic signed [SCORE_W-1:0] score_q [0:4];
    logic [31:0] qsilent_rows_q;
    logic [31:0] identk_rows_q;
    logic [31:0] overlap_accepts_q;
    logic [0:0] retire_kind_q [0:1];
    logic retire_wr_q;
    logic retire_rd_q;
    logic [1:0] retire_count_q;
    logic can_push;
    logic can_accept_fast;
    logic can_accept_base;
    logic emit_fast;
    logic emit_base;
    logic issue_fire;
    logic retire_fire;
    logic retire_head;

    logic [5:0] pop_w [0:4];
    logic signed [12:0] raw_w [0:4];
    logic signed [SCORE_W-1:0] score_w [0:4];
    logic [5*SCORE_W-1:0] score_bus;
    logic [5*GATE_W-1:0] gate_bus;

    function automatic logic [5:0] popcount32(
        input logic [31:0] bits
    );
        logic [5:0] count;
        count = '0;
        for (int lane = 0; lane < 32; lane = lane + 1)
            count = count + 6'(bits[lane]);
        popcount32 = count;
    endfunction

    function automatic logic signed [SCORE_W-1:0] rne_q7(
        input logic signed [12:0] raw_value
    );
        logic [12:0] nonnegative;
        logic [8:0] quotient;
        logic [3:0] remainder;
        logic increment;
        nonnegative = raw_value[12] ? 13'd0 : raw_value;
        quotient = nonnegative[12:4];
        remainder = nonnegative[3:0];
        increment = (remainder > 4'd8)
                 || ((remainder == 4'd8) && quotient[0]);
        rne_q7 = $signed({7'b0, quotient})
               + SCORE_W'(increment);
    endfunction

    function automatic logic [12:0] raw16(
        input logic [31:0] q_bits,
        input logic [31:0] k_bits
    );
        logic [12:0] value;
        value = '0;
        for (int lane = 0; lane < 32; lane = lane + 1) begin
            if (q_bits[lane] && k_bits[lane])
                value = value + 13'd64;
            else if (!q_bits[lane] && !k_bits[lane])
                value = value + 13'd1;
        end
        raw16 = value;
    endfunction

    always_comb begin
        ident_k_valid = 1'b0;
        ident_k_w = 32'd0;
        take_identk = 1'b0;
        if (ENABLE_QSILENT && ENABLE_IDENTK
            && (in_q != 32'd0) && (in_valid_mask != 5'd0)) begin
            take_identk = 1'b1;
            for (int cand = 0; cand < 5; cand = cand + 1) begin
                if (in_valid_mask[cand]) begin
                    if (!ident_k_valid) begin
                        ident_k_w = in_k[cand*32 +: 32];
                        ident_k_valid = 1'b1;
                    end else if (in_k[cand*32 +: 32] != ident_k_w)
                        take_identk = 1'b0;
                end
            end
        end
    end

    assign take_qsilent = ENABLE_QSILENT && (in_q == 32'd0);
    assign take_fast = take_qsilent || take_identk;
    assign can_push = retire_count_q < 2'd2;
    assign can_accept_fast = ENABLE_QSILENT
                           && (state_q == ST_IDLE)
                           && can_push
                           && (ENABLE_OVERLAP
                               || ((retire_count_q == 2'd0) && base_in_ready));
    assign can_accept_base = base_in_ready
                           && can_push
                           && (ENABLE_OVERLAP
                               || ((retire_count_q == 2'd0)
                                   && (state_q == ST_IDLE)));
    assign retire_head = retire_kind_q[retire_rd_q];
    assign emit_fast = ENABLE_QSILENT
                    && (retire_count_q != 2'd0)
                    && retire_head
                    && (state_q == ST_OUT);
    assign emit_base = ENABLE_QSILENT
                    && (retire_count_q != 2'd0)
                    && !retire_head
                    && base_out_valid;
    assign issue_fire = in_valid && in_ready;
    assign retire_fire = out_valid && out_ready;
    assign base_in_valid = in_valid && in_ready && !take_fast;
    assign base_out_ready = ENABLE_QSILENT
                          ? (emit_base && out_ready)
                          : out_ready;

    qfit_local5_score_leaf #(
        .ARCH_QFSA(ARCH_QFSA),
        .PIPE_COMPACTOR(PIPE_COMPACTOR),
        .XBF_BANKED(XBF_BANKED),
        .USE_THRESHOLD_ROUTE(USE_THRESHOLD_ROUTE),
        .ROUTE_THRESHOLD(ROUTE_THRESHOLD),
        .USE_BANK_PRESSURE_ROUTE(USE_BANK_PRESSURE_ROUTE),
        .BANK_PRESSURE_THRESHOLD(BANK_PRESSURE_THRESHOLD),
        .TAG_W(TAG_W),
        .SCORE_W(SCORE_W),
        .GATE_W(GATE_W)
    ) u_base (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(base_in_valid),
        .in_ready(base_in_ready),
        .in_tag(in_tag),
        .in_q(in_q),
        .in_k(in_k),
        .in_valid_mask(in_valid_mask),
        .out_valid(base_out_valid),
        .out_ready(base_out_ready),
        .out_tag(base_out_tag),
        .out_score_q7(base_out_score),
        .out_gate_q17(base_out_gate),
        .out_k_self(base_out_k_self),
        .out_valid_mask(base_out_mask),
        .perf_service_cycles(base_service_cycles),
        .perf_route_direct_mask(base_route_mask)
    );

    always_comb begin
        for (int cand = 0; cand < 5; cand = cand + 1) begin
            pop_w[cand] = popcount32(in_k[cand*32 +: 32]);
            raw_w[cand] = take_identk
                        ? raw16(in_q, ident_k_w)
                        : (13'sd32 - $signed({7'b0, pop_w[cand]}));
            score_w[cand] = in_valid_mask[cand]
                          ? rne_q7(raw_w[cand])
                          : SCORE_W'(-256);
            score_bus[cand*SCORE_W +: SCORE_W] = score_q[cand];
        end
    end

    local5_shiftmax5_q17 #(
        .N_CAND(5),
        .SCORE_W(SCORE_W),
        .GATE_W(GATE_W)
    ) u_shiftmax (
        .score_q7(score_bus),
        .valid(valid_q),
        .gate_q17(gate_bus)
    );

    assign in_ready = ENABLE_QSILENT
                    ? (take_fast ? can_accept_fast : can_accept_base)
                    : base_in_ready;
    assign out_valid = ENABLE_QSILENT
                     ? (emit_fast || emit_base)
                     : base_out_valid;
    assign out_tag = emit_fast ? tag_q : base_out_tag;
    assign out_score_q7 = emit_fast ? score_bus : base_out_score;
    assign out_gate_q17 = emit_fast ? gate_bus : base_out_gate;
    assign out_k_self = emit_fast ? k_self_q : base_out_k_self;
    assign out_valid_mask = emit_fast ? valid_q : base_out_mask;
    assign perf_service_cycles = emit_fast ? 16'd0 : base_service_cycles;
    assign perf_route_direct_mask = emit_fast ? 4'd0 : base_route_mask;
    assign perf_qsilent_rows = qsilent_rows_q;
    assign perf_identk_rows = identk_rows_q;
    assign perf_overlap_accepts = overlap_accepts_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            tag_q <= '0;
            k_self_q <= '0;
            valid_q <= '0;
            qsilent_rows_q <= '0;
            identk_rows_q <= '0;
            overlap_accepts_q <= '0;
            retire_kind_q[0] <= 1'b0;
            retire_kind_q[1] <= 1'b0;
            retire_wr_q <= 1'b0;
            retire_rd_q <= 1'b0;
            retire_count_q <= 2'd0;
            for (int cand = 0; cand < 5; cand = cand + 1)
                score_q[cand] <= '0;
        end else if (ENABLE_QSILENT) begin
            if (issue_fire) begin
                retire_kind_q[retire_wr_q] <= take_fast;
                retire_wr_q <= retire_wr_q + 1'b1;
                if (take_fast && (retire_count_q != 2'd0))
                    overlap_accepts_q <= overlap_accepts_q + 32'd1;
            end
            unique case (state_q)
                ST_IDLE: begin
                    if (in_valid && in_ready && take_fast) begin
                        tag_q <= in_tag;
                        k_self_q <= in_k[31:0];
                        valid_q <= in_valid_mask;
                        for (int cand = 0; cand < 5; cand = cand + 1)
                            score_q[cand] <= score_w[cand];
                        if (take_qsilent)
                            qsilent_rows_q <= qsilent_rows_q + 32'd1;
                        if (take_identk)
                            identk_rows_q <= identk_rows_q + 32'd1;
                        state_q <= ST_OUT;
                    end
                end
                ST_OUT: begin
                    if (emit_fast && out_ready)
                        state_q <= ST_IDLE;
                end
                default: state_q <= ST_IDLE;
            endcase
            unique case ({issue_fire, retire_fire})
                2'b10: retire_count_q <= retire_count_q + 2'd1;
                2'b01: begin
                    retire_count_q <= retire_count_q - 2'd1;
                    retire_rd_q <= retire_rd_q + 1'b1;
                end
                2'b11: retire_rd_q <= retire_rd_q + 1'b1;
                default: ;
            endcase
        end
    end
endmodule

`default_nettype wire
