// A commit is valid for every U and tau inside their supplied intervals.
// A replay result is a request, not a completed recovery calculation.
module atlif_threshold_packet_verifier #(
    parameter int B = 32,
    parameter int K = 4,
    parameter int W = 32,
    parameter int PACKET_W = B + 2*(W+1) + 2
                          + K*(2*(W+1)+$clog2(B)+1) + 82
) (
    input  logic clk,
    input  logic rst_n,
    input  logic request_valid_i,
    output logic request_ready_o,
    input  logic [PACKET_W-1:0] packet_i,
    input  logic [31:0] context_i,
    input  logic [15:0] epoch_i,
    input  logic signed [W-1:0] tau_lo_i,
    input  logic signed [W-1:0] tau_hi_i,
    output logic result_valid_o,
    input  logic result_ready_i,
    output logic [1:0] status_o, // 0=COMMIT, 1=REPLAY, 2=ERROR
    output logic [B-1:0] bits_o,
    output logic [31:0] theta_payload_o,
    output logic [31:0] context_o,
    output logic [15:0] epoch_o
);
    localparam int NW = W+1;
    localparam int IW = $clog2(B);
    localparam int ENTRY_W = 2*NW + IW + 1;
    localparam int A_OFF = B;
    localparam int B_OFF = A_OFF + NW;
    localparam int AV_OFF = B_OFF + NW;
    localparam int BV_OFF = AV_OFF + 1;
    localparam int KEEP_OFF = BV_OFF + 1;
    localparam int META_OFF = KEEP_OFF + K*ENTRY_W;
    logic bad, uncertain;
    logic [B-1:0] final_bits, seen_index;
    logic signed [NW-1:0] tau_lo, tau_hi, a, b;
    logic signed [NW-1:0] value_lo, value_hi;
    logic [IW-1:0] value_index;
    logic [1:0] status_next;

    initial begin
        if (B < 2 || K < 0 || K > B || W < 2)
            $fatal(1, "Unsupported B/K/W parameters");
        if (PACKET_W != META_OFF + 82)
            $fatal(1, "Packet width mismatch");
    end

    always_comb begin
        tau_lo = $signed({tau_lo_i[W-1], tau_lo_i});
        tau_hi = $signed({tau_hi_i[W-1], tau_hi_i});
        if (packet_i[META_OFF+80]) begin
            tau_lo = -$signed({tau_hi_i[W-1], tau_hi_i});
            tau_hi = -$signed({tau_lo_i[W-1], tau_lo_i});
        end
        bad = packet_i[META_OFF+81]
           || packet_i[META_OFF +: 32] != context_i
           || packet_i[META_OFF+32 +: 16] != epoch_i
           || tau_lo_i > tau_hi_i;
        a = $signed(packet_i[A_OFF +: NW]);
        b = $signed(packet_i[B_OFF +: NW]);
        uncertain = (packet_i[AV_OFF] && !(a < tau_lo))
                 || (packet_i[BV_OFF] && !(tau_hi <= b));
        final_bits = packet_i[0 +: B];
        seen_index = '0;
        value_lo = '0;
        value_hi = '0;
        value_index = '0;
        for (int i = 0; i < K; i++) begin
            value_lo = $signed(packet_i[KEEP_OFF+i*ENTRY_W +: NW]);
            value_hi = $signed(packet_i[KEEP_OFF+i*ENTRY_W+NW +: NW]);
            value_index = packet_i[KEEP_OFF+i*ENTRY_W+2*NW +: IW];
            if (!packet_i[KEEP_OFF+i*ENTRY_W+2*NW+IW]) bad = 1'b1;
            if (int'(value_index) >= B || value_lo > value_hi) bad = 1'b1;
            else begin
                if (seen_index[value_index]) bad = 1'b1;
                seen_index[value_index] = 1'b1;
                if (value_lo >= tau_hi) final_bits[value_index] = 1'b1;
                else if (value_hi < tau_lo) final_bits[value_index] = 1'b0;
                else uncertain = 1'b1;
            end
        end
        status_next = bad ? 2'd2 : uncertain ? 2'd1 : 2'd0;
    end

    assign request_ready_o = !result_valid_o || result_ready_i;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            result_valid_o <= 1'b0;
            status_o <= 2'd2;
            bits_o <= '0;
            theta_payload_o <= '0;
            context_o <= '0;
            epoch_o <= '0;
        end else if (request_ready_o) begin
            result_valid_o <= request_valid_i;
            if (request_valid_i) begin
                status_o <= status_next;
                bits_o <= (status_next == 2'd0) ? final_bits : '0;
                theta_payload_o <= packet_i[META_OFF+48 +: 32];
                context_o <= packet_i[META_OFF +: 32];
                epoch_o <= packet_i[META_OFF+32 +: 16];
            end
        end
    end
endmodule
