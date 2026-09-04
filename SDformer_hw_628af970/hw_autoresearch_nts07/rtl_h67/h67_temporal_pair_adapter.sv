`default_nettype none

module h67_temporal_pair_adapter #(
    parameter int HEAD_DIM = 32
)(
    input  logic [2*HEAD_DIM-1:0] k_pair_bits,
    input  logic                    time_sel,
    output logic [HEAD_DIM-1:0]     k_current_bits,
    output logic [HEAD_DIM-1:0]     k_peer_bits
);
    always_comb begin
        if (time_sel) begin
            k_current_bits = k_pair_bits[2*HEAD_DIM-1:HEAD_DIM];
            k_peer_bits = k_pair_bits[HEAD_DIM-1:0];
        end else begin
            k_current_bits = k_pair_bits[HEAD_DIM-1:0];
            k_peer_bits = k_pair_bits[2*HEAD_DIM-1:HEAD_DIM];
        end
    end
endmodule

`default_nettype wire
