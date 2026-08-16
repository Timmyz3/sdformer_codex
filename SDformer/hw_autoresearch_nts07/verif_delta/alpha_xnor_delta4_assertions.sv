`timescale 1ns/1ps
`default_nettype none

module alpha_xnor_delta4_assertions (
    input logic [3:0] lane_valid,
    input logic [19:0] lane_ids
);

    wire [4:0] lane0 = lane_ids[4:0];
    wire [4:0] lane1 = lane_ids[9:5];
    wire [4:0] lane2 = lane_ids[14:10];
    wire [4:0] lane3 = lane_ids[19:15];

    always_comb begin
        if (lane_valid[0] && lane_valid[1]) begin
            assert (lane0 != lane1);
        end
        if (lane_valid[0] && lane_valid[2]) begin
            assert (lane0 != lane2);
        end
        if (lane_valid[0] && lane_valid[3]) begin
            assert (lane0 != lane3);
        end
        if (lane_valid[1] && lane_valid[2]) begin
            assert (lane1 != lane2);
        end
        if (lane_valid[1] && lane_valid[3]) begin
            assert (lane1 != lane3);
        end
        if (lane_valid[2] && lane_valid[3]) begin
            assert (lane2 != lane3);
        end
    end

endmodule

`default_nettype wire
