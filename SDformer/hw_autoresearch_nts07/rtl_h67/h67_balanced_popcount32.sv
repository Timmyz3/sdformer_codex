`default_nettype none

module h67_balanced_popcount32 (
    input  logic [31:0] bits,
    output logic [5:0] count
);
    logic [1:0] level1 [0:15];
    logic [2:0] level2 [0:7];
    logic [3:0] level3 [0:3];
    logic [4:0] level4 [0:1];

    generate
        for (genvar index1 = 0; index1 < 16; index1 = index1 + 1)
            assign level1[index1] = {1'b0, bits[2*index1]}
                                 + {1'b0, bits[2*index1+1]};
        for (genvar index2 = 0; index2 < 8; index2 = index2 + 1)
            assign level2[index2] = {1'b0, level1[2*index2]}
                                 + {1'b0, level1[2*index2+1]};
        for (genvar index3 = 0; index3 < 4; index3 = index3 + 1)
            assign level3[index3] = {1'b0, level2[2*index3]}
                                 + {1'b0, level2[2*index3+1]};
        for (genvar index4 = 0; index4 < 2; index4 = index4 + 1)
            assign level4[index4] = {1'b0, level3[2*index4]}
                                 + {1'b0, level3[2*index4+1]};
    endgenerate

    assign count = {1'b0, level4[0]} + {1'b0, level4[1]};
endmodule

`default_nettype wire
