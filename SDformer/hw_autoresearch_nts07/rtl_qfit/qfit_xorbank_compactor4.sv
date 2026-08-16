`timescale 1ns/1ps
`default_nettype none

// 四个固定bank各选择一个事件。正交映射：
//   bank = lane[1:0] XOR direction
// 每个bank只包含32个(dir,lane)位置，避免全局128->4优先链。
module qfit_xorbank_compactor4 (
    input  logic [31:0] mask_n,
    input  logic [31:0] mask_s,
    input  logic [31:0] mask_e,
    input  logic [31:0] mask_w,
    output logic [31:0] next_mask_n,
    output logic [31:0] next_mask_s,
    output logic [31:0] next_mask_e,
    output logic [31:0] next_mask_w,
    output logic [3:0]  event_valid,
    output logic [7:0]  event_dir,
    output logic [19:0] event_lane
);

    logic [31:0] work [0:3];
    logic found;

    always_comb begin
        work[0] = mask_n;
        work[1] = mask_s;
        work[2] = mask_e;
        work[3] = mask_w;
        event_valid = '0;
        event_dir = '0;
        event_lane = '0;

        for (int bank = 0; bank < 4; bank = bank + 1) begin
            found = 1'b0;
            for (int dir = 0; dir < 4; dir = dir + 1) begin
                for (int lane = 0; lane < 32; lane = lane + 1) begin
                    if (
                        !found
                        && ((lane[1:0] ^ dir[1:0]) == bank[1:0])
                        && work[dir][lane]
                    ) begin
                        event_valid[bank] = 1'b1;
                        event_dir[bank*2 +: 2] = 2'(dir);
                        event_lane[bank*5 +: 5] = 5'(lane);
                        work[dir][lane] = 1'b0;
                        found = 1'b1;
                    end
                end
            end
        end

        next_mask_n = work[0];
        next_mask_s = work[1];
        next_mask_e = work[2];
        next_mask_w = work[3];
    end

endmodule

`default_nettype wire
