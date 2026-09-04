`timescale 1ns/1ps
`default_nettype none

// Map the five Local5 roles {self, down, up, right, left} onto five colors.
module qfit_local5_color_map #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int Y_W = $clog2(HEIGHT),
    parameter int X_W = $clog2(WIDTH),
    parameter int PLANE_W = (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int BANK_DEPTH = TIME_PLANES * HEIGHT * ((WIDTH + 4) / 5),
    parameter int BANK_ADDR_W = $clog2(BANK_DEPTH)
) (
    input  logic [PLANE_W-1:0]             source_plane,
    input  logic [Y_W-1:0]                 source_y,
    input  logic [X_W-1:0]                 source_x,
    input  logic [4:0]                     role_mask,
    output logic [4:0]                     bank_geometry_valid,
    output logic [4:0]                     bank_enable,
    output logic [5*BANK_ADDR_W-1:0]       bank_address_packed,
    output logic                           boundary_error
);
    localparam int X_GROUPS = (WIDTH + 4) / 5;
    localparam int PLANE_BANK_DEPTH = HEIGHT * X_GROUPS;

    always_comb begin
        integer role_y [0:4];
        integer role_x [0:4];
        logic role_valid [0:4];
        integer bank;
        integer address;

        role_y[0] = 32'(source_y); role_x[0] = 32'(source_x);
        role_y[1] = 32'(source_y) + 1; role_x[1] = 32'(source_x);
        role_y[2] = 32'(source_y) - 1; role_x[2] = 32'(source_x);
        role_y[3] = 32'(source_y); role_x[3] = 32'(source_x) + 1;
        role_y[4] = 32'(source_y); role_x[4] = 32'(source_x) - 1;
        role_valid[0] = 1'b1;
        role_valid[1] = 32'(source_y) < HEIGHT - 1;
        role_valid[2] = source_y != 0;
        role_valid[3] = 32'(source_x) < WIDTH - 1;
        role_valid[4] = source_x != 0;

        bank_geometry_valid = '0;
        bank_enable = '0;
        bank_address_packed = '0;
        boundary_error = 1'b0;
        bank = 0;
        address = 0;
        for (integer role = 0; role < 5; role++) begin
            if (role_mask[role] && !role_valid[role])
                boundary_error = 1'b1;
            if (role_valid[role]) begin
                bank = (role_x[role] + 2 * role_y[role]) % 5;
                address = 32'(source_plane) * PLANE_BANK_DEPTH
                        + role_y[role] * X_GROUPS + role_x[role] / 5;
                bank_geometry_valid[bank] = 1'b1;
                bank_address_packed[bank*BANK_ADDR_W +: BANK_ADDR_W]
                    = BANK_ADDR_W'(address);
                if (role_mask[role])
                    bank_enable[bank] = 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
