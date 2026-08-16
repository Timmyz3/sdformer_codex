`timescale 1ns/1ps
`default_nettype none

// Local5双向五色前端。
// 输入位序为bit[4:0]={RIGHT, LEFT, DOWN, UP, SELF}。
// 对同一destination的五个source，bank(x,y)=(x+2*y) mod 5互不冲突。
// 每个bank使用本地next-set-bit编码器，seal后只发射活跃source。
module qfit_dual_color_active_source_index #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int SOURCE_ID_W =
        (HEIGHT * WIDTH * TIME_PLANES <= 1)
        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,
    input  logic                       build_start,
    input  logic                       build_seal,
    output logic                       build_active,
    output logic                       build_done,

    input  logic                       in_valid,
    output logic                       in_ready,
    input  logic [PLANE_W-1:0]         in_plane,
    input  logic [Y_W-1:0]             in_destination_y,
    input  logic [X_W-1:0]             in_destination_x,
    input  logic [4:0]                 in_active_candidate_mask,

    output logic                       out_valid,
    input  logic                       out_ready,
    output logic [SOURCE_ID_W-1:0]     out_source_id,
    output logic [PLANE_W-1:0]         out_source_plane,
    output logic [Y_W-1:0]             out_source_y,
    output logic [X_W-1:0]             out_source_x,
    output logic                       out_last,

    output logic                       protocol_error,
    output logic [31:0]                perf_input_candidates,
    output logic [31:0]                perf_unique_sources,
    output logic [31:0]                perf_duplicate_sets,
    output logic [31:0]                perf_bank_conflicts
);
    localparam int BANKS = 5;
    localparam int X_GROUPS = (WIDTH + 4) / 5;
    localparam int PLANE_BANK_DEPTH = HEIGHT * X_GROUPS;
    localparam int BANK_DEPTH = TIME_PLANES * PLANE_BANK_DEPTH;
    localparam int BANK_ADDR_W =
        (BANK_DEPTH <= 1) ? 1 : $clog2(BANK_DEPTH);
    localparam int TOTAL_SOURCES = HEIGHT * WIDTH * TIME_PLANES;
    localparam int SOURCE_COUNT_W = $clog2(TOTAL_SOURCES + 1);

    typedef enum logic [1:0] {
        ST_IDLE,
        ST_BUILD,
        ST_EMIT,
        ST_DONE
    } state_t;

    state_t state_q;
    logic [BANK_DEPTH-1:0] bank_bits_q [0:BANKS-1];
    logic [4:0] bank_write_enable;
    logic [BANK_ADDR_W-1:0] bank_write_addr [0:BANKS-1];
    logic [4:0] bank_selected_valid;
    logic [BANK_ADDR_W-1:0] bank_selected_addr [0:BANKS-1];
    logic [2:0] selected_bank;
    logic selected_valid;
    logic [BANK_ADDR_W-1:0] selected_addr;
    logic [2:0] round_robin_q;
    logic [SOURCE_COUNT_W-1:0] active_count_q;
    logic protocol_error_q;
    logic [31:0] input_candidates_q;
    logic [31:0] unique_sources_q;
    logic [31:0] duplicate_sets_q;
    logic [31:0] bank_conflicts_q;
    logic [2:0] input_candidate_count;
    logic [2:0] new_write_count;
    logic [2:0] duplicate_write_count;
    logic [3:0] conflict_count;
    logic invalid_candidate;

    function automatic logic [2:0] color_xy(input integer x, input integer y);
        integer value;
        begin
            value = (x + 2 * y) % 5;
            color_xy = 3'(value);
        end
    endfunction

    function automatic logic [2:0] add_mod5(
        input logic [2:0] lhs,
        input integer rhs
    );
        integer value;
        begin
            value = (lhs + rhs) % 5;
            add_mod5 = 3'(value);
        end
    endfunction

    function automatic integer x_residue_for_color(
        input integer bank,
        input integer y
    );
        integer residue;
        begin
            x_residue_for_color = 0;
            for (residue = 0; residue < 5; residue = residue + 1)
                if (((residue + 2 * y) % 5) == bank)
                    x_residue_for_color = residue;
        end
    endfunction

    always_comb begin : build_write_map
        integer role_x [0:4];
        integer role_y [0:4];
        integer bank;
        integer address;
        integer role;
        integer other;
        bank_write_enable = '0;
        input_candidate_count = {2'b0, in_active_candidate_mask[0]}
                              + {2'b0, in_active_candidate_mask[1]}
                              + {2'b0, in_active_candidate_mask[2]}
                              + {2'b0, in_active_candidate_mask[3]}
                              + {2'b0, in_active_candidate_mask[4]};
        new_write_count = '0;
        duplicate_write_count = '0;
        conflict_count = '0;
        invalid_candidate = 1'b0;
        bank = 0;
        address = 0;
        for (bank = 0; bank < BANKS; bank = bank + 1)
            bank_write_addr[bank] = '0;

        role_x[0] = in_destination_x;
        role_x[1] = in_destination_x;
        role_x[2] = in_destination_x;
        role_x[3] = in_destination_x;
        role_x[4] = in_destination_x;
        role_y[0] = in_destination_y;
        role_y[1] = in_destination_y;
        role_y[2] = in_destination_y;
        role_y[3] = in_destination_y;
        role_y[4] = in_destination_y;
        role_y[1] = role_y[1] - 1;
        role_y[2] = role_y[2] + 1;
        role_x[3] = role_x[3] - 1;
        role_x[4] = role_x[4] + 1;

        for (role = 0; role < 5; role = role + 1) begin
            if (in_active_candidate_mask[role]) begin
                if (
                    role_x[role] < 0 || role_x[role] >= WIDTH
                    || role_y[role] < 0 || role_y[role] >= HEIGHT
                    || in_plane >= TIME_PLANES
                ) begin
                    invalid_candidate = 1'b1;
                end else begin
                    bank = color_xy(role_x[role], role_y[role]);
                    address = in_plane * PLANE_BANK_DEPTH
                            + role_y[role] * X_GROUPS
                            + role_x[role] / 5;
                    bank_write_enable[bank] = 1'b1;
                    bank_write_addr[bank] = BANK_ADDR_W'(address);
                end
            end
        end

        // 该循环仅用于在合同被破坏时计数；正常五点拓扑永不命中。
        for (role = 0; role < 5; role = role + 1)
            for (other = 0; other < 5; other = other + 1)
                if (
                    other > role
                    &&
                    in_active_candidate_mask[role]
                    && in_active_candidate_mask[other]
                    && role_x[role] >= 0 && role_x[role] < WIDTH
                    && role_y[role] >= 0 && role_y[role] < HEIGHT
                    && role_x[other] >= 0 && role_x[other] < WIDTH
                    && role_y[other] >= 0 && role_y[other] < HEIGHT
                    && color_xy(role_x[role], role_y[role])
                       == color_xy(role_x[other], role_y[other])
                ) begin
                    conflict_count = conflict_count + 1'b1;
                end
        for (bank = 0; bank < BANKS; bank = bank + 1)
            if (bank_write_enable[bank]) begin
                if (bank_bits_q[bank][bank_write_addr[bank]])
                    duplicate_write_count = duplicate_write_count + 1'b1;
                else
                    new_write_count = new_write_count + 1'b1;
            end
    end

    always_comb begin : select_local_bits
        integer bank;
        integer address;
        integer offset;
        logic [2:0] candidate_bank;
        for (bank = 0; bank < BANKS; bank = bank + 1) begin
            bank_selected_valid[bank] = 1'b0;
            bank_selected_addr[bank] = '0;
            for (address = 0; address < BANK_DEPTH; address = address + 1)
                if (!bank_selected_valid[bank] && bank_bits_q[bank][address]) begin
                    bank_selected_valid[bank] = 1'b1;
                    bank_selected_addr[bank] = BANK_ADDR_W'(address);
                end
        end
        selected_valid = 1'b0;
        selected_bank = '0;
        selected_addr = '0;
        for (offset = 0; offset < BANKS; offset = offset + 1) begin
            candidate_bank = add_mod5(round_robin_q, offset);
            if (!selected_valid && bank_selected_valid[candidate_bank]) begin
                selected_valid = 1'b1;
                selected_bank = candidate_bank;
                selected_addr = bank_selected_addr[candidate_bank];
            end
        end
    end

    always_comb begin : decode_selected_source
        integer local_address;
        integer plane;
        integer plane_address;
        integer y;
        integer x_group;
        integer x_residue;
        integer x;
        local_address = selected_addr;
        plane = local_address / PLANE_BANK_DEPTH;
        plane_address = local_address % PLANE_BANK_DEPTH;
        y = plane_address / X_GROUPS;
        x_group = plane_address % X_GROUPS;
        x_residue = x_residue_for_color(selected_bank, y);
        x = 5 * x_group + x_residue;
        out_source_plane = PLANE_W'(plane);
        out_source_y = Y_W'(y);
        out_source_x = X_W'(x);
        out_source_id = SOURCE_ID_W'(plane * HEIGHT * WIDTH + y * WIDTH + x);
    end

    assign in_ready = state_q == ST_BUILD && !build_seal;
    assign build_active = state_q == ST_BUILD || state_q == ST_EMIT;
    assign build_done = state_q == ST_DONE;
    assign out_valid = state_q == ST_EMIT && selected_valid;
    assign out_last = out_valid && active_count_q == SOURCE_COUNT_W'(1);
    assign protocol_error = protocol_error_q;
    assign perf_input_candidates = input_candidates_q;
    assign perf_unique_sources = unique_sources_q;
    assign perf_duplicate_sets = duplicate_sets_q;
    assign perf_bank_conflicts = bank_conflicts_q;

    always_ff @(posedge clk_core) begin : state_update
        integer bank;
        if (rst_core) begin
            state_q <= ST_IDLE;
            round_robin_q <= '0;
            active_count_q <= '0;
            protocol_error_q <= 1'b0;
            input_candidates_q <= '0;
            unique_sources_q <= '0;
            duplicate_sets_q <= '0;
            bank_conflicts_q <= '0;
            for (bank = 0; bank < BANKS; bank = bank + 1)
                bank_bits_q[bank] <= '0;
        end else begin
            if (build_start) begin
                if (state_q != ST_IDLE && state_q != ST_DONE)
                    protocol_error_q <= 1'b1;
                else begin
                    state_q <= ST_BUILD;
                    round_robin_q <= '0;
                    active_count_q <= '0;
                    protocol_error_q <= 1'b0;
                    input_candidates_q <= '0;
                    unique_sources_q <= '0;
                    duplicate_sets_q <= '0;
                    bank_conflicts_q <= '0;
                    for (bank = 0; bank < BANKS; bank = bank + 1)
                        bank_bits_q[bank] <= '0;
                end
            end else begin
                if (in_valid && in_ready) begin
                    input_candidates_q <= input_candidates_q
                        + 32'(input_candidate_count);
                    active_count_q <= active_count_q
                        + SOURCE_COUNT_W'(new_write_count);
                    unique_sources_q <= unique_sources_q
                        + 32'(new_write_count);
                    duplicate_sets_q <= duplicate_sets_q
                        + 32'(duplicate_write_count);
                    bank_conflicts_q <= bank_conflicts_q
                        + 32'(conflict_count);
                    if (invalid_candidate || conflict_count != 0)
                        protocol_error_q <= 1'b1;
                    for (bank = 0; bank < BANKS; bank = bank + 1)
                        if (bank_write_enable[bank])
                            bank_bits_q[bank][bank_write_addr[bank]] <= 1'b1;
                end

                if (build_seal) begin
                    if (state_q != ST_BUILD || in_valid)
                        protocol_error_q <= 1'b1;
                    else if (active_count_q == 0)
                        state_q <= ST_DONE;
                    else
                        state_q <= ST_EMIT;
                end

                if (out_valid && out_ready) begin
                    bank_bits_q[selected_bank][selected_addr] <= 1'b0;
                    active_count_q <= active_count_q - 1'b1;
                    round_robin_q <= add_mod5(selected_bank, 1);
                    if (active_count_q == SOURCE_COUNT_W'(1))
                        state_q <= ST_DONE;
                end
            end
        end
    end

    initial begin
        if (HEIGHT < 3 || WIDTH < 3 || TIME_PLANES < 1)
            $fatal(1, "active source index参数非法");
    end
endmodule

`default_nettype wire
