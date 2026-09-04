`timescale 1ns/1ps
`default_nettype none

// Strong dynamic-retirement baseline for the fixed Local5 cross stencil.
// Five-color banking gives one counter access per bank for each destination:
//   bank(x, y) = (x + 2*y) mod 5.
// The five source candidates of a cross stencil therefore never conflict.
module qfit_banked_dynamic_retirement_scheduler #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int DILATION = 1,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int SOURCE_ID_W =
        (HEIGHT * WIDTH * TIME_PLANES <= 1)
        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES)
) (
    input  logic                   clk_core,
    input  logic                   rst_core,
    input  logic                   plane_start,
    input  logic [PLANE_W-1:0]     plane_id,
    input  logic                   in_valid,
    output logic                   in_ready,
    input  logic [Y_W-1:0]         in_y,
    input  logic [X_W-1:0]         in_x,
    input  logic [4:0]             in_candidate_valid,
    output logic                   retire_valid,
    input  logic                   retire_ready,
    output logic [SOURCE_ID_W-1:0] retire_source_id,
    output logic [Y_W-1:0]         retire_y,
    output logic [X_W-1:0]         retire_x,
    output logic                   plane_idle,
    output logic [31:0]            perf_producer_stalls,
    output logic [2:0]             perf_max_pending
);

    localparam int TOKENS = HEIGHT * WIDTH;
    localparam int COUNTER_ROWS = 2 * DILATION + 1;
    localparam int BANKS = 5;
    localparam int BANK_COLS = (WIDTH + BANKS - 1) / BANKS;
    localparam int BANK_DEPTH = COUNTER_ROWS * BANK_COLS;
    localparam int BANK_ADDR_W =
        (BANK_DEPTH <= 1) ? 1 : $clog2(BANK_DEPTH);
    localparam int ROW_GENERATIONS =
        (HEIGHT + COUNTER_ROWS - 1) / COUNTER_ROWS;
    // One plane visits each ring slot at most ROW_GENERATIONS times.  The
    // extra code point keeps the reset-invalid all-ones tag distinct from
    // generation zero and every incremented live generation.
    localparam int EPOCH_W =
        (ROW_GENERATIONS <= 1) ? 1 : $clog2(ROW_GENERATIONS + 2);

    logic [PLANE_W-1:0] plane_q;
    logic retire_valid_q;
    logic [SOURCE_ID_W-1:0] retire_source_q;
    logic [Y_W-1:0] retire_y_q;
    logic [X_W-1:0] retire_x_q;
    logic [31:0] stalls_q;
    logic [2:0] max_pending_q;

    logic [2:0] count_bank_q [0:BANKS-1][0:BANK_DEPTH-1];
    logic [EPOCH_W-1:0]
        epoch_bank_q [0:BANKS-1][0:BANK_DEPTH-1];
    logic [EPOCH_W-1:0] row_epoch_q [0:COUNTER_ROWS-1];

    logic bank_valid [0:BANKS-1];
    logic [2:0] bank_role [0:BANKS-1];
    logic [Y_W-1:0] bank_y [0:BANKS-1];
    logic [X_W-1:0] bank_x [0:BANKS-1];
    logic [SOURCE_ID_W-1:0] bank_source [0:BANKS-1];
    logic [BANK_ADDR_W-1:0] bank_addr [0:BANKS-1];
    logic [EPOCH_W-1:0] bank_epoch [0:BANKS-1];
    logic [2:0] bank_seen [0:BANKS-1];
    logic bank_complete [0:BANKS-1];
    logic role_complete [0:4];
    logic [SOURCE_ID_W-1:0] role_source [0:4];
    logic [Y_W-1:0] role_y [0:4];
    logic [X_W-1:0] role_x [0:4];

    logic [SOURCE_ID_W-1:0] event_id [0:4];
    logic [Y_W-1:0] event_y [0:4];
    logic [X_W-1:0] event_x [0:4];
    logic [2:0] event_count;
    logic [SOURCE_ID_W-1:0] pending_id_q [0:3];
    logic [Y_W-1:0] pending_y_q [0:3];
    logic [X_W-1:0] pending_x_q [0:3];
    logic [2:0] pending_count_q;
    logic output_slot_available;

    function automatic logic [SOURCE_ID_W-1:0] make_source_id(
        input logic [PLANE_W-1:0] p,
        input int y,
        input int x
    );
        int value;
        value = p * TOKENS + y * WIDTH + x;
        make_source_id = value;
    endfunction

    function automatic logic [2:0] expected_consumers(
        input int y,
        input int x
    );
        int count;
        count = 1;
        if (y >= DILATION)
            count = count + 1;
        if (y < HEIGHT - DILATION)
            count = count + 1;
        if (x >= DILATION)
            count = count + 1;
        if (x < WIDTH - DILATION)
            count = count + 1;
        expected_consumers = count;
    endfunction

    function automatic int bank_color(input int y, input int x);
        bank_color = (x + 2 * y) % BANKS;
    endfunction

    function automatic int bank_address(input int y, input int x);
        bank_address = (y % COUNTER_ROWS) * BANK_COLS + x / BANKS;
    endfunction

    function automatic logic [EPOCH_W-1:0] effective_epoch(input int y);
        int row_slot;
        row_slot = y % COUNTER_ROWS;
        if (in_x == 0 && y == in_y + DILATION)
            effective_epoch = row_epoch_q[row_slot] + EPOCH_W'(1);
        else
            effective_epoch = row_epoch_q[row_slot];
    endfunction

    always_comb begin
        int sy [0:4];
        int sx [0:4];
        int selected_bank;
        int selected_addr;
        int role_order [0:4];

        selected_bank = 0;
        selected_addr = 0;

        sy[0] = in_y;
        sx[0] = in_x;
        sy[1] = in_y - DILATION;
        sx[1] = in_x;
        sy[2] = in_y + DILATION;
        sx[2] = in_x;
        sy[3] = in_y;
        sx[3] = in_x - DILATION;
        sy[4] = in_y;
        sx[4] = in_x + DILATION;
        role_order[0] = 1;
        role_order[1] = 3;
        role_order[2] = 0;
        role_order[3] = 2;
        role_order[4] = 4;

        for (int bank = 0; bank < BANKS; bank = bank + 1) begin
            bank_valid[bank] = 1'b0;
            bank_role[bank] = '0;
            bank_y[bank] = '0;
            bank_x[bank] = '0;
            bank_source[bank] = '0;
            bank_addr[bank] = '0;
            bank_epoch[bank] = 1'b0;
            bank_seen[bank] = '0;
            bank_complete[bank] = 1'b0;
        end
        for (int role = 0; role < 5; role = role + 1) begin
            role_complete[role] = 1'b0;
            role_source[role] = '0;
            role_y[role] = '0;
            role_x[role] = '0;
        end

        for (int role = 0; role < 5; role = role + 1) begin
            if (in_candidate_valid[role]) begin
                selected_bank = bank_color(sy[role], sx[role]);
                selected_addr = bank_address(sy[role], sx[role]);
                bank_valid[selected_bank] = 1'b1;
                bank_role[selected_bank] = role[2:0];
                bank_y[selected_bank] = Y_W'(sy[role]);
                bank_x[selected_bank] = X_W'(sx[role]);
                bank_source[selected_bank] = make_source_id(
                    plane_q,
                    sy[role],
                    sx[role]
                );
                bank_addr[selected_bank] = BANK_ADDR_W'(selected_addr);
                bank_epoch[selected_bank] = effective_epoch(sy[role]);
            end
        end

        for (int bank = 0; bank < BANKS; bank = bank + 1) begin
            if (bank_valid[bank]) begin
                if (
                    epoch_bank_q[bank][bank_addr[bank]]
                    == bank_epoch[bank]
                )
                    bank_seen[bank] =
                        count_bank_q[bank][bank_addr[bank]];
                bank_complete[bank] =
                    bank_seen[bank] + 3'd1
                    == expected_consumers(bank_y[bank], bank_x[bank]);
                role_complete[bank_role[bank]] = bank_complete[bank];
                role_source[bank_role[bank]] = bank_source[bank];
                role_y[bank_role[bank]] = bank_y[bank];
                role_x[bank_role[bank]] = bank_x[bank];
            end
        end

        event_count = '0;
        for (int slot = 0; slot < 5; slot = slot + 1) begin
            event_id[slot] = '0;
            event_y[slot] = '0;
            event_x[slot] = '0;
        end
        for (int slot = 0; slot < 5; slot = slot + 1) begin
            if (role_complete[role_order[slot]]) begin
                event_id[event_count] = role_source[role_order[slot]];
                event_y[event_count] = role_y[role_order[slot]];
                event_x[event_count] = role_x[role_order[slot]];
                event_count = event_count + 3'd1;
            end
        end
    end

    assign output_slot_available = !retire_valid_q || retire_ready;
    assign in_ready = !plane_start
                   && pending_count_q == 0
                   && output_slot_available;
    assign retire_valid = retire_valid_q;
    assign retire_source_id = retire_source_q;
    assign retire_y = retire_y_q;
    assign retire_x = retire_x_q;
    assign plane_idle = pending_count_q == 0 && !retire_valid_q;
    assign perf_producer_stalls = stalls_q;
    assign perf_max_pending = max_pending_q;

    always_ff @(posedge clk_core) begin
        if (rst_core || plane_start) begin
            plane_q <= plane_id;
            retire_valid_q <= 1'b0;
            retire_source_q <= '0;
            retire_y_q <= '0;
            retire_x_q <= '0;
            stalls_q <= '0;
            max_pending_q <= '0;
            pending_count_q <= '0;
            for (int slot = 0; slot < 4; slot = slot + 1) begin
                pending_id_q[slot] <= '0;
                pending_y_q[slot] <= '0;
                pending_x_q[slot] <= '0;
            end
            for (int row = 0; row < COUNTER_ROWS; row = row + 1)
                row_epoch_q[row] <= 1'b0;
            for (int bank = 0; bank < BANKS; bank = bank + 1) begin
                for (int addr = 0; addr < BANK_DEPTH; addr = addr + 1) begin
                    count_bank_q[bank][addr] <= '0;
                    epoch_bank_q[bank][addr] <= '1;
                end
            end
        end else begin
            if (retire_valid_q && retire_ready)
                retire_valid_q <= 1'b0;

            if (pending_count_q != 0 && output_slot_available) begin
                retire_valid_q <= 1'b1;
                retire_source_q <= pending_id_q[0];
                retire_y_q <= pending_y_q[0];
                retire_x_q <= pending_x_q[0];
                for (int slot = 0; slot < 3; slot = slot + 1) begin
                    pending_id_q[slot] <= pending_id_q[slot+1];
                    pending_y_q[slot] <= pending_y_q[slot+1];
                    pending_x_q[slot] <= pending_x_q[slot+1];
                end
                pending_id_q[3] <= '0;
                pending_y_q[3] <= '0;
                pending_x_q[3] <= '0;
                pending_count_q <= pending_count_q - 3'd1;
            end else if (in_valid && in_ready) begin
                if (event_count != 0) begin
                    retire_valid_q <= 1'b1;
                    retire_source_q <= event_id[0];
                    retire_y_q <= event_y[0];
                    retire_x_q <= event_x[0];
                    for (int slot = 0; slot < 4; slot = slot + 1) begin
                        pending_id_q[slot] <= event_id[slot+1];
                        pending_y_q[slot] <= event_y[slot+1];
                        pending_x_q[slot] <= event_x[slot+1];
                    end
                    pending_count_q <= event_count - 3'd1;
                end

                if (in_x == 0 && in_y < HEIGHT - DILATION)
                    row_epoch_q[(in_y + DILATION) % COUNTER_ROWS]
                        <= row_epoch_q[(in_y + DILATION) % COUNTER_ROWS]
                           + EPOCH_W'(1);
                for (int bank = 0; bank < BANKS; bank = bank + 1) begin
                    if (bank_valid[bank]) begin
                        count_bank_q[bank][bank_addr[bank]]
                            <= bank_seen[bank] + 3'd1;
                        epoch_bank_q[bank][bank_addr[bank]]
                            <= bank_epoch[bank];
                    end
                end
            end

            if (in_valid && !in_ready)
                stalls_q <= stalls_q + 32'd1;
            if (pending_count_q > max_pending_q)
                max_pending_q <= pending_count_q;
            if (
                in_valid
                && in_ready
                && event_count != 0
                && event_count - 3'd1 > max_pending_q
            )
                max_pending_q <= event_count - 3'd1;
        end
    end

`ifndef SYNTHESIS
    // Sample only accepted inputs. An always_comb assertion can observe
    // in_candidate_valid and bank_valid in different Icarus delta cycles.
    always_ff @(posedge clk_core) begin
        int active_banks;
        active_banks = 0;
        for (int bank = 0; bank < BANKS; bank = bank + 1)
            active_banks += bank_valid[bank];
        if (!rst_core && in_valid && in_ready)
            assert (active_banks == $countones(in_candidate_valid)) else
            $fatal(
                1,
                "bank collision valid=%0b y=%0d x=%0d mask=%05b active_banks=%0d",
                in_valid,
                in_y,
                in_x,
                in_candidate_valid,
                active_banks
            );
    end
`endif

endmodule

`default_nettype wire
