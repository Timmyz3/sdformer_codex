`timescale 1ns/1ps
`default_nettype none

module tb_qfit_dqfs_segmented_leaf;
    localparam int LANES = 2;
    localparam int WAYS = 2;
    localparam int TERM_CAPACITY = 6;
    localparam int ROW_ID_W = 3;
    localparam int EPOCH_W = 3;
    localparam int TILE_W = 3;
    localparam int PLANE_W = 1;
    localparam int Y_W = 3;
    localparam int X_W = 4;
    localparam int GATE_W = 9;
    localparam int DEST_MASK_W = 5;
    localparam int COUNT_W = $clog2(TERM_CAPACITY + 1);
    localparam int MAX_EXPECTED = 64;

    logic clk_core;
    logic rst_core;
    logic txn_start_valid;
    logic txn_start_ready;
    logic [EPOCH_W-1:0] txn_epoch;
    logic [TILE_W-1:0] txn_output_tile;
    logic txn_close_valid;
    logic txn_close_ready;
    logic txn_done;
    logic in_valid;
    logic in_ready;
    logic [ROW_ID_W-1:0] in_row_id;
    logic in_row_last;
    logic in_window_last;
    logic [$clog2(LANES)-1:0] in_lane;
    logic [GATE_W-1:0] in_gate;
    logic [PLANE_W-1:0] in_source_plane;
    logic [Y_W-1:0] in_source_y;
    logic [X_W-1:0] in_source_x;
    logic [DEST_MASK_W-1:0] in_destination_mask;
    logic group_valid;
    logic group_ready;
    logic [$clog2(LANES)-1:0] group_lane;
    logic [GATE_W-1:0] group_gate;
    logic [EPOCH_W-1:0] group_epoch;
    logic [TILE_W-1:0] group_output_tile;
    logic [COUNT_W-1:0] group_member_count;
    logic member_valid;
    logic member_ready;
    logic [PLANE_W-1:0] member_source_plane;
    logic [Y_W-1:0] member_source_y;
    logic [X_W-1:0] member_source_x;
    logic [DEST_MASK_W-1:0] member_destination_mask;
    logic member_group_last;
    logic member_row_last;
    logic member_window_last;
    logic protocol_error;
    logic [31:0] perf_accepted_terms;
    logic [31:0] perf_emitted_members;
    logic [31:0] perf_emitted_groups;
    logic [31:0] perf_capacity_seals;
    logic [31:0] perf_way_seals;
    logic [31:0] perf_input_stalls;

    integer exp_count;
    logic exp_seen [0:MAX_EXPECTED-1];
    integer exp_row [0:MAX_EXPECTED-1];
    integer exp_lane [0:MAX_EXPECTED-1];
    integer exp_gate [0:MAX_EXPECTED-1];
    integer exp_plane [0:MAX_EXPECTED-1];
    integer exp_y [0:MAX_EXPECTED-1];
    integer exp_x [0:MAX_EXPECTED-1];
    integer exp_mask [0:MAX_EXPECTED-1];
    integer cycle_count;
    integer output_rows_last;
    integer output_windows_last;
    integer group_members_left;
    logic group_active;
    integer active_group_lane;
    integer active_group_gate;
    integer active_group_epoch;
    integer active_group_tile;

    qfit_dqfs_segmented_leaf #(
        .CONTEXTS(2),
        .LANES(LANES),
        .WAYS(WAYS),
        .TERM_CAPACITY(TERM_CAPACITY),
        .ROW_ID_W(ROW_ID_W),
        .EPOCH_W(EPOCH_W),
        .TILE_W(TILE_W),
        .PLANE_W(PLANE_W),
        .Y_W(Y_W),
        .X_W(X_W),
        .GATE_W(GATE_W),
        .DEST_MASK_W(DEST_MASK_W)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    task automatic start_txn(
        input logic [EPOCH_W-1:0] epoch,
        input logic [TILE_W-1:0] tile
    );
        @(negedge clk_core);
        txn_epoch = epoch;
        txn_output_tile = tile;
        txn_start_valid = 1'b1;
        #1;
        while (!txn_start_ready) begin
            @(negedge clk_core);
            #1;
        end
        @(negedge clk_core);
        txn_start_valid = 1'b0;
    endtask

    task automatic close_txn;
        @(negedge clk_core);
        txn_close_valid = 1'b1;
        #1;
        while (!txn_close_ready) begin
            @(negedge clk_core);
            #1;
        end
        @(negedge clk_core);
        txn_close_valid = 1'b0;
    endtask

    task automatic send_term(
        input logic [ROW_ID_W-1:0] row_id,
        input logic row_last,
        input logic window_last,
        input logic [$clog2(LANES)-1:0] lane,
        input logic [GATE_W-1:0] gate,
        input logic [PLANE_W-1:0] plane,
        input logic [Y_W-1:0] y,
        input logic [X_W-1:0] x,
        input logic [DEST_MASK_W-1:0] mask,
        input logic track
    );
        @(negedge clk_core);
        in_row_id = row_id;
        in_row_last = row_last;
        in_window_last = window_last;
        in_lane = lane;
        in_gate = gate;
        in_source_plane = plane;
        in_source_y = y;
        in_source_x = x;
        in_destination_mask = mask;
        in_valid = 1'b1;
        #1;
        while (!in_ready) begin
            @(negedge clk_core);
            #1;
        end
        if (track) begin
            if (exp_count == MAX_EXPECTED)
                $fatal(1, "expected scoreboard overflow");
            exp_seen[exp_count] = 1'b0;
            exp_row[exp_count] = integer'(row_id);
            exp_lane[exp_count] = integer'(lane);
            exp_gate[exp_count] = integer'(gate);
            exp_plane[exp_count] = integer'(plane);
            exp_y[exp_count] = integer'(y);
            exp_x[exp_count] = integer'(x);
            exp_mask[exp_count] = integer'(mask);
            exp_count = exp_count + 1;
        end
        @(negedge clk_core);
        in_valid = 1'b0;
        in_row_last = 1'b0;
        in_window_last = 1'b0;
    endtask

    task automatic wait_done;
        integer timeout;
        timeout = 0;
        while (!txn_done && timeout < 5000) begin
            @(negedge clk_core);
            timeout = timeout + 1;
        end
        if (!txn_done)
            $fatal(1, "DQFS transaction timeout");
    endtask

    always @(posedge clk_core) begin
        integer match;
        integer pending_same_row;
        cycle_count <= cycle_count + 1;
        if (rst_core) begin
            group_ready <= 1'b0;
            member_ready <= 1'b0;
            group_active <= 1'b0;
            group_members_left <= 0;
            output_rows_last <= 0;
            output_windows_last <= 0;
        end else begin
            group_ready <= (cycle_count % 3) != 0;
            member_ready <= (cycle_count % 4) != 1;
            if (group_valid && group_ready) begin
                if (group_active)
                    $fatal(1, "new group before prior group completed");
                if (group_member_count == 0)
                    $fatal(1, "zero member group");
                group_active <= 1'b1;
                group_members_left <= integer'(group_member_count);
                active_group_lane <= integer'(group_lane);
                active_group_gate <= integer'(group_gate);
                active_group_epoch <= integer'(group_epoch);
                active_group_tile <= integer'(group_output_tile);
            end
            if (member_valid && member_ready) begin
                if (!group_active)
                    $fatal(1, "member without active group");
                match = -1;
                for (int index = 0; index < exp_count; index = index + 1) begin
                    if (
                        match < 0
                        && !exp_seen[index]
                        && exp_lane[index] == active_group_lane
                        && exp_gate[index] == active_group_gate
                        && exp_plane[index] == integer'(member_source_plane)
                        && exp_y[index] == integer'(member_source_y)
                        && exp_x[index] == integer'(member_source_x)
                        && exp_mask[index]
                            == integer'(member_destination_mask)
                    )
                        match = index;
                end
                if (match < 0)
                    $fatal(
                        1,
                        "unexpected member lane=%0d gate=%0d y=%0d x=%0d mask=%0d",
                        active_group_lane,
                        active_group_gate,
                        member_source_y,
                        member_source_x,
                        member_destination_mask
                    );
                if (active_group_epoch != 3 || active_group_tile != 2)
                    $fatal(1, "epoch/tile mismatch");
                pending_same_row = 0;
                for (int index = 0; index < exp_count; index = index + 1)
                    if (
                        index != match
                        && !exp_seen[index]
                        && exp_row[index] == exp_row[match]
                    )
                        pending_same_row = pending_same_row + 1;
                if (member_row_last && pending_same_row != 0)
                    $fatal(
                        1,
                        "row_last emitted before %0d older row=%0d members",
                        pending_same_row,
                        exp_row[match]
                    );
                exp_seen[match] <= 1'b1;
                if (member_group_last != (group_members_left == 1))
                    $fatal(1, "member_group_last mismatch");
                group_members_left <= group_members_left - 1;
                if (member_group_last)
                    group_active <= 1'b0;
                if (member_row_last)
                    output_rows_last <= output_rows_last + 1;
                if (member_window_last)
                    output_windows_last <= output_windows_last + 1;
            end
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        txn_start_valid = 1'b0;
        txn_epoch = '0;
        txn_output_tile = '0;
        txn_close_valid = 1'b0;
        in_valid = 1'b0;
        in_row_id = '0;
        in_row_last = 1'b0;
        in_window_last = 1'b0;
        in_lane = '0;
        in_gate = '0;
        in_source_plane = '0;
        in_source_y = '0;
        in_source_x = '0;
        in_destination_mask = '0;
        group_ready = 1'b0;
        member_ready = 1'b0;
        exp_count = 0;
        cycle_count = 0;
        output_rows_last = 0;
        output_windows_last = 0;
        group_active = 1'b0;
        group_members_left = 0;
        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;

        // Transaction 1 verifies atomic abort/drain of an illegal final term.
        start_txn(1, 1);
        send_term(0, 1, 1, 0, 0, 0, 0, 0, 1, 0);
        wait_done();
        if (!protocol_error)
            $fatal(1, "illegal term did not set protocol_error");
        if (perf_accepted_terms != 0 || perf_emitted_members != 0)
            $fatal(1, "illegal term caused architectural side effects");

        // Transaction 2 stresses interleaved rows, way seals and capacity seals.
        start_txn(3, 2);
        if (protocol_error)
            $fatal(1, "txn_start did not clear protocol_error");
        send_term(0, 0, 0, 0, 1, 0, 0, 0, 5'b00001, 1);
        send_term(1, 0, 0, 1, 4, 0, 1, 8, 5'b00100, 1);
        send_term(0, 0, 0, 0, 2, 0, 0, 1, 5'b00010, 1);
        send_term(1, 0, 0, 1, 4, 0, 1, 9, 5'b01000, 1);
        send_term(0, 0, 0, 0, 3, 0, 0, 2, 5'b00100, 1);
        send_term(0, 0, 0, 0, 3, 0, 0, 3, 5'b01000, 1);
        send_term(1, 1, 0, 1, 5, 0, 1, 10, 5'b10000, 1);
        send_term(0, 1, 0, 0, 1, 0, 0, 4, 5'b10001, 1);

        for (int item = 0; item < 7; item = item + 1)
            send_term(
                2,
                item == 6,
                0,
                1,
                7,
                0,
                2,
                X_W'(item),
                5'b00101,
                1
            );

        send_term(3, 0, 0, 0, 9, 0, 3, 0, 5'b00011, 1);
        send_term(3, 1, 1, 0, 9, 0, 3, 1, 5'b11000, 1);
        wait_done();

        if (protocol_error)
            $fatal(1, "legal DQFS transaction raised protocol_error");
        if (perf_accepted_terms != exp_count)
            $fatal(
                1,
                "accepted count mismatch got=%0d exp=%0d emitted=%0d groups=%0d cap=%0d way=%0d",
                perf_accepted_terms,
                exp_count,
                perf_emitted_members,
                perf_emitted_groups,
                perf_capacity_seals,
                perf_way_seals
            );
        if (perf_emitted_members != exp_count)
            $fatal(1, "emitted count mismatch");
        if (perf_way_seals == 0)
            $fatal(1, "way seal was not exercised");
        if (perf_capacity_seals == 0)
            $fatal(1, "capacity seal was not exercised");
        if (output_rows_last != 4)
            $fatal(1, "row_last count mismatch got=%0d", output_rows_last);
        if (output_windows_last != 1)
            $fatal(
                1,
                "window_last count mismatch got=%0d",
                output_windows_last
            );
        for (int index = 0; index < exp_count; index = index + 1)
            if (!exp_seen[index])
                $fatal(1, "expected term %0d was not emitted", index);
        $display(
            "PASS dqfs terms=%0d groups=%0d stalls=%0d cap_seals=%0d way_seals=%0d",
            perf_accepted_terms,
            perf_emitted_groups,
            perf_input_stalls,
            perf_capacity_seals,
            perf_way_seals
        );
        $finish;
    end
endmodule

`default_nettype wire
