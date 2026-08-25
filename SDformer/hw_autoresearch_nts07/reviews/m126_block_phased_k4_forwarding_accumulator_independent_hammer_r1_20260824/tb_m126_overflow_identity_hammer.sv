`timescale 1ns/1ps
`default_nettype none

module tb_m126_overflow_identity_hammer;
    localparam int LANES = 96;
    localparam int ROWS = 384;

    logic clk_core, rst_core;
    logic window_start_valid, window_start_ready, window_start_accept;
    logic weight_fill_valid, weight_fill_ready;
    logic [2:0] weight_fill_block;
    logic [3:0] weight_fill_source;
    logic [1:0] weight_fill_beat;
    logic [255:0] weight_fill_data;
    logic weight_fill_accept;
    logic row_valid, row_ready;
    logic [2:0] row_block;
    logic [8:0] row_offset;
    logic [15:0] row_source_mask, row_negate_mask;
    logic row_accept, row_done;
    logic window_end_valid, window_end_ready, window_end_accept;
    logic commit_valid, commit_ready;
    logic [2:0] commit_block;
    logic [8:0] commit_row;
    logic [1823:0] commit_data;
    logic commit_last, window_done;
    logic lane_mem_rd_en;
    logic [11:0] lane_mem_rd_addr;
    logic [18:0] lane_mem_rd_data [0:LANES-1];
    logic lane_mem_wr_en;
    logic [11:0] lane_mem_wr_addr;
    logic [18:0] lane_mem_wr_data [0:LANES-1];
    logic observed_fold_update_accept;
    logic observed_accumulator_update_accept;
    logic [2:0] observed_fold_update_block;
    logic [8:0] observed_fold_update_row;
    logic [1823:0] observed_fold_update_delta;
    logic [15:0] observed_fold_selected_mask;
    logic [15:0] observed_fold_remaining_mask;
    logic [15:0] observed_cache_valid;
    logic [2:0] observed_resident_block;
    logic observed_resident_block_valid;
    logic fold_protocol_error, accumulator_protocol_error;
    logic protocol_error, window_active, busy;

    logic [18:0] lane_memory [0:LANES-1][0:3071];
    int row_accept_count;
    int fold_accept_count;
    int lane_write_count;
    int identity_row_accepts;
    int identity_fold_accepts;
    int identity_writes;
    int overflow_row_accepts;
    int overflow_fold_accepts;
    int overflow_writes;

    m126_block_phased_k4_forwarding_accumulator_island dut (.*);
    always #1 clk_core = ~clk_core;

    always @(posedge clk_core) begin
        if (lane_mem_rd_en) begin
            for (int lane = 0; lane < LANES; lane++)
                lane_mem_rd_data[lane] <= lane_memory[lane][lane_mem_rd_addr];
        end
        if (lane_mem_wr_en) begin
            for (int lane = 0; lane < LANES; lane++)
                lane_memory[lane][lane_mem_wr_addr] <= lane_mem_wr_data[lane];
            lane_write_count = lane_write_count + 1;
        end
        if (row_accept)
            row_accept_count = row_accept_count + 1;
        if (observed_fold_update_accept)
            fold_accept_count = fold_accept_count + 1;
        if (observed_fold_update_accept
                !== observed_accumulator_update_accept)
            $fatal(1, "M126 boundary fold/acc accept divergence");
        if (rst_core && (window_start_ready || window_start_accept
                || weight_fill_ready || weight_fill_accept
                || row_ready || row_accept || row_done
                || window_end_ready || window_end_accept
                || commit_valid || window_done || lane_mem_rd_en
                || lane_mem_wr_en || observed_fold_update_accept
                || observed_accumulator_update_accept || protocol_error))
            $fatal(1, "M126 boundary reset isolation failure");
    end

    task automatic drive_idle;
        begin
            window_start_valid = 0;
            weight_fill_valid = 0;
            weight_fill_block = 0;
            weight_fill_source = 0;
            weight_fill_beat = 0;
            weight_fill_data = 0;
            row_valid = 0;
            row_block = 0;
            row_offset = 0;
            row_source_mask = 0;
            row_negate_mask = 0;
            window_end_valid = 0;
            commit_ready = 0;
        end
    endtask

    task automatic clean_reset;
        begin
            @(negedge clk_core);
            drive_idle();
            rst_core = 1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 0;
            repeat (2) @(posedge clk_core);
            if (protocol_error || window_active || busy)
                $fatal(1, "M126 boundary reset recovery failure");
        end
    endtask

    task automatic start_window;
        begin
            @(negedge clk_core);
            window_start_valid = 1;
            do @(posedge clk_core); while (!window_start_accept);
            @(negedge clk_core);
            window_start_valid = 0;
        end
    endtask

    task automatic fill_boundary_source(input int source);
        logic [255:0] payload;
        int lane;
        begin
            for (int beat = 0; beat < 3; beat++) begin
                payload = 0;
                for (int item = 0; item < 32; item++) begin
                    lane = beat * 32 + item;
                    payload[item * 8 +: 8]
                        = (lane == 0 && source < 4) ? 8'h80 : 8'h00;
                end
                @(negedge clk_core);
                weight_fill_valid = 1;
                weight_fill_block = 0;
                weight_fill_source = source[3:0];
                weight_fill_beat = beat[1:0];
                weight_fill_data = payload;
                do @(posedge clk_core); while (!weight_fill_accept);
            end
            @(negedge clk_core);
            weight_fill_valid = 0;
        end
    endtask

    task automatic send_single_row(
        input int row_id,
        input logic [15:0] mask,
        input logic [15:0] negate
    );
        int watchdog;
        begin
            @(negedge clk_core);
            row_valid = 1;
            row_block = 0;
            row_offset = row_id[8:0];
            row_source_mask = mask;
            row_negate_mask = negate;
            watchdog = 0;
            do begin
                @(posedge clk_core);
                watchdog = watchdog + 1;
                if (watchdog > 30)
                    $fatal(1, "M126 boundary row accept watchdog row=%0d", row_id);
            end while (!row_accept);
            @(negedge clk_core);
            row_valid = 0;
            watchdog = 0;
            while (!row_done && !protocol_error) begin
                @(posedge clk_core);
                watchdog = watchdog + 1;
                if (watchdog > 30)
                    $fatal(1, "M126 boundary row completion watchdog row=%0d", row_id);
            end
            @(posedge clk_core);
        end
    endtask

    initial begin
        clk_core = 0;
        rst_core = 1;
        row_accept_count = 0;
        fold_accept_count = 0;
        lane_write_count = 0;
        identity_row_accepts = 0;
        identity_fold_accepts = 0;
        identity_writes = 0;
        overflow_row_accepts = 0;
        overflow_fold_accepts = 0;
        overflow_writes = 0;
        drive_idle();
        for (int lane = 0; lane < LANES; lane++) begin
            lane_mem_rd_data[lane] = 0;
            for (int address = 0; address < 3072; address++)
                lane_memory[lane][address] = 0;
        end

        clean_reset();
        start_window();
        fill_boundary_source(0);
        send_single_row(384, 16'h0001, 16'h0000);
        repeat (2) @(posedge clk_core);
        identity_row_accepts = row_accept_count;
        identity_fold_accepts = fold_accept_count;
        identity_writes = lane_write_count;
        if (identity_row_accepts != 1 || identity_fold_accepts != 0
                || identity_writes != 0 || !protocol_error
                || !accumulator_protocol_error || fold_protocol_error)
            $fatal(1, "M126 identity boundary mismatch rows=%0d folds=%0d writes=%0d protocol=%0b fold_fault=%0b acc_fault=%0b",
                   identity_row_accepts, identity_fold_accepts,
                   identity_writes, protocol_error,
                   fold_protocol_error, accumulator_protocol_error);

        clean_reset();
        row_accept_count = 0;
        fold_accept_count = 0;
        lane_write_count = 0;
        start_window();
        for (int source = 0; source < 4; source++)
            fill_boundary_source(source);
        for (int transaction = 0; transaction < 512; transaction++)
            send_single_row(7, 16'h000f, 16'h000f);
        repeat (3) @(posedge clk_core);
        overflow_row_accepts = row_accept_count;
        overflow_fold_accepts = fold_accept_count;
        overflow_writes = lane_write_count;
        if (overflow_row_accepts != 512 || overflow_fold_accepts != 512
                || overflow_writes != 511 || !protocol_error
                || !accumulator_protocol_error || fold_protocol_error
                || $signed(lane_memory[0][7]) != 261632
                || lane_mem_wr_en)
            $fatal(1, "M126 overflow boundary mismatch rows=%0d folds=%0d writes=%0d mem0=%0d protocol=%0b fold_fault=%0b acc_fault=%0b",
                   overflow_row_accepts, overflow_fold_accepts,
                   overflow_writes, $signed(lane_memory[0][7]),
                   protocol_error, fold_protocol_error,
                   accumulator_protocol_error);

        $display("PASS M126 overflow identity hammer identity_row_accepts=1 identity_fold_accepts=0 identity_writes=0 out_of_range_row_fail_closed=true overflow_row_accepts=512 overflow_fold_accepts=512 overflow_writes=511 last_valid_lane0=261632 overflow_fail_closed=true overflow_retry=false physical_speedup=false system_speedup=false");
        $finish;
    end
endmodule

`default_nettype wire
