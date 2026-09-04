`timescale 1ns/1ps
`default_nettype none

// Full-chain Local5 T=450 check driven by an independent Python integer model.
// Two consecutive windows read back every accumulator row/output so stale data,
// address aliasing and incomplete scrub are observable.
module tb_local5_window_t450;
    localparam int HEAD_DIM = 32;
    localparam int OUT_DIM = 4;
    localparam int MAX_DEST = 450;
    localparam int DEST_W = 9;

    logic clk_core;
    logic rst_core;
    logic w_load_valid;
    logic w_load_ready;
    logic [4:0] w_load_lane;
    logic [1:0] w_load_out;
    logic signed [7:0] w_load_data;
    logic w_load_last;
    logic run_start;
    logic run_busy;
    logic run_done;
    logic dest_valid;
    logic dest_ready;
    logic [15:0] dest_tag;
    logic [DEST_W-1:0] dest_id;
    logic [31:0] dest_q;
    logic [31:0] dest_k_self;
    logic [4:0] dest_valid_mask;
    logic [31:0] dest_k_n;
    logic [31:0] dest_k_s;
    logic [31:0] dest_k_e;
    logic [31:0] dest_k_w;
    logic dest_last_in_window;
    logic acc_read_valid;
    logic acc_read_ready;
    logic [DEST_W-1:0] acc_read_dest;
    logic [1:0] acc_read_out;
    logic acc_data_valid;
    logic signed [31:0] acc_data;
    logic protocol_error;
    logic [31:0] perf_dest_count;
    logic [31:0] perf_cmd_count;
    logic [31:0] perf_cycle_count;

    integer vector_fd;
    integer scan_rc;
    integer errors;
    integer file_head_dim;
    integer file_out_dim;
    integer file_max_dest;
    integer file_runs;
    string vector_path;

    local5_window_attention_top #(
        .HEAD_DIM(HEAD_DIM),
        .OUT_DIM(OUT_DIM),
        .MAX_DEST(MAX_DEST),
        .DEST_W(DEST_W),
        .EXPLODE_MULT(1'b0),
        .USE_TARE(1'b0)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    task automatic load_weight_from_file;
        integer lane;
        integer out_idx;
        integer value;
        begin
            scan_rc = $fscanf(vector_fd, "%d %d %d\n", lane, out_idx, value);
            if (scan_rc != 3)
                $fatal(1, "weight vector parse failure rc=%0d", scan_rc);
            @(negedge clk_core);
            w_load_valid = 1'b1;
            w_load_lane = 5'(lane);
            w_load_out = 2'(out_idx);
            w_load_data = 8'(value);
            w_load_last = (lane == HEAD_DIM-1 && out_idx == OUT_DIM-1);
            do @(posedge clk_core); while (!w_load_ready);
        end
    endtask

    task automatic send_destination_from_file;
        integer tag_value;
        integer dest_value;
        integer mask_value;
        integer last_value;
        logic [31:0] q_value;
        logic [31:0] k0_value;
        logic [31:0] k1_value;
        logic [31:0] k2_value;
        logic [31:0] k3_value;
        logic [31:0] k4_value;
        begin
            scan_rc = $fscanf(vector_fd,
                "%h %d %h %h %h %h %h %h %h %d\n",
                tag_value, dest_value, q_value, k0_value, k1_value,
                k2_value, k3_value, k4_value, mask_value, last_value);
            if (scan_rc != 10)
                $fatal(1, "destination vector parse failure rc=%0d", scan_rc);
            while (!dest_ready) @(posedge clk_core);
            @(negedge clk_core);
            dest_valid = 1'b1;
            dest_tag = 16'(tag_value);
            dest_id = DEST_W'(dest_value);
            dest_q = q_value;
            dest_k_self = k0_value;
            dest_k_n = k1_value;
            dest_k_s = k2_value;
            dest_k_e = k3_value;
            dest_k_w = k4_value;
            dest_valid_mask = 5'(mask_value);
            dest_last_in_window = 1'(last_value);
            do @(posedge clk_core); while (!dest_ready);
            @(negedge clk_core);
            dest_valid = 1'b0;
        end
    endtask

    task automatic check_acc_from_file;
        integer read_dest;
        integer read_out;
        integer expected;
        begin
            scan_rc = $fscanf(vector_fd, "%d %d %d\n",
                              read_dest, read_out, expected);
            if (scan_rc != 3)
                $fatal(1, "acc vector parse failure rc=%0d", scan_rc);
            @(negedge clk_core);
            acc_read_valid = 1'b1;
            acc_read_dest = DEST_W'(read_dest);
            acc_read_out = 2'(read_out);
            #1;
            if (!acc_read_ready || !acc_data_valid) begin
                $error("acc read rejected dest=%0d out=%0d", read_dest, read_out);
                errors++;
            end else if (acc_data !== expected) begin
                $error("acc mismatch dest=%0d out=%0d got=%0d expected=%0d",
                       read_dest, read_out, acc_data, expected);
                errors++;
            end
            @(negedge clk_core);
            acc_read_valid = 1'b0;
        end
    endtask

    task automatic wait_done;
        integer guard;
        begin
            guard = 0;
            while (!run_done) begin
                @(posedge clk_core);
                guard++;
                if (guard > 500000)
                    $fatal(1, "timeout waiting T450 completion");
            end
        end
    endtask

    task automatic check_read_blocked_during_run;
        begin
            @(negedge clk_core);
            acc_read_valid = 1'b1;
            acc_read_dest = DEST_W'(MAX_DEST-1);
            acc_read_out = '0;
            #1;
            if (acc_read_ready || acc_data_valid) begin
                $error("acc read unexpectedly accepted before DONE");
                errors++;
            end
            @(negedge clk_core);
            acc_read_valid = 1'b0;
        end
    endtask

    task automatic check_invalid_read_after_done;
        begin
            @(negedge clk_core);
            acc_read_valid = 1'b1;
            acc_read_dest = DEST_W'(500);
            acc_read_out = '0;
            #1;
            if (acc_read_ready || acc_data_valid) begin
                $error("invalid acc address unexpectedly accepted");
                errors++;
            end
            @(posedge clk_core);
            #1;
            if (!protocol_error) begin
                $error("invalid acc address did not raise protocol_error");
                errors++;
            end
            @(negedge clk_core);
            acc_read_valid = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        w_load_valid = 1'b0;
        w_load_lane = '0;
        w_load_out = '0;
        w_load_data = '0;
        w_load_last = 1'b0;
        run_start = 1'b0;
        dest_valid = 1'b0;
        dest_tag = '0;
        dest_id = '0;
        dest_q = '0;
        dest_k_self = '0;
        dest_valid_mask = '0;
        dest_k_n = '0;
        dest_k_s = '0;
        dest_k_e = '0;
        dest_k_w = '0;
        dest_last_in_window = 1'b0;
        acc_read_valid = 1'b0;
        acc_read_dest = '0;
        acc_read_out = '0;
        errors = 0;

        if (!$value$plusargs("VECTORS=%s", vector_path))
            vector_path = "build_local5/parity/local5_t450_window_vectors.txt";
        vector_fd = $fopen(vector_path, "r");
        if (vector_fd == 0)
            $fatal(1, "cannot open vector file %0s", vector_path);
        scan_rc = $fscanf(vector_fd, "%d %d %d %d\n",
                          file_head_dim, file_out_dim, file_max_dest, file_runs);
        if (scan_rc != 4 || file_head_dim != HEAD_DIM ||
            file_out_dim != OUT_DIM || file_max_dest != MAX_DEST || file_runs != 2)
            $fatal(1, "vector header mismatch rc=%0d values=%0d/%0d/%0d/%0d",
                   scan_rc, file_head_dim, file_out_dim, file_max_dest, file_runs);

        repeat (4) @(posedge clk_core);
        rst_core = 1'b0;

        @(negedge clk_core);
        run_start = 1'b1;
        @(negedge clk_core);
        run_start = 1'b0;
        @(posedge clk_core);
        #1;
        if (!protocol_error || run_busy || run_done) begin
            $error("pre-weight run_start was not rejected cleanly");
            errors++;
        end

        for (int weight_idx = 0; weight_idx < HEAD_DIM*OUT_DIM; weight_idx++)
            load_weight_from_file();
        @(negedge clk_core);
        w_load_valid = 1'b0;
        w_load_last = 1'b0;

        for (int run_idx = 0; run_idx < 2; run_idx++) begin
            integer destination_count;
            scan_rc = $fscanf(vector_fd, "%d\n", destination_count);
            if (scan_rc != 1 || destination_count <= 0)
                $fatal(1, "run header parse failure rc=%0d count=%0d",
                       scan_rc, destination_count);

            @(negedge clk_core);
            run_start = 1'b1;
            @(negedge clk_core);
            run_start = 1'b0;
            check_read_blocked_during_run();
            for (int dest_idx = 0; dest_idx < destination_count; dest_idx++)
                send_destination_from_file();
            wait_done();

            if (protocol_error) begin
                $error("protocol_error run=%0d", run_idx);
                errors++;
            end
            if (perf_dest_count !== destination_count) begin
                $error("destination count run=%0d got=%0d expected=%0d",
                       run_idx, perf_dest_count, destination_count);
                errors++;
            end
            for (int check_idx = 0; check_idx < MAX_DEST*OUT_DIM; check_idx++)
                check_acc_from_file();

            $display("RUN %0d PASS dests=%0d cmds=%0d cycles=%0d",
                     run_idx, perf_dest_count, perf_cmd_count, perf_cycle_count);
        end

        check_invalid_read_after_done();

        $fclose(vector_fd);
        if (errors != 0)
            $fatal(1, "FAIL tb_local5_window_t450 errors=%0d", errors);
        $display("PASS tb_local5_window_t450 runs=2 checked_acc=%0d",
                 2*MAX_DEST*OUT_DIM);
        $finish;
    end
endmodule

`default_nettype wire
