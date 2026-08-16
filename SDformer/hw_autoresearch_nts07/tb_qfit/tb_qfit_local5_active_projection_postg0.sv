`timescale 1ns/1ps
`default_nettype none

module tb_qfit_local5_active_projection_postg0 #(
    parameter int BACKEND_KIND = 0,
    parameter int NEW_1RW_BACKEND = 0,
    parameter int MODE = 1,
    parameter int GEOMETRY_SYNC_MODE = 1,
    parameter int GROUPS = 100,
    parameter int RUN_GROUPS = GROUPS,
    parameter int RANDOM_INPUT_GAPS = 0,
    parameter int RANDOM_READ_GAPS = 0,
    parameter int OUT_DIM = 2,
    parameter int RELATION_READ_LATENCY = 1,
    parameter int RELATION_MEMORY_IMPL = 0,
    parameter int ACC_MEMORY_IMPL = 0
);
    localparam int HEIGHT = 15;
    localparam int WIDTH = 15;
    localparam int TIME_PLANES = 2;
    localparam int HEAD_DIM = 32;
    localparam int SOURCES = HEIGHT * WIDTH * TIME_PLANES;
    localparam int GATE_W = 9;
    localparam int W_W = 8;
    localparam int ACC_W = 32;
    localparam int Y_W = $clog2(HEIGHT);
    localparam int X_W = $clog2(WIDTH);
    localparam int PLANE_W = 1;
    localparam int LANE_W = $clog2(HEAD_DIM);
    localparam int OUT_W = $clog2(OUT_DIM);

    logic clk_core = 1'b0;
    logic rst_core;
    logic weight_valid;
    logic weight_ready;
    logic [LANE_W-1:0] weight_lane;
    logic [OUT_W-1:0] weight_out;
    logic signed [W_W-1:0] weight_data;
    logic weight_last;
    logic weight_context_release;
    logic weight_context_release_ready;
    logic projection_start;
    logic projection_accumulate;
    logic projection_close;
    logic projection_close_ready;
    logic projection_busy;
    logic projection_done;
    logic relation_start;
    logic relation_seal;
    logic relation_active;
    logic relation_done;
    logic relation_valid;
    logic relation_ready;
    logic [PLANE_W-1:0] relation_plane;
    logic [Y_W-1:0] relation_destination_y;
    logic [X_W-1:0] relation_destination_x;
    logic [4:0] relation_candidate_valid;
    logic [4:0] relation_active_candidate_mask;
    logic [HEAD_DIM-1:0] relation_k_self;
    logic [5*GATE_W-1:0] relation_direction_gates;
    logic read_valid;
    logic read_ready;
    logic [PLANE_W-1:0] read_plane;
    logic [Y_W-1:0] read_y;
    logic [X_W-1:0] read_x;
    logic [OUT_W-1:0] read_out;
    logic read_data_valid;
    logic signed [ACC_W-1:0] read_data;
    logic protocol_error;
    logic [31:0] perf_relation_writes;
    logic [31:0] perf_active_source_reads;
    logic [31:0] perf_dense_reads_avoided;
    logic [31:0] perf_memory_wait_cycles;
    logic [31:0] perf_descriptors;
    logic [31:0] perf_product_terms;
    logic [31:0] perf_destination_updates;
    logic [31:0] perf_term_stall_cycles;
    logic [31:0] perf_sram_reads;
    logic [31:0] perf_sram_writes;

    logic [4:0] input_valid_mem [0:GROUPS*SOURCES-1];
    logic [4:0] input_active_mem [0:GROUPS*SOURCES-1];
    logic [HEAD_DIM-1:0] input_k_mem [0:GROUPS*SOURCES-1];
    logic [5*GATE_W-1:0] input_gates_mem [0:GROUPS*SOURCES-1];
    logic signed [W_W-1:0]
        input_weight_mem [0:GROUPS*HEAD_DIM*OUT_DIM-1];
    logic signed [ACC_W-1:0]
        expected_acc_mem [0:GROUPS*SOURCES*OUT_DIM-1];
    logic [15:0] expected_active_mem [0:GROUPS-1];
    logic [31:0] expected_terms_mem [0:GROUPS-1];
    logic [31:0] expected_updates_mem [0:GROUPS-1];
    integer busy_cycles;
    integer cumulative_descriptors;
    integer total_cycles;
    integer execution_sram_reads;
    integer execution_sram_writes;
    integer first_relation_accept_cycles;
    integer last_relation_accept_cycles;
    integer fill_end_cycles;
    integer execute_end_cycles;
    string vector_dir;
    bit debug_progress;
    bit checkpoint_weights;
    bit erep_phase_trace_v4;
    bit no_acc_check;
    string actual_acc_file;
    integer actual_acc_fd;

    generate
        if (NEW_1RW_BACKEND == 0) begin : g_old
            qfit_local5_active_projection_tile #(
                .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES),
                .HEAD_DIM(HEAD_DIM), .OUT_DIM(OUT_DIM), .GATE_W(GATE_W),
                .W_W(W_W), .ACC_W(ACC_W), .BACKEND_KIND(BACKEND_KIND),
                .RELATION_READ_LATENCY(RELATION_READ_LATENCY),
                .RELATION_MEMORY_IMPL(RELATION_MEMORY_IMPL)
            ) dut (.*);
            assign perf_term_stall_cycles = '0;
            assign perf_sram_reads = '0;
            assign perf_sram_writes = '0;
        end else begin : g_new
            qfit_local5_1rw_active_projection_tile #(
                .MODE(MODE), .GEOMETRY_SYNC_MODE(GEOMETRY_SYNC_MODE),
                .HEIGHT(HEIGHT), .WIDTH(WIDTH),
                .TIME_PLANES(TIME_PLANES), .HEAD_DIM(HEAD_DIM),
                .OUT_DIM(OUT_DIM), .GATE_W(GATE_W), .W_W(W_W),
                .ACC_W(ACC_W),
                .RELATION_READ_LATENCY(RELATION_READ_LATENCY),
                .RELATION_MEMORY_IMPL(RELATION_MEMORY_IMPL),
                .ACC_MEMORY_IMPL(ACC_MEMORY_IMPL)
            ) dut (.*);
        end
    endgenerate

    always #1 clk_core = ~clk_core;

    function automatic integer weight_value(
        input integer lane, input integer out
    );
        weight_value = (lane % 5 + 1) * (out == 0 ? 1 : -2);
    endfunction

    task automatic load_weight(
        input integer group, input integer lane, input integer out,
        input bit last
    );
        integer weight_address;
        begin
            @(negedge clk_core);
            weight_lane = LANE_W'(lane);
            weight_out = OUT_W'(out);
            weight_address = (group * HEAD_DIM + lane) * OUT_DIM + out;
            weight_data = checkpoint_weights
                ? input_weight_mem[weight_address]
                : W_W'(weight_value(lane, out));
            weight_last = last;
            weight_valid = 1'b1;
            do @(posedge clk_core); while (!weight_ready);
            @(negedge clk_core);
            weight_valid = 1'b0;
            weight_last = 1'b0;
        end
    endtask

    task automatic drive_group(input integer group);
        integer address;
        bit accepted;
        begin
            for (integer source = 0; source < SOURCES; source = source + 1) begin
                if (RANDOM_INPUT_GAPS != 0)
                    repeat ($urandom_range(0, 3)) @(negedge clk_core);
                address = group * SOURCES + source;
                relation_plane = PLANE_W'(source / (HEIGHT * WIDTH));
                relation_destination_y = Y_W'(
                    (source % (HEIGHT * WIDTH)) / WIDTH
                );
                relation_destination_x = X_W'(source % WIDTH);
                relation_candidate_valid = input_valid_mem[address];
                relation_active_candidate_mask = input_active_mem[address];
                relation_k_self = input_k_mem[address];
                relation_direction_gates = input_gates_mem[address];
                relation_valid = 1'b1;
                do begin
                    @(posedge clk_core);
                    accepted = relation_ready;
                    @(negedge clk_core);
                end while (!accepted);
                if (source == 0)
                    first_relation_accept_cycles = busy_cycles;
                if (source == SOURCES - 1)
                    last_relation_accept_cycles = busy_cycles;
                relation_valid = 1'b0;
            end
        end
    endtask

    task automatic check_acc(
        input integer group,
        input integer source,
        input integer out
    );
        integer expected_address;
        begin
            if (RANDOM_READ_GAPS != 0)
                repeat ($urandom_range(0, 3)) @(negedge clk_core);
            @(negedge clk_core);
            read_plane = PLANE_W'(source / (HEIGHT * WIDTH));
            read_y = Y_W'((source % (HEIGHT * WIDTH)) / WIDTH);
            read_x = X_W'(source % WIDTH);
            read_out = OUT_W'(out);
            read_valid = 1'b1;
            do @(posedge clk_core); while (!read_ready);
            @(negedge clk_core);
            read_valid = 1'b0;
            wait (read_data_valid);
            #1;
            if (actual_acc_fd != 0)
                $fwrite(actual_acc_fd, "%08x\n", read_data);
            expected_address = (group * SOURCES + source) * OUT_DIM + out;
            if (!no_acc_check
                && read_data !== expected_acc_mem[expected_address])
                $fatal(1,
                    "post-G0 acc mismatch backend=%0d group=%0d source=%0d out=%0d got=%0d expected=%0d",
                    BACKEND_KIND, group, source, out, $signed(read_data),
                    $signed(expected_acc_mem[expected_address]));
        end
    endtask

    always_ff @(posedge clk_core) begin
        if (rst_core || projection_start)
            busy_cycles <= 0;
        else if (projection_busy)
            busy_cycles <= busy_cycles + 1;
    end

    initial begin
        debug_progress = $test$plusargs("DEBUG_PROGRESS");
        checkpoint_weights = $test$plusargs("CHECKPOINT_WEIGHTS");
        erep_phase_trace_v4 = $test$plusargs("EREP_PHASE_TRACE_V4");
        no_acc_check = $test$plusargs("NO_ACC_CHECK");
        actual_acc_fd = 0;
        if ($value$plusargs("ACTUAL_ACC_FILE=%s", actual_acc_file)) begin
            actual_acc_fd = $fopen(actual_acc_file, "w");
            if (actual_acc_fd == 0)
                $fatal(1, "cannot open ACTUAL_ACC_FILE=%s", actual_acc_file);
        end
        if (!$value$plusargs("VECTOR_DIR=%s", vector_dir))
            vector_dir = "tb_qfit/vectors/local5_active_projection_postg0_100";
        $readmemh({vector_dir, "/input_valid.memh"}, input_valid_mem);
        $readmemh({vector_dir, "/input_active.memh"}, input_active_mem);
        $readmemh({vector_dir, "/input_k.memh"}, input_k_mem);
        $readmemh({vector_dir, "/input_gates.memh"}, input_gates_mem);
        if (checkpoint_weights)
            $readmemh({vector_dir, "/input_weights.memh"}, input_weight_mem);
        if (!no_acc_check)
            $readmemh({vector_dir, "/expected_acc.memh"}, expected_acc_mem);
        $readmemh({vector_dir, "/expected_active.memh"}, expected_active_mem);
        $readmemh({vector_dir, "/expected_terms.memh"}, expected_terms_mem);
        $readmemh({vector_dir, "/expected_updates.memh"}, expected_updates_mem);

        rst_core = 1'b1;
        weight_valid = 1'b0;
        weight_lane = '0;
        weight_out = '0;
        weight_data = '0;
        weight_last = 1'b0;
        weight_context_release = 1'b0;
        projection_start = 1'b0;
        projection_accumulate = 1'b0;
        projection_close = 1'b0;
        relation_start = 1'b0;
        relation_seal = 1'b0;
        relation_valid = 1'b0;
        relation_plane = '0;
        relation_destination_y = '0;
        relation_destination_x = '0;
        relation_candidate_valid = '0;
        relation_active_candidate_mask = '0;
        relation_k_self = '0;
        relation_direction_gates = '0;
        read_valid = 1'b0;
        read_plane = '0;
        read_y = '0;
        read_x = '0;
        read_out = '0;
        busy_cycles = 0;
        cumulative_descriptors = 0;
        total_cycles = 0;
        first_relation_accept_cycles = -1;
        last_relation_accept_cycles = -1;
        repeat (5) @(negedge clk_core);
        rst_core = 1'b0;
        for (integer lane = 0; lane < HEAD_DIM; lane = lane + 1)
            for (integer out = 0; out < OUT_DIM; out = out + 1)
                load_weight(0, lane, out,
                    lane == HEAD_DIM - 1 && out == OUT_DIM - 1);

        if (RUN_GROUPS > GROUPS)
            $fatal(1, "RUN_GROUPS must not exceed GROUPS");
        for (integer group = 0; group < RUN_GROUPS; group = group + 1) begin
            first_relation_accept_cycles = -1;
            last_relation_accept_cycles = -1;
            if (debug_progress)
                $display("DEBUG group=%0d stage=start time=%0t", group, $time);
            @(negedge clk_core);
            projection_start = 1'b1;
            relation_start = 1'b1;
            @(negedge clk_core);
            projection_start = 1'b0;
            relation_start = 1'b0;
            drive_group(group);
            if (debug_progress)
                $display("DEBUG group=%0d stage=relations_written time=%0t", group, $time);
            @(negedge clk_core);
            relation_seal = 1'b1;
            @(negedge clk_core);
            relation_seal = 1'b0;
            fill_end_cycles = busy_cycles;
            if (debug_progress)
                $display("DEBUG group=%0d stage=sealed time=%0t", group, $time);
            wait (projection_close_ready);
            execute_end_cycles = busy_cycles;
            if (debug_progress)
                $display("DEBUG group=%0d stage=close_ready time=%0t", group, $time);
            @(negedge clk_core);
            projection_close = 1'b1;
            @(negedge clk_core);
            projection_close = 1'b0;
            wait (projection_done);
            if (debug_progress)
                $display("DEBUG group=%0d stage=done time=%0t", group, $time);
            execution_sram_reads = perf_sram_reads;
            execution_sram_writes = perf_sram_writes;
            if (first_relation_accept_cycles < 1
                || last_relation_accept_cycles < first_relation_accept_cycles
                || fill_end_cycles < last_relation_accept_cycles
                || execute_end_cycles < fill_end_cycles
                || busy_cycles < execute_end_cycles)
                $fatal(1, "EREP phase boundary order mismatch group=%0d", group);
            if (erep_phase_trace_v4)
                $display(
                    "EREP_PHASE_V4 schema=local5_erep_t450_phase_v4 group=%0d first_relation_accept_cycle=%0d last_relation_accept_cycle=%0d execute_begin_cycle=%0d execute_end_cycle=%0d done_cycle=%0d prepare=%0d relation_fill=%0d relation_commit=%0d execute=%0d compute_drain=%0d total=%0d active=%0d terms=%0d updates=%0d term_stall=%0d sram_reads=%0d sram_writes=%0d",
                    group, first_relation_accept_cycles,
                    last_relation_accept_cycles, fill_end_cycles,
                    execute_end_cycles, busy_cycles,
                    first_relation_accept_cycles - 1,
                    last_relation_accept_cycles
                        - first_relation_accept_cycles + 1,
                    fill_end_cycles - last_relation_accept_cycles,
                    execute_end_cycles - fill_end_cycles,
                    busy_cycles - execute_end_cycles, busy_cycles,
                    perf_active_source_reads, perf_product_terms,
                    perf_destination_updates, perf_term_stall_cycles,
                    execution_sram_reads, execution_sram_writes);
            cumulative_descriptors = cumulative_descriptors
                                   + expected_active_mem[group];
            if (perf_relation_writes != SOURCES)
                $fatal(1, "relation write count mismatch group=%0d", group);
            if (perf_active_source_reads != expected_active_mem[group])
                $fatal(1, "active source mismatch group=%0d got=%0d expected=%0d",
                    group, perf_active_source_reads, expected_active_mem[group]);
            if (perf_descriptors != expected_active_mem[group])
                $fatal(1, "per-window descriptor mismatch group=%0d", group);
            if (perf_product_terms != expected_terms_mem[group])
                $fatal(1, "term count mismatch group=%0d got=%0d expected=%0d",
                    group, perf_product_terms, expected_terms_mem[group]);
            if (perf_destination_updates != expected_updates_mem[group])
                $fatal(1, "update count mismatch group=%0d got=%0d expected=%0d",
                    group, perf_destination_updates, expected_updates_mem[group]);
            for (integer source = 0; source < SOURCES; source = source + 1)
                for (integer out = 0; out < OUT_DIM; out = out + 1)
                    check_acc(group, source, out);
            total_cycles = total_cycles + busy_cycles;
            $display(
                "GROUP backend=%0d new1rw=%0d mode=%0d latency=%0d group=%0d cycles=%0d active=%0d avoided=%0d memory_wait=%0d terms=%0d updates=%0d term_stall=%0d sram_reads=%0d sram_writes=%0d",
                BACKEND_KIND, NEW_1RW_BACKEND, MODE,
                RELATION_READ_LATENCY, group, busy_cycles,
                perf_active_source_reads, perf_dense_reads_avoided,
                perf_memory_wait_cycles, perf_product_terms,
                perf_destination_updates, perf_term_stall_cycles,
                execution_sram_reads, execution_sram_writes);
            if (protocol_error)
                $fatal(1, "protocol error group=%0d", group);
            if (checkpoint_weights && group + 1 < RUN_GROUPS) begin
                // Weight SRAM is loaded in ST_LOAD. Reset between independently
                // sampled block/head groups so each group uses its own real slice.
                @(negedge clk_core);
                rst_core = 1'b1;
                repeat (3) @(negedge clk_core);
                rst_core = 1'b0;
                for (integer lane = 0; lane < HEAD_DIM; lane = lane + 1)
                    for (integer out = 0; out < OUT_DIM; out = out + 1)
                        load_weight(group + 1, lane, out,
                            lane == HEAD_DIM - 1
                            && out == OUT_DIM - 1);
            end
        end
        $display(
            "PASS post-G0 active projection backend=%0d latency=%0d groups=%0d total_cycles=%0d descriptors=%0d",
            BACKEND_KIND, RELATION_READ_LATENCY, RUN_GROUPS, total_cycles,
            cumulative_descriptors);
        if (actual_acc_fd != 0)
            $fclose(actual_acc_fd);
        $finish;
    end

    initial begin
        repeat (20_000_000) @(posedge clk_core);
        $fatal(1, "post-G0 active projection timeout");
    end

endmodule

`default_nettype wire
