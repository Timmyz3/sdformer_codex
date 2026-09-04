`timescale 1ns/1ps
`default_nettype none

module tb_qfit_local5_score_projection_postg0 #(
    parameter int BACKEND_KIND = 0,
    parameter int GROUPS = 100,
    parameter int RUN_GROUPS = GROUPS,
    parameter int RANDOM_INPUT_GAPS = 0,
    parameter int RANDOM_READ_GAPS = 0,
    parameter int OUT_DIM = 2,
    parameter int RELATION_READ_LATENCY = 1,
    parameter int ACC_BACKEND_KIND = 0,
    parameter bit GROUP_EQUAL_GATES = 1'b1,
    parameter int PRODUCT_CACHE_WAYS = 0,
    parameter bit ARCH_QSILENT = 1'b0,
    parameter bit ARCH_IDENTK = 1'b1,
    parameter bit ARCH_QSILENT_OVERLAP = 1'b1
);
    localparam int HEIGHT = 15;
    localparam int WIDTH = 15;
    localparam int PLANES = 2;
    localparam int SOURCES = HEIGHT * WIDTH * PLANES;
    localparam int HEAD_DIM = 32;
    localparam int GATE_W = 9;
    localparam int W_W = 8;
    localparam int ACC_W = 32;
    localparam int Y_W = $clog2(HEIGHT);
    localparam int X_W = $clog2(WIDTH);
    localparam int PLANE_W = 1;
    localparam int LANE_W = $clog2(HEAD_DIM);
    localparam int OUT_W = $clog2(OUT_DIM);
`ifdef QFIT_LOCAL5_1RW_DC_WRAPPER
    localparam bit USE_DC_WRAPPER = 1'b1;
    localparam int WRAPPER_ACC_BACKEND_KIND = 1;
    localparam string ACTIVITY_DESIGN_NAME = "local5_unified_out2_1rw_dc_top";
`elsif QFIT_LOCAL5_DC_WRAPPER
    localparam bit USE_DC_WRAPPER = 1'b1;
    localparam int WRAPPER_ACC_BACKEND_KIND = 0;
    localparam string ACTIVITY_DESIGN_NAME = "local5_unified_out2_dc_top";
`else
    localparam bit USE_DC_WRAPPER = 1'b0;
    localparam int WRAPPER_ACC_BACKEND_KIND = ACC_BACKEND_KIND;
    localparam string ACTIVITY_DESIGN_NAME = "local5_core_no_activity_wrapper";
`endif

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
    logic relation_seal_ready;
    logic relation_active;
    logic relation_done;
    logic row_valid;
    logic row_ready;
    logic [PLANE_W-1:0] row_plane;
    logic [Y_W-1:0] row_destination_y;
    logic [X_W-1:0] row_destination_x;
    logic [HEAD_DIM-1:0] row_q;
    logic [5*HEAD_DIM-1:0] row_candidate_k;
    logic [4:0] row_candidate_valid;
    logic read_valid;
    logic read_ready;
    logic [PLANE_W-1:0] read_plane;
    logic [Y_W-1:0] read_y;
    logic [X_W-1:0] read_x;
    logic [OUT_W-1:0] read_out;
    logic read_data_valid;
    logic signed [ACC_W-1:0] read_data;
    logic protocol_error;
    logic [31:0] perf_score_rows;
    logic [31:0] perf_score_service_cycles;
    logic [31:0] perf_score_direct_rows;
    logic [31:0] perf_relation_writes;
    logic [31:0] perf_active_source_reads;
    logic [31:0] perf_dense_reads_avoided;
    logic [31:0] perf_memory_wait_cycles;
    logic [31:0] perf_descriptors;
    logic [31:0] perf_product_terms;
    logic [31:0] perf_destination_updates;
    logic [31:0] perf_qsilent_rows;
    logic [31:0] perf_identk_rows;
    logic [31:0] perf_overlap_accepts;
    logic [31:0] perf_cache_hits;
    logic [31:0] perf_cache_misses;
    logic [31:0] perf_tag_compares;
    logic [31:0] perf_lru_writes;
    logic [31:0] perf_product_reads;
    logic [31:0] perf_product_writes;
    logic [31:0] perf_product_starts;
    logic [31:0] perf_weight_reads;

    logic [31:0] input_q_mem [0:GROUPS*SOURCES-1];
    logic [5*HEAD_DIM-1:0]
        input_candidate_k_mem [0:GROUPS*SOURCES-1];
    logic [4:0] input_valid_mem [0:GROUPS*SOURCES-1];
    logic [5*16-1:0] expected_scores_mem [0:GROUPS*SOURCES-1];
    logic [5*GATE_W-1:0] expected_gates_mem [0:GROUPS*SOURCES-1];
    logic signed [W_W-1:0]
        input_weight_mem [0:GROUPS*HEAD_DIM*OUT_DIM-1];
    logic signed [ACC_W-1:0]
        expected_acc_mem [0:GROUPS*SOURCES*OUT_DIM-1];
    logic [15:0] expected_active_mem [0:GROUPS-1];
    logic [31:0] expected_terms_mem [0:GROUPS-1];
    logic [31:0] expected_updates_mem [0:GROUPS-1];

    integer busy_cycles;
    integer total_cycles;
    integer score_check_index;
    integer score_mismatches;
    integer active_group;
    string vector_dir;
    string actual_acc_file;
    integer actual_acc_fd;
    logic observed_score_out_fire;
    logic [5*16-1:0] observed_score_out_q7;
    logic [5*GATE_W-1:0] observed_score_out_gate;
    bit dump_configured;
    bit dump_active;
    integer dump_start_group;
    integer dump_group_count;
    integer dump_measured_cycles;
    string dump_file;
    string dump_scope;
    bit dump_busy_only;

    generate
`ifdef QFIT_LOCAL5_1RW_DC_WRAPPER
        begin : g_dc_wrapper
            local5_unified_out2_1rw_dc_top dut (.*);
            assign observed_score_out_fire = dut.u_core.score_out_fire;
            assign observed_score_out_q7 = dut.u_core.score_out_q7;
            assign observed_score_out_gate = dut.u_core.score_out_gate;
            assign perf_cache_hits = '0;
            assign perf_cache_misses = '0;
            assign perf_tag_compares = '0;
            assign perf_lru_writes = '0;
            assign perf_product_reads = '0;
            assign perf_product_writes = '0;
            assign perf_product_starts = perf_product_terms;
            assign perf_weight_reads = perf_product_terms * OUT_DIM;
            initial begin
                #0;
                if (dump_configured) begin
                    $dumpfile(dump_file);
                    $dumpvars(0, dut);
                    $dumpoff;
                end
            end
        end
`elsif QFIT_LOCAL5_DC_WRAPPER
        begin : g_dc_wrapper
            local5_unified_out2_dc_top dut (.*);
            assign observed_score_out_fire = dut.u_core.score_out_fire;
            assign observed_score_out_q7 = dut.u_core.score_out_q7;
            assign observed_score_out_gate = dut.u_core.score_out_gate;
            assign perf_cache_hits = '0;
            assign perf_cache_misses = '0;
            assign perf_tag_compares = '0;
            assign perf_lru_writes = '0;
            assign perf_product_reads = '0;
            assign perf_product_writes = '0;
            assign perf_product_starts = perf_product_terms;
            assign perf_weight_reads = perf_product_terms * OUT_DIM;
            initial begin
                #0;
                if (dump_configured) begin
                    $dumpfile(dump_file);
                    $dumpvars(0, dut);
                    $dumpoff;
                end
            end
        end
`else
        begin : g_core
            qfit_local5_score_active_projection_tile #(
                .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(PLANES),
                .HEAD_DIM(HEAD_DIM), .OUT_DIM(OUT_DIM), .GATE_W(GATE_W),
                .W_W(W_W), .ACC_W(ACC_W), .BACKEND_KIND(BACKEND_KIND),
                .ACC_BACKEND_KIND(ACC_BACKEND_KIND),
                .GROUP_EQUAL_GATES(GROUP_EQUAL_GATES),
                .PRODUCT_CACHE_WAYS(PRODUCT_CACHE_WAYS),
                .RELATION_READ_LATENCY(RELATION_READ_LATENCY),
                .ARCH_QSILENT(ARCH_QSILENT),
                .ARCH_IDENTK(ARCH_IDENTK),
                .ARCH_QSILENT_OVERLAP(ARCH_QSILENT_OVERLAP)
            ) dut (.*);
            assign observed_score_out_fire = dut.score_out_fire;
            assign observed_score_out_q7 = dut.score_out_q7;
            assign observed_score_out_gate = dut.score_out_gate;
            initial begin
                #0;
                if (dump_configured) begin
                    $dumpfile(dump_file);
                    $dumpvars(0, dut);
                    $dumpoff;
                end
            end
        end
`endif
    endgenerate

    always #1 clk_core = ~clk_core;

    task automatic load_weight(
        input integer group, input integer lane, input integer out,
        input bit last
    );
        integer address;
        begin
            @(negedge clk_core);
            weight_lane = LANE_W'(lane);
            weight_out = OUT_W'(out);
            address = (group * HEAD_DIM + lane) * OUT_DIM + out;
            weight_data = input_weight_mem[address];
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
            for (integer destination = 0; destination < SOURCES;
                 destination = destination + 1) begin
                if (RANDOM_INPUT_GAPS != 0)
                    repeat ($urandom_range(0, 3)) @(negedge clk_core);
                address = group * SOURCES + destination;
                row_plane = PLANE_W'(destination / (HEIGHT * WIDTH));
                row_destination_y = Y_W'(
                    (destination % (HEIGHT * WIDTH)) / WIDTH
                );
                row_destination_x = X_W'(destination % WIDTH);
                row_q = input_q_mem[address];
                row_candidate_k = input_candidate_k_mem[address];
                row_candidate_valid = input_valid_mem[address];
                row_valid = 1'b1;
                do begin
                    @(posedge clk_core);
                    accepted = row_ready;
                    @(negedge clk_core);
                end while (!accepted);
                row_valid = 1'b0;
            end
        end
    endtask

    task automatic check_acc(
        input integer group, input integer source, input integer out
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
            if (read_data !== expected_acc_mem[expected_address])
                $fatal(1,
                    "score-projection Acc mismatch backend=%0d group=%0d source=%0d out=%0d got=%0d expected=%0d",
                    BACKEND_KIND, group, source, out, $signed(read_data),
                    $signed(expected_acc_mem[expected_address]));
        end
    endtask

    always_ff @(posedge clk_core) begin
        integer expected_address;
        if (rst_core || projection_start) begin
            busy_cycles <= 0;
            score_check_index <= 0;
            score_mismatches <= 0;
        end else begin
            if (projection_busy)
                busy_cycles <= busy_cycles + 1;
            if (observed_score_out_fire) begin
                expected_address = active_group * SOURCES + score_check_index;
                if (observed_score_out_q7 !== expected_scores_mem[expected_address]) begin
                    $error("score mismatch backend=%0d group=%0d row=%0d",
                           BACKEND_KIND, active_group, score_check_index);
                    score_mismatches <= score_mismatches + 1;
                end
                if (observed_score_out_gate !== expected_gates_mem[expected_address]) begin
                    $error("gate mismatch backend=%0d group=%0d row=%0d",
                           BACKEND_KIND, active_group, score_check_index);
                    score_mismatches <= score_mismatches + 1;
                end
                score_check_index <= score_check_index + 1;
            end
        end
    end

    // VCS ICPD: do not mix always_ff with initial assignment of the same integer.
    always @(posedge clk_core) begin
        if (dump_active)
            dump_measured_cycles <= dump_measured_cycles + 1;
    end

    initial begin
        dump_configured = $value$plusargs("DUMP_FILE=%s", dump_file);
        dump_start_group = 0;
        dump_group_count = 1;
        dump_scope = "full";
        void'($value$plusargs("DUMP_START_GROUP=%d", dump_start_group));
        void'($value$plusargs("DUMP_GROUPS=%d", dump_group_count));
        void'($value$plusargs("DUMP_SCOPE=%s", dump_scope));
        dump_busy_only = dump_scope == "busy";
        dump_active = 1'b0;
        dump_measured_cycles = 0;
        if (dump_configured && !USE_DC_WRAPPER)
            $fatal(1, "DUMP_FILE is reserved for USE_DC_WRAPPER=1");
        if (dump_group_count <= 0)
            $fatal(1, "DUMP_GROUPS must be positive");
        if (dump_scope != "full" && dump_scope != "busy")
            $fatal(1, "DUMP_SCOPE must be full or busy");
        if (USE_DC_WRAPPER && (BACKEND_KIND != 0
            || ACC_BACKEND_KIND != WRAPPER_ACC_BACKEND_KIND
            || OUT_DIM != 2
            || RELATION_READ_LATENCY != 1 || !ARCH_QSILENT
            || !ARCH_IDENTK || !ARCH_QSILENT_OVERLAP))
            $fatal(1, "USE_DC_WRAPPER parameters do not match frozen Local5 top");
        if (!$value$plusargs("VECTOR_DIR=%s", vector_dir))
            vector_dir = "tb_qfit/vectors/local5_score_projection_100";
        actual_acc_fd = 0;
        if ($value$plusargs("ACTUAL_ACC_FILE=%s", actual_acc_file)) begin
            actual_acc_fd = $fopen(actual_acc_file, "w");
            if (actual_acc_fd == 0)
                $fatal(1, "cannot open ACTUAL_ACC_FILE=%s", actual_acc_file);
        end
        $readmemh({vector_dir, "/input_q.memh"}, input_q_mem);
        $readmemh(
            {vector_dir, "/input_candidate_k.memh"}, input_candidate_k_mem
        );
        $readmemh({vector_dir, "/input_valid.memh"}, input_valid_mem);
        $readmemh({vector_dir, "/expected_scores.memh"}, expected_scores_mem);
        $readmemh({vector_dir, "/expected_gates.memh"}, expected_gates_mem);
        $readmemh({vector_dir, "/input_weights.memh"}, input_weight_mem);
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
        row_valid = 1'b0;
        row_plane = '0;
        row_destination_y = '0;
        row_destination_x = '0;
        row_q = '0;
        row_candidate_k = '0;
        row_candidate_valid = '0;
        read_valid = 1'b0;
        read_plane = '0;
        read_y = '0;
        read_x = '0;
        read_out = '0;
        active_group = 0;
        total_cycles = 0;
        repeat (5) @(negedge clk_core);
        rst_core = 1'b0;

        if (RUN_GROUPS > GROUPS)
            $fatal(1, "RUN_GROUPS must not exceed GROUPS");
        for (integer group = 0; group < RUN_GROUPS; group = group + 1) begin
            active_group = group;
            if (dump_configured && !dump_busy_only
                && group == dump_start_group) begin
                @(negedge clk_core);
                dump_active = 1'b1;
                $dumpon;
            end
            for (integer lane = 0; lane < HEAD_DIM; lane = lane + 1)
                for (integer out = 0; out < OUT_DIM; out = out + 1)
                    load_weight(group, lane, out,
                        lane == HEAD_DIM - 1 && out == OUT_DIM - 1);
            if (dump_configured && dump_busy_only
                && group >= dump_start_group
                && group < dump_start_group + dump_group_count) begin
                @(negedge clk_core);
                dump_active = 1'b1;
                $dumpon;
            end
            @(negedge clk_core);
            projection_start = 1'b1;
            relation_start = 1'b1;
            @(negedge clk_core);
            projection_start = 1'b0;
            relation_start = 1'b0;
            drive_group(group);
            wait (relation_seal_ready);
            @(negedge clk_core);
            relation_seal = 1'b1;
            @(negedge clk_core);
            relation_seal = 1'b0;
            wait (projection_close_ready);
            @(negedge clk_core);
            projection_close = 1'b1;
            @(negedge clk_core);
            projection_close = 1'b0;
            wait (projection_done);

            if (dump_configured && dump_busy_only
                && group >= dump_start_group
                && group < dump_start_group + dump_group_count) begin
                @(negedge clk_core);
                dump_active = 1'b0;
                $dumpoff;
                if (group + 1 == dump_start_group + dump_group_count)
                    $display(
                        "SAIF_MEASUREMENT design=%s start_group=%0d groups=%0d measured_cycles=%0d scope=busy_projection",
                        ACTIVITY_DESIGN_NAME, dump_start_group,
                        dump_group_count, dump_measured_cycles
                    );
            end

            if (score_check_index != SOURCES || perf_score_rows != SOURCES)
                $fatal(1, "score row count mismatch group=%0d checked=%0d perf=%0d",
                       group, score_check_index, perf_score_rows);
            if (score_mismatches != 0)
                $fatal(1, "score/gate mismatches group=%0d count=%0d",
                       group, score_mismatches);
            if (perf_relation_writes != SOURCES)
                $fatal(1, "relation writes mismatch group=%0d", group);
            if (perf_active_source_reads != expected_active_mem[group]
                || perf_descriptors != expected_active_mem[group])
                $fatal(
                    1,
                    "active descriptor mismatch group=%0d expected=%0d reads=%0d descriptors=%0d",
                    group,
                    expected_active_mem[group],
                    perf_active_source_reads,
                    perf_descriptors
                );
            if (
                perf_product_terms
                != (GROUP_EQUAL_GATES
                    ? expected_terms_mem[group]
                    : expected_updates_mem[group])
            )
                $fatal(1, "term mismatch group=%0d", group);
            if (perf_destination_updates != expected_updates_mem[group])
                $fatal(1, "update mismatch group=%0d", group);
            for (integer source = 0; source < SOURCES; source = source + 1)
                for (integer out = 0; out < OUT_DIM; out = out + 1)
                    check_acc(group, source, out);

            total_cycles = total_cycles + busy_cycles;
            $display(
                "GROUP backend=%0d latency=%0d group=%0d cycles=%0d score_rows=%0d score_service=%0d score_direct_rows=%0d qsilent_rows=%0d identk_rows=%0d overlap=%0d active=%0d memory_wait=%0d terms=%0d updates=%0d cache_hits=%0d cache_misses=%0d tag_compares=%0d lru_writes=%0d product_reads=%0d product_writes=%0d product_starts=%0d weight_reads=%0d",
                BACKEND_KIND, RELATION_READ_LATENCY, group, busy_cycles,
                perf_score_rows, perf_score_service_cycles,
                perf_score_direct_rows, perf_qsilent_rows, perf_identk_rows,
                perf_overlap_accepts,
                perf_active_source_reads,
                perf_memory_wait_cycles, perf_product_terms,
                perf_destination_updates, perf_cache_hits,
                perf_cache_misses, perf_tag_compares, perf_lru_writes,
                perf_product_reads, perf_product_writes,
                perf_product_starts, perf_weight_reads);
            if (protocol_error)
                $fatal(1, "protocol error group=%0d", group);
            if (dump_configured && !dump_busy_only
                && group + 1 == dump_start_group + dump_group_count) begin
                @(negedge clk_core);
                dump_active = 1'b0;
                $dumpoff;
                $display(
                    "SAIF_MEASUREMENT design=%s start_group=%0d groups=%0d measured_cycles=%0d scope=full_load_compute_readback",
                    ACTIVITY_DESIGN_NAME, dump_start_group,
                    dump_group_count, dump_measured_cycles
                );
            end
            if (group + 1 < RUN_GROUPS) begin
                @(negedge clk_core);
                rst_core = 1'b1;
                repeat (3) @(negedge clk_core);
                rst_core = 1'b0;
            end
        end
        $display(
            "PASS Local5 score-to-projection backend=%0d latency=%0d groups=%0d total_cycles=%0d",
            BACKEND_KIND, RELATION_READ_LATENCY, RUN_GROUPS, total_cycles);
        if (actual_acc_fd != 0)
            $fclose(actual_acc_fd);
        $finish;
    end

    initial begin
        repeat (30_000_000) @(posedge clk_core);
        $fatal(1, "Local5 score-to-projection timeout");
    end

`ifdef SNPS_FSDB
    string fsdb_file;
    initial begin
        if ($value$plusargs("FSDB_FILE=%s", fsdb_file)) begin
            $fsdbDumpfile(fsdb_file);
            $fsdbDumpvars(0, tb_qfit_local5_score_projection_postg0);
        end
    end
`endif
endmodule

`default_nettype wire
