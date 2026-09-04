`timescale 1ns/1ps
`default_nettype none

module tb_qfit_gasr2c_singlebank_postg0 #(
    parameter int GROUPS = 100,
    parameter bit RANDOM_GAPS = 1'b0
);
    localparam int DEPTH = 90;
    localparam int OUT_DIM = 2;
    localparam int ACC_W = 32;
    localparam int ADDR_W = $clog2(DEPTH);
    localparam int VEC_W = OUT_DIM * ACC_W;
    localparam int MAX_SOURCES = 8948;
    localparam int MAX_UPDATES = 33271;

    logic clk_core = 1'b0;
    logic rst_core;
    longint unsigned cycle_counter;

    logic direct_run_start;
    logic direct_update_valid;
    logic direct_update_ready;
    logic [ADDR_W-1:0] direct_update_addr;
    logic [VEC_W-1:0] direct_update_delta;
    logic direct_flush_valid;
    logic direct_flush_ready;
    logic direct_flush_done;
    logic direct_read_valid;
    logic direct_read_ready;
    logic [ADDR_W-1:0] direct_read_addr;
    logic direct_read_data_valid;
    logic [VEC_W-1:0] direct_read_data;
    logic direct_protocol_error;
    logic [31:0] direct_perf_updates;
    logic [31:0] direct_perf_reads;
    logic [31:0] direct_perf_writes;

    logic gasr_run_start;
    logic gasr_prepare_valid;
    logic gasr_prepare_ready;
    logic [ADDR_W-1:0] gasr_prepare_addr;
    logic gasr_activate_valid;
    logic gasr_activate_ready;
    logic [ADDR_W-1:0] gasr_activate_addr;
    logic gasr_update_valid;
    logic gasr_update_ready;
    logic [ADDR_W-1:0] gasr_update_addr;
    logic [VEC_W-1:0] gasr_update_delta;
    logic gasr_flush_valid;
    logic gasr_flush_ready;
    logic gasr_flush_done;
    logic gasr_read_valid;
    logic gasr_read_ready;
    logic [ADDR_W-1:0] gasr_read_addr;
    logic gasr_read_data_valid;
    logic [VEC_W-1:0] gasr_read_data;
    logic gasr_protocol_error;
    logic [31:0] gasr_perf_updates;
    logic [31:0] gasr_perf_hits;
    logic [31:0] gasr_perf_misses;
    logic [31:0] gasr_perf_reads;
    logic [31:0] gasr_perf_writes;

    logic [31:0] group_source_offsets [0:GROUPS];
    logic [31:0] source_update_offsets [0:MAX_SOURCES];
    logic [ADDR_W-1:0] source_addr_mem [0:MAX_SOURCES-1];
    logic [VEC_W-1:0] update_delta_mem [0:MAX_UPDATES-1];
    logic [VEC_W-1:0] expected_acc_mem [0:GROUPS*DEPTH-1];
    string vector_dir;
    longint unsigned total_direct_cycles;
    longint unsigned total_gasr_cycles;
    longint unsigned total_direct_reads;
    longint unsigned total_direct_writes;
    longint unsigned total_gasr_reads;
    longint unsigned total_gasr_writes;
    logic [31:0] lfsr_q;

    qfit_direct_1rw_acc_bank #(
        .DEPTH(DEPTH), .OUT_DIM(OUT_DIM), .ACC_W(ACC_W)
    ) u_direct (
        .clk_core(clk_core), .rst_core(rst_core), .run_start(direct_run_start),
        .run_accumulate(1'b0),
        .update_valid(direct_update_valid), .update_ready(direct_update_ready),
        .update_addr(direct_update_addr), .update_delta(direct_update_delta),
        .flush_valid(direct_flush_valid), .flush_ready(direct_flush_ready),
        .flush_done(direct_flush_done), .read_valid(direct_read_valid),
        .read_ready(direct_read_ready), .read_addr(direct_read_addr),
        .read_data_valid(direct_read_data_valid), .read_data(direct_read_data),
        .protocol_error(direct_protocol_error), .perf_updates(direct_perf_updates),
        .perf_sram_reads(direct_perf_reads), .perf_sram_writes(direct_perf_writes)
    );

    qfit_gasr2c_acc_bank #(
        .DEPTH(DEPTH), .OUT_DIM(OUT_DIM), .ACC_W(ACC_W)
    ) u_gasr (
        .clk_core(clk_core), .rst_core(rst_core), .run_start(gasr_run_start),
        .prepare_valid(gasr_prepare_valid), .prepare_ready(gasr_prepare_ready),
        .prepare_addr(gasr_prepare_addr), .activate_valid(gasr_activate_valid),
        .activate_ready(gasr_activate_ready), .activate_addr(gasr_activate_addr),
        .update_valid(gasr_update_valid), .update_ready(gasr_update_ready),
        .update_addr(gasr_update_addr), .update_delta(gasr_update_delta),
        .flush_valid(gasr_flush_valid), .flush_ready(gasr_flush_ready),
        .flush_done(gasr_flush_done), .read_valid(gasr_read_valid),
        .read_ready(gasr_read_ready), .read_addr(gasr_read_addr),
        .read_data_valid(gasr_read_data_valid), .read_data(gasr_read_data),
        .protocol_error(gasr_protocol_error), .perf_updates(gasr_perf_updates),
        .perf_prepare_hits(gasr_perf_hits), .perf_prepare_misses(gasr_perf_misses),
        .perf_sram_reads(gasr_perf_reads), .perf_sram_writes(gasr_perf_writes)
    );

    always #1 clk_core = ~clk_core;
    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_counter <= 0;
            lfsr_q <= 32'h1ace_b00c;
        end else begin
            cycle_counter <= cycle_counter + 1;
            lfsr_q <= {lfsr_q[30:0], lfsr_q[31] ^ lfsr_q[21]
                                     ^ lfsr_q[1] ^ lfsr_q[0]};
        end
    end

    task automatic pulse_direct_start;
        begin
            @(negedge clk_core);
            direct_run_start = 1'b1;
            @(negedge clk_core);
            direct_run_start = 1'b0;
        end
    endtask

    task automatic pulse_gasr_start;
        begin
            @(negedge clk_core);
            gasr_run_start = 1'b1;
            @(negedge clk_core);
            gasr_run_start = 1'b0;
        end
    endtask

    task automatic run_direct_group(
        input integer group,
        output longint unsigned elapsed
    );
        integer source_index;
        integer update_index;
        integer source_stop;
        integer update_stop;
        longint unsigned start_cycle;
        logic accepted;
        begin
            pulse_direct_start();
            start_cycle = cycle_counter;
            source_index = group_source_offsets[group];
            source_stop = group_source_offsets[group + 1];
            update_index = source_update_offsets[source_index];
            update_stop = source_update_offsets[source_stop];
            while (update_index < update_stop) begin
                direct_update_addr = source_addr_mem[source_index];
                direct_update_delta = update_delta_mem[update_index];
                if (!direct_update_valid)
                    direct_update_valid = !RANDOM_GAPS || lfsr_q[0];
                @(posedge clk_core);
                accepted = direct_update_valid && direct_update_ready;
                @(negedge clk_core);
                if (accepted) begin
                    direct_update_valid = 1'b0;
                    update_index = update_index + 1;
                    if (update_index == source_update_offsets[source_index + 1])
                        source_index = source_index + 1;
                end
            end
            direct_update_valid = 1'b0;
            direct_flush_valid = 1'b1;
            do begin
                @(posedge clk_core);
                accepted = direct_flush_valid && direct_flush_ready;
                @(negedge clk_core);
            end while (!accepted);
            direct_flush_valid = 1'b0;
            wait (direct_flush_done);
            elapsed = cycle_counter - start_cycle;
        end
    endtask

    task automatic prepare_and_activate_first(input integer source_index);
        logic accepted;
        begin
            gasr_prepare_addr = source_addr_mem[source_index];
            gasr_prepare_valid = 1'b1;
            do begin
                @(posedge clk_core);
                accepted = gasr_prepare_valid && gasr_prepare_ready;
                @(negedge clk_core);
            end while (!accepted);
            gasr_prepare_valid = 1'b0;
            gasr_activate_addr = source_addr_mem[source_index];
            gasr_activate_valid = 1'b1;
            do begin
                @(posedge clk_core);
                accepted = gasr_activate_valid && gasr_activate_ready;
                @(negedge clk_core);
            end while (!accepted);
            gasr_activate_valid = 1'b0;
        end
    endtask

    task automatic run_gasr_group(
        input integer group,
        output longint unsigned elapsed
    );
        integer source_index;
        integer source_stop;
        integer update_index;
        integer update_stop;
        logic source_complete;
        logic update_accepted;
        logic activate_accepted;
        logic has_next;
        longint unsigned start_cycle;
        begin
            pulse_gasr_start();
            start_cycle = cycle_counter;
            source_index = group_source_offsets[group];
            source_stop = group_source_offsets[group + 1];
            if (source_index < source_stop) begin
                prepare_and_activate_first(source_index);
                update_index = source_update_offsets[source_index];
                update_stop = source_update_offsets[source_index + 1];
                source_complete = 1'b0;
                while (source_index < source_stop) begin
                    has_next = source_index + 1 < source_stop;
                    gasr_prepare_valid = has_next;
                    if (has_next)
                        gasr_prepare_addr = source_addr_mem[source_index + 1];

                    if (!gasr_update_valid)
                        gasr_update_valid = !source_complete
                                          && (!RANDOM_GAPS || lfsr_q[1]);
                    gasr_update_addr = source_addr_mem[source_index];
                    gasr_update_delta = update_delta_mem[update_index];

                    gasr_activate_valid = has_next && (
                        source_complete
                        || (gasr_update_valid
                            && update_index + 1 == update_stop
                            && gasr_update_ready)
                    );
                    if (has_next)
                        gasr_activate_addr = source_addr_mem[source_index + 1];

                    @(posedge clk_core);
                    update_accepted = gasr_update_valid && gasr_update_ready;
                    activate_accepted = gasr_activate_valid && gasr_activate_ready;
                    @(negedge clk_core);

                    if (update_accepted) begin
                        gasr_update_valid = 1'b0;
                        update_index = update_index + 1;
                        if (update_index == update_stop)
                            source_complete = 1'b1;
                    end
                    if (activate_accepted) begin
                        source_index = source_index + 1;
                        update_index = source_update_offsets[source_index];
                        update_stop = source_update_offsets[source_index + 1];
                        source_complete = 1'b0;
                    end else if (source_complete && !has_next) begin
                        source_index = source_stop;
                    end
                end
            end
            gasr_prepare_valid = 1'b0;
            gasr_activate_valid = 1'b0;
            gasr_update_valid = 1'b0;
            gasr_flush_valid = 1'b1;
            do begin
                @(posedge clk_core);
                update_accepted = gasr_flush_valid && gasr_flush_ready;
                @(negedge clk_core);
            end while (!update_accepted);
            gasr_flush_valid = 1'b0;
            wait (gasr_flush_done);
            elapsed = cycle_counter - start_cycle;
        end
    endtask

    task automatic check_direct(input integer group, input integer addr);
        logic accepted;
        begin
            direct_read_addr = ADDR_W'(addr);
            direct_read_valid = 1'b1;
            do begin
                @(posedge clk_core);
                accepted = direct_read_valid && direct_read_ready;
                @(negedge clk_core);
            end while (!accepted);
            direct_read_valid = 1'b0;
            wait (direct_read_data_valid);
            #1;
            if (direct_read_data !== expected_acc_mem[group * DEPTH + addr])
                $fatal(1,
                    "direct Acc32 mismatch group=%0d addr=%0d got=%h expected=%h",
                    group, addr, direct_read_data,
                    expected_acc_mem[group * DEPTH + addr]);
        end
    endtask

    task automatic check_gasr(input integer group, input integer addr);
        logic accepted;
        begin
            gasr_read_addr = ADDR_W'(addr);
            gasr_read_valid = 1'b1;
            do begin
                @(posedge clk_core);
                accepted = gasr_read_valid && gasr_read_ready;
                @(negedge clk_core);
            end while (!accepted);
            gasr_read_valid = 1'b0;
            wait (gasr_read_data_valid);
            #1;
            if (gasr_read_data !== expected_acc_mem[group * DEPTH + addr])
                $fatal(1,
                    "GASR Acc32 mismatch group=%0d addr=%0d got=%h expected=%h",
                    group, addr, gasr_read_data,
                    expected_acc_mem[group * DEPTH + addr]);
        end
    endtask

    initial begin
        if (!$value$plusargs("VECTOR_DIR=%s", vector_dir))
            vector_dir = "tb_qfit/vectors/local5_gasr_singlebank_postg0_100";
        $readmemh({vector_dir, "/group_source_offsets.memh"}, group_source_offsets);
        $readmemh({vector_dir, "/source_update_offsets.memh"}, source_update_offsets);
        $readmemh({vector_dir, "/source_addr.memh"}, source_addr_mem);
        $readmemh({vector_dir, "/update_delta.memh"}, update_delta_mem);
        $readmemh({vector_dir, "/expected_acc.memh"}, expected_acc_mem);

        rst_core = 1'b1;
        direct_run_start = 1'b0;
        direct_update_valid = 1'b0;
        direct_update_addr = '0;
        direct_update_delta = '0;
        direct_flush_valid = 1'b0;
        direct_read_valid = 1'b0;
        direct_read_addr = '0;
        gasr_run_start = 1'b0;
        gasr_prepare_valid = 1'b0;
        gasr_prepare_addr = '0;
        gasr_activate_valid = 1'b0;
        gasr_activate_addr = '0;
        gasr_update_valid = 1'b0;
        gasr_update_addr = '0;
        gasr_update_delta = '0;
        gasr_flush_valid = 1'b0;
        gasr_read_valid = 1'b0;
        gasr_read_addr = '0;
        total_direct_cycles = 0;
        total_gasr_cycles = 0;
        total_direct_reads = 0;
        total_direct_writes = 0;
        total_gasr_reads = 0;
        total_gasr_writes = 0;
        repeat (5) @(negedge clk_core);
        rst_core = 1'b0;

        for (integer group = 0; group < GROUPS; group++) begin
            longint unsigned direct_cycles;
            longint unsigned gasr_cycles;
            logic [31:0] direct_exec_reads;
            logic [31:0] direct_exec_writes;
            logic [31:0] gasr_exec_reads;
            logic [31:0] gasr_exec_writes;
            run_direct_group(group, direct_cycles);
            direct_exec_reads = direct_perf_reads;
            direct_exec_writes = direct_perf_writes;
            for (integer addr = 0; addr < DEPTH; addr++)
                check_direct(group, addr);
            run_gasr_group(group, gasr_cycles);
            gasr_exec_reads = gasr_perf_reads;
            gasr_exec_writes = gasr_perf_writes;
            for (integer addr = 0; addr < DEPTH; addr++)
                check_gasr(group, addr);
            if (direct_protocol_error || gasr_protocol_error)
                $fatal(1, "protocol error group=%0d direct=%0b gasr=%0b",
                    group, direct_protocol_error, gasr_protocol_error);
            if (direct_perf_updates !== gasr_perf_updates)
                $fatal(1, "update count mismatch group=%0d", group);
            total_direct_cycles = total_direct_cycles + direct_cycles;
            total_gasr_cycles = total_gasr_cycles + gasr_cycles;
            total_direct_reads = total_direct_reads + 64'(direct_exec_reads);
            total_direct_writes = total_direct_writes + 64'(direct_exec_writes);
            total_gasr_reads = total_gasr_reads + 64'(gasr_exec_reads);
            total_gasr_writes = total_gasr_writes + 64'(gasr_exec_writes);
            $display(
                "GROUP group=%0d direct_cycles=%0d gasr_cycles=%0d updates=%0d direct_reads=%0d direct_writes=%0d gasr_reads=%0d gasr_writes=%0d gasr_hits=%0d gasr_misses=%0d",
                group, direct_cycles, gasr_cycles, direct_perf_updates,
                direct_exec_reads, direct_exec_writes, gasr_exec_reads,
                gasr_exec_writes, gasr_perf_hits, gasr_perf_misses);
        end
        $display(
            "PASS GASR2C singlebank groups=%0d random_gaps=%0d direct_cycles=%0d gasr_cycles=%0d direct_reads=%0d direct_writes=%0d gasr_reads=%0d gasr_writes=%0d",
            GROUPS, RANDOM_GAPS, total_direct_cycles, total_gasr_cycles,
            total_direct_reads, total_direct_writes,
            total_gasr_reads, total_gasr_writes);
        $finish;
    end

    initial begin
        repeat (20_000_000) @(posedge clk_core);
        $fatal(1, "GASR2C singlebank timeout");
    end
endmodule

`default_nettype wire
