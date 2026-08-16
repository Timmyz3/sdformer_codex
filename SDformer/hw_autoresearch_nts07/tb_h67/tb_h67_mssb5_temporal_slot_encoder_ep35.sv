`timescale 1ns/1ps
`default_nettype none

module tb_h67_mssb5_temporal_slot_encoder_ep35;
    localparam int ROWS = 138;
    localparam int TOKENS = 450;
    localparam int PAIRS = 225;
    localparam int PAIR_ID_W = $clog2(PAIRS);

    logic clk = 1'b0;
    logic rst;
    logic window_start;
    logic pair_valid;
    logic base_ready;
    logic cse_ready;
    logic dut_ready;
    logic [PAIR_ID_W-1:0] pair_id;
    logic [63:0] q_pair;
    logic [63:0] k_pair;
    logic sink_ready;
    logic base_valid;
    logic cse_valid;
    logic dut_valid;
    logic [1:0] base_count;
    logic [1:0] cse_count;
    logic [1:0] dut_count;
    logic [15:0] base_slot0;
    logic [15:0] cse_slot0;
    logic [15:0] dut_slot0;
    logic [15:0] base_slot1;
    logic [15:0] cse_slot1;
    logic [15:0] dut_slot1;
    logic base_commit;
    logic cse_commit;
    logic dut_commit;
    logic base_error;
    logic cse_error;
    logic dut_error;
    logic [31:0] base_pairs, base_slots, base_equal;
    logic [31:0] cse_pairs, cse_slots, cse_equal;
    logic [31:0] dut_pairs, dut_slots, dut_equal;
    logic [31:0] q_mem [0:TOKENS-1];
    logic [31:0] k_mem [0:TOKENS-1];
    logic [31:0] peer_mem [0:TOKENS-1];
    logic [31:0] lfsr;
    integer fd;
    integer scan_count;
    integer file_rows;
    integer file_tokens;
    integer row_idx;
    integer token_idx;
    integer pair_idx;
    integer row_tag, stage_tag, block_tag, head_tag;
    integer expected_outputs, expected_folded;
    integer expected_gate_unused;
    integer commits;
    integer mismatches;
    string vector_path;

    always #1 clk = ~clk;

    h67_temporal_slot_encoder u_base (
        .clk_core(clk), .rst_core(rst), .window_start(window_start),
        .pair_valid(pair_valid), .pair_ready(base_ready), .pair_id(pair_id),
        .q_pair(q_pair), .k_pair(k_pair), .packet_valid(base_valid),
        .packet_ready(sink_ready), .packet_slot_count(base_count),
        .packet_slot0(base_slot0), .packet_slot1(base_slot1),
        .pair_commit(base_commit), .protocol_error(base_error),
        .perf_pairs(base_pairs), .perf_slots(base_slots),
        .perf_equal_pairs(base_equal)
    );

    h67_mssb5_temporal_slot_encoder u_dut (
        .clk_core(clk), .rst_core(rst), .window_start(window_start),
        .pair_valid(pair_valid), .pair_ready(dut_ready), .pair_id(pair_id),
        .q_pair(q_pair), .k_pair(k_pair), .packet_valid(dut_valid),
        .packet_ready(sink_ready), .packet_slot_count(dut_count),
        .packet_slot0(dut_slot0), .packet_slot1(dut_slot1),
        .pair_commit(dut_commit), .protocol_error(dut_error),
        .perf_pairs(dut_pairs), .perf_slots(dut_slots),
        .perf_equal_pairs(dut_equal)
    );

    h67_cse7_temporal_slot_encoder u_cse (
        .clk_core(clk), .rst_core(rst), .window_start(window_start),
        .pair_valid(pair_valid), .pair_ready(cse_ready), .pair_id(pair_id),
        .q_pair(q_pair), .k_pair(k_pair), .packet_valid(cse_valid),
        .packet_ready(sink_ready), .packet_slot_count(cse_count),
        .packet_slot0(cse_slot0), .packet_slot1(cse_slot1),
        .pair_commit(cse_commit), .protocol_error(cse_error),
        .perf_pairs(cse_pairs), .perf_slots(cse_slots),
        .perf_equal_pairs(cse_equal)
    );

    task automatic check_comb;
        begin
            #0;
            if ({base_ready, base_valid, base_count, base_slot0, base_slot1,
                 base_commit} !==
                {dut_ready, dut_valid, dut_count, dut_slot0, dut_slot1,
                 dut_commit}) begin
                mismatches = mismatches + 1;
                $fatal(1, "packet mismatch row=%0d pair=%0d", row_idx, pair_idx);
            end
            if ({base_ready, base_valid, base_count, base_slot0, base_slot1,
                 base_commit} !==
                {cse_ready, cse_valid, cse_count, cse_slot0, cse_slot1,
                 cse_commit}) begin
                mismatches = mismatches + 1;
                $fatal(1, "CSE packet mismatch row=%0d pair=%0d",
                    row_idx, pair_idx);
            end
        end
    endtask

    initial begin
        if (!$value$plusargs("VECTORS=%s", vector_path))
            $fatal(1, "missing +VECTORS");
        fd = $fopen(vector_path, "r");
        if (!fd) $fatal(1, "vector open failed");
        scan_count = $fscanf(fd, "%d %d", file_rows, file_tokens);
        if (scan_count != 2 || file_rows != ROWS || file_tokens != TOKENS)
            $fatal(1, "bad vector header");

        rst = 1'b1;
        window_start = 1'b0;
        pair_valid = 1'b0;
        pair_id = '0;
        q_pair = '0;
        k_pair = '0;
        sink_ready = 1'b0;
        lfsr = 32'hc0de_35a1;
        commits = 0;
        mismatches = 0;
        repeat (4) @(negedge clk);
        rst = 1'b0;

        for (row_idx = 0; row_idx < ROWS; row_idx = row_idx + 1) begin
            scan_count = $fscanf(fd, "%d %d %d %d %d %d", row_tag,
                stage_tag, block_tag, head_tag, expected_outputs,
                expected_folded);
            if (scan_count != 6) $fatal(1, "bad row header");
            for (token_idx = 0; token_idx < TOKENS;
                 token_idx = token_idx + 1) begin
                scan_count = $fscanf(fd, "%h %h %h %d", q_mem[token_idx],
                    k_mem[token_idx], peer_mem[token_idx], expected_gate_unused);
                if (scan_count != 4) $fatal(1, "bad token row");
            end
            @(negedge clk);
            window_start = 1'b1;
            @(negedge clk);
            window_start = 1'b0;

            for (pair_idx = 0; pair_idx < PAIRS; pair_idx = pair_idx + 1) begin
                pair_id = PAIR_ID_W'(pair_idx);
                q_pair = {q_mem[pair_idx+PAIRS], q_mem[pair_idx]};
                k_pair = {k_mem[pair_idx+PAIRS], k_mem[pair_idx]};
                pair_valid = 1'b1;
                sink_ready = lfsr[0] || lfsr[7];
                check_comb();
                while (!(base_ready && cse_ready && dut_ready)) begin
                    @(negedge clk);
                    lfsr = {lfsr[30:0], lfsr[31] ^ lfsr[21]
                        ^ lfsr[1] ^ lfsr[0]};
                    sink_ready = lfsr[0] || lfsr[7];
                    check_comb();
                end
                @(posedge clk);
                if (base_commit && cse_commit && dut_commit)
                    commits = commits + 1;
                @(negedge clk);
                pair_valid = 1'b0;
            end
            if (base_pairs != PAIRS || cse_pairs != PAIRS
                || dut_pairs != PAIRS || base_slots != cse_slots
                || base_slots != dut_slots || base_equal != cse_equal
                || base_equal != dut_equal)
                $fatal(1, "counter mismatch row=%0d", row_idx);
            if (base_error || cse_error || dut_error)
                $fatal(1, "protocol error row=%0d", row_idx);
        end
        $fclose(fd);
        if (commits != ROWS*PAIRS || mismatches != 0)
            $fatal(1, "final mismatch commits=%0d mismatches=%0d",
                commits, mismatches);
        $display("MSSB5_SLOT_EP35 rows=%0d pairs=%0d packet_mismatch=0",
            ROWS, commits);
        $display("PASS tb_h67_mssb5_temporal_slot_encoder_ep35");
        $finish;
    end
endmodule

`default_nettype wire
