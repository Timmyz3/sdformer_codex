`timescale 1ns/1ps
`default_nettype none

module tb_h67_temporal_slot_flow_real_trace_1s_random #(
    parameter int MAX_TOKENS = 450,
    parameter int SLOT_FIFO_DEPTH = 32
);
    localparam int HEAD_DIM = 32;
    localparam int PAIRS = MAX_TOKENS / 2;
    localparam int PAIR_ID_W = $clog2(PAIRS);
    localparam int TOKEN_W = $clog2(MAX_TOKENS + 1);
    localparam int FIFO_OCC_W = $clog2(SLOT_FIFO_DEPTH + 1);

    logic clk;
    logic rst_core;
    logic descriptor_issue_enable;
    logic common_out_ready;
    logic [15:0] backpressure_lfsr;
    logic backpressure_reseed;
    integer global_cycle;

    logic fixed_start;
    logic fixed_seal;
    logic fixed_seal_ready;
    logic fixed_done;
    logic fixed_pair_valid;
    logic fixed_pair_ready;
    logic [PAIR_ID_W-1:0] fixed_pair_id;
    logic [63:0] fixed_q_pair;
    logic [63:0] fixed_k_pair;
    logic fixed_out_valid;
    logic fixed_out_last;
    logic [TOKEN_W-1:0] fixed_out_token;
    logic [31:0] fixed_out_k;
    logic [8:0] fixed_out_gate;
    logic fixed_error;
    logic [31:0] fixed_pairs;
    logic [31:0] fixed_slots;
    logic [31:0] fixed_equal;
    logic [31:0] fixed_desc;
    logic [31:0] fixed_tokens;
    logic [31:0] fixed_active;
    logic [31:0] fixed_classes;
    logic [31:0] fixed_exp;
    logic [31:0] fixed_emitted;
    logic [31:0] fixed_k_reads;
    logic [31:0] fixed_k_bits;
    logic [31:0] fixed_cycles;
    logic [31:0] fixed_pair_stalls;
    logic [31:0] fixed_desc_stalls;
    logic [31:0] fixed_out_stalls;
    logic [FIFO_OCC_W-1:0] fixed_fifo_occ;
    logic [FIFO_OCC_W-1:0] fixed_fifo_max;

    logic rqtb_start;
    logic rqtb_seal;
    logic rqtb_seal_ready;
    logic rqtb_done;
    logic rqtb_pair_valid;
    logic rqtb_pair_ready;
    logic [PAIR_ID_W-1:0] rqtb_pair_id;
    logic [63:0] rqtb_q_pair;
    logic [63:0] rqtb_k_pair;
    logic rqtb_out_valid;
    logic rqtb_out_last;
    logic [TOKEN_W-1:0] rqtb_out_token;
    logic [31:0] rqtb_out_k;
    logic [8:0] rqtb_out_gate;
    logic rqtb_error;
    logic [31:0] rqtb_pairs;
    logic [31:0] rqtb_slots;
    logic [31:0] rqtb_equal;
    logic [31:0] rqtb_desc;
    logic [31:0] rqtb_tokens;
    logic [31:0] rqtb_active;
    logic [31:0] rqtb_classes;
    logic [31:0] rqtb_exp;
    logic [31:0] rqtb_emitted;
    logic [31:0] rqtb_k_reads;
    logic [31:0] rqtb_k_bits;
    logic [31:0] rqtb_cycles;
    logic [31:0] rqtb_pair_stalls;
    logic [31:0] rqtb_desc_stalls;
    logic [31:0] rqtb_out_stalls;
    logic [FIFO_OCC_W-1:0] rqtb_fifo_occ;
    logic [FIFO_OCC_W-1:0] rqtb_fifo_max;

    logic [31:0] q_vector [0:MAX_TOKENS-1];
    logic [31:0] k_vector [0:MAX_TOKENS-1];
    logic [31:0] peer_vector [0:MAX_TOKENS-1];
    integer expected_gate [0:MAX_TOKENS-1];
    logic [TOKEN_W-1:0] fixed_token_log [0:MAX_TOKENS-1];
    logic [31:0] fixed_k_log [0:MAX_TOKENS-1];
    logic [8:0] fixed_gate_log [0:MAX_TOKENS-1];
    logic fixed_last_log [0:MAX_TOKENS-1];
    logic [TOKEN_W-1:0] rqtb_token_log [0:MAX_TOKENS-1];
    logic [31:0] rqtb_k_log [0:MAX_TOKENS-1];
    logic [8:0] rqtb_gate_log [0:MAX_TOKENS-1];
    logic rqtb_last_log [0:MAX_TOKENS-1];
    logic [TOKEN_W-1:0] expected_token_log [0:MAX_TOKENS-1];
    logic fixed_seen [0:MAX_TOKENS-1];
    logic rqtb_seen [0:MAX_TOKENS-1];
    integer signed expected_acc [0:HEAD_DIM-1];
    integer signed fixed_acc [0:HEAD_DIM-1];
    integer signed rqtb_acc [0:HEAD_DIM-1];
    integer fixed_occ_hist [0:SLOT_FIFO_DEPTH];
    integer rqtb_occ_hist [0:SLOT_FIFO_DEPTH];

    integer fd;
    integer scan_count;
    integer file_rows;
    integer file_tokens;
    integer row_limit;
    integer row_index;
    integer row_tag;
    integer stage_tag;
    integer block_tag;
    integer head_tag;
    integer expected_outputs_header;
    integer expected_folded_header;
    integer fixed_count;
    integer rqtb_count;
    integer total_checked;
    integer total_rows;
    integer total_fixed_cycles;
    integer total_rqtb_cycles;
    integer total_fixed_slots;
    integer total_rqtb_slots;
    integer total_fixed_exp;
    integer total_rqtb_exp;
    integer total_errors;
    logic fixed_row_active;
    logic rqtb_row_active;
    string vector_path;
    string dump_path;

    function automatic integer signed lane_weight(input integer lane);
        lane_weight = (lane % 17) - 8;
    endfunction

    function automatic integer motion_score(
        input logic [31:0] q_bits,
        input logic [31:0] current_k,
        input logic [31:0] peer_k
    );
        integer overlap;
        integer same_zero;
        integer motion;
        integer integer_base;
        integer quotient;
        integer remainder;
        begin
            overlap = $countones(q_bits & current_k);
            same_zero = $countones(~q_bits & ~current_k);
            motion = $countones(current_k ^ peer_k);
            integer_base = 4 * overlap + motion;
            quotient = same_zero / 16;
            remainder = same_zero % 16;
            if (remainder > 8
                || (remainder == 8 && ((integer_base + quotient) & 1)))
                quotient = quotient + 1;
            motion_score = integer_base + quotient;
        end
    endfunction

    h67_temporal_slot_shiftmax_sync_k_top #(
        .PAIRS(PAIRS),
        .PAIR_ID_W(PAIR_ID_W),
        .TOKEN_W(TOKEN_W),
        .SLOT_FIFO_DEPTH(SLOT_FIFO_DEPTH),
        .FIFO_OCC_W(FIFO_OCC_W),
        .QUOTIENT_ENABLE(1'b0)
    ) u_fixed (
        .clk_core(clk), .rst_core(rst_core),
        .window_start(fixed_start), .window_seal(fixed_seal),
        .descriptor_issue_enable(descriptor_issue_enable),
        .cfg_preserve_mean(1'b1), .cfg_threshold_q8(8'd64),
        .seal_ready(fixed_seal_ready), .window_done(fixed_done),
        .pair_valid(fixed_pair_valid), .pair_ready(fixed_pair_ready),
        .pair_id(fixed_pair_id), .q_pair(fixed_q_pair), .k_pair(fixed_k_pair),
        .out_valid(fixed_out_valid), .out_ready(common_out_ready),
        .out_last(fixed_out_last), .out_token_id(fixed_out_token),
        .out_k_bits(fixed_out_k), .out_gate_q17(fixed_out_gate),
        .out_threshold_q8(), .protocol_error(fixed_error),
        .perf_pairs(fixed_pairs), .perf_slots(fixed_slots),
        .perf_equal_pairs(fixed_equal),
        .perf_quotient_descriptors(fixed_desc),
        .perf_original_tokens(fixed_tokens), .perf_active_entries(fixed_active),
        .perf_class_transactions(fixed_classes), .perf_exp_transactions(fixed_exp),
        .perf_emitted_tokens(fixed_emitted),
        .perf_k_read_transactions(fixed_k_reads), .perf_k_read_bits(fixed_k_bits),
        .perf_total_cycles(fixed_cycles),
        .perf_pair_stall_cycles(fixed_pair_stalls),
        .perf_descriptor_stall_cycles(fixed_desc_stalls),
        .perf_output_stall_cycles(fixed_out_stalls),
        .perf_fifo_occupancy(fixed_fifo_occ),
        .perf_fifo_max_occupancy(fixed_fifo_max)
    );

    h67_temporal_slot_shiftmax_sync_k_top #(
        .PAIRS(PAIRS),
        .PAIR_ID_W(PAIR_ID_W),
        .TOKEN_W(TOKEN_W),
        .SLOT_FIFO_DEPTH(SLOT_FIFO_DEPTH),
        .FIFO_OCC_W(FIFO_OCC_W),
        .QUOTIENT_ENABLE(1'b1)
    ) u_rqtb (
        .clk_core(clk), .rst_core(rst_core),
        .window_start(rqtb_start), .window_seal(rqtb_seal),
        .descriptor_issue_enable(descriptor_issue_enable),
        .cfg_preserve_mean(1'b1), .cfg_threshold_q8(8'd64),
        .seal_ready(rqtb_seal_ready), .window_done(rqtb_done),
        .pair_valid(rqtb_pair_valid), .pair_ready(rqtb_pair_ready),
        .pair_id(rqtb_pair_id), .q_pair(rqtb_q_pair), .k_pair(rqtb_k_pair),
        .out_valid(rqtb_out_valid), .out_ready(common_out_ready),
        .out_last(rqtb_out_last), .out_token_id(rqtb_out_token),
        .out_k_bits(rqtb_out_k), .out_gate_q17(rqtb_out_gate),
        .out_threshold_q8(), .protocol_error(rqtb_error),
        .perf_pairs(rqtb_pairs), .perf_slots(rqtb_slots),
        .perf_equal_pairs(rqtb_equal),
        .perf_quotient_descriptors(rqtb_desc),
        .perf_original_tokens(rqtb_tokens), .perf_active_entries(rqtb_active),
        .perf_class_transactions(rqtb_classes), .perf_exp_transactions(rqtb_exp),
        .perf_emitted_tokens(rqtb_emitted),
        .perf_k_read_transactions(rqtb_k_reads), .perf_k_read_bits(rqtb_k_bits),
        .perf_total_cycles(rqtb_cycles),
        .perf_pair_stall_cycles(rqtb_pair_stalls),
        .perf_descriptor_stall_cycles(rqtb_desc_stalls),
        .perf_output_stall_cycles(rqtb_out_stalls),
        .perf_fifo_occupancy(rqtb_fifo_occ),
        .perf_fifo_max_occupancy(rqtb_fifo_max)
    );

    always #5 clk = ~clk;

    always @(negedge clk) begin
        if (rst_core || backpressure_reseed) begin
            global_cycle <= 0;
            backpressure_lfsr <= 16'h1d3f;
            descriptor_issue_enable <= 1'b0;
            common_out_ready <= 1'b0;
        end else begin
            global_cycle <= global_cycle + 1;
            backpressure_lfsr <= {backpressure_lfsr[14:0],
                backpressure_lfsr[15] ^ backpressure_lfsr[13]
                ^ backpressure_lfsr[12] ^ backpressure_lfsr[10]};
            descriptor_issue_enable <= backpressure_lfsr[0]
                                    || backpressure_lfsr[5];
            common_out_ready <= backpressure_lfsr[2]
                             || backpressure_lfsr[9];
        end
    end

    always @(posedge clk) begin
        integer lane;
        if (!rst_core) begin
            if (fixed_pair_valid && fixed_pair_ready && !u_fixed.pair_commit)
                $fatal(1, "fixed pair rejected id=%0d next=%0d id_legal=%0d score_legal=%0d score0=%0d score1=%0d packet_ready=%0d",
                    fixed_pair_id, u_fixed.u_encoder.next_pair_q,
                    u_fixed.u_encoder.id_legal, u_fixed.u_encoder.score_legal,
                    u_fixed.u_encoder.score0_w, u_fixed.u_encoder.score1_w,
                    u_fixed.packet_ready);
            if (rqtb_pair_valid && rqtb_pair_ready && !u_rqtb.pair_commit)
                $fatal(1, "rqtb pair rejected id=%0d next=%0d id_legal=%0d score_legal=%0d score0=%0d score1=%0d packet_ready=%0d",
                    rqtb_pair_id, u_rqtb.u_encoder.next_pair_q,
                    u_rqtb.u_encoder.id_legal, u_rqtb.u_encoder.score_legal,
                    u_rqtb.u_encoder.score0_w, u_rqtb.u_encoder.score1_w,
                    u_rqtb.packet_ready);
            if (fixed_row_active && !fixed_done)
                fixed_occ_hist[fixed_fifo_occ] = fixed_occ_hist[fixed_fifo_occ] + 1;
            if (rqtb_row_active && !rqtb_done)
                rqtb_occ_hist[rqtb_fifo_occ] = rqtb_occ_hist[rqtb_fifo_occ] + 1;
            if (fixed_out_valid && common_out_ready) begin
                if (fixed_count >= MAX_TOKENS || fixed_seen[fixed_out_token])
                    $fatal(1, "fixed duplicate/out-of-range token=%0d count=%0d",
                        fixed_out_token, fixed_count);
                fixed_seen[fixed_out_token] = 1'b1;
                fixed_token_log[fixed_count] = fixed_out_token;
                fixed_k_log[fixed_count] = fixed_out_k;
                fixed_gate_log[fixed_count] = fixed_out_gate;
                fixed_last_log[fixed_count] = fixed_out_last;
                for (lane = 0; lane < HEAD_DIM; lane = lane + 1)
                    if (fixed_out_k[lane])
                        fixed_acc[lane] = fixed_acc[lane]
                                        + lane_weight(lane) * fixed_out_gate;
                fixed_count = fixed_count + 1;
            end
            if (rqtb_out_valid && common_out_ready) begin
                if (rqtb_count >= MAX_TOKENS || rqtb_seen[rqtb_out_token])
                    $fatal(1, "rqtb duplicate/out-of-range token=%0d count=%0d",
                        rqtb_out_token, rqtb_count);
                rqtb_seen[rqtb_out_token] = 1'b1;
                rqtb_token_log[rqtb_count] = rqtb_out_token;
                rqtb_k_log[rqtb_count] = rqtb_out_k;
                rqtb_gate_log[rqtb_count] = rqtb_out_gate;
                rqtb_last_log[rqtb_count] = rqtb_out_last;
                for (lane = 0; lane < HEAD_DIM; lane = lane + 1)
                    if (rqtb_out_k[lane])
                        rqtb_acc[lane] = rqtb_acc[lane]
                                       + lane_weight(lane) * rqtb_out_gate;
                rqtb_count = rqtb_count + 1;
            end
        end
    end

    task automatic drive_fixed_pairs;
        integer pair;
        integer wait_cycles;
        begin
            for (pair = 0; pair < PAIRS; pair = pair + 1) begin
                if ((pair % 13) == 7) @(negedge clk);
                fixed_pair_id = PAIR_ID_W'(pair);
                fixed_q_pair = {q_vector[pair + PAIRS], q_vector[pair]};
                fixed_k_pair = {k_vector[pair + PAIRS], k_vector[pair]};
                fixed_pair_valid = 1'b1;
                wait_cycles = 0;
                @(posedge clk);
                while (!fixed_pair_ready && wait_cycles < 5000) begin
                    wait_cycles = wait_cycles + 1;
                    @(posedge clk);
                end
                if (!fixed_pair_ready)
                    $fatal(1, "fixed pair wait timeout pair=%0d pairs=%0d slots=%0d occ=%0d decoded=%0d open=%0d error=%0d",
                        pair, fixed_pairs, fixed_slots, fixed_fifo_occ,
                        u_fixed.decoded_pairs_q, u_fixed.pair_open_q, fixed_error);
                @(negedge clk);
                fixed_pair_valid = 1'b0;
            end
            wait_cycles = 0;
            while (!fixed_seal_ready && wait_cycles < 5000) begin
                @(negedge clk);
                wait_cycles = wait_cycles + 1;
            end
            if (!fixed_seal_ready)
                $fatal(1, "fixed seal wait timeout pairs=%0d slots=%0d occ=%0d decoded=%0d open=%0d error=%0d",
                    fixed_pairs, fixed_slots, fixed_fifo_occ,
                    u_fixed.decoded_pairs_q, u_fixed.pair_open_q, fixed_error);
            fixed_seal = 1'b1;
            @(negedge clk);
            fixed_seal = 1'b0;
        end
    endtask

    task automatic wait_fixed_done;
        integer timeout;
        begin
            timeout = 0;
            while (!fixed_done && timeout < 20000) begin
                @(negedge clk);
                timeout = timeout + 1;
            end
            fixed_row_active = 1'b0;
            if (!fixed_done)
                $fatal(1, "fixed row timeout row=%0d", row_tag);
        end
    endtask

    task automatic wait_rqtb_done;
        integer timeout;
        begin
            timeout = 0;
            while (!rqtb_done && timeout < 20000) begin
                @(negedge clk);
                timeout = timeout + 1;
            end
            rqtb_row_active = 1'b0;
            if (!rqtb_done)
                $fatal(1, "rqtb row timeout row=%0d", row_tag);
        end
    endtask

    task automatic drive_rqtb_pairs;
        integer pair;
        integer wait_cycles;
        begin
            for (pair = 0; pair < PAIRS; pair = pair + 1) begin
                if ((pair % 13) == 7) @(negedge clk);
                rqtb_pair_id = PAIR_ID_W'(pair);
                rqtb_q_pair = {q_vector[pair + PAIRS], q_vector[pair]};
                rqtb_k_pair = {k_vector[pair + PAIRS], k_vector[pair]};
                rqtb_pair_valid = 1'b1;
                wait_cycles = 0;
                @(posedge clk);
                while (!rqtb_pair_ready && wait_cycles < 5000) begin
                    wait_cycles = wait_cycles + 1;
                    @(posedge clk);
                end
                if (!rqtb_pair_ready)
                    $fatal(1, "rqtb pair wait timeout pair=%0d pairs=%0d slots=%0d occ=%0d decoded=%0d open=%0d error=%0d",
                        pair, rqtb_pairs, rqtb_slots, rqtb_fifo_occ,
                        u_rqtb.decoded_pairs_q, u_rqtb.pair_open_q, rqtb_error);
                @(negedge clk);
                rqtb_pair_valid = 1'b0;
            end
            wait_cycles = 0;
            while (!rqtb_seal_ready && wait_cycles < 5000) begin
                @(negedge clk);
                wait_cycles = wait_cycles + 1;
            end
            if (!rqtb_seal_ready)
                $fatal(1, "rqtb seal wait timeout pairs=%0d slots=%0d occ=%0d decoded=%0d open=%0d error=%0d",
                    rqtb_pairs, rqtb_slots, rqtb_fifo_occ,
                    u_rqtb.decoded_pairs_q, u_rqtb.pair_open_q, rqtb_error);
            rqtb_seal = 1'b1;
            @(negedge clk);
            rqtb_seal = 1'b0;
        end
    endtask

    task automatic run_loaded_row;
        integer token;
        integer lane;
        integer index;
        integer active_count;
        integer trace_index;
        integer pair;
        integer expected_equal_count;
        begin
            fixed_count = 0;
            rqtb_count = 0;
            active_count = 0;
            expected_equal_count = 0;
            for (lane = 0; lane < HEAD_DIM; lane = lane + 1) begin
                expected_acc[lane] = 0;
                fixed_acc[lane] = 0;
                rqtb_acc[lane] = 0;
            end
            for (token = 0; token < MAX_TOKENS; token = token + 1) begin
                fixed_seen[token] = 1'b0;
                rqtb_seen[token] = 1'b0;
                if (k_vector[token] != 0) begin
                    for (lane = 0; lane < HEAD_DIM; lane = lane + 1)
                        if (k_vector[token][lane])
                            expected_acc[lane] = expected_acc[lane]
                                + lane_weight(lane) * expected_gate[token];
                end
            end
            for (pair = 0; pair < PAIRS; pair = pair + 1) begin
                if (peer_vector[pair] !== k_vector[pair + PAIRS]
                    || peer_vector[pair + PAIRS] !== k_vector[pair])
                    $fatal(1, "peer temporal pairing mismatch row=%0d pair=%0d",
                        row_tag, pair);
                if (motion_score(q_vector[pair], k_vector[pair], peer_vector[pair])
                    == motion_score(q_vector[pair + PAIRS], k_vector[pair + PAIRS],
                                    peer_vector[pair + PAIRS]))
                    expected_equal_count = expected_equal_count + 1;
                if (k_vector[pair] != 0) begin
                    expected_token_log[active_count] = TOKEN_W'(2 * pair);
                    active_count = active_count + 1;
                end
                if (k_vector[pair + PAIRS] != 0) begin
                    expected_token_log[active_count] = TOKEN_W'(2 * pair + 1);
                    active_count = active_count + 1;
                end
            end
            if (active_count != expected_outputs_header)
                $fatal(1, "row header active mismatch row=%0d", row_tag);

            @(posedge clk);
            backpressure_reseed = 1'b1;
            @(posedge clk);
            backpressure_reseed = 1'b0;
            @(negedge clk);
            fixed_pair_id = '0;
            fixed_q_pair = {q_vector[PAIRS], q_vector[0]};
            fixed_k_pair = {k_vector[PAIRS], k_vector[0]};
            fixed_pair_valid = 1'b1;
            rqtb_pair_id = '0;
            rqtb_q_pair = {q_vector[PAIRS], q_vector[0]};
            rqtb_k_pair = {k_vector[PAIRS], k_vector[0]};
            rqtb_pair_valid = 1'b1;
            fixed_start = 1'b1;
            rqtb_start = 1'b1;
            fixed_row_active = 1'b1;
            rqtb_row_active = 1'b1;
            @(negedge clk);
            if (fixed_pair_ready || rqtb_pair_ready
                || u_fixed.pair_commit || u_rqtb.pair_commit)
                $fatal(1, "window_start accepted a pair row=%0d", row_tag);
            fixed_start = 1'b0;
            rqtb_start = 1'b0;
            fixed_pair_valid = 1'b0;
            rqtb_pair_valid = 1'b0;
            fork
                drive_fixed_pairs();
                drive_rqtb_pairs();
                wait_fixed_done();
                wait_rqtb_done();
            join
            if (fixed_error || rqtb_error)
                $fatal(1, "protocol error row=%0d fixed=%0d rqtb=%0d", row_tag,
                    fixed_error, rqtb_error);
            if (fixed_count != active_count || rqtb_count != active_count)
                $fatal(1, "output count mismatch row=%0d expected=%0d fixed=%0d rqtb=%0d",
                    row_tag, active_count, fixed_count, rqtb_count);

            for (index = 0; index < active_count; index = index + 1) begin
                if (fixed_token_log[index] !== expected_token_log[index]
                    || rqtb_token_log[index] !== expected_token_log[index])
                    $fatal(1, "output order mismatch row=%0d index=%0d expected=%0d fixed=%0d rqtb=%0d",
                        row_tag, index, expected_token_log[index],
                        fixed_token_log[index], rqtb_token_log[index]);
                if (fixed_token_log[index] !== rqtb_token_log[index]
                    || fixed_k_log[index] !== rqtb_k_log[index]
                    || fixed_gate_log[index] !== rqtb_gate_log[index]
                    || fixed_last_log[index] !== rqtb_last_log[index])
                    $fatal(1, "fixed/RQTB miter mismatch row=%0d index=%0d", row_tag, index);
                if (fixed_last_log[index] !== (index == active_count - 1)
                    || rqtb_last_log[index] !== (index == active_count - 1))
                    $fatal(1, "last mismatch row=%0d index=%0d active=%0d fixed=%0d rqtb=%0d",
                        row_tag, index, active_count,
                        fixed_last_log[index], rqtb_last_log[index]);
                trace_index = fixed_token_log[index][0]
                    ? PAIRS + (32'(fixed_token_log[index]) >> 1)
                    : (32'(fixed_token_log[index]) >> 1);
                if (fixed_k_log[index] !== k_vector[trace_index]
                    || fixed_gate_log[index] !== expected_gate[trace_index][8:0])
                    $fatal(1, "trace mismatch row=%0d index=%0d token=%0d trace=%0d k=%08x/%08x gate=%0d/%0d",
                        row_tag, index, fixed_token_log[index], trace_index,
                        fixed_k_log[index], k_vector[trace_index],
                        fixed_gate_log[index], expected_gate[trace_index]);
            end
            for (lane = 0; lane < HEAD_DIM; lane = lane + 1)
                if (fixed_acc[lane] != expected_acc[lane]
                    || rqtb_acc[lane] != expected_acc[lane])
                    $fatal(1, "Acc32 mismatch row=%0d lane=%0d expected=%0d fixed=%0d rqtb=%0d",
                        row_tag, lane, expected_acc[lane], fixed_acc[lane], rqtb_acc[lane]);

            if (fixed_pairs != PAIRS || rqtb_pairs != PAIRS
                || fixed_equal != expected_equal_count
                || rqtb_equal != expected_equal_count
                || fixed_slots != MAX_TOKENS || rqtb_slots > fixed_slots
                || fixed_tokens != MAX_TOKENS || rqtb_tokens != MAX_TOKENS
                || fixed_emitted != active_count || rqtb_emitted != active_count
                || fixed_k_reads != active_count || rqtb_k_reads != active_count
                || fixed_k_bits != 32 * active_count || rqtb_k_bits != 32 * active_count
                || fixed_classes != rqtb_classes || rqtb_exp > fixed_exp)
                $fatal(1, "perf contract mismatch row=%0d", row_tag);

            total_checked = total_checked + active_count;
            total_fixed_cycles = total_fixed_cycles + fixed_cycles;
            total_rqtb_cycles = total_rqtb_cycles + rqtb_cycles;
            total_fixed_slots = total_fixed_slots + fixed_slots;
            total_rqtb_slots = total_rqtb_slots + rqtb_slots;
            total_fixed_exp = total_fixed_exp + fixed_exp;
            total_rqtb_exp = total_rqtb_exp + rqtb_exp;
            total_rows = total_rows + 1;
            $display("RQTB_ROW row=%0d stage=%0d block=%0d head=%0d active=%0d equal=%0d fixed_cycles=%0d rqtb_cycles=%0d fixed_slots=%0d rqtb_slots=%0d fixed_desc=%0d rqtb_desc=%0d fixed_exp=%0d rqtb_exp=%0d fixed_pair_stall=%0d rqtb_pair_stall=%0d fixed_desc_stall=%0d rqtb_desc_stall=%0d fixed_out_stall=%0d rqtb_out_stall=%0d fixed_fifo_max=%0d rqtb_fifo_max=%0d",
                row_tag, stage_tag, block_tag, head_tag, active_count,
                rqtb_equal, fixed_cycles, rqtb_cycles, fixed_slots, rqtb_slots,
                fixed_desc, rqtb_desc, fixed_exp, rqtb_exp,
                fixed_pair_stalls, rqtb_pair_stalls,
                fixed_desc_stalls, rqtb_desc_stalls,
                fixed_out_stalls, rqtb_out_stalls,
                fixed_fifo_max, rqtb_fifo_max);
            @(negedge clk);
        end
    endtask

    initial begin
        integer token;
        integer occ;
        clk = 1'b0;
        rst_core = 1'b1;
        descriptor_issue_enable = 1'b0;
        common_out_ready = 1'b0;
        backpressure_lfsr = 16'h1d3f;
        backpressure_reseed = 1'b0;
        global_cycle = 0;
        fixed_start = 1'b0;
        fixed_seal = 1'b0;
        fixed_pair_valid = 1'b0;
        fixed_pair_id = '0;
        fixed_q_pair = '0;
        fixed_k_pair = '0;
        rqtb_start = 1'b0;
        rqtb_seal = 1'b0;
        rqtb_pair_valid = 1'b0;
        rqtb_pair_id = '0;
        rqtb_q_pair = '0;
        rqtb_k_pair = '0;
        fixed_row_active = 1'b0;
        rqtb_row_active = 1'b0;
        total_checked = 0;
        total_rows = 0;
        total_fixed_cycles = 0;
        total_rqtb_cycles = 0;
        total_fixed_slots = 0;
        total_rqtb_slots = 0;
        total_fixed_exp = 0;
        total_rqtb_exp = 0;
        total_errors = 0;
        for (occ = 0; occ <= SLOT_FIFO_DEPTH; occ = occ + 1) begin
            fixed_occ_hist[occ] = 0;
            rqtb_occ_hist[occ] = 0;
        end

        if (!$value$plusargs("VECTORS=%s", vector_path))
            $fatal(1, "missing +VECTORS=<path>");
        if (!$value$plusargs("ROW_LIMIT=%d", row_limit))
            row_limit = 0;
        if ($value$plusargs("DUMP=%s", dump_path)) begin
            $dumpfile(dump_path);
            $dumpvars(2, u_fixed);
            $dumpvars(2, u_rqtb);
        end
        fd = $fopen(vector_path, "r");
        if (fd == 0) $fatal(1, "cannot open vectors: %s", vector_path);
        scan_count = $fscanf(fd, "%d %d", file_rows, file_tokens);
        if (scan_count != 2 || file_rows <= 0 || file_tokens != MAX_TOKENS)
            $fatal(1, "invalid vector header rows=%0d tokens=%0d", file_rows, file_tokens);
        if (row_limit <= 0 || row_limit > file_rows)
            row_limit = file_rows;

        repeat (4) @(negedge clk);
        rst_core = 1'b0;
        for (row_index = 0; row_index < file_rows; row_index = row_index + 1) begin
            scan_count = $fscanf(fd, "%d %d %d %d %d %d",
                row_tag, stage_tag, block_tag, head_tag,
                expected_outputs_header, expected_folded_header);
            if (scan_count != 6 || row_tag != row_index)
                $fatal(1, "invalid row header row=%0d", row_index);
            for (token = 0; token < MAX_TOKENS; token = token + 1) begin
                scan_count = $fscanf(fd, "%h %h %h %d",
                    q_vector[token], k_vector[token], peer_vector[token],
                    expected_gate[token]);
                if (scan_count != 4)
                    $fatal(1, "invalid vector row=%0d token=%0d", row_index, token);
            end
            if (row_index < row_limit)
                run_loaded_row();
        end
        $fclose(fd);

        $write("RQTB_OCC fixed=");
        for (occ = 0; occ <= SLOT_FIFO_DEPTH; occ = occ + 1)
            $write("%0d%s", fixed_occ_hist[occ],
                   occ == SLOT_FIFO_DEPTH ? "\n" : ",");
        $write("RQTB_OCC rqtb=");
        for (occ = 0; occ <= SLOT_FIFO_DEPTH; occ = occ + 1)
            $write("%0d%s", rqtb_occ_hist[occ],
                   occ == SLOT_FIFO_DEPTH ? "\n" : ",");
        $display("PASS H67 RQTB 1S random physical flow rows=%0d checked=%0d fixed_cycles=%0d rqtb_cycles=%0d fixed_slots=%0d rqtb_slots=%0d fixed_exp=%0d rqtb_exp=%0d acc32_mismatch=0",
            total_rows, total_checked, total_fixed_cycles, total_rqtb_cycles,
            total_fixed_slots, total_rqtb_slots, total_fixed_exp, total_rqtb_exp);
        $finish;
    end

    initial begin
        repeat (5000000) @(posedge clk);
        $fatal(1, "global watchdog timeout");
    end
endmodule

`default_nettype wire
