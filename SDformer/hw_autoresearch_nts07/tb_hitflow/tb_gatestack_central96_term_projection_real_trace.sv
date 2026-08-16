`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off BLKSEQ */
/* verilator lint_off UNUSEDSIGNAL */

module tb_gatestack_central96_term_projection_real_trace #(
    parameter integer STAGE = 0,
    parameter integer HEADS = 3
);
    localparam integer TOKENS = 162;
    localparam integer LANES = 32;
    localparam integer DIM = HEADS * LANES;
    localparam integer SUPERTILES = HEADS / 3;
    localparam integer EVENT_WAYS = 4;
    localparam integer OUT_TILE = 96;
    localparam integer BANKS = 2;
    localparam integer PHYSICAL_SLICES = 3;
    localparam integer GATE_W = 9;
    localparam integer WEIGHT_W = 8;
    localparam integer PRODUCT_W = GATE_W + WEIGHT_W;
    localparam integer ACC_W = 32;
    localparam integer TAG_W = 32;
    localparam integer ISSUE_SEQ_W = 13;
    localparam integer INPUT_CH_W = 10;
    localparam integer LANE_ID_W = 5;
    localparam integer TOKEN_ID_W = 8;
    localparam integer OUTPUT_TILE_W = 5;
    localparam integer HEAD_COUNT_W = 5;
    localparam integer COUNTER_W = 32;
    localparam integer WAY_COUNT_W = 3;
    localparam integer MAX_TERMS = HEADS * LANES * TOKENS;
    localparam integer MAX_EVENTS = HEADS * LANES * TOKENS;
    localparam logic [TAG_W-1:0] TAG_BASE = 32'hce96_0000;

    logic clk_core;
    logic rst_core;
    logic tile_start_valid;
    logic tile_start_ready;
    logic [TAG_W-1:0] tile_start_tag;
    logic [OUTPUT_TILE_W-1:0] tile_start_output_tile;
    logic [HEAD_COUNT_W-1:0] tile_start_head_count;
    logic head_start_valid;
    logic head_start_ready;
    logic [TAG_W-1:0] head_start_tag;
    logic [HEAD_COUNT_W-1:0] head_start_index;
    logic [INPUT_CH_W-1:0] head_start_input_channel_base;
    logic head_start_last;
    logic term_valid;
    logic term_ready;
    logic [GATE_W-1:0] term_gate_code;
    logic [LANE_ID_W-1:0] term_lane_id;
    logic [7:0] term_destination_count;
    logic [ISSUE_SEQ_W-1:0] term_issue_seq;
    logic term_head_last;
    logic event_valid;
    logic event_ready;
    logic [GATE_W-1:0] event_gate_code;
    logic [LANE_ID_W-1:0] event_lane_id;
    logic [EVENT_WAYS-1:0] event_token_valid;
    logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] event_token_ids;
    logic [WAY_COUNT_W-1:0] event_count;
    logic [ISSUE_SEQ_W-1:0] event_issue_seq;
    logic event_term_first;
    logic event_term_last;
    logic event_head_last;
    logic source_done_valid;
    logic source_done_ready;
    logic [TAG_W-1:0] source_done_tag;
    logic source_done_error;
    logic head_done_valid;
    logic head_done_ready;
    logic [TAG_W-1:0] head_done_tag;
    logic [HEAD_COUNT_W-1:0] head_done_index;
    logic head_done_last;
    logic head_done_error;
    logic weight_req_valid;
    logic weight_req_ready;
    logic [TAG_W-1:0] weight_req_tag;
    logic [INPUT_CH_W-1:0] weight_req_input_channel;
    logic [OUTPUT_TILE_W-1:0] weight_req_output_tile;
    logic weight_rsp_valid;
    logic weight_rsp_ready;
    logic [TAG_W-1:0] weight_rsp_tag;
    logic [INPUT_CH_W-1:0] weight_rsp_input_channel;
    logic [OUTPUT_TILE_W-1:0] weight_rsp_output_tile;
    logic [(OUT_TILE*WEIGHT_W)-1:0] weight_rsp_weights;
    logic bias_req_valid;
    logic bias_req_ready;
    logic [TAG_W-1:0] bias_req_tag;
    logic [OUTPUT_TILE_W-1:0] bias_req_output_tile;
    logic [TOKEN_ID_W-1:0] bias_req_token_id;
    logic bias_rsp_valid;
    logic bias_rsp_ready;
    logic [TAG_W-1:0] bias_rsp_tag;
    logic [TOKEN_ID_W-1:0] bias_rsp_token_id;
    logic [(OUT_TILE*ACC_W)-1:0] bias_rsp_values;
    logic [BANKS-1:0] final_valid;
    logic [BANKS-1:0] final_ready;
    logic [(BANKS*TOKEN_ID_W)-1:0] final_token_ids;
    logic [TAG_W-1:0] final_tag;
    logic [(BANKS*OUT_TILE*ACC_W)-1:0] final_values;
    logic tile_done_valid;
    logic tile_done_ready;
    logic [TAG_W-1:0] tile_done_tag;
    logic protocol_error;
    logic accumulator_overflow;
    logic [COUNTER_W-1:0] count_heads;
    logic [COUNTER_W-1:0] count_terms;
    logic [COUNTER_W-1:0] count_completed_terms;
    logic [COUNTER_W-1:0] count_bias_commits;

    logic [31:0] metadata_mem [0:8];
    logic [31:0] head_term_offsets_mem [0:HEADS];
    logic [31:0] term_token_offsets_mem [0:MAX_TERMS];
    logic [GATE_W-1:0] term_gate_mem [0:MAX_TERMS-1];
    logic [LANE_ID_W-1:0] term_lane_mem [0:MAX_TERMS-1];
    logic [7:0] term_count_mem [0:MAX_TERMS-1];
    logic [TOKEN_ID_W-1:0] term_token_mem [0:MAX_EVENTS-1];
    logic [WEIGHT_W-1:0] weight_mem [0:(DIM*DIM)-1];
    logic [ACC_W-1:0] bias_mem [0:DIM-1];
    logic [ACC_W-1:0] expected_mem [0:(TOKENS*DIM)-1];
    logic [TOKENS-1:0] final_seen [0:SUPERTILES-1];

    integer total_terms;
    integer total_events;
    integer cycle_count;
    integer logical_weight_req_count;
    integer physical_weight_access_count;
    integer logical_bias_req_count;
    integer physical_bias_access_count;
    integer final_beat_count;
    integer final_check_count;
    integer current_supertile;
    logic [TAG_W-1:0] current_tag;
    string vector_dir;

    gatestack_multihead_tile_projection_top #(
        .TOKENS(TOKENS),
        .LANES(LANES),
        .EVENT_WAYS(EVENT_WAYS),
        .OUT_TILE(OUT_TILE),
        .BANKS(BANKS),
        .SEGMENT_TOKENS(18),
        .GATE_W(GATE_W),
        .WEIGHT_W(WEIGHT_W),
        .PRODUCT_W(PRODUCT_W),
        .ACC_W(ACC_W),
        .TAG_W(TAG_W),
        .INPUT_CH_W(INPUT_CH_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .HEAD_COUNT_W(HEAD_COUNT_W),
        .COUNTER_W(COUNTER_W),
        .TOKEN_ID_W(TOKEN_ID_W),
        .LANE_ID_W(LANE_ID_W),
        .WAY_COUNT_W(WAY_COUNT_W)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    assign head_done_ready = 1'b1;
    assign tile_done_ready = 1'b1;
    assign final_ready = {BANKS{1'b1}};
    assign weight_req_ready = ~weight_rsp_valid;
    assign bias_req_ready = ~bias_rsp_valid;

    task automatic start_tile(input integer supertile);
        begin
            current_supertile = supertile;
            current_tag = TAG_BASE + TAG_W'(supertile);
            @(negedge clk_core);
            tile_start_tag = current_tag;
            tile_start_output_tile = OUTPUT_TILE_W'(supertile);
            tile_start_head_count = HEAD_COUNT_W'(HEADS);
            tile_start_valid = 1'b1;
            do @(posedge clk_core); while (!tile_start_ready);
            @(negedge clk_core);
            tile_start_valid = 1'b0;
        end
    endtask

    task automatic start_head(input integer head);
        begin
            @(negedge clk_core);
            head_start_tag = current_tag;
            head_start_index = HEAD_COUNT_W'(head);
            head_start_input_channel_base = INPUT_CH_W'(head * LANES);
            head_start_last = head == HEADS - 1;
            head_start_valid = 1'b1;
            do @(posedge clk_core); while (!head_start_ready);
            @(negedge clk_core);
            head_start_valid = 1'b0;
        end
    endtask

    task automatic drive_term(
        input integer term_index,
        input integer issue_index,
        input logic head_last_term
    );
        integer token_cursor;
        integer token_end;
        integer beat_count;
        integer way;
        begin
            @(negedge clk_core);
            term_gate_code = term_gate_mem[term_index];
            term_lane_id = term_lane_mem[term_index];
            term_destination_count = term_count_mem[term_index];
            term_issue_seq = ISSUE_SEQ_W'(issue_index);
            term_head_last = head_last_term;
            term_valid = 1'b1;
            do @(posedge clk_core); while (!term_ready);
            @(negedge clk_core);
            term_valid = 1'b0;

            token_cursor = term_token_offsets_mem[term_index];
            token_end = term_token_offsets_mem[term_index + 1];
            while (token_cursor < token_end) begin
                beat_count = token_end - token_cursor;
                if (beat_count > EVENT_WAYS)
                    beat_count = EVENT_WAYS;
                @(negedge clk_core);
                event_gate_code = term_gate_mem[term_index];
                event_lane_id = term_lane_mem[term_index];
                event_token_valid = '0;
                event_token_ids = '0;
                for (way = 0; way < EVENT_WAYS; way = way + 1) begin
                    if (way < beat_count) begin
                        event_token_valid[way] = 1'b1;
                        event_token_ids[(way*TOKEN_ID_W) +: TOKEN_ID_W] =
                            term_token_mem[token_cursor + way];
                    end
                end
                event_count = WAY_COUNT_W'(beat_count);
                event_issue_seq = ISSUE_SEQ_W'(issue_index);
                event_term_first =
                    token_cursor == term_token_offsets_mem[term_index];
                event_term_last = token_cursor + beat_count == token_end;
                event_head_last = event_term_last && head_last_term;
                event_valid = 1'b1;
                do @(posedge clk_core); while (!event_ready);
                @(negedge clk_core);
                event_valid = 1'b0;
                token_cursor = token_cursor + beat_count;
            end
        end
    endtask

    task automatic finish_source;
        begin
            @(negedge clk_core);
            source_done_tag = current_tag;
            source_done_error = 1'b0;
            source_done_valid = 1'b1;
            do @(posedge clk_core); while (!source_done_ready);
            @(negedge clk_core);
            source_done_valid = 1'b0;
        end
    endtask

    task automatic wait_head_done(input integer head);
        begin
            do @(posedge clk_core); while (!head_done_valid);
            if (head_done_tag !== current_tag ||
                head_done_index !== HEAD_COUNT_W'(head) ||
                head_done_last !== (head == HEADS - 1) || head_done_error)
                $fatal(1, "S%0d head完成错误: head=%0d", STAGE, head);
            @(negedge clk_core);
        end
    endtask

    task automatic wait_tile_done;
        begin
            do @(posedge clk_core); while (!tile_done_valid);
            if (tile_done_tag !== current_tag || protocol_error ||
                accumulator_overflow)
                $fatal(1, "S%0d supertile完成状态错误: supertile=%0d", STAGE,
                       current_supertile);
            if (final_seen[current_supertile] !== {TOKENS{1'b1}})
                $fatal(1, "S%0d输出supertile缺失: supertile=%0d mask=%h",
                       STAGE, current_supertile,
                       final_seen[current_supertile]);
            @(negedge clk_core);
        end
    endtask

    // 行为模型只表达固定一拍接口。每个96-lane响应显式拼接3个32-lane切片。
    always_ff @(posedge clk_core) begin : p_fixed_one_cycle_memory
        integer physical_bank;
        integer lane;
        integer input_channel;
        integer output_tile;
        integer token;
        integer output_channel;
        integer weight_index;
        integer bias_index;
        if (rst_core) begin
            weight_rsp_valid <= 1'b0;
            weight_rsp_tag <= '0;
            weight_rsp_input_channel <= '0;
            weight_rsp_output_tile <= '0;
            weight_rsp_weights <= '0;
            bias_rsp_valid <= 1'b0;
            bias_rsp_tag <= '0;
            bias_rsp_token_id <= '0;
            bias_rsp_values <= '0;
        end else begin
            if (weight_rsp_valid && weight_rsp_ready)
                weight_rsp_valid <= 1'b0;
            if (weight_req_valid && weight_req_ready) begin
                input_channel = 32'(weight_req_input_channel);
                output_tile = 32'(weight_req_output_tile);
                if (weight_req_tag !== current_tag ||
                    output_tile != current_supertile ||
                    input_channel < 0 || input_channel >= DIM)
                    $fatal(1, "S%0d Central96 weight请求身份错误", STAGE);
                weight_rsp_tag <= weight_req_tag;
                weight_rsp_input_channel <= weight_req_input_channel;
                weight_rsp_output_tile <= weight_req_output_tile;
                for (physical_bank = 0; physical_bank < PHYSICAL_SLICES;
                     physical_bank = physical_bank + 1) begin
                    for (lane = 0; lane < LANES; lane = lane + 1) begin
                        output_channel =
                            ((output_tile * PHYSICAL_SLICES + physical_bank) *
                             LANES) + lane;
                        weight_index = output_channel * DIM + input_channel;
                        weight_rsp_weights[
                            ((physical_bank*LANES + lane)*WEIGHT_W) +:
                            WEIGHT_W] <= weight_mem[weight_index];
                    end
                end
                weight_rsp_valid <= 1'b1;
            end

            if (bias_rsp_valid && bias_rsp_ready)
                bias_rsp_valid <= 1'b0;
            if (bias_req_valid && bias_req_ready) begin
                output_tile = 32'(bias_req_output_tile);
                token = 32'(bias_req_token_id);
                if (bias_req_tag !== current_tag ||
                    output_tile != current_supertile ||
                    token < 0 || token >= TOKENS)
                    $fatal(1, "S%0d Central96 bias请求身份错误", STAGE);
                bias_rsp_tag <= bias_req_tag;
                bias_rsp_token_id <= bias_req_token_id;
                for (physical_bank = 0; physical_bank < PHYSICAL_SLICES;
                     physical_bank = physical_bank + 1) begin
                    for (lane = 0; lane < LANES; lane = lane + 1) begin
                        output_channel =
                            ((output_tile * PHYSICAL_SLICES + physical_bank) *
                             LANES) + lane;
                        bias_index = output_channel;
                        bias_rsp_values[
                            ((physical_bank*LANES + lane)*ACC_W) +:
                            ACC_W] <= bias_mem[bias_index];
                    end
                end
                bias_rsp_valid <= 1'b1;
            end
        end
    end

    always @(posedge clk_core) begin : p_scoreboard
        integer port;
        integer lane;
        integer token;
        integer output_channel;
        integer expected_index;
        if (rst_core) begin
            cycle_count = 0;
        end else begin
            cycle_count = cycle_count + 1;
            if (protocol_error)
                $fatal(1, "S%0d DUT protocol_error", STAGE);
            if (accumulator_overflow)
                $fatal(1, "S%0d DUT accumulator_overflow", STAGE);
            if (weight_req_valid && weight_req_ready) begin
                logical_weight_req_count = logical_weight_req_count + 1;
                physical_weight_access_count = physical_weight_access_count +
                                               PHYSICAL_SLICES;
            end
            if (bias_req_valid && bias_req_ready) begin
                logical_bias_req_count = logical_bias_req_count + 1;
                physical_bias_access_count = physical_bias_access_count +
                                             PHYSICAL_SLICES;
            end
            for (port = 0; port < BANKS; port = port + 1) begin
                if (final_valid[port] && final_ready[port]) begin
                    token = 32'(final_token_ids[
                        (port*TOKEN_ID_W) +: TOKEN_ID_W]);
                    if (final_tag !== current_tag ||
                        token < 0 || token >= TOKENS)
                        $fatal(1, "S%0d final身份错误: port=%0d token=%0d",
                               STAGE, port, token);
                    if (final_seen[current_supertile][token])
                        $fatal(1, "S%0d final重复: supertile=%0d token=%0d",
                               STAGE, current_supertile, token);
                    for (lane = 0; lane < OUT_TILE; lane = lane + 1) begin
                        output_channel = current_supertile * OUT_TILE + lane;
                        expected_index = token * DIM + output_channel;
                        if (final_values[
                                (port*OUT_TILE*ACC_W) + (lane*ACC_W) +:
                                ACC_W] !== expected_mem[expected_index])
                            $fatal(1,
                                "S%0d bit-exact失败: supertile=%0d port=%0d token=%0d lane=%0d expected=%0d actual=%0d",
                                STAGE, current_supertile, port, token, lane,
                                $signed(expected_mem[expected_index]),
                                $signed(final_values[
                                    (port*OUT_TILE*ACC_W) + (lane*ACC_W) +:
                                    ACC_W]));
                        final_check_count = final_check_count + 1;
                    end
                    final_seen[current_supertile][token] = 1'b1;
                    final_beat_count = final_beat_count + 1;
                end
            end
        end
    end

    initial begin : p_test
        integer head;
        integer supertile;
        integer term_index;
        integer term_begin;
        integer term_end;
        integer issue_index;
        integer expected_issued_terms;
        integer expected_logical_weight_requests;
        integer expected_physical_weight_accesses;
        integer expected_logical_bias_requests;
        integer expected_physical_bias_accesses;
        integer expected_final_beats;
        integer expected_final_checks;

        if (!$value$plusargs("VECTOR_DIR=%s", vector_dir))
            $fatal(1, "缺少+VECTOR_DIR=<path>");
        $readmemh($sformatf("%s/metadata.memh", vector_dir), metadata_mem);
        if (metadata_mem[0] != 1 || metadata_mem[1] != STAGE ||
            metadata_mem[2] != HEADS || metadata_mem[3] != SUPERTILES ||
            metadata_mem[4] != DIM || metadata_mem[7] != TOKENS ||
            metadata_mem[8] != LANES)
            $fatal(1, "S%0d向量metadata与TB参数不一致", STAGE);
        total_terms = metadata_mem[5];
        total_events = metadata_mem[6];
        $readmemh($sformatf("%s/head_term_offsets.memh", vector_dir),
                  head_term_offsets_mem);
        $readmemh($sformatf("%s/term_token_offsets.memh", vector_dir),
                  term_token_offsets_mem, 0, total_terms);
        if (total_terms > 0) begin
            $readmemh($sformatf("%s/term_gate_codes.memh", vector_dir),
                      term_gate_mem, 0, total_terms - 1);
            $readmemh($sformatf("%s/term_lane_ids.memh", vector_dir),
                      term_lane_mem, 0, total_terms - 1);
            $readmemh($sformatf("%s/term_destination_counts.memh", vector_dir),
                      term_count_mem, 0, total_terms - 1);
        end
        if (total_events > 0)
            $readmemh($sformatf("%s/term_tokens.memh", vector_dir),
                      term_token_mem, 0, total_events - 1);
        $readmemh($sformatf("%s/projection_weights_int8.memh", vector_dir),
                  weight_mem);
        $readmemh($sformatf("%s/projection_bias_acc32.memh", vector_dir),
                  bias_mem);
        $readmemh($sformatf("%s/expected_output_acc32.memh", vector_dir),
                  expected_mem);

        if (head_term_offsets_mem[HEADS] != total_terms ||
            term_token_offsets_mem[total_terms] != total_events)
            $fatal(1, "S%0d stream offset尾界错误", STAGE);

        clk_core = 1'b0;
        rst_core = 1'b1;
        tile_start_valid = 1'b0;
        tile_start_tag = '0;
        tile_start_output_tile = '0;
        tile_start_head_count = '0;
        head_start_valid = 1'b0;
        head_start_tag = '0;
        head_start_index = '0;
        head_start_input_channel_base = '0;
        head_start_last = 1'b0;
        term_valid = 1'b0;
        term_gate_code = '0;
        term_lane_id = '0;
        term_destination_count = '0;
        term_issue_seq = '0;
        term_head_last = 1'b0;
        event_valid = 1'b0;
        event_gate_code = '0;
        event_lane_id = '0;
        event_token_valid = '0;
        event_token_ids = '0;
        event_count = '0;
        event_issue_seq = '0;
        event_term_first = 1'b0;
        event_term_last = 1'b0;
        event_head_last = 1'b0;
        source_done_valid = 1'b0;
        source_done_tag = '0;
        source_done_error = 1'b0;
        cycle_count = 0;
        logical_weight_req_count = 0;
        physical_weight_access_count = 0;
        logical_bias_req_count = 0;
        physical_bias_access_count = 0;
        final_beat_count = 0;
        final_check_count = 0;
        current_supertile = 0;
        current_tag = TAG_BASE;
        for (supertile = 0; supertile < SUPERTILES;
             supertile = supertile + 1)
            final_seen[supertile] = '0;

        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        for (supertile = 0; supertile < SUPERTILES;
             supertile = supertile + 1) begin
            start_tile(supertile);
            for (head = 0; head < HEADS; head = head + 1) begin
                start_head(head);
                issue_index = 0;
                term_begin = head_term_offsets_mem[head];
                term_end = head_term_offsets_mem[head + 1];
                for (term_index = term_begin; term_index < term_end;
                     term_index = term_index + 1) begin
                    drive_term(term_index, issue_index,
                               term_index == term_end - 1);
                    issue_index = issue_index + 1;
                end
                finish_source();
                wait_head_done(head);
            end
            wait_tile_done();
        end

        expected_issued_terms = total_terms * SUPERTILES;
        expected_logical_weight_requests = expected_issued_terms;
        expected_physical_weight_accesses =
            expected_logical_weight_requests * PHYSICAL_SLICES;
        expected_logical_bias_requests = TOKENS * SUPERTILES;
        expected_physical_bias_accesses =
            expected_logical_bias_requests * PHYSICAL_SLICES;
        expected_final_beats = TOKENS * SUPERTILES;
        expected_final_checks = TOKENS * DIM;
        if (count_heads != COUNTER_W'(HEADS * SUPERTILES) ||
            count_terms != COUNTER_W'(expected_issued_terms) ||
            count_completed_terms != COUNTER_W'(expected_issued_terms) ||
            count_bias_commits != COUNTER_W'(expected_logical_bias_requests))
            $fatal(1,
                "S%0d DUT计数错误: heads=%0d terms=%0d completed=%0d bias=%0d",
                STAGE, count_heads, count_terms, count_completed_terms,
                count_bias_commits);
        if (logical_weight_req_count != expected_logical_weight_requests ||
            physical_weight_access_count != expected_physical_weight_accesses ||
            logical_bias_req_count != expected_logical_bias_requests ||
            physical_bias_access_count != expected_physical_bias_accesses ||
            final_beat_count != expected_final_beats ||
            final_check_count != expected_final_checks)
            $fatal(1,
                "S%0d TB计数错误: logical_weight=%0d/%0d physical_weight=%0d/%0d logical_bias=%0d/%0d physical_bias=%0d/%0d beats=%0d/%0d checks=%0d/%0d",
                STAGE, logical_weight_req_count,
                expected_logical_weight_requests,
                physical_weight_access_count,
                expected_physical_weight_accesses, logical_bias_req_count,
                expected_logical_bias_requests, physical_bias_access_count,
                expected_physical_bias_accesses, final_beat_count,
                expected_final_beats, final_check_count,
                expected_final_checks);

        $display("PASS CENTRAL96 REAL TRACE stage=S%0d heads=%0d cycles=%0d terms=%0d logical_weight_req=%0d physical_weight_access=%0d logical_bias_req=%0d physical_bias_access=%0d final_beats=%0d final_checks=%0d",
                 STAGE, HEADS, cycle_count, expected_issued_terms,
                 logical_weight_req_count, physical_weight_access_count,
                 logical_bias_req_count, physical_bias_access_count,
                 final_beat_count, final_check_count);
        $finish;
    end

    initial begin
        repeat (2000000) @(posedge clk_core);
        $fatal(1, "Central96真实trace TB超时");
    end
endmodule

`default_nettype wire
