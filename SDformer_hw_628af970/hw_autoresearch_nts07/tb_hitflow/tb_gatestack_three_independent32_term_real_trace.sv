`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off BLKSEQ */
/* verilator lint_off UNUSEDSIGNAL */

module tb_gatestack_three_independent32_term_real_trace #(
    parameter integer STAGE = 0,
    parameter integer HEADS = 3
);
    localparam integer ENGINES = 3;
    localparam integer TOKENS = 162;
    localparam integer LANES = 32;
    localparam integer DIM = HEADS * LANES;
    localparam integer SUPERTILES = HEADS / ENGINES;
    localparam integer EVENT_WAYS = 4;
    localparam integer BANKS = 2;
    localparam integer OUT_TILE = 32;
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
    localparam logic [TAG_W-1:0] TAG_BASE = 32'h31d3_0000;

    logic clk_core;
    logic rst_core;
    logic [2:0] tile_start_valid;
    logic [2:0] tile_start_ready;
    logic [(ENGINES*TAG_W)-1:0] tile_start_tags;
    logic [(ENGINES*OUTPUT_TILE_W)-1:0] tile_start_output_tiles;
    logic [(ENGINES*HEAD_COUNT_W)-1:0] tile_start_head_counts;
    logic [2:0] head_start_valid;
    logic [2:0] head_start_ready;
    logic [(ENGINES*TAG_W)-1:0] head_start_tags;
    logic [(ENGINES*HEAD_COUNT_W)-1:0] head_start_indices;
    logic [(ENGINES*INPUT_CH_W)-1:0] head_start_input_channel_bases;
    logic [2:0] head_start_last;
    logic [2:0] term_valid;
    logic [2:0] term_ready;
    logic [(ENGINES*GATE_W)-1:0] term_gate_codes;
    logic [(ENGINES*LANE_ID_W)-1:0] term_lane_ids;
    logic [(ENGINES*8)-1:0] term_destination_counts;
    logic [(ENGINES*ISSUE_SEQ_W)-1:0] term_issue_seqs;
    logic [2:0] term_head_last;
    logic [2:0] event_valid;
    logic [2:0] event_ready;
    logic [(ENGINES*GATE_W)-1:0] event_gate_codes;
    logic [(ENGINES*LANE_ID_W)-1:0] event_lane_ids;
    logic [(ENGINES*EVENT_WAYS)-1:0] event_token_valids;
    logic [(ENGINES*EVENT_WAYS*TOKEN_ID_W)-1:0] event_token_ids;
    logic [(ENGINES*WAY_COUNT_W)-1:0] event_counts;
    logic [(ENGINES*ISSUE_SEQ_W)-1:0] event_issue_seqs;
    logic [2:0] event_term_first;
    logic [2:0] event_term_last;
    logic [2:0] event_head_last;
    logic [2:0] source_done_valid;
    logic [2:0] source_done_ready;
    logic [(ENGINES*TAG_W)-1:0] source_done_tags;
    logic [2:0] source_done_error;
    logic [2:0] head_done_valid;
    logic [2:0] head_done_ready;
    logic [(ENGINES*TAG_W)-1:0] head_done_tags;
    logic [(ENGINES*HEAD_COUNT_W)-1:0] head_done_indices;
    logic [2:0] head_done_last;
    logic [2:0] head_done_error;
    logic [2:0] weight_req_valid;
    logic [2:0] weight_req_ready;
    logic [(ENGINES*TAG_W)-1:0] weight_req_tags;
    logic [(ENGINES*INPUT_CH_W)-1:0] weight_req_input_channels;
    logic [(ENGINES*OUTPUT_TILE_W)-1:0] weight_req_output_tiles;
    logic [2:0] weight_rsp_valid;
    logic [2:0] weight_rsp_ready;
    logic [(ENGINES*TAG_W)-1:0] weight_rsp_tags;
    logic [(ENGINES*INPUT_CH_W)-1:0] weight_rsp_input_channels;
    logic [(ENGINES*OUTPUT_TILE_W)-1:0] weight_rsp_output_tiles;
    logic [(ENGINES*OUT_TILE*WEIGHT_W)-1:0] weight_rsp_weights;
    logic [2:0] bias_req_valid;
    logic [2:0] bias_req_ready;
    logic [(ENGINES*TAG_W)-1:0] bias_req_tags;
    logic [(ENGINES*OUTPUT_TILE_W)-1:0] bias_req_output_tiles;
    logic [(ENGINES*TOKEN_ID_W)-1:0] bias_req_token_ids;
    logic [2:0] bias_rsp_valid;
    logic [2:0] bias_rsp_ready;
    logic [(ENGINES*TAG_W)-1:0] bias_rsp_tags;
    logic [(ENGINES*TOKEN_ID_W)-1:0] bias_rsp_token_ids;
    logic [(ENGINES*OUT_TILE*ACC_W)-1:0] bias_rsp_values;
    logic [(ENGINES*BANKS)-1:0] final_valid;
    logic [(ENGINES*BANKS)-1:0] final_ready;
    logic [(ENGINES*BANKS*TOKEN_ID_W)-1:0] final_token_ids;
    logic [(ENGINES*TAG_W)-1:0] final_tags;
    logic [(ENGINES*BANKS*OUT_TILE*ACC_W)-1:0] final_values;
    logic [2:0] tile_done_valid;
    logic [2:0] tile_done_ready;
    logic [(ENGINES*TAG_W)-1:0] tile_done_tags;
    logic [2:0] protocol_error;
    logic [2:0] accumulator_overflow;
    logic [(ENGINES*COUNTER_W)-1:0] count_heads;
    logic [(ENGINES*COUNTER_W)-1:0] count_terms;
    logic [(ENGINES*COUNTER_W)-1:0] count_completed_terms;
    logic [(ENGINES*COUNTER_W)-1:0] count_bias_commits;

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
    logic [TOKENS-1:0] final_seen [0:HEADS-1];

    integer total_terms;
    integer total_events;
    integer event_beats_per_replay;
    integer cycle_count;
    integer term_port_read_count;
    integer event_port_read_count;
    integer source_done_port_count;
    integer physical_weight_req_count;
    integer bias_req_count;
    integer final_beat_count;
    integer final_check_count;
    integer current_supertile;
    logic [TAG_W-1:0] current_tag;
    string vector_dir;

    gatestack_three_independent32_term_projection_top #(
        .TOKENS(TOKENS),
        .LANES(LANES),
        .EVENT_WAYS(EVENT_WAYS),
        .BANKS(BANKS),
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

    assign final_ready = {ENGINES*BANKS{1'b1}};
    assign weight_req_ready = ~weight_rsp_valid;
    assign bias_req_ready = ~bias_rsp_valid;

    task automatic start_tile(input integer supertile);
        logic [2:0] accepted;
        integer engine;
        begin
            current_supertile = supertile;
            current_tag = TAG_BASE + TAG_W'(supertile);
            @(negedge clk_core);
            for (engine = 0; engine < ENGINES; engine = engine + 1) begin
                tile_start_tags[(engine*TAG_W) +: TAG_W] = current_tag;
                tile_start_output_tiles[
                    (engine*OUTPUT_TILE_W) +: OUTPUT_TILE_W] =
                    OUTPUT_TILE_W'(supertile * ENGINES + engine);
                tile_start_head_counts[
                    (engine*HEAD_COUNT_W) +: HEAD_COUNT_W] = HEAD_COUNT_W'(HEADS);
            end
            tile_start_valid = 3'b111;
            while (tile_start_valid != 3'b000) begin
                @(posedge clk_core);
                accepted = tile_start_valid & tile_start_ready;
                @(negedge clk_core);
                tile_start_valid = tile_start_valid & ~accepted;
            end
        end
    endtask

    task automatic start_head(input integer head);
        logic [2:0] accepted;
        integer engine;
        begin
            @(negedge clk_core);
            for (engine = 0; engine < ENGINES; engine = engine + 1) begin
                head_start_tags[(engine*TAG_W) +: TAG_W] = current_tag;
                head_start_indices[(engine*HEAD_COUNT_W) +: HEAD_COUNT_W] =
                    HEAD_COUNT_W'(head);
                head_start_input_channel_bases[
                    (engine*INPUT_CH_W) +: INPUT_CH_W] =
                    INPUT_CH_W'(head * LANES);
            end
            head_start_last = {ENGINES{head == HEADS - 1}};
            head_start_valid = 3'b111;
            while (head_start_valid != 3'b000) begin
                @(posedge clk_core);
                accepted = head_start_valid & head_start_ready;
                @(negedge clk_core);
                head_start_valid = head_start_valid & ~accepted;
            end
        end
    endtask

    task automatic drive_term(
        input integer term_index,
        input integer issue_index,
        input logic head_last_term
    );
        logic [2:0] accepted;
        integer engine;
        begin
            @(negedge clk_core);
            for (engine = 0; engine < ENGINES; engine = engine + 1) begin
                term_gate_codes[(engine*GATE_W) +: GATE_W] =
                    term_gate_mem[term_index];
                term_lane_ids[(engine*LANE_ID_W) +: LANE_ID_W] =
                    term_lane_mem[term_index];
                term_destination_counts[(engine*8) +: 8] =
                    term_count_mem[term_index];
                term_issue_seqs[(engine*ISSUE_SEQ_W) +: ISSUE_SEQ_W] =
                    ISSUE_SEQ_W'(issue_index);
            end
            term_head_last = {ENGINES{head_last_term}};
            term_valid = 3'b111;
            while (term_valid != 3'b000) begin
                @(posedge clk_core);
                accepted = term_valid & term_ready;
                @(negedge clk_core);
                term_valid = term_valid & ~accepted;
            end
        end
    endtask

    task automatic drive_event_beat(
        input integer term_index,
        input integer issue_index,
        input integer token_cursor,
        input integer token_end,
        input integer beat_count,
        input logic head_last_term
    );
        logic [2:0] accepted;
        logic [EVENT_WAYS-1:0] beat_valid;
        logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] beat_tokens;
        integer engine;
        integer way;
        begin
            beat_valid = '0;
            beat_tokens = '0;
            for (way = 0; way < EVENT_WAYS; way = way + 1) begin
                if (way < beat_count) begin
                    beat_valid[way] = 1'b1;
                    beat_tokens[(way*TOKEN_ID_W) +: TOKEN_ID_W] =
                        term_token_mem[token_cursor + way];
                end
            end
            @(negedge clk_core);
            for (engine = 0; engine < ENGINES; engine = engine + 1) begin
                event_gate_codes[(engine*GATE_W) +: GATE_W] =
                    term_gate_mem[term_index];
                event_lane_ids[(engine*LANE_ID_W) +: LANE_ID_W] =
                    term_lane_mem[term_index];
                event_token_valids[
                    (engine*EVENT_WAYS) +: EVENT_WAYS] = beat_valid;
                event_token_ids[
                    (engine*EVENT_WAYS*TOKEN_ID_W) +:
                    (EVENT_WAYS*TOKEN_ID_W)] = beat_tokens;
                event_counts[(engine*WAY_COUNT_W) +: WAY_COUNT_W] =
                    WAY_COUNT_W'(beat_count);
                event_issue_seqs[(engine*ISSUE_SEQ_W) +: ISSUE_SEQ_W] =
                    ISSUE_SEQ_W'(issue_index);
            end
            event_term_first = {ENGINES{
                token_cursor == 32'(term_token_offsets_mem[term_index])}};
            event_term_last = {ENGINES{token_cursor + beat_count == token_end}};
            event_head_last = {ENGINES{
                head_last_term && token_cursor + beat_count == token_end}};
            event_valid = 3'b111;
            while (event_valid != 3'b000) begin
                @(posedge clk_core);
                accepted = event_valid & event_ready;
                @(negedge clk_core);
                event_valid = event_valid & ~accepted;
            end
        end
    endtask

    task automatic drive_events(
        input integer term_index,
        input integer issue_index,
        input logic head_last_term
    );
        integer token_cursor;
        integer token_end;
        integer beat_count;
        begin
            token_cursor = term_token_offsets_mem[term_index];
            token_end = term_token_offsets_mem[term_index + 1];
            while (token_cursor < token_end) begin
                beat_count = token_end - token_cursor;
                if (beat_count > EVENT_WAYS)
                    beat_count = EVENT_WAYS;
                drive_event_beat(term_index, issue_index, token_cursor,
                                 token_end, beat_count, head_last_term);
                token_cursor = token_cursor + beat_count;
            end
        end
    endtask

    task automatic finish_source;
        logic [2:0] accepted;
        integer engine;
        begin
            @(negedge clk_core);
            for (engine = 0; engine < ENGINES; engine = engine + 1)
                source_done_tags[(engine*TAG_W) +: TAG_W] = current_tag;
            source_done_error = '0;
            source_done_valid = 3'b111;
            while (source_done_valid != 3'b000) begin
                @(posedge clk_core);
                accepted = source_done_valid & source_done_ready;
                @(negedge clk_core);
                source_done_valid = source_done_valid & ~accepted;
            end
        end
    endtask

    task automatic wait_head_done(input integer head);
        logic [2:0] accepted;
        integer engine;
        begin
            @(negedge clk_core);
            head_done_ready = 3'b111;
            while (head_done_ready != 3'b000) begin
                @(posedge clk_core);
                accepted = head_done_ready & head_done_valid;
                for (engine = 0; engine < ENGINES; engine = engine + 1) begin
                    if (accepted[engine] &&
                        (head_done_tags[(engine*TAG_W) +: TAG_W] !== current_tag ||
                         head_done_indices[
                             (engine*HEAD_COUNT_W) +: HEAD_COUNT_W] !==
                             HEAD_COUNT_W'(head) ||
                         head_done_last[engine] !== (head == HEADS - 1) ||
                         head_done_error[engine]))
                        $fatal(1, "S%0d engine%0d head完成错误: head=%0d",
                               STAGE, engine, head);
                end
                @(negedge clk_core);
                head_done_ready = head_done_ready & ~accepted;
            end
        end
    endtask

    task automatic wait_tile_done;
        logic [2:0] accepted;
        integer engine;
        integer output_head;
        begin
            @(negedge clk_core);
            tile_done_ready = 3'b111;
            while (tile_done_ready != 3'b000) begin
                @(posedge clk_core);
                accepted = tile_done_ready & tile_done_valid;
                for (engine = 0; engine < ENGINES; engine = engine + 1) begin
                    if (accepted[engine] &&
                        tile_done_tags[(engine*TAG_W) +: TAG_W] !== current_tag)
                        $fatal(1, "S%0d engine%0d tile完成tag错误",
                               STAGE, engine);
                end
                @(negedge clk_core);
                tile_done_ready = tile_done_ready & ~accepted;
            end
            if (protocol_error != '0 || accumulator_overflow != '0)
                $fatal(1, "S%0d supertile完成状态错误: protocol=%b overflow=%b",
                       STAGE, protocol_error, accumulator_overflow);
            for (output_head = current_supertile * ENGINES;
                 output_head < current_supertile * ENGINES + ENGINES;
                 output_head = output_head + 1) begin
                if (final_seen[output_head] !== {TOKENS{1'b1}})
                    $fatal(1, "S%0d输出head缺失: head=%0d mask=%h", STAGE,
                           output_head, final_seen[output_head]);
            end
        end
    endtask

    // Each engine has its own fixed-one-cycle weight and bias memory port.
    always_ff @(posedge clk_core) begin : p_fixed_one_cycle_memory
        integer engine;
        integer lane;
        integer input_channel;
        integer output_tile;
        integer token;
        integer weight_index;
        integer bias_index;
        if (rst_core) begin
            weight_rsp_valid <= '0;
            weight_rsp_tags <= '0;
            weight_rsp_input_channels <= '0;
            weight_rsp_output_tiles <= '0;
            weight_rsp_weights <= '0;
            bias_rsp_valid <= '0;
            bias_rsp_tags <= '0;
            bias_rsp_token_ids <= '0;
            bias_rsp_values <= '0;
        end else begin
            for (engine = 0; engine < ENGINES; engine = engine + 1) begin
                if (weight_rsp_valid[engine] && weight_rsp_ready[engine])
                    weight_rsp_valid[engine] <= 1'b0;
                if (weight_req_valid[engine] && weight_req_ready[engine]) begin
                    input_channel = 32'(weight_req_input_channels[
                        (engine*INPUT_CH_W) +: INPUT_CH_W]);
                    output_tile = 32'(weight_req_output_tiles[
                        (engine*OUTPUT_TILE_W) +: OUTPUT_TILE_W]);
                    if (weight_req_tags[(engine*TAG_W) +: TAG_W] !== current_tag ||
                        output_tile != current_supertile * ENGINES + engine ||
                        input_channel < 0 || input_channel >= DIM)
                        $fatal(1, "S%0d engine%0d weight请求身份错误",
                               STAGE, engine);
                    weight_rsp_tags[(engine*TAG_W) +: TAG_W] <=
                        weight_req_tags[(engine*TAG_W) +: TAG_W];
                    weight_rsp_input_channels[
                        (engine*INPUT_CH_W) +: INPUT_CH_W] <=
                        weight_req_input_channels[
                            (engine*INPUT_CH_W) +: INPUT_CH_W];
                    weight_rsp_output_tiles[
                        (engine*OUTPUT_TILE_W) +: OUTPUT_TILE_W] <=
                        weight_req_output_tiles[
                            (engine*OUTPUT_TILE_W) +: OUTPUT_TILE_W];
                    for (lane = 0; lane < OUT_TILE; lane = lane + 1) begin
                        weight_index = ((output_tile * OUT_TILE + lane) * DIM) +
                                       input_channel;
                        weight_rsp_weights[
                            (engine*OUT_TILE*WEIGHT_W) + (lane*WEIGHT_W) +:
                            WEIGHT_W] <= weight_mem[weight_index];
                    end
                    weight_rsp_valid[engine] <= 1'b1;
                end

                if (bias_rsp_valid[engine] && bias_rsp_ready[engine])
                    bias_rsp_valid[engine] <= 1'b0;
                if (bias_req_valid[engine] && bias_req_ready[engine]) begin
                    output_tile = 32'(bias_req_output_tiles[
                        (engine*OUTPUT_TILE_W) +: OUTPUT_TILE_W]);
                    token = 32'(bias_req_token_ids[
                        (engine*TOKEN_ID_W) +: TOKEN_ID_W]);
                    if (bias_req_tags[(engine*TAG_W) +: TAG_W] !== current_tag ||
                        output_tile != current_supertile * ENGINES + engine ||
                        token < 0 || token >= TOKENS)
                        $fatal(1, "S%0d engine%0d bias请求身份错误",
                               STAGE, engine);
                    bias_rsp_tags[(engine*TAG_W) +: TAG_W] <=
                        bias_req_tags[(engine*TAG_W) +: TAG_W];
                    bias_rsp_token_ids[(engine*TOKEN_ID_W) +: TOKEN_ID_W] <=
                        bias_req_token_ids[(engine*TOKEN_ID_W) +: TOKEN_ID_W];
                    for (lane = 0; lane < OUT_TILE; lane = lane + 1) begin
                        bias_index = output_tile * OUT_TILE + lane;
                        bias_rsp_values[
                            (engine*OUT_TILE*ACC_W) + (lane*ACC_W) +: ACC_W] <=
                            bias_mem[bias_index];
                    end
                    bias_rsp_valid[engine] <= 1'b1;
                end
            end
        end
    end

    always @(posedge clk_core) begin : p_scoreboard
        integer engine;
        integer port;
        integer lane;
        integer token;
        integer output_head;
        integer output_channel;
        integer expected_index;
        if (rst_core) begin
            cycle_count = 0;
        end else begin
            cycle_count = cycle_count + 1;
            if (protocol_error != '0)
                $fatal(1, "S%0d DUT protocol_error=%b", STAGE, protocol_error);
            if (accumulator_overflow != '0)
                $fatal(1, "S%0d DUT accumulator_overflow=%b",
                       STAGE, accumulator_overflow);
            for (engine = 0; engine < ENGINES; engine = engine + 1) begin
                if (term_valid[engine] && term_ready[engine])
                    term_port_read_count = term_port_read_count + 1;
                if (event_valid[engine] && event_ready[engine])
                    event_port_read_count = event_port_read_count + 1;
                if (source_done_valid[engine] && source_done_ready[engine])
                    source_done_port_count = source_done_port_count + 1;
                if (weight_req_valid[engine] && weight_req_ready[engine])
                    physical_weight_req_count = physical_weight_req_count + 1;
                if (bias_req_valid[engine] && bias_req_ready[engine])
                    bias_req_count = bias_req_count + 1;
            end
            for (port = 0; port < ENGINES*BANKS; port = port + 1) begin
                if (final_valid[port] && final_ready[port]) begin
                    engine = port / BANKS;
                    token = 32'(final_token_ids[
                        (port*TOKEN_ID_W) +: TOKEN_ID_W]);
                    output_head = current_supertile * ENGINES + engine;
                    if (final_tags[(engine*TAG_W) +: TAG_W] !== current_tag ||
                        token < 0 || token >= TOKENS)
                        $fatal(1, "S%0d final身份错误: port=%0d token=%0d",
                               STAGE, port, token);
                    if (final_seen[output_head][token])
                        $fatal(1, "S%0d final重复: head=%0d token=%0d",
                               STAGE, output_head, token);
                    for (lane = 0; lane < OUT_TILE; lane = lane + 1) begin
                        output_channel = output_head * OUT_TILE + lane;
                        expected_index = token * DIM + output_channel;
                        if (final_values[
                                (port*OUT_TILE*ACC_W) + (lane*ACC_W) +: ACC_W]
                            !== expected_mem[expected_index])
                            $fatal(1,
                                "S%0d bit-exact失败: supertile=%0d port=%0d token=%0d lane=%0d expected=%0d actual=%0d",
                                STAGE, current_supertile, port, token, lane,
                                $signed(expected_mem[expected_index]),
                                $signed(final_values[
                                    (port*OUT_TILE*ACC_W) + (lane*ACC_W) +:
                                    ACC_W]));
                        final_check_count = final_check_count + 1;
                    end
                    final_seen[output_head][token] = 1'b1;
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
        integer token_count;
        integer expected_term_port_reads;
        integer expected_event_port_reads;
        integer expected_source_done_ports;
        integer expected_weight_requests;
        integer expected_bias_requests;
        integer expected_final_beats;
        integer expected_final_checks;
        integer engine;

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

        event_beats_per_replay = 0;
        for (term_index = 0; term_index < total_terms;
             term_index = term_index + 1) begin
            token_count = term_token_offsets_mem[term_index + 1] -
                          term_token_offsets_mem[term_index];
            event_beats_per_replay = event_beats_per_replay +
                                     ((token_count + EVENT_WAYS - 1) /
                                      EVENT_WAYS);
        end

        clk_core = 1'b0;
        rst_core = 1'b1;
        tile_start_valid = '0;
        tile_start_tags = '0;
        tile_start_output_tiles = '0;
        tile_start_head_counts = '0;
        head_start_valid = '0;
        head_start_tags = '0;
        head_start_indices = '0;
        head_start_input_channel_bases = '0;
        head_start_last = '0;
        term_valid = '0;
        term_gate_codes = '0;
        term_lane_ids = '0;
        term_destination_counts = '0;
        term_issue_seqs = '0;
        term_head_last = '0;
        event_valid = '0;
        event_gate_codes = '0;
        event_lane_ids = '0;
        event_token_valids = '0;
        event_token_ids = '0;
        event_counts = '0;
        event_issue_seqs = '0;
        event_term_first = '0;
        event_term_last = '0;
        event_head_last = '0;
        source_done_valid = '0;
        source_done_tags = '0;
        source_done_error = '0;
        head_done_ready = '0;
        tile_done_ready = '0;
        cycle_count = 0;
        term_port_read_count = 0;
        event_port_read_count = 0;
        source_done_port_count = 0;
        physical_weight_req_count = 0;
        bias_req_count = 0;
        final_beat_count = 0;
        final_check_count = 0;
        current_supertile = 0;
        current_tag = TAG_BASE;
        for (head = 0; head < HEADS; head = head + 1)
            final_seen[head] = '0;

        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        for (supertile = 0; supertile < SUPERTILES;
             supertile = supertile + 1) begin
            start_tile(supertile);
            for (head = 0; head < HEADS; head = head + 1) begin
                start_head(head);
                term_begin = head_term_offsets_mem[head];
                term_end = head_term_offsets_mem[head + 1];
                issue_index = 0;
                for (term_index = term_begin; term_index < term_end;
                     term_index = term_index + 1) begin
                    drive_term(term_index, issue_index,
                               term_index == term_end - 1);
                    drive_events(term_index, issue_index,
                                 term_index == term_end - 1);
                    issue_index = issue_index + 1;
                end
                finish_source();
                wait_head_done(head);
            end
            wait_tile_done();
        end

        expected_term_port_reads = total_terms * SUPERTILES * ENGINES;
        expected_event_port_reads = event_beats_per_replay * SUPERTILES * ENGINES;
        expected_source_done_ports = HEADS * SUPERTILES * ENGINES;
        expected_weight_requests = expected_term_port_reads;
        expected_bias_requests = TOKENS * SUPERTILES * ENGINES;
        expected_final_beats = TOKENS * HEADS;
        expected_final_checks = TOKENS * DIM;
        for (engine = 0; engine < ENGINES; engine = engine + 1) begin
            if (count_heads[(engine*COUNTER_W) +: COUNTER_W] !=
                    COUNTER_W'(HEADS * SUPERTILES) ||
                count_terms[(engine*COUNTER_W) +: COUNTER_W] !=
                    COUNTER_W'(total_terms * SUPERTILES) ||
                count_completed_terms[(engine*COUNTER_W) +: COUNTER_W] !=
                    COUNTER_W'(total_terms * SUPERTILES) ||
                count_bias_commits[(engine*COUNTER_W) +: COUNTER_W] !=
                    COUNTER_W'(TOKENS * SUPERTILES))
                $fatal(1, "S%0d engine%0d DUT计数错误", STAGE, engine);
        end
        if (term_port_read_count != expected_term_port_reads ||
            event_port_read_count != expected_event_port_reads ||
            source_done_port_count != expected_source_done_ports ||
            physical_weight_req_count != expected_weight_requests ||
            bias_req_count != expected_bias_requests ||
            final_beat_count != expected_final_beats ||
            final_check_count != expected_final_checks)
            $fatal(1,
                "S%0d TB计数错误: term_reads=%0d/%0d event_reads=%0d/%0d done=%0d/%0d weight=%0d/%0d bias=%0d/%0d beats=%0d/%0d checks=%0d/%0d",
                STAGE, term_port_read_count, expected_term_port_reads,
                event_port_read_count, expected_event_port_reads,
                source_done_port_count, expected_source_done_ports,
                physical_weight_req_count, expected_weight_requests,
                bias_req_count, expected_bias_requests,
                final_beat_count, expected_final_beats,
                final_check_count, expected_final_checks);

        $display("PASS THREE_INDEPENDENT32 TERM REAL TRACE stage=S%0d heads=%0d cycles=%0d logical_terms=%0d term_port_reads=%0d event_port_reads=%0d source_done_ports=%0d weight_req=%0d bias_req=%0d final_checks=%0d",
                 STAGE, HEADS, cycle_count, total_terms * SUPERTILES,
                 term_port_read_count, event_port_read_count,
                 source_done_port_count, physical_weight_req_count,
                 bias_req_count, final_check_count);
        $finish;
    end

    initial begin
        repeat (3000000) @(posedge clk_core);
        $fatal(1, "3xIndependent32真实trace TB超时");
    end
endmodule

`default_nettype wire
