`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_three_independent32_projection_top;
    localparam int ENGINES = 3;
    localparam int TOKENS = 4;
    localparam int LANES = 32;
    localparam int OUT_TILE = 32;
    localparam int BANKS = 2;
    localparam int TAG_W = 16;
    localparam int INPUT_CH_W = 6;
    localparam int OUTPUT_TILE_W = 4;
    localparam int HEAD_COUNT_W = 3;
    localparam int WORD_INDEX_W = 7;
    localparam int EVENT_COUNT_W = 13;
    localparam int COUNTER_W = 32;
    localparam int TOKEN_ID_W = 8;
    localparam int LANE_ID_W = 5;
    localparam int GATE_W = 9;
    localparam int WEIGHT_W = 8;
    localparam int ACC_W = 32;
    localparam int ROUTE_W = 2;
    localparam int FORMAT_W = 2;
    localparam int RES_TERM_IDX_W = 7;

    logic clk_core, rst_core;
    logic [2:0] tile_start_valid, tile_start_ready;
    logic [(ENGINES*TAG_W)-1:0] tile_start_tag;
    logic [(ENGINES*OUTPUT_TILE_W)-1:0] tile_start_output_tile;
    logic [(ENGINES*HEAD_COUNT_W)-1:0] tile_start_head_count;
    logic [2:0] head_start_valid, head_start_ready;
    logic [(ENGINES*TAG_W)-1:0] head_start_tag;
    logic [(ENGINES*TAG_W)-1:0] head_start_payload_tag;
    logic [(ENGINES*HEAD_COUNT_W)-1:0] head_start_index;
    logic [(ENGINES*ROUTE_W)-1:0] head_start_route_select;
    logic [(ENGINES*FORMAT_W)-1:0] head_start_csr_format;
    logic [(ENGINES*INPUT_CH_W)-1:0] head_start_input_channel_base;
    logic [2:0] head_start_last;
    logic [(ENGINES*8)-1:0] resident_term_count;
    logic [(ENGINES*EVENT_COUNT_W)-1:0] resident_event_count;
    logic [2:0] resident_descriptor_valid, resident_descriptor_ready;
    logic [(ENGINES*GATE_W)-1:0] resident_descriptor_gate_code;
    logic [(ENGINES*LANE_ID_W)-1:0] resident_descriptor_lane_id;
    logic [(ENGINES*8)-1:0] resident_descriptor_destination_count;
    logic [(ENGINES*RES_TERM_IDX_W)-1:0] resident_descriptor_term_index;
    logic [2:0] resident_descriptor_last;
    logic [2:0] resident_word_valid, resident_word_ready;
    logic [(ENGINES*64)-1:0] resident_word_data;
    logic [(ENGINES*WORD_INDEX_W)-1:0] resident_word_index;
    logic [2:0] resident_word_last;
    logic [2:0] ipd_word_valid, ipd_word_ready;
    logic [(ENGINES*64)-1:0] ipd_word_data;
    logic [(ENGINES*WORD_INDEX_W)-1:0] ipd_word_index;
    logic [2:0] ipd_word_last;
    logic [2:0] raw_word_valid, raw_word_ready;
    logic [(ENGINES*64)-1:0] raw_word_data;
    logic [(ENGINES*WORD_INDEX_W)-1:0] raw_word_index;
    logic [2:0] raw_word_last;
    logic [2:0] ipd_fill_begin_valid, ipd_fill_begin_ready;
    logic [(ENGINES*TAG_W)-1:0] ipd_fill_begin_tag;
    logic [(ENGINES*8)-1:0] ipd_fill_begin_term_count;
    logic [2:0] ipd_fill_entry_valid, ipd_fill_entry_ready;
    logic [(ENGINES*GATE_W)-1:0] ipd_fill_gate_code;
    logic [(ENGINES*LANE_ID_W)-1:0] ipd_fill_lane_id;
    logic [(ENGINES*8)-1:0] ipd_fill_destination_count;
    logic [2:0] ipd_fill_entry_last, ipd_fill_cache_allowed;
    logic [2:0] weight_req_valid, weight_req_ready;
    logic [(ENGINES*TAG_W)-1:0] weight_req_tag;
    logic [(ENGINES*INPUT_CH_W)-1:0] weight_req_input_channel;
    logic [(ENGINES*OUTPUT_TILE_W)-1:0] weight_req_output_tile;
    logic [2:0] weight_rsp_valid, weight_rsp_ready;
    logic [(ENGINES*TAG_W)-1:0] weight_rsp_tag;
    logic [(ENGINES*INPUT_CH_W)-1:0] weight_rsp_input_channel;
    logic [(ENGINES*OUTPUT_TILE_W)-1:0] weight_rsp_output_tile;
    logic [(ENGINES*OUT_TILE*WEIGHT_W)-1:0] weight_rsp_weights;
    logic [2:0] bias_req_valid, bias_req_ready;
    logic [(ENGINES*TAG_W)-1:0] bias_req_tag;
    logic [(ENGINES*OUTPUT_TILE_W)-1:0] bias_req_output_tile;
    logic [(ENGINES*TOKEN_ID_W)-1:0] bias_req_token_id;
    logic [2:0] bias_rsp_valid, bias_rsp_ready;
    logic [(ENGINES*TAG_W)-1:0] bias_rsp_tag;
    logic [(ENGINES*TOKEN_ID_W)-1:0] bias_rsp_token_id;
    logic [(ENGINES*OUT_TILE*ACC_W)-1:0] bias_rsp_values;
    logic [(ENGINES*BANKS)-1:0] final_valid, final_ready;
    logic [(ENGINES*BANKS*TOKEN_ID_W)-1:0] final_token_ids;
    logic [(ENGINES*TAG_W)-1:0] final_tag;
    logic [(ENGINES*BANKS*OUT_TILE*ACC_W)-1:0] final_values;
    logic [2:0] decoder_done_valid, decoder_done_ready;
    logic [(ENGINES*TAG_W)-1:0] decoder_done_payload_tag;
    logic [2:0] decoder_done_error;
    logic [2:0] head_done_valid, head_done_ready;
    logic [(ENGINES*TAG_W)-1:0] head_done_tag;
    logic [(ENGINES*HEAD_COUNT_W)-1:0] head_done_index;
    logic [2:0] head_done_last, head_done_error;
    logic [2:0] tile_done_valid, tile_done_ready;
    logic [(ENGINES*TAG_W)-1:0] tile_done_tag;
    logic [2:0] protocol_error, accumulator_overflow;
    logic [(ENGINES*COUNTER_W)-1:0] count_heads, count_terms;
    logic [(ENGINES*COUNTER_W)-1:0] count_completed_terms;
    logic [(ENGINES*COUNTER_W)-1:0] count_bias_commits;

    logic [2:0] weight_pending, bias_pending;
    integer weight_delay [0:ENGINES-1];
    integer bias_delay [0:ENGINES-1];
    integer weight_wait [0:ENGINES-1];
    integer final_hold [0:ENGINES-1];
    integer cycle_count;
    integer final_count [0:ENGINES-1];
    integer decoder_done_count [0:ENGINES-1];
    integer head_done_count [0:ENGINES-1];
    logic [TOKENS-1:0] seen_tokens [0:ENGINES-1];
    logic [2:0] weight_stall_seen, final_stall_seen;

    gatestack_three_independent32_projection_top #(
        .TOKENS(TOKENS), .LANES(LANES), .MAX_TERMS(128),
        .RESIDENT_TERMS(80), .EVENT_WAYS(4), .BANKS(BANKS),
        .SEGMENT_TOKENS(2), .TAG_W(TAG_W),
        .INPUT_CH_W(INPUT_CH_W), .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .HEAD_COUNT_W(HEAD_COUNT_W), .WORD_INDEX_W(WORD_INDEX_W),
        .EVENT_COUNT_W(EVENT_COUNT_W), .COUNTER_W(COUNTER_W),
        .TOKEN_ID_W(TOKEN_ID_W), .LANE_ID_W(LANE_ID_W),
        .RES_TERM_IDX_W(RES_TERM_IDX_W), .IPD_TERM_IDX_W(7)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    function automatic integer signed weight_value(
        input integer engine,
        input integer lane
    );
        if (engine == 0)
            weight_value = lane + 1;
        else if (engine == 1)
            weight_value = -(lane + 1);
        else if ((lane % 2) == 0)
            weight_value = 3;
        else
            weight_value = -2;
    endfunction

    function automatic integer signed bias_value(
        input integer engine,
        input integer token,
        input integer lane
    );
        bias_value = 1000 * engine + 100 * token + lane;
    endfunction

    always_comb begin
        for (int engine = 0; engine < ENGINES; engine = engine + 1) begin
            weight_req_ready[engine] = !weight_pending[engine] &&
                !weight_rsp_valid[engine] && weight_wait[engine] >= engine;
            bias_req_ready[engine] = !bias_pending[engine] &&
                !bias_rsp_valid[engine] &&
                ((cycle_count + engine) % (engine + 2) != 0);
            head_done_ready[engine] =
                ((cycle_count + engine) % (engine + 2)) != 0;
            for (int acc_bank = 0; acc_bank < BANKS;
                 acc_bank = acc_bank + 1) begin
                final_ready[(engine*BANKS) + acc_bank] =
                    final_hold[engine] == 0;
            end
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            weight_pending <= '0;
            bias_pending <= '0;
            weight_rsp_valid <= '0;
            bias_rsp_valid <= '0;
            weight_rsp_tag <= '0;
            weight_rsp_input_channel <= '0;
            weight_rsp_output_tile <= '0;
            weight_rsp_weights <= '0;
            bias_rsp_tag <= '0;
            bias_rsp_token_id <= '0;
            bias_rsp_values <= '0;
            weight_stall_seen <= '0;
            final_stall_seen <= '0;
            for (int engine = 0; engine < ENGINES; engine = engine + 1) begin
                weight_delay[engine] = 0;
                bias_delay[engine] = 0;
                weight_wait[engine] = 0;
                final_hold[engine] = engine;
                final_count[engine] = 0;
                decoder_done_count[engine] = 0;
                head_done_count[engine] = 0;
                seen_tokens[engine] = '0;
            end
        end else begin
            cycle_count <= cycle_count + 1;
            for (int engine = 0; engine < ENGINES; engine = engine + 1) begin
                if (weight_req_valid[engine] && !weight_req_ready[engine]) begin
                    weight_wait[engine] = weight_wait[engine] + 1;
                    weight_stall_seen[engine] <= 1'b1;
                end
                if (weight_req_valid[engine] && weight_req_ready[engine]) begin
                    if (weight_req_tag[(engine*TAG_W) +: TAG_W] !=
                            TAG_W'(16'h3100 + engine) ||
                        weight_req_input_channel[(engine*INPUT_CH_W) +: INPUT_CH_W] !=
                            INPUT_CH_W'(engine + engine) ||
                        weight_req_output_tile[(engine*OUTPUT_TILE_W) +: OUTPUT_TILE_W] !=
                            OUTPUT_TILE_W'(engine))
                        $fatal(1, "engine %0d weight request crosstalk", engine);
                    weight_rsp_tag[(engine*TAG_W) +: TAG_W] <=
                        weight_req_tag[(engine*TAG_W) +: TAG_W];
                    weight_rsp_input_channel[(engine*INPUT_CH_W) +: INPUT_CH_W] <=
                        weight_req_input_channel[(engine*INPUT_CH_W) +: INPUT_CH_W];
                    weight_rsp_output_tile[(engine*OUTPUT_TILE_W) +: OUTPUT_TILE_W] <=
                        weight_req_output_tile[(engine*OUTPUT_TILE_W) +: OUTPUT_TILE_W];
                    for (int lane = 0; lane < OUT_TILE; lane = lane + 1)
                        weight_rsp_weights[
                            (engine*OUT_TILE*WEIGHT_W) + (lane*WEIGHT_W) +:
                            WEIGHT_W] <= WEIGHT_W'(weight_value(engine, lane));
                    weight_pending[engine] <= 1'b1;
                    weight_delay[engine] = engine + 1;
                end
                if (weight_pending[engine]) begin
                    if (weight_delay[engine] == 0) begin
                        weight_rsp_valid[engine] <= 1'b1;
                        weight_pending[engine] <= 1'b0;
                    end else begin
                        weight_delay[engine] = weight_delay[engine] - 1;
                    end
                end
                if (weight_rsp_valid[engine] && weight_rsp_ready[engine])
                    weight_rsp_valid[engine] <= 1'b0;

                if (bias_req_valid[engine] && bias_req_ready[engine]) begin
                    if (bias_req_tag[(engine*TAG_W) +: TAG_W] !=
                            TAG_W'(16'h3100 + engine) ||
                        bias_req_output_tile[(engine*OUTPUT_TILE_W) +: OUTPUT_TILE_W] !=
                            OUTPUT_TILE_W'(engine))
                        $fatal(1, "engine %0d bias request crosstalk", engine);
                    bias_rsp_tag[(engine*TAG_W) +: TAG_W] <=
                        bias_req_tag[(engine*TAG_W) +: TAG_W];
                    bias_rsp_token_id[(engine*TOKEN_ID_W) +: TOKEN_ID_W] <=
                        bias_req_token_id[(engine*TOKEN_ID_W) +: TOKEN_ID_W];
                    for (int lane = 0; lane < OUT_TILE; lane = lane + 1)
                        bias_rsp_values[
                            (engine*OUT_TILE*ACC_W) + (lane*ACC_W) +:
                            ACC_W] <= ACC_W'(bias_value(
                                engine,
                                bias_req_token_id[
                                    (engine*TOKEN_ID_W) +: TOKEN_ID_W],
                                lane));
                    bias_pending[engine] <= 1'b1;
                    bias_delay[engine] = 1 + (2-engine);
                end
                if (bias_pending[engine]) begin
                    if (bias_delay[engine] == 0) begin
                        bias_rsp_valid[engine] <= 1'b1;
                        bias_pending[engine] <= 1'b0;
                    end else begin
                        bias_delay[engine] = bias_delay[engine] - 1;
                    end
                end
                if (bias_rsp_valid[engine] && bias_rsp_ready[engine])
                    bias_rsp_valid[engine] <= 1'b0;

                if (|(final_valid[(engine*BANKS) +: BANKS])) begin
                    if (final_hold[engine] != 0) begin
                        final_hold[engine] = final_hold[engine] - 1;
                        final_stall_seen[engine] <= 1'b1;
                    end
                end
                if (decoder_done_valid[engine] && decoder_done_ready[engine]) begin
                    if (decoder_done_payload_tag[(engine*TAG_W) +: TAG_W] !=
                            TAG_W'(16'h4100 + engine) ||
                        decoder_done_error[engine])
                        $fatal(1, "engine %0d decoder completion crosstalk", engine);
                    decoder_done_count[engine] = decoder_done_count[engine] + 1;
                end
                if (head_done_valid[engine] && head_done_ready[engine]) begin
                    if (head_done_tag[(engine*TAG_W) +: TAG_W] !=
                            TAG_W'(16'h3100 + engine) ||
                        head_done_index[(engine*HEAD_COUNT_W) +: HEAD_COUNT_W] != 0 ||
                        !head_done_last[engine] || head_done_error[engine])
                        $fatal(1, "engine %0d head completion crosstalk", engine);
                    head_done_count[engine] = head_done_count[engine] + 1;
                end
                for (int acc_bank = 0; acc_bank < BANKS;
                     acc_bank = acc_bank + 1) begin
                    if (final_valid[(engine*BANKS) + acc_bank] &&
                        final_ready[(engine*BANKS) + acc_bank]) begin
                        integer token;
                        token = final_token_ids[
                            (engine*BANKS*TOKEN_ID_W) +
                            (acc_bank*TOKEN_ID_W) +: TOKEN_ID_W];
                        if (token >= TOKENS || (token % BANKS) != acc_bank ||
                            seen_tokens[engine][token])
                            $fatal(1, "engine %0d invalid/duplicate token %0d",
                                   engine, token);
                        if (final_tag[(engine*TAG_W) +: TAG_W] !=
                            TAG_W'(16'h3100 + engine))
                            $fatal(1, "engine %0d final tag crosstalk", engine);
                        for (int lane = 0; lane < OUT_TILE; lane = lane + 1) begin
                            integer signed actual_value;
                            integer signed expected_value;
                            actual_value = $signed(final_values[
                                (engine*BANKS*OUT_TILE*ACC_W) +
                                (acc_bank*OUT_TILE*ACC_W) + (lane*ACC_W) +:
                                ACC_W]);
                            expected_value = bias_value(engine, token, lane);
                            if (token == engine)
                                expected_value = expected_value +
                                    ((engine + 1) * weight_value(engine, lane));
                            if (actual_value != expected_value)
                                $fatal(1, "engine=%0d token=%0d lane=%0d got=%0d expected=%0d",
                                       engine, token, lane, actual_value,
                                       expected_value);
                        end
                        seen_tokens[engine][token] = 1'b1;
                        final_count[engine] = final_count[engine] + 1;
                    end
                end
            end
        end
    end

    task automatic run_engine(input integer engine);
        begin
            repeat (engine + 1) @(posedge clk_core);
            @(negedge clk_core);
            tile_start_valid[engine] = 1'b1;
            do @(posedge clk_core); while (!tile_start_ready[engine]);
            @(negedge clk_core);
            tile_start_valid[engine] = 1'b0;

            repeat (2-engine) @(posedge clk_core);
            @(negedge clk_core);
            head_start_valid[engine] = 1'b1;
            do @(posedge clk_core); while (!head_start_ready[engine]);
            @(negedge clk_core);
            head_start_valid[engine] = 1'b0;

            resident_descriptor_valid[engine] = 1'b1;
            do @(posedge clk_core);
            while (!resident_descriptor_ready[engine]);
            @(negedge clk_core);
            resident_descriptor_valid[engine] = 1'b0;

            resident_word_valid[engine] = 1'b1;
            do @(posedge clk_core); while (!resident_word_ready[engine]);
            @(negedge clk_core);
            resident_word_valid[engine] = 1'b0;
            resident_word_last[engine] = 1'b0;

            wait (tile_done_valid[engine]);
            if (tile_done_tag[(engine*TAG_W) +: TAG_W] !=
                    TAG_W'(16'h3100 + engine))
                $fatal(1, "engine %0d tile completion crosstalk", engine);
            @(negedge clk_core);
            tile_done_ready[engine] = 1'b1;
            do @(posedge clk_core); while (!tile_done_valid[engine]);
            @(negedge clk_core);
            tile_done_ready[engine] = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        tile_start_valid = '0;
        head_start_valid = '0;
        resident_descriptor_valid = '0;
        resident_word_valid = '0;
        ipd_word_valid = '0;
        raw_word_valid = '0;
        tile_done_ready = '0;
        decoder_done_ready = '1;
        ipd_fill_begin_ready = '1;
        ipd_fill_entry_ready = '1;
        tile_start_tag = '0;
        tile_start_output_tile = '0;
        tile_start_head_count = '0;
        head_start_tag = '0;
        head_start_payload_tag = '0;
        head_start_index = '0;
        head_start_route_select = '0;
        head_start_csr_format = '0;
        head_start_input_channel_base = '0;
        head_start_last = '0;
        resident_term_count = '0;
        resident_event_count = '0;
        resident_descriptor_gate_code = '0;
        resident_descriptor_lane_id = '0;
        resident_descriptor_destination_count = '0;
        resident_descriptor_term_index = '0;
        resident_descriptor_last = '0;
        resident_word_data = '0;
        resident_word_index = '0;
        resident_word_last = '0;
        ipd_word_data = '0;
        ipd_word_index = '0;
        ipd_word_last = '0;
        raw_word_data = '0;
        raw_word_index = '0;
        raw_word_last = '0;
        for (int engine = 0; engine < ENGINES; engine = engine + 1) begin
            tile_start_tag[(engine*TAG_W) +: TAG_W] =
                TAG_W'(16'h3100 + engine);
            tile_start_output_tile[(engine*OUTPUT_TILE_W) +: OUTPUT_TILE_W] =
                OUTPUT_TILE_W'(engine);
            tile_start_head_count[(engine*HEAD_COUNT_W) +: HEAD_COUNT_W] = 1;
            head_start_tag[(engine*TAG_W) +: TAG_W] =
                TAG_W'(16'h3100 + engine);
            head_start_payload_tag[(engine*TAG_W) +: TAG_W] =
                TAG_W'(16'h4100 + engine);
            head_start_index[(engine*HEAD_COUNT_W) +: HEAD_COUNT_W] = 0;
            head_start_route_select[(engine*ROUTE_W) +: ROUTE_W] = 0;
            head_start_csr_format[(engine*FORMAT_W) +: FORMAT_W] = 1;
            head_start_input_channel_base[
                (engine*INPUT_CH_W) +: INPUT_CH_W] = INPUT_CH_W'(engine);
            head_start_last[engine] = 1'b1;
            resident_term_count[(engine*8) +: 8] = 1;
            resident_event_count[(engine*EVENT_COUNT_W) +: EVENT_COUNT_W] = 1;
            resident_descriptor_gate_code[(engine*GATE_W) +: GATE_W] =
                GATE_W'(engine + 1);
            resident_descriptor_lane_id[(engine*LANE_ID_W) +: LANE_ID_W] =
                LANE_ID_W'(engine);
            resident_descriptor_destination_count[(engine*8) +: 8] = 1;
            resident_descriptor_term_index[
                (engine*RES_TERM_IDX_W) +: RES_TERM_IDX_W] = 0;
            resident_descriptor_last[engine] = 1'b1;
            resident_word_data[(engine*64) +: 64] = 64'(engine);
            resident_word_index[(engine*WORD_INDEX_W) +: WORD_INDEX_W] = 0;
            resident_word_last[engine] = 1'b1;
        end
        repeat (6) @(posedge clk_core);
        rst_core = 1'b0;

        fork
            run_engine(0);
            run_engine(1);
            run_engine(2);
        join

        repeat (3) @(posedge clk_core);
        for (int engine = 0; engine < ENGINES; engine = engine + 1) begin
            if (final_count[engine] != TOKENS ||
                seen_tokens[engine] != {TOKENS{1'b1}} ||
                decoder_done_count[engine] != 1 ||
                head_done_count[engine] != 1 ||
                count_heads[(engine*COUNTER_W) +: COUNTER_W] != 1 ||
                count_terms[(engine*COUNTER_W) +: COUNTER_W] != 1 ||
                count_completed_terms[(engine*COUNTER_W) +: COUNTER_W] != 1 ||
                count_bias_commits[(engine*COUNTER_W) +: COUNTER_W] != TOKENS ||
                protocol_error[engine] || accumulator_overflow[engine])
                $fatal(1, "engine %0d status mismatch finals=%0d seen=%b heads=%0d terms=%0d completed=%0d bias=%0d",
                       engine, final_count[engine], seen_tokens[engine],
                       count_heads[(engine*COUNTER_W) +: COUNTER_W],
                       count_terms[(engine*COUNTER_W) +: COUNTER_W],
                       count_completed_terms[(engine*COUNTER_W) +: COUNTER_W],
                       count_bias_commits[(engine*COUNTER_W) +: COUNTER_W]);
        end
        if (!weight_stall_seen[1] || !weight_stall_seen[2] ||
            !final_stall_seen[1] || !final_stall_seen[2])
            $fatal(1, "independent backpressure was not exercised weight=%b final=%b",
                   weight_stall_seen, final_stall_seen);
        $display("RESULT suite=three_independent32 status=PASS transactions=3 product_lanes=96 exact_elements=%0d crosstalk=0 weight_stalls=%b final_stalls=%b cycles=%0d",
                 ENGINES*TOKENS*OUT_TILE, weight_stall_seen,
                 final_stall_seen, cycle_count);
        $finish;
    end

    initial begin
        repeat (20000) @(posedge clk_core);
        $fatal(1, "three-independent32 projection TB timeout");
    end
endmodule

`default_nettype wire
