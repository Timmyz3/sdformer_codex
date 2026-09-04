`timescale 1ns/1ps
`default_nettype none

module tb_qfit_dual_line_descriptor_stateful_real #(
    parameter int STATE_QUEUE_DEPTH = 4,
    parameter bit USE_SHARED_WIDE_METADATA = 1'b1
);
    localparam int TILE_BITS = 256;
    localparam int MAX_CHUNKS = 12;
    localparam int MAX_LANE_TILES = 32;
    localparam int ISSUE_WIDTH = 16;
    localparam int CONTEXTS = 4;
    localparam int REDUCE_SLOTS = 4;
    localparam int OUT_LANES = 96;
    localparam int TAG_W = 32;
    localparam int OBJECT_W = 64;
    localparam int W_W = 8;
    localparam int ACC_W = 32;
    localparam int STATE_CONTEXTS = 4;
    localparam int STATE_BASE_TILES = 32;
    localparam int STATE_BANKS = 6;
    localparam int STATE_LANES_PER_BANK = 16;
    localparam int EPOCH_W = 16;
    localparam int DOMAIN_W = 32;
    localparam int STEP_W = 4;
    localparam int LEN_W = 4;
    localparam int BANK_ADDR_W = $clog2(TILE_BITS)-$clog2(ISSUE_WIDTH);
    localparam int CTX_W = $clog2(CONTEXTS);
    localparam int CTX_COUNT_W = $clog2(CONTEXTS+1);
    localparam int CHUNK_W = $clog2(MAX_CHUNKS);
    localparam int CHUNK_COUNT_W = $clog2(MAX_CHUNKS+1);
    localparam int LANE_TILE_W = $clog2(MAX_LANE_TILES);
    localparam int LANE_COUNT_W = $clog2(MAX_LANE_TILES+1);
    localparam int SLOT_W = $clog2(REDUCE_SLOTS);
    localparam int STATE_CTX_W = $clog2(STATE_CONTEXTS);
    localparam int STATE_BASE_TILE_W = $clog2(STATE_BASE_TILES);
    localparam int MAX_EXPECTED_OUTPUTS = 1024;

    logic clk_core = 1'b0;
    logic por_core;
    logic rst_core;
    logic [DOMAIN_W-1:0] active_domain;
    always #1.5 clk_core = ~clk_core;

    logic descriptor_valid, descriptor_ready;
    logic descriptor_row_first, descriptor_row_last, descriptor_batch_last;
    logic [TAG_W-1:0] descriptor_tag;
    logic [OBJECT_W-1:0] descriptor_object_tag;
    logic [CHUNK_W-1:0] descriptor_chunk_index;
    logic [CHUNK_COUNT_W-1:0] descriptor_chunk_count;
    logic [LANE_COUNT_W-1:0] descriptor_lane_tile_count;
    logic descriptor_use_motion;
    logic [TILE_BITS-1:0] descriptor_source_bits;
    logic [TILE_BITS-1:0] descriptor_negative_bits;
    logic [STATE_CTX_W-1:0] descriptor_state_context;
    logic [STATE_BASE_TILE_W-1:0] descriptor_state_base_tile;
    logic [EPOCH_W-1:0] descriptor_epoch;
    logic [DOMAIN_W-1:0] descriptor_domain;
    logic [STEP_W-1:0] descriptor_temporal_step;
    logic [LEN_W-1:0] descriptor_temporal_length;
    logic descriptor_temporal_first, descriptor_temporal_last;
    logic weight_request_valid, weight_request_ready;
    logic [OBJECT_W-1:0] weight_request_object_tag;
    logic [CHUNK_W-1:0] weight_request_chunk_index;
    logic [LANE_TILE_W-1:0] weight_request_lane_tile;
    logic [ISSUE_WIDTH-1:0] weight_request_bank_valid;
    logic [ISSUE_WIDTH*BANK_ADDR_W-1:0] weight_request_bank_addr;
    logic [ISSUE_WIDTH*CTX_W-1:0] weight_request_bank_context;
    logic [ISSUE_WIDTH*SLOT_W-1:0] weight_request_bank_slot;
    logic [ISSUE_WIDTH-1:0] weight_request_bank_negative;
    logic weight_response_valid, weight_response_ready;
    logic [ISSUE_WIDTH-1:0] weight_response_bank_valid;
    logic [ISSUE_WIDTH*OUT_LANES*W_W-1:0] weight_response_data;
    logic output_valid, output_ready;
    logic [STATE_CTX_W-1:0] output_state_context;
    logic [STATE_BASE_TILE_W-1:0] output_state_base_tile;
    logic [EPOCH_W-1:0] output_epoch;
    logic [DOMAIN_W-1:0] output_domain;
    logic [STEP_W-1:0] output_temporal_step;
    logic [LEN_W-1:0] output_temporal_length;
    logic output_temporal_first, output_temporal_last, output_used_motion;
    logic [TAG_W-1:0] output_tag;
    logic [OUT_LANES*ACC_W-1:0] output_current_acc;
    logic [2:0] controller_state;
    logic [CTX_COUNT_W-1:0] resident_contexts;
    logic state_rmw_busy, domain_fence_ready, domain_fence_error;
    logic protocol_error;

    logic [TILE_BITS-1:0] golden_current [0:CONTEXTS-1][0:MAX_CHUNKS-1];
    integer golden_residue_count
        [0:CONTEXTS-1][0:MAX_CHUNKS-1][0:16];
    logic [TAG_W-1:0] expected_tag [0:CONTEXTS-1];
    logic expected_motion [0:CONTEXTS-1];
    integer expected_epoch [0:CONTEXTS-1];
    integer expected_step [0:CONTEXTS-1];
    integer expected_length [0:CONTEXTS-1];
    integer expected_base [0:CONTEXTS-1];
    integer expected_chunks, expected_lane_tiles, expected_contexts;
    logic [OBJECT_W-1:0] expected_object;
    logic [STATE_CTX_W-1:0] scoreboard_context
        [0:MAX_EXPECTED_OUTPUTS-1];
    logic [STATE_BASE_TILE_W-1:0] scoreboard_base
        [0:MAX_EXPECTED_OUTPUTS-1];
    logic [EPOCH_W-1:0] scoreboard_epoch [0:MAX_EXPECTED_OUTPUTS-1];
    logic [DOMAIN_W-1:0] scoreboard_domain [0:MAX_EXPECTED_OUTPUTS-1];
    logic [STEP_W-1:0] scoreboard_step [0:MAX_EXPECTED_OUTPUTS-1];
    logic [LEN_W-1:0] scoreboard_length [0:MAX_EXPECTED_OUTPUTS-1];
    logic scoreboard_first [0:MAX_EXPECTED_OUTPUTS-1];
    logic scoreboard_last [0:MAX_EXPECTED_OUTPUTS-1];
    logic scoreboard_motion [0:MAX_EXPECTED_OUTPUTS-1];
    logic [TAG_W-1:0] scoreboard_tag [0:MAX_EXPECTED_OUTPUTS-1];
    logic signed [ACC_W-1:0] scoreboard_acc
        [0:MAX_EXPECTED_OUTPUTS-1][0:OUT_LANES-1];
    integer expected_write_count, expected_read_count;
    integer descriptors, batches, outputs_seen, local_outputs, motion_outputs;
    integer request_beats, bank_reads, output_stalls, request_stalls;
    integer rmw_backpressure_cycles, cycle_count;
    integer current_sequence, sequence_start_cycle;
    integer sequence_start_outputs, sequence_start_motion_outputs;
    integer sequence_start_descriptors;
    integer perf_local_cycles, perf_hybrid_cycles;
    integer perf_local_sequences, perf_hybrid_sequences;
    bit perf_mode, stream_mode;

    qfit_dual_line_descriptor_stateful_engine #(
        .TILE_BITS(TILE_BITS), .MAX_CHUNKS(MAX_CHUNKS),
        .MAX_LANE_TILES(MAX_LANE_TILES), .ISSUE_WIDTH(ISSUE_WIDTH),
        .CONTEXTS(CONTEXTS), .REDUCE_SLOTS(REDUCE_SLOTS),
        .OUT_LANES(OUT_LANES), .TAG_W(TAG_W), .OBJECT_W(OBJECT_W),
        .W_W(W_W), .ACC_W(ACC_W), .STATE_CONTEXTS(STATE_CONTEXTS),
        .STATE_BASE_TILES(STATE_BASE_TILES), .STATE_BANKS(STATE_BANKS),
        .STATE_LANES_PER_BANK(STATE_LANES_PER_BANK), .EPOCH_W(EPOCH_W),
        .DOMAIN_W(DOMAIN_W), .STEP_W(STEP_W), .LEN_W(LEN_W),
        .STATE_QUEUE_DEPTH(STATE_QUEUE_DEPTH),
        .USE_SHARED_WIDE_METADATA(USE_SHARED_WIDE_METADATA)
    ) dut (.*);

    function automatic integer signed model_weight(
        input logic [OBJECT_W-1:0] object_tag,
        input integer lane_tile, input integer chunk,
        input integer source, input integer lane
    );
        longint unsigned mixed;
        begin
            mixed = $unsigned(object_tag[31:0]) + lane_tile*13 +
                    chunk*7 + source*5 + lane*3;
            model_weight = (mixed % 17) - 8;
        end
    endfunction

    task automatic clear_batch_expected;
        begin
            expected_contexts = 0;
            expected_chunks = 0;
            expected_lane_tiles = 0;
            expected_object = '0;
            for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
                expected_tag[ctx] = '0;
                expected_motion[ctx] = 1'b0;
                expected_epoch[ctx] = 0;
                expected_step[ctx] = 0;
                expected_length[ctx] = 0;
                expected_base[ctx] = 0;
                for (int chunk = 0; chunk < MAX_CHUNKS; chunk = chunk + 1)
                    golden_current[ctx][chunk] = '0;
                for (int chunk = 0; chunk < MAX_CHUNKS; chunk = chunk + 1)
                    for (int residue = 0; residue < 17; residue = residue + 1)
                        golden_residue_count[ctx][chunk][residue] = 0;
            end
        end
    endtask

    task automatic enqueue_batch_expected;
        integer batch_outputs, slot;
        integer signed expected;
        begin
            batch_outputs = expected_contexts*expected_lane_tiles;
            if (expected_contexts <= 0 || expected_chunks <= 0 ||
                    expected_lane_tiles <= 0)
                $fatal(1, "cannot enqueue empty real-state batch");
            if ((expected_write_count - expected_read_count) +
                    batch_outputs > MAX_EXPECTED_OUTPUTS)
                $fatal(1,
                    "real-state expected-output FIFO overflow pending=%0d add=%0d",
                    expected_write_count - expected_read_count, batch_outputs);
            // The resident kernel emits every context for a lane tile before
            // advancing to the next lane tile.  The independent scoreboard
            // deliberately encodes that contract instead of indexing golden
            // data with the DUT-provided address.
            for (int tile = 0; tile < expected_lane_tiles;
                    tile = tile + 1) begin
                for (int ctx = 0; ctx < expected_contexts;
                        ctx = ctx + 1) begin
                    slot = expected_write_count % MAX_EXPECTED_OUTPUTS;
                    scoreboard_context[slot] = STATE_CTX_W'(ctx);
                    scoreboard_base[slot] =
                        STATE_BASE_TILE_W'(expected_base[ctx] + tile);
                    scoreboard_epoch[slot] = EPOCH_W'(expected_epoch[ctx]);
                    scoreboard_domain[slot] = active_domain;
                    scoreboard_step[slot] = STEP_W'(expected_step[ctx]);
                    scoreboard_length[slot] = LEN_W'(expected_length[ctx]);
                    scoreboard_first[slot] = expected_step[ctx] == 0;
                    scoreboard_last[slot] =
                        expected_step[ctx] == expected_length[ctx]-1;
                    scoreboard_motion[slot] = expected_motion[ctx];
                    scoreboard_tag[slot] = expected_tag[ctx];
                    for (int lane = 0; lane < OUT_LANES;
                            lane = lane + 1) begin
                        expected = 0;
                        for (int chunk = 0; chunk < expected_chunks;
                                chunk = chunk + 1)
                            for (int residue = 0; residue < 17;
                                    residue = residue + 1)
                                expected += golden_residue_count[
                                    ctx][chunk][residue] *
                                    model_weight(expected_object, tile, chunk,
                                        residue, lane);
                        scoreboard_acc[slot][lane] = ACC_W'(expected);
                    end
                    expected_write_count += 1;
                end
            end
        end
    endtask

    always_ff @(posedge clk_core) begin
        if (por_core || rst_core) begin
            weight_request_ready <= 1'b1;
            request_stalls <= 0;
        end else begin
            if (perf_mode)
                weight_request_ready <= 1'b1;
            else
                weight_request_ready <= ($urandom % 7) != 0;
            if (weight_request_valid && !weight_request_ready)
                request_stalls <= request_stalls + 1;
        end
    end

    always_ff @(posedge clk_core) begin
        if (por_core || rst_core) begin
            weight_response_valid <= 1'b0;
            weight_response_bank_valid <= '0;
            weight_response_data <= '0;
            request_beats <= 0;
            bank_reads <= 0;
        end else begin
            weight_response_valid <= weight_request_valid && weight_request_ready;
            weight_response_bank_valid <= weight_request_bank_valid;
            weight_response_data <= '0;
            if (weight_request_valid && weight_request_ready) begin
                integer beat_reads;
                beat_reads = 0;
                request_beats <= request_beats + 1;
                for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin
                    if (weight_request_bank_valid[bank]) begin
                        integer address, source;
                        integer signed value;
                        address = weight_request_bank_addr[
                            bank*BANK_ADDR_W +: BANK_ADDR_W];
                        source = address*ISSUE_WIDTH + bank;
                        beat_reads += 1;
                        for (int lane = 0; lane < OUT_LANES; lane = lane + 1) begin
                            value = model_weight(weight_request_object_tag,
                                weight_request_lane_tile,
                                weight_request_chunk_index, source, lane);
                            weight_response_data[
                                (bank*OUT_LANES+lane)*W_W +: W_W] <= W_W'(value);
                        end
                    end
                end
                bank_reads <= bank_reads + beat_reads;
            end
        end
    end

    always_ff @(posedge clk_core) begin
        if (por_core || rst_core) begin
            output_ready <= 1'b0;
            outputs_seen <= 0;
            local_outputs <= 0;
            motion_outputs <= 0;
            output_stalls <= 0;
            rmw_backpressure_cycles <= 0;
            cycle_count <= 0;
            expected_read_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (perf_mode)
                output_ready <= 1'b1;
            else
                output_ready <= (cycle_count % 11) >= 3;
            if (output_valid && !output_ready)
                output_stalls <= output_stalls + 1;
            if (state_rmw_busy && output_valid && !output_ready)
                rmw_backpressure_cycles <= rmw_backpressure_cycles + 1;
            if (output_valid && output_ready) begin
                integer slot;
                if (expected_read_count >= expected_write_count)
                    $fatal(1, "real-state expected-output FIFO underflow");
                slot = expected_read_count % MAX_EXPECTED_OUTPUTS;
                if (output_state_context != scoreboard_context[slot] ||
                        output_state_base_tile != scoreboard_base[slot])
                    $fatal(1,
                        "real state output address/order mismatch got_ctx=%0d got_base=%0d exp_ctx=%0d exp_base=%0d",
                        output_state_context, output_state_base_tile,
                        scoreboard_context[slot], scoreboard_base[slot]);
                if (output_tag != scoreboard_tag[slot] ||
                        output_epoch != scoreboard_epoch[slot] ||
                        output_domain != scoreboard_domain[slot] ||
                        output_temporal_step != scoreboard_step[slot] ||
                        output_temporal_length != scoreboard_length[slot] ||
                        output_temporal_first != scoreboard_first[slot] ||
                        output_temporal_last != scoreboard_last[slot] ||
                        output_used_motion != scoreboard_motion[slot])
                    $fatal(1,
                        "real state output identity/order mismatch index=%0d",
                        expected_read_count);
                for (int lane = 0; lane < OUT_LANES; lane = lane + 1) begin
                    integer signed expected, actual;
                    expected = scoreboard_acc[slot][lane];
                    actual = $signed(output_current_acc[lane*ACC_W +: ACC_W]);
                    if (actual != expected)
                        $fatal(1,
                            "real state Acc/order mismatch index=%0d ctx=%0d base=%0d lane=%0d got=%0d exp=%0d",
                            expected_read_count, output_state_context,
                            output_state_base_tile, lane, actual, expected);
                end
                expected_read_count <= expected_read_count + 1;
                outputs_seen <= outputs_seen + 1;
                if (output_used_motion)
                    motion_outputs <= motion_outputs + 1;
                else
                    local_outputs <= local_outputs + 1;
            end
        end
    end

    task automatic drive_descriptor;
        begin
            descriptor_valid = 1'b1;
            do @(posedge clk_core); while (!descriptor_ready);
            @(negedge clk_core);
            descriptor_valid = 1'b0;
        end
    endtask

    task automatic finish_sequence(input integer sequence_id);
        integer sequence_cycles, sequence_outputs;
        integer sequence_motion_outputs, sequence_descriptors;
        string mode;
        begin
            sequence_cycles = cycle_count - sequence_start_cycle;
            sequence_outputs = outputs_seen - sequence_start_outputs;
            sequence_motion_outputs = motion_outputs -
                                      sequence_start_motion_outputs;
            sequence_descriptors = descriptors - sequence_start_descriptors;
            if ((sequence_id % 2) == 1) begin
                mode = "local_only";
                if (sequence_motion_outputs != 0)
                    $fatal(1, "local-only sequence emitted Motion state output");
                perf_local_cycles += sequence_cycles;
                perf_local_sequences += 1;
            end else begin
                mode = "hybrid_local_motion";
                if (sequence_motion_outputs == 0)
                    $fatal(1, "hybrid sequence lacks Motion state output");
                perf_hybrid_cycles += sequence_cycles;
                perf_hybrid_sequences += 1;
            end
            if (perf_mode)
                $display("M4_STATE_SEQ id=%0d mode=%s cycles=%0d descriptors=%0d outputs=%0d motion_outputs=%0d",
                    sequence_id, mode, sequence_cycles, sequence_descriptors,
                    sequence_outputs, sequence_motion_outputs);
        end
    endtask

    task automatic wait_outputs(input integer target);
        integer timeout;
        begin
            timeout = 0;
            while (outputs_seen < target && timeout < 20_000_000) begin
                @(posedge clk_core);
                timeout += 1;
            end
            if (outputs_seen != target)
                $fatal(1, "real state output timeout got=%0d target=%0d state=%0d",
                    outputs_seen, target, controller_state);
        end
    endtask

    initial begin : stimulus
        string trace_path;
        integer fd, scan, trace_context, target_outputs;
        integer trace_chunk, trace_chunks, trace_lane_tiles, trace_motion;
        integer trace_first, trace_last, trace_batch_last;
        integer trace_state_context, trace_state_base, trace_epoch;
        integer trace_step, trace_length, trace_tfirst, trace_tlast;
        integer trace_sequence;
        logic [OBJECT_W-1:0] trace_object;
        logic [TAG_W-1:0] trace_tag;
        logic [TILE_BITS-1:0] trace_selected, trace_negative, trace_golden;

`ifdef VCS
        $display("SIMULATOR=Synopsys VCS");
        $display("ASSERTIONS=enabled");
`else
        $fatal(1, "real M4-state miter requires Synopsys VCS");
`endif
        por_core = 1'b1;
        perf_mode = $test$plusargs("PERF_MODE");
        stream_mode = $test$plusargs("STREAM_MODE");
        rst_core = 1'b0;
        active_domain = 32'h4d345232;
        descriptor_valid = 1'b0;
        descriptor_row_first = 1'b0;
        descriptor_row_last = 1'b0;
        descriptor_batch_last = 1'b0;
        descriptor_tag = '0;
        descriptor_object_tag = '0;
        descriptor_chunk_index = '0;
        descriptor_chunk_count = '0;
        descriptor_lane_tile_count = '0;
        descriptor_use_motion = 1'b0;
        descriptor_source_bits = '0;
        descriptor_negative_bits = '0;
        descriptor_state_context = '0;
        descriptor_state_base_tile = '0;
        descriptor_epoch = '0;
        descriptor_domain = '0;
        descriptor_temporal_step = '0;
        descriptor_temporal_length = '0;
        descriptor_temporal_first = 1'b0;
        descriptor_temporal_last = 1'b0;
        descriptors = 0;
        batches = 0;
        current_sequence = 0;
        sequence_start_cycle = 0;
        sequence_start_outputs = 0;
        sequence_start_motion_outputs = 0;
        sequence_start_descriptors = 0;
        perf_local_cycles = 0;
        perf_hybrid_cycles = 0;
        perf_local_sequences = 0;
        perf_hybrid_sequences = 0;
        expected_write_count = 0;
        clear_batch_expected();
        repeat (6) @(posedge clk_core);
        @(negedge clk_core);
        por_core = 1'b0;
        wait (domain_fence_ready);
        if (!$value$plusargs("REAL_STATE_TRACE=%s", trace_path))
            $fatal(1, "REAL_STATE_TRACE is required");
        fd = $fopen(trace_path, "r");
        if (fd == 0)
            $fatal(1, "cannot open real state trace");
        trace_context = -1;
        target_outputs = 0;
        while (!$feof(fd)) begin
            scan = $fscanf(fd,
                "%h %h %d %d %d %d %h %h %h %d %d %d %d %d %d %d %d %d %d %d\n",
                trace_object, trace_tag, trace_chunk, trace_chunks,
                trace_lane_tiles, trace_motion, trace_selected, trace_negative,
                trace_golden, trace_first, trace_last, trace_batch_last,
                trace_state_context, trace_state_base, trace_epoch, trace_step,
                trace_length, trace_tfirst, trace_tlast, trace_sequence);
            if (scan == 20) begin
                if (trace_sequence != current_sequence) begin
                    if (current_sequence != 0 && !stream_mode)
                        finish_sequence(current_sequence);
                    if (trace_sequence != current_sequence + 1)
                        $fatal(1, "real state sequence IDs are not contiguous");
                    current_sequence = trace_sequence;
                    sequence_start_cycle = cycle_count;
                    sequence_start_outputs = outputs_seen;
                    sequence_start_motion_outputs = motion_outputs;
                    sequence_start_descriptors = descriptors;
                end
                if (trace_first != 0) begin
                    trace_context += 1;
                    if (trace_context >= CONTEXTS ||
                            trace_state_context != trace_context)
                        $fatal(1, "real state context order invalid");
                    expected_tag[trace_context] = trace_tag;
                    expected_motion[trace_context] = trace_motion != 0;
                    expected_epoch[trace_context] = trace_epoch;
                    expected_step[trace_context] = trace_step;
                    expected_length[trace_context] = trace_length;
                    expected_base[trace_context] = trace_state_base;
                    if (trace_context == 0) begin
                        expected_object = trace_object;
                        expected_chunks = trace_chunks;
                        expected_lane_tiles = trace_lane_tiles;
                    end else if (trace_object != expected_object ||
                            trace_chunks != expected_chunks ||
                            trace_lane_tiles != expected_lane_tiles)
                        $fatal(1, "real state batch geometry changed");
                end
                golden_current[trace_context][trace_chunk] = trace_golden;
                for (int residue = 0; residue < 17; residue = residue + 1) begin
                    golden_residue_count[trace_context][trace_chunk][residue] = 0;
                    for (int source = residue; source < TILE_BITS;
                            source = source + 17)
                        if (trace_golden[source])
                            golden_residue_count[
                                trace_context][trace_chunk][residue] += 1;
                end
                @(negedge clk_core);
                descriptor_object_tag = trace_object;
                descriptor_tag = trace_tag;
                descriptor_chunk_index = CHUNK_W'(trace_chunk);
                descriptor_chunk_count = CHUNK_COUNT_W'(trace_chunks);
                descriptor_lane_tile_count = LANE_COUNT_W'(trace_lane_tiles);
                descriptor_use_motion = trace_motion != 0;
                descriptor_source_bits = trace_selected;
                descriptor_negative_bits = trace_negative;
                descriptor_row_first = trace_first != 0;
                descriptor_row_last = trace_last != 0;
                descriptor_batch_last = trace_batch_last != 0;
                descriptor_state_context = STATE_CTX_W'(trace_state_context);
                descriptor_state_base_tile = STATE_BASE_TILE_W'(trace_state_base);
                descriptor_epoch = EPOCH_W'(trace_epoch);
                descriptor_domain = active_domain;
                descriptor_temporal_step = STEP_W'(trace_step);
                descriptor_temporal_length = LEN_W'(trace_length);
                descriptor_temporal_first = trace_tfirst != 0;
                descriptor_temporal_last = trace_tlast != 0;
                drive_descriptor();
                descriptors += 1;
                if (trace_batch_last != 0) begin
                    expected_contexts = trace_context + 1;
                    target_outputs += expected_contexts*expected_lane_tiles;
                    enqueue_batch_expected();
                    if (!stream_mode)
                        wait_outputs(target_outputs);
                    batches += 1;
                    trace_context = -1;
                    clear_batch_expected();
                end
            end else if (!$feof(fd)) begin
                $fatal(1, "malformed real state trace scan=%0d", scan);
            end
        end
        $fclose(fd);
        if (stream_mode)
            wait_outputs(target_outputs);
        if (current_sequence != 0 && !stream_mode)
            finish_sequence(current_sequence);
        if (trace_context != -1 || protocol_error || domain_fence_error ||
                descriptors == 0 || outputs_seen == 0 ||
                expected_read_count != expected_write_count ||
                (!perf_mode && (request_stalls == 0 || output_stalls == 0 ||
                                rmw_backpressure_cycles == 0)) ||
                (perf_mode && (request_stalls != 0 || output_stalls != 0)) ||
                (!stream_mode && (perf_local_sequences != 40 ||
                                  perf_hybrid_sequences != 40)) ||
                current_sequence != 80)
            $fatal(1, "real M4-state admission/coverage failed");
        $display("PASS_M4_STATEFUL_REAL sequences=80 batches=%0d descriptors=%0d outputs=%0d local_outputs=%0d motion_outputs=%0d request_beats=%0d bank_reads=%0d request_stalls=%0d output_stalls=%0d rmw_backpressure=%0d",
            batches, descriptors, outputs_seen, local_outputs, motion_outputs,
            request_beats, bank_reads, request_stalls, output_stalls,
            rmw_backpressure_cycles);
        if (perf_mode && !stream_mode)
            $display("PASS_M4_STATEFUL_PERF pairs=40 local_cycles=%0d hybrid_cycles=%0d",
                perf_local_cycles, perf_hybrid_cycles);
        if (stream_mode)
            $display("PASS_M4_STATEFUL_STREAMING sequences=80 batches=%0d outputs=%0d fifo_writes=%0d fifo_reads=%0d",
                batches, outputs_seen, expected_write_count,
                expected_read_count);
        if (stream_mode && perf_mode)
            $display("PASS_M4_STATEFUL_STREAMING_PERF sequences=80 batches=%0d outputs=%0d cycles=%0d",
                batches, outputs_seen, cycle_count);
        $finish;
    end
endmodule

`default_nettype wire
