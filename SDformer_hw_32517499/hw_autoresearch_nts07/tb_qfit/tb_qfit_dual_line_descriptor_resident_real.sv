`timescale 1ns/1ps
`default_nettype none

module tb_qfit_dual_line_descriptor_resident_real;
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
    localparam int BANK_ADDR_W = $clog2(TILE_BITS) - $clog2(ISSUE_WIDTH);
    localparam int CTX_W = $clog2(CONTEXTS);
    localparam int CTX_COUNT_W = $clog2(CONTEXTS + 1);
    localparam int CHUNK_W = $clog2(MAX_CHUNKS);
    localparam int CHUNK_COUNT_W = $clog2(MAX_CHUNKS + 1);
    localparam int LANE_TILE_W = $clog2(MAX_LANE_TILES);
    localparam int LANE_COUNT_W = $clog2(MAX_LANE_TILES + 1);
    localparam int SLOT_W = $clog2(REDUCE_SLOTS);
    localparam int SOURCE_COUNT_W = $clog2(MAX_CHUNKS*TILE_BITS + 1);

    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    always #1.5 clk_core = ~clk_core;

    logic descriptor_valid;
    logic descriptor_ready;
    logic descriptor_row_first;
    logic descriptor_row_last;
    logic descriptor_batch_last;
    logic [TAG_W-1:0] descriptor_tag;
    logic [OBJECT_W-1:0] descriptor_object_tag;
    logic [CHUNK_W-1:0] descriptor_chunk_index;
    logic [CHUNK_COUNT_W-1:0] descriptor_chunk_count;
    logic [LANE_COUNT_W-1:0] descriptor_lane_tile_count;
    logic descriptor_use_motion;
    logic [TILE_BITS-1:0] descriptor_source_bits;
    logic [TILE_BITS-1:0] descriptor_negative_bits;
    logic weight_request_valid;
    logic weight_request_ready;
    logic [OBJECT_W-1:0] weight_request_object_tag;
    logic [CHUNK_W-1:0] weight_request_chunk_index;
    logic [LANE_TILE_W-1:0] weight_request_lane_tile;
    logic [ISSUE_WIDTH-1:0] weight_request_bank_valid;
    logic [ISSUE_WIDTH*BANK_ADDR_W-1:0] weight_request_bank_addr;
    logic [ISSUE_WIDTH*CTX_W-1:0] weight_request_bank_context;
    logic [ISSUE_WIDTH*SLOT_W-1:0] weight_request_bank_slot;
    logic [ISSUE_WIDTH-1:0] weight_request_bank_negative;
    logic weight_response_valid;
    logic weight_response_ready;
    logic [ISSUE_WIDTH-1:0] weight_response_bank_valid;
    logic [ISSUE_WIDTH*OUT_LANES*W_W-1:0] weight_response_data;
    logic output_valid;
    logic output_ready;
    logic [TAG_W-1:0] output_tag;
    logic [OBJECT_W-1:0] output_object_tag;
    logic [LANE_TILE_W-1:0] output_lane_tile;
    logic output_use_motion;
    logic [SOURCE_COUNT_W-1:0] output_source_count;
    logic [OUT_LANES*ACC_W-1:0] output_acc;
    logic [2:0] controller_state;
    logic [CTX_COUNT_W-1:0] resident_contexts;
    logic protocol_error;

    logic [TILE_BITS-1:0] expected_bits [0:CONTEXTS-1][0:MAX_CHUNKS-1];
    logic [TILE_BITS-1:0] expected_negative [0:CONTEXTS-1][0:MAX_CHUNKS-1];
    logic [TILE_BITS-1:0] requested_bits
        [0:CONTEXTS-1][0:MAX_LANE_TILES-1][0:MAX_CHUNKS-1];
    logic [TAG_W-1:0] expected_tag [0:CONTEXTS-1];
    logic expected_motion [0:CONTEXTS-1];
    logic [OBJECT_W-1:0] expected_object;
    integer expected_chunks;
    integer expected_lane_tiles;
    integer expected_contexts;
    integer outputs_seen;
    integer request_beats;
    integer bank_reads;
    integer request_stalls;
    integer source_scoreboard_checks;
    integer output_stalls;
    integer ideal_wall_cycles;
    integer batch_wall_cycles;
    integer expected_wall_cycles;
    logic ideal_mode;
    logic weight_stall_mode;
    logic batch_wall_active;

    qfit_dual_line_descriptor_resident_engine #(
        .TILE_BITS(TILE_BITS), .MAX_CHUNKS(MAX_CHUNKS),
        .MAX_LANE_TILES(MAX_LANE_TILES), .ISSUE_WIDTH(ISSUE_WIDTH),
        .CONTEXTS(CONTEXTS), .REDUCE_SLOTS(REDUCE_SLOTS),
        .OUT_LANES(OUT_LANES), .TAG_W(TAG_W), .OBJECT_W(OBJECT_W),
        .W_W(W_W), .ACC_W(ACC_W)
    ) dut (.*);

    function automatic integer signed model_weight(
        input logic [OBJECT_W-1:0] object_tag,
        input integer lane_tile,
        input integer chunk,
        input integer source,
        input integer lane
    );
        longint unsigned mixed;
        begin
            mixed = $unsigned(object_tag[31:0])
                + lane_tile*13 + chunk*7 + source*5 + lane*3;
            model_weight = (mixed % 17) - 8;
        end
    endfunction

    function automatic integer popcount(input logic [TILE_BITS-1:0] value);
        integer count;
        begin
            count = 0;
            for (int source = 0; source < TILE_BITS; source = source + 1)
                count += value[source];
            popcount = count;
        end
    endfunction

    task automatic clear_expected;
        begin
            expected_contexts = 0;
            expected_chunks = 0;
            expected_lane_tiles = 0;
            expected_object = '0;
            for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
                expected_tag[ctx] = '0;
                expected_motion[ctx] = 1'b0;
                for (int chunk = 0; chunk < MAX_CHUNKS; chunk = chunk + 1) begin
                    expected_bits[ctx][chunk] = '0;
                    expected_negative[ctx][chunk] = '0;
                    for (int lane_tile = 0; lane_tile < MAX_LANE_TILES;
                            lane_tile = lane_tile + 1)
                        requested_bits[ctx][lane_tile][chunk] = '0;
                end
            end
        end
    endtask

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            weight_request_ready <= 1'b1;
            request_stalls <= 0;
        end else begin
            weight_request_ready <= weight_stall_mode
                ? (($urandom % 5) != 0) : 1'b1;
            if (weight_request_valid && !weight_request_ready)
                request_stalls <= request_stalls + 1;
        end
    end

    // Testbench response model also owns the per-batch source scoreboard,
    // which is synchronously cleared by the stimulus task at batch fences.
    always @(posedge clk_core) begin
        if (rst_core) begin
            weight_response_valid <= 1'b0;
            weight_response_bank_valid <= '0;
            weight_response_data <= '0;
            request_beats <= 0;
            bank_reads <= 0;
            source_scoreboard_checks <= 0;
        end else begin
            weight_response_valid <= weight_request_valid && weight_request_ready;
            weight_response_bank_valid <= weight_request_bank_valid;
            weight_response_data <= '0;
            if (weight_request_valid && weight_request_ready) begin
                integer slot_use [0:CONTEXTS-1];
                integer beat_reads;
                request_beats <= request_beats + 1;
                beat_reads = 0;
                for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1)
                    slot_use[ctx] = 0;
                for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin
                    if (weight_request_bank_valid[bank]) begin
                        integer address;
                        integer source;
                        integer ctx_id;
                        integer slot;
                        integer signed value;
                        address = weight_request_bank_addr[bank*BANK_ADDR_W +: BANK_ADDR_W];
                        source = address*ISSUE_WIDTH + bank;
                        ctx_id = weight_request_bank_context[bank*CTX_W +: CTX_W];
                        slot = weight_request_bank_slot[bank*SLOT_W +: SLOT_W];
                        if (ctx_id >= CONTEXTS || slot >= REDUCE_SLOTS)
                            $fatal(1, "real request context/slot out of range");
                        if (weight_request_chunk_index >= expected_chunks
                                || weight_request_lane_tile >= expected_lane_tiles)
                            $fatal(1, "real request geometry out of range");
                        if (!expected_bits[ctx_id][weight_request_chunk_index][source])
                            $fatal(1,
                                "real request selected inactive source ctx=%0d tile=%0d chunk=%0d source=%0d",
                                ctx_id, weight_request_lane_tile,
                                weight_request_chunk_index, source);
                        if (requested_bits[ctx_id][weight_request_lane_tile]
                                [weight_request_chunk_index][source])
                            $fatal(1,
                                "real request duplicated source ctx=%0d tile=%0d chunk=%0d source=%0d",
                                ctx_id, weight_request_lane_tile,
                                weight_request_chunk_index, source);
                        if (weight_request_bank_negative[bank]
                                != expected_negative[ctx_id]
                                    [weight_request_chunk_index][source])
                            $fatal(1,
                                "real request sign mismatch ctx=%0d tile=%0d chunk=%0d source=%0d",
                                ctx_id, weight_request_lane_tile,
                                weight_request_chunk_index, source);
                        requested_bits[ctx_id][weight_request_lane_tile]
                            [weight_request_chunk_index][source] <= 1'b1;
                        slot_use[ctx_id] += 1;
                        beat_reads += 1;
                        for (int lane = 0; lane < OUT_LANES; lane = lane + 1) begin
                            value = model_weight(
                                weight_request_object_tag, weight_request_lane_tile,
                                weight_request_chunk_index, source, lane
                            );
                            weight_response_data[(bank*OUT_LANES + lane)*W_W +: W_W]
                                <= W_W'(value);
                        end
                    end
                end
                for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1)
                    if (slot_use[ctx] > REDUCE_SLOTS)
                        $fatal(1, "real compact reducer overbooked");
                bank_reads <= bank_reads + beat_reads;
                source_scoreboard_checks <= source_scoreboard_checks + beat_reads;
            end
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            outputs_seen <= 0;
            output_stalls <= 0;
            output_ready <= 1'b0;
        end else begin
            output_ready <= ideal_mode ? 1'b1 : (($urandom % 7) != 0);
            if (output_valid && !output_ready)
                output_stalls <= output_stalls + 1;
            if (output_valid && output_ready) begin
                integer ctx_id;
                integer count;
                ctx_id = -1;
                for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1)
                    if (ctx < expected_contexts && output_tag == expected_tag[ctx])
                        ctx_id = ctx;
                if (ctx_id < 0)
                    $fatal(1, "real output tag not resident: %0h", output_tag);
                if (output_object_tag != expected_object)
                    $fatal(1, "real output object mismatch");
                if (output_lane_tile >= expected_lane_tiles)
                    $fatal(1, "real output lane tile out of range");
                if (output_use_motion != expected_motion[ctx_id])
                    $fatal(1, "real motion identity mismatch");
                for (int chunk = 0; chunk < expected_chunks; chunk = chunk + 1)
                    if (requested_bits[ctx_id][output_lane_tile][chunk]
                            != expected_bits[ctx_id][chunk])
                        $fatal(1,
                            "real requested-source conservation mismatch ctx=%0d tile=%0d chunk=%0d",
                            ctx_id, output_lane_tile, chunk);
                count = 0;
                for (int chunk = 0; chunk < expected_chunks; chunk = chunk + 1)
                    count += popcount(expected_bits[ctx_id][chunk]);
                if (output_source_count != count)
                    $fatal(1, "real source count mismatch got=%0d expected=%0d",
                        output_source_count, count);
                for (int lane = 0; lane < OUT_LANES; lane = lane + 1) begin
                    integer signed expected;
                    integer signed actual;
                    expected = 0;
                    for (int chunk = 0; chunk < expected_chunks; chunk = chunk + 1) begin
                        for (int source = 0; source < TILE_BITS; source = source + 1) begin
                            if (expected_bits[ctx_id][chunk][source]) begin
                                if (expected_negative[ctx_id][chunk][source])
                                    expected -= model_weight(
                                        expected_object, output_lane_tile, chunk, source, lane
                                    );
                                else
                                    expected += model_weight(
                                        expected_object, output_lane_tile, chunk, source, lane
                                    );
                            end
                        end
                    end
                    actual = $signed(output_acc[lane*ACC_W +: ACC_W]);
                    if (actual != expected)
                        $fatal(1, "real acc mismatch ctx=%0d tile=%0d lane=%0d got=%0d expected=%0d",
                            ctx_id, output_lane_tile, lane, actual, expected);
                end
                outputs_seen <= outputs_seen + 1;
            end
        end
    end

    // Sum controller-active cycles per batch, excluding intentional TB gaps
    // between batches. In ideal mode descriptors are contiguous within LOAD
    // and output_ready is fixed high, so this must equal the executable model.
    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            ideal_wall_cycles <= 0;
            batch_wall_cycles <= 0;
            batch_wall_active <= 1'b0;
        end else if (!batch_wall_active && descriptor_valid && descriptor_ready) begin
            batch_wall_active <= 1'b1;
            batch_wall_cycles <= 1;
        end else if (batch_wall_active) begin
            if (output_valid && output_ready
                    && output_lane_tile == expected_lane_tiles - 1
                    && output_tag == expected_tag[expected_contexts - 1]) begin
                ideal_wall_cycles <= ideal_wall_cycles + batch_wall_cycles + 1;
                batch_wall_cycles <= 0;
                batch_wall_active <= 1'b0;
            end else begin
                batch_wall_cycles <= batch_wall_cycles + 1;
            end
        end
    end

    task automatic drive_descriptor(
        input logic [OBJECT_W-1:0] object_tag,
        input logic [TAG_W-1:0] tag,
        input integer chunk,
        input integer chunks,
        input integer lane_tiles,
        input logic motion,
        input logic [TILE_BITS-1:0] bits,
        input logic [TILE_BITS-1:0] negative,
        input logic first,
        input logic last,
        input logic batch_last
    );
        begin
            @(negedge clk_core);
            descriptor_valid = 1'b1;
            descriptor_object_tag = object_tag;
            descriptor_tag = tag;
            descriptor_chunk_index = CHUNK_W'(chunk);
            descriptor_chunk_count = CHUNK_COUNT_W'(chunks);
            descriptor_lane_tile_count = LANE_COUNT_W'(lane_tiles);
            descriptor_use_motion = motion;
            descriptor_source_bits = bits;
            descriptor_negative_bits = negative;
            descriptor_row_first = first;
            descriptor_row_last = last;
            descriptor_batch_last = batch_last;
            do @(posedge clk_core); while (!descriptor_ready);
            if (!ideal_mode || batch_last) begin
                @(negedge clk_core);
                descriptor_valid = 1'b0;
            end
        end
    endtask

    task automatic wait_outputs(input integer target);
        integer timeout;
        begin
            timeout = 0;
            while (outputs_seen < target && timeout < 10_000_000) begin
                @(posedge clk_core);
                timeout += 1;
            end
            if (outputs_seen != target)
                $fatal(1, "real output timeout got=%0d target=%0d state=%0d",
                    outputs_seen, target, controller_state);
        end
    endtask

    initial begin : stimulus
        string trace_path;
        integer trace_fd;
        integer trace_scan;
        integer trace_chunk;
        integer trace_chunks;
        integer trace_lane_tiles;
        integer trace_motion;
        integer trace_first;
        integer trace_last;
        integer trace_batch_last;
        integer trace_context;
        integer target_outputs;
        integer descriptors;
        integer batches;
        logic [OBJECT_W-1:0] trace_object;
        logic [TAG_W-1:0] trace_tag;
        logic [TILE_BITS-1:0] trace_bits;
        logic [TILE_BITS-1:0] trace_negative;

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
        ideal_mode = $test$plusargs("IDEAL_WALL_CYCLES");
        weight_stall_mode = $test$plusargs("RANDOM_WEIGHT_BACKPRESSURE");
        if (ideal_mode && weight_stall_mode)
            $fatal(1, "ideal wall-cycle mode forbids weight backpressure");
        expected_wall_cycles = 0;
        if (ideal_mode && !$value$plusargs(
                "EXPECTED_WALL_CYCLES=%d", expected_wall_cycles))
            $fatal(1, "EXPECTED_WALL_CYCLES is required in ideal mode");
        clear_expected();
        repeat (8) @(posedge clk_core);
        rst_core = 1'b0;
        if (!$value$plusargs("REAL_TRACE=%s", trace_path))
            $fatal(1, "REAL_TRACE is required");
        trace_fd = $fopen(trace_path, "r");
        if (trace_fd == 0)
            $fatal(1, "cannot open REAL_TRACE=%s", trace_path);
        trace_context = -1;
        target_outputs = 0;
        descriptors = 0;
        batches = 0;
        while (!$feof(trace_fd)) begin
            trace_scan = $fscanf(
                trace_fd, "%h %h %d %d %d %d %h %h %d %d %d\n",
                trace_object, trace_tag, trace_chunk, trace_chunks,
                trace_lane_tiles, trace_motion, trace_bits, trace_negative,
                trace_first, trace_last, trace_batch_last
            );
            if (trace_scan == 11) begin
                if (trace_first != 0) begin
                    trace_context += 1;
                    if (trace_context >= CONTEXTS)
                        $fatal(1, "real batch exceeds context capacity");
                    expected_tag[trace_context] = trace_tag;
                    expected_motion[trace_context] = trace_motion != 0;
                    if (trace_context == 0) begin
                        expected_object = trace_object;
                        expected_chunks = trace_chunks;
                        expected_lane_tiles = trace_lane_tiles;
                    end else if (trace_object != expected_object
                            || trace_chunks != expected_chunks
                            || trace_lane_tiles != expected_lane_tiles) begin
                        $fatal(1, "real batch geometry changed across contexts");
                    end
                end
                if (trace_context < 0 || trace_chunk < 0 || trace_chunk >= trace_chunks
                        || trace_chunks > MAX_CHUNKS || trace_lane_tiles > MAX_LANE_TILES)
                    $fatal(1, "real descriptor geometry invalid");
                expected_bits[trace_context][trace_chunk] = trace_bits;
                expected_negative[trace_context][trace_chunk] = trace_negative;
                drive_descriptor(
                    trace_object, trace_tag, trace_chunk, trace_chunks,
                    trace_lane_tiles, trace_motion != 0, trace_bits, trace_negative,
                    trace_first != 0, trace_last != 0, trace_batch_last != 0
                );
                descriptors += 1;
                if (trace_batch_last != 0) begin
                    expected_contexts = trace_context + 1;
                    target_outputs += expected_contexts * expected_lane_tiles;
                    wait_outputs(target_outputs);
                    batches += 1;
                    trace_context = -1;
                    clear_expected();
                end
            end else if (!$feof(trace_fd)) begin
                $fatal(1, "malformed real descriptor trace scan=%0d", trace_scan);
            end
        end
        $fclose(trace_fd);
        if (trace_context != -1)
            $fatal(1, "real descriptor trace ended inside a batch");
        if (protocol_error || descriptors == 0 || batches == 0
                || request_beats == 0 || bank_reads == 0
                || source_scoreboard_checks == 0
                || (!ideal_mode && output_stalls == 0)
                || (weight_stall_mode && request_stalls == 0))
            $fatal(1, "real M4 coverage/admission failed");
        if (batch_wall_active)
            $fatal(1, "ideal wall-cycle counter ended inside a batch");
        if (ideal_mode && ideal_wall_cycles != expected_wall_cycles)
            $fatal(1, "ideal wall cycles mismatch got=%0d expected=%0d",
                ideal_wall_cycles, expected_wall_cycles);
        $display("PASS_M4_DESCRIPTOR_RESIDENT_REAL batches=%0d descriptors=%0d outputs=%0d request_beats=%0d bank_reads=%0d output_stalls=%0d request_stalls=%0d source_checks=%0d wall_cycles=%0d ideal=%0d",
            batches, descriptors, outputs_seen, request_beats, bank_reads,
            output_stalls, request_stalls, source_scoreboard_checks,
            ideal_wall_cycles, ideal_mode);
        if ($test$plusargs("UCLI_SAIF_STOP"))
            $stop;
        else
            $finish;
    end
endmodule

`default_nettype wire
