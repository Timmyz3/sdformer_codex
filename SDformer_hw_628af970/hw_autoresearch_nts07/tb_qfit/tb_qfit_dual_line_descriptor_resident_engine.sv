`timescale 1ns/1ps
`default_nettype none

module tb_qfit_dual_line_descriptor_resident_engine;
    localparam int TILE_BITS = 32;
    localparam int MAX_CHUNKS = 3;
    localparam int MAX_LANE_TILES = 3;
    localparam int ISSUE_WIDTH = 8;
    localparam int CONTEXTS = 2;
    localparam int REDUCE_SLOTS = 2;
    localparam int OUT_LANES = 4;
    localparam int TAG_W = 16;
    localparam int OBJECT_W = 16;
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
    integer outputs_seen;
    integer request_beats;
    integer bank_reads;
    integer output_stalls;
    integer first_output_stall_cycles;

    qfit_dual_line_descriptor_resident_engine #(
        .TILE_BITS(TILE_BITS), .MAX_CHUNKS(MAX_CHUNKS),
        .MAX_LANE_TILES(MAX_LANE_TILES), .ISSUE_WIDTH(ISSUE_WIDTH),
        .CONTEXTS(CONTEXTS), .REDUCE_SLOTS(REDUCE_SLOTS),
        .OUT_LANES(OUT_LANES), .TAG_W(TAG_W), .OBJECT_W(OBJECT_W),
        .W_W(W_W), .ACC_W(ACC_W)
    ) dut (.*);

    function automatic integer signed model_weight(
        input integer object_tag,
        input integer lane_tile,
        input integer chunk,
        input integer source,
        input integer lane
    );
        model_weight = ((object_tag + lane_tile*13 + chunk*7 + source*5 + lane*3) % 17) - 8;
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

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
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
                integer slot_use [0:CONTEXTS-1];
                integer beat_bank_reads;
                request_beats <= request_beats + 1;
                beat_bank_reads = 0;
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
                        source = address * ISSUE_WIDTH + bank;
                        ctx_id = weight_request_bank_context[bank*CTX_W +: CTX_W];
                        slot = weight_request_bank_slot[bank*SLOT_W +: SLOT_W];
                        if (ctx_id >= CONTEXTS || slot >= REDUCE_SLOTS)
                            $fatal(1, "request context/slot out of range");
                        slot_use[ctx_id] += 1;
                        beat_bank_reads += 1;
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
                        $fatal(1, "compact reducer overbooked ctx=%0d slots=%0d", ctx, slot_use[ctx]);
                bank_reads <= bank_reads + beat_bank_reads;
            end
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            outputs_seen <= 0;
            output_stalls <= 0;
            first_output_stall_cycles <= 0;
            output_ready <= 1'b0;
        end else begin
            if (outputs_seen == 0 && (!output_valid || first_output_stall_cycles < 3)) begin
                output_ready <= 1'b0;
                if (output_valid)
                    first_output_stall_cycles <= first_output_stall_cycles + 1;
            end else begin
                output_ready <= ($urandom % 5) != 0;
            end
            if (output_valid && !output_ready)
                output_stalls <= output_stalls + 1;
            if (output_valid && output_ready) begin
                integer ctx_id;
                integer count;
                ctx_id = output_tag - 16'h40;
                if (ctx_id < 0 || ctx_id >= CONTEXTS)
                    $fatal(1, "bad output tag=%0h", output_tag);
                if (output_object_tag != 16'h1234)
                    $fatal(1, "bad output object=%0h", output_object_tag);
                if (output_use_motion != ctx_id[0])
                    $fatal(1, "motion identity mismatch context=%0d", ctx_id);
                count = 0;
                for (int chunk = 0; chunk < MAX_CHUNKS; chunk = chunk + 1)
                    count += popcount(expected_bits[ctx_id][chunk]);
                if (output_source_count != count)
                    $fatal(1, "source count mismatch got=%0d expected=%0d", output_source_count, count);
                for (int lane = 0; lane < OUT_LANES; lane = lane + 1) begin
                    integer signed expected;
                    integer signed actual;
                    expected = 0;
                    for (int chunk = 0; chunk < MAX_CHUNKS; chunk = chunk + 1) begin
                        for (int source = 0; source < TILE_BITS; source = source + 1) begin
                            if (expected_bits[ctx_id][chunk][source]) begin
                                if (expected_negative[ctx_id][chunk][source])
                                    expected -= model_weight(16'h1234, output_lane_tile, chunk, source, lane);
                                else
                                    expected += model_weight(16'h1234, output_lane_tile, chunk, source, lane);
                            end
                        end
                    end
                    actual = $signed(output_acc[lane*ACC_W +: ACC_W]);
                    if (actual != expected)
                        $fatal(1, "acc mismatch ctx=%0d tile=%0d lane=%0d got=%0d expected=%0d",
                            ctx_id, output_lane_tile, lane, actual, expected);
                end
                outputs_seen <= outputs_seen + 1;
            end
        end
    end

    task automatic drive_descriptor(
        input logic first,
        input logic last,
        input logic batch_last,
        input logic [TAG_W-1:0] tag,
        input logic [CHUNK_W-1:0] chunk,
        input logic motion,
        input logic [TILE_BITS-1:0] bits,
        input logic [TILE_BITS-1:0] negative
    );
        begin
            @(negedge clk_core);
            descriptor_valid = 1'b1;
            descriptor_row_first = first;
            descriptor_row_last = last;
            descriptor_batch_last = batch_last;
            descriptor_tag = tag;
            descriptor_object_tag = 16'h1234;
            descriptor_chunk_index = chunk;
            descriptor_chunk_count = CHUNK_COUNT_W'(MAX_CHUNKS);
            descriptor_lane_tile_count = LANE_COUNT_W'(2);
            descriptor_use_motion = motion;
            descriptor_source_bits = bits;
            descriptor_negative_bits = negative;
            do @(posedge clk_core); while (!descriptor_ready);
            @(negedge clk_core);
            descriptor_valid = 1'b0;
        end
    endtask

    task automatic wait_outputs(input integer target);
        integer timeout;
        begin
            timeout = 0;
            while (outputs_seen < target && timeout < 20000) begin
                @(posedge clk_core);
                timeout += 1;
            end
            if (outputs_seen != target)
                $fatal(1, "output timeout got=%0d target=%0d state=%0d", outputs_seen, target, controller_state);
        end
    endtask

    initial begin : stimulus
        logic [TILE_BITS-1:0] bits;
        logic [TILE_BITS-1:0] negative;
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
        weight_request_ready = 1'b1;
        for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1)
            for (int chunk = 0; chunk < MAX_CHUNKS; chunk = chunk + 1) begin
                expected_bits[ctx][chunk] = '0;
                expected_negative[ctx][chunk] = '0;
            end
        repeat (8) @(posedge clk_core);
        rst_core = 1'b0;

        for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
            for (int chunk = 0; chunk < MAX_CHUNKS; chunk = chunk + 1) begin
                bits = '0;
                negative = '0;
                for (int source = 0; source < TILE_BITS; source = source + 1) begin
                    if (((source*3 + chunk*5 + ctx*7) % 11) < 5) begin
                        bits[source] = 1'b1;
                        if (ctx == 1 && ((source + chunk) % 3) == 0)
                            negative[source] = 1'b1;
                    end
                end
                // Exercise an empty resident chunk without special-casing it.
                if (ctx == 0 && chunk == 1) begin
                    bits = '0;
                    negative = '0;
                end
                expected_bits[ctx][chunk] = bits;
                expected_negative[ctx][chunk] = negative;
                drive_descriptor(
                    chunk == 0, chunk == MAX_CHUNKS-1,
                    ctx == CONTEXTS-1 && chunk == MAX_CHUNKS-1,
                    TAG_W'(16'h40 + ctx), CHUNK_W'(chunk), ctx[0], bits, negative
                );
            end
        end

        // Hold a legal next-batch descriptor while the resident batch owns the
        // engine.  ready may stay low, but the payload must not trigger fault.
        fork
            begin
                @(negedge clk_core);
                descriptor_valid = 1'b1;
                descriptor_row_first = 1'b1;
                descriptor_row_last = 1'b1;
                descriptor_batch_last = 1'b1;
                descriptor_tag = 16'h99;
                descriptor_object_tag = 16'h9999;
                descriptor_chunk_index = '0;
                descriptor_chunk_count = CHUNK_COUNT_W'(1);
                descriptor_lane_tile_count = LANE_COUNT_W'(1);
                descriptor_use_motion = 1'b0;
                descriptor_source_bits = '0;
                descriptor_negative_bits = '0;
                repeat (5) @(posedge clk_core);
                if (descriptor_ready || protocol_error)
                    $fatal(1, "held descriptor ready/fault contract failed");
                @(negedge clk_core);
                descriptor_valid = 1'b0;
            end
            wait_outputs(CONTEXTS * 2);
        join

        if (protocol_error)
            $fatal(1, "unexpected protocol error");
        if (output_stalls == 0)
            $fatal(1, "output backpressure was not exercised");
        if (request_beats == 0 || bank_reads == 0)
            $fatal(1, "weight interface was not exercised");
        $display("PASS_M4_DESCRIPTOR_RESIDENT outputs=%0d request_beats=%0d bank_reads=%0d stalls=%0d",
            outputs_seen, request_beats, bank_reads, output_stalls);
        $finish;
    end
endmodule

`default_nettype wire
