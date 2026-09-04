`timescale 1ns/1ps
`default_nettype none

module tb_qfit_dual_line_descriptor_stateful_engine;
    localparam int TILE_BITS = 16;
    localparam int MAX_CHUNKS = 1;
    localparam int MAX_LANE_TILES = 2;
    localparam int ISSUE_WIDTH = 4;
    localparam int CONTEXTS = 2;
    localparam int REDUCE_SLOTS = 2;
    localparam int OUT_LANES = 4;
    localparam int TAG_W = 16;
    localparam int OBJECT_W = 16;
    localparam int W_W = 8;
    localparam int ACC_W = 32;
    localparam int STATE_CONTEXTS = 2;
    localparam int STATE_BASE_TILES = 4;
    localparam int STATE_BANKS = 2;
    localparam int STATE_LANES_PER_BANK = 2;
    localparam int EPOCH_W = 16;
    localparam int DOMAIN_W = 32;
    localparam int STEP_W = 4;
    localparam int LEN_W = 4;
    localparam int INDEX_W = $clog2(TILE_BITS);
    localparam int BANK_BITS = $clog2(ISSUE_WIDTH);
    localparam int BANK_ADDR_W = INDEX_W-BANK_BITS;
    localparam int CTX_W = $clog2(CONTEXTS);
    localparam int CTX_COUNT_W = $clog2(CONTEXTS+1);
    localparam int CHUNK_W = 1;
    localparam int CHUNK_COUNT_W = 1;
    localparam int LANE_TILE_W = 1;
    localparam int LANE_COUNT_W = 2;
    localparam int SLOT_W = 1;
    localparam int STATE_CTX_W = 1;
    localparam int STATE_BASE_TILE_W = 2;

    logic clk_core = 1'b0;
    logic por_core;
    logic rst_core;
    logic [DOMAIN_W-1:0] active_domain;
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
    logic [STATE_CTX_W-1:0] descriptor_state_context;
    logic [STATE_BASE_TILE_W-1:0] descriptor_state_base_tile;
    logic [EPOCH_W-1:0] descriptor_epoch;
    logic [DOMAIN_W-1:0] descriptor_domain;
    logic [STEP_W-1:0] descriptor_temporal_step;
    logic [LEN_W-1:0] descriptor_temporal_length;
    logic descriptor_temporal_first;
    logic descriptor_temporal_last;
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
    logic [STATE_CTX_W-1:0] output_state_context;
    logic [STATE_BASE_TILE_W-1:0] output_state_base_tile;
    logic [EPOCH_W-1:0] output_epoch;
    logic [DOMAIN_W-1:0] output_domain;
    logic [STEP_W-1:0] output_temporal_step;
    logic [LEN_W-1:0] output_temporal_length;
    logic output_temporal_first;
    logic output_temporal_last;
    logic output_used_motion;
    logic [TAG_W-1:0] output_tag;
    logic [OUT_LANES*ACC_W-1:0] output_current_acc;
    logic [2:0] controller_state;
    logic [CTX_COUNT_W-1:0] resident_contexts;
    logic state_rmw_busy;
    logic domain_fence_ready;
    logic domain_fence_error;
    logic protocol_error;

    logic [TILE_BITS-1:0] local_bits [0:CONTEXTS-1];
    logic [TILE_BITS-1:0] next_bits [0:CONTEXTS-1];
    logic [TILE_BITS-1:0] expected_bits [0:CONTEXTS-1];
    integer outputs_seen;
    integer local_outputs;
    integer motion_outputs;
    integer request_beats;
    integer bank_reads;
    integer output_stalls;
    integer cycle_count;
    integer local_start_cycle;
    integer local_end_cycle;
    integer motion_start_cycle;
    integer motion_end_cycle;

    qfit_dual_line_descriptor_stateful_engine #(
        .TILE_BITS(TILE_BITS), .MAX_CHUNKS(MAX_CHUNKS),
        .MAX_LANE_TILES(MAX_LANE_TILES), .ISSUE_WIDTH(ISSUE_WIDTH),
        .CONTEXTS(CONTEXTS), .REDUCE_SLOTS(REDUCE_SLOTS),
        .OUT_LANES(OUT_LANES), .TAG_W(TAG_W), .OBJECT_W(OBJECT_W),
        .W_W(W_W), .ACC_W(ACC_W), .STATE_CONTEXTS(STATE_CONTEXTS),
        .STATE_BASE_TILES(STATE_BASE_TILES), .STATE_BANKS(STATE_BANKS),
        .STATE_LANES_PER_BANK(STATE_LANES_PER_BANK), .EPOCH_W(EPOCH_W),
        .DOMAIN_W(DOMAIN_W), .STEP_W(STEP_W), .LEN_W(LEN_W)
    ) dut (.*);

    function automatic integer signed model_weight(
        input integer object_tag,
        input integer lane_tile,
        input integer source,
        input integer lane
    );
        model_weight = ((object_tag + lane_tile*13 + source*5 + lane*3) % 17) - 8;
    endfunction

    function automatic integer signed model_acc(
        input logic [TILE_BITS-1:0] bits,
        input integer lane_tile,
        input integer lane
    );
        integer signed value;
        begin
            value = 0;
            for (int source = 0; source < TILE_BITS; source = source + 1)
                if (bits[source])
                    value += model_weight(16'h1234, lane_tile, source, lane);
            model_acc = value;
        end
    endfunction

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
                        integer address;
                        integer source;
                        integer signed value;
                        address = weight_request_bank_addr[
                            bank*BANK_ADDR_W +: BANK_ADDR_W];
                        source = address*ISSUE_WIDTH + bank;
                        beat_reads = beat_reads + 1;
                        for (int lane = 0; lane < OUT_LANES; lane = lane + 1) begin
                            value = model_weight(weight_request_object_tag,
                                weight_request_lane_tile, source, lane);
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
            cycle_count <= 0;
            output_ready <= 1'b0;
            outputs_seen <= 0;
            local_outputs <= 0;
            motion_outputs <= 0;
            output_stalls <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            output_ready <= (cycle_count % 5) != 2;
            if (output_valid && !output_ready)
                output_stalls <= output_stalls + 1;
            if (output_valid && output_ready) begin
                integer ctx;
                integer tile;
                integer signed expected;
                ctx = output_state_context;
                tile = output_state_base_tile;
                if (ctx >= CONTEXTS || tile >= MAX_LANE_TILES)
                    $fatal(1, "M4-state output address out of range");
                if (output_epoch != 1 || output_domain != active_domain ||
                        output_temporal_length != 2 ||
                        output_tag != TAG_W'(16'h40 + ctx))
                    $fatal(1, "M4-state output identity mismatch");
                if (output_used_motion != (output_temporal_step == 1))
                    $fatal(1, "M4-state Local/Motion output mode mismatch");
                for (int lane = 0; lane < OUT_LANES; lane = lane + 1) begin
                    expected = model_acc(expected_bits[ctx], tile, lane);
                    if ($signed(output_current_acc[lane*ACC_W +: ACC_W]) !==
                            expected)
                        $fatal(1,
                            "M4-state Acc mismatch ctx=%0d tile=%0d lane=%0d got=%0d exp=%0d",
                            ctx, tile, lane,
                            $signed(output_current_acc[lane*ACC_W +: ACC_W]),
                            expected);
                end
                outputs_seen <= outputs_seen + 1;
                if (output_used_motion)
                    motion_outputs <= motion_outputs + 1;
                else
                    local_outputs <= local_outputs + 1;
            end
        end
    end

    task automatic send_row(
        input int ctx,
        input logic [TILE_BITS-1:0] selected,
        input logic [TILE_BITS-1:0] negative,
        input bit motion,
        input int step
    );
        begin
            @(negedge clk_core);
            descriptor_row_first = 1'b1;
            descriptor_row_last = 1'b1;
            descriptor_batch_last = (ctx == CONTEXTS-1);
            descriptor_tag = TAG_W'(16'h40 + ctx);
            descriptor_object_tag = 16'h1234;
            descriptor_chunk_index = 0;
            descriptor_chunk_count = 1;
            descriptor_lane_tile_count = MAX_LANE_TILES;
            descriptor_use_motion = motion;
            descriptor_source_bits = selected;
            descriptor_negative_bits = negative;
            descriptor_state_context = STATE_CTX_W'(ctx);
            descriptor_state_base_tile = 0;
            descriptor_epoch = 1;
            descriptor_domain = active_domain;
            descriptor_temporal_step = STEP_W'(step);
            descriptor_temporal_length = 2;
            descriptor_temporal_first = (step == 0);
            descriptor_temporal_last = (step == 1);
            descriptor_valid = 1'b1;
            do @(posedge clk_core); while (!descriptor_ready);
            @(negedge clk_core);
            descriptor_valid = 1'b0;
        end
    endtask

    initial begin
`ifdef VCS
        $display("SIMULATOR=Synopsys VCS");
        $display("ASSERTIONS=enabled");
`else
        $fatal(1, "M4-state integration requires Synopsys VCS");
`endif
        por_core = 1'b1;
        // POR alone must initialize M4, the adapter, and the state fabric.
        rst_core = 1'b0;
        active_domain = 32'h4d345331;
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
        weight_request_ready = 1'b1;
        local_bits[0] = 16'b1001_0010_0101_1011;
        local_bits[1] = 16'b0110_1101_1000_0110;
        next_bits[0] = 16'b1100_0011_0101_0010;
        next_bits[1] = 16'b0011_1100_1001_0110;
        expected_bits[0] = local_bits[0];
        expected_bits[1] = local_bits[1];

        repeat (4) @(posedge clk_core);
        @(negedge clk_core);
        por_core = 1'b0;
        wait (domain_fence_ready);

        local_start_cycle = cycle_count;
        for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1)
            send_row(ctx, local_bits[ctx], '0, 1'b0, 0);
        wait (outputs_seen == CONTEXTS*MAX_LANE_TILES);
        local_end_cycle = cycle_count;

        expected_bits[0] = next_bits[0];
        expected_bits[1] = next_bits[1];
        motion_start_cycle = cycle_count;
        for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1)
            send_row(ctx, local_bits[ctx] ^ next_bits[ctx],
                local_bits[ctx] & ~next_bits[ctx], 1'b1, 1);
        wait (outputs_seen == 2*CONTEXTS*MAX_LANE_TILES);
        motion_end_cycle = cycle_count;

        repeat (4) @(posedge clk_core);
        $display("M4_STATE_RESULT outputs=%0d local=%0d motion=%0d request_beats=%0d bank_reads=%0d output_stalls=%0d local_cycles=%0d motion_cycles=%0d",
            outputs_seen, local_outputs, motion_outputs, request_beats,
            bank_reads, output_stalls, local_end_cycle-local_start_cycle,
            motion_end_cycle-motion_start_cycle);
        if (outputs_seen != 8 || local_outputs != 4 || motion_outputs != 4 ||
                protocol_error || domain_fence_error)
            $fatal(1, "M4-state integration ledger mismatch");
        $display("PASS: M4 descriptor-resident Local absolute plus Motion delta shared-state miter exact");
        $finish;
    end
endmodule

`default_nettype wire
