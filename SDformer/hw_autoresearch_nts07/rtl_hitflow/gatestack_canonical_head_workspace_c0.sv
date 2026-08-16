`timescale 1ns/1ps
`default_nettype none

// C0 canonical workspace. It captures one complete final-gate/K head, builds
// the exact gate/lane destination directory, and replays canonical streams.
module gatestack_canonical_head_workspace_c0 #(
    parameter int TOKENS        = 162,
    parameter int LANES         = 32,
    parameter int GATE_W        = 9,
    parameter int CLASS_SLOTS   = 4,
    parameter int CONTEXTS      = 2,
    parameter int HEADS         = 24,
    parameter int TAG_W         = 32,
    parameter int FORMAT_W      = 2,
    parameter int COUNTER_W     = 32,
    parameter int DESTINATION_SCAN_MODE = 1,
    parameter int BITMAP_BYPASS_ENABLE = 0,
    parameter int EXPLICIT_BITMAP_BANK_ENABLE = 0,
    parameter int CONTEXT_ID_W  = (CONTEXTS <= 1) ? 1 : $clog2(CONTEXTS),
    parameter int HEAD_ID_W     = (HEADS <= 1) ? 1 : $clog2(HEADS),
    parameter int TOKEN_ID_W    = (TOKENS <= 1) ? 1 : $clog2(TOKENS),
    parameter int TERM_INDEX_W  =
        ((CLASS_SLOTS * LANES) <= 1) ? 1 : $clog2(CLASS_SLOTS * LANES),
    parameter int CLASS_ID_W    =
        (CLASS_SLOTS <= 1) ? 1 : $clog2(CLASS_SLOTS)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         head_begin_valid,
    output logic                         head_begin_ready,
    input  logic [CONTEXT_ID_W-1:0]      head_context_id,
    input  logic [HEAD_ID_W-1:0]         head_id,
    input  logic [TAG_W-1:0]             head_tag,

    input  logic                         token_valid,
    output logic                         token_ready,
    input  logic [TOKEN_ID_W-1:0]        token_id,
    input  logic [GATE_W-1:0]            token_gate_code,
    input  logic [LANES-1:0]             token_k_bits,
    input  logic                         token_last,

    output logic                         metadata_valid,
    input  logic                         metadata_ready,
    output logic [CONTEXT_ID_W-1:0]      metadata_context_id,
    output logic [HEAD_ID_W-1:0]         metadata_head_id,
    output logic [TAG_W-1:0]             metadata_tag,
    output logic [3:0]                   metadata_active_classes,
    output logic [7:0]                   metadata_active_tokens,
    output logic [7:0]                   metadata_term_count,
    output logic [12:0]                  metadata_event_count,
    output logic [7:0]                   metadata_bitmap_term_count,
    output logic [12:0]                  metadata_fadc_destination_bytes,
    output logic                         metadata_overflow,
    output logic                         raw_capture_error,

    input  logic                         emit_start_valid,
    output logic                         emit_start_ready,
    input  logic [FORMAT_W-1:0]          emit_start_format,

    output logic                         descriptor_valid,
    input  logic                         descriptor_ready,
    output logic [GATE_W-1:0]            descriptor_gate_code,
    output logic [4:0]                   descriptor_lane_id,
    output logic [7:0]                   descriptor_destination_count,
    output logic                         descriptor_last,

    output logic                         destination_valid,
    input  logic                         destination_ready,
    output logic [7:0]                   destination_token_id,
    output logic                         destination_last_for_term,
    output logic                         destination_bitmap_valid,
    input  logic                         destination_bitmap_ready,
    output logic [TOKENS-1:0]            destination_bitmap,

    output logic                         raw_token_valid,
    input  logic                         raw_token_ready,
    output logic [7:0]                   raw_token_id,
    output logic [GATE_W-1:0]            raw_gate_code,
    output logic [LANES-1:0]             raw_k_bits,

    output logic                         emit_done_valid,
    input  logic                         emit_done_ready,
    output logic [TAG_W-1:0]             emit_done_tag,
    output logic                         emit_done_error,

    output logic                         protocol_error,
    output logic [COUNTER_W-1:0]         count_heads,
    output logic [COUNTER_W-1:0]         count_raw_fallback_heads,
    output logic [COUNTER_W-1:0]         count_emitted_terms,
    output logic [COUNTER_W-1:0]         count_emitted_destinations,
    output logic [COUNTER_W-1:0]         count_destination_scan_cycles,
    output logic [COUNTER_W-1:0]         count_output_stall_cycles
);

    localparam logic [FORMAT_W-1:0] FORMAT_RAW = FORMAT_W'(0);
    localparam logic [FORMAT_W-1:0] FORMAT_FADC24 = FORMAT_W'(2);
    localparam int DIRECTORY_ENTRIES = CLASS_SLOTS * LANES;
    localparam int DEST_SEGMENT_W = 32'd16;
    localparam int DEST_SEGMENTS =
        (TOKENS + DEST_SEGMENT_W - 1) / DEST_SEGMENT_W;
    localparam int DEST_PADDED_TOKENS = DEST_SEGMENTS * DEST_SEGMENT_W;
    localparam int DEST_SEGMENT_INDEX_W =
        (DEST_SEGMENTS <= 1) ? 1 : $clog2(DEST_SEGMENTS);
    localparam int LANE_ID_W = (LANES <= 1) ? 1 : $clog2(LANES);
    // Untyped aliases keep the independent Verilog-2001 lint parser aware
    // that parameterized loop bounds are elaboration-time constants.
    localparam LOOP_LANES = LANES;
    localparam LOOP_CLASS_SLOTS = CLASS_SLOTS;
    localparam LOOP_DEST_SEGMENTS = DEST_SEGMENTS;
    localparam LOOP_DEST_SEGMENT_W = DEST_SEGMENT_W;

    typedef enum logic [3:0] {
        ST_IDLE,
        ST_CAPTURE,
        ST_ANALYZE_SELECT,
        ST_ANALYZE_LANES,
        ST_METADATA,
        ST_READY,
        ST_DESCRIPTORS,
        ST_DESTINATIONS,
        ST_RAW,
        ST_DONE
    } state_t;

    state_t state_q;
    logic [CONTEXT_ID_W-1:0] context_q;
    logic [HEAD_ID_W-1:0] head_q;
    logic [TAG_W-1:0] tag_q;
    logic [FORMAT_W-1:0] emit_format_q;
    logic [TOKEN_ID_W-1:0] capture_index_q;
    logic [3:0] class_count_q;
    logic [7:0] active_token_count_q;
    logic [12:0] event_count_q;
    logic metadata_overflow_q;
    logic raw_capture_error_q;

    logic [CLASS_SLOTS-1:0] class_valid_q;
    logic [GATE_W-1:0] class_gate_q [0:CLASS_SLOTS-1];
    logic [7:0] fanout_q [0:CLASS_SLOTS-1][0:LANES-1];
    logic [GATE_W+LANES-1:0] raw_record_q [0:TOKENS-1];

    // {class slot, destination count, lane, gate}
    logic [CLASS_ID_W+8+5+GATE_W-1:0] term_q [0:DIRECTORY_ENTRIES-1];
    logic [7:0] term_count_q;
    logic [7:0] bitmap_term_count_q;
    logic [12:0] fadc_destination_bytes_q;

    logic [CLASS_SLOTS-1:0] analyzed_class_mask_q;
    logic [CLASS_ID_W-1:0] analyze_class_q;
    logic [LANE_ID_W-1:0] analyze_lane_q;
    logic [3:0] analyzed_class_count_q;

    logic [TERM_INDEX_W-1:0] emit_term_index_q;
    logic [TOKEN_ID_W-1:0] destination_scan_index_q;
    logic [TOKENS-1:0] destination_remaining_q;
    logic [7:0] destination_seen_q;
    logic [TOKEN_ID_W-1:0] raw_emit_index_q;
    logic emit_done_error_q;

    logic token_fire;
    logic token_active;
    logic class_found;
    logic free_found;
    logic [CLASS_ID_W-1:0] found_class;
    logic [CLASS_ID_W-1:0] free_class;
    logic [CLASS_ID_W-1:0] selected_class;
    logic selected_class_found;
    logic [GATE_W-1:0] selected_class_gate;
    logic [5:0] token_popcount;
    logic descriptor_fire;
    logic destination_fire;
    logic destination_bitmap_fire;
    logic destination_bitmap_bypass;
    logic raw_fire;
    logic [CLASS_ID_W-1:0] emit_class;
    logic [LANE_ID_W-1:0] emit_lane;
    logic [7:0] emit_fanout;
    logic linear_destination_bit;
    logic [DEST_PADDED_TOKENS-1:0] destination_padded;
    logic segmented_destination_found;
    logic [7:0] segmented_destination_token;
    logic [DEST_SEGMENT_W-1:0] selected_segment_bits;
    logic [DEST_SEGMENT_INDEX_W-1:0] selected_segment_index;
    logic [$clog2(DEST_SEGMENT_W)-1:0] selected_bit_index;
    logic selected_segment_found;
    logic selected_bit_found;
    logic [TERM_INDEX_W-1:0] next_emit_term_index;
    logic [CLASS_ID_W-1:0] next_emit_class;
    logic [LANE_ID_W-1:0] next_emit_lane;
    logic [CLASS_ID_W-1:0] bitmap_read_class;
    logic [LANE_ID_W-1:0] bitmap_read_lane;
    logic [TOKENS-1:0] bitmap_read_data;

    // Descriptor completion loads term zero. Destination completion prepares
    // the following term. Only one transposed bitmap read is needed per cycle.
    always_comb begin
        bitmap_read_class = emit_class;
        bitmap_read_lane = emit_lane;
        if (state_q == ST_DESCRIPTORS) begin
            bitmap_read_class = CLASS_ID_W'(
                term_q[0][GATE_W+5+8 +: CLASS_ID_W]);
            bitmap_read_lane = LANE_ID_W'(
                term_q[0][GATE_W +: 5]);
        end else if (state_q == ST_DESTINATIONS &&
                     32'(emit_term_index_q) + 1 < 32'(term_count_q)) begin
            bitmap_read_class = next_emit_class;
            bitmap_read_lane = next_emit_lane;
        end
    end

    generate
        if (EXPLICIT_BITMAP_BANK_ENABLE != 0) begin : g_explicit_bitmap_bank
            gatestack_transposed_bitmap_bank #(
                .TOKENS(TOKENS),
                .LANES(LANES),
                .CLASS_SLOTS(CLASS_SLOTS),
                .SEGMENT_W(DEST_SEGMENT_W)
            ) u_transposed_bitmap_bank (
                .clk_core(clk_core),
                .rst_core(rst_core),
                .clear_valid(head_begin_valid && head_begin_ready),
                .write_valid(token_fire && token_active &&
                             (class_found || free_found)),
                .write_class_id(class_found ? found_class : free_class),
                .write_token_id(capture_index_q),
                .write_lane_bits(token_k_bits),
                .read_class_id(bitmap_read_class),
                .read_lane_id(bitmap_read_lane),
                .read_bitmap(bitmap_read_data)
            );
        end else begin : g_implicit_bitmap
            logic [TOKENS-1:0] destination_bitmap_q
                [0:CLASS_SLOTS-1][0:LANES-1];

            assign bitmap_read_data =
                destination_bitmap_q[bitmap_read_class][bitmap_read_lane];

            always_ff @(posedge clk_core) begin
                if (!rst_core && token_fire && token_active) begin
                    if (class_found) begin
                        for (int lane = 32'd0; lane < LOOP_LANES;
                             lane = lane + 32'd1) begin
                            if (token_k_bits[lane])
                                destination_bitmap_q[found_class][lane]
                                                    [capture_index_q] <= 1'b1;
                        end
                    end else if (free_found) begin
                        for (int lane = 32'd0; lane < LOOP_LANES;
                             lane = lane + 32'd1) begin
                            destination_bitmap_q[free_class][lane] <=
                                token_k_bits[lane] ?
                                ({{(TOKENS-1){1'b0}}, 1'b1} <<
                                 capture_index_q) : '0;
                        end
                    end
                end
            end
        end
    endgenerate

    always_comb begin
        token_popcount = '0;
        for (int lane = 32'd0; lane < LOOP_LANES; lane = lane + 32'd1)
            token_popcount = token_popcount + 6'(token_k_bits[lane]);
    end

    always_comb begin
        destination_padded = '0;
        destination_padded[TOKENS-1:0] = destination_remaining_q;
        selected_segment_found = 1'b0;
        selected_segment_index = '0;
        for (int segment = 32'd0; segment < LOOP_DEST_SEGMENTS;
             segment = segment + 32'd1) begin
            if (!selected_segment_found &&
                |destination_padded[segment * DEST_SEGMENT_W +:
                                    DEST_SEGMENT_W]) begin
                selected_segment_found = 1'b1;
                selected_segment_index = DEST_SEGMENT_INDEX_W'(segment);
            end
        end
        selected_segment_bits = destination_padded[
            32'(selected_segment_index) * DEST_SEGMENT_W +: DEST_SEGMENT_W];
        selected_bit_found = 1'b0;
        selected_bit_index = '0;
        for (int bit_index = 32'd0; bit_index < LOOP_DEST_SEGMENT_W;
             bit_index = bit_index + 32'd1) begin
            if (!selected_bit_found && selected_segment_bits[bit_index]) begin
                selected_bit_found = 1'b1;
                selected_bit_index = $clog2(DEST_SEGMENT_W)'(bit_index);
            end
        end
        segmented_destination_found =
            selected_segment_found && selected_bit_found;
        segmented_destination_token =
            8'((32'(selected_segment_index) * DEST_SEGMENT_W) +
               32'(selected_bit_index));
    end

    always_comb begin
        class_found = 1'b0;
        free_found = 1'b0;
        found_class = '0;
        free_class = '0;
        for (int slot = 32'd0; slot < LOOP_CLASS_SLOTS;
             slot = slot + 32'd1) begin
            if (!class_found && class_valid_q[slot] &&
                class_gate_q[slot] == token_gate_code) begin
                class_found = 1'b1;
                found_class = CLASS_ID_W'(slot);
            end
            if (!free_found && !class_valid_q[slot]) begin
                free_found = 1'b1;
                free_class = CLASS_ID_W'(slot);
            end
        end
    end

    // Select the smallest not-yet-analyzed gate code. This makes the directory
    // independent of first-arrival class allocation order.
    always_comb begin
        selected_class_found = 1'b0;
        selected_class = '0;
        selected_class_gate = {GATE_W{1'b1}};
        for (int slot = 32'd0; slot < LOOP_CLASS_SLOTS;
             slot = slot + 32'd1) begin
            if (class_valid_q[slot] && !analyzed_class_mask_q[slot] &&
                (!selected_class_found ||
                 class_gate_q[slot] < selected_class_gate)) begin
                selected_class_found = 1'b1;
                selected_class = CLASS_ID_W'(slot);
                selected_class_gate = class_gate_q[slot];
            end
        end
    end

    assign head_begin_ready = state_q == ST_IDLE;
    assign token_ready = state_q == ST_CAPTURE;
    assign token_fire = token_valid && token_ready;
    assign token_active = |token_k_bits;

    assign metadata_valid = state_q == ST_METADATA;
    assign metadata_context_id = context_q;
    assign metadata_head_id = head_q;
    assign metadata_tag = tag_q;
    assign metadata_active_classes = class_count_q;
    assign metadata_active_tokens = active_token_count_q;
    assign metadata_term_count = term_count_q;
    assign metadata_event_count = event_count_q;
    assign metadata_bitmap_term_count = bitmap_term_count_q;
    assign metadata_fadc_destination_bytes = fadc_destination_bytes_q;
    assign metadata_overflow = metadata_overflow_q;
    assign raw_capture_error = raw_capture_error_q;

    assign emit_start_ready = state_q == ST_READY;
    assign descriptor_valid = state_q == ST_DESCRIPTORS;
    assign descriptor_gate_code = term_q[emit_term_index_q][GATE_W-1:0];
    assign descriptor_lane_id =
        term_q[emit_term_index_q][GATE_W +: 5];
    assign descriptor_destination_count =
        term_q[emit_term_index_q][GATE_W+5 +: 8];
    assign descriptor_last =
        32'(emit_term_index_q) + 1 == 32'(term_count_q);
    assign descriptor_fire = descriptor_valid && descriptor_ready;

    assign emit_class = term_q[emit_term_index_q][GATE_W+5+8 +: CLASS_ID_W];
    assign emit_lane = LANE_ID_W'(
        term_q[emit_term_index_q][GATE_W +: 5]);
    assign emit_fanout = term_q[emit_term_index_q][GATE_W+5 +: 8];
    assign destination_bitmap_bypass =
        BITMAP_BYPASS_ENABLE != 32'd0 && emit_format_q == FORMAT_FADC24 &&
        32'(emit_fanout) > 32'd21;
    assign linear_destination_bit =
        destination_remaining_q[destination_scan_index_q];
    assign destination_valid = state_q == ST_DESTINATIONS &&
        !destination_bitmap_bypass &&
        ((DESTINATION_SCAN_MODE == 0) ? linear_destination_bit :
                                       segmented_destination_found);
    assign destination_token_id = (DESTINATION_SCAN_MODE == 0) ?
        8'(destination_scan_index_q) : segmented_destination_token;
    assign destination_last_for_term =
        32'(destination_seen_q) + 1 == 32'(emit_fanout);
    assign destination_fire = destination_valid && destination_ready;
    assign destination_bitmap_valid = state_q == ST_DESTINATIONS &&
        destination_bitmap_bypass;
    assign destination_bitmap = destination_remaining_q;
    assign destination_bitmap_fire =
        destination_bitmap_valid && destination_bitmap_ready;
    assign next_emit_term_index = emit_term_index_q + 1'b1;
    assign next_emit_class =
        term_q[next_emit_term_index][GATE_W+5+8 +: CLASS_ID_W];
    assign next_emit_lane = LANE_ID_W'(
        term_q[next_emit_term_index][GATE_W +: 5]);

    assign raw_token_valid = state_q == ST_RAW;
    assign raw_token_id = 8'(raw_emit_index_q);
    assign raw_k_bits = raw_record_q[raw_emit_index_q][LANES-1:0];
    assign raw_gate_code =
        raw_record_q[raw_emit_index_q][LANES +: GATE_W];
    assign raw_fire = raw_token_valid && raw_token_ready;

    assign emit_done_valid = state_q == ST_DONE;
    assign emit_done_tag = tag_q;
    assign emit_done_error = emit_done_error_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            context_q <= '0;
            head_q <= '0;
            tag_q <= '0;
            emit_format_q <= FORMAT_RAW;
            capture_index_q <= '0;
            class_count_q <= '0;
            active_token_count_q <= '0;
            event_count_q <= '0;
            metadata_overflow_q <= 1'b0;
            raw_capture_error_q <= 1'b0;
            class_valid_q <= '0;
            term_count_q <= '0;
            bitmap_term_count_q <= '0;
            fadc_destination_bytes_q <= '0;
            analyzed_class_mask_q <= '0;
            analyze_class_q <= '0;
            analyze_lane_q <= '0;
            analyzed_class_count_q <= '0;
            emit_term_index_q <= '0;
            destination_scan_index_q <= '0;
            destination_remaining_q <= '0;
            destination_seen_q <= '0;
            raw_emit_index_q <= '0;
            emit_done_error_q <= 1'b0;
            protocol_error <= 1'b0;
            count_heads <= '0;
            count_raw_fallback_heads <= '0;
            count_emitted_terms <= '0;
            count_emitted_destinations <= '0;
            count_destination_scan_cycles <= '0;
            count_output_stall_cycles <= '0;
            for (int slot = 32'd0; slot < LOOP_CLASS_SLOTS;
                 slot = slot + 32'd1)
                class_gate_q[slot] <= '0;
        end else begin
            if ((descriptor_valid && !descriptor_ready) ||
                (destination_valid && !destination_ready) ||
                (destination_bitmap_valid && !destination_bitmap_ready) ||
                (raw_token_valid && !raw_token_ready))
                count_output_stall_cycles <= count_output_stall_cycles + 1'b1;

            case (state_q)
                ST_IDLE: begin
                    if (head_begin_valid && head_begin_ready) begin
                        context_q <= head_context_id;
                        head_q <= head_id;
                        tag_q <= head_tag;
                        capture_index_q <= '0;
                        class_count_q <= '0;
                        active_token_count_q <= '0;
                        event_count_q <= '0;
                        metadata_overflow_q <= 1'b0;
                        raw_capture_error_q <= 1'b0;
                        class_valid_q <= '0;
                        term_count_q <= '0;
                        bitmap_term_count_q <= '0;
                        fadc_destination_bytes_q <= '0;
                        analyzed_class_mask_q <= '0;
                        analyzed_class_count_q <= '0;
                        emit_done_error_q <= 1'b0;
                        count_heads <= count_heads + 1'b1;
                        state_q <= ST_CAPTURE;
                    end
                end

                ST_CAPTURE: begin
                    if (token_fire) begin
                        raw_record_q[capture_index_q] <=
                            {token_gate_code, token_k_bits};
                        if (32'(token_id) != 32'(capture_index_q) ||
                            token_last !=
                                (32'(capture_index_q) + 1 == TOKENS)) begin
                            raw_capture_error_q <= 1'b1;
                            protocol_error <= 1'b1;
                        end
                        if (token_active) begin
                            active_token_count_q <= active_token_count_q + 1'b1;
                            event_count_q <=
                                event_count_q + 13'(token_popcount);
                            if (class_found) begin
                                for (int lane = 32'd0; lane < LOOP_LANES;
                                     lane = lane + 32'd1) begin
                                    if (token_k_bits[lane])
                                        fanout_q[found_class][lane] <=
                                            fanout_q[found_class][lane] + 1'b1;
                                end
                            end else if (free_found) begin
                                class_valid_q[free_class] <= 1'b1;
                                class_gate_q[free_class] <= token_gate_code;
                                class_count_q <= class_count_q + 1'b1;
                                for (int lane = 32'd0; lane < LOOP_LANES;
                                     lane = lane + 32'd1) begin
                                    fanout_q[free_class][lane] <=
                                        token_k_bits[lane] ? 8'd1 : 8'd0;
                                end
                            end else begin
                                metadata_overflow_q <= 1'b1;
                            end
                        end
                        if (32'(capture_index_q) + 1 == TOKENS) begin
                            analyzed_class_mask_q <= '0;
                            analyzed_class_count_q <= '0;
                            state_q <= ST_ANALYZE_SELECT;
                        end else begin
                            capture_index_q <= capture_index_q + 1'b1;
                        end
                    end
                end

                ST_ANALYZE_SELECT: begin
                    if (class_count_q == 0 ||
                        analyzed_class_count_q == class_count_q) begin
                        state_q <= ST_METADATA;
                    end else if (selected_class_found) begin
                        analyze_class_q <= selected_class;
                        analyze_lane_q <= '0;
                        state_q <= ST_ANALYZE_LANES;
                    end else begin
                        metadata_overflow_q <= 1'b1;
                        protocol_error <= 1'b1;
                        state_q <= ST_METADATA;
                    end
                end

                ST_ANALYZE_LANES: begin
                    if (fanout_q[analyze_class_q][analyze_lane_q] != 0) begin
                        term_q[TERM_INDEX_W'(term_count_q)] <= {
                            analyze_class_q,
                            fanout_q[analyze_class_q][analyze_lane_q],
                            5'(analyze_lane_q),
                            class_gate_q[analyze_class_q]
                        };
                        term_count_q <= term_count_q + 1'b1;
                        if (fanout_q[analyze_class_q][analyze_lane_q] > 21)
                            bitmap_term_count_q <=
                                bitmap_term_count_q + 1'b1;
                        fadc_destination_bytes_q <=
                            fadc_destination_bytes_q +
                            ((fanout_q[analyze_class_q][analyze_lane_q] > 21) ?
                             13'd21 :
                             13'(fanout_q[analyze_class_q][analyze_lane_q]));
                    end
                    if (32'(analyze_lane_q) == LANES - 1) begin
                        analyzed_class_mask_q[analyze_class_q] <= 1'b1;
                        analyzed_class_count_q <=
                            analyzed_class_count_q + 1'b1;
                        state_q <= ST_ANALYZE_SELECT;
                    end else begin
                        analyze_lane_q <= analyze_lane_q + 1'b1;
                    end
                end

                ST_METADATA: begin
                    if (metadata_valid && metadata_ready)
                        state_q <= ST_READY;
                end

                ST_READY: begin
                    if (emit_start_valid && emit_start_ready) begin
                        emit_format_q <= emit_start_format;
                        emit_term_index_q <= '0;
                        destination_scan_index_q <= '0;
                        destination_remaining_q <= '0;
                        destination_seen_q <= '0;
                        raw_emit_index_q <= '0;
                        if (raw_capture_error_q) begin
                            emit_done_error_q <= 1'b1;
                            state_q <= ST_DONE;
                        end else if (emit_start_format == FORMAT_RAW) begin
                            if (metadata_overflow_q)
                                count_raw_fallback_heads <=
                                    count_raw_fallback_heads + 1'b1;
                            state_q <= ST_RAW;
                        end else if (term_count_q == 0) begin
                            state_q <= ST_DONE;
                        end else begin
                            state_q <= ST_DESCRIPTORS;
                        end
                    end
                end

                ST_DESCRIPTORS: begin
                    if (descriptor_fire) begin
                        count_emitted_terms <= count_emitted_terms + 1'b1;
                        if (descriptor_last) begin
                            emit_term_index_q <= '0;
                            destination_scan_index_q <= '0;
                            destination_seen_q <= '0;
                            destination_remaining_q <= bitmap_read_data;
                            state_q <= ST_DESTINATIONS;
                        end else begin
                            emit_term_index_q <= emit_term_index_q + 1'b1;
                        end
                    end
                end

                ST_DESTINATIONS: begin
                    if (destination_bitmap_bypass) begin
                        if (destination_bitmap_fire) begin
                            count_destination_scan_cycles <=
                                count_destination_scan_cycles + 1'b1;
                            count_emitted_destinations <=
                                count_emitted_destinations +
                                COUNTER_W'(emit_fanout);
                            if (32'(emit_term_index_q) + 1 ==
                                32'(term_count_q)) begin
                                state_q <= ST_DONE;
                            end else begin
                                emit_term_index_q <= next_emit_term_index;
                                destination_seen_q <= '0;
                                destination_remaining_q <= bitmap_read_data;
                            end
                        end
                    end else if (DESTINATION_SCAN_MODE == 0) begin
                        if (!linear_destination_bit || destination_fire)
                            count_destination_scan_cycles <=
                                count_destination_scan_cycles + 1'b1;
                        if (linear_destination_bit) begin
                            if (destination_fire) begin
                                count_emitted_destinations <=
                                    count_emitted_destinations + 1'b1;
                                if (destination_last_for_term) begin
                                    if (32'(emit_term_index_q) + 1 ==
                                        32'(term_count_q)) begin
                                        state_q <= ST_DONE;
                                    end else begin
                                        emit_term_index_q <=
                                            emit_term_index_q + 1'b1;
                                        destination_scan_index_q <= '0;
                                        destination_seen_q <= '0;
                                        destination_remaining_q <=
                                            bitmap_read_data;
                                    end
                                end else if (
                                    32'(destination_scan_index_q) + 1 < TOKENS
                                ) begin
                                    destination_scan_index_q <=
                                        destination_scan_index_q + 1'b1;
                                    destination_seen_q <=
                                        destination_seen_q + 1'b1;
                                end else begin
                                    emit_done_error_q <= 1'b1;
                                    protocol_error <= 1'b1;
                                    state_q <= ST_DONE;
                                end
                            end
                        end else if (
                            32'(destination_scan_index_q) + 1 < TOKENS
                        ) begin
                            destination_scan_index_q <=
                                destination_scan_index_q + 1'b1;
                        end else begin
                            emit_done_error_q <= 1'b1;
                            protocol_error <= 1'b1;
                            state_q <= ST_DONE;
                        end
                    end else if (segmented_destination_found) begin
                        if (destination_fire) begin
                            count_destination_scan_cycles <=
                                count_destination_scan_cycles + 1'b1;
                            count_emitted_destinations <=
                                count_emitted_destinations + 1'b1;
                            destination_remaining_q[
                                TOKEN_ID_W'(segmented_destination_token)] <=
                                1'b0;
                            if (destination_last_for_term) begin
                                if (32'(emit_term_index_q) + 1 ==
                                    32'(term_count_q)) begin
                                    state_q <= ST_DONE;
                                end else begin
                                    emit_term_index_q <= next_emit_term_index;
                                    destination_seen_q <= '0;
                                    destination_remaining_q <= bitmap_read_data;
                                end
                            end else begin
                                destination_seen_q <= destination_seen_q + 1'b1;
                            end
                        end
                    end else begin
                        emit_done_error_q <= 1'b1;
                        protocol_error <= 1'b1;
                        state_q <= ST_DONE;
                    end
                end

                ST_RAW: begin
                    if (raw_fire) begin
                        if (32'(raw_emit_index_q) + 1 == TOKENS)
                            state_q <= ST_DONE;
                        else
                            raw_emit_index_q <= raw_emit_index_q + 1'b1;
                    end
                end

                ST_DONE: begin
                    if (emit_done_valid && emit_done_ready)
                        state_q <= ST_IDLE;
                end

                default: begin
                    protocol_error <= 1'b1;
                    emit_done_error_q <= 1'b1;
                    state_q <= ST_DONE;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
