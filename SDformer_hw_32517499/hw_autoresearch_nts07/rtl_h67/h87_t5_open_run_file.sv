`timescale 1ns/1ps
`default_nettype none

// D1/H87 crosses five existing T=2 windows without materializing a 10-slot
// score tensor.  Each spatial position keeps only its currently open run.
// Closed runs are emitted in raster arrival order with exact five-slot
// membership and active-K masks.  The T=5 boundary is between slots 4 and 5.
module h87_t5_open_run_file #(
    parameter int POSITIONS = 225,
    parameter int SCORE_W = 8,
    parameter int MAX_SCORE = 162,
    parameter int POSITION_W = (POSITIONS <= 1) ? 1 : $clog2(POSITIONS),
    parameter int GENERATION_W = 3
) (
    input  logic                    clk_core,
    input  logic                    rst_core,

    input  logic                    row_start,
    output logic                    row_start_ready,

    input  logic                    in_valid,
    output logic                    in_ready,
    input  logic [2:0]              in_window_index,
    input  logic [POSITION_W-1:0]   in_position_id,
    input  logic [SCORE_W-1:0]      in_score0_q7,
    input  logic [SCORE_W-1:0]      in_score1_q7,
    input  logic [1:0]              in_active_mask,

    output logic                    packet_valid,
    input  logic                    packet_ready,
    output logic [1:0]              packet_desc_count,

    output logic [POSITION_W-1:0]   packet_desc0_position,
    output logic                    packet_desc0_group,
    output logic [SCORE_W-1:0]      packet_desc0_score_q7,
    output logic [4:0]              packet_desc0_temporal_mask,
    output logic [4:0]              packet_desc0_active_mask,
    output logic                    packet_desc0_group_last,
    output logic                    packet_desc0_row_last,

    output logic [POSITION_W-1:0]   packet_desc1_position,
    output logic                    packet_desc1_group,
    output logic [SCORE_W-1:0]      packet_desc1_score_q7,
    output logic [4:0]              packet_desc1_temporal_mask,
    output logic [4:0]              packet_desc1_active_mask,
    output logic                    packet_desc1_group_last,
    output logic                    packet_desc1_row_last,

    output logic                    row_done,
    output logic                    protocol_error,
    output logic [31:0]             perf_input_pairs,
    output logic [31:0]             perf_original_slots,
    output logic [31:0]             perf_run_descriptors,
    output logic [31:0]             perf_equal_edges,
    output logic [31:0]             perf_spill_packets,
    output logic [31:0]             perf_input_stall_cycles
);
    typedef struct packed {
        logic [POSITION_W-1:0] position;
        logic                  group_id;
        logic [SCORE_W-1:0]    score_q7;
        logic [4:0]            temporal_mask;
        logic [4:0]            active_mask;
        logic                  group_last;
        logic                  row_last;
    } descriptor_t;

    function automatic descriptor_t make_descriptor(
        input logic [POSITION_W-1:0] position,
        input logic group_id,
        input logic [SCORE_W-1:0] score_q7,
        input logic [4:0] temporal_mask,
        input logic [4:0] active_mask,
        input logic group_last,
        input logic row_last
    );
        descriptor_t descriptor;
        begin
            descriptor.position = position;
            descriptor.group_id = group_id;
            descriptor.score_q7 = score_q7;
            descriptor.temporal_mask = temporal_mask;
            descriptor.active_mask = active_mask;
            descriptor.group_last = group_last;
            descriptor.row_last = row_last;
            return descriptor;
        end
    endfunction

    logic [SCORE_W-1:0] state_score_mem [0:POSITIONS-1];
    logic [4:0] state_temporal_mem [0:POSITIONS-1];
    logic [4:0] state_active_mem [0:POSITIONS-1];
    logic [GENERATION_W-1:0] state_generation_mem [0:POSITIONS-1];

    logic [GENERATION_W-1:0] generation_q;
    logic row_open_q;
    logic [2:0] next_window_q;
    logic [POSITION_W-1:0] next_position_q;

    logic packet_valid_q;
    logic [1:0] packet_count_q;
    descriptor_t packet_desc0_q;
    descriptor_t packet_desc1_q;
    logic spill_valid_q;
    descriptor_t spill_desc_q;

    logic protocol_error_q;
    logic [31:0] input_pairs_q;
    logic [31:0] original_slots_q;
    logic [31:0] run_descriptors_q;
    logic [31:0] equal_edges_q;
    logic [31:0] spill_packets_q;
    logic [31:0] input_stall_cycles_q;

    logic accept_w;
    logic sequence_legal_w;
    logic score_legal_w;
    logic state_legal_w;
    logic input_legal_w;
    logic state_present_w;
    logic position_legal_w;
    logic [POSITION_W-1:0] state_index_w;

    logic [SCORE_W-1:0] open_score_w;
    logic [4:0] open_temporal_w;
    logic [4:0] open_active_w;
    logic [SCORE_W-1:0] new_score_w [0:1];
    logic new_active_w [0:1];
    logic [2:0] new_slot_w [0:1];
    logic [1:0] new_count_w;
    logic finalize_w;
    logic group_id_w;
    logic state_write_w;
    logic [SCORE_W-1:0] state_write_score_w;
    logic [4:0] state_write_temporal_w;
    logic [4:0] state_write_active_w;
    logic [1:0] generated_count_w;
    logic [2:0] generated_equal_edges_w;
    descriptor_t generated_desc0_w;
    descriptor_t generated_desc1_w;
    descriptor_t generated_desc2_w;

    integer step_i;

    assign row_start_ready = !row_open_q && !packet_valid_q && !spill_valid_q;
    assign in_ready = row_open_q
                   && !spill_valid_q
                   && (!packet_valid_q || packet_ready);
    assign accept_w = in_valid && in_ready;

    assign sequence_legal_w = in_window_index == next_window_q
                           && in_position_id == next_position_q
                           && in_window_index <= 3'd4
                           && 32'(in_position_id) < 32'(POSITIONS);
    assign position_legal_w = 32'(in_position_id) < 32'(POSITIONS);
    assign state_index_w = position_legal_w ? in_position_id : '0;
    assign score_legal_w = in_score0_q7 <= SCORE_W'(MAX_SCORE)
                        && in_score1_q7 <= SCORE_W'(MAX_SCORE);
    assign state_present_w = state_generation_mem[state_index_w] == generation_q;
    assign state_legal_w = (in_window_index == 3'd0) || state_present_w;
    assign input_legal_w = sequence_legal_w && score_legal_w && state_legal_w;

    assign packet_valid = packet_valid_q;
    assign packet_desc_count = packet_count_q;
    assign packet_desc0_position = packet_desc0_q.position;
    assign packet_desc0_group = packet_desc0_q.group_id;
    assign packet_desc0_score_q7 = packet_desc0_q.score_q7;
    assign packet_desc0_temporal_mask = packet_desc0_q.temporal_mask;
    assign packet_desc0_active_mask = packet_desc0_q.active_mask;
    assign packet_desc0_group_last = packet_desc0_q.group_last;
    assign packet_desc0_row_last = packet_desc0_q.row_last;
    assign packet_desc1_position = packet_desc1_q.position;
    assign packet_desc1_group = packet_desc1_q.group_id;
    assign packet_desc1_score_q7 = packet_desc1_q.score_q7;
    assign packet_desc1_temporal_mask = packet_desc1_q.temporal_mask;
    assign packet_desc1_active_mask = packet_desc1_q.active_mask;
    assign packet_desc1_group_last = packet_desc1_q.group_last;
    assign packet_desc1_row_last = packet_desc1_q.row_last;
    assign row_done = packet_valid_q && packet_ready
                   && ((packet_count_q == 2'd1 && packet_desc0_q.row_last)
                       || (packet_count_q == 2'd2 && packet_desc1_q.row_last));
    assign protocol_error = protocol_error_q;
    assign perf_input_pairs = input_pairs_q;
    assign perf_original_slots = original_slots_q;
    assign perf_run_descriptors = run_descriptors_q;
    assign perf_equal_edges = equal_edges_q;
    assign perf_spill_packets = spill_packets_q;
    assign perf_input_stall_cycles = input_stall_cycles_q;

    // Build zero to three descriptors for one accepted T=2 pair.  A third
    // descriptor is possible only at window 4 when group 1 is finalized.
    always_comb begin
        open_score_w = state_score_mem[state_index_w];
        open_temporal_w = state_temporal_mem[state_index_w];
        open_active_w = state_active_mem[state_index_w];
        new_score_w[0] = '0;
        new_score_w[1] = '0;
        new_active_w[0] = 1'b0;
        new_active_w[1] = 1'b0;
        new_slot_w[0] = '0;
        new_slot_w[1] = '0;
        new_count_w = '0;
        finalize_w = 1'b0;
        group_id_w = in_window_index >= 3'd3;
        state_write_w = 1'b1;
        generated_count_w = '0;
        generated_equal_edges_w = '0;
        generated_desc0_w = '0;
        generated_desc1_w = '0;
        generated_desc2_w = '0;

        unique case (in_window_index)
            3'd0: begin
                open_score_w = in_score0_q7;
                open_temporal_w = 5'b00001;
                open_active_w = {4'b0000, in_active_mask[0]};
                new_score_w[0] = in_score1_q7;
                new_active_w[0] = in_active_mask[1];
                new_slot_w[0] = 3'd1;
                new_count_w = 2'd1;
                group_id_w = 1'b0;
            end
            3'd1: begin
                new_score_w[0] = in_score0_q7;
                new_score_w[1] = in_score1_q7;
                new_active_w[0] = in_active_mask[0];
                new_active_w[1] = in_active_mask[1];
                new_slot_w[0] = 3'd2;
                new_slot_w[1] = 3'd3;
                new_count_w = 2'd2;
                group_id_w = 1'b0;
            end
            3'd2: begin
                new_score_w[0] = in_score0_q7;
                new_active_w[0] = in_active_mask[0];
                new_slot_w[0] = 3'd4;
                new_count_w = 2'd1;
                finalize_w = 1'b1;
                group_id_w = 1'b0;
            end
            3'd3: begin
                new_score_w[0] = in_score0_q7;
                new_score_w[1] = in_score1_q7;
                new_active_w[0] = in_active_mask[0];
                new_active_w[1] = in_active_mask[1];
                new_slot_w[0] = 3'd1;
                new_slot_w[1] = 3'd2;
                new_count_w = 2'd2;
                group_id_w = 1'b1;
            end
            3'd4: begin
                new_score_w[0] = in_score0_q7;
                new_score_w[1] = in_score1_q7;
                new_active_w[0] = in_active_mask[0];
                new_active_w[1] = in_active_mask[1];
                new_slot_w[0] = 3'd3;
                new_slot_w[1] = 3'd4;
                new_count_w = 2'd2;
                finalize_w = 1'b1;
                group_id_w = 1'b1;
            end
            default: begin
                new_count_w = '0;
                state_write_w = 1'b0;
            end
        endcase

        for (step_i = 0; step_i < 2; step_i = step_i + 1) begin
            if (step_i < new_count_w) begin
                if (new_score_w[step_i] == open_score_w) begin
                    open_temporal_w[new_slot_w[step_i]] = 1'b1;
                    open_active_w[new_slot_w[step_i]] = new_active_w[step_i];
                    generated_equal_edges_w = generated_equal_edges_w + 1'b1;
                end else begin
                    unique case (generated_count_w)
                        2'd0: generated_desc0_w = make_descriptor(
                            in_position_id, group_id_w, open_score_w,
                            open_temporal_w, open_active_w, 1'b0, 1'b0);
                        2'd1: generated_desc1_w = make_descriptor(
                            in_position_id, group_id_w, open_score_w,
                            open_temporal_w, open_active_w, 1'b0, 1'b0);
                        default: generated_desc2_w = make_descriptor(
                            in_position_id, group_id_w, open_score_w,
                            open_temporal_w, open_active_w, 1'b0, 1'b0);
                    endcase
                    generated_count_w = generated_count_w + 1'b1;
                    open_score_w = new_score_w[step_i];
                    open_temporal_w = 5'b00001 << new_slot_w[step_i];
                    open_active_w = ({4'b0000, new_active_w[step_i]})
                                  << new_slot_w[step_i];
                end
            end
        end

        if (finalize_w) begin
            unique case (generated_count_w)
                2'd0: generated_desc0_w = make_descriptor(
                    in_position_id, group_id_w, open_score_w,
                    open_temporal_w, open_active_w, 1'b1,
                    in_window_index == 3'd4
                    && 32'(in_position_id) == 32'(POSITIONS - 1));
                2'd1: generated_desc1_w = make_descriptor(
                    in_position_id, group_id_w, open_score_w,
                    open_temporal_w, open_active_w, 1'b1,
                    in_window_index == 3'd4
                    && 32'(in_position_id) == 32'(POSITIONS - 1));
                default: generated_desc2_w = make_descriptor(
                    in_position_id, group_id_w, open_score_w,
                    open_temporal_w, open_active_w, 1'b1,
                    in_window_index == 3'd4
                    && 32'(in_position_id) == 32'(POSITIONS - 1));
            endcase
            generated_count_w = generated_count_w + 1'b1;
            if (in_window_index == 3'd2) begin
                // score1 belongs to slot 0 of the second T=5 group.  There is
                // deliberately no equality edge across the group boundary.
                state_write_w = 1'b1;
                state_write_score_w = in_score1_q7;
                state_write_temporal_w = 5'b00001;
                state_write_active_w = {4'b0000, in_active_mask[1]};
            end else begin
                state_write_w = 1'b0;
                state_write_score_w = '0;
                state_write_temporal_w = '0;
                state_write_active_w = '0;
            end
        end else begin
            state_write_score_w = open_score_w;
            state_write_temporal_w = open_temporal_w;
            state_write_active_w = open_active_w;
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            generation_q <= '0;
            row_open_q <= 1'b0;
            next_window_q <= '0;
            next_position_q <= '0;
            packet_valid_q <= 1'b0;
            packet_count_q <= '0;
            packet_desc0_q <= '0;
            packet_desc1_q <= '0;
            spill_valid_q <= 1'b0;
            spill_desc_q <= '0;
            protocol_error_q <= 1'b0;
            input_pairs_q <= '0;
            original_slots_q <= '0;
            run_descriptors_q <= '0;
            equal_edges_q <= '0;
            spill_packets_q <= '0;
            input_stall_cycles_q <= '0;
        end else begin
            if (row_start && !row_start_ready)
                protocol_error_q <= 1'b1;

            if (row_start && row_start_ready) begin
                generation_q <= generation_q + 1'b1;
                row_open_q <= 1'b1;
                next_window_q <= '0;
                next_position_q <= '0;
                packet_valid_q <= 1'b0;
                packet_count_q <= '0;
                spill_valid_q <= 1'b0;
                protocol_error_q <= 1'b0;
                input_pairs_q <= '0;
                original_slots_q <= '0;
                run_descriptors_q <= '0;
                equal_edges_q <= '0;
                spill_packets_q <= '0;
                input_stall_cycles_q <= '0;
            end else begin
                if (in_valid && !in_ready)
                    input_stall_cycles_q <= input_stall_cycles_q + 1'b1;

                if (packet_valid_q && packet_ready) begin
                    if (spill_valid_q) begin
                        packet_valid_q <= 1'b1;
                        packet_count_q <= 2'd1;
                        packet_desc0_q <= spill_desc_q;
                        packet_desc1_q <= '0;
                        spill_valid_q <= 1'b0;
                    end else begin
                        packet_valid_q <= 1'b0;
                        packet_count_q <= '0;
                    end
                end

                if (accept_w) begin
                    input_pairs_q <= input_pairs_q + 1'b1;
                    original_slots_q <= original_slots_q + 32'd2;
                    if (next_position_q == POSITION_W'(POSITIONS - 1)) begin
                        next_position_q <= '0;
                        if (next_window_q == 3'd4) begin
                            next_window_q <= '0;
                            row_open_q <= 1'b0;
                        end else begin
                            next_window_q <= next_window_q + 1'b1;
                        end
                    end else begin
                        next_position_q <= next_position_q + 1'b1;
                    end

                    if (!input_legal_w) begin
                        protocol_error_q <= 1'b1;
                        packet_valid_q <= 1'b0;
                        packet_count_q <= '0;
                        spill_valid_q <= 1'b0;
                    end else begin
                        run_descriptors_q <= run_descriptors_q
                                           + 32'(generated_count_w);
                        equal_edges_q <= equal_edges_q
                                       + 32'(generated_equal_edges_w);
                        if (state_write_w) begin
                            state_score_mem[state_index_w] <= state_write_score_w;
                            state_temporal_mem[state_index_w] <= state_write_temporal_w;
                            state_active_mem[state_index_w] <= state_write_active_w;
                            state_generation_mem[state_index_w] <= generation_q;
                        end

                        unique case (generated_count_w)
                            2'd0: begin
                                packet_valid_q <= 1'b0;
                                packet_count_q <= '0;
                                spill_valid_q <= 1'b0;
                            end
                            2'd1: begin
                                packet_valid_q <= 1'b1;
                                packet_count_q <= 2'd1;
                                packet_desc0_q <= generated_desc0_w;
                                packet_desc1_q <= '0;
                                spill_valid_q <= 1'b0;
                            end
                            2'd2: begin
                                packet_valid_q <= 1'b1;
                                packet_count_q <= 2'd2;
                                packet_desc0_q <= generated_desc0_w;
                                packet_desc1_q <= generated_desc1_w;
                                spill_valid_q <= 1'b0;
                            end
                            2'd3: begin
                                packet_valid_q <= 1'b1;
                                packet_count_q <= 2'd2;
                                packet_desc0_q <= generated_desc0_w;
                                packet_desc1_q <= generated_desc1_w;
                                spill_valid_q <= 1'b1;
                                spill_desc_q <= generated_desc2_w;
                                spill_packets_q <= spill_packets_q + 1'b1;
                            end
                            default: begin
                                packet_valid_q <= 1'b0;
                                packet_count_q <= '0;
                                spill_valid_q <= 1'b0;
                                protocol_error_q <= 1'b1;
                            end
                        endcase
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
