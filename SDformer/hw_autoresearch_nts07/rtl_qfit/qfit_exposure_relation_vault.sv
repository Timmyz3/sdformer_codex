`timescale 1ns/1ps
`default_nettype none

module qfit_exposure_relation_vault #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int GATE_W = 9,
    parameter int MAX_HEADS = 24,
    parameter int DEPTH = 512,
    parameter int RECORD_W = 112,
    parameter int RELATION_BUILD_CYCLES = 450,
    parameter int SCAN_CYCLES = 15,
    parameter int SOURCE_ID_W =
        (HEIGHT * WIDTH * TIME_PLANES <= 1)
        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES),
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int HEAD_W = (MAX_HEADS <= 1) ? 1 : $clog2(MAX_HEADS),
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter int PTR_W = $clog2(DEPTH + 1)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,
    input  logic                       window_start,

    input  logic                       head_start,
    output logic                       head_ready,
    input  logic [HEAD_W-1:0]          head_index,

    input  logic                       in_valid,
    output logic                       in_ready,
    input  logic [SOURCE_ID_W-1:0]     in_source_id,
    input  logic [Y_W-1:0]             in_y,
    input  logic [X_W-1:0]             in_x,
    input  logic [HEAD_DIM-1:0]        in_k,
    input  logic [5*GATE_W-1:0]        in_gates,
    input  logic [4:0]                 in_valid_mask,
    input  logic                       in_last,

    output logic                       live_valid,
    input  logic                       live_ready,
    output logic [SOURCE_ID_W-1:0]     live_source_id,
    output logic [Y_W-1:0]             live_y,
    output logic [X_W-1:0]             live_x,
    output logic [HEAD_DIM-1:0]        live_k,
    output logic [5*GATE_W-1:0]        live_gates,
    output logic [4:0]                 live_valid_mask,
    output logic                       live_last,

    output logic                       head_done,
    output logic                       head_resident,
    output logic                       head_critical,
    output logic                       head_overflow,
    output logic [31:0]                head_service_cycles,
    output logic [PTR_W-1:0]           head_record_count,

    input  logic                       replay_start,
    output logic                       replay_cmd_ready,
    input  logic [HEAD_W-1:0]          replay_head_index,
    output logic                       replay_valid,
    input  logic                       replay_ready,
    output logic [SOURCE_ID_W-1:0]     replay_source_id,
    output logic [Y_W-1:0]             replay_y,
    output logic [X_W-1:0]             replay_x,
    output logic [HEAD_DIM-1:0]        replay_k,
    output logic [5*GATE_W-1:0]        replay_gates,
    output logic [4:0]                 replay_valid_mask,
    output logic                       replay_last,
    output logic                       replay_done,
    output logic                       replay_miss,

    output logic                       protocol_error,
    output logic [31:0]                perf_speculative_writes,
    output logic [31:0]                perf_discarded_writes,
    output logic [31:0]                perf_committed_records,
    output logic [31:0]                perf_replay_reads,
    output logic [31:0]                perf_capacity_misses
);
    localparam int TOTAL_SOURCES = HEIGHT * WIDTH * TIME_PLANES;
    localparam int DESC_COUNT_W = $clog2(TOTAL_SOURCES + 1);
    localparam int TERM_COUNT_W = 17;
    localparam int SOURCE_LSB = 0;
    localparam int K_LSB = SOURCE_LSB + SOURCE_ID_W;
    localparam int GATE_LSB = K_LSB + HEAD_DIM;
    localparam int MASK_LSB = GATE_LSB + 5 * GATE_W;
    localparam int PAYLOAD_W = MASK_LSB + 5;
    localparam int REPLAY_FIFO_DEPTH = 4;
    localparam int REPLAY_FIFO_PTR_W = $clog2(REPLAY_FIFO_DEPTH);
    localparam int REPLAY_FIFO_COUNT_W = $clog2(REPLAY_FIFO_DEPTH + 1);

    logic [PTR_W-1:0] directory_base_q [0:MAX_HEADS-1];
    logic [PTR_W-1:0] directory_length_q [0:MAX_HEADS-1];
    logic directory_resident_q [0:MAX_HEADS-1];

    logic head_active_q;
    logic [HEAD_W-1:0] current_head_q;
    logic [PTR_W-1:0] committed_ptr_q;
    logic [PTR_W-1:0] head_base_q;
    logic [PTR_W-1:0] speculative_ptr_q;
    logic head_overflow_q;
    logic [TERM_COUNT_W-1:0] term_count_q;
    logic [DESC_COUNT_W-1:0] descriptor_count_q;
    logic [31:0] current_speculative_writes_q;

    logic replay_active_q;
    logic [PTR_W-1:0] replay_issue_ptr_q;
    logic [PTR_W-1:0] replay_issue_remaining_q;
    logic [RECORD_W-1:0] replay_fifo_record_q [0:REPLAY_FIFO_DEPTH-1];
    logic replay_fifo_last_q [0:REPLAY_FIFO_DEPTH-1];
    logic [REPLAY_FIFO_PTR_W-1:0] replay_fifo_write_ptr_q;
    logic [REPLAY_FIFO_PTR_W-1:0] replay_fifo_read_ptr_q;
    logic [REPLAY_FIFO_COUNT_W-1:0] replay_fifo_count_q;
    logic read_tag_last_q [0:REPLAY_FIFO_DEPTH-1];
    logic [REPLAY_FIFO_PTR_W-1:0] read_tag_write_ptr_q;
    logic [REPLAY_FIFO_PTR_W-1:0] read_tag_read_ptr_q;
    logic [REPLAY_FIFO_COUNT_W-1:0] read_tag_count_q;

    logic mem_write_valid;
    logic [ADDR_W-1:0] mem_write_addr;
    logic [RECORD_W-1:0] mem_write_data;
    logic mem_read_valid;
    logic [ADDR_W-1:0] mem_read_addr;
    logic mem_read_data_valid;
    logic [RECORD_W-1:0] mem_read_data;

    logic input_fire;
    logic replay_pop;
    logic [REPLAY_FIFO_COUNT_W:0] replay_reserved_after_pop;
    logic active_record;
    logic [7:0] input_term_increment;
    logic [TERM_COUNT_W-1:0] term_count_after_input;
    logic [PTR_W-1:0] speculative_ptr_after_input;
    logic overflow_after_input;
    logic [RECORD_W-1:0] input_record;

    initial begin
        if (PAYLOAD_W > RECORD_W)
            $error("relation vault payload exceeds physical record width");
        if (DEPTH != 512 || RECORD_W != 112)
            $error("current relation vault contract requires 512x112 memory");
        if (TOTAL_SOURCES != 450 || HEAD_DIM != 32 || GATE_W != 9)
            $error("current relation vault contract requires T450/K32/gate9");
    end

    function automatic logic [5:0] popcount_k(
        input logic [HEAD_DIM-1:0] value
    );
        logic [5:0] count;
        count = '0;
        for (integer lane = 0; lane < HEAD_DIM; lane = lane + 1)
            count = count + 6'(value[lane]);
        popcount_k = count;
    endfunction

    function automatic logic [2:0] unique_nonzero_gates(
        input logic [5*GATE_W-1:0] gates,
        input logic [4:0] mask
    );
        logic [2:0] count;
        logic duplicate;
        logic [GATE_W-1:0] current_gate;
        logic [GATE_W-1:0] previous_gate;
        count = '0;
        for (integer role = 0; role < 5; role = role + 1) begin
            current_gate = gates[role*GATE_W +: GATE_W];
            duplicate = 1'b0;
            for (integer previous = 0; previous < role; previous = previous + 1) begin
                previous_gate = gates[previous*GATE_W +: GATE_W];
                if (
                    mask[previous]
                    && previous_gate != '0
                    && previous_gate == current_gate
                )
                    duplicate = 1'b1;
            end
            if (mask[role] && current_gate != '0 && !duplicate)
                count = count + 3'd1;
        end
        unique_nonzero_gates = count;
    endfunction

    function automatic logic [Y_W-1:0] source_y_from_id(
        input logic [SOURCE_ID_W-1:0] source_id
    );
        integer local_id;
        local_id = 32'(source_id) % (HEIGHT * WIDTH);
        source_y_from_id = Y_W'(local_id / WIDTH);
    endfunction

    function automatic logic [X_W-1:0] source_x_from_id(
        input logic [SOURCE_ID_W-1:0] source_id
    );
        integer local_id;
        local_id = 32'(source_id) % (HEIGHT * WIDTH);
        source_x_from_id = X_W'(local_id % WIDTH);
    endfunction

    always_comb begin
        logic [5:0] k_count;
        logic [2:0] gate_count;
        k_count = popcount_k(in_k);
        gate_count = unique_nonzero_gates(in_gates, in_valid_mask);
        input_term_increment = 8'(k_count * gate_count);
    end

    assign active_record = input_term_increment != 0;
    assign input_fire = in_valid && in_ready;
    assign live_valid = in_valid && head_active_q;
    assign in_ready = head_active_q && live_ready && !replay_active_q;
    assign live_source_id = in_source_id;
    assign live_y = in_y;
    assign live_x = in_x;
    assign live_k = in_k;
    assign live_gates = in_gates;
    assign live_valid_mask = in_valid_mask;
    assign live_last = in_last;

    always_comb begin
        input_record = '0;
        input_record[SOURCE_LSB +: SOURCE_ID_W] = in_source_id;
        input_record[K_LSB +: HEAD_DIM] = in_k;
        input_record[GATE_LSB +: 5*GATE_W] = in_gates;
        input_record[MASK_LSB +: 5] = in_valid_mask;
        term_count_after_input = term_count_q;
        speculative_ptr_after_input = speculative_ptr_q;
        overflow_after_input = head_overflow_q;
        if (input_fire) begin
            term_count_after_input =
                term_count_q + TERM_COUNT_W'(input_term_increment);
            if (active_record) begin
                if (!head_overflow_q && speculative_ptr_q < PTR_W'(DEPTH))
                    speculative_ptr_after_input = speculative_ptr_q + 1'b1;
                else
                    overflow_after_input = 1'b1;
            end
        end
    end

    assign mem_write_valid = input_fire
                          && active_record
                          && !head_overflow_q
                          && speculative_ptr_q < PTR_W'(DEPTH);
    assign mem_write_addr = speculative_ptr_q[ADDR_W-1:0];
    assign mem_write_data = input_record;

    assign replay_pop = replay_valid && replay_ready;
    always_comb begin
        replay_reserved_after_pop =
            (REPLAY_FIFO_COUNT_W + 1)'(replay_fifo_count_q)
            + (REPLAY_FIFO_COUNT_W + 1)'(read_tag_count_q);
        if (replay_pop)
            replay_reserved_after_pop = replay_reserved_after_pop - 1'b1;
    end
    assign mem_read_valid = replay_active_q
                         && replay_issue_remaining_q != 0
                         && replay_reserved_after_pop
                            < (REPLAY_FIFO_COUNT_W + 1)'(REPLAY_FIFO_DEPTH);
    assign mem_read_addr = replay_issue_ptr_q[ADDR_W-1:0];

    qfit_sync_relation_bank #(
        .DEPTH(DEPTH),
        .DATA_W(RECORD_W),
        .READ_LATENCY(1)
    ) u_vault_memory (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .write_valid(mem_write_valid),
        .write_addr(mem_write_addr),
        .write_data(mem_write_data),
        .read_valid(mem_read_valid),
        .read_addr(mem_read_addr),
        .read_data_valid(mem_read_data_valid),
        .read_data(mem_read_data)
    );

    assign head_ready = !head_active_q
                     && !replay_active_q
                     && replay_fifo_count_q == 0
                     && read_tag_count_q == 0;
    assign replay_cmd_ready = head_ready;
    assign replay_valid = replay_fifo_count_q != 0;
    assign replay_source_id =
        replay_fifo_record_q[replay_fifo_read_ptr_q]
            [SOURCE_LSB +: SOURCE_ID_W];
    assign replay_y = source_y_from_id(replay_source_id);
    assign replay_x = source_x_from_id(replay_source_id);
    assign replay_k = replay_fifo_record_q[replay_fifo_read_ptr_q]
        [K_LSB +: HEAD_DIM];
    assign replay_gates = replay_fifo_record_q[replay_fifo_read_ptr_q]
        [GATE_LSB +: 5*GATE_W];
    assign replay_valid_mask = replay_fifo_record_q[replay_fifo_read_ptr_q]
        [MASK_LSB +: 5];
    assign replay_last = replay_fifo_last_q[replay_fifo_read_ptr_q];

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            head_active_q <= 1'b0;
            current_head_q <= '0;
            committed_ptr_q <= '0;
            head_base_q <= '0;
            speculative_ptr_q <= '0;
            head_overflow_q <= 1'b0;
            term_count_q <= '0;
            descriptor_count_q <= '0;
            current_speculative_writes_q <= '0;
            replay_active_q <= 1'b0;
            replay_issue_ptr_q <= '0;
            replay_issue_remaining_q <= '0;
            replay_fifo_write_ptr_q <= '0;
            replay_fifo_read_ptr_q <= '0;
            replay_fifo_count_q <= '0;
            read_tag_write_ptr_q <= '0;
            read_tag_read_ptr_q <= '0;
            read_tag_count_q <= '0;
            head_done <= 1'b0;
            head_resident <= 1'b0;
            head_critical <= 1'b0;
            head_overflow <= 1'b0;
            head_service_cycles <= '0;
            head_record_count <= '0;
            replay_done <= 1'b0;
            replay_miss <= 1'b0;
            protocol_error <= 1'b0;
            perf_speculative_writes <= '0;
            perf_discarded_writes <= '0;
            perf_committed_records <= '0;
            perf_replay_reads <= '0;
            perf_capacity_misses <= '0;
            for (integer head = 0; head < MAX_HEADS; head = head + 1) begin
                directory_base_q[head] <= '0;
                directory_length_q[head] <= '0;
                directory_resident_q[head] <= 1'b0;
            end
        end else begin
            head_done <= 1'b0;
            replay_done <= 1'b0;
            replay_miss <= 1'b0;

            if (window_start) begin
                if (!head_ready)
                    protocol_error <= 1'b1;
                committed_ptr_q <= '0;
                for (integer head = 0; head < MAX_HEADS; head = head + 1) begin
                    directory_base_q[head] <= '0;
                    directory_length_q[head] <= '0;
                    directory_resident_q[head] <= 1'b0;
                end
            end

            if (head_start) begin
                if (!head_ready || head_index >= HEAD_W'(MAX_HEADS)) begin
                    protocol_error <= 1'b1;
                end else begin
                    head_active_q <= 1'b1;
                    current_head_q <= head_index;
                    head_base_q <= committed_ptr_q;
                    speculative_ptr_q <= committed_ptr_q;
                    head_overflow_q <= 1'b0;
                    term_count_q <= '0;
                    descriptor_count_q <= '0;
                    current_speculative_writes_q <= '0;
                end
            end

            if (in_valid && !head_active_q)
                protocol_error <= 1'b1;

            if (input_fire) begin
                term_count_q <= term_count_after_input;
                speculative_ptr_q <= speculative_ptr_after_input;
                head_overflow_q <= overflow_after_input;
                descriptor_count_q <= descriptor_count_q + 1'b1;
                if (mem_write_valid) begin
                    current_speculative_writes_q
                        <= current_speculative_writes_q + 32'd1;
                    perf_speculative_writes
                        <= perf_speculative_writes + 32'd1;
                end
                if (
                    descriptor_count_q >= DESC_COUNT_W'(TOTAL_SOURCES)
                    || (
                        in_last
                        && descriptor_count_q
                           != DESC_COUNT_W'(TOTAL_SOURCES - 1)
                    )
                )
                    protocol_error <= 1'b1;

                if (in_last) begin
                    logic critical;
                    logic resident;
                    logic [PTR_W-1:0] record_count;
                    critical =
                        32'(SCAN_CYCLES) + 32'(term_count_after_input)
                        < 32'(RELATION_BUILD_CYCLES);
                    resident = critical && !overflow_after_input;
                    record_count = speculative_ptr_after_input - head_base_q;
                    directory_base_q[current_head_q] <= head_base_q;
                    directory_length_q[current_head_q]
                        <= resident ? record_count : '0;
                    directory_resident_q[current_head_q] <= resident;
                    head_active_q <= 1'b0;
                    head_done <= 1'b1;
                    head_resident <= resident;
                    head_critical <= critical;
                    head_overflow <= overflow_after_input;
                    head_service_cycles
                        <= 32'(SCAN_CYCLES) + 32'(term_count_after_input);
                    head_record_count <= resident ? record_count : '0;
                    if (resident) begin
                        committed_ptr_q <= speculative_ptr_after_input;
                        perf_committed_records
                            <= perf_committed_records + 32'(record_count);
                    end else begin
                        speculative_ptr_q <= head_base_q;
                        perf_discarded_writes
                            <= perf_discarded_writes
                             + current_speculative_writes_q
                             + 32'(mem_write_valid);
                        if (overflow_after_input)
                            perf_capacity_misses
                                <= perf_capacity_misses + 32'd1;
                    end
                end
            end

            if (replay_pop) begin
                replay_fifo_read_ptr_q <= replay_fifo_read_ptr_q + 1'b1;
                if (replay_last) begin
                    replay_active_q <= 1'b0;
                    replay_done <= 1'b1;
                end
            end

            if (mem_read_data_valid) begin
                if (read_tag_count_q == 0) begin
                    protocol_error <= 1'b1;
                end else begin
                    replay_fifo_record_q[replay_fifo_write_ptr_q]
                        <= mem_read_data;
                    replay_fifo_last_q[replay_fifo_write_ptr_q]
                        <= read_tag_last_q[read_tag_read_ptr_q];
                    replay_fifo_write_ptr_q
                        <= replay_fifo_write_ptr_q + 1'b1;
                    read_tag_read_ptr_q <= read_tag_read_ptr_q + 1'b1;
                end
            end

            if (mem_read_valid) begin
                replay_issue_ptr_q <= replay_issue_ptr_q + 1'b1;
                replay_issue_remaining_q
                    <= replay_issue_remaining_q - 1'b1;
                read_tag_last_q[read_tag_write_ptr_q]
                    <= replay_issue_remaining_q == 1;
                read_tag_write_ptr_q <= read_tag_write_ptr_q + 1'b1;
                perf_replay_reads <= perf_replay_reads + 32'd1;
            end

            case ({mem_read_data_valid, replay_pop})
                2'b10: replay_fifo_count_q <= replay_fifo_count_q + 1'b1;
                2'b01: replay_fifo_count_q <= replay_fifo_count_q - 1'b1;
                default: replay_fifo_count_q <= replay_fifo_count_q;
            endcase

            case ({mem_read_valid, mem_read_data_valid})
                2'b10: read_tag_count_q <= read_tag_count_q + 1'b1;
                2'b01: read_tag_count_q <= read_tag_count_q - 1'b1;
                default: read_tag_count_q <= read_tag_count_q;
            endcase

            if (
                mem_read_data_valid
                && replay_fifo_count_q == REPLAY_FIFO_COUNT_W'(REPLAY_FIFO_DEPTH)
                && !replay_pop
            )
                protocol_error <= 1'b1;

            if (replay_start) begin
                if (
                    !replay_cmd_ready
                    || replay_head_index >= HEAD_W'(MAX_HEADS)
                ) begin
                    protocol_error <= 1'b1;
                end else if (!directory_resident_q[replay_head_index]) begin
                    replay_miss <= 1'b1;
                    replay_done <= 1'b1;
                end else if (directory_length_q[replay_head_index] == 0) begin
                    replay_done <= 1'b1;
                end else begin
                    replay_active_q <= 1'b1;
                    replay_issue_ptr_q
                        <= directory_base_q[replay_head_index];
                    replay_issue_remaining_q
                        <= directory_length_q[replay_head_index];
                    replay_fifo_write_ptr_q <= '0;
                    replay_fifo_read_ptr_q <= '0;
                    replay_fifo_count_q <= '0;
                    read_tag_write_ptr_q <= '0;
                    read_tag_read_ptr_q <= '0;
                    read_tag_count_q <= '0;
                end
            end
        end
    end
endmodule

`default_nettype wire
