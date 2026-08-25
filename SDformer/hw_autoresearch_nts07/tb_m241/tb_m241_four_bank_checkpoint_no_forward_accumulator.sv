`timescale 1ns/1ps
`default_nettype none

module tb_m241_four_bank_checkpoint_no_forward_accumulator;
    localparam int LANES = 8;
    localparam int ROWS = 384;
    localparam int ACC_BITS = 19;
    localparam int ROW_BITS = 9;
    localparam int ACC_ADDR_BITS = 10;
    localparam int VECTOR_BITS = LANES * ACC_BITS;
    localparam int MAX_DESCRIPTORS = 126;

    logic clk_core;
    logic rst_core;
    logic context_open_valid;
    logic context_open_ready;
    logic [31:0] context_open_sequence;
    logic [1:0] context_open_operator;
    logic [8:0] context_open_partition;
    logic [15:0] context_open_weight_epoch;
    logic context_open_accept;
    logic descriptor_valid;
    logic descriptor_ready;
    logic [31:0] descriptor_sequence;
    logic [1:0] descriptor_operator;
    logic [8:0] descriptor_partition;
    logic [15:0] descriptor_weight_epoch;
    logic [15:0] descriptor_order;
    logic [ROW_BITS-1:0] descriptor_row;
    logic [3:0] descriptor_source;
    logic [3:0] descriptor_destination_valid;
    logic [2:0] descriptor_destination [0:3];
    logic [3:0] descriptor_negate;
    logic descriptor_last;
    logic descriptor_accept;
    logic context_close_valid;
    logic context_close_ready;
    logic context_close_accept;
    logic window_done;
    logic [3:0] weight_rd_en;
    logic [4:0] weight_rd_addr [0:3];
    logic signed [7:0] weight_rd_data [0:3][0:LANES-1];
    logic [3:0] weight_cache_hit;
    logic [3:0] weight_cache_miss;
    logic [3:0] acc_rd_en;
    logic [ACC_ADDR_BITS-1:0] acc_rd_addr [0:3];
    logic signed [ACC_BITS-1:0] acc_rd_data [0:3][0:LANES-1];
    logic [3:0] acc_wr_en;
    logic [ACC_ADDR_BITS-1:0] acc_wr_addr [0:3];
    logic signed [ACC_BITS-1:0] acc_wr_data [0:3][0:LANES-1];
    logic commit_valid;
    logic commit_ready;
    logic commit_accept;
    logic [15:0] commit_order;
    logic [ROW_BITS-1:0] commit_row;
    logic [3:0] commit_bank_valid;
    logic [2:0] commit_destination [0:3];
    logic commit_last;
    logic rmw_alias_stall;
    logic [15:0] next_descriptor_order;
    logic context_active;
    logic protocol_error;
    logic overflow_error;
    logic busy;

    logic [63:0] descriptor_memory [0:MAX_DESCRIPTORS-1];
    logic [31:0] meta_memory [0:15];
    logic [7:0] weight_flat [0:4*32*LANES-1];
    logic signed [ACC_BITS-1:0] acc_memory
        [0:3][0:2*ROWS-1][0:LANES-1];
    integer signed reference_memory
        [0:3][0:2*ROWS-1][0:LANES-1];

    typedef struct packed {
        logic [ACC_ADDR_BITS-1:0] address;
        logic [VECTOR_BITS-1:0] data;
    } expected_write_t;
    expected_write_t expected_write_q [0:3][$];

    string vector_dir;
    logic auto_commit_stall;
    logic force_commit_stall;
    int cycle_count;
    int accepted_descriptors;
    int accepted_groups;
    int exact_write_checks;
    int exact_lane_checks;
    int exact_mismatches;
    int weight_macro_reads;
    int weight_cache_hits;
    int accumulator_macro_reads;
    int alias_stall_cycles;
    int commit_stall_cycles;
    int full4_descriptors;
    int tail_descriptors;
    int negate_descriptors;
    int real_descriptors;
    int real_weight_macro_reads;
    int real_weight_cache_hits;
    int real_accumulator_macro_reads;
    int real_exact_write_checks;
    int real_exact_lane_checks;
    int protocol_attacks;
    int younger_fault_atomicity_checks;
    int reset_flush_checks;
    int overflow_attacks;
    int expected_overflows;
    int window_count;

    m241_four_bank_checkpoint_no_forward_accumulator #(
        .LANES(LANES), .ROWS(ROWS), .ACC_BITS(ACC_BITS)
    ) dut (.*);

    m241_four_bank_checkpoint_no_forward_accumulator_assertions #(
        .LANES(LANES), .ROWS(ROWS), .ACC_BITS(ACC_BITS)
    ) checks (.*);

    always #1.5 clk_core = ~clk_core;
    assign commit_ready = !force_commit_stall
                        && (!auto_commit_stall || cycle_count % 7 != 3);

    always @(posedge clk_core) begin : counters
        if (!rst_core) begin
            cycle_count <= cycle_count + 1;
            weight_macro_reads <= weight_macro_reads
                                + $countones(weight_rd_en);
            weight_cache_hits <= weight_cache_hits
                               + $countones(weight_cache_hit);
            accumulator_macro_reads <= accumulator_macro_reads
                                     + $countones(acc_rd_en);
            if (rmw_alias_stall)
                alias_stall_cycles <= alias_stall_cycles + 1;
            if (commit_valid && !commit_ready)
                commit_stall_cycles <= commit_stall_cycles + 1;
            if (context_open_accept)
                window_count <= window_count + 1;
        end
    end

    always @(posedge clk_core) begin : behavioral_synchronous_macros
        if (!rst_core) begin
            for (int bank = 0; bank < 4; bank++) begin
                if (weight_rd_en[bank]) begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        weight_rd_data[bank][lane] <= $signed(weight_flat[
                            (bank * 32 + weight_rd_addr[bank]) * LANES
                            + lane]);
                    end
                end
                if (acc_rd_en[bank]) begin
                    for (int lane = 0; lane < LANES; lane++)
                        acc_rd_data[bank][lane] <=
                            acc_memory[bank][acc_rd_addr[bank]][lane];
                end
                if (acc_wr_en[bank]) begin
                    for (int lane = 0; lane < LANES; lane++)
                        acc_memory[bank][acc_wr_addr[bank]][lane]
                            <= acc_wr_data[bank][lane];
                end
            end
        end
    end

    always @(posedge clk_core) begin : exact_integer_scoreboard
        expected_write_t expected;
        expected_write_t produced;
        int bank;
        int address;
        int weight_address;
        int signed weight_value;
        int signed contribution;
        int signed new_value;
        bit descriptor_overflow;
        if (rst_core) begin
            for (int bank_index = 0; bank_index < 4; bank_index++) begin
                expected_write_q[bank_index].delete();
                for (int address_index = 0; address_index < 2 * ROWS;
                        address_index++) begin
                    for (int lane = 0; lane < LANES; lane++)
                        reference_memory[bank_index][address_index][lane] = 0;
                end
            end
        end else begin
            for (int bank_index = 0; bank_index < 4; bank_index++) begin
                if (acc_wr_en[bank_index]) begin
                    if (expected_write_q[bank_index].size() == 0)
                        $fatal(1, "M241 unexpected accumulator write bank=%0d",
                               bank_index);
                    expected = expected_write_q[bank_index].pop_front();
                    if (acc_wr_addr[bank_index] !== expected.address) begin
                        exact_mismatches = exact_mismatches + 1;
                        $fatal(1, "M241 write address mismatch bank=%0d",
                               bank_index);
                    end
                    for (int lane = 0; lane < LANES; lane++) begin
                        if (acc_wr_data[bank_index][lane]
                                !== $signed(expected.data[
                                    lane * ACC_BITS +: ACC_BITS])) begin
                            exact_mismatches = exact_mismatches + 1;
                            $fatal(1,
                                "M241 integer miter mismatch bank=%0d lane=%0d expected=%0d actual=%0d",
                                bank_index, lane,
                                $signed(expected.data[
                                    lane * ACC_BITS +: ACC_BITS]),
                                $signed(acc_wr_data[bank_index][lane]));
                        end
                        exact_lane_checks = exact_lane_checks + 1;
                    end
                    exact_write_checks = exact_write_checks + 1;
                end
            end

            if (descriptor_accept) begin
                accepted_descriptors = accepted_descriptors + 1;
                accepted_groups = accepted_groups
                                + $countones(descriptor_destination_valid);
                if (descriptor_destination_valid == 4'b1111)
                    full4_descriptors = full4_descriptors + 1;
                else
                    tail_descriptors = tail_descriptors + 1;
                if (|descriptor_negate)
                    negate_descriptors = negate_descriptors + 1;

                descriptor_overflow = 1'b0;
                for (int tuple = 0; tuple < 4; tuple++) begin
                    if (descriptor_destination_valid[tuple]) begin
                        bank = descriptor_destination[tuple][1:0];
                        address = (descriptor_destination[tuple][2]
                                   ? ROWS : 0) + descriptor_row;
                        weight_address = {
                            descriptor_destination[tuple][2],
                            descriptor_source};
                        for (int lane = 0; lane < LANES; lane++) begin
                            weight_value = $signed(weight_flat[
                                (bank * 32 + weight_address) * LANES
                                + lane]);
                            contribution = descriptor_negate[tuple]
                                ? -weight_value : weight_value;
                            new_value = reference_memory[bank][address][lane]
                                      + contribution;
                            if (new_value < -(1 << (ACC_BITS - 1))
                                    || new_value
                                       > (1 << (ACC_BITS - 1)) - 1)
                                descriptor_overflow = 1'b1;
                        end
                    end
                end
                if (descriptor_overflow) begin
                    expected_overflows = expected_overflows + 1;
                end else begin
                    for (int tuple = 0; tuple < 4; tuple++) begin
                        if (descriptor_destination_valid[tuple]) begin
                            bank = descriptor_destination[tuple][1:0];
                            address = (descriptor_destination[tuple][2]
                                       ? ROWS : 0) + descriptor_row;
                            weight_address = {
                                descriptor_destination[tuple][2],
                                descriptor_source};
                            produced.address = address;
                            produced.data = '0;
                            for (int lane = 0; lane < LANES; lane++) begin
                                weight_value = $signed(weight_flat[
                                    (bank * 32 + weight_address) * LANES
                                    + lane]);
                                contribution = descriptor_negate[tuple]
                                    ? -weight_value : weight_value;
                                reference_memory[bank][address][lane]
                                    = reference_memory[bank][address][lane]
                                      + contribution;
                                produced.data[
                                    lane * ACC_BITS +: ACC_BITS]
                                    = reference_memory[bank][address][lane]
                                      [ACC_BITS-1:0];
                            end
                            expected_write_q[bank].push_back(produced);
                        end
                    end
                end
            end
        end
    end

    task automatic clear_drivers;
        begin
            context_open_valid = 1'b0;
            descriptor_valid = 1'b0;
            context_close_valid = 1'b0;
            force_commit_stall = 1'b0;
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            clear_drivers();
            auto_commit_stall = 1'b0;
            rst_core = 1'b1;
            repeat (3) @(negedge clk_core);
            rst_core = 1'b0;
            repeat (1) @(negedge clk_core);
        end
    endtask

    task automatic open_context(
        input logic [31:0] sequence_id,
        input logic [1:0] operator_id,
        input logic [8:0] partition_id,
        input logic [15:0] epoch_id);
        begin
            context_open_sequence = sequence_id;
            context_open_operator = operator_id;
            context_open_partition = partition_id;
            context_open_weight_epoch = epoch_id;
            context_open_valid = 1'b1;
            do @(posedge clk_core); while (!context_open_accept);
            @(negedge clk_core);
            context_open_valid = 1'b0;
        end
    endtask

    task automatic drive_descriptor(
        input logic [15:0] order_id,
        input logic [ROW_BITS-1:0] row_id,
        input logic [3:0] source_id,
        input logic [3:0] valid_mask,
        input logic [2:0] destination0,
        input logic [2:0] destination1,
        input logic [2:0] destination2,
        input logic [2:0] destination3,
        input logic [3:0] negate_mask,
        input logic last_flag,
        input logic [31:0] sequence_id,
        input logic [1:0] operator_id,
        input logic [8:0] partition_id,
        input logic [15:0] epoch_id);
        begin
            descriptor_sequence = sequence_id;
            descriptor_operator = operator_id;
            descriptor_partition = partition_id;
            descriptor_weight_epoch = epoch_id;
            descriptor_order = order_id;
            descriptor_row = row_id;
            descriptor_source = source_id;
            descriptor_destination_valid = valid_mask;
            descriptor_destination[0] = destination0;
            descriptor_destination[1] = destination1;
            descriptor_destination[2] = destination2;
            descriptor_destination[3] = destination3;
            descriptor_negate = negate_mask;
            descriptor_last = last_flag;
            descriptor_valid = 1'b1;
            do @(posedge clk_core); while (!descriptor_accept);
            @(negedge clk_core);
            descriptor_valid = 1'b0;
        end
    endtask

    task automatic drive_packed(input logic [63:0] packed_word);
        begin
            drive_descriptor(
                packed_word[15:0], packed_word[24:16],
                packed_word[28:25], packed_word[32:29],
                packed_word[35:33], packed_word[38:36],
                packed_word[41:39], packed_word[44:42],
                packed_word[48:45], packed_word[49],
                meta_memory[1], meta_memory[2][1:0], meta_memory[3][8:0],
                meta_memory[4][15:0]);
        end
    endtask

    task automatic wait_scoreboard_empty;
        begin
            wait (expected_write_q[0].size() == 0
                  && expected_write_q[1].size() == 0
                  && expected_write_q[2].size() == 0
                  && expected_write_q[3].size() == 0
                  && !commit_valid);
            @(negedge clk_core);
        end
    endtask

    task automatic close_context;
        begin
            wait_scoreboard_empty();
            context_close_valid = 1'b1;
            do @(posedge clk_core); while (!context_close_accept);
            @(negedge clk_core);
            context_close_valid = 1'b0;
        end
    endtask

    task automatic inject_invalid_younger(
        input int attack_kind,
        input int writes_expected);
        int writes_before;
        begin
            writes_before = exact_write_checks;
            descriptor_sequence = attack_kind == 0
                ? meta_memory[1] ^ 32'h0000_0100 : meta_memory[1];
            descriptor_operator = meta_memory[2][1:0];
            descriptor_partition = meta_memory[3][8:0];
            descriptor_weight_epoch = attack_kind == 2
                ? meta_memory[4][15:0] ^ 16'h0001
                : meta_memory[4][15:0];
            descriptor_order = attack_kind == 1 ? 16'd0 : 16'd1;
            descriptor_row = 9'd11;
            descriptor_source = 4'd3;
            descriptor_destination_valid = 4'b1111;
            descriptor_destination[0] = 3'd0;
            descriptor_destination[1] = 3'd1;
            descriptor_destination[2] = 3'd2;
            descriptor_destination[3] = 3'd3;
            descriptor_negate = 4'b0101;
            descriptor_last = 1'b0;
            descriptor_valid = 1'b1;
            #0.1;
            if (!protocol_error || descriptor_ready)
                $fatal(1, "M241 invalid younger request escaped kind=%0d",
                       attack_kind);
            @(posedge clk_core);
            @(negedge clk_core);
            descriptor_valid = 1'b0;
            wait_scoreboard_empty();
            if (exact_write_checks != writes_before + writes_expected)
                $fatal(1,
                    "M241 younger fault erased older token kind=%0d expected_writes=%0d actual_delta=%0d",
                    attack_kind, writes_expected,
                    exact_write_checks - writes_before);
            protocol_attacks = protocol_attacks + 1;
            younger_fault_atomicity_checks
                = younger_fault_atomicity_checks + 1;
        end
    endtask

    task automatic run_younger_fault(input int attack_kind);
        begin
            reset_dut();
            open_context(meta_memory[1], meta_memory[2][1:0],
                         meta_memory[3][8:0], meta_memory[4][15:0]);
            drive_descriptor(16'd0, 9'd10, 4'd2, 4'b1111,
                             3'd0, 3'd1, 3'd2, 3'd3, 4'b0010, 1'b0,
                             meta_memory[1], meta_memory[2][1:0],
                             meta_memory[3][8:0], meta_memory[4][15:0]);
            inject_invalid_younger(attack_kind, 4);
        end
    endtask

    initial begin : test
        int real_groups_before;
        int real_reads_before;
        int real_hits_before;
        int real_acc_reads_before;
        int real_writes_before;
        int real_lanes_before;
        int writes_before_reset;
        int best_abs;
        int best_bank;
        int best_address;
        int best_lane;
        int signed candidate;
        int overflow_iterations;
        logic [2:0] overflow_destination;
        logic overflow_negate;

        clk_core = 1'b0;
        rst_core = 1'b1;
        auto_commit_stall = 1'b0;
        force_commit_stall = 1'b0;
        cycle_count = 0;
        accepted_descriptors = 0;
        accepted_groups = 0;
        exact_write_checks = 0;
        exact_lane_checks = 0;
        exact_mismatches = 0;
        weight_macro_reads = 0;
        weight_cache_hits = 0;
        accumulator_macro_reads = 0;
        alias_stall_cycles = 0;
        commit_stall_cycles = 0;
        full4_descriptors = 0;
        tail_descriptors = 0;
        negate_descriptors = 0;
        real_descriptors = 0;
        real_weight_macro_reads = 0;
        real_weight_cache_hits = 0;
        real_accumulator_macro_reads = 0;
        real_exact_write_checks = 0;
        real_exact_lane_checks = 0;
        protocol_attacks = 0;
        younger_fault_atomicity_checks = 0;
        reset_flush_checks = 0;
        overflow_attacks = 0;
        expected_overflows = 0;
        window_count = 0;
        clear_drivers();
        context_open_sequence = '0;
        context_open_operator = '0;
        context_open_partition = '0;
        context_open_weight_epoch = '0;
        descriptor_sequence = '0;
        descriptor_operator = '0;
        descriptor_partition = '0;
        descriptor_weight_epoch = '0;
        descriptor_order = '0;
        descriptor_row = '0;
        descriptor_source = '0;
        descriptor_destination_valid = '0;
        descriptor_negate = '0;
        descriptor_last = 1'b0;
        for (int tuple = 0; tuple < 4; tuple++)
            descriptor_destination[tuple] = '0;
        for (int bank = 0; bank < 4; bank++) begin
            for (int lane = 0; lane < LANES; lane++) begin
                weight_rd_data[bank][lane] = '0;
                acc_rd_data[bank][lane] = '0;
            end
            for (int address = 0; address < 2 * ROWS; address++) begin
                for (int lane = 0; lane < LANES; lane++) begin
                    acc_memory[bank][address][lane]
                        = ((bank + 1) * 97 + address * 11 + lane * 5)
                          & ((1 << ACC_BITS) - 1);
                    reference_memory[bank][address][lane] = 0;
                end
            end
        end

        if (!$value$plusargs("VECTOR_DIR=%s", vector_dir))
            $fatal(1, "M241 VECTOR_DIR plusarg missing");
        $readmemh({vector_dir, "/descriptor.mem"}, descriptor_memory);
        $readmemh({vector_dir, "/weight.mem"}, weight_flat);
        $readmemh({vector_dir, "/meta.mem"}, meta_memory);
        if (meta_memory[0] != MAX_DESCRIPTORS
                || meta_memory[7] != LANES || meta_memory[8] != ROWS
                || meta_memory[10] == 0 || meta_memory[11] == 0
                || meta_memory[12] != 0 || meta_memory[13] == 0
                || meta_memory[14] == 0)
            $fatal(1, "M241 vector metadata drift");

        // Exact real ordered signed-INT8 checkpoint subset.
        reset_dut();
        open_context(meta_memory[1], meta_memory[2][1:0],
                     meta_memory[3][8:0], meta_memory[4][15:0]);
        auto_commit_stall = 1'b1;
        real_groups_before = accepted_groups;
        real_reads_before = weight_macro_reads;
        real_hits_before = weight_cache_hits;
        real_acc_reads_before = accumulator_macro_reads;
        real_writes_before = exact_write_checks;
        real_lanes_before = exact_lane_checks;
        for (int index = 0; index < MAX_DESCRIPTORS; index++) begin
            drive_packed(descriptor_memory[index]);
            real_descriptors = real_descriptors + 1;
        end
        close_context();
        auto_commit_stall = 1'b0;
        real_weight_macro_reads = weight_macro_reads - real_reads_before;
        real_weight_cache_hits = weight_cache_hits - real_hits_before;
        real_accumulator_macro_reads =
            accumulator_macro_reads - real_acc_reads_before;
        real_exact_write_checks = exact_write_checks - real_writes_before;
        real_exact_lane_checks = exact_lane_checks - real_lanes_before;
        if (accepted_groups - real_groups_before != 4 * MAX_DESCRIPTORS
                || real_weight_macro_reads != 56
                || real_weight_cache_hits != 448
                || real_exact_write_checks != 504
                || real_exact_lane_checks != 4032)
            $fatal(1, "M241 real subset cache accounting failed");

        // Directed prefix tails plus same-address interlock.  Values still
        // come from the same exact checkpoint partition.
        reset_dut();
        open_context(meta_memory[1], meta_memory[2][1:0],
                     meta_memory[3][8:0], meta_memory[4][15:0]);
        drive_descriptor(16'd0, 9'd5, 4'd1, 4'b0001,
                         3'd0, 3'd0, 3'd0, 3'd0, 4'b0001, 1'b0,
                         meta_memory[1], meta_memory[2][1:0],
                         meta_memory[3][8:0], meta_memory[4][15:0]);
        drive_descriptor(16'd1, 9'd5, 4'd1, 4'b0001,
                         3'd0, 3'd0, 3'd0, 3'd0, 4'b0000, 1'b0,
                         meta_memory[1], meta_memory[2][1:0],
                         meta_memory[3][8:0], meta_memory[4][15:0]);
        drive_descriptor(16'd2, 9'd6, 4'd2, 4'b0011,
                         3'd0, 3'd1, 3'd0, 3'd0, 4'b0001, 1'b0,
                         meta_memory[1], meta_memory[2][1:0],
                         meta_memory[3][8:0], meta_memory[4][15:0]);
        drive_descriptor(16'd3, 9'd7, 4'd3, 4'b0111,
                         3'd4, 3'd5, 3'd6, 3'd0, 4'b0010, 1'b1,
                         meta_memory[1], meta_memory[2][1:0],
                         meta_memory[3][8:0], meta_memory[4][15:0]);
        close_context();
        if (alias_stall_cycles == 0 || tail_descriptors < 4)
            $fatal(1, "M241 tail/alias coverage missing");

        // Three independently reset fail-closed attacks: stale sequence,
        // replayed order and wrong checkpoint/cache epoch.  Each is younger
        // than one accepted full4 token, which must still commit exactly once.
        run_younger_fault(0);
        run_younger_fault(1);
        run_younger_fault(2);

        // Reset flushes an accepted but uncommitted token; stale macro content
        // remains and is ignored by the next lazy-valid context.
        reset_dut();
        open_context(meta_memory[1], meta_memory[2][1:0],
                     meta_memory[3][8:0], meta_memory[4][15:0]);
        force_commit_stall = 1'b1;
        drive_descriptor(16'd0, 9'd21, 4'd4, 4'b1111,
                         3'd0, 3'd1, 3'd2, 3'd3, 4'b0000, 1'b0,
                         meta_memory[1], meta_memory[2][1:0],
                         meta_memory[3][8:0], meta_memory[4][15:0]);
        wait (commit_valid);
        writes_before_reset = exact_write_checks;
        reset_dut();
        repeat (5) @(posedge clk_core);
        if (exact_write_checks != writes_before_reset)
            $fatal(1, "M241 reset failed to flush uncommitted token");
        reset_flush_checks = reset_flush_checks + 1;

        // Retain the runtime overflow tree.  Repeated exact checkpoint INT8
        // weights target one address with no younger token in flight.  The
        // first out-of-range update atomically suppresses all bank writes.
        best_abs = 0;
        best_bank = 0;
        best_address = 0;
        best_lane = 0;
        for (int bank = 0; bank < 4; bank++) begin
            for (int address = 0; address < 32; address++) begin
                for (int lane = 0; lane < LANES; lane++) begin
                    candidate = $signed(weight_flat[
                        (bank * 32 + address) * LANES + lane]);
                    if (candidate < 0)
                        candidate = -candidate;
                    if (candidate > best_abs) begin
                        best_abs = candidate;
                        best_bank = bank;
                        best_address = address;
                        best_lane = lane;
                    end
                end
            end
        end
        if (best_abs < 64)
            $fatal(1, "M241 checkpoint overflow stimulus too weak");
        overflow_destination = best_bank + 4 * (best_address >> 4);
        overflow_negate = $signed(weight_flat[
            (best_bank * 32 + best_address) * LANES + best_lane]) < 0;
        reset_dut();
        open_context(meta_memory[1], meta_memory[2][1:0],
                     meta_memory[3][8:0], meta_memory[4][15:0]);
        overflow_iterations = 0;
        while (!overflow_error && overflow_iterations < 5000) begin
            drive_descriptor(overflow_iterations[15:0], 9'd31,
                             best_address[3:0], 4'b0001,
                             overflow_destination, 3'd0, 3'd0, 3'd0,
                             {3'b000, overflow_negate}, 1'b0,
                             meta_memory[1], meta_memory[2][1:0],
                             meta_memory[3][8:0], meta_memory[4][15:0]);
            repeat (4) @(posedge clk_core);
            overflow_iterations = overflow_iterations + 1;
        end
        if (!overflow_error || acc_wr_en != 4'b0000
                || expected_overflows != 1)
            $fatal(1,
                "M241 overflow guard failed iter=%0d expected=%0d wr=%b",
                overflow_iterations, expected_overflows, acc_wr_en);
        overflow_attacks = overflow_attacks + 1;

        if (real_descriptors != MAX_DESCRIPTORS
                || exact_mismatches != 0
                || protocol_attacks != 3
                || younger_fault_atomicity_checks != 3
                || reset_flush_checks != 1 || overflow_attacks != 1
                || commit_stall_cycles == 0 || alias_stall_cycles == 0
                || negate_descriptors == 0 || full4_descriptors == 0
                || tail_descriptors == 0 || exact_lane_checks == 0)
            $fatal(1,
                "M241 final coverage mismatch real=%0d mismatch=%0d attacks=%0d atomic=%0d reset=%0d overflow=%0d stalls=%0d alias=%0d neg=%0d full4=%0d tail=%0d lanes=%0d",
                real_descriptors, exact_mismatches, protocol_attacks,
                younger_fault_atomicity_checks, reset_flush_checks,
                overflow_attacks, commit_stall_cycles, alias_stall_cycles,
                negate_descriptors, full4_descriptors, tail_descriptors,
                exact_lane_checks);
        $display("PASS M241 checkpoint descriptors=%0d real_groups=%0d real_exact_writes=%0d real_exact_lanes=%0d real_weight_macro_reads=%0d real_cache_hits=%0d real_acc_macro_reads=%0d total_exact_write_checks=%0d total_exact_lane_checks=%0d mismatches=%0d commit_stalls=%0d alias_stalls=%0d directed_tail_descriptors=4 real_negative_descriptors=2 protocol_attacks=%0d younger_fault_atomicity=%0d reset_flush=%0d overflow_attacks=%0d overflow_iterations=%0d banks=4 lanes=%0d acc_bits=19 lazy_valid=true overflow_guard=true forwarding_payload_bits=0 dense_high_half_address=true real_full_trace=false m238_target_speedup=1.687018 physical_speedup=false system_speedup=false headline=false",
                 real_descriptors, 4 * real_descriptors,
                 real_exact_write_checks, real_exact_lane_checks,
                 real_weight_macro_reads, real_weight_cache_hits,
                 real_accumulator_macro_reads, exact_write_checks,
                 exact_lane_checks, exact_mismatches, commit_stall_cycles,
                 alias_stall_cycles,
                 protocol_attacks, younger_fault_atomicity_checks,
                 reset_flush_checks, overflow_attacks,
                 overflow_iterations, LANES);
        $finish;
    end

    initial begin
        #2000000;
        $fatal(1, "M241 watchdog timeout");
    end
endmodule

`default_nettype wire
