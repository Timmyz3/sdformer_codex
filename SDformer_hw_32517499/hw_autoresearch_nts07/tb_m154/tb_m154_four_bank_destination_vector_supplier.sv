`timescale 1ns/1ps
`default_nettype none

module tb_m154_four_bank_destination_vector_supplier;
    localparam int LANES = 96;
    localparam int SEQUENCE_BITS = 32;
    localparam int PARTITION_BITS = 9;

    logic clk_core;
    logic rst_core;
    logic context_open_valid;
    logic context_open_ready;
    logic [31:0] context_open_sequence;
    logic [1:0] context_open_operator;
    logic [8:0] context_open_partition;
    logic context_open_accept;
    logic descriptor_valid;
    logic descriptor_ready;
    logic [31:0] descriptor_sequence;
    logic [1:0] descriptor_operator;
    logic [8:0] descriptor_partition;
    logic [8:0] descriptor_row;
    logic [3:0] descriptor_source;
    logic [3:0] descriptor_destination_valid;
    logic [2:0] descriptor_destination [0:3];
    logic [3:0] descriptor_negate;
    logic descriptor_last;
    logic descriptor_accept;
    logic context_close_valid;
    logic context_close_ready;
    logic context_close_accept;
    logic [3:0] bank_rd_en;
    logic [4:0] bank_rd_addr [0:3];
    logic signed [7:0] bank_rd_data [0:3][0:LANES-1];
    logic result_valid;
    logic result_ready;
    logic [31:0] result_sequence;
    logic [1:0] result_operator;
    logic [8:0] result_partition;
    logic [8:0] result_row;
    logic [3:0] result_source;
    logic [3:0] result_destination_valid;
    logic [2:0] result_destination [0:3];
    logic [3:0] result_negate;
    logic result_last;
    logic signed [7:0] result_vector [0:3][0:LANES-1];
    logic result_accept;
    logic context_active;
    logic protocol_error;
    logic busy;

    logic force_result_stall;
    int cycle_count;
    int context_count;
    int descriptor_count;
    int result_count;
    int vector_group_count;
    int vector_lane_checks;
    int full4_count;
    int tail_count;
    int result_stall_count;
    int consecutive_accept_pairs;
    int protocol_attack_count;
    logic previous_descriptor_accept;

    typedef struct packed {
        logic [31:0] sequence_id;
        logic [1:0] operator_id;
        logic [8:0] partition;
        logic [8:0] row_id;
        logic [3:0] source;
        logic [3:0] destination_valid;
        logic [11:0] destination;
        logic [3:0] negate;
        logic last;
    } expected_t;
    expected_t expected_q[$];

    m154_four_bank_destination_vector_supplier dut (.*);
    m154_four_bank_destination_vector_supplier_assertions checks (.*);

    always #1.5 clk_core = ~clk_core;

    function automatic logic signed [7:0] frozen_weight(
        input int bank, input int address, input int lane);
        int signed value;
        begin
            if (bank == 0 && address == 0 && lane == 0)
                value = -128;
            else if (bank == 3 && address == 31 && lane == 95)
                value = 127;
            else begin
                value = (bank * 73 + address * 29 + lane * 11 + 17) % 255;
                value = value - 127;
            end
            frozen_weight = value;
        end
    endfunction

    // One-cycle synchronous behavioral SRAM response.  The arrays themselves
    // remain outside the DUT, matching the explicit macro cut.
    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            for (int bank = 0; bank < 4; bank++)
                for (int lane = 0; lane < LANES; lane++)
                    bank_rd_data[bank][lane] <= '0;
        end else begin
            for (int bank = 0; bank < 4; bank++) begin
                if (bank_rd_en[bank]) begin
                    for (int lane = 0; lane < LANES; lane++)
                        bank_rd_data[bank][lane]
                            <= frozen_weight(bank, bank_rd_addr[bank], lane);
                end
            end
        end
    end

    always @(negedge clk_core) begin
        if (rst_core)
            result_ready <= 1'b0;
        else if (force_result_stall)
            result_ready <= 1'b0;
        else
            result_ready <= (cycle_count % 7 != 2)
                         && (cycle_count % 11 != 5);
    end

    always @(posedge clk_core) begin : scoreboard
        expected_t expected;
        expected_t accepted;
        if (rst_core) begin
            cycle_count <= 0;
            previous_descriptor_accept <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (result_valid && !result_ready)
                result_stall_count <= result_stall_count + 1;
            if (result_accept) begin
                if (expected_q.size() == 0)
                    $fatal(1, "M154 unexpected result");
                expected = expected_q.pop_front();
                if (result_sequence !== expected.sequence_id
                        || result_operator !== expected.operator_id
                        || result_partition !== expected.partition
                        || result_row !== expected.row_id
                        || result_source !== expected.source
                        || result_destination_valid
                           !== expected.destination_valid
                        || result_negate !== expected.negate
                        || result_last !== expected.last)
                    $fatal(1, "M154 result metadata mismatch");
                vector_group_count <= vector_group_count
                    + $countones(expected.destination_valid);
                vector_lane_checks <= vector_lane_checks
                    + $countones(expected.destination_valid) * LANES;
                for (int tuple = 0; tuple < 4; tuple++) begin
                    if (result_destination[tuple]
                            !== expected.destination[tuple * 3 +: 3])
                        $fatal(1, "M154 destination order mismatch");
                    if (expected.destination_valid[tuple]) begin
                        int bank;
                        int address;
                        bank = expected.destination[tuple * 3 +: 2];
                        address = {
                            expected.destination[tuple * 3 + 2],
                            expected.source};
                        for (int lane = 0; lane < LANES; lane++) begin
                            if (result_vector[tuple][lane]
                                    !== frozen_weight(bank, address, lane))
                                $fatal(1,
                                    "M154 vector mismatch tuple=%0d lane=%0d",
                                    tuple, lane);
                        end
                    end else begin
                        for (int lane = 0; lane < LANES; lane++) begin
                            if (result_vector[tuple][lane] !== 8'sd0)
                                $fatal(1, "M154 invalid tuple was nonzero");
                        end
                    end
                end
                result_count <= result_count + 1;
            end
            if (descriptor_accept) begin
                accepted.sequence_id = descriptor_sequence;
                accepted.operator_id = descriptor_operator;
                accepted.partition = descriptor_partition;
                accepted.row_id = descriptor_row;
                accepted.source = descriptor_source;
                accepted.destination_valid = descriptor_destination_valid;
                for (int tuple = 0; tuple < 4; tuple++)
                    accepted.destination[tuple * 3 +: 3]
                        = descriptor_destination[tuple];
                accepted.negate = descriptor_negate;
                accepted.last = descriptor_last;
                expected_q.push_back(accepted);
                descriptor_count <= descriptor_count + 1;
                if (descriptor_destination_valid == 4'b1111)
                    full4_count <= full4_count + 1;
                else
                    tail_count <= tail_count + 1;
                if (previous_descriptor_accept)
                    consecutive_accept_pairs <= consecutive_accept_pairs + 1;
            end
            previous_descriptor_accept <= descriptor_accept;
        end
    end

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            context_open_valid = 1'b0;
            descriptor_valid = 1'b0;
            context_close_valid = 1'b0;
            force_result_stall = 1'b0;
            repeat (3) @(negedge clk_core);
            rst_core = 1'b0;
        end
    endtask

    task automatic open_context(
        input logic [31:0] sequence_id,
        input logic [1:0] operator_id,
        input logic [8:0] partition_id);
        begin
            @(negedge clk_core);
            context_open_sequence = sequence_id;
            context_open_operator = operator_id;
            context_open_partition = partition_id;
            context_open_valid = 1'b1;
            do @(posedge clk_core); while (!context_open_accept);
            @(negedge clk_core);
            context_open_valid = 1'b0;
            context_count = context_count + 1;
        end
    endtask

    task automatic send_descriptor(
        input logic [31:0] sequence_id,
        input logic [1:0] operator_id,
        input logic [8:0] partition_id,
        input logic [8:0] row_id,
        input logic [3:0] source_id,
        input logic [3:0] valid_mask,
        input int destination_base,
        input logic last_flag);
        begin
            descriptor_sequence = sequence_id;
            descriptor_operator = operator_id;
            descriptor_partition = partition_id;
            descriptor_row = row_id;
            descriptor_source = source_id;
            descriptor_destination_valid = valid_mask;
            descriptor_negate = {
                source_id[0], ~source_id[0], source_id[1], ~source_id[1]};
            descriptor_last = last_flag;
            for (int tuple = 0; tuple < 4; tuple++)
                descriptor_destination[tuple] = destination_base + tuple;
            descriptor_valid = 1'b1;
            do @(posedge clk_core); while (!descriptor_accept);
            @(negedge clk_core);
            descriptor_valid = 1'b0;
        end
    endtask

    task automatic close_context;
        begin
            wait (expected_q.size() == 0 && !result_valid && !descriptor_valid);
            @(negedge clk_core);
            context_close_valid = 1'b1;
            do @(posedge clk_core); while (!context_close_accept);
            @(negedge clk_core);
            context_close_valid = 1'b0;
        end
    endtask

    task automatic inject_illegal(
        input logic [3:0] valid_mask,
        input int destination0,
        input int destination1,
        input logic mismatch_context);
        begin
            @(negedge clk_core);
            descriptor_sequence = mismatch_context ? 32'hdeadc0de : 32'h1540_0001;
            descriptor_operator = 2'd2;
            descriptor_partition = 9'd173;
            descriptor_row = 9'd7;
            descriptor_source = 4'd3;
            descriptor_destination_valid = valid_mask;
            descriptor_destination[0] = destination0;
            descriptor_destination[1] = destination1;
            descriptor_destination[2] = 3'd2;
            descriptor_destination[3] = 3'd3;
            descriptor_negate = 4'b0101;
            descriptor_last = 1'b0;
            descriptor_valid = 1'b1;
            #0.1;
            if (!protocol_error || descriptor_ready)
                $fatal(1, "M154 illegal descriptor did not fail closed");
            @(posedge clk_core);
            @(negedge clk_core);
            descriptor_valid = 1'b0;
            protocol_attack_count = protocol_attack_count + 1;
        end
    endtask

    initial begin : test
        clk_core = 1'b0;
        rst_core = 1'b1;
        context_open_valid = 1'b0;
        context_open_sequence = '0;
        context_open_operator = '0;
        context_open_partition = '0;
        descriptor_valid = 1'b0;
        descriptor_sequence = '0;
        descriptor_operator = '0;
        descriptor_partition = '0;
        descriptor_row = '0;
        descriptor_source = '0;
        descriptor_destination_valid = '0;
        descriptor_negate = '0;
        descriptor_last = 1'b0;
        context_close_valid = 1'b0;
        result_ready = 1'b0;
        force_result_stall = 1'b0;
        context_count = 0;
        descriptor_count = 0;
        result_count = 0;
        vector_group_count = 0;
        vector_lane_checks = 0;
        full4_count = 0;
        tail_count = 0;
        result_stall_count = 0;
        consecutive_accept_pairs = 0;
        protocol_attack_count = 0;
        previous_descriptor_accept = 1'b0;
        for (int tuple = 0; tuple < 4; tuple++)
            descriptor_destination[tuple] = '0;

        reset_dut();
        open_context(32'h1540_0001, 2'd2, 9'd173);
        for (int item = 0; item < 24; item++) begin
            send_descriptor(32'h1540_0001, 2'd2, 9'd173,
                            item, item[3:0], 4'b1111, 0, 1'b0);
            send_descriptor(32'h1540_0001, 2'd2, 9'd173,
                            item, item[3:0], 4'b1111, 4, item == 23);
        end
        for (int count = 1; count <= 3; count++) begin
            send_descriptor(32'h1540_0001, 2'd2, 9'd173,
                            100 + count, count, (1 << count) - 1,
                            0, 1'b0);
            send_descriptor(32'h1540_0001, 2'd2, 9'd173,
                            110 + count, count + 4, (1 << count) - 1,
                            4, count == 3);
        end
        close_context();

        // A younger bank-conflict fault must not erase an accepted stalled
        // result.  Destinations 0 and 4 collide in physical bank zero.
        reset_dut();
        open_context(32'h1540_0001, 2'd2, 9'd173);
        force_result_stall = 1'b1;
        send_descriptor(32'h1540_0001, 2'd2, 9'd173,
                        9'd301, 4'd15, 4'b1111, 0, 1'b1);
        wait (result_valid);
        inject_illegal(4'b0011, 0, 4, 1'b0);
        repeat (2) @(posedge clk_core);
        if (!result_valid || !protocol_error)
            $fatal(1, "M154 pending result was lost on fault");
        force_result_stall = 1'b0;
        wait (expected_q.size() == 0);

        // Non-prefix valid mask and context mismatch are separately attacked.
        reset_dut();
        open_context(32'h1540_0001, 2'd2, 9'd173);
        inject_illegal(4'b0101, 0, 1, 1'b0);
        reset_dut();
        open_context(32'h1540_0001, 2'd2, 9'd173);
        inject_illegal(4'b0001, 0, 1, 1'b1);

        repeat (4) @(posedge clk_core);
        if (descriptor_count != 55 || result_count != 55
                || vector_group_count != 208
                || vector_lane_checks != 19968
                || full4_count != 49 || tail_count != 6
                || protocol_attack_count != 3
                || result_stall_count == 0
                || consecutive_accept_pairs == 0)
            $fatal(1,
                "M154 coverage/count mismatch d=%0d r=%0d g=%0d l=%0d f=%0d t=%0d a=%0d stalls=%0d ii1=%0d",
                descriptor_count, result_count, vector_group_count,
                vector_lane_checks, full4_count, tail_count,
                protocol_attack_count, result_stall_count,
                consecutive_accept_pairs);
        $display("PASS M154 four-bank destination-vector supplier VCS contexts=%0d descriptors=%0d results=%0d vector_groups=%0d vector_lane_checks=%0d full4=%0d tails=%0d result_stalls=%0d ii1_pairs=%0d protocol_attacks=%0d banks=4 vectors_per_full_descriptor=4 lanes=96 weight_bits=8 resident_partition_bits=98304 synchronous_read_latency=1 macro_external=true single_vector_multicast=false accumulator_commit=false physical_speedup=false system_speedup=false headline=false",
                 context_count, descriptor_count, result_count,
                 vector_group_count, vector_lane_checks, full4_count,
                 tail_count, result_stall_count, consecutive_accept_pairs,
                 protocol_attack_count);
        $finish;
    end

    initial begin
        #200000;
        $fatal(1, "M154 watchdog timeout");
    end
endmodule

`default_nettype wire
