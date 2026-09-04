`timescale 1ns/1ps
`default_nettype none

module tb_m149_destination_conflict_resolved_k4_combiner;
    localparam int LANES = 96;
    localparam int JOBS = 72;
    localparam int QUEUE_DEPTH = 128;

    logic clk_core = 1'b0;
    logic rst_core;
    always #1.5 clk_core = ~clk_core;

    logic descriptor_valid, descriptor_ready, descriptor_accept;
    logic [31:0] descriptor_sequence;
    logic [8:0] descriptor_row;
    logic descriptor_last;
    logic [3:0] descriptor_tuple_valid;
    logic [2:0] tuple_destination [0:3];
    logic tuple_negate [0:3];
    logic signed [7:0] tuple_vector [0:3][0:LANES-1];
    logic result_valid, result_ready, result_accept;
    logic [31:0] result_sequence;
    logic [8:0] result_row;
    logic result_last;
    logic [3:0] result_group_valid;
    logic [2:0] result_destination [0:3];
    logic signed [10:0] result_vector [0:3][0:LANES-1];
    logic protocol_error, busy;

    int unsigned cycle_count;
    int unsigned accepted_descriptors, accepted_results;
    int unsigned input_tuples, output_groups, combined_tuples;
    int unsigned result_stalls, ii1_pairs, protocol_attacks;
    int unsigned prior_accept_cycle;
    bit saw_all_same, saw_two_two, saw_two_one_one, saw_distinct;
    bit saw_tail1, saw_tail2, saw_tail3;
    bit saw_pos512, saw_neg512, saw_pos508;
    logic force_result_ready;
    logic attack_mode;

    logic [31:0] expected_sequence [0:QUEUE_DEPTH-1];
    logic [8:0] expected_row [0:QUEUE_DEPTH-1];
    logic expected_last [0:QUEUE_DEPTH-1];
    logic [3:0] expected_group_valid [0:QUEUE_DEPTH-1];
    logic [2:0] expected_destination [0:QUEUE_DEPTH-1][0:3];
    logic signed [10:0] expected_vector
        [0:QUEUE_DEPTH-1][0:3][0:LANES-1];
    int unsigned queue_head, queue_tail;

    m149_destination_conflict_resolved_k4_combiner dut (.*);
    m149_destination_conflict_resolved_k4_combiner_assertions sva (.*);

    function automatic int unsigned count4(input logic [3:0] value);
        int unsigned count;
        count = 0;
        for (int bit_index = 0; bit_index < 4; bit_index++)
            count += value[bit_index];
        return count;
    endfunction

    function automatic logic signed [7:0] generated_value(
        input int unsigned seed,
        input int unsigned tuple,
        input int unsigned lane
    );
        int signed raw;
        raw = (seed * 53 + tuple * 71 + lane * 29 + lane / 3) % 256;
        raw -= 128;
        return raw[7:0];
    endfunction

    task automatic clear_descriptor;
        descriptor_valid = 1'b0;
        descriptor_sequence = '0;
        descriptor_row = '0;
        descriptor_last = 1'b0;
        descriptor_tuple_valid = 4'b0001;
        for (int tuple = 0; tuple < 4; tuple++) begin
            tuple_destination[tuple] = '0;
            tuple_negate[tuple] = 1'b0;
            for (int lane = 0; lane < LANES; lane++)
                tuple_vector[tuple][lane] = '0;
        end
    endtask

    task automatic apply_reset;
        @(negedge clk_core);
        rst_core = 1'b1;
        clear_descriptor();
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
    endtask

    task automatic send_descriptor(
        input int unsigned sequence_id,
        input logic [3:0] valid_shape,
        input logic [2:0] destination0,
        input logic [2:0] destination1,
        input logic [2:0] destination2,
        input logic [2:0] destination3,
        input logic [3:0] negate_mask,
        input int signed boundary_mode
    );
        descriptor_valid = 1'b1;
        descriptor_sequence = sequence_id;
        descriptor_row = sequence_id[8:0];
        descriptor_last = (sequence_id % 5) == 4;
        descriptor_tuple_valid = valid_shape;
        tuple_destination[0] = destination0;
        tuple_destination[1] = destination1;
        tuple_destination[2] = destination2;
        tuple_destination[3] = destination3;
        for (int tuple = 0; tuple < 4; tuple++) begin
            tuple_negate[tuple] = negate_mask[tuple];
            for (int lane = 0; lane < LANES; lane++) begin
                case (boundary_mode)
                    1: tuple_vector[tuple][lane] = -8'sd128;
                    2: tuple_vector[tuple][lane] = -8'sd128;
                    3: tuple_vector[tuple][lane] = 8'sd127;
                    default:
                        tuple_vector[tuple][lane]
                            = generated_value(sequence_id, tuple, lane);
                endcase
            end
        end
        do @(posedge clk_core); while (!descriptor_accept);
        @(negedge clk_core);
    endtask

    always_comb begin
        if (rst_core)
            result_ready = 1'b0;
        else if (force_result_ready)
            result_ready = 1'b1;
        else
            result_ready = (cycle_count % 9) != 3
                        && (cycle_count % 9) != 4;
    end

    always @(posedge clk_core) begin
        if (rst_core)
            cycle_count <= 0;
        else
            cycle_count <= cycle_count + 1;
    end

    always @(posedge clk_core) begin : exact_scoreboard
        if (rst_core) begin
            queue_head = 0;
            queue_tail = 0;
        end else if (!attack_mode) begin
            if (result_valid) begin
                if (queue_head == queue_tail)
                    $fatal(1, "M149 result without expected transaction");
                if (result_sequence !== expected_sequence[queue_head]
                        || result_row !== expected_row[queue_head]
                        || result_last !== expected_last[queue_head]
                        || result_group_valid
                           !== expected_group_valid[queue_head])
                    $fatal(1, "M149 result metadata mismatch");
                for (int group = 0; group < 4; group++) begin
                    if (result_destination[group]
                            !== expected_destination[queue_head][group])
                        $fatal(1, "M149 result destination mismatch");
                    for (int lane = 0; lane < LANES; lane++) begin
                        if (result_vector[group][lane]
                                !== expected_vector[queue_head][group][lane])
                            $fatal(1,
                                "M149 numeric mismatch group=%0d lane=%0d got=%0d expected=%0d",
                                group, lane, result_vector[group][lane],
                                expected_vector[queue_head][group][lane]);
                    end
                end
                if (!result_ready)
                    result_stalls <= result_stalls + 1;
            end
            if (result_accept) begin
                accepted_results <= accepted_results + 1;
                output_groups <= output_groups
                               + count4(result_group_valid);
                if (result_group_valid == 4'b0001)
                    saw_all_same <= 1'b1;
                if (count4(result_group_valid) == 2)
                    saw_two_two <= 1'b1;
                if (count4(result_group_valid) == 3)
                    saw_two_one_one <= 1'b1;
                if (result_group_valid == 4'b1111)
                    saw_distinct <= 1'b1;
                if (result_vector[0][0] == 11'sd512)
                    saw_pos512 <= 1'b1;
                if (result_vector[0][0] == -11'sd512)
                    saw_neg512 <= 1'b1;
                if (result_vector[0][0] == 11'sd508)
                    saw_pos508 <= 1'b1;
                queue_head = queue_head + 1;
            end

            if (descriptor_accept) begin
                accepted_descriptors <= accepted_descriptors + 1;
                input_tuples <= input_tuples
                              + count4(descriptor_tuple_valid);
                if (descriptor_tuple_valid == 4'b0001)
                    saw_tail1 <= 1'b1;
                if (descriptor_tuple_valid == 4'b0011)
                    saw_tail2 <= 1'b1;
                if (descriptor_tuple_valid == 4'b0111)
                    saw_tail3 <= 1'b1;
                if (accepted_descriptors != 0
                        && cycle_count == prior_accept_cycle + 1)
                    ii1_pairs <= ii1_pairs + 1;
                prior_accept_cycle <= cycle_count;

                expected_sequence[queue_tail] = descriptor_sequence;
                expected_row[queue_tail] = descriptor_row;
                expected_last[queue_tail] = descriptor_last;
                expected_group_valid[queue_tail] = '0;
                for (int owner = 0; owner < 4; owner++) begin
                    bit first_occurrence;
                    first_occurrence = descriptor_tuple_valid[owner];
                    for (int earlier = 0; earlier < owner; earlier++) begin
                        if (descriptor_tuple_valid[earlier]
                                && tuple_destination[earlier]
                                   == tuple_destination[owner])
                            first_occurrence = 1'b0;
                    end
                    expected_group_valid[queue_tail][owner]
                        = first_occurrence;
                    expected_destination[queue_tail][owner]
                        = tuple_destination[owner];
                    for (int lane = 0; lane < LANES; lane++) begin
                        int signed total;
                        total = 0;
                        if (first_occurrence) begin
                            for (int tuple = 0; tuple < 4; tuple++) begin
                                if (descriptor_tuple_valid[tuple]
                                        && tuple_destination[tuple]
                                           == tuple_destination[owner]) begin
                                    if (tuple_negate[tuple])
                                        total -= $signed(
                                            tuple_vector[tuple][lane]);
                                    else
                                        total += $signed(
                                            tuple_vector[tuple][lane]);
                                end
                            end
                        end
                        expected_vector[queue_tail][owner][lane]
                            = total[10:0];
                    end
                end
                combined_tuples <= combined_tuples
                    + count4(descriptor_tuple_valid)
                    - count4(expected_group_valid[queue_tail]);
                queue_tail = queue_tail + 1;
            end
        end
    end

    initial begin : stimulus
        int watchdog;
        logic [2:0] d0, d1, d2, d3;
        logic [3:0] shape;

        rst_core = 1'b1;
        clear_descriptor();
        force_result_ready = 1'b0;
        attack_mode = 1'b0;
        cycle_count = 0;
        accepted_descriptors = 0;
        accepted_results = 0;
        input_tuples = 0;
        output_groups = 0;
        combined_tuples = 0;
        result_stalls = 0;
        ii1_pairs = 0;
        protocol_attacks = 0;
        prior_accept_cycle = 0;
        saw_all_same = 1'b0;
        saw_two_two = 1'b0;
        saw_two_one_one = 1'b0;
        saw_distinct = 1'b0;
        saw_tail1 = 1'b0;
        saw_tail2 = 1'b0;
        saw_tail3 = 1'b0;
        saw_pos512 = 1'b0;
        saw_neg512 = 1'b0;
        saw_pos508 = 1'b0;

        apply_reset();
        send_descriptor(0, 4'b1111, 3, 3, 3, 3, 4'b1111, 1);
        send_descriptor(1, 4'b1111, 4, 4, 4, 4, 4'b0000, 2);
        send_descriptor(2, 4'b1111, 5, 5, 5, 5, 4'b0000, 3);
        send_descriptor(3, 4'b1111, 1, 1, 6, 6, 4'b0101, 0);
        send_descriptor(4, 4'b1111, 2, 2, 5, 7, 4'b1010, 0);
        send_descriptor(5, 4'b1111, 0, 3, 5, 7, 4'b0110, 0);
        send_descriptor(6, 4'b0001, 7, 0, 0, 0, 4'b0001, 0);
        send_descriptor(7, 4'b0011, 6, 6, 0, 0, 4'b0010, 0);
        send_descriptor(8, 4'b0111, 1, 2, 1, 0, 4'b0100, 0);

        for (int sequence_id = 9; sequence_id < JOBS; sequence_id++) begin
            case (sequence_id % 4)
                0: begin d0 = 1; d1 = 1; d2 = 1; d3 = 1; end
                1: begin d0 = 0; d1 = 0; d2 = 6; d3 = 6; end
                2: begin d0 = 2; d1 = 5; d2 = 2; d3 = 7; end
                default: begin d0 = 0; d1 = 2; d2 = 4; d3 = 6; end
            endcase
            case (sequence_id % 11)
                0: shape = 4'b0001;
                1: shape = 4'b0011;
                2: shape = 4'b0111;
                default: shape = 4'b1111;
            endcase
            send_descriptor(sequence_id, shape, d0, d1, d2, d3,
                            sequence_id[3:0], 0);
        end
        clear_descriptor();

        watchdog = 0;
        while ((accepted_results != JOBS || queue_head != queue_tail)
                && watchdog < 2000) begin
            @(posedge clk_core);
            watchdog++;
        end
        if (watchdog == 2000)
            $fatal(1, "M149 normal traffic failed to drain");
        repeat (3) @(posedge clk_core);
        if (protocol_error || busy
                || accepted_descriptors != JOBS
                || accepted_results != JOBS
                || input_tuples <= output_groups
                || combined_tuples != input_tuples - output_groups
                || result_stalls == 0 || ii1_pairs == 0
                || !saw_all_same || !saw_two_two || !saw_two_one_one
                || !saw_distinct || !saw_tail1 || !saw_tail2 || !saw_tail3
                || !saw_pos512 || !saw_neg512 || !saw_pos508)
            $fatal(1, "M149 normal contract incomplete");

        attack_mode = 1'b1;
        force_result_ready = 1'b1;
        apply_reset();
        descriptor_valid = 1'b1;
        descriptor_sequence = 32'hbad00001;
        descriptor_tuple_valid = 4'b0101;
        @(posedge clk_core);
        #0.1;
        if (!protocol_error || descriptor_accept || result_valid)
            $fatal(1, "M149 gapped tuple-valid attack not quarantined");
        protocol_attacks++;

        apply_reset();
        descriptor_valid = 1'b1;
        descriptor_sequence = 32'hbad00002;
        descriptor_tuple_valid = 4'b0000;
        @(posedge clk_core);
        #0.1;
        if (!protocol_error || descriptor_accept || result_valid)
            $fatal(1, "M149 empty descriptor attack not quarantined");
        protocol_attacks++;

        $display("PASS M149 destination-conflict-resolved K4 combiner VCS descriptors=%0d results=%0d input_tuples=%0d output_groups=%0d combined_tuples=%0d result_stalls=%0d ii1_pairs=%0d protocol_attacks=%0d lanes=96 signed_input_bits=8 negate_bits=9 pair_sum_bits=10 result_bits=11 numeric_range=-512_to_512 stable_first_occurrence=true all_vectors_preavailable_assumption=true weight_storage=false sram_macro=false accumulator_commit=false physical_speedup=false system_speedup=false headline=false",
                 JOBS, accepted_results, input_tuples, output_groups,
                 combined_tuples, result_stalls, ii1_pairs,
                 protocol_attacks);
        $finish;
    end

    initial begin : watchdog
        #300000;
        $fatal(1, "M149 watchdog timeout");
    end
endmodule

`default_nettype wire
