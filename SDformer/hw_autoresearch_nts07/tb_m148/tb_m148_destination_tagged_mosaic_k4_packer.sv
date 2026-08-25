`timescale 1ns/1ps
`default_nettype none

module tb_m148_destination_tagged_mosaic_k4_packer;
    localparam int MASK_BITS = 128;
    localparam int JOBS = 68;

    logic clk_core = 1'b0;
    logic rst_core;
    always #1.5 clk_core = ~clk_core;

    logic row_valid, row_ready, row_accept;
    logic [31:0] row_sequence;
    logic [8:0] row_id;
    logic [MASK_BITS-1:0] row_event_mask;
    logic descriptor_valid, descriptor_ready, descriptor_accept;
    logic [31:0] descriptor_sequence;
    logic [8:0] descriptor_row;
    logic [1:0] descriptor_count_m1;
    logic [2:0] descriptor_destination [0:3];
    logic [3:0] descriptor_source [0:3];
    logic [3:0] descriptor_tuple_valid;
    logic descriptor_last;
    logic done_valid;
    logic [31:0] done_sequence;
    logic [8:0] done_row;
    logic observed_active;
    logic [MASK_BITS-1:0] observed_remaining_mask;
    logic [7:0] observed_work_popcount;
    logic [31:0] observed_next_sequence;
    logic protocol_error, busy;

    logic force_descriptor_ready;
    logic attack_mode;
    int unsigned cycle_count;
    int unsigned accepted_rows, accepted_descriptors, emitted_tuples;
    int unsigned completed_rows, descriptor_stalls, protocol_attacks;
    int unsigned expected_descriptors_total, expected_events_total;
    int unsigned block_k4_descriptors_total, consecutive_descriptors;
    int unsigned prior_descriptor_cycle;
    bit saw_zero, saw_tail1, saw_tail2, saw_tail3, saw_full4;
    bit saw_cross_destination, saw_same_destination, saw_fallthrough;

    logic model_active;
    logic [127:0] model_mask;
    logic [31:0] model_sequence;
    logic [8:0] model_row;

    m148_destination_tagged_mosaic_k4_packer dut (.*);
    m148_destination_tagged_mosaic_k4_packer_assertions sva (.*);

    function automatic int unsigned popcount128(input logic [127:0] value);
        int unsigned count;
        count = 0;
        for (int bit_index = 0; bit_index < 128; bit_index++)
            count += value[bit_index];
        return count;
    endfunction

    function automatic int unsigned block_k4_count(
        input logic [127:0] value);
        int unsigned total;
        total = 0;
        for (int destination = 0; destination < 8; destination++) begin
            int unsigned per_destination;
            per_destination = 0;
            for (int source = 0; source < 16; source++)
                per_destination += value[destination * 16 + source];
            total += (per_destination + 3) / 4;
        end
        return total;
    endfunction

    function automatic logic [127:0] generated_mask(input int unsigned seed);
        logic [127:0] result;
        result = '0;
        for (int bit_index = 0; bit_index < 128; bit_index++) begin
            if (((seed * 37 + bit_index * 13 + bit_index / 7) % 17) < 4)
                result[bit_index] = 1'b1;
        end
        return result;
    endfunction

    task automatic clear_row;
        row_valid = 1'b0;
        row_sequence = '0;
        row_id = '0;
        row_event_mask = '0;
    endtask

    task automatic apply_reset;
        @(negedge clk_core);
        rst_core = 1'b1;
        clear_row();
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        repeat (2) @(posedge clk_core);
    endtask

    task automatic drive_row(
        input int unsigned sequence_id,
        input logic [127:0] event_mask
    );
        int unsigned events;
        events = popcount128(event_mask);
        expected_events_total += events;
        expected_descriptors_total += (events + 3) / 4;
        block_k4_descriptors_total += block_k4_count(event_mask);
        @(negedge clk_core);
        row_valid = 1'b1;
        row_sequence = sequence_id;
        row_id = sequence_id[8:0];
        row_event_mask = event_mask;
        do @(posedge clk_core); while (!row_accept);
        @(negedge clk_core);
        row_valid = 1'b0;
    endtask

    always_comb begin
        if (rst_core)
            descriptor_ready = 1'b0;
        else if (force_descriptor_ready)
            descriptor_ready = 1'b1;
        else
            descriptor_ready = (cycle_count % 7) != 2;
    end

    always @(posedge clk_core) begin : cycle_counter
        if (rst_core)
            cycle_count <= 0;
        else
            cycle_count <= cycle_count + 1;
    end

    always @(posedge clk_core) begin : exact_scoreboard
        logic [127:0] expected_mask;
        logic [127:0] expected_remainder;
        logic [31:0] expected_sequence;
        logic [8:0] expected_row;
        logic [3:0] expected_valid;
        logic [6:0] expected_index [0:3];
        logic source_was_active;

        if (rst_core) begin
            model_active = 1'b0;
            model_mask = '0;
            model_sequence = '0;
            model_row = '0;
        end else if (!attack_mode) begin
            source_was_active = model_active;
            expected_mask = model_active ? model_mask : row_event_mask;
            expected_sequence = model_active
                ? model_sequence : row_sequence;
            expected_row = model_active ? model_row : row_id;
            expected_remainder = expected_mask;
            expected_valid = '0;
            for (int pick = 0; pick < 4; pick++) begin
                bit found;
                found = 1'b0;
                expected_index[pick] = '0;
                for (int linear = 0; linear < 128; linear++) begin
                    if (!found && expected_remainder[linear]) begin
                        found = 1'b1;
                        expected_valid[pick] = 1'b1;
                        expected_index[pick] = linear[6:0];
                        expected_remainder[linear] = 1'b0;
                    end
                end
            end

            if (descriptor_valid) begin
                if (descriptor_sequence !== expected_sequence
                        || descriptor_row !== expected_row
                        || descriptor_tuple_valid !== expected_valid
                        || descriptor_last !== (expected_remainder == 0))
                    $fatal(1, "M148 descriptor header mismatch");
                for (int pick = 0; pick < 4; pick++) begin
                    if (descriptor_tuple_valid[pick]) begin
                        if ({descriptor_destination[pick],
                             descriptor_source[pick]}
                                !== expected_index[pick])
                            $fatal(1, "M148 canonical tuple mismatch");
                    end else if ({descriptor_destination[pick],
                                  descriptor_source[pick]} != 0) begin
                        $fatal(1, "M148 dirty descriptor padding");
                    end
                end
                if (!descriptor_ready)
                    descriptor_stalls <= descriptor_stalls + 1;
            end

            if (row_accept) begin
                accepted_rows <= accepted_rows + 1;
                if (row_event_mask == 0)
                    saw_zero <= 1'b1;
            end
            if (descriptor_accept) begin
                accepted_descriptors <= accepted_descriptors + 1;
                emitted_tuples <= emitted_tuples
                                + popcount128({124'd0,
                                               descriptor_tuple_valid});
                if (row_accept)
                    saw_fallthrough <= 1'b1;
                if (descriptor_tuple_valid == 4'b0001)
                    saw_tail1 <= 1'b1;
                if (descriptor_tuple_valid == 4'b0011)
                    saw_tail2 <= 1'b1;
                if (descriptor_tuple_valid == 4'b0111)
                    saw_tail3 <= 1'b1;
                if (descriptor_tuple_valid == 4'b1111)
                    saw_full4 <= 1'b1;
                if (descriptor_tuple_valid[1]
                        && descriptor_destination[0]
                           != descriptor_destination[1])
                    saw_cross_destination <= 1'b1;
                if (descriptor_tuple_valid == 4'b1111
                        && descriptor_destination[0]
                           == descriptor_destination[1]
                        && descriptor_destination[1]
                           == descriptor_destination[2]
                        && descriptor_destination[2]
                           == descriptor_destination[3])
                    saw_same_destination <= 1'b1;
                if (accepted_descriptors != 0
                        && cycle_count == prior_descriptor_cycle + 1)
                    consecutive_descriptors <= consecutive_descriptors + 1;
                prior_descriptor_cycle <= cycle_count;
                if (expected_remainder == 0) begin
                    model_active = 1'b0;
                    model_mask = '0;
                end else begin
                    model_active = 1'b1;
                    model_mask = expected_remainder;
                    if (!source_was_active) begin
                        model_sequence = row_sequence;
                        model_row = row_id;
                    end
                end
            end
            if (done_valid) begin
                if (descriptor_accept && descriptor_last) begin
                    if (done_sequence !== descriptor_sequence
                            || done_row !== descriptor_row)
                        $fatal(1, "M148 descriptor done identity mismatch");
                end else if (row_accept && row_event_mask == 0) begin
                    if (done_sequence !== row_sequence
                            || done_row !== row_id)
                        $fatal(1, "M148 zero-row done identity mismatch");
                end else begin
                    $fatal(1, "M148 spurious done");
                end
                completed_rows <= completed_rows + 1;
            end
        end
    end

    initial begin : stimulus
        logic [127:0] mask;
        int watchdog;

        rst_core = 1'b1;
        clear_row();
        force_descriptor_ready = 1'b0;
        attack_mode = 1'b0;
        cycle_count = 0;
        accepted_rows = 0;
        accepted_descriptors = 0;
        emitted_tuples = 0;
        completed_rows = 0;
        descriptor_stalls = 0;
        protocol_attacks = 0;
        expected_descriptors_total = 0;
        expected_events_total = 0;
        block_k4_descriptors_total = 0;
        consecutive_descriptors = 0;
        prior_descriptor_cycle = 0;
        saw_zero = 1'b0;
        saw_tail1 = 1'b0;
        saw_tail2 = 1'b0;
        saw_tail3 = 1'b0;
        saw_full4 = 1'b0;
        saw_cross_destination = 1'b0;
        saw_same_destination = 1'b0;
        saw_fallthrough = 1'b0;

        apply_reset();
        drive_row(0, '0);
        mask = '0; mask[3] = 1'b1;
        drive_row(1, mask);
        mask = '0; mask[2] = 1'b1; mask[17] = 1'b1;
        drive_row(2, mask);
        mask = '0; mask[1] = 1'b1; mask[35] = 1'b1; mask[100] = 1'b1;
        drive_row(3, mask);
        mask = '0;
        for (int source = 0; source < 16; source++)
            mask[3 * 16 + source] = 1'b1;
        drive_row(4, mask);
        mask = '0;
        for (int destination = 0; destination < 8; destination++) begin
            mask[destination * 16] = 1'b1;
            if ((destination % 2) == 0)
                mask[destination * 16 + 7] = 1'b1;
        end
        drive_row(5, mask);
        for (int sequence_id = 6; sequence_id < JOBS; sequence_id++)
            drive_row(sequence_id, generated_mask(sequence_id));

        watchdog = 0;
        while (completed_rows != JOBS && watchdog < 5000) begin
            @(posedge clk_core);
            watchdog++;
        end
        if (watchdog == 5000)
            $fatal(1, "M148 normal traffic failed to drain");
        repeat (4) @(posedge clk_core);
        if (protocol_error || busy || accepted_rows != JOBS
                || completed_rows != JOBS
                || accepted_descriptors != expected_descriptors_total
                || emitted_tuples != expected_events_total
                || block_k4_descriptors_total
                   <= expected_descriptors_total
                || descriptor_stalls == 0 || consecutive_descriptors == 0
                || !saw_zero || !saw_tail1 || !saw_tail2 || !saw_tail3
                || !saw_full4 || !saw_cross_destination
                || !saw_same_destination || !saw_fallthrough)
            $fatal(1, "M148 normal contract incomplete");

        attack_mode = 1'b1;
        force_descriptor_ready = 1'b1;
        apply_reset();
        @(negedge clk_core);
        row_valid = 1'b1;
        row_sequence = 1;
        row_id = 9'h155;
        row_event_mask = 128'h1;
        @(posedge clk_core);
        #0.1;
        if (!protocol_error || row_accept)
            $fatal(1, "M148 wrong initial sequence not quarantined");
        protocol_attacks++;

        apply_reset();
        @(negedge clk_core);
        row_valid = 1'b1;
        row_sequence = 0;
        row_id = 9'h077;
        row_event_mask = {128{1'b1}};
        @(posedge clk_core);
        if (!row_accept || !descriptor_accept)
            $fatal(1, "M148 active attack setup not accepted");
        @(negedge clk_core);
        row_sequence = 2;
        row_id = 9'h078;
        row_event_mask = 128'h3;
        repeat (2) @(posedge clk_core);
        #0.1;
        if (!protocol_error || row_accept || descriptor_accept)
            $fatal(1, "M148 active wrong sequence not quarantined");
        protocol_attacks++;

        $display("PASS M148 destination-tagged mosaic K4 packer VCS jobs=%0d rows=%0d events=%0d descriptors=%0d block_k4_descriptors=%0d descriptor_savings=%0d stalls=%0d ii1_pairs=%0d protocol_attacks=%0d stable_order=true exact_tuple_conservation=true zero_row_floor=true first_descriptor_fallthrough=true engine_arithmetic=false sram_macro=false physical_speedup=false system_speedup=false headline=false",
                 JOBS, accepted_rows, expected_events_total,
                 expected_descriptors_total, block_k4_descriptors_total,
                 block_k4_descriptors_total - expected_descriptors_total,
                 descriptor_stalls, consecutive_descriptors,
                 protocol_attacks);
        $finish;
    end

    initial begin : watchdog
        #200000;
        $fatal(1, "M148 watchdog timeout");
    end
endmodule

`default_nettype wire
