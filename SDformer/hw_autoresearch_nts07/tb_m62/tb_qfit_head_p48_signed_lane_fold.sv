`timescale 1ns/1ps
`default_nettype none

module tb_qfit_head_p48_signed_lane_fold;
    localparam int PIXELS = 48;
    localparam int OUTPUTS = 2;
    localparam int SLOTS = 8;
    localparam int ACC_W = 13;
    localparam int LANES = 96;

    logic clk_core, rst_core;
    logic command_valid, command_ready, command_accept;
    logic [47:0] command_tag;
    logic [LANES*ACC_W-1:0] command_seed_acc;
    logic command_zero_event_group;
    logic event_valid, event_ready, event_accept, event_last;
    logic [SLOTS-1:0] event_source_valid;
    logic [SLOTS*PIXELS-1:0] event_positive_mask;
    logic [SLOTS*PIXELS-1:0] event_negative_mask;
    logic [SLOTS*OUTPUTS*8-1:0] event_weight;
    logic output_valid, output_ready, output_accept;
    logic [47:0] output_tag;
    logic [LANES*ACC_W-1:0] output_acc;
    logic [15:0] output_source_issues, output_signed_events;
    logic protocol_error, busy;

    integer signed golden [0:LANES-1];
    integer expected_sources, expected_events;
    integer groups_checked, events_accepted, zero_groups;
    integer seed_state;

    qfit_head_p48_signed_lane_fold dut (.*);

    qfit_head_p48_signed_lane_fold_assertions m62_sva (.*);

    always #1.5 clk_core = ~clk_core;

    function automatic integer signed slot_weight(
        input integer group_id, input integer event_id,
        input integer slot, input integer output_index
    );
        integer value;
        begin
            value = ((group_id * 5 + event_id * 3 + slot * 7
                      + output_index * 11) % 15) - 7;
            if (value == 0) value = output_index ? -3 : 2;
            slot_weight = value;
        end
    endfunction

    task automatic drive_command(
        input integer group_id,
        input logic zero_group
    );
        begin
            command_seed_acc = '0;
            for (int pixel = 0; pixel < PIXELS; pixel++) begin
                golden[pixel*OUTPUTS] = 5 + ((group_id + pixel) % 3) - 1;
                golden[pixel*OUTPUTS+1] = 3 - ((group_id + pixel) % 3) + 1;
                command_seed_acc[(pixel*OUTPUTS)*ACC_W +: ACC_W]
                    = ACC_W'(golden[pixel*OUTPUTS]);
                command_seed_acc[(pixel*OUTPUTS+1)*ACC_W +: ACC_W]
                    = ACC_W'(golden[pixel*OUTPUTS+1]);
            end
            expected_sources = 0;
            expected_events = 0;
            @(negedge clk_core);
            command_tag = group_id;
            command_zero_event_group = zero_group;
            command_valid = 1'b1;
            do @(posedge clk_core); while (!command_accept);
            @(negedge clk_core);
            command_valid = 1'b0;
            command_zero_event_group = 1'b0;
        end
    endtask

    task automatic drive_event(
        input integer group_id,
        input integer event_id,
        input logic last_event
    );
        integer signed weight;
        integer selector;
        begin
            event_source_valid = '0;
            event_positive_mask = '0;
            event_negative_mask = '0;
            event_weight = '0;
            for (int slot = 0; slot < SLOTS; slot++) begin
                event_source_valid[slot]
                    = ((group_id + event_id + slot) % 5) != 0;
                for (int output_index = 0; output_index < OUTPUTS;
                        output_index++) begin
                    weight = slot_weight(group_id, event_id, slot,
                                         output_index);
                    event_weight[(slot*OUTPUTS+output_index)*8 +: 8]
                        = 8'(weight);
                end
                for (int pixel = 0; pixel < PIXELS; pixel++) begin
                    selector = (group_id * 13 + event_id * 7
                                + slot * 5 + pixel * 3) % 11;
                    if (event_source_valid[slot] && selector < 3)
                        event_positive_mask[slot*PIXELS+pixel] = 1'b1;
                    else if (event_source_valid[slot] && selector == 3)
                        event_negative_mask[slot*PIXELS+pixel] = 1'b1;
                end
                if (event_source_valid[slot]
                        && !(|event_positive_mask[slot*PIXELS +: PIXELS])
                        && !(|event_negative_mask[slot*PIXELS +: PIXELS]))
                    event_positive_mask[slot*PIXELS
                        + ((group_id + slot) % PIXELS)] = 1'b1;
            end
            if (!(|event_source_valid)) begin
                event_source_valid[0] = 1'b1;
                event_positive_mask[0] = 1'b1;
            end
            for (int slot = 0; slot < SLOTS; slot++) begin
                if (event_source_valid[slot]) expected_sources++;
                for (int pixel = 0; pixel < PIXELS; pixel++) begin
                    if (event_source_valid[slot]
                            && event_positive_mask[slot*PIXELS+pixel]) begin
                        expected_events++;
                        for (int output_index = 0;
                                output_index < OUTPUTS; output_index++)
                            golden[pixel*OUTPUTS+output_index]
                                += slot_weight(group_id, event_id, slot,
                                               output_index);
                    end else if (event_source_valid[slot]
                            && event_negative_mask[slot*PIXELS+pixel]) begin
                        expected_events++;
                        for (int output_index = 0;
                                output_index < OUTPUTS; output_index++)
                            golden[pixel*OUTPUTS+output_index]
                                -= slot_weight(group_id, event_id, slot,
                                               output_index);
                    end
                end
            end
            @(negedge clk_core);
            event_last = last_event;
            event_valid = 1'b1;
            do @(posedge clk_core); while (!event_accept);
            events_accepted++;
            @(negedge clk_core);
            event_valid = 1'b0;
            event_last = 1'b0;
        end
    endtask

    task automatic check_output(input integer group_id, input logic stall_it);
        logic [LANES*ACC_W-1:0] held_acc;
        logic [47:0] held_tag;
        begin
            output_ready = 1'b0;
            do @(posedge clk_core); while (!output_valid);
            #1;
            if (output_tag !== 48'(group_id)
                    || output_source_issues !== expected_sources[15:0]
                    || output_signed_events !== expected_events[15:0])
                $fatal(1, "M62 metadata mismatch group=%0d tag=%0d src=%0d/%0d events=%0d/%0d",
                       group_id, output_tag, output_source_issues,
                       expected_sources, output_signed_events, expected_events);
            for (int lane = 0; lane < LANES; lane++)
                if ($signed(output_acc[lane*ACC_W +: ACC_W]) !== golden[lane])
                    $fatal(1, "M62 accumulator mismatch group=%0d lane=%0d got=%0d expected=%0d",
                           group_id, lane,
                           $signed(output_acc[lane*ACC_W +: ACC_W]),
                           golden[lane]);
            if (stall_it) begin
                held_acc = output_acc;
                held_tag = output_tag;
                repeat (3) begin
                    @(posedge clk_core); #1;
                    if (!output_valid || output_acc !== held_acc
                            || output_tag !== held_tag)
                        $fatal(1, "M62 output changed under backpressure");
                end
            end
            @(negedge clk_core); output_ready = 1'b1;
            do @(posedge clk_core); while (!output_accept);
            groups_checked++;
            @(negedge clk_core);
        end
    endtask

    initial begin
        integer event_count;
        clk_core = 1'b0;
        rst_core = 1'b1;
        command_valid = 1'b0;
        command_tag = '0;
        command_seed_acc = '0;
        command_zero_event_group = 1'b0;
        event_valid = 1'b0;
        event_last = 1'b0;
        event_source_valid = '0;
        event_positive_mask = '0;
        event_negative_mask = '0;
        event_weight = '0;
        output_ready = 1'b1;
        groups_checked = 0;
        events_accepted = 0;
        zero_groups = 0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core); rst_core = 1'b0;

        for (int group_id = 0; group_id < 240; group_id++) begin
            logic zero_group;
            zero_group = (group_id % 17) == 0;
            drive_command(group_id, zero_group);
            if (zero_group) begin
                zero_groups++;
            end else begin
                event_count = 1 + (group_id % 6);
                for (int event_id = 0; event_id < event_count; event_id++)
                    drive_event(group_id, event_id,
                                event_id == event_count-1);
            end
            check_output(group_id, (group_id % 9) == 0);
            if (protocol_error)
                $fatal(1, "M62 legal traffic triggered protocol error");
        end

        // Directed cover: the random generator never lights all eight source
        // slots in one cycle because 8 consecutive integers always include
        // a multiple of 5.  Drive one full-8 legal group explicitly.
        begin
            integer slot;
            integer pixel;
            integer output_index;
            integer signed weight;
            drive_command(240, 1'b0);
            event_source_valid = {SLOTS{1'b1}};
            event_positive_mask = '0;
            event_negative_mask = '0;
            event_weight = '0;
            expected_sources = 0;
            expected_events = 0;
            for (slot = 0; slot < SLOTS; slot++) begin
                expected_sources++;
                for (output_index = 0; output_index < OUTPUTS;
                        output_index++) begin
                    weight = slot_weight(240, 0, slot, output_index);
                    event_weight[(slot*OUTPUTS+output_index)*8 +: 8]
                        = 8'(weight);
                end
                for (pixel = 0; pixel < PIXELS; pixel++) begin
                    if ((pixel + slot) % 4 == 0) begin
                        event_negative_mask[slot*PIXELS+pixel] = 1'b1;
                        expected_events++;
                        for (output_index = 0; output_index < OUTPUTS;
                                output_index++)
                            golden[pixel*OUTPUTS+output_index]
                                -= slot_weight(240, 0, slot, output_index);
                    end else begin
                        event_positive_mask[slot*PIXELS+pixel] = 1'b1;
                        expected_events++;
                        for (output_index = 0; output_index < OUTPUTS;
                                output_index++)
                            golden[pixel*OUTPUTS+output_index]
                                += slot_weight(240, 0, slot, output_index);
                    end
                end
            end
            @(negedge clk_core);
            event_last = 1'b1;
            event_valid = 1'b1;
            do @(posedge clk_core); while (!event_accept);
            events_accepted++;
            @(negedge clk_core);
            event_valid = 1'b0;
            event_last = 1'b0;
            check_output(240, 1'b0);
            if (protocol_error)
                $fatal(1, "M62 full-eight legal group triggered protocol error");
        end

        // One fail-closed protocol attack: an event cannot precede a command.
        @(negedge clk_core);
        event_source_valid = 8'h01;
        event_positive_mask = '0;
        event_positive_mask[0] = 1'b1;
        event_valid = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core); event_valid = 1'b0;
        @(posedge clk_core); #1;
        if (!protocol_error)
            $fatal(1, "M62 orphan-event attack was not rejected");

        $display("PASS M62 directed groups=%0d events=%0d zero=%0d lanes=%0d protocol_attacks=1 full8=1",
                 groups_checked, events_accepted, zero_groups, LANES);
        $finish;
    end
endmodule

`default_nettype wire
