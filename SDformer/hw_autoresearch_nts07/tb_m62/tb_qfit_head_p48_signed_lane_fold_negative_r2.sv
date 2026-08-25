`timescale 1ns/1ps
`default_nettype none

module tb_qfit_head_p48_signed_lane_fold_negative_r2;
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
    logic legal_case_active, attack_case_active;
    logic [3:0] legal_case_id, attack_case_id;

    integer signed golden [0:LANES-1];
    integer expected_sources, expected_events;
    integer legal_groups, lane_checks, attacks, attack_accepts;
    integer sticky_cycles, mismatches;

    qfit_head_p48_signed_lane_fold dut (.*);

    qfit_head_p48_signed_lane_fold_negative_assertions_r2 m62_r2_sva (
        .clk_core,
        .rst_core,
        .command_valid,
        .command_ready,
        .command_accept,
        .event_valid,
        .event_ready,
        .event_accept,
        .output_valid,
        .output_ready,
        .output_accept,
        .output_tag,
        .output_acc,
        .output_source_issues,
        .output_signed_events,
        .protocol_error,
        .mask_overlap(dut.mask_overlap),
        .invalid_slot_mask(dut.invalid_slot_mask),
        .reserved_negative_weight(dut.reserved_negative_weight),
        .event_has_source(dut.event_has_source),
        .event_signed_count(dut.event_signed_count),
        .accumulator_overflow(dut.accumulator_overflow),
        .legal_case_active,
        .legal_case_id,
        .attack_case_active,
        .attack_case_id
    );

    always #1.5 clk_core = ~clk_core;

    function automatic integer signed legal_weight(
        input integer case_id,
        input integer slot,
        input integer output_index
    );
        integer signed ladder;
        begin
            case (case_id)
                0: legal_weight = ((slot + output_index) % 2) ? -127 : 127;
                1: legal_weight = 127;
                2: legal_weight = (slot < 4) ? 127 : -127;
                3: legal_weight = 127;
                4: legal_weight = -127;
                default: begin
                    case (slot)
                        0: ladder = 127;
                        1: ladder = -127;
                        2: ladder = 63;
                        3: ladder = -63;
                        4: ladder = 31;
                        5: ladder = -31;
                        6: ladder = 1;
                        default: ladder = -1;
                    endcase
                    legal_weight = output_index ? -ladder : ladder;
                end
            endcase
        end
    endfunction

    task automatic clear_drivers;
        begin
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
            legal_case_active = 1'b0;
            legal_case_id = '0;
            attack_case_active = 1'b0;
            attack_case_id = '0;
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            clear_drivers();
            rst_core = 1'b1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
        end
    endtask

    task automatic drive_command(
        input integer tag_value,
        input integer signed seed_value
    );
        begin
            command_seed_acc = '0;
            for (int lane = 0; lane < LANES; lane++) begin
                golden[lane] = seed_value;
                command_seed_acc[lane*ACC_W +: ACC_W] = ACC_W'(seed_value);
            end
            expected_sources = 0;
            expected_events = 0;
            @(negedge clk_core);
            command_tag = 48'(tag_value);
            command_zero_event_group = 1'b0;
            command_valid = 1'b1;
            do @(posedge clk_core); while (!command_accept);
            @(negedge clk_core);
            command_valid = 1'b0;
        end
    endtask

    task automatic build_legal_event(input integer case_id);
        integer signed weight;
        begin
            event_source_valid = 8'hff;
            event_positive_mask = '0;
            event_negative_mask = '0;
            event_weight = '0;
            expected_sources = SLOTS;
            expected_events = 0;
            for (int slot = 0; slot < SLOTS; slot++) begin
                for (int output_index = 0; output_index < OUTPUTS;
                        output_index++) begin
                    weight = legal_weight(case_id, slot, output_index);
                    event_weight[(slot*OUTPUTS+output_index)*8 +: 8]
                        = 8'(weight);
                end
                for (int pixel = 0; pixel < PIXELS; pixel++) begin
                    case (case_id)
                        0: begin
                            if (pixel == (slot * 6) % PIXELS) begin
                                if (slot % 2)
                                    event_negative_mask[slot*PIXELS+pixel]
                                        = 1'b1;
                                else
                                    event_positive_mask[slot*PIXELS+pixel]
                                        = 1'b1;
                            end
                        end
                        1: begin
                            if (slot < 4)
                                event_positive_mask[slot*PIXELS+pixel] = 1'b1;
                            else
                                event_negative_mask[slot*PIXELS+pixel] = 1'b1;
                        end
                        5: begin
                            if ((slot + pixel) % 2)
                                event_negative_mask[slot*PIXELS+pixel] = 1'b1;
                            else
                                event_positive_mask[slot*PIXELS+pixel] = 1'b1;
                        end
                        default:
                            event_positive_mask[slot*PIXELS+pixel] = 1'b1;
                    endcase
                end
            end

            // Build the reference independently from the packed transaction.
            for (int slot = 0; slot < SLOTS; slot++) begin
                for (int pixel = 0; pixel < PIXELS; pixel++) begin
                    if (event_positive_mask[slot*PIXELS+pixel]
                            ^ event_negative_mask[slot*PIXELS+pixel]) begin
                        expected_events++;
                        for (int output_index = 0; output_index < OUTPUTS;
                                output_index++) begin
                            weight = $signed(event_weight[
                                (slot*OUTPUTS+output_index)*8 +: 8]);
                            if (event_positive_mask[slot*PIXELS+pixel])
                                golden[pixel*OUTPUTS+output_index] += weight;
                            else
                                golden[pixel*OUTPUTS+output_index] -= weight;
                        end
                    end
                end
            end
        end
    endtask

    task automatic drive_legal_event(input integer case_id);
        begin
            legal_case_active = 1'b1;
            legal_case_id = 4'(case_id);
            build_legal_event(case_id);
            @(negedge clk_core);
            output_ready = 1'b0;
            event_last = 1'b1;
            event_valid = 1'b1;
            do @(posedge clk_core); while (!event_accept);
            @(negedge clk_core);
            event_valid = 1'b0;
            event_last = 1'b0;
        end
    endtask

    task automatic check_legal_output(
        input integer case_id,
        input integer stall_count
    );
        logic [LANES*ACC_W-1:0] held_acc;
        logic [47:0] held_tag;
        begin
            do @(posedge clk_core); while (!output_valid);
            #1;
            if (protocol_error || output_tag !== 48'(16'h6200 + case_id)
                    || output_source_issues !== expected_sources[15:0]
                    || output_signed_events !== expected_events[15:0]) begin
                mismatches++;
                $fatal(1, "M62 r2 legal metadata/fault mismatch case=%0d", case_id);
            end
            for (int lane = 0; lane < LANES; lane++) begin
                lane_checks++;
                if ($signed(output_acc[lane*ACC_W +: ACC_W])
                        !== golden[lane]) begin
                    mismatches++;
                    $fatal(1, "M62 r2 lane mismatch case=%0d lane=%0d got=%0d expected=%0d",
                           case_id, lane,
                           $signed(output_acc[lane*ACC_W +: ACC_W]),
                           golden[lane]);
                end
            end
            if (case_id == 3 && golden[0] != 4094)
                $fatal(1, "M62 r2 positive near-limit oracle drift");
            if (case_id == 4 && golden[0] != -4095)
                $fatal(1, "M62 r2 negative near-limit oracle drift");
            held_acc = output_acc;
            held_tag = output_tag;
            repeat (stall_count) begin
                @(posedge clk_core); #1;
                if (!output_valid || output_acc !== held_acc
                        || output_tag !== held_tag) begin
                    mismatches++;
                    $fatal(1, "M62 r2 output changed under case-5 stall");
                end
            end
            @(negedge clk_core);
            output_ready = 1'b1;
            do @(posedge clk_core); while (!output_accept);
            legal_groups++;
            @(negedge clk_core);
            legal_case_active = 1'b0;
        end
    endtask

    task automatic build_attack(input integer case_id);
        begin
            event_source_valid = '0;
            event_positive_mask = '0;
            event_negative_mask = '0;
            event_weight = '0;
            event_source_valid[0] = 1'b1;
            event_weight[0 +: 8] = 8'sd1;
            event_weight[8 +: 8] = 8'sd1;
            case (case_id)
                0: begin
                    event_positive_mask[0] = 1'b1;
                    event_negative_mask[0] = 1'b1;
                    event_positive_mask[1] = 1'b1;
                end
                1: begin
                    event_positive_mask[0] = 1'b1;
                    event_positive_mask[PIXELS+1] = 1'b1;
                end
                2: begin
                    event_positive_mask[0] = 1'b1;
                    event_weight[0 +: 8] = 8'h80;
                end
                3: begin
                    // Valid source and legal weights, but no signed mask work.
                end
                4: begin
                    for (int pixel = 0; pixel < PIXELS; pixel++)
                        event_positive_mask[pixel] = 1'b1;
                    event_weight[0 +: 8] = 8'sd7;
                    event_weight[8 +: 8] = 8'sd7;
                end
                default: $fatal(1, "M62 r2 unknown attack case");
            endcase
        end
    endtask

    task automatic run_attack(input integer case_id);
        logic accepted_on_edge;
        integer signed seed_value;
        begin
            reset_dut();
            seed_value = (case_id == 4) ? 4090 : 0;
            drive_command(16'h6280 + case_id, seed_value);
            attack_case_active = 1'b1;
            attack_case_id = 4'(case_id);
            build_attack(case_id);
            @(negedge clk_core);
            event_last = 1'b1;
            event_valid = 1'b1;
            @(posedge clk_core);
            accepted_on_edge = event_accept;
            if (!accepted_on_edge)
                $fatal(1, "M62 r2 malformed event was not accepted case=%0d",
                       case_id);
            attack_accepts++;
            #1;
            if (!protocol_error)
                $fatal(1, "M62 r2 accepted attack did not fault case=%0d",
                       case_id);
            attacks++;
            @(negedge clk_core);
            event_valid = 1'b0;
            event_last = 1'b0;

            // Prove the fault is sticky and all three interfaces fail closed.
            command_valid = 1'b1;
            event_valid = 1'b1;
            output_ready = 1'b1;
            repeat (3) begin
                @(posedge clk_core); #1;
                sticky_cycles++;
                if (!protocol_error || command_ready || event_ready
                        || output_valid || command_accept || event_accept
                        || output_accept)
                    $fatal(1, "M62 r2 fault did not stay closed case=%0d",
                           case_id);
            end
            @(negedge clk_core);
            command_valid = 1'b0;
            event_valid = 1'b0;
            attack_case_active = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        clear_drivers();
        legal_groups = 0;
        lane_checks = 0;
        attacks = 0;
        attack_accepts = 0;
        sticky_cycles = 0;
        mismatches = 0;
        repeat (4) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        for (int case_id = 0; case_id < 6; case_id++) begin
            integer signed seed_value;
            case (case_id)
                3: seed_value = 3078;
                4: seed_value = -3079;
                default: seed_value = 0;
            endcase
            drive_command(16'h6200 + case_id, seed_value);
            drive_legal_event(case_id);
            check_legal_output(case_id, (case_id == 5) ? 5 : 0);
        end

        for (int case_id = 0; case_id < 5; case_id++)
            run_attack(case_id);

        if (legal_groups != 6 || lane_checks != 576 || attacks != 5
                || attack_accepts != 5 || sticky_cycles != 15
                || mismatches != 0)
            $fatal(1, "M62 r2 terminal counter mismatch");
        $display("PASS M62 R2 directed_negative legal_full8=6 lane_checks=576 attacks=5 attack_accepts=5 sticky_cycles=15 mismatches=0");
        $finish;
    end
endmodule

`default_nettype wire
