`timescale 1ns/1ps
`default_nettype none

module tb_m104_independent_adversarial;
    logic clk_core, rst_core;
    logic load_valid, load_ready;
    logic [3:0] load_source;
    logic [2:0] load_block;
    logic [1:0] load_beat;
    logic [255:0] load_data;
    logic load_accept;
    logic event_valid, event_ready;
    logic [3:0] event_source;
    logic [2:0] event_block;
    logic event_negate, event_last_for_key;
    logic [31:0] event_tag;
    logic event_accept;
    logic output_valid, output_ready;
    logic [31:0] output_tag;
    logic [3:0] output_source;
    logic [2:0] output_block;
    logic output_negate;
    logic [1151:0] output_values;
    logic output_accept;
    logic held_valid, collecting;
    logic [1:0] expected_load_beat;
    logic protocol_error, busy;

    logic [767:0] payload;
    logic [1151:0] expected_positive, expected_negative;
    integer signed_codes_checked;
    integer ii1_turnovers;
    integer sticky_cycles;

    m104_held_weight_correction_broadcaster dut (.*);
    m104_independent_adversarial_assertions audit_sva (
        .clk_core(clk_core), .rst_core(rst_core),
        .load_ready(load_ready),
        .event_valid(event_valid), .event_ready(event_ready),
        .event_accept(event_accept),
        .event_last_for_key(event_last_for_key),
        .output_valid(output_valid), .output_ready(output_ready),
        .output_accept(output_accept), .output_tag(output_tag),
        .output_values(output_values), .held_valid(held_valid),
        .collecting(collecting), .protocol_error(protocol_error),
        .illegal_request(dut.illegal_request),
        .request_fault(dut.request_fault_q),
        .output_valid_q(dut.output_valid_q)
    );

    always #1.5 clk_core = ~clk_core;

    initial begin
        #200000;
        $fatal(1, "M104 independent watchdog timeout");
    end

    task automatic clear_drives;
        begin
            load_valid = 1'b0;
            load_source = '0;
            load_block = '0;
            load_beat = '0;
            load_data = '0;
            event_valid = 1'b0;
            event_source = '0;
            event_block = '0;
            event_negate = 1'b0;
            event_last_for_key = 1'b0;
            event_tag = '0;
            output_ready = 1'b1;
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            clear_drives();
            rst_core = 1'b1;
            repeat (2) @(posedge clk_core);
            #0.1;
            if (dut.request_fault_q || dut.output_valid_q || held_valid || collecting)
                $fatal(1, "M104 reset failed to clear state");
            @(negedge clk_core);
            rst_core = 1'b0;
        end
    endtask

    task automatic build_exhaustive_payload(input integer group_index);
        integer code;
        integer signed_value;
        logic signed [11:0] extended;
        begin
            payload = '0;
            expected_positive = '0;
            expected_negative = '0;
            for (int lane = 0; lane < 96; lane++) begin
                code = (group_index * 96 + lane) & 255;
                signed_value = code < 128 ? code : code - 256;
                payload[lane*8 +: 8] = code[7:0];
                extended = signed_value;
                expected_positive[lane*12 +: 12] = extended;
                expected_negative[lane*12 +: 12] = -extended;
            end
        end
    endtask

    task automatic load_payload(
        input integer source,
        input integer block,
        input integer idle_gap
    );
        begin
            for (int beat = 0; beat < 3; beat++) begin
                @(negedge clk_core);
                load_valid = 1'b1;
                load_source = source[3:0];
                load_block = block[2:0];
                load_beat = beat[1:0];
                load_data = payload[beat*256 +: 256];
                #0.1;
                if (!load_ready || !load_accept || protocol_error)
                    $fatal(1, "M104 legal load rejected beat=%0d", beat);
                @(posedge clk_core);
                @(negedge clk_core);
                load_valid = 1'b0;
                if (beat != 2 && idle_gap != 0) begin
                    repeat (idle_gap) begin
                        @(posedge clk_core); #0.1;
                        if (!collecting || held_valid
                                || expected_load_beat != beat + 1
                                || protocol_error)
                            $fatal(1, "M104 load gap state drift beat=%0d", beat);
                    end
                end
            end
            #0.1;
            if (!held_valid || collecting || expected_load_beat != 0)
                $fatal(1, "M104 three-beat load did not expose held vector");
        end
    endtask

    task automatic check_output(
        input logic [31:0] tag,
        input logic negate
    );
        logic [1151:0] expected_value;
        begin
            expected_value = negate ? expected_negative : expected_positive;
            if (!output_valid || output_tag != tag || output_negate != negate
                    || output_values !== expected_value) begin
                for (int lane = 0; lane < 96; lane++) begin
                    if (output_values[lane*12 +: 12]
                            !== expected_value[lane*12 +: 12])
                        $display("M104 first mismatching lane=%0d got=%03x expected=%03x",
                                 lane, output_values[lane*12 +: 12],
                                 expected_value[lane*12 +: 12]);
                end
                $fatal(1, "M104 signed output mismatch tag=%08x negate=%0d",
                       tag, negate);
            end
        end
    endtask

    initial begin : independent_test
        logic [1151:0] stalled_value;

        clk_core = 1'b0;
        rst_core = 1'b1;
        signed_codes_checked = 0;
        ii1_turnovers = 0;
        sticky_cycles = 0;
        clear_drives();
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        // Across three groups, all 256 INT8 bit patterns appear in every
        // checked positive and negative lane result.  Events are II=1.
        for (int group_index = 0; group_index < 3; group_index++) begin
            build_exhaustive_payload(group_index);
            load_payload(group_index + 1, group_index, 0);
            for (int sign_index = 0; sign_index < 2; sign_index++) begin
                @(negedge clk_core);
                event_valid = 1'b1;
                event_source = group_index + 1;
                event_block = group_index;
                event_negate = sign_index[0];
                event_last_for_key = sign_index == 1;
                event_tag = 32'h104a0000 + group_index*2 + sign_index;
                #0.1;
                if (!event_ready || !event_accept || protocol_error)
                    $fatal(1, "M104 II=1 event rejected group=%0d sign=%0d",
                           group_index, sign_index);
                @(posedge clk_core);
                if (sign_index == 1) begin
                    #0.01;
                    event_valid = 1'b0;
                end
                #0.09;
                check_output(32'h104a0000 + group_index*2 + sign_index,
                             sign_index[0]);
                if (sign_index == 1)
                    ii1_turnovers++;
            end
            signed_codes_checked += group_index == 2 ? 64 : 96;
            @(negedge clk_core);
            event_valid = 1'b0;
            @(posedge clk_core); #0.1;
            if (output_valid || held_valid || protocol_error)
                $fatal(1, "M104 II=1 group failed drain/release");
        end
        if (signed_codes_checked != 256)
            $fatal(1, "M104 exhaustive code accounting drift");

        // Legal last descriptor waits under backpressure, then replaces the
        // old output in the exact cycle ready releases it: no bubble, no loss.
        build_exhaustive_payload(0);
        load_payload(9, 5, 0);
        @(negedge clk_core);
        output_ready = 1'b0;
        event_valid = 1'b1;
        event_source = 4'd9;
        event_block = 3'd5;
        event_negate = 1'b0;
        event_last_for_key = 1'b0;
        event_tag = 32'h104b0001;
        #0.1;
        if (!event_ready) $fatal(1, "M104 empty output did not accept event");
        @(posedge clk_core); #0.1;
        check_output(32'h104b0001, 1'b0);
        stalled_value = output_values;

        @(negedge clk_core);
        event_negate = 1'b1;
        event_last_for_key = 1'b1;
        event_tag = 32'h104b0002;
        #0.1;
        if (event_ready || protocol_error || !held_valid)
            $fatal(1, "M104 stalled last descriptor protocol drift");
        @(posedge clk_core); #0.1;
        if (!output_valid || output_values !== stalled_value || !held_valid)
            $fatal(1, "M104 unaccepted last released or corrupted state");

        @(negedge clk_core);
        output_ready = 1'b1;
        #0.1;
        if (!event_ready || !event_accept || !output_accept)
            $fatal(1, "M104 ready-release turnover not accepted");
        @(posedge clk_core); #0.01;
        event_valid = 1'b0;
        #0.09;
        check_output(32'h104b0002, 1'b1);
        if (held_valid) $fatal(1, "M104 accepted last did not release key");
        ii1_turnovers++;
        @(negedge clk_core);
        event_valid = 1'b0;
        @(posedge clk_core); #0.1;
        if (output_valid) $fatal(1, "M104 turnover output failed to drain");

        // Hold an old result, then release ready in the same cycle as a bad
        // new request.  The old result must never handshake or escape.
        build_exhaustive_payload(1);
        load_payload(6, 2, 0);
        @(negedge clk_core);
        output_ready = 1'b0;
        event_valid = 1'b1;
        event_source = 4'd6;
        event_block = 3'd2;
        event_negate = 1'b0;
        event_last_for_key = 1'b0;
        event_tag = 32'h104c0001;
        #0.1;
        if (!event_ready) $fatal(1, "M104 fault setup event not ready");
        @(posedge clk_core); #0.1;
        check_output(32'h104c0001, 1'b0);

        @(negedge clk_core);
        output_ready = 1'b1;
        event_source = 4'd7;
        event_tag = 32'h104cbad0;
        #0.1;
        if (!protocol_error || output_valid || output_accept
                || load_ready || event_ready || !dut.output_valid_q)
            $fatal(1, "M104 same-cycle invalid+ready quarantine failed");
        @(posedge clk_core); #0.1;
        if (!dut.request_fault_q || !dut.output_valid_q
                || output_valid || output_accept)
            $fatal(1, "M104 fault edge retired old stalled output");

        @(negedge clk_core);
        event_valid = 1'b0;
        repeat (3) begin
            @(posedge clk_core); #0.1;
            if (!protocol_error || !dut.request_fault_q || output_valid
                    || output_accept || !dut.output_valid_q)
                $fatal(1, "M104 sticky quarantine escaped without reset");
            sticky_cycles++;
        end

        // Reset is the only recovery and must discard the quarantined result.
        reset_dut();
        if (protocol_error || dut.output_valid_q || output_valid || held_valid)
            $fatal(1, "M104 reset-only recovery failed");

        // Three-beat loading permits bubbles but not beat/identity drift.
        build_exhaustive_payload(2);
        load_payload(4, 7, 2);
        @(negedge clk_core);
        event_valid = 1'b1;
        event_source = 4'd4;
        event_block = 3'd7;
        event_negate = 1'b1;
        event_last_for_key = 1'b1;
        event_tag = 32'h104d0001;
        #0.1;
        if (!event_ready) $fatal(1, "M104 post-gap event not ready");
        @(posedge clk_core); #0.01;
        event_valid = 1'b0;
        #0.09;
        check_output(32'h104d0001, 1'b1);
        @(negedge clk_core);
        event_valid = 1'b0;
        @(posedge clk_core); #0.1;
        if (output_valid || held_valid || protocol_error)
            $fatal(1, "M104 post-reset/load-gap recovery did not drain");

        if (ii1_turnovers != 4 || sticky_cycles != 3)
            $fatal(1, "M104 independent counter mismatch");
        $display("PASS M104 independent adversarial VCS signed_codes=256 lanes=96 signs=2 ready_release_fault=1 sticky_cycles=3 reset_recovery=1 ii1_turnovers=4 load_gap=1 last_wait=1");
        $finish;
    end
endmodule

`default_nettype wire
