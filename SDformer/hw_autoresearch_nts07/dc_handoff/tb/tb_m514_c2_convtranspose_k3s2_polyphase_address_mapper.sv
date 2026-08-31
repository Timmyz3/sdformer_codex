`timescale 1ns/1ps
`default_nettype none

module tb_m514_c2_convtranspose_k3s2_polyphase_address_mapper;
    localparam int TAG_BITS = 8;
    localparam int CHANNEL_BITS = 8;
    localparam int COORD_BITS = 6;
    localparam int TIME_BITS = 4;
    localparam int MAX_EXPECTED = 256;

    logic clk_core = 0;
    logic rst_core = 1;
    always #1.5 clk_core = ~clk_core;

    logic event_valid, event_ready, event_accept;
    logic [TAG_BITS-1:0] event_tag;
    logic [TIME_BITS-1:0] event_time;
    logic [CHANNEL_BITS-1:0] event_source_channel;
    logic [COORD_BITS-1:0] event_source_y, event_source_x;
    logic [COORD_BITS-1:0] event_input_height, event_input_width;
    logic event_last;
    logic tap_valid, tap_ready, tap_accept;
    logic [TAG_BITS-1:0] tap_tag;
    logic [TIME_BITS-1:0] tap_time;
    logic [CHANNEL_BITS-1:0] tap_source_channel;
    logic [COORD_BITS-1:0] tap_source_y, tap_source_x;
    logic [1:0] tap_kernel_y, tap_kernel_x;
    logic [3:0] tap_kernel_index;
    logic [COORD_BITS-1:0] tap_destination_y, tap_destination_x;
    logic [1:0] tap_phase_bank;
    logic tap_last_for_event, tap_stream_last;
    logic protocol_error, busy;

    logic [TAG_BITS-1:0] exp_tag [0:MAX_EXPECTED-1];
    logic [TIME_BITS-1:0] exp_time [0:MAX_EXPECTED-1];
    logic [CHANNEL_BITS-1:0] exp_channel [0:MAX_EXPECTED-1];
    logic [COORD_BITS-1:0] exp_sy [0:MAX_EXPECTED-1];
    logic [COORD_BITS-1:0] exp_sx [0:MAX_EXPECTED-1];
    logic [1:0] exp_ky [0:MAX_EXPECTED-1];
    logic [1:0] exp_kx [0:MAX_EXPECTED-1];
    logic [COORD_BITS-1:0] exp_dy [0:MAX_EXPECTED-1];
    logic [COORD_BITS-1:0] exp_dx [0:MAX_EXPECTED-1];
    logic [1:0] exp_phase [0:MAX_EXPECTED-1];
    logic exp_last [0:MAX_EXPECTED-1];
    logic exp_stream_last [0:MAX_EXPECTED-1];
    integer expected_count = 0;
    integer observed_count = 0;
    integer replacement_count = 0;
    integer stall_count = 0;
    integer phase_count [0:3];
    logic [15:0] lfsr = 16'h1ace;
    logic force_stall = 0;

    m514_c2_convtranspose_k3s2_polyphase_address_mapper #(
        .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
        .COORD_BITS(COORD_BITS), .TIME_BITS(TIME_BITS)
    ) dut (.*);

    function automatic logic slot_valid(
        input integer slot, input integer sy, input integer sx);
        case (slot)
            0: slot_valid = sy != 0 && sx != 0;
            1, 4: slot_valid = sy != 0;
            2, 6: slot_valid = sx != 0;
            default: slot_valid = 1'b1;
        endcase
    endfunction

    task automatic slot_to_kernel(
        input integer slot, output integer ky, output integer kx);
        case (slot)
            0: begin ky = 0; kx = 0; end
            1: begin ky = 0; kx = 2; end
            2: begin ky = 2; kx = 0; end
            3: begin ky = 2; kx = 2; end
            4: begin ky = 0; kx = 1; end
            5: begin ky = 2; kx = 1; end
            6: begin ky = 1; kx = 0; end
            7: begin ky = 1; kx = 2; end
            default: begin ky = 1; kx = 1; end
        endcase
    endtask

    task automatic enqueue_event(
        input integer tag_value, input integer time_value,
        input integer channel_value, input integer sy, input integer sx,
        input logic stream_last);
        integer slot, ky, kx, last_slot;
        begin
            last_slot = -1;
            for (slot = 0; slot < 9; slot = slot + 1)
                if (slot_valid(slot, sy, sx)) last_slot = slot;
            for (slot = 0; slot < 9; slot = slot + 1) begin
                if (slot_valid(slot, sy, sx)) begin
                    if (expected_count >= MAX_EXPECTED)
                        $fatal(1, "M514 expected queue overflow");
                    slot_to_kernel(slot, ky, kx);
                    exp_tag[expected_count] = tag_value;
                    exp_time[expected_count] = time_value;
                    exp_channel[expected_count] = channel_value;
                    exp_sy[expected_count] = sy;
                    exp_sx[expected_count] = sx;
                    exp_ky[expected_count] = ky;
                    exp_kx[expected_count] = kx;
                    exp_dy[expected_count] = (2 * sy) - 1 + ky;
                    exp_dx[expected_count] = (2 * sx) - 1 + kx;
                    exp_phase[expected_count] =
                        {exp_dy[expected_count][0],
                         exp_dx[expected_count][0]};
                    exp_last[expected_count] = slot == last_slot;
                    exp_stream_last[expected_count] =
                        stream_last && slot == last_slot;
                    expected_count = expected_count + 1;
                end
            end
        end
    endtask

    task automatic drive_event(
        input integer tag_value, input integer time_value,
        input integer channel_value, input integer sy, input integer sx,
        input integer height, input integer width,
        input logic stream_last);
        begin
            enqueue_event(tag_value, time_value, channel_value, sy, sx,
                          stream_last);
            @(negedge clk_core);
            event_tag = tag_value;
            event_time = time_value;
            event_source_channel = channel_value;
            event_source_y = sy;
            event_source_x = sx;
            event_input_height = height;
            event_input_width = width;
            event_last = stream_last;
            event_valid = 1'b1;
            do @(posedge clk_core); while (!event_accept);
            @(negedge clk_core);
            event_valid = 1'b0;
        end
    endtask

    always @(negedge clk_core) begin
        if (rst_core) begin
            lfsr <= 16'h1ace;
            tap_ready <= 1'b0;
        end else if (force_stall) begin
            tap_ready <= 1'b0;
        end else begin
            lfsr <= {lfsr[14:0],
                     lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
            tap_ready <= lfsr[0] | lfsr[3];
        end
    end

    always @(posedge clk_core) begin
        if (!rst_core && tap_valid && !tap_ready)
            stall_count = stall_count + 1;
        if (!rst_core && event_accept && tap_accept && tap_last_for_event)
            replacement_count = replacement_count + 1;
        if (!rst_core && tap_accept) begin
            if (observed_count >= expected_count)
                $fatal(1, "M514 unexpected tap");
            if (tap_tag !== exp_tag[observed_count]
                    || tap_time !== exp_time[observed_count]
                    || tap_source_channel !== exp_channel[observed_count]
                    || tap_source_y !== exp_sy[observed_count]
                    || tap_source_x !== exp_sx[observed_count]
                    || tap_kernel_y !== exp_ky[observed_count]
                    || tap_kernel_x !== exp_kx[observed_count]
                    || tap_kernel_index !==
                       exp_ky[observed_count] * 3 + exp_kx[observed_count]
                    || tap_destination_y !== exp_dy[observed_count]
                    || tap_destination_x !== exp_dx[observed_count]
                    || tap_phase_bank !== exp_phase[observed_count]
                    || tap_last_for_event !== exp_last[observed_count]
                    || tap_stream_last !== exp_stream_last[observed_count])
                $fatal(1, "M514 tuple mismatch at %0d", observed_count);
            phase_count[tap_phase_bank] = phase_count[tap_phase_bank] + 1;
            observed_count = observed_count + 1;
        end
    end

    property p_tap_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        tap_valid && !tap_ready |=> tap_valid &&
            $stable({tap_tag, tap_time, tap_source_channel,
                     tap_source_y, tap_source_x, tap_kernel_y, tap_kernel_x,
                     tap_kernel_index, tap_destination_y, tap_destination_x,
                     tap_phase_bank, tap_last_for_event, tap_stream_last});
    endproperty
    assert property (p_tap_stable_under_stall);

    initial begin
        event_valid = 0;
        event_tag = 0;
        event_time = 0;
        event_source_channel = 0;
        event_source_y = 0;
        event_source_x = 0;
        event_input_height = 0;
        event_input_width = 0;
        event_last = 0;
        tap_ready = 0;
        for (int phase = 0; phase < 4; phase++) phase_count[phase] = 0;

        repeat (5) @(posedge clk_core);
        rst_core = 0;

        // Fanout 4, 6, 6, and 9 plus a second interior event.  The final
        // event carries the stream fence.  Backpressure is active throughout.
        drive_event(1, 0, 3, 0, 0, 3, 4, 0);
        drive_event(2, 1, 5, 0, 2, 3, 4, 0);
        drive_event(3, 2, 7, 2, 0, 3, 4, 0);
        drive_event(4, 3, 9, 1, 2, 3, 4, 0);
        drive_event(5, 4, 11, 2, 3, 3, 4, 0);

        while (observed_count != expected_count || busy)
            @(posedge clk_core);
        if (expected_count != 34 || observed_count != 34)
            $fatal(1, "M514 exact fanout total mismatch");
        if (replacement_count == 0 || stall_count == 0)
            $fatal(1, "M514 replacement/stall cover missing");
        for (int phase = 0; phase < 4; phase++)
            if (phase_count[phase] == 0)
                $fatal(1, "M514 phase cover missing %0d", phase);

        // Accept another legal event, stall one of its taps, then inject an
        // illegal successor.  The fault must lock out future events without
        // retracting or losing the already advertised/pending legal taps.
        // The maximum legal dimension for COORD_BITS=6 is 32; source 31
        // must emit nine taps including destination coordinate 63.
        drive_event(6, 5, 13, 31, 31, 32, 32, 1);
        @(posedge clk_core);
        force_stall = 1'b1;
        while (!(tap_valid && !tap_ready)) @(posedge clk_core);
        @(negedge clk_core);
        event_tag = 8'hff;
        event_source_y = 32;
        event_source_x = 0;
        event_input_height = 32;
        event_input_width = 32;
        event_valid = 1;
        @(posedge clk_core);
        @(negedge clk_core);
        event_valid = 0;
        force_stall = 0;
        while (observed_count != expected_count || busy)
            @(posedge clk_core);
        if (expected_count != 43 || observed_count != 43)
            $fatal(1, "M514 fault-drain tuple total mismatch");
        if (!protocol_error || event_ready || tap_valid || busy)
            $fatal(1, "M514 illegal successor did not lock then drain");

        $display("PASS M514 exact_taps=%0d stalls=%0d replacements=%0d phases=%0d/%0d/%0d/%0d protocol_attack=1",
                 observed_count, stall_count, replacement_count,
                 phase_count[0], phase_count[1], phase_count[2],
                 phase_count[3]);
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "M514 timeout");
    end
endmodule

`default_nettype wire
