`timescale 1ns/1ps
`default_nettype none

module tb_m204_fc2_descriptor4_paired_window_fixed_bank_frontend;
    localparam int TAG_BITS = 24;
    logic clk_core = 0, rst_core;
    always #1.5 clk_core = ~clk_core;

    logic header_valid, header_ready, header_accept;
    logic [23:0] header_tag;
    logic [3:0] header_output_blocks;
    logic descriptor_valid, descriptor_ready, descriptor_accept;
    logic [2:0] descriptor_count;
    logic [23:0] descriptor_token_tag;
    logic [4:0] descriptor_beat_index [0:3];
    logic [95:0] descriptor_bitmap [0:3];
    logic [3:0] descriptor_window_last;
    logic upstream_done_valid, upstream_done_ready, upstream_done_accept;
    logic [23:0] upstream_done_tag;
    logic [5:0] upstream_done_descriptor_count;
    logic group_valid, group_ready, group_accept;
    logic [23:0] group_tag;
    logic [2:0] group_output_block;
    logic [3:0] group_source_count;
    logic [7:0] group_bank_valid;
    logic [11:0] group_source_channel [0:7];
    logic token_done_valid, token_done_ready, token_done_accept;
    logic [23:0] token_done_tag;
    logic [5:0] token_done_descriptor_count;
    logic token_done_had_event, protocol_error, busy;

    logic [95:0] ref_bitmap [0:1][0:7];
    logic [4:0] ref_index [0:1][0:7];
    int ref_count [0:1];
    logic [11:0] held_channel [0:7];
    logic [7:0] held_mask;
    int ref_output_block, ref_drain_buffer, active_output_blocks;
    int accepted_headers, accepted_packets, accepted_descriptors;
    int accepted_groups, accepted_done, paired_groups, odd_groups;
    int group_stalls, protocol_attacks;

    m204_fc2_descriptor4_paired_window_fixed_bank_frontend dut (.*);

    function automatic logic [95:0] pattern(input int ordinal);
        logic [95:0] value;
        int bank0, bank1, row0, row1;
        begin
            value = 0;
            bank0 = ordinal % 8; row0 = (ordinal * 3 + 1) % 12;
            bank1 = (ordinal * 5 + 3) % 8; row1 = (ordinal * 7 + 2) % 12;
            value[row0*8+bank0] = 1;
            value[row1*8+bank1] = 1;
            if (ordinal % 3 == 0)
                value[((ordinal+5)%12)*8+((ordinal+6)%8)] = 1;
            return value;
        end
    endfunction

    function automatic int bit_count8(input logic [7:0] value);
        int count = 0;
        for (int bit_index = 0; bit_index < 8; bit_index++)
            count += value[bit_index];
        return count;
    endfunction

    task automatic clear_reference;
        begin
            for (int buffer = 0; buffer < 2; buffer++) begin
                ref_count[buffer] = 0;
                for (int entry = 0; entry < 8; entry++) begin
                    ref_bitmap[buffer][entry] = 0;
                    ref_index[buffer][entry] = 0;
                end
            end
            ref_output_block = 0; ref_drain_buffer = 0; held_mask = 0;
            for (int bank = 0; bank < 8; bank++) held_channel[bank] = 0;
        end
    endtask

    task automatic apply_reset;
        begin
            rst_core = 1; header_valid = 0; descriptor_valid = 0;
            upstream_done_valid = 0; group_ready = 0; token_done_ready = 0;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core); rst_core = 0; clear_reference();
        end
    endtask

    task automatic send_header(input logic [23:0] tag, input int blocks);
        begin
            @(negedge clk_core); header_tag = tag;
            header_output_blocks = blocks; header_valid = 1;
            do @(posedge clk_core); while (!header_accept);
            accepted_headers++;
            @(negedge clk_core); header_valid = 0;
            descriptor_token_tag = tag; upstream_done_tag = tag;
            active_output_blocks = blocks;
        end
    endtask

    task automatic send_packet(
        input int buffer, input int count, input int base_index,
        input int ordinal_base, input logic closes_window
    );
        begin
            @(negedge clk_core);
            descriptor_count = count;
            descriptor_window_last = closes_window ? (1 << (count-1)) : 0;
            for (int lane = 0; lane < 4; lane++) begin
                descriptor_beat_index[lane] = base_index + lane;
                descriptor_bitmap[lane] = lane < count
                    ? pattern(ordinal_base + lane) : 0;
            end
            descriptor_valid = 1;
            do @(posedge clk_core); while (!descriptor_accept);
            for (int lane = 0; lane < count; lane++) begin
                ref_bitmap[buffer][ref_count[buffer]] = descriptor_bitmap[lane];
                ref_index[buffer][ref_count[buffer]] = descriptor_beat_index[lane];
                ref_count[buffer]++;
            end
            accepted_packets++; accepted_descriptors += count;
            @(negedge clk_core); descriptor_valid = 0;
        end
    endtask

    task automatic send_done(input int descriptors);
        begin
            @(negedge clk_core);
            upstream_done_descriptor_count = descriptors;
            upstream_done_valid = 1;
            do @(posedge clk_core); while (!upstream_done_accept);
            @(negedge clk_core); upstream_done_valid = 0;
        end
    endtask

    task automatic expected_new_source;
        logic found;
        begin
            held_mask = 0;
            for (int bank = 0; bank < 8; bank++) begin
                found = 0; held_channel[bank] = 0;
                for (int buffer = 0; buffer < 2; buffer++) begin
                    for (int entry = 0; entry < ref_count[buffer]; entry++) begin
                        for (int row = 0; row < 12; row++) begin
                            if (!found
                                    && (active_output_blocks != 1
                                        || buffer == ref_drain_buffer)
                                    && ref_bitmap[buffer][entry][row*8+bank]) begin
                                held_mask[bank] = 1;
                                held_channel[bank]
                                    = (ref_index[buffer][entry]*12+row)*8+bank;
                                ref_bitmap[buffer][entry][row*8+bank] = 0;
                                found = 1;
                            end
                        end
                    end
                end
            end
            if (held_mask == 0) $fatal(1, "M204 reference underflow");
        end
    endtask

    task automatic drain_token(input logic expect_event);
        logic [23:0] stalled_tag;
        logic [7:0] stalled_mask;
        logic [2:0] stalled_block;
        logic [11:0] stalled_channel [0:7];
        begin
            group_ready = 0; token_done_ready = 0;
            repeat (3) begin
                @(posedge clk_core);
                if (group_valid) begin
                    stalled_tag = group_tag; stalled_mask = group_bank_valid;
                    stalled_block = group_output_block;
                    for (int bank = 0; bank < 8; bank++)
                        stalled_channel[bank] = group_source_channel[bank];
                    @(posedge clk_core); group_stalls++;
                    if (group_tag !== stalled_tag
                            || group_bank_valid !== stalled_mask
                            || group_output_block !== stalled_block)
                        $fatal(1, "M204 group changed under stall");
                    for (int bank = 0; bank < 8; bank++)
                        if (group_source_channel[bank] !== stalled_channel[bank])
                            $fatal(1, "M204 channel changed under stall");
                end
            end
            @(negedge clk_core); group_ready = 1;
            while (!token_done_valid) begin
                @(posedge clk_core);
                if (group_accept) begin
                    if (ref_output_block == 0) expected_new_source();
                    if (group_output_block != ref_output_block
                            || group_bank_valid !== held_mask
                            || group_source_count != bit_count8(held_mask))
                        $fatal(1, "M204 group metadata mismatch actual block=%0d mask=%h count=%0d expected block=%0d mask=%h count=%0d",
                            group_output_block, group_bank_valid,
                            group_source_count, ref_output_block, held_mask,
                            bit_count8(held_mask));
                    for (int bank = 0; bank < 8; bank++)
                        if (held_mask[bank]
                                && group_source_channel[bank] !== held_channel[bank])
                            $fatal(1, "M204 source channel mismatch bank=%0d", bank);
                    accepted_groups++;
                    if (active_output_blocks != 1 && ref_count[1] != 0)
                        paired_groups++;
                    else odd_groups++;
                    if (ref_output_block + 1 == active_output_blocks) begin
                        ref_output_block = 0;
                        if (active_output_blocks == 1) begin
                            logic buffer_empty;
                            buffer_empty = 1;
                            for (int entry = 0;
                                    entry < ref_count[ref_drain_buffer]; entry++)
                                if (ref_bitmap[ref_drain_buffer][entry] != 0)
                                    buffer_empty = 0;
                            if (buffer_empty) begin
                                ref_count[ref_drain_buffer] = 0;
                                ref_drain_buffer = 1 - ref_drain_buffer;
                            end
                        end
                    end else ref_output_block++;
                end
            end
            if (token_done_had_event !== expect_event
                    || token_done_descriptor_count
                        != (expect_event ? accepted_descriptors : 0))
                $fatal(1, "M204 token done mismatch");
            @(negedge clk_core); token_done_ready = 1;
            @(posedge clk_core);
            accepted_done++;
            @(negedge clk_core); group_ready = 0; token_done_ready = 0;
        end
    endtask

    initial begin
        accepted_headers=0; accepted_packets=0; accepted_descriptors=0;
        accepted_groups=0; accepted_done=0; paired_groups=0; odd_groups=0;
        group_stalls=0; protocol_attacks=0;
        apply_reset();
        send_header(24'h204001, 2);
        send_packet(0,4,0,0,1); send_packet(1,4,4,4,1);
        send_done(8); drain_token(1);

        accepted_descriptors=0; clear_reference();
        send_header(24'h204002, 1);
        send_packet(0,3,0,8,0); send_done(3); drain_token(1);

        accepted_descriptors=0; clear_reference();
        send_header(24'h204003, 1);
        send_packet(0,4,0,11,1); send_packet(1,4,4,15,1);
        send_done(8); drain_token(1);

        accepted_descriptors=0; clear_reference();
        send_header(24'h204004, 4); send_done(0); drain_token(0);

        accepted_descriptors=0; clear_reference();
        send_header(24'h204005, 8);
        send_packet(0,4,0,19,0); send_packet(0,4,4,23,1);
        send_packet(1,2,8,27,0); send_done(10); drain_token(1);

        apply_reset(); @(negedge clk_core);
        header_tag=24'hbad001; header_output_blocks=3; header_valid=1;
        @(posedge clk_core); if (!protocol_error) $fatal(1,"bad header missed");
        protocol_attacks++;
        @(negedge clk_core); header_valid=0; @(posedge clk_core);
        apply_reset(); send_header(24'hbad002,2); @(negedge clk_core);
        descriptor_count=2; descriptor_token_tag=24'hbad003;
        descriptor_window_last=0; descriptor_bitmap[0]=1; descriptor_bitmap[1]=2;
        descriptor_beat_index[0]=0; descriptor_beat_index[1]=1;
        descriptor_valid=1; @(posedge clk_core);
        if (!protocol_error) $fatal(1,"bad tag missed"); protocol_attacks++;
        @(negedge clk_core); descriptor_valid=0; @(posedge clk_core);
        apply_reset(); send_header(24'hbad004,2); @(negedge clk_core);
        upstream_done_tag=24'hbad004; upstream_done_descriptor_count=1;
        upstream_done_valid=1; @(posedge clk_core);
        if (!protocol_error) $fatal(1,"bad done count missed"); protocol_attacks++;
        @(negedge clk_core); upstream_done_valid=0; @(posedge clk_core);

        $display("PASS M204 descriptor4 paired-window frontend VCS headers=%0d packets=%0d groups=%0d done=%0d paired_groups=%0d odd_groups=%0d group_stalls=%0d protocol_attacks=%0d complete_fc2=false physical_speedup=false system_speedup=false headline=false",
            accepted_headers, accepted_packets, accepted_groups, accepted_done,
            paired_groups, odd_groups, group_stalls, protocol_attacks);
        $finish;
    end
    initial begin #3000000; $fatal(1,"M204 watchdog timeout"); end
endmodule

`default_nettype wire
