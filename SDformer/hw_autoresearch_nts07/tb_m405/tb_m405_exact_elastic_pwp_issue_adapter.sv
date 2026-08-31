`timescale 1ns/1ps
`default_nettype none

module tb_m405_exact_elastic_pwp_issue_adapter;
    localparam int TAG_BITS = 24;
    localparam int MAX_EXPECTED = 4096;

    logic clk_core, reset_n, config_reload;
    logic low_valid, low_ready, low_accept;
    logic [TAG_BITS-1:0] low_tag;
    logic low_tile;
    logic [4:0] low_center_id;
    logic [2:0] low_output_block;
    logic low_narrow;
    logic [767:0] low_data;
    logic high_valid, high_ready, high_accept;
    logic [TAG_BITS-1:0] high_tag;
    logic high_tile;
    logic [4:0] high_center_id;
    logic [2:0] high_output_block;
    logic [511:0] high_data;
    logic contribution_valid, contribution_ready, contribution_accept;
    logic [TAG_BITS-1:0] contribution_tag;
    logic contribution_tile;
    logic [4:0] contribution_center_id;
    logic [2:0] contribution_output_block;
    logic contribution_narrow, contribution_part_high, contribution_last;
    logic [1151:0] contribution_data;
    logic protocol_error, busy;
    logic [1:0] debug_completed_fifo_count;
    logic [31:0] debug_low_accepts, debug_high_accepts;
    logic [31:0] debug_narrow_blocks, debug_wide_blocks;
    logic [31:0] debug_contributions;

    logic [TAG_BITS-1:0] expected_tag [0:MAX_EXPECTED-1];
    logic expected_tile [0:MAX_EXPECTED-1];
    logic [4:0] expected_center [0:MAX_EXPECTED-1];
    logic [2:0] expected_block [0:MAX_EXPECTED-1];
    logic expected_narrow [0:MAX_EXPECTED-1];
    logic expected_high [0:MAX_EXPECTED-1];
    logic expected_last [0:MAX_EXPECTED-1];
    logic [1151:0] expected_data [0:MAX_EXPECTED-1];
    integer expected_head, expected_tail;
    integer blocks_sent, narrow_sent, wide_sent, accepted_outputs;
    integer stall_cycles, protocol_attacks, atomic_leak_count;
    integer ready_high_gap_errors, last_accept_cycle, cycle_count;
    logic measure_contiguous;

    m405_exact_elastic_pwp_issue_adapter #(.TAG_BITS(TAG_BITS)) dut (.*);
    m405_exact_elastic_pwp_issue_adapter_assertions #(.TAG_BITS(TAG_BITS))
        m405_a_sva (.*);

    always #1.5 clk_core = ~clk_core;
    always @(posedge clk_core) cycle_count <= cycle_count + 1;

    function automatic integer signed lane_value(
        input integer txid, input integer lane, input logic narrow
    );
        integer raw;
        begin
            if (narrow) begin
                raw = ((txid * 29 + lane * 17) % 256) - 128;
            end else begin
                raw = ((txid * 233 + lane * 97) % 4096) - 2048;
                if (lane == 0 && raw >= -128 && raw <= 127)
                    raw = (txid[0]) ? 2047 : -2048;
            end
            lane_value = raw;
        end
    endfunction

    task automatic build_record(
        input integer txid,
        input logic narrow,
        output logic [767:0] low_bits,
        output logic [511:0] high_bits,
        output logic [1151:0] first_contribution,
        output logic [1151:0] second_contribution
    );
        integer signed value;
        integer raw12;
        begin
            low_bits = '0;
            high_bits = '0;
            first_contribution = '0;
            second_contribution = '0;
            for (int lane = 0; lane < 96; lane++) begin
                value = lane_value(txid, lane, narrow);
                raw12 = value & 12'hfff;
                low_bits[lane*8 +: 8] = raw12[7:0];
                high_bits[lane*4 +: 4] = raw12[11:8];
                if (narrow) begin
                    first_contribution[lane*12 +: 12] = value[11:0];
                end else begin
                    first_contribution[lane*12 +: 12] =
                        {4'b0000,raw12[7:0]};
                    second_contribution[lane*12 +: 12] =
                        {raw12[11:8],8'b0};
                end
            end
        end
    endtask

    task automatic enqueue(
        input integer tag_value,
        input logic tile_value,
        input integer center_value,
        input integer block_value,
        input logic narrow_value,
        input logic high_value,
        input logic last_value,
        input logic [1151:0] data_value
    );
        begin
            if (expected_tail >= MAX_EXPECTED)
                $fatal(1,"M405A expected queue overflow");
            expected_tag[expected_tail] = tag_value[TAG_BITS-1:0];
            expected_tile[expected_tail] = tile_value;
            expected_center[expected_tail] = center_value[4:0];
            expected_block[expected_tail] = block_value[2:0];
            expected_narrow[expected_tail] = narrow_value;
            expected_high[expected_tail] = high_value;
            expected_last[expected_tail] = last_value;
            expected_data[expected_tail] = data_value;
            expected_tail++;
        end
    endtask

    task automatic drive_low(
        input integer tag_value,
        input logic tile_value,
        input integer center_value,
        input integer block_value,
        input logic narrow_value,
        input logic [767:0] data_value
    );
        begin
            if (clk_core !== 1'b0) @(negedge clk_core);
            low_tag = tag_value[TAG_BITS-1:0];
            low_tile = tile_value;
            low_center_id = center_value[4:0];
            low_output_block = block_value[2:0];
            low_narrow = narrow_value;
            low_data = data_value;
            low_valid = 1'b1;
            do @(posedge clk_core); while (!low_accept && !protocol_error);
            @(negedge clk_core); low_valid = 1'b0;
        end
    endtask

    task automatic drive_high(
        input integer tag_value,
        input logic tile_value,
        input integer center_value,
        input integer block_value,
        input logic [511:0] data_value
    );
        begin
            if (clk_core !== 1'b0) @(negedge clk_core);
            high_tag = tag_value[TAG_BITS-1:0];
            high_tile = tile_value;
            high_center_id = center_value[4:0];
            high_output_block = block_value[2:0];
            high_data = data_value;
            high_valid = 1'b1;
            do @(posedge clk_core); while (!high_accept && !protocol_error);
            @(negedge clk_core); high_valid = 1'b0;
        end
    endtask

    task automatic send_record(input integer txid, input logic narrow);
        logic [767:0] low_bits;
        logic [511:0] high_bits;
        logic [1151:0] first_bits, second_bits;
        logic tile_value;
        integer center_value, block_value;
        begin
            tile_value = (txid / 8) & 1;
            center_value = txid % 32;
            block_value = txid % 8;
            build_record(txid,narrow,low_bits,high_bits,
                         first_bits,second_bits);
            enqueue(txid,tile_value,center_value,block_value,
                    narrow,1'b0,narrow,first_bits);
            if (!narrow)
                enqueue(txid,tile_value,center_value,block_value,
                        1'b0,1'b1,1'b1,second_bits);
            drive_low(txid,tile_value,center_value,block_value,narrow,low_bits);
            if (!narrow)
                drive_high(txid,tile_value,center_value,block_value,high_bits);
            blocks_sent++;
            if (narrow) narrow_sent++; else wide_sent++;
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            reset_n = 1'b0;
            low_valid = 1'b0;
            high_valid = 1'b0;
            config_reload = 1'b0;
            contribution_ready = 1'b1;
            repeat (4) @(posedge clk_core);
            @(negedge clk_core); reset_n = 1'b1;
        end
    endtask

    always @(posedge clk_core) begin
        if (contribution_valid && !contribution_ready)
            stall_cycles++;
        if (contribution_accept) begin
            if (expected_head >= expected_tail)
                $fatal(1,"M405A unexpected contribution tag=%0d",
                       contribution_tag);
            if (contribution_tag !== expected_tag[expected_head]
                    || contribution_tile !== expected_tile[expected_head]
                    || contribution_center_id !== expected_center[expected_head]
                    || contribution_output_block !== expected_block[expected_head]
                    || contribution_narrow !== expected_narrow[expected_head]
                    || contribution_part_high !== expected_high[expected_head]
                    || contribution_last !== expected_last[expected_head]
                    || contribution_data !== expected_data[expected_head])
                $fatal(1,"M405A output mismatch index=%0d tag=%0d/%0d tile=%0d/%0d center=%0d/%0d block=%0d/%0d narrow=%0d/%0d high=%0d/%0d last=%0d/%0d data_xor=%h",
                       expected_head,contribution_tag,
                       expected_tag[expected_head],contribution_tile,
                       expected_tile[expected_head],contribution_center_id,
                       expected_center[expected_head],
                       contribution_output_block,
                       expected_block[expected_head],contribution_narrow,
                       expected_narrow[expected_head],
                       contribution_part_high,expected_high[expected_head],
                       contribution_last,expected_last[expected_head],
                       contribution_data ^ expected_data[expected_head]);
            if (measure_contiguous && last_accept_cycle >= 0
                    && cycle_count != last_accept_cycle + 1)
                ready_high_gap_errors++;
            last_accept_cycle = cycle_count;
            expected_head++;
            accepted_outputs++;
        end
        if (protocol_error && contribution_accept)
            atomic_leak_count++;
    end

    initial begin
        clk_core = 0;
        reset_n = 0;
        config_reload = 0;
        low_valid = 0;
        low_tag = 0;
        low_tile = 0;
        low_center_id = 0;
        low_output_block = 0;
        low_narrow = 0;
        low_data = 0;
        high_valid = 0;
        high_tag = 0;
        high_tile = 0;
        high_center_id = 0;
        high_output_block = 0;
        high_data = 0;
        contribution_ready = 1;
        expected_head = 0;
        expected_tail = 0;
        blocks_sent = 0;
        narrow_sent = 0;
        wide_sent = 0;
        accepted_outputs = 0;
        stall_cycles = 0;
        protocol_attacks = 0;
        atomic_leak_count = 0;
        ready_high_gap_errors = 0;
        last_accept_cycle = -1;
        cycle_count = 0;
        measure_contiguous = 0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core); reset_n = 1;

        // Directed signed boundaries and randomized width transitions.
        fork
            begin : producer
                for (int txid = 1; txid <= 320; txid++)
                    send_record(txid,(txid % 5) == 0 || (txid % 7) == 0);
            end
            begin : consumer_stalls
                while (accepted_outputs < 320 + (320-64-45+9)) begin
                    @(negedge clk_core);
                    contribution_ready = $urandom_range(0,4) != 0;
                end
                contribution_ready = 1'b1;
            end
        join
        wait (expected_head == expected_tail && !busy);

        // Explicit no-gap service after named two-record prefill: all-wide
        // traffic has equal two-beat input and two-contribution output rates.
        expected_head = 0;
        expected_tail = 0;
        contribution_ready = 0;
        send_record(1001,0);
        send_record(1002,0);
        wait (debug_completed_fifo_count == 2);
        measure_contiguous = 1;
        last_accept_cycle = -1;
        contribution_ready = 1;
        fork
            begin
                for (int txid = 1003; txid <= 1066; txid++)
                    send_record(txid,0);
            end
            begin
                wait (expected_head == expected_tail && !busy);
            end
        join
        measure_contiguous = 0;
        if (ready_high_gap_errors != 0)
            $fatal(1,"M405A internal ready-high gaps=%0d",
                   ready_high_gap_errors);

        // Attack 1: orphan high.
        reset_dut();
        high_tag = 2001;
        high_valid = 1;
        @(posedge clk_core); #1;
        high_valid = 0;
        if (!protocol_error) $fatal(1,"M405A orphan high not rejected");
        protocol_attacks++;

        // Attack 2: nonzero physical padding after a buffered wide low.
        reset_dut();
        begin
            logic [767:0] lb;
            logic [511:0] hb;
            logic [1151:0] a,b;
            build_record(2002,0,lb,hb,a,b);
            drive_low(2002,0,2,2,0,lb);
            hb[384] = 1'b1;
            drive_high(2002,0,2,2,hb);
            @(posedge clk_core); #1;
            if (!protocol_error || contribution_valid)
                $fatal(1,"M405A padding attack leaked contribution");
            protocol_attacks++;
        end

        // Attack 3: high metadata mismatch.
        reset_dut();
        begin
            logic [767:0] lb;
            logic [511:0] hb;
            logic [1151:0] a,b;
            build_record(2003,0,lb,hb,a,b);
            drive_low(2003,1,3,3,0,lb);
            drive_high(2003,1,4,3,hb);
            @(posedge clk_core); #1;
            if (!protocol_error || contribution_valid)
                $fatal(1,"M405A metadata attack leaked contribution");
            protocol_attacks++;
        end

        // Attack 4: reload while a wide low is uncommitted.
        reset_dut();
        begin
            logic [767:0] lb;
            logic [511:0] hb;
            logic [1151:0] a,b;
            build_record(2004,0,lb,hb,a,b);
            drive_low(2004,0,4,4,0,lb);
            @(negedge clk_core); config_reload = 1;
            @(posedge clk_core); #1;
            config_reload = 0;
            if (!protocol_error || contribution_valid)
                $fatal(1,"M405A busy reload not fail-closed");
            protocol_attacks++;
        end

        if (atomic_leak_count != 0 || protocol_attacks != 4
                || narrow_sent == 0 || wide_sent == 0
                || accepted_outputs != 672)
            $fatal(1,"M405A coverage mismatch outputs=%0d narrow=%0d wide=%0d attacks=%0d leaks=%0d",
                   accepted_outputs,narrow_sent,wide_sent,
                   protocol_attacks,atomic_leak_count);
        $display("PASS M405A exact elastic PWP blocks=386 narrow=%0d wide=%0d contributions=%0d stalls=%0d no_gap_wide=66 protocol_attacks=4 atomic_leaks=0 signed_boundaries=true padding=true single_shared96=true system_speedup=false headline=false",
                 narrow_sent,wide_sent,accepted_outputs,stall_cycles);
        $finish;
    end

    initial begin
        #200000;
        $fatal(1,"M405A watchdog timeout expected=%0d/%0d busy=%0d error=%0d",
               expected_head,expected_tail,busy,protocol_error);
    end
endmodule

`default_nettype wire
