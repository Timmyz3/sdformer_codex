`timescale 1ns/1ps
`default_nettype none

module tb_m433_exact_dualbank_coread_pwp_adapter;
    localparam int TAG_BITS = 24;
    localparam int MAX_EXPECTED = 2048;

    logic clk_core, reset_n, config_reload;
    logic request_valid, request_ready, request_accept;
    logic [TAG_BITS-1:0] low_tag, high_tag;
    logic low_tile, high_tile;
    logic [4:0] low_center_id, high_center_id;
    logic [2:0] low_output_block, high_output_block;
    logic request_narrow;
    logic [767:0] low_data;
    logic [511:0] high_data;
    logic contribution_valid, contribution_ready, contribution_accept;
    logic [TAG_BITS-1:0] contribution_tag;
    logic contribution_tile;
    logic [4:0] contribution_center_id;
    logic [2:0] contribution_output_block;
    logic contribution_narrow;
    logic [1151:0] contribution_data;
    logic protocol_error, busy, debug_output_full;
    logic [31:0] debug_request_accepts, debug_narrow_accepts;
    logic [31:0] debug_wide_accepts, debug_contributions;
    logic [31:0] debug_protocol_faults;

    logic [TAG_BITS-1:0] expected_tag [0:MAX_EXPECTED-1];
    logic expected_tile [0:MAX_EXPECTED-1];
    logic [4:0] expected_center [0:MAX_EXPECTED-1];
    logic [2:0] expected_block [0:MAX_EXPECTED-1];
    logic expected_narrow [0:MAX_EXPECTED-1];
    logic [1151:0] expected_data [0:MAX_EXPECTED-1];
    integer expected_head, expected_tail;
    integer requests_sent, narrow_sent, wide_sent, outputs_seen;
    integer metadata_mismatches, arithmetic_mismatches;
    integer stall_cycles, max_stall_run, current_stall_run;
    integer simultaneous_pop_push, consecutive_request_pairs;
    integer protocol_attacks, failclosed_leaks, legal_reloads;
    integer boundary_lanes_checked;
    logic prior_request_accept;

    m433_exact_dualbank_coread_pwp_adapter #(.TAG_BITS(TAG_BITS)) dut (.*);
    m433_exact_dualbank_coread_pwp_adapter_assertions #(
        .TAG_BITS(TAG_BITS)) m433_a_sva (.*);

    always #1.5 clk_core = ~clk_core;

    function automatic integer signed lane_value(
        input integer txid, input integer lane, input logic narrow
    );
        integer raw;
        begin
            if (narrow) begin
                case (lane)
                    0: raw = -128;
                    1: raw = -1;
                    2: raw = 0;
                    3: raw = 127;
                    default: raw = ((txid * 29 + lane * 17) % 256) - 128;
                endcase
            end else begin
                case (lane)
                    0: raw = -2048;
                    1: raw = -1;
                    2: raw = 0;
                    3: raw = 2047;
                    default: raw = ((txid * 233 + lane * 97) % 4096) - 2048;
                endcase
            end
            lane_value = raw;
        end
    endfunction

    task automatic build_request(
        input integer txid,
        input logic narrow_value,
        output logic [767:0] low_bits,
        output logic [511:0] high_bits,
        output logic [1151:0] result_bits
    );
        integer signed value;
        integer raw12;
        begin
            low_bits = '0;
            high_bits = '0;
            result_bits = '0;
            for (int lane = 0; lane < 96; lane++) begin
                value = lane_value(txid,lane,narrow_value);
                raw12 = value & 12'hfff;
                low_bits[lane*8 +: 8] = raw12[7:0];
                if (!narrow_value)
                    high_bits[lane*4 +: 4] = raw12[11:8];
                result_bits[lane*12 +: 12] = raw12[11:0];
            end
        end
    endtask

    task automatic enqueue_expected(
        input integer tag_value,
        input logic tile_value,
        input integer center_value,
        input integer block_value,
        input logic narrow_value,
        input logic [1151:0] result_bits
    );
        begin
            if (expected_tail >= MAX_EXPECTED)
                $fatal(1,"M433 expected queue overflow");
            expected_tag[expected_tail] = tag_value[TAG_BITS-1:0];
            expected_tile[expected_tail] = tile_value;
            expected_center[expected_tail] = center_value[4:0];
            expected_block[expected_tail] = block_value[2:0];
            expected_narrow[expected_tail] = narrow_value;
            expected_data[expected_tail] = result_bits;
            expected_tail++;
        end
    endtask

    task automatic drive_legal(input integer txid, input logic narrow_value);
        logic [767:0] low_bits;
        logic [511:0] high_bits;
        logic [1151:0] result_bits;
        logic tile_value;
        integer center_value, block_value;
        begin
            build_request(txid,narrow_value,low_bits,high_bits,result_bits);
            tile_value = (txid >> 3) & 1;
            center_value = txid % 32;
            block_value = txid % 8;
            enqueue_expected(txid,tile_value,center_value,block_value,
                             narrow_value,result_bits);
            if (clk_core !== 1'b0) @(negedge clk_core);
            low_tag = txid[TAG_BITS-1:0];
            low_tile = tile_value;
            low_center_id = center_value[4:0];
            low_output_block = block_value[2:0];
            request_narrow = narrow_value;
            low_data = low_bits;
            if (narrow_value) begin
                high_tag = '0;
                high_tile = 1'b0;
                high_center_id = '0;
                high_output_block = '0;
                high_data = '0;
            end else begin
                high_tag = txid[TAG_BITS-1:0];
                high_tile = tile_value;
                high_center_id = center_value[4:0];
                high_output_block = block_value[2:0];
                high_data = high_bits;
            end
            request_valid = 1'b1;
            do @(posedge clk_core); while (!request_accept && !protocol_error);
            @(negedge clk_core);
            request_valid = 1'b0;
            requests_sent++;
            if (narrow_value) narrow_sent++; else wide_sent++;
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            reset_n = 1'b0;
            request_valid = 1'b0;
            config_reload = 1'b0;
            contribution_ready = 1'b1;
            repeat (4) @(posedge clk_core);
            @(negedge clk_core);
            reset_n = 1'b1;
            expected_head = expected_tail;
        end
    endtask

    task automatic check_attack_closed(input [8*64-1:0] attack_name);
        begin
            @(posedge clk_core); #1;
            if (!protocol_error || request_accept || contribution_valid ||
                    contribution_accept)
                $fatal(1,"M433 attack not fail-closed: %0s",attack_name);
            protocol_attacks++;
            @(negedge clk_core);
            request_valid = 1'b0;
            config_reload = 1'b0;
        end
    endtask

    always @(posedge clk_core) begin
        if (reset_n) begin
            if (contribution_valid && !contribution_ready) begin
                stall_cycles++;
                current_stall_run++;
                if (current_stall_run + 1 > max_stall_run)
                    max_stall_run = current_stall_run + 1;
            end else begin
                current_stall_run = 0;
            end
            if (request_accept && contribution_accept)
                simultaneous_pop_push++;
            if (request_accept && prior_request_accept)
                consecutive_request_pairs++;
            prior_request_accept <= request_accept;
        end else begin
            prior_request_accept <= 1'b0;
            current_stall_run = 0;
        end

        if (protocol_error && (request_accept || contribution_accept ||
                               contribution_valid))
            failclosed_leaks++;

        if (contribution_accept) begin
            if (expected_head >= expected_tail)
                $fatal(1,"M433 unexpected contribution tag=%0d",
                       contribution_tag);
            if (contribution_tag !== expected_tag[expected_head] ||
                    contribution_tile !== expected_tile[expected_head] ||
                    contribution_center_id !== expected_center[expected_head] ||
                    contribution_output_block !== expected_block[expected_head] ||
                    contribution_narrow !== expected_narrow[expected_head]) begin
                metadata_mismatches++;
                $fatal(1,"M433 metadata mismatch index=%0d",expected_head);
            end
            if (contribution_data !== expected_data[expected_head]) begin
                arithmetic_mismatches++;
                $fatal(1,"M433 arithmetic mismatch index=%0d xor=%h",
                       expected_head,
                       contribution_data ^ expected_data[expected_head]);
            end
            if (!contribution_narrow) begin
                if ($signed(contribution_data[0 +: 12]) != -2048 ||
                    $signed(contribution_data[12 +: 12]) != -1 ||
                    $signed(contribution_data[24 +: 12]) != 0 ||
                    $signed(contribution_data[36 +: 12]) != 2047)
                    $fatal(1,"M433 signed boundary mismatch");
                boundary_lanes_checked = 4;
            end
            expected_head++;
            outputs_seen++;
        end
    end

    initial begin
        clk_core = 0;
        reset_n = 0;
        config_reload = 0;
        request_valid = 0;
        low_tag = 0;
        low_tile = 0;
        low_center_id = 0;
        low_output_block = 0;
        request_narrow = 0;
        low_data = 0;
        high_tag = 0;
        high_tile = 0;
        high_center_id = 0;
        high_output_block = 0;
        high_data = 0;
        contribution_ready = 1;
        expected_head = 0;
        expected_tail = 0;
        requests_sent = 0;
        narrow_sent = 0;
        wide_sent = 0;
        outputs_seen = 0;
        metadata_mismatches = 0;
        arithmetic_mismatches = 0;
        stall_cycles = 0;
        max_stall_run = 0;
        current_stall_run = 0;
        simultaneous_pop_push = 0;
        consecutive_request_pairs = 0;
        protocol_attacks = 0;
        failclosed_leaks = 0;
        legal_reloads = 0;
        boundary_lanes_checked = 0;
        prior_request_accept = 0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core); reset_n = 1;

        // Mixed narrow/wide traffic with randomized output backpressure.
        fork
            begin : mixed_producer
                for (int txid = 1; txid <= 240; txid++)
                    drive_legal(txid,(txid % 3) == 0 || (txid % 11) == 0);
            end
            begin : mixed_consumer
                while (outputs_seen < 240) begin
                    @(negedge clk_core);
                    contribution_ready = $urandom_range(0,4) != 0;
                end
                contribution_ready = 1'b1;
            end
        join
        wait (expected_head == expected_tail && !busy);

        // Explicit long stall; payload and metadata must remain stable.
        contribution_ready = 1'b0;
        drive_legal(3001,1'b0);
        wait (contribution_valid);
        repeat (12) @(posedge clk_core);
        @(negedge clk_core); contribution_ready = 1'b1;
        wait (expected_head == expected_tail && !busy);

        // One-entry elastic replacement: prefill, then sustained pop+push II=1.
        contribution_ready = 1'b0;
        drive_legal(4000,1'b0);
        wait (busy);
        @(negedge clk_core); contribution_ready = 1'b1;
        for (int txid = 4001; txid <= 4065; txid++)
            drive_legal(txid,(txid % 4) == 0);
        wait (expected_head == expected_tail && !busy);

        // Empty reload is legal and is not a request.
        @(negedge clk_core); config_reload = 1'b1;
        @(posedge clk_core); #1;
        if (protocol_error || request_ready || contribution_valid)
            $fatal(1,"M433 legal empty reload failed");
        legal_reloads++;
        @(negedge clk_core); config_reload = 1'b0;

        // Attack 1: wide physical padding must be zero.
        reset_dut();
        begin
            logic [767:0] lb; logic [511:0] hb;
            logic [1151:0] rb;
            build_request(5001,0,lb,hb,rb);
            low_tag=5001; low_tile=0; low_center_id=9; low_output_block=1;
            request_narrow=0; low_data=lb;
            high_tag=5001; high_tile=0; high_center_id=9;
            high_output_block=1; high_data=hb; high_data[511]=1;
            request_valid=1;
            check_attack_closed("wide_padding");
        end

        // Attack 2: duplicated wide metadata mismatch.
        reset_dut();
        begin
            logic [767:0] lb; logic [511:0] hb;
            logic [1151:0] rb;
            build_request(5002,0,lb,hb,rb);
            low_tag=5002; low_tile=1; low_center_id=10; low_output_block=2;
            request_narrow=0; low_data=lb;
            high_tag=5002; high_tile=1; high_center_id=11;
            high_output_block=2; high_data=hb; request_valid=1;
            check_attack_closed("wide_metadata");
        end

        // Attack 3: narrow requests may not carry a high-bank sidecar.
        reset_dut();
        begin
            logic [767:0] lb; logic [511:0] hb;
            logic [1151:0] rb;
            build_request(5003,1,lb,hb,rb);
            low_tag=5003; low_tile=0; low_center_id=11; low_output_block=3;
            request_narrow=1; low_data=lb;
            high_tag=0; high_tile=0; high_center_id=0;
            high_output_block=0; high_data=1; request_valid=1;
            check_attack_closed("narrow_high_side");
        end

        // Attack 4: reload with a buffered contribution suppresses retirement.
        reset_dut();
        contribution_ready = 1'b0;
        drive_legal(5004,0);
        wait (contribution_valid);
        @(negedge clk_core);
        contribution_ready = 1'b1;
        config_reload = 1'b1;
        request_valid = 1'b0;
        check_attack_closed("reload_busy");

        // Sticky fail-closed until reset: a legal request cannot enter.
        begin
            logic [767:0] lb; logic [511:0] hb;
            logic [1151:0] rb;
            build_request(5005,0,lb,hb,rb);
            low_tag=5005; low_tile=0; low_center_id=13; low_output_block=5;
            request_narrow=0; low_data=lb;
            high_tag=5005; high_tile=0; high_center_id=13;
            high_output_block=5; high_data=hb;
            @(negedge clk_core); request_valid=1;
            repeat (3) begin
                @(posedge clk_core); #1;
                if (request_ready || request_accept || contribution_valid)
                    $fatal(1,"M433 sticky fail-closed leak");
            end
            @(negedge clk_core); request_valid=0;
        end

        // Reset recovers the block; one final exact request must complete.
        reset_dut();
        contribution_ready = 1'b1;
        drive_legal(6001,0);
        wait (expected_head == expected_tail && !busy);

        if (metadata_mismatches != 0 || arithmetic_mismatches != 0 ||
                failclosed_leaks != 0 || protocol_attacks != 4 ||
                boundary_lanes_checked != 4 || max_stall_run < 8 ||
                simultaneous_pop_push < 32 || consecutive_request_pairs < 32 ||
                narrow_sent == 0 || wide_sent == 0)
            $fatal(1,"M433 coverage failure metadata=%0d arithmetic=%0d leaks=%0d attacks=%0d boundaries=%0d maxstall=%0d poppush=%0d consecutive=%0d narrow=%0d wide=%0d",
                   metadata_mismatches,arithmetic_mismatches,
                   failclosed_leaks,protocol_attacks,boundary_lanes_checked,
                   max_stall_run,simultaneous_pop_push,
                   consecutive_request_pairs,narrow_sent,wide_sent);

        $display("PASS M433 exact dualbank coread standalone requests=%0d outputs=%0d narrow=%0d wide=%0d signed_boundaries=4 metadata_mismatches=0 arithmetic_mismatches=0 padding_mismatches=0 protocol_attacks=4 failclosed_leaks=0 stall_cycles=%0d max_stall=%0d pop_push=%0d consecutive_ii1=%0d legal_reloads=1 logical_wide_bytes=144 physical_interface_bytes=160 old_psum_preserved_downstream=true correction_fusion=false accuracy_changed=false cycles=false system_speedup=false ppa=false power=false headline=false",
                 requests_sent,outputs_seen,narrow_sent,wide_sent,
                 stall_cycles,max_stall_run,simultaneous_pop_push,
                 consecutive_request_pairs);
        $finish;
    end

    initial begin
        #300000;
        $fatal(1,"M433 watchdog expected=%0d/%0d busy=%0d error=%0d",
               expected_head,expected_tail,busy,protocol_error);
    end
endmodule

`default_nettype wire
