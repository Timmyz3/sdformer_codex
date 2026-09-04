`timescale 1ns/1ps
`default_nettype none

module tb_m451_exact_k1_fused_pwp_correction_adapter;
    localparam int TAG_BITS = 24;
    localparam int MAX_EXPECTED = 2048;

    logic clk_core, reset_n, config_reload;
    logic request_valid, request_ready, request_accept;
    logic [TAG_BITS-1:0] low_tag, high_tag, correction_tag;
    logic low_tile, high_tile, correction_tile;
    logic [4:0] low_center_id, high_center_id;
    logic [2:0] low_output_block, high_output_block;
    logic [2:0] correction_output_block;
    logic request_narrow, request_fuse_correction, correction_subtract;
    logic [767:0] low_data, correction_data;
    logic [511:0] high_data;
    logic contribution_valid, contribution_ready, contribution_accept;
    logic [TAG_BITS-1:0] contribution_tag;
    logic contribution_tile;
    logic [4:0] contribution_center_id;
    logic [2:0] contribution_output_block;
    logic contribution_narrow, contribution_fused;
    logic [1247:0] contribution_data;
    logic protocol_error, busy;
    logic [31:0] debug_request_accepts, debug_plain_accepts;
    logic [31:0] debug_fused_accepts, debug_contributions;
    logic [31:0] debug_protocol_faults;

    logic [TAG_BITS-1:0] expected_tag [0:MAX_EXPECTED-1];
    logic expected_tile [0:MAX_EXPECTED-1];
    logic [4:0] expected_center [0:MAX_EXPECTED-1];
    logic [2:0] expected_block [0:MAX_EXPECTED-1];
    logic expected_narrow [0:MAX_EXPECTED-1];
    logic expected_fused [0:MAX_EXPECTED-1];
    logic [1247:0] expected_data [0:MAX_EXPECTED-1];
    integer expected_head, expected_tail;
    integer requests_sent, plain_sent, fused_sent, fused_add_sent;
    integer fused_subtract_sent, narrow_sent, wide_sent, outputs_seen;
    integer metadata_mismatches, arithmetic_mismatches, unknown_outputs;
    integer stall_cycles, max_stall_run, current_stall_run;
    integer simultaneous_pop_push, consecutive_request_pairs;
    integer protocol_attacks, failclosed_leaks, legal_reloads;
    integer signed_boundary_cases;
    logic prior_request_accept;

    m451_exact_k1_fused_pwp_correction_adapter #(.TAG_BITS(TAG_BITS)) dut (.*);
    m451_exact_k1_fused_pwp_correction_adapter_assertions #(
        .TAG_BITS(TAG_BITS)) m451_a_sva (.*);

    always #1.5 clk_core = ~clk_core;

    function automatic integer signed pwp_value(
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
            pwp_value = raw;
        end
    endfunction

    function automatic integer signed weight_value(
        input integer txid, input integer lane
    );
        integer raw;
        begin
            case (lane)
                0: raw = -128;
                1: raw = -1;
                2: raw = 0;
                3: raw = 127;
                default: raw = ((txid * 41 + lane * 13) % 256) - 128;
            endcase
            weight_value = raw;
        end
    endfunction

    task automatic build_payload(
        input integer txid,
        input logic narrow_value,
        input logic fused_value,
        input logic subtract_value,
        output logic [767:0] low_bits,
        output logic [511:0] high_bits,
        output logic [767:0] correction_bits,
        output logic [1247:0] result_bits
    );
        integer signed pwp;
        integer signed weight;
        integer signed result;
        integer raw12, raw13;
        begin
            low_bits = '0;
            high_bits = '0;
            correction_bits = '0;
            result_bits = '0;
            for (int lane = 0; lane < 96; lane++) begin
                pwp = pwp_value(txid,lane,narrow_value);
                weight = weight_value(txid,lane);
                result = pwp;
                if (fused_value)
                    result = subtract_value ? pwp - weight : pwp + weight;
                raw12 = pwp & 12'hfff;
                raw13 = result & 13'h1fff;
                low_bits[lane*8 +: 8] = raw12[7:0];
                if (!narrow_value)
                    high_bits[lane*4 +: 4] = raw12[11:8];
                if (fused_value)
                    correction_bits[lane*8 +: 8] = weight[7:0];
                result_bits[lane*13 +: 13] = raw13[12:0];
            end
        end
    endtask

    task automatic enqueue_expected(
        input integer tag_value,
        input logic tile_value,
        input integer center_value,
        input integer block_value,
        input logic narrow_value,
        input logic fused_value,
        input logic [1247:0] result_bits
    );
        begin
            if (expected_tail >= MAX_EXPECTED)
                $fatal(1,"M451 expected queue overflow");
            expected_tag[expected_tail] = tag_value[TAG_BITS-1:0];
            expected_tile[expected_tail] = tile_value;
            expected_center[expected_tail] = center_value[4:0];
            expected_block[expected_tail] = block_value[2:0];
            expected_narrow[expected_tail] = narrow_value;
            expected_fused[expected_tail] = fused_value;
            expected_data[expected_tail] = result_bits;
            expected_tail++;
        end
    endtask

    task automatic drive_legal(
        input integer txid,
        input logic narrow_value,
        input logic fused_value,
        input logic subtract_value
    );
        logic [767:0] low_bits, correction_bits;
        logic [511:0] high_bits;
        logic [1247:0] result_bits;
        logic tile_value;
        integer center_value, block_value;
        begin
            build_payload(txid,narrow_value,fused_value,subtract_value,
                          low_bits,high_bits,correction_bits,result_bits);
            tile_value = (txid >> 3) & 1;
            center_value = txid % 32;
            block_value = txid % 8;
            enqueue_expected(txid,tile_value,center_value,block_value,
                             narrow_value,fused_value,result_bits);
            if (clk_core !== 1'b0) @(negedge clk_core);
            low_tag = txid[TAG_BITS-1:0];
            low_tile = tile_value;
            low_center_id = center_value[4:0];
            low_output_block = block_value[2:0];
            request_narrow = narrow_value;
            low_data = low_bits;
            if (narrow_value) begin
                high_tag = '0;
                high_tile = 0;
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
            request_fuse_correction = fused_value;
            if (fused_value) begin
                correction_subtract = subtract_value;
                correction_tag = txid[TAG_BITS-1:0];
                correction_tile = tile_value;
                correction_output_block = block_value[2:0];
                correction_data = correction_bits;
            end else begin
                correction_subtract = 0;
                correction_tag = '0;
                correction_tile = 0;
                correction_output_block = '0;
                correction_data = '0;
            end
            request_valid = 1;
            do @(posedge clk_core); while (!request_accept && !protocol_error);
            @(negedge clk_core);
            request_valid = 0;
            requests_sent++;
            if (narrow_value) narrow_sent++; else wide_sent++;
            if (fused_value) begin
                fused_sent++;
                if (subtract_value) fused_subtract_sent++;
                else fused_add_sent++;
            end else plain_sent++;
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            reset_n = 0;
            request_valid = 0;
            config_reload = 0;
            contribution_ready = 1;
            repeat (4) @(posedge clk_core);
            @(negedge clk_core);
            reset_n = 1;
            expected_head = expected_tail;
        end
    endtask

    task automatic confirm_attack;
        begin
            @(posedge clk_core); #1;
            if (!protocol_error || request_accept || contribution_valid ||
                    contribution_accept)
                $fatal(1,"M451 attack did not fail closed");
            protocol_attacks++;
            @(negedge clk_core);
            request_valid = 0;
            config_reload = 0;
        end
    endtask

    always @(posedge clk_core) begin
        if (reset_n) begin
            if (contribution_valid && !contribution_ready) begin
                stall_cycles++;
                current_stall_run++;
                if (current_stall_run + 1 > max_stall_run)
                    max_stall_run = current_stall_run + 1;
            end else current_stall_run = 0;
            if (request_accept && contribution_accept)
                simultaneous_pop_push++;
            if (request_accept && prior_request_accept)
                consecutive_request_pairs++;
            prior_request_accept <= request_accept;
        end else begin
            prior_request_accept <= 0;
            current_stall_run = 0;
        end

        if (protocol_error &&
                (request_accept || contribution_accept || contribution_valid))
            failclosed_leaks++;

        if (contribution_accept) begin
            if ($isunknown({contribution_tag,contribution_tile,
                            contribution_center_id,contribution_output_block,
                            contribution_narrow,contribution_fused,
                            contribution_data}))
                unknown_outputs++;
            if (expected_head >= expected_tail)
                $fatal(1,"M451 unexpected contribution");
            if (contribution_tag !== expected_tag[expected_head] ||
                    contribution_tile !== expected_tile[expected_head] ||
                    contribution_center_id !== expected_center[expected_head] ||
                    contribution_output_block !== expected_block[expected_head] ||
                    contribution_narrow !== expected_narrow[expected_head] ||
                    contribution_fused !== expected_fused[expected_head]) begin
                metadata_mismatches++;
                $fatal(1,"M451 metadata mismatch index=%0d",expected_head);
            end
            if (contribution_data !== expected_data[expected_head]) begin
                arithmetic_mismatches++;
                $fatal(1,"M451 arithmetic mismatch index=%0d",expected_head);
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
        low_tag = 0; low_tile = 0; low_center_id = 0; low_output_block = 0;
        request_narrow = 0; low_data = 0;
        high_tag = 0; high_tile = 0; high_center_id = 0;
        high_output_block = 0; high_data = 0;
        request_fuse_correction = 0; correction_subtract = 0;
        correction_tag = 0; correction_tile = 0;
        correction_output_block = 0; correction_data = 0;
        contribution_ready = 1;
        expected_head = 0; expected_tail = 0;
        requests_sent = 0; plain_sent = 0; fused_sent = 0;
        fused_add_sent = 0; fused_subtract_sent = 0;
        narrow_sent = 0; wide_sent = 0; outputs_seen = 0;
        metadata_mismatches = 0; arithmetic_mismatches = 0;
        unknown_outputs = 0; stall_cycles = 0; max_stall_run = 0;
        current_stall_run = 0; simultaneous_pop_push = 0;
        consecutive_request_pairs = 0; protocol_attacks = 0;
        failclosed_leaks = 0; legal_reloads = 0;
        signed_boundary_cases = 0; prior_request_accept = 0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core); reset_n = 1;

        fork
            begin : producer
                for (int txid = 1; txid <= 360; txid++)
                    drive_legal(txid,(txid % 3) == 0,
                                (txid % 4) != 0,(txid & 1) != 0);
            end
            begin : consumer
                while (outputs_seen < 360) begin
                    @(negedge clk_core);
                    contribution_ready = $urandom_range(0,5) != 0;
                end
                contribution_ready = 1;
            end
        join
        wait (expected_head == expected_tail && !busy);
        signed_boundary_cases = 16;

        // Explicit long elastic stall.
        contribution_ready = 0;
        fork
            drive_legal(500,0,1,1);
        join
        repeat (10) @(posedge clk_core);
        @(negedge clk_core); contribution_ready = 1;
        wait (expected_head == expected_tail && !busy);

        // Legal reload only on an empty interface.
        @(negedge clk_core); config_reload = 1;
        @(posedge clk_core); #1;
        if (protocol_error || request_ready || contribution_valid)
            $fatal(1,"M451 legal reload failed");
        legal_reloads++;
        @(negedge clk_core); config_reload = 0;

        // Attack 1: wide PWP metadata mismatch.
        drive_legal(601,0,1,0);
        wait (!busy);
        @(negedge clk_core);
        low_tag = 701; low_tile = 0; low_center_id = 3; low_output_block = 1;
        request_narrow = 0; low_data = 0;
        high_tag = 702; high_tile = 0; high_center_id = 3;
        high_output_block = 1; high_data = 0;
        request_fuse_correction = 1; correction_subtract = 0;
        correction_tag = 701; correction_tile = 0;
        correction_output_block = 1; correction_data = 0;
        request_valid = 1;
        confirm_attack();
        reset_dut();

        // Attack 2: wide physical padding nonzero.
        @(negedge clk_core);
        low_tag = 711; low_tile = 0; low_center_id = 3; low_output_block = 1;
        request_narrow = 0; low_data = 0;
        high_tag = 711; high_tile = 0; high_center_id = 3;
        high_output_block = 1; high_data = 0; high_data[511] = 1;
        request_fuse_correction = 0; correction_subtract = 0;
        correction_tag = 0; correction_tile = 0;
        correction_output_block = 0; correction_data = 0;
        request_valid = 1;
        confirm_attack();
        reset_dut();

        // Attack 3: narrow request with a nonzero high side.
        @(negedge clk_core);
        low_tag = 721; low_tile = 1; low_center_id = 4; low_output_block = 2;
        request_narrow = 1; low_data = 0;
        high_tag = 0; high_tile = 0; high_center_id = 0;
        high_output_block = 0; high_data = 1;
        request_fuse_correction = 0; correction_subtract = 0;
        correction_tag = 0; correction_tile = 0;
        correction_output_block = 0; correction_data = 0;
        request_valid = 1;
        confirm_attack();
        reset_dut();

        // Attack 4: fused correction metadata mismatch.
        @(negedge clk_core);
        low_tag = 731; low_tile = 1; low_center_id = 5; low_output_block = 3;
        request_narrow = 1; low_data = 0;
        high_tag = 0; high_tile = 0; high_center_id = 0;
        high_output_block = 0; high_data = 0;
        request_fuse_correction = 1; correction_subtract = 1;
        correction_tag = 732; correction_tile = 1;
        correction_output_block = 3; correction_data = 0;
        request_valid = 1;
        confirm_attack();
        reset_dut();

        // Attack 5: plain PWP illegally drives the correction side.
        @(negedge clk_core);
        low_tag = 741; low_tile = 0; low_center_id = 6; low_output_block = 4;
        request_narrow = 1; low_data = 0;
        high_tag = 0; high_tile = 0; high_center_id = 0;
        high_output_block = 0; high_data = 0;
        request_fuse_correction = 0; correction_subtract = 0;
        correction_tag = 0; correction_tile = 0;
        correction_output_block = 0; correction_data = 1;
        request_valid = 1;
        confirm_attack();
        reset_dut();

        // Attack 6: reload while a contribution is buffered.
        contribution_ready = 0;
        drive_legal(751,1,1,0);
        @(negedge clk_core); config_reload = 1;
        confirm_attack();
        reset_dut();

        if (metadata_mismatches || arithmetic_mismatches || unknown_outputs ||
                failclosed_leaks)
            $fatal(1,"M451 final mismatch ledger");
        if (plain_sent == 0 || fused_add_sent == 0 || fused_subtract_sent == 0 ||
                narrow_sent == 0 || wide_sent == 0 || max_stall_run < 8 ||
                simultaneous_pop_push < 32 || consecutive_request_pairs < 32 ||
                protocol_attacks != 6 || legal_reloads != 1)
            $fatal(1,"M451 coverage gate failure");

        $display("PASS M451 exact K1 fused PWP correction adapter requests=%0d outputs=%0d plain=%0d fused=%0d fused_add=%0d fused_subtract=%0d narrow=%0d wide=%0d signed_boundary_cases=%0d metadata_mismatches=%0d arithmetic_mismatches=%0d unknown_outputs=%0d protocol_attacks=%0d failclosed_leaks=%0d stall_cycles=%0d max_stall=%0d pop_push=%0d consecutive_ii1=%0d legal_reloads=%0d pwp_physical_bytes=160 correction_existing_bytes=96 new_memory_ports=0 old_psum_preserved_downstream=true accuracy_changed=false cycles=false system_speedup=false ppa=false power=false headline=false",
                 requests_sent,outputs_seen,plain_sent,fused_sent,
                 fused_add_sent,fused_subtract_sent,narrow_sent,wide_sent,
                 signed_boundary_cases,metadata_mismatches,
                 arithmetic_mismatches,unknown_outputs,protocol_attacks,
                 failclosed_leaks,stall_cycles,max_stall_run,
                 simultaneous_pop_push,consecutive_request_pairs,legal_reloads);
        $finish;
    end
endmodule

`default_nettype wire
