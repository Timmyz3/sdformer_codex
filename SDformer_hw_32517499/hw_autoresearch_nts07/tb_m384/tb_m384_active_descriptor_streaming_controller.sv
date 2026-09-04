`timescale 1ns/1ps
`default_nettype none

module tb_m384_active_descriptor_streaming_controller;
    localparam int TAG_BITS = 24;
    logic clk_core, reset_n;
    logic config_reload, config_reload_accept;
    logic phase_valid, phase_ready, phase_accept;
    logic [TAG_BITS-1:0] phase_tag;
    logic phase_bank;
    logic [511:0] phase_centers_q16;
    logic row_valid, row_ready, row_accept;
    logic [11:0] row_id;
    logic [15:0] row_original;
    logic [6:0] row_center_id;
    logic [4:0] row_distance;
    logic row_use_pwp, row_last;
    logic descriptor_write_valid, descriptor_write_ready;
    logic descriptor_write_accept;
    logic [TAG_BITS-1:0] descriptor_write_tag;
    logic descriptor_write_bank;
    logic [11:0] descriptor_write_address;
    logic [47:0] descriptor_write_data;
    logic phase_seal_valid, phase_seal_ready, phase_seal_accept;
    logic [TAG_BITS-1:0] phase_seal_tag;
    logic phase_seal_bank;
    logic [11:0] phase_seal_active_count;
    logic [31:0] phase_seal_used_center_bitmap;
    logic phase_seal_empty;
    logic pwp_run_valid, pwp_run_ready, pwp_run_accept;
    logic [4:0] pwp_run_start_center;
    logic [5:0] pwp_run_length_centers;
    logic [15:0] pwp_run_tile0_address, pwp_run_tile1_address;
    logic [15:0] pwp_run_bytes;
    logic pwp_run_last;
    logic tile1_prefetch_valid, tile1_prefetch_ready;
    logic tile1_prefetch_accept;
    logic [TAG_BITS-1:0] tile1_prefetch_tag;
    logic tile1_prefetch_bank;
    logic [15:0] tile1_prefetch_weight_address;
    logic [15:0] tile1_prefetch_pwp_base_address;
    logic [31:0] tile1_prefetch_used_center_bitmap;
    logic tile1_prefetch_done_valid, tile1_prefetch_done_ready;
    logic tile1_prefetch_done_accept;
    logic [TAG_BITS-1:0] tile1_prefetch_done_tag;
    logic tile1_prefetch_done_bank;
    logic replay_start_valid, replay_start_ready, replay_start_accept;
    logic replay_start_tile;
    logic descriptor_read_req_valid, descriptor_read_req_ready;
    logic descriptor_read_req_accept;
    logic [TAG_BITS-1:0] descriptor_read_req_tag;
    logic descriptor_read_req_bank;
    logic [11:0] descriptor_read_req_address;
    logic descriptor_read_rsp_valid, descriptor_read_rsp_ready;
    logic descriptor_read_rsp_accept;
    logic [TAG_BITS-1:0] descriptor_read_rsp_tag;
    logic descriptor_read_rsp_bank;
    logic [11:0] descriptor_read_rsp_address;
    logic [47:0] descriptor_read_rsp_data;
    logic bundle_valid, bundle_ready, bundle_accept;
    logic [TAG_BITS-1:0] bundle_tag;
    logic bundle_tile;
    logic [11:0] bundle_row_id;
    logic [15:0] bundle_original;
    logic [6:0] bundle_center_id;
    logic [15:0] bundle_center;
    logic [4:0] bundle_distance;
    logic bundle_use_pwp, bundle_fallback_bit_sparse;
    logic [15:0] bundle_plus_mask, bundle_minus_mask;
    logic replay_done_valid, replay_done_ready, replay_done_accept;
    logic [TAG_BITS-1:0] replay_done_tag;
    logic replay_done_tile;
    logic [11:0] replay_done_count;
    logic phase_done_valid, phase_done_ready, phase_done_accept;
    logic [TAG_BITS-1:0] phase_done_tag;
    logic [11:0] phase_done_active_count;
    logic [31:0] phase_done_used_center_bitmap;
    logic phase_done_empty;
    logic protocol_error, busy;
    logic [3:0] debug_state;
    logic [11:0] debug_rows_accepted, debug_active_count;
    logic [3:0] debug_fifo_occupancy, debug_outstanding_reads;
    logic [3:0] debug_credit_used;
    logic [1:0] debug_replays_completed;
    logic [31:0] debug_descriptor_writes, debug_descriptor_requests;
    logic [31:0] debug_descriptor_responses, debug_bundle_accepts;
    logic [31:0] debug_pwp_runs_issued;

    logic [47:0] descriptor_store [0:1][0:2999];
    logic [TAG_BITS-1:0] request_tag_queue [0:4095];
    logic request_bank_queue [0:4095];
    logic [11:0] request_address_queue [0:4095];
    integer request_due_queue [0:4095];
    integer memory_head, memory_tail, memory_cycle, memory_latency;
    logic memory_flush, response_enable;
    logic corrupt_response_flags, corrupt_response_address;
    logic inject_unexpected_response;

    integer normal_phases, normal_replays, checked_bundles;
    integer checked_pwp_runs;
    integer zero_rows, active_rows, pop1_rows, pwp_rows, fallback_rows;
    integer write_stalls, request_stalls, response_stalls, backend_stalls;
    integer protocol_attacks, sticky_cycles;
    integer max_fifo, max_outstanding, max_credit;
    integer latency_mask;
    integer account_replay, state_reset_mask;
    integer prefetch_starts, prefetch_dones;

    m384_active_descriptor_streaming_controller #(
        .TAG_BITS(TAG_BITS), .FIFO_DEPTH(8)
    ) dut (.*);

    m384_active_descriptor_streaming_controller_assertions #(
        .TAG_BITS(TAG_BITS)
    ) sva (.*);

    initial clk_core = 1'b0;
    always #1.5 clk_core = ~clk_core;

    function automatic [4:0] pc16(input logic [15:0] value);
        integer bit_index;
        begin
            pc16 = 0;
            for (bit_index=0; bit_index<16; bit_index=bit_index+1)
                pc16 = pc16 + value[bit_index];
        end
    endfunction

    function automatic [15:0] center_value(input integer index);
        begin
            case (index)
                0: center_value = 16'haa55;
                1: center_value = 16'h00ff;
                2: center_value = 16'h0f0f;
                3: center_value = 16'h3333;
                default: center_value = (16'h00ff << (index % 8))
                    | (16'h00ff >> (8-(index % 8)));
            endcase
        end
    endfunction

    task automatic make_centers;
        integer index;
        begin
            phase_centers_q16 = '0;
            for (index=0; index<32; index=index+1)
                phase_centers_q16[index*16 +: 16] = center_value(index);
        end
    endtask

    task automatic clear_drivers;
        begin
            config_reload = 0;
            phase_valid = 0;
            phase_tag = 0;
            phase_bank = 0;
            phase_centers_q16 = 0;
            row_valid = 0;
            row_id = 0;
            row_original = 0;
            row_center_id = 0;
            row_distance = 0;
            row_use_pwp = 0;
            row_last = 0;
            descriptor_write_ready = 1;
            phase_seal_ready = 0;
            pwp_run_ready = 0;
            tile1_prefetch_ready = 1;
            tile1_prefetch_done_valid = 0;
            tile1_prefetch_done_tag = 0;
            tile1_prefetch_done_bank = 0;
            replay_start_valid = 0;
            replay_start_tile = 0;
            descriptor_read_req_ready = 1;
            bundle_ready = 0;
            replay_done_ready = 0;
            phase_done_ready = 0;
            memory_flush = 0;
            response_enable = 1;
            corrupt_response_flags = 0;
            corrupt_response_address = 0;
            inject_unexpected_response = 0;
            memory_latency = 1;
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            if (reset_n === 1'b1)
                state_reset_mask = state_reset_mask | (1 << debug_state);
            clear_drivers();
            reset_n = 0;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            reset_n = 1;
            repeat (2) @(posedge clk_core);
            if (protocol_error || busy)
                $fatal(1,"reset failed");
        end
    endtask

    always @(posedge clk_core) begin
        if (reset_n) begin
            if (descriptor_write_accept)
                descriptor_store[descriptor_write_bank]
                    [descriptor_write_address] <= descriptor_write_data;
            if (debug_fifo_occupancy > max_fifo)
                max_fifo <= debug_fifo_occupancy;
            if (debug_outstanding_reads > max_outstanding)
                max_outstanding <= debug_outstanding_reads;
            if (debug_credit_used > max_credit)
                max_credit <= debug_credit_used;
            if (tile1_prefetch_accept)
                prefetch_starts <= prefetch_starts + 1;
            if (tile1_prefetch_done_accept)
                prefetch_dones <= prefetch_dones + 1;
        end
    end

    // In-order external SRAM with configurable L1..L8 first-response latency.
    // Response valid holds under backpressure.  Requests and responses remain
    // independently backpressurable; the DUT owns the eight-credit invariant.
    always @(posedge clk_core or negedge reset_n) begin : descriptor_memory
        integer next_head;
        if (!reset_n) begin
            descriptor_read_rsp_valid <= 0;
            descriptor_read_rsp_tag <= 0;
            descriptor_read_rsp_bank <= 0;
            descriptor_read_rsp_address <= 0;
            descriptor_read_rsp_data <= 0;
            memory_head <= 0;
            memory_tail <= 0;
            memory_cycle <= 0;
        end else if (memory_flush) begin
            descriptor_read_rsp_valid <= 0;
            memory_head <= 0;
            memory_tail <= 0;
            memory_cycle <= 0;
        end else begin
            memory_cycle <= memory_cycle + 1;
            if (descriptor_read_req_accept) begin
                request_tag_queue[memory_tail] <= descriptor_read_req_tag;
                request_bank_queue[memory_tail] <= descriptor_read_req_bank;
                request_address_queue[memory_tail] <=
                    descriptor_read_req_address;
                request_due_queue[memory_tail] <= memory_cycle + memory_latency;
                memory_tail <= memory_tail + 1;
            end

            next_head = memory_head;
            if (descriptor_read_rsp_valid && descriptor_read_rsp_ready) begin
                descriptor_read_rsp_valid <= 0;
                next_head = memory_head + 1;
                memory_head <= memory_head + 1;
            end
            if ((!descriptor_read_rsp_valid
                 || (descriptor_read_rsp_valid && descriptor_read_rsp_ready))
                && response_enable && next_head < memory_tail
                && request_due_queue[next_head] <= memory_cycle) begin
                descriptor_read_rsp_valid <= 1;
                descriptor_read_rsp_tag <= request_tag_queue[next_head];
                descriptor_read_rsp_bank <= request_bank_queue[next_head];
                descriptor_read_rsp_address <=
                    request_address_queue[next_head]
                    + (corrupt_response_address ? 1'b1 : 1'b0);
                descriptor_read_rsp_data <= descriptor_store
                    [request_bank_queue[next_head]]
                    [request_address_queue[next_head]]
                    ^ (corrupt_response_flags ? 48'h800000000000 : 48'b0);
            end else if (!descriptor_read_rsp_valid
                         && inject_unexpected_response) begin
                descriptor_read_rsp_valid <= 1;
                descriptor_read_rsp_tag <= phase_tag;
                descriptor_read_rsp_bank <= phase_bank;
                descriptor_read_rsp_address <= 0;
                descriptor_read_rsp_data <= 48'h000000001001;
            end
        end
    end

    task automatic idle_reload;
        begin
            @(negedge clk_core);
            config_reload = 1;
            @(posedge clk_core);
            if (!config_reload_accept) $fatal(1,"idle reload not accepted");
            @(negedge clk_core);
            config_reload = 0;
        end
    endtask

    task automatic start_phase(input integer tag_value, input integer bank_value);
        begin
            @(negedge clk_core);
            make_centers();
            phase_tag = tag_value;
            phase_bank = bank_value;
            phase_valid = 1;
            do @(posedge clk_core); while (!phase_accept);
            @(negedge clk_core);
            phase_valid = 0;
        end
    endtask

    task automatic choose_row(
        input integer source_row,
        input integer active_target,
        output logic [15:0] original,
        output logic [6:0] center_id_value,
        output logic [4:0] distance_value,
        output logic use_value
    );
        integer kind;
        logic [15:0] center;
        begin
            if (source_row < 3000-active_target) begin
                original = 0;
                center_id_value = 0;
                center = center_value(0);
            end else if (active_target == 3000) begin
                active_rows = active_rows + 1;
                center_id_value = source_row % 32;
                center = center_value(center_id_value);
                original = center;
            end else begin
                active_rows = active_rows + 1;
                kind = (active_target == 1) ? 1 : (source_row % 4);
                case (kind)
                    0: begin
                        original = 16'h0001 << (source_row % 16);
                        center_id_value = 0;
                        pop1_rows = pop1_rows + 1;
                    end
                    1: begin
                        center_id_value = (source_row & 4) ? 4 : 2;
                        original = center_value(center_id_value);
                    end
                    2: begin
                        original = 16'h01fe;
                        center_id_value = 1;
                    end
                    default: begin
                        original = 16'h0003;
                        center_id_value = 0;
                    end
                endcase
                center = center_value(center_id_value);
            end
            distance_value = pc16(original ^ center);
            use_value = ({1'b0,distance_value}+1) < pc16(original);
            if (original == 0)
                zero_rows = zero_rows + 1;
            else if (use_value)
                pwp_rows = pwp_rows + 1;
            else
                fallback_rows = fallback_rows + 1;
        end
    endtask

    task automatic send_rows(input integer active_target);
        integer source_row;
        logic [15:0] original;
        logic [6:0] center_id_value;
        logic [4:0] distance_value;
        logic use_value;
        logic accepted;
        begin
            for (source_row=0; source_row<3000; source_row=source_row+1) begin
                choose_row(source_row,active_target,original,center_id_value,
                           distance_value,use_value);
                @(negedge clk_core);
                row_id = source_row;
                row_original = original;
                row_center_id = center_id_value;
                row_distance = distance_value;
                row_use_pwp = use_value;
                row_last = source_row == 2999;
                row_valid = 1;
                descriptor_write_ready = (original == 0)
                    ? $urandom_range(0,1) : ($urandom_range(0,4) != 0);
                accepted = 0;
                while (!accepted) begin
                    @(posedge clk_core);
                    if (row_accept) begin
                        accepted = 1;
                    end else if (protocol_error) begin
                        $fatal(1,"fault in legal row stream state=%0d id=%0d expected=%0d original=%h center=%0d distance=%0d use=%0d last=%0d ready=%0d",
                               debug_state,row_id,debug_rows_accepted,
                               row_original,row_center_id,row_distance,
                               row_use_pwp,row_last,descriptor_write_ready);
                    end
                    if (!accepted) begin
                        if (row_valid && !row_ready)
                            write_stalls = write_stalls+1;
                        @(negedge clk_core);
                        descriptor_write_ready = $urandom_range(0,4) != 0;
                    end
                end
                @(negedge clk_core);
                row_valid = 0;
            end
        end
    endtask

    task automatic accept_seal(input integer expected_active);
        logic [31:0] expected_bitmap;
        logic [31:0] expected_remaining;
        integer address;
        integer start_center, run_length, center_index;
        logic found, open_run, run_accepted;
        begin
            expected_bitmap = 0;
            for (address=0; address<expected_active; address=address+1)
                if (descriptor_store[phase_bank][address][40])
                    expected_bitmap[
                        descriptor_store[phase_bank][address][34:28]] = 1;
            @(negedge clk_core);
            phase_seal_ready = 1;
            @(posedge clk_core);
            if (!phase_seal_accept) $fatal(1,"seal not accepted");
            if (phase_seal_active_count !== expected_active)
                $fatal(1,"seal active count mismatch");
            if (phase_seal_used_center_bitmap !== expected_bitmap)
                $fatal(1,"seal bitmap mismatch got=%h expected=%h",
                       phase_seal_used_center_bitmap,expected_bitmap);
            if (phase_seal_empty !== (expected_active==0))
                $fatal(1,"seal empty mismatch");
            @(negedge clk_core);
            phase_seal_ready = 0;
            expected_remaining = expected_bitmap;
            while (expected_remaining != 0) begin
                start_center = 0;
                run_length = 0;
                found = 0;
                open_run = 0;
                for (center_index=0; center_index<32;
                     center_index=center_index+1) begin
                    if (!found && expected_remaining[center_index]) begin
                        start_center = center_index;
                        found = 1;
                        open_run = 1;
                    end
                    if (found && open_run && center_index>=start_center) begin
                        if (expected_remaining[center_index])
                            run_length = run_length + 1;
                        else
                            open_run = 0;
                    end
                end
                pwp_run_ready = $urandom_range(0,2) != 0;
                run_accepted = 0;
                while (!run_accepted) begin
                    @(posedge clk_core);
                    if (pwp_run_accept) begin
                        if (pwp_run_start_center != start_center
                            || pwp_run_length_centers != run_length
                            || pwp_run_tile0_address !=
                               6240+start_center*640
                            || pwp_run_tile1_address !=
                               38912+start_center*640
                            || pwp_run_bytes != run_length*640)
                            $fatal(1,"PWP run/address mismatch");
                        if (pwp_run_last !=
                            ((expected_remaining >>
                              (start_center+run_length)) == 0))
                            $fatal(1,"PWP run last mismatch");
                        run_accepted = 1;
                    end else begin
                        @(negedge clk_core);
                        pwp_run_ready = $urandom_range(0,2) != 0;
                    end
                end
                for (center_index=start_center;
                     center_index<start_center+run_length;
                     center_index=center_index+1)
                    expected_remaining[center_index] = 0;
                checked_pwp_runs = checked_pwp_runs + 1;
                @(negedge clk_core);
                pwp_run_ready = 0;
            end
        end
    endtask

    task automatic flush_memory;
        begin
            @(negedge clk_core);
            memory_flush = 1;
            @(posedge clk_core);
            @(negedge clk_core);
            memory_flush = 0;
        end
    endtask

    task automatic run_replay(
        input integer tile_value,
        input integer active_target,
        input integer latency_value,
        input integer pressure_mode
    );
        integer expected_address, local_cycles, req_seen, req_stall_left;
        logic [47:0] expected;
        logic [15:0] expected_center, expected_plus, expected_minus;
        begin
            flush_memory();
            memory_latency = latency_value;
            latency_mask = latency_mask | (1 << latency_value);
            response_enable = 1;
            descriptor_read_req_ready = 1;
            bundle_ready = pressure_mode ? 0 : 1;
            replay_done_ready = 0;
            @(negedge clk_core);
            replay_start_tile = tile_value;
            replay_start_valid = 1;
            do @(posedge clk_core); while (!replay_start_accept);
            @(negedge clk_core);
            replay_start_valid = 0;
            if (tile_value == 0) begin
                tile1_prefetch_done_tag = phase_tag;
                tile1_prefetch_done_bank = phase_bank;
                tile1_prefetch_done_valid = 1;
                @(posedge clk_core);
                if (!tile1_prefetch_done_accept)
                    $fatal(1,"tile1 prefetch completion not accepted");
                @(negedge clk_core);
                tile1_prefetch_done_valid = 0;
            end
            expected_address = 0;
            local_cycles = 0;
            req_seen = 0;
            req_stall_left = 0;
            while (!replay_done_valid) begin
                @(posedge clk_core);
                local_cycles = local_cycles + 1;
                if (descriptor_read_req_accept) begin
                    req_seen = req_seen + 1;
                    if (pressure_mode && req_seen == 32)
                        req_stall_left = 16;
                end
                if (bundle_accept) begin
                    if (expected_address >= active_target)
                        $fatal(1,"bundle overrun");
                    expected = descriptor_store[phase_bank][expected_address];
                    expected_center = center_value(expected[34:28]);
                    expected_plus = expected[40]
                        ? (expected[27:12] & ~expected_center)
                        : expected[27:12];
                    expected_minus = expected[40]
                        ? (expected_center & ~expected[27:12]) : 0;
                    if ({bundle_use_pwp,bundle_distance,bundle_center_id,
                         bundle_original,bundle_row_id} !== expected[40:0])
                        $fatal(1,"bundle descriptor mismatch addr=%0d",
                               expected_address);
                    if (bundle_center !== expected_center
                        || bundle_plus_mask !== expected_plus
                        || bundle_minus_mask !== expected_minus)
                        $fatal(1,"bundle exact reconstruction mismatch");
                    if (bundle_tile !== tile_value)
                        $fatal(1,"bundle tile mismatch");
                    expected_address = expected_address + 1;
                    if (account_replay)
                        checked_bundles = checked_bundles + 1;
                end
                if (descriptor_read_req_valid && !descriptor_read_req_ready)
                    request_stalls = request_stalls + 1;
                if (descriptor_read_rsp_valid && !descriptor_read_rsp_ready)
                    response_stalls = response_stalls + 1;
                if (bundle_valid && !bundle_ready)
                    backend_stalls = backend_stalls + 1;
                if (local_cycles > active_target*20+5000)
                    $fatal(1,"replay watchdog");
                @(negedge clk_core);
                if (pressure_mode && local_cycles < 30)
                    bundle_ready = 0;
                else
                    bundle_ready = $urandom_range(0,7) != 0;
                if (req_stall_left > 0) begin
                    descriptor_read_req_ready = 0;
                    req_stall_left = req_stall_left - 1;
                end else begin
                    descriptor_read_req_ready = $urandom_range(0,7) != 0;
                end
                response_enable = $urandom_range(0,15) != 0;
            end
            if (expected_address != active_target)
                $fatal(1,"replay under-run got=%0d expected=%0d",
                       expected_address,active_target);
            if (replay_done_count != active_target
                || replay_done_tile != tile_value)
                $fatal(1,"replay done mismatch");
            @(negedge clk_core);
            replay_done_ready = 1;
            @(posedge clk_core);
            if (!replay_done_accept) $fatal(1,"done not accepted");
            @(negedge clk_core);
            replay_done_ready = 0;
            bundle_ready = 0;
            if (account_replay)
                normal_replays = normal_replays + 1;
        end
    endtask

    task automatic accept_phase_done(input integer expected_active);
        begin
            @(negedge clk_core);
            phase_done_ready = 1;
            @(posedge clk_core);
            if (!phase_done_accept || phase_done_active_count != expected_active)
                $fatal(1,"phase done mismatch");
            @(negedge clk_core);
            phase_done_ready = 0;
            normal_phases = normal_phases + 1;
        end
    endtask

    task automatic run_phase(
        input integer tag_value,
        input integer bank_value,
        input integer active_target,
        input integer latency0,
        input integer latency1,
        input integer pressure_mode
    );
        begin
            start_phase(tag_value,bank_value);
            send_rows(active_target);
            accept_seal(active_target);
            if (active_target != 0) begin
                run_replay(0,active_target,latency0,pressure_mode);
                run_replay(1,active_target,latency1,pressure_mode);
            end
            accept_phase_done(active_target);
            if (protocol_error) $fatal(1,"legal phase faulted");
        end
    endtask

    task automatic expect_sticky_fault;
        integer cycle_index;
        begin
            @(posedge clk_core);
            if (!protocol_error)
                $fatal(1,"expected protocol fault absent state=%0d rspv=%0d rspr=%0d rspa=%0d rspaddr=%0d rspdata=%h corrupt_flags=%0d corrupt_addr=%0d replayv=%0d tile=%0d",
                       debug_state,descriptor_read_rsp_valid,
                       descriptor_read_rsp_ready,descriptor_read_rsp_accept,
                       descriptor_read_rsp_address,descriptor_read_rsp_data,
                       corrupt_response_flags,corrupt_response_address,
                       replay_start_valid,replay_start_tile);
            protocol_attacks = protocol_attacks + 1;
            @(negedge clk_core);
            phase_valid = 1;
            config_reload = 1;
            row_valid = 1;
            replay_start_valid = 1;
            for (cycle_index=0; cycle_index<10; cycle_index=cycle_index+1) begin
                @(posedge clk_core);
                if (!protocol_error || phase_accept || config_reload_accept
                    || row_accept || replay_start_accept
                    || descriptor_write_accept || descriptor_read_req_accept
                    || bundle_accept || phase_done_accept)
                    $fatal(1,"sticky fail-close violation");
                sticky_cycles = sticky_cycles + 1;
            end
            @(negedge clk_core);
            phase_valid = 0;
            config_reload = 0;
            row_valid = 0;
            replay_start_valid = 0;
        end
    endtask

    task automatic attack_bad_row(input integer attack_kind);
        begin
            reset_dut();
            start_phase(24'h800000+attack_kind,attack_kind&1);
            @(negedge clk_core);
            row_valid = 1;
            row_id = attack_kind==0 ? 12'd1 : 12'd0;
            row_original = attack_kind==3 ? 16'h0001 : 16'h0f0f;
            row_center_id = attack_kind==1 ? 7'd32 :
                (attack_kind==3 ? 7'd0 : 7'd2);
            row_distance = attack_kind==2 ? 5'd1 :
                pc16(row_original ^ center_value(row_center_id[4:0]));
            row_use_pwp = attack_kind==3 ? 1'b1 :
                (({1'b0,row_distance}+1) < pc16(row_original));
            row_last = attack_kind==4;
            descriptor_write_ready = 1;
            expect_sticky_fault();
        end
    endtask

    task automatic setup_active_one(input integer tag_value);
        begin
            start_phase(tag_value,tag_value&1);
            send_rows(1);
            accept_seal(1);
        end
    endtask

    task automatic attack_wrong_tile;
        begin
            reset_dut();
            setup_active_one(24'h900001);
            @(negedge clk_core);
            replay_start_tile = 1;
            replay_start_valid = 1;
            expect_sticky_fault();
        end
    endtask

    task automatic attack_corrupt_response(input integer corrupt_kind);
        integer wait_cycles;
        begin
            reset_dut();
            setup_active_one(24'h900010+corrupt_kind);
            flush_memory();
            memory_latency = 1;
            descriptor_read_req_ready = 1;
            response_enable = 1;
            bundle_ready = 1;
            corrupt_response_flags = corrupt_kind==0;
            corrupt_response_address = corrupt_kind==1;
            @(negedge clk_core);
            replay_start_tile = 0;
            replay_start_valid = 1;
            do @(posedge clk_core); while (!replay_start_accept);
            @(negedge clk_core);
            replay_start_valid = 0;
            @(posedge clk_core);
            wait_cycles = 0;
            while (!protocol_error && wait_cycles<40) begin
                @(posedge clk_core);
                wait_cycles = wait_cycles + 1;
            end
            if (!protocol_error) $fatal(1,"corrupt response did not fault");
            expect_sticky_fault();
        end
    endtask

    task automatic attack_third_replay;
        begin
            reset_dut();
            setup_active_one(24'h900020);
            run_replay(0,1,1,0);
            run_replay(1,1,1,0);
            @(negedge clk_core);
            replay_start_tile = 0;
            replay_start_valid = 1;
            expect_sticky_fault();
        end
    endtask

    task automatic accept_seal_without_runs;
        begin
            @(negedge clk_core);
            phase_seal_ready = 1;
            @(posedge clk_core);
            if (!phase_seal_accept) $fatal(1,"manual seal not accepted");
            @(negedge clk_core);
            phase_seal_ready = 0;
        end
    endtask

    task automatic start_replay_and_hold(input integer tile_value);
        begin
            flush_memory();
            descriptor_read_req_ready = 0;
            bundle_ready = 0;
            replay_done_ready = 0;
            @(negedge clk_core);
            replay_start_tile = tile_value;
            replay_start_valid = 1;
            do @(posedge clk_core); while (!replay_start_accept);
            @(negedge clk_core);
            replay_start_valid = 0;
        end
    endtask

    task automatic reach_replay_done(input integer tile_value);
        integer cycles;
        begin
            flush_memory();
            memory_latency = 1;
            descriptor_read_req_ready = 1;
            response_enable = 1;
            bundle_ready = 1;
            replay_done_ready = 0;
            @(negedge clk_core);
            replay_start_tile = tile_value;
            replay_start_valid = 1;
            do @(posedge clk_core); while (!replay_start_accept);
            @(negedge clk_core);
            replay_start_valid = 0;
            if (tile_value == 0) begin
                tile1_prefetch_done_tag = phase_tag;
                tile1_prefetch_done_bank = phase_bank;
                tile1_prefetch_done_valid = 1;
                @(posedge clk_core);
                if (!tile1_prefetch_done_accept)
                    $fatal(1,"reach-done prefetch completion failed");
                @(negedge clk_core);
                tile1_prefetch_done_valid = 0;
            end
            cycles = 0;
            while (!replay_done_valid && cycles<100) begin
                @(posedge clk_core);
                cycles = cycles + 1;
            end
            if (!replay_done_valid) $fatal(1,"failed to reach replay done");
            @(negedge clk_core);
            bundle_ready = 0;
        end
    endtask

    task automatic reset_state_coverage;
        begin
            account_replay = 0;

            reset_dut();
            start_phase(24'ha00001,0); // MATCH
            reset_dut();

            start_phase(24'ha00002,0);
            send_rows(0);              // SEAL
            reset_dut();

            start_phase(24'ha00003,0);
            send_rows(1);
            accept_seal_without_runs(); // RUNS
            reset_dut();

            setup_active_one(24'ha00004); // WAIT0
            reset_dut();

            setup_active_one(24'ha00005);
            start_replay_and_hold(0);   // REPLAY0
            reset_dut();

            setup_active_one(24'ha00006);
            reach_replay_done(0);       // DONE0
            reset_dut();

            setup_active_one(24'ha00007);
            run_replay(0,1,1,0);        // WAIT1
            reset_dut();

            setup_active_one(24'ha00008);
            run_replay(0,1,1,0);
            start_replay_and_hold(1);   // REPLAY1
            reset_dut();

            setup_active_one(24'ha00009);
            run_replay(0,1,1,0);
            reach_replay_done(1);       // DONE1
            reset_dut();

            start_phase(24'ha0000a,0);
            send_rows(0);
            accept_seal(0);             // PHASE_DONE
            reset_dut();

            account_replay = 1;
        end
    endtask

    initial begin : watchdog
        #12000000;
        $fatal(1,"M384 global watchdog timeout");
    end

    initial begin : test
        reset_n = 0;
        normal_phases = 0;
        normal_replays = 0;
        checked_bundles = 0;
        checked_pwp_runs = 0;
        zero_rows = 0;
        active_rows = 0;
        pop1_rows = 0;
        pwp_rows = 0;
        fallback_rows = 0;
        write_stalls = 0;
        request_stalls = 0;
        response_stalls = 0;
        backend_stalls = 0;
        protocol_attacks = 0;
        sticky_cycles = 0;
        max_fifo = 0;
        max_outstanding = 0;
        max_credit = 0;
        latency_mask = 0;
        account_replay = 1;
        state_reset_mask = 0;
        prefetch_starts = 0;
        prefetch_dones = 0;
        clear_drivers();
        reset_dut();
        idle_reload();

        run_phase(24'h100001,0,0,1,1,0);
        run_phase(24'h100002,1,1,1,2,0);
        run_phase(24'h100003,0,2400,2,4,0);
        run_phase(24'h100004,1,3000,8,8,1);

        reset_state_coverage();

        attack_bad_row(0); // non-monotonic row
        attack_bad_row(1); // center 32
        attack_bad_row(2); // bad distance
        attack_bad_row(3); // pop1 illegally marked PWP
        attack_bad_row(4); // early row_last
        attack_wrong_tile();
        attack_corrupt_response(0); // nonzero reserved flags
        attack_corrupt_response(1); // wrong SRAM response address
        attack_third_replay();

        // Unsolicited response is a sticky fault even with no active replay.
        reset_dut();
        @(negedge clk_core);
        inject_unexpected_response = 1;
        @(posedge clk_core);
        @(negedge clk_core);
        expect_sticky_fault();
        inject_unexpected_response = 0;

        if (normal_phases != 4 || normal_replays != 8)
            $fatal(1,"normal extent mismatch");
        if (checked_bundles != 2*(1+2400+3000)+2)
            $fatal(1,"checked bundle extent mismatch");
        if (pop1_rows == 0 || pwp_rows == 0 || fallback_rows == 0)
            $fatal(1,"source coverage missing");
        if (checked_pwp_runs < 4)
            $fatal(1,"PWP run coverage missing");
        if (max_fifo != 8 || max_outstanding != 8 || max_credit != 8)
            $fatal(1,"credit coverage missing fifo=%0d out=%0d credit=%0d",
                   max_fifo,max_outstanding,max_credit);
        if ((latency_mask & ((1<<1)|(1<<2)|(1<<4)|(1<<8)))
            != ((1<<1)|(1<<2)|(1<<4)|(1<<8)))
            $fatal(1,"latency coverage missing mask=%h",latency_mask);
        if (protocol_attacks != 10 || sticky_cycles < 100)
            $fatal(1,"protocol coverage mismatch");
        if ((state_reset_mask & 32'h0000_87ff) != 32'h0000_87ff)
            $fatal(1,"state reset coverage missing mask=%h",state_reset_mask);
        if (prefetch_starts == 0 || prefetch_dones == 0
            || prefetch_dones > prefetch_starts)
            $fatal(1,"tile1 prefetch coverage mismatch starts=%0d dones=%0d",
                   prefetch_starts,prefetch_dones);

        $display("PASS M384 active descriptor streaming controller phases=%0d replays=%0d bundles=%0d pwp_runs=%0d prefetch_starts=%0d prefetch_dones=%0d zero=%0d active=%0d pop1=%0d pwp=%0d fallback=%0d write_stalls=%0d request_stalls=%0d response_stalls=%0d backend_stalls=%0d protocol_attacks=%0d sticky_cycles=%0d max_fifo=%0d max_outstanding=%0d max_credit=%0d latency_mask=%0h reset_mask=%0h mismatches=0 exact_compaction=true direct_address_runs=true tile1_overlap=true dual_replay=true ii1_credit=true system_speedup=false headline=false",
                 normal_phases,normal_replays,checked_bundles,
                 checked_pwp_runs,prefetch_starts,prefetch_dones,zero_rows,
                 active_rows,pop1_rows,pwp_rows,fallback_rows,write_stalls,
                 request_stalls,response_stalls,backend_stalls,
                 protocol_attacks,sticky_cycles,max_fifo,max_outstanding,
                 max_credit,latency_mask,state_reset_mask);
        $finish;
    end
endmodule

`default_nettype wire
