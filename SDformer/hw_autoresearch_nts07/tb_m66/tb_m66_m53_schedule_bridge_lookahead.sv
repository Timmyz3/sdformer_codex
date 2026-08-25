`timescale 1ns/1ps
`default_nettype none

module tb_m66_m53_schedule_bridge_lookahead;
    localparam int LANES = 96;
    localparam int ACC_W = 19;
    localparam int RSP_DEPTH = 32;
    localparam int EXP_DEPTH = 64;

    logic clk_core, rst_core;
    logic command_valid, command_ready, command_accept;
    logic [47:0] command_tag;
    logic [255:0] command_add_bits, command_subtract_bits;
    logic [1823:0] command_seed_acc;
    logic [3:0] command_accept_context;
    logic launch_valid, launch_ready, launch_accept;
    logic [2:0] launch_context_count;
    logic [15:0] launch_contexts;
    logic weight_request_valid, weight_request_ready, request_accept;
    logic [15:0] weight_request_tag;
    logic [2:0] weight_request_context_count;
    logic [15:0] weight_request_contexts;
    logic [7:0] weight_request_bank_valid;
    logic [39:0] weight_request_bank_addr;
    logic [31:0] weight_request_context_valid;
    logic [31:0] weight_request_context_subtract;
    logic weight_request_last;
    logic weight_response_valid, weight_response_ready, response_accept;
    logic [15:0] weight_response_tag;
    logic [2:0] weight_response_context_count;
    logic [15:0] weight_response_contexts;
    logic [7:0] weight_response_bank_valid;
    logic [6143:0] weight_response_data;
    logic output_valid, output_ready, output_accept;
    logic [47:0] output_tag;
    logic [8:0] output_source_count;
    logic [1823:0] output_acc;
    logic protocol_error, busy, group_active;
    logic [4:0] context_occupancy;
    logic [4:0] response_metadata_occupancy, complete_occupancy;
    logic [63:0] telemetry_cycles, telemetry_commands, telemetry_launches;
    logic [63:0] telemetry_requests, telemetry_responses, telemetry_outputs;
    logic [63:0] telemetry_command_stalls, telemetry_launch_stalls;
    logic [63:0] telemetry_request_stalls, telemetry_response_stalls;
    logic [63:0] telemetry_output_stalls, telemetry_context_reuses;
    logic [63:0] telemetry_response_tag_wraps;
    logic [4:0] telemetry_max_context_occupancy;
    logic [4:0] telemetry_max_metadata_occupancy;
    logic [4:0] telemetry_max_complete_occupancy;

    byte unsigned io_buf [0:255];
    integer stream_fd, ledger_fd;
    string stream_path, ledger_path;
    integer response_latency;
    longint unsigned expected_groups, expected_commands;
    longint unsigned expected_model_cycles, expected_source_cycles;
    longint unsigned sample_start_cycle, schedule_late_cycles;
    longint unsigned schedule_late_groups;
    longint unsigned launch_phase_direct_groups, launch_phase_aligned_groups;
    longint unsigned prelaunch_artificial_bubbles;
    longint unsigned descriptor_index, group_index;
    integer sample_id;

    logic [15:0] rsp_tag_q [0:RSP_DEPTH-1];
    logic [2:0] rsp_count_q [0:RSP_DEPTH-1];
    logic [15:0] rsp_contexts_q [0:RSP_DEPTH-1];
    logic [7:0] rsp_bank_valid_q [0:RSP_DEPTH-1];
    longint unsigned rsp_due_q [0:RSP_DEPTH-1];
    longint unsigned rsp_write, rsp_read;
    logic response_consumed_since_negedge;

    logic [47:0] context_expected_tag [0:15];
    integer context_expected_count [0:15];
    integer signed context_expected_scalar [0:15];
    logic [47:0] expected_tag_q [0:EXP_DEPTH-1];
    integer expected_count_q [0:EXP_DEPTH-1];
    integer signed expected_scalar_q [0:EXP_DEPTH-1];
    longint unsigned expected_write, expected_read;

    longint unsigned parent_zero, parent_left, parent_up, parent_previous;
    longint unsigned signed_add_terms, signed_subtract_terms;
    integer functional_mismatches;
    longint unsigned seamless_launches;

    qfit_m66_m53_schedule_bridge_lookahead dut (.*);

    qfit_k4_parent_delta_p8_l96_ctx16_assertions m54_sva (
        .clk_core, .rst_core,
        .command_valid, .command_ready, .command_accept,
        .command_accept_context,
        .launch_valid, .launch_ready, .launch_accept,
        .launch_context_count, .launch_contexts,
        .launch_legal(dut.core.launch_legal),
        .launch_zero(dut.core.launch_zero),
        .weight_request_valid, .weight_request_ready, .weight_request_tag,
        .weight_request_context_count, .weight_request_contexts,
        .weight_request_bank_valid, .weight_request_bank_addr,
        .weight_request_context_valid, .weight_request_context_subtract,
        .weight_request_last, .request_accept,
        .weight_response_valid, .weight_response_ready, .weight_response_tag,
        .weight_response_context_count, .weight_response_contexts,
        .weight_response_bank_valid, .response_accept,
        .response_contract_valid(dut.core.response_contract_valid),
        .response_acc_overflow(dut.core.response_acc_overflow),
        .output_valid, .output_ready, .output_tag, .output_source_count,
        .output_acc, .output_accept, .protocol_error, .busy,
        .context_occupancy, .response_metadata_occupancy,
        .complete_occupancy, .group_active,
        .complete_push_count(dut.core.complete_push_count),
        .final_response_success(dut.core.final_response_success),
        .zero_launch_success(dut.core.zero_launch_success),
        .context_allocated_vector(dut.core.context_allocated_vector),
        .context_launched_vector(dut.core.context_launched_vector),
        .meta_head(dut.core.meta_head_q), .meta_tail(dut.core.meta_tail_q),
        .complete_head(dut.core.complete_head_q),
        .complete_tail(dut.core.complete_tail_q)
    );

    qfit_k4_parent_delta_lookahead_assertions m66_seam_sva (
        .clk_core, .rst_core,
        .launch_valid, .launch_ready, .launch_accept,
        .launch_context_count, .launch_contexts,
        .launch_legal(dut.core.launch_legal),
        .launch_zero(dut.core.launch_zero),
        .final_response_success(dut.core.final_response_success),
        .command_accept, .command_accept_context,
        .request_accept, .output_accept,
        .complete_push_count(dut.core.complete_push_count),
        .response_metadata_occupancy, .complete_occupancy,
        .group_active,
        .active_count_state(dut.core.active_count_q),
        .active_contexts_state({dut.core.active_context_q[3],
                                dut.core.active_context_q[2],
                                dut.core.active_context_q[1],
                                dut.core.active_context_q[0]}),
        .context_allocated_vector(dut.core.context_allocated_vector),
        .context_launched_vector(dut.core.context_launched_vector),
        .protocol_error
    );

    always #1.5 clk_core = ~clk_core;

    function automatic longint unsigned load_u64(input integer offset);
        longint unsigned value;
        begin
            value = 0;
            for (int index = 0; index < 8; index++)
                value |= longint'(io_buf[offset+index]) << (8*index);
            load_u64 = value;
        end
    endfunction

    function automatic integer load_u32(input integer offset);
        integer value;
        begin
            value = 0;
            for (int index = 0; index < 4; index++)
                value |= integer'(io_buf[offset+index]) << (8*index);
            load_u32 = value;
        end
    endfunction

    function automatic integer load_u16(input integer offset);
        load_u16 = integer'(io_buf[offset])
            | (integer'(io_buf[offset+1]) << 8);
    endfunction

    function automatic integer popcount256(input logic [255:0] value);
        popcount256 = $countones(value);
    endfunction

    task automatic read_exact(input integer count);
        integer got;
        begin
            got = $fread(io_buf, stream_fd, 0, count);
            if (got != count)
                $fatal(1, "M57 stream short read got=%0d expected=%0d", got, count);
        end
    endtask

    task automatic require_magic(
        input byte unsigned a, input byte unsigned b,
        input byte unsigned c, input byte unsigned d
    );
        begin
            if (io_buf[0] != a || io_buf[1] != b
                    || io_buf[2] != c || io_buf[3] != d)
                $fatal(1, "M57 binary record magic mismatch %02x%02x%02x%02x",
                       io_buf[0], io_buf[1], io_buf[2], io_buf[3]);
        end
    endtask

    task automatic send_descriptor(
        input logic [255:0] add_mask,
        input logic [255:0] subtract_mask,
        output logic [3:0] context_id
    );
        integer add_count, subtract_count;
        begin
            add_count = popcount256(add_mask);
            subtract_count = popcount256(subtract_mask);
            // The trace bridge is a one-command-per-cycle streaming source.
            // The caller is already aligned to a negedge (reset release or
            // the prior descriptor's accept), so inserting another negedge
            // here would create an artificial bubble absent from the RTL
            // ready/valid contract.
            command_tag = descriptor_index[47:0];
            command_add_bits = add_mask;
            command_subtract_bits = subtract_mask;
            command_seed_acc = '0;
            command_valid = 1'b1;
            do @(posedge clk_core); while (!command_accept);
            context_id = command_accept_context;
            context_expected_tag[context_id] = descriptor_index[47:0];
            context_expected_count[context_id] = add_count + subtract_count;
            context_expected_scalar[context_id] = add_count - subtract_count;
            descriptor_index = descriptor_index + 1;
            @(negedge clk_core);
            command_valid = 1'b0;
        end
    endtask

    task automatic launch_one_group(
        input integer count,
        input logic [15:0] contexts,
        input longint unsigned target_cycle
    );
        longint unsigned relative_cycle;
        integer slot, ctx, queue_index;
        logic waited_for_target_posedge;
        begin
            relative_cycle = telemetry_cycles - sample_start_cycle;
            waited_for_target_posedge = 1'b0;
            while (relative_cycle < target_cycle) begin
                @(posedge clk_core);
                waited_for_target_posedge = 1'b1;
                relative_cycle = telemetry_cycles - sample_start_cycle;
            end
            if (relative_cycle > target_cycle) begin
                schedule_late_cycles = schedule_late_cycles
                    + (relative_cycle - target_cycle);
                schedule_late_groups = schedule_late_groups + 1;
            end
            for (slot = 0; slot < count; slot++) begin
                ctx = contexts[slot*4 +: 4];
                queue_index = expected_write % EXP_DEPTH;
                expected_tag_q[queue_index] = context_expected_tag[ctx];
                expected_count_q[queue_index] = context_expected_count[ctx];
                expected_scalar_q[queue_index] = context_expected_scalar[ctx];
                expected_write = expected_write + 1;
                if (expected_write - expected_read > EXP_DEPTH)
                    $fatal(1, "M57 expected output ring overflow");
            end
            // send_descriptor returns on a negedge.  If target waiting did not
            // execute, another unconditional @(negedge) would add one
            // artificial cycle per group.  Only a path that actually exited
            // the target wait on a posedge needs phase alignment here.
            if (waited_for_target_posedge) begin
                @(negedge clk_core);
                launch_phase_aligned_groups = launch_phase_aligned_groups + 1;
            end else begin
                if (clk_core !== 1'b0)
                    $fatal(1, "M57 direct launch path not entered at negedge");
                launch_phase_direct_groups = launch_phase_direct_groups + 1;
            end
            launch_context_count = count[2:0];
            launch_contexts = contexts;
            launch_valid = 1'b1;
            do @(posedge clk_core); while (!launch_accept);
            @(negedge clk_core);
            launch_valid = 1'b0;
        end
    endtask

    // Every source supplies signed +1 to all 96 lanes.  This keeps the full
    // signed19x96 check inexpensive while retaining add/subtract semantics.
    initial weight_response_data = {768{8'h01}};

    always @(posedge clk_core) begin
        response_consumed_since_negedge <= response_accept;
        if (!rst_core && request_accept) begin
            integer index;
            index = rsp_write % RSP_DEPTH;
            if (rsp_write - rsp_read >= RSP_DEPTH)
                $fatal(1, "M57 response ring overflow");
            rsp_tag_q[index] = weight_request_tag;
            rsp_count_q[index] = weight_request_context_count;
            rsp_contexts_q[index] = weight_request_contexts;
            rsp_bank_valid_q[index] = weight_request_bank_valid;
            rsp_due_q[index] = telemetry_cycles + response_latency;
            rsp_write = rsp_write + 1;
        end
    end

    always @(negedge clk_core) begin
        if (rst_core) begin
            weight_response_valid = 1'b0;
        end else begin
            if (response_consumed_since_negedge) begin
                weight_response_valid = 1'b0;
                rsp_read = rsp_read + 1;
            end
            if (!weight_response_valid && rsp_read < rsp_write) begin
                integer index;
                index = rsp_read % RSP_DEPTH;
                if (telemetry_cycles >= rsp_due_q[index]) begin
                    weight_response_valid = 1'b1;
                    weight_response_tag = rsp_tag_q[index];
                    weight_response_context_count = rsp_count_q[index];
                    weight_response_contexts = rsp_contexts_q[index];
                    weight_response_bank_valid = rsp_bank_valid_q[index];
                end
            end
        end
    end

    always @(posedge clk_core) begin
        if (!rst_core) begin
            if (launch_accept && dut.core.final_response_success
                    && !dut.core.launch_zero)
                seamless_launches = seamless_launches + 1;
            if (command_accept && |(command_add_bits & command_subtract_bits))
                $display("M57_FAULT_CAUSE overlapping_command cycle=%0d tag=%0h",
                         telemetry_cycles, command_tag);
            if (launch_accept && !dut.core.launch_legal)
                $display("M57_FAULT_CAUSE illegal_launch cycle=%0d count=%0d contexts=%0h allocated=%0h launched=%0h",
                         telemetry_cycles, launch_context_count, launch_contexts,
                         dut.core.context_allocated_vector,
                         dut.core.context_launched_vector);
            if (response_accept && (!dut.core.response_contract_valid
                                     || dut.core.response_acc_overflow))
                $display("M57_FAULT_CAUSE bad_response cycle=%0d contract=%0d overflow=%0d got_tag=%0h exp_tag=%0h got_count=%0d exp_count=%0d",
                         telemetry_cycles, dut.core.response_contract_valid,
                         dut.core.response_acc_overflow, weight_response_tag,
                         dut.core.expected_tag, weight_response_context_count,
                         dut.core.expected_count);
            if (weight_response_valid && response_metadata_occupancy == 0)
                $display("M57_FAULT_CAUSE unexpected_response cycle=%0d tag=%0h",
                         telemetry_cycles, weight_response_tag);
            if (protocol_error)
                $fatal(1, "M57 legal stream triggered protocol_error cycle=%0d",
                       telemetry_cycles);
            if (output_accept) begin
                integer index;
                logic signed [ACC_W-1:0] expected_lane;
                index = expected_read % EXP_DEPTH;
                if (expected_read >= expected_write) begin
                    functional_mismatches = functional_mismatches + 1;
                    $fatal(1, "M57 unexpected output tag=%0h", output_tag);
                end
                if (output_tag !== expected_tag_q[index]
                        || output_source_count !== expected_count_q[index][8:0]) begin
                    functional_mismatches = functional_mismatches + 1;
                    $fatal(1, "M57 output tag/count mismatch idx=%0d tag=%0h/%0h count=%0d/%0d",
                           expected_read, output_tag, expected_tag_q[index],
                           output_source_count, expected_count_q[index]);
                end
                expected_lane = expected_scalar_q[index];
                if (output_acc !== {LANES{expected_lane}}) begin
                    functional_mismatches = functional_mismatches + 1;
                    $fatal(1, "M57 signed19x96 mismatch idx=%0d got_lane0=%0d exp=%0d",
                           expected_read, $signed(output_acc[ACC_W-1:0]),
                           expected_scalar_q[index]);
                end
                expected_read = expected_read + 1;
            end
            if (request_accept || response_accept || output_accept
                    || (weight_request_valid && !weight_request_ready)
                    || (weight_response_valid && !weight_response_ready)
                    || (output_valid && !output_ready))
                // Compact, independently replayable accepted-event record:
                // cycle, flags(req/rsp/out/last), packed occupancy, and the
                // three observed tags.  Full stall counts remain in END.
                $fdisplay(ledger_fd, "E %0d %0x %0x %0h %0h %0h",
                    telemetry_cycles,
                    {28'b0, weight_request_last, output_accept,
                     response_accept, request_accept},
                    {17'b0, context_occupancy, complete_occupancy,
                     response_metadata_occupancy},
                    weight_request_tag, weight_response_tag, output_tag);
        end
    end

    initial begin
        integer version, header_sample;
        integer count, group_cycles, parent_code, task_index;
        integer timeout;
        longint unsigned target_cycle, encoded_group_id;
        logic [3:0] contexts [0:3];
        logic [15:0] packed_contexts;
        logic [255:0] add_mask, subtract_mask;

        clk_core = 0; rst_core = 1;
        command_valid = 0; command_tag = 0;
        command_add_bits = 0; command_subtract_bits = 0;
        command_seed_acc = 0;
        launch_valid = 0; launch_context_count = 0; launch_contexts = 0;
        weight_request_ready = 1;
        weight_response_valid = 0; weight_response_tag = 0;
        weight_response_context_count = 0; weight_response_contexts = 0;
        weight_response_bank_valid = 0;
        output_ready = 1;
        rsp_write = 0; rsp_read = 0;
        response_consumed_since_negedge = 0;
        expected_write = 0; expected_read = 0;
        descriptor_index = 0; group_index = 0;
        expected_source_cycles = 0;
        schedule_late_cycles = 0;
        schedule_late_groups = 0;
        seamless_launches = 0;
        launch_phase_direct_groups = 0;
        launch_phase_aligned_groups = 0;
        prelaunch_artificial_bubbles = 0;
        parent_zero = 0; parent_left = 0; parent_up = 0; parent_previous = 0;
        signed_add_terms = 0; signed_subtract_terms = 0;
        functional_mismatches = 0;
        response_latency = 1;
        if (!$value$plusargs("STREAM=%s", stream_path))
            $fatal(1, "M57 +STREAM is required");
        if (!$value$plusargs("LEDGER=%s", ledger_path))
            $fatal(1, "M57 +LEDGER is required");
        void'($value$plusargs("RSP_LATENCY=%d", response_latency));
        if (response_latency < 1 || response_latency > 24)
            $fatal(1, "M57 response latency outside frozen range");
        stream_fd = $fopen(stream_path, "rb");
        if (stream_fd == 0) $fatal(1, "M57 cannot open stream %s", stream_path);
        ledger_fd = $fopen(ledger_path, "w");
        if (ledger_fd == 0) $fatal(1, "M57 cannot open ledger %s", ledger_path);

        read_exact(40);
        if (io_buf[0] != "M" || io_buf[1] != "5" || io_buf[2] != "7"
                || io_buf[3] != "R" || io_buf[4] != "1"
                || io_buf[5] != "B" || io_buf[6] != "I" || io_buf[7] != "N")
            $fatal(1, "M57 file header magic mismatch");
        version = load_u32(8);
        header_sample = load_u32(12);
        expected_groups = load_u64(16);
        expected_commands = load_u64(24);
        expected_model_cycles = load_u64(32);
        if (version != 1 || header_sample < 0 || header_sample > 9)
            $fatal(1, "M57 file header identity mismatch");
        sample_id = header_sample;
        $fdisplay(ledger_fd,
            "BEGIN sample=%0d groups=%0d commands=%0d model_cycles=%0d latency=%0d ledger=compact_v1",
            sample_id, expected_groups, expected_commands,
            expected_model_cycles, response_latency);

        repeat (5) @(posedge clk_core);
        @(negedge clk_core); rst_core = 0;
        sample_start_cycle = telemetry_cycles;

        for (group_index = 0; group_index < expected_groups; group_index++) begin
            if ((group_index % 100000) == 0)
                $display("M57_PROGRESS sample=%0d group=%0d/%0d cycle=%0d",
                         sample_id, group_index, expected_groups,
                         telemetry_cycles - sample_start_cycle);
            read_exact(28);
            require_magic("G", "R", "P", "1");
            target_cycle = load_u64(4);
            encoded_group_id = load_u64(12);
            if (encoded_group_id != group_index || io_buf[20] != sample_id)
                $fatal(1, "M57 group identity drift got=%0d expected=%0d",
                       encoded_group_id, group_index);
            count = io_buf[25];
            group_cycles = load_u16(26);
            if (count < 1 || count > 4 || group_cycles < 0 || group_cycles > 32)
                $fatal(1, "M57 group geometry drift");
            expected_source_cycles = expected_source_cycles + group_cycles;
            packed_contexts = '0;
            for (int slot = 0; slot < count; slot++) begin
                read_exact(68);
                task_index = load_u16(0);
                parent_code = io_buf[2];
                if (task_index < 0 || task_index >= 300 || parent_code > 3)
                    $fatal(1, "M57 descriptor metadata drift");
                add_mask = '0; subtract_mask = '0;
                for (int byte_index = 0; byte_index < 32; byte_index++) begin
                    add_mask[byte_index*8 +: 8] = io_buf[4+byte_index];
                    subtract_mask[byte_index*8 +: 8] = io_buf[36+byte_index];
                end
                if (|(add_mask & subtract_mask))
                    $fatal(1, "M57 overlapping signed masks");
                case (parent_code)
                    0: parent_zero = parent_zero + 1;
                    1: parent_left = parent_left + 1;
                    2: parent_up = parent_up + 1;
                    3: parent_previous = parent_previous + 1;
                endcase
                signed_add_terms = signed_add_terms + popcount256(add_mask);
                signed_subtract_terms = signed_subtract_terms
                    + popcount256(subtract_mask);
                send_descriptor(add_mask, subtract_mask, contexts[slot]);
                packed_contexts[slot*4 +: 4] = contexts[slot];
            end
            launch_one_group(count, packed_contexts, target_cycle);
        end

        read_exact(28);
        require_magic("E", "N", "D", "1");
        if (load_u64(4) != expected_groups
                || load_u64(12) != expected_commands
                || load_u64(20) != expected_source_cycles)
            $fatal(1, "M57 trailer mismatch");
        timeout = 0;
        while ((busy || expected_read != expected_write || rsp_read != rsp_write)
                && timeout < 100000) begin
            @(posedge clk_core); timeout++;
        end
        if (timeout == 100000)
            $fatal(1, "M57 final drain timeout busy=%0d exp=%0d/%0d rsp=%0d/%0d",
                   busy, expected_read, expected_write, rsp_read, rsp_write);
        #1;
        if (descriptor_index != expected_commands
                || telemetry_commands != expected_commands
                || telemetry_launches != expected_groups
                || telemetry_requests != expected_source_cycles
                || telemetry_responses != expected_source_cycles
                || telemetry_outputs != expected_commands
                || expected_read != expected_commands
                || functional_mismatches != 0)
            $fatal(1, "M57 final conservation mismatch cmd=%0d/%0d launch=%0d/%0d req=%0d/%0d rsp=%0d out=%0d/%0d mismatch=%0d",
                   telemetry_commands, expected_commands,
                   telemetry_launches, expected_groups,
                   telemetry_requests, expected_source_cycles,
                   telemetry_responses, telemetry_outputs, expected_commands,
                   functional_mismatches);
        $fdisplay(ledger_fd,
            "END sample=%0d rtl_cycles=%0d model_cycles=%0d commands=%0d groups=%0d requests=%0d responses=%0d outputs=%0d max_meta=%0d max_ctx=%0d max_complete=%0d cmd_stall=%0d launch_stall=%0d seamless=%0d req_stall=%0d rsp_stall=%0d out_stall=%0d reuse=%0d tag_wrap=%0d late=%0d late_groups=%0d phase_direct=%0d phase_aligned=%0d prelaunch_artificial_bubbles=%0d parent=%0d,%0d,%0d,%0d add=%0d sub=%0d mismatches=%0d",
            sample_id, telemetry_cycles - sample_start_cycle,
            expected_model_cycles, telemetry_commands, telemetry_launches,
            telemetry_requests, telemetry_responses, telemetry_outputs,
            telemetry_max_metadata_occupancy,
            telemetry_max_context_occupancy,
            telemetry_max_complete_occupancy,
            telemetry_command_stalls, telemetry_launch_stalls, seamless_launches,
            telemetry_request_stalls, telemetry_response_stalls,
            telemetry_output_stalls, telemetry_context_reuses,
            telemetry_response_tag_wraps, schedule_late_cycles,
            schedule_late_groups, launch_phase_direct_groups,
            launch_phase_aligned_groups, prelaunch_artificial_bubbles,
            parent_zero, parent_left, parent_up, parent_previous,
            signed_add_terms, signed_subtract_terms, functional_mismatches);
        $fclose(stream_fd);
        $fclose(ledger_fd);
        $display("PASS M66 S%0d groups=%0d commands=%0d requests=%0d outputs=%0d rtl_cycles=%0d model_cycles=%0d seamless=%0d max_meta=%0d tag_wrap=%0d",
                 sample_id, telemetry_launches, telemetry_commands,
                 telemetry_requests, telemetry_outputs,
                 telemetry_cycles - sample_start_cycle,
                 expected_model_cycles, seamless_launches,
                 telemetry_max_metadata_occupancy,
                 telemetry_response_tag_wraps);
        $finish;
    end
endmodule

`default_nettype wire
