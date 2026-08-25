`timescale 1ns/1ps
`default_nettype none

module tb_m67_qfit_k4_parent_delta_lookahead_pressure;
    localparam int TILE_BITS = 256;
    localparam int BANKS = 8;
    localparam int LANES = 96;
    localparam int ACC_W = 19;
    localparam int MAX_K = 4;
    localparam int MAX_REQS = 4096;
    localparam int MAX_TAGS = 512;

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

    logic [15:0] rsp_tag_q [0:MAX_REQS-1];
    logic [2:0] rsp_count_q [0:MAX_REQS-1];
    logic [15:0] rsp_contexts_q [0:MAX_REQS-1];
    logic [7:0] rsp_bank_valid_q [0:MAX_REQS-1];
    logic [6143:0] rsp_data_q [0:MAX_REQS-1];
    integer rsp_write, rsp_read;
    logic automatic_responses, random_response_gaps;
    logic response_consumed_since_negedge;
    logic random_backpressure;

    integer ctx_tag_model [0:15];
    integer expected_count [0:MAX_TAGS-1];
    integer signed expected_acc [0:MAX_TAGS-1][0:LANES-1];
    integer expected_order [0:MAX_TAGS-1];
    integer expected_order_count, outputs_seen, legal_tag_next;
    integer cycle_count, ledger_fd;
    string ledger_path;
    logic ledger_enable;

    logic saw_context16, saw_meta16, saw_complete16;
    logic saw_push4, saw_complete13_pop_push4;
    logic saw_meta_tail_wrap, saw_complete_tail_wrap;
    logic saw_k1, saw_k2, saw_k3, saw_k4;
    logic saw_k4_full, saw_k4_partial, saw_k4_no_share;
    logic saw_request_stall, saw_response_stall, saw_output_stall;
    logic saw_context_reuse;
    logic saw_seam, saw_seam_command, saw_seam_output;
    logic saw_seam_command_output, saw_zero_next_wait;
    integer sealed_commands, sealed_outputs, sealed_requests, sealed_groups;

    qfit_k4_parent_delta_p8_l96_ctx16_lookahead dut (.*);

    qfit_k4_parent_delta_p8_l96_ctx16_assertions sva (
        .clk_core, .rst_core,
        .command_valid, .command_ready, .command_accept,
        .command_accept_context,
        .launch_valid, .launch_ready, .launch_accept,
        .launch_context_count, .launch_contexts,
        .launch_legal(dut.launch_legal), .launch_zero(dut.launch_zero),
        .weight_request_valid, .weight_request_ready, .weight_request_tag,
        .weight_request_context_count, .weight_request_contexts,
        .weight_request_bank_valid, .weight_request_bank_addr,
        .weight_request_context_valid, .weight_request_context_subtract,
        .weight_request_last, .request_accept,
        .weight_response_valid, .weight_response_ready, .weight_response_tag,
        .weight_response_context_count, .weight_response_contexts,
        .weight_response_bank_valid, .response_accept,
        .response_contract_valid(dut.response_contract_valid),
        .response_acc_overflow(dut.response_acc_overflow),
        .output_valid, .output_ready, .output_tag, .output_source_count,
        .output_acc, .output_accept, .protocol_error, .busy,
        .context_occupancy, .response_metadata_occupancy,
        .complete_occupancy, .group_active,
        .complete_push_count(dut.complete_push_count),
        .final_response_success(dut.final_response_success),
        .zero_launch_success(dut.zero_launch_success),
        .context_allocated_vector(dut.context_allocated_vector),
        .context_launched_vector(dut.context_launched_vector),
        .meta_head(dut.meta_head_q), .meta_tail(dut.meta_tail_q),
        .complete_head(dut.complete_head_q),
        .complete_tail(dut.complete_tail_q)
    );

    qfit_k4_parent_delta_lookahead_assertions m66_seam_sva (
        .clk_core, .rst_core,
        .launch_valid, .launch_ready, .launch_accept,
        .launch_context_count, .launch_contexts,
        .launch_legal(dut.launch_legal), .launch_zero(dut.launch_zero),
        .final_response_success(dut.final_response_success),
        .command_accept, .command_accept_context,
        .request_accept, .output_accept,
        .complete_push_count(dut.complete_push_count),
        .response_metadata_occupancy, .complete_occupancy,
        .group_active,
        .active_count_state(dut.active_count_q),
        .active_contexts_state({dut.active_context_q[3],
                                dut.active_context_q[2],
                                dut.active_context_q[1],
                                dut.active_context_q[0]}),
        .context_allocated_vector(dut.context_allocated_vector),
        .context_launched_vector(dut.context_launched_vector),
        .protocol_error
    );

    function automatic signed [7:0] model_weight(
        input integer source, input integer lane
    );
        integer value;
        begin
            if (source < BANKS) value = -128;
            else value = ((source * 37 + lane * 13 + 19) % 255) - 127;
            model_weight = value;
        end
    endfunction

    function automatic [15:0] pack4(
        input logic [3:0] c0, input logic [3:0] c1,
        input logic [3:0] c2, input logic [3:0] c3
    );
        pack4 = {c3, c2, c1, c0};
    endfunction

    task automatic make_seed(input integer mode, output logic [1823:0] seed);
        integer signed value;
        begin
            seed = '0;
            for (int lane = 0; lane < LANES; lane++) begin
                if (mode > 0) value = (1 << (ACC_W-1)) - 1;
                else if (mode < 0) value = -(1 << (ACC_W-1));
                else value = (lane % 9) - 4;
                seed[lane*ACC_W +: ACC_W] = value[ACC_W-1:0];
            end
        end
    endtask

    task automatic send_descriptor(
        input logic [255:0] add_mask,
        input logic [255:0] sub_mask,
        input integer seed_mode,
        output logic [3:0] context_id,
        output integer tag
    );
        logic [1823:0] seed;
        begin
            make_seed(seed_mode, seed);
            tag = legal_tag_next;
            @(negedge clk_core);
            while (!command_ready) @(negedge clk_core);
            command_tag = tag;
            command_add_bits = add_mask;
            command_subtract_bits = sub_mask;
            command_seed_acc = seed;
            command_valid = 1'b1;
            @(posedge clk_core);
            if (!command_accept) $fatal(1, "M67 command did not accept");
            context_id = command_accept_context;
            @(negedge clk_core);
            command_valid = 1'b0;
            legal_tag_next = legal_tag_next + 1;
        end
    endtask

    task automatic launch_group(
        input integer count,
        input logic [15:0] contexts
    );
        begin
            @(negedge clk_core);
            launch_context_count = count[2:0];
            launch_contexts = contexts;
            launch_valid = 1'b1;
            // Observe the handshake on posedges.  A ready pulse generated by
            // final-response lookahead can begin between negedges; polling
            // ready only at negedges can miss that legal accept and leave the
            // same context valid until a later, illegal duplicate launch.
            do @(posedge clk_core); while (!launch_accept);
            @(negedge clk_core);
            launch_valid = 1'b0;
        end
    endtask

    task automatic wait_for_all_outputs;
        integer timeout;
        begin
            timeout = 0;
            while (outputs_seen != expected_order_count && timeout < 20000) begin
                @(posedge clk_core); timeout++;
            end
            if (outputs_seen != expected_order_count)
                $fatal(1, "M67 output timeout got=%0d exp=%0d",
                       outputs_seen, expected_order_count);
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            command_valid = 1'b0;
            launch_valid = 1'b0;
            weight_response_valid = 1'b0;
            automatic_responses = 1'b0;
            random_response_gaps = 1'b0;
            random_backpressure = 1'b0;
            rsp_write = 0;
            rsp_read = 0;
            response_consumed_since_negedge = 1'b0;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
        end
    endtask

    task automatic require_fail_closed(input string attack);
        begin
            @(negedge clk_core);
            if (!protocol_error)
                $fatal(1, "M67 attack did not fault: %s", attack);
            repeat (2) @(posedge clk_core);
            @(negedge clk_core);
            if (!protocol_error || command_ready || launch_ready
                    || weight_request_valid || weight_response_ready
                    || output_valid)
                $fatal(1, "M67 fault not sticky/fail-closed: %s", attack);
        end
    endtask

    task automatic prepare_one_request(
        output logic [3:0] context_id,
        output integer tag
    );
        logic [255:0] add_mask, sub_mask;
        begin
            add_mask = '0; sub_mask = '0;
            for (int bank = 0; bank < BANKS; bank++)
                add_mask[2*BANKS + bank] = 1'b1;
            weight_request_ready = 1'b1;
            send_descriptor(add_mask, sub_mask, 0, context_id, tag);
            launch_group(1, pack4(context_id, 0, 0, 0));
            do @(posedge clk_core); while (response_metadata_occupancy == 0);
        end
    endtask

    always #1.5 clk_core = ~clk_core;

    // Store every accepted request and construct deterministic response data.
    always @(posedge clk_core) begin
        response_consumed_since_negedge <= response_accept;
        if (!rst_core && request_accept) begin
            if (rsp_write >= MAX_REQS) $fatal(1, "M67 response model overflow");
            rsp_tag_q[rsp_write] <= weight_request_tag;
            rsp_count_q[rsp_write] <= weight_request_context_count;
            rsp_contexts_q[rsp_write] <= weight_request_contexts;
            rsp_bank_valid_q[rsp_write] <= weight_request_bank_valid;
            for (int bank = 0; bank < BANKS; bank++) begin
                integer source;
                source = weight_request_bank_addr[bank*5 +: 5] * BANKS + bank;
                for (int lane = 0; lane < LANES; lane++)
                    rsp_data_q[rsp_write][(bank*LANES+lane)*8 +: 8]
                        <= model_weight(source, lane);
            end
            rsp_write <= rsp_write + 1;
        end
    end

    // Response payload remains stable until accepted.
    always @(negedge clk_core) begin
        if (rst_core) begin
            weight_response_valid = 1'b0;
        end else if (automatic_responses) begin
            if (response_consumed_since_negedge) begin
                weight_response_valid = 1'b0;
                rsp_read = rsp_read + 1;
            end
            if (!weight_response_valid && rsp_read < rsp_write
                    && (!random_response_gaps || $urandom_range(0, 3) != 0)) begin
                weight_response_valid = 1'b1;
                weight_response_tag = rsp_tag_q[rsp_read];
                weight_response_context_count = rsp_count_q[rsp_read];
                weight_response_contexts = rsp_contexts_q[rsp_read];
                weight_response_bank_valid = rsp_bank_valid_q[rsp_read];
                weight_response_data = rsp_data_q[rsp_read];
            end
        end
    end

    always @(negedge clk_core) begin
        if (rst_core) begin
            weight_request_ready = 1'b0;
            output_ready = 1'b0;
        end else if (random_backpressure) begin
            weight_request_ready = $urandom_range(0, 7) != 0;
            output_ready = $urandom_range(0, 5) != 0;
        end
    end

    // Independent in-TB oracle consumes accepted handshakes only.
    always @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (context_occupancy == 16) saw_context16 <= 1'b1;
            if (response_metadata_occupancy == 16) saw_meta16 <= 1'b1;
            if (complete_occupancy == 16) saw_complete16 <= 1'b1;
            if (dut.complete_push_count == 4) saw_push4 <= 1'b1;
            if (complete_occupancy == 13 && output_accept
                    && dut.complete_push_count == 4)
                saw_complete13_pop_push4 <= 1'b1;
            if (request_accept && dut.meta_tail_q == 4'hf)
                saw_meta_tail_wrap <= 1'b1;
            if (dut.complete_push_count != 0
                    && (int'(dut.complete_tail_q)
                        + int'(dut.complete_push_count)) >= 16)
                saw_complete_tail_wrap <= 1'b1;
            if (request_accept && weight_request_context_count == 1) saw_k1 <= 1;
            if (request_accept && weight_request_context_count == 2) saw_k2 <= 1;
            if (request_accept && weight_request_context_count == 3) saw_k3 <= 1;
            if (request_accept && weight_request_context_count == 4) saw_k4 <= 1;
            if (request_accept && weight_request_context_count == 4
                    && weight_request_context_valid[7:0] == weight_request_bank_valid
                    && weight_request_context_valid[15:8] == weight_request_bank_valid
                    && weight_request_context_valid[23:16] == weight_request_bank_valid
                    && weight_request_context_valid[31:24] == weight_request_bank_valid)
                saw_k4_full <= 1;
            if (request_accept && weight_request_context_count == 4
                    && |(weight_request_context_valid[7:0]
                         & weight_request_context_valid[15:8])
                    && weight_request_context_valid[7:0]
                        != weight_request_context_valid[15:8])
                saw_k4_partial <= 1;
            if (request_accept && weight_request_context_count == 4
                    && (weight_request_context_valid[7:0]
                        & weight_request_context_valid[15:8]) == 0
                    && (weight_request_context_valid[7:0]
                        & weight_request_context_valid[23:16]) == 0
                    && (weight_request_context_valid[7:0]
                        & weight_request_context_valid[31:24]) == 0
                    && (weight_request_context_valid[15:8]
                        & weight_request_context_valid[23:16]) == 0
                    && (weight_request_context_valid[15:8]
                        & weight_request_context_valid[31:24]) == 0
                    && (weight_request_context_valid[23:16]
                        & weight_request_context_valid[31:24]) == 0)
                saw_k4_no_share <= 1;
            if (weight_request_valid && !weight_request_ready)
                saw_request_stall <= 1;
            if (weight_response_valid && !weight_response_ready)
                saw_response_stall <= 1;
            if (output_valid && !output_ready) saw_output_stall <= 1;
            if (launch_accept && dut.launch_legal
                    && dut.final_response_success) begin
                saw_seam <= 1;
                if (command_accept) saw_seam_command <= 1;
                if (output_accept) saw_seam_output <= 1;
                if (command_accept && output_accept)
                    saw_seam_command_output <= 1;
            end
            if (dut.final_response_success && launch_valid
                    && dut.launch_legal && dut.launch_zero
                    && !launch_accept)
                saw_zero_next_wait <= 1;

            if (command_accept && ledger_enable) begin
                integer tag;
                tag = command_tag;
                ctx_tag_model[command_accept_context] = tag;
                expected_count[tag] = 0;
                for (int lane = 0; lane < LANES; lane++)
                    expected_acc[tag][lane]
                        = $signed(command_seed_acc[lane*ACC_W +: ACC_W]);
                $fdisplay(ledger_fd, "C %0d %0d %0h %0h %0h %0h",
                    cycle_count, command_accept_context, command_tag,
                    command_add_bits, command_subtract_bits, command_seed_acc);
            end
            if (launch_accept && dut.launch_legal && ledger_enable) begin
                for (int slot = 0; slot < launch_context_count; slot++) begin
                    integer ctx;
                    ctx = launch_contexts[slot*4 +: 4];
                    expected_order[expected_order_count] = ctx_tag_model[ctx];
                    expected_order_count = expected_order_count + 1;
                end
                sealed_groups = sealed_groups + 1;
                $fdisplay(ledger_fd, "L %0d %0d %0h", cycle_count,
                          launch_context_count, launch_contexts);
            end
            if (request_accept && ledger_enable) begin
                for (int slot = 0; slot < weight_request_context_count; slot++) begin
                    integer ctx, tag;
                    ctx = weight_request_contexts[slot*4 +: 4];
                    tag = ctx_tag_model[ctx];
                    for (int bank = 0; bank < BANKS; bank++) begin
                        if (weight_request_context_valid[slot*8 + bank]) begin
                            integer source;
                            integer signed value;
                            source = weight_request_bank_addr[bank*5 +: 5]
                                * BANKS + bank;
                            expected_count[tag] = expected_count[tag] + 1;
                            for (int lane = 0; lane < LANES; lane++) begin
                                value = model_weight(source, lane);
                                if (weight_request_context_subtract[slot*8 + bank])
                                    expected_acc[tag][lane]
                                        = expected_acc[tag][lane] - value;
                                else
                                    expected_acc[tag][lane]
                                        = expected_acc[tag][lane] + value;
                            end
                        end
                    end
                end
                $fdisplay(ledger_fd,
                    "R %0d %0h %0d %0h %0h %0h %0h %0h %0d",
                    cycle_count, weight_request_tag,
                    weight_request_context_count, weight_request_contexts,
                    weight_request_bank_valid, weight_request_bank_addr,
                    weight_request_context_valid,
                    weight_request_context_subtract, weight_request_last);
            end
            if (output_accept && ledger_enable) begin
                integer tag;
                tag = output_tag;
                if (outputs_seen >= expected_order_count
                        || tag != expected_order[outputs_seen])
                    $fatal(1, "M67 output order mismatch idx=%0d got=%0d exp=%0d",
                           outputs_seen, tag, expected_order[outputs_seen]);
                if (output_source_count !== expected_count[tag][8:0])
                    $fatal(1, "M67 count mismatch tag=%0d got=%0d exp=%0d",
                           tag, output_source_count, expected_count[tag]);
                for (int lane = 0; lane < LANES; lane++)
                    if ($signed(output_acc[lane*ACC_W +: ACC_W])
                            !== expected_acc[tag][lane])
                        $fatal(1, "M67 acc mismatch tag=%0d lane=%0d got=%0d exp=%0d",
                               tag, lane,
                               $signed(output_acc[lane*ACC_W +: ACC_W]),
                               expected_acc[tag][lane]);
                outputs_seen = outputs_seen + 1;
                $fdisplay(ledger_fd, "O %0d %0h %0d %0h", cycle_count,
                          output_tag, output_source_count, output_acc);
            end
            if (ledger_enable && (command_accept || launch_accept
                    || request_accept || output_accept))
                $fflush(ledger_fd);
            if (ledger_enable && response_accept
                    && (!dut.response_contract_valid
                        || dut.response_acc_overflow))
                $fatal(1,
                    "M67 response fault cycle=%0d got_tag=%0h exp_tag=%0h got_count=%0d exp_count=%0d got_ctx=%0h exp_ctx=%0h got_bv=%0h exp_bv=%0h overflow=%0d rsp_read=%0d rsp_write=%0d",
                    cycle_count, weight_response_tag, dut.expected_tag,
                    weight_response_context_count, dut.expected_count,
                    weight_response_contexts, dut.expected_contexts,
                    weight_response_bank_valid, dut.expected_bank_valid,
                    dut.response_acc_overflow, rsp_read, rsp_write);
            if (ledger_enable && weight_response_valid
                    && response_metadata_occupancy == 0)
                $fatal(1,
                    "M67 response-without-metadata cycle=%0d got_tag=%0h rsp_read=%0d rsp_write=%0d",
                    cycle_count, weight_response_tag, rsp_read, rsp_write);
            if (ledger_enable && command_accept
                    && |(command_add_bits & command_subtract_bits))
                $fatal(1,
                    "M67 overlapping-command fault cycle=%0d tag=%0h overlap=%0h",
                    cycle_count, command_tag,
                    command_add_bits & command_subtract_bits);
            if (ledger_enable && launch_accept && !dut.launch_legal)
                $fatal(1,
                    "M67 illegal-launch fault cycle=%0d count=%0d contexts=%0h allocated=%0h launched=%0h active=%0d final_rsp=%0d",
                    cycle_count, launch_context_count, launch_contexts,
                    dut.context_allocated_vector,
                    dut.context_launched_vector, group_active,
                    dut.final_response_success);
            if (ledger_enable && protocol_error)
                $fatal(1, "M67 unexpected legal-phase fault cycle=%0d",
                       cycle_count);
        end
    end

    initial begin
        logic [255:0] add_mask [0:3];
        logic [255:0] sub_mask [0:3];
        logic [3:0] ctx [0:15];
        logic [3:0] gctx [0:3];
        logic [3:0] c0, seam_zero_ctx, seam_old_ctx;
        logic [3:0] seam_next_ctx, seam_command_ctx;
        logic [3:0] zero_wait_old_ctx, zero_wait_next_ctx;
        logic [1823:0] seam_seed;
        integer tag_dummy, timeout, seed, manual_rsp_index;
        logic [3:0] first_reused_context;

        clk_core = 0; rst_core = 1;
        command_valid = 0; command_tag = 0;
        command_add_bits = 0; command_subtract_bits = 0;
        command_seed_acc = 0;
        launch_valid = 0; launch_context_count = 0; launch_contexts = 0;
        weight_request_ready = 0;
        weight_response_valid = 0; weight_response_tag = 0;
        weight_response_context_count = 0; weight_response_contexts = 0;
        weight_response_bank_valid = 0; weight_response_data = 0;
        output_ready = 0;
        rsp_write = 0; rsp_read = 0; automatic_responses = 0;
        random_response_gaps = 0; response_consumed_since_negedge = 0;
        random_backpressure = 0;
        expected_order_count = 0; outputs_seen = 0; legal_tag_next = 0;
        cycle_count = 0; ledger_enable = 1; sealed_groups = 0;
        saw_context16 = 0; saw_meta16 = 0; saw_complete16 = 0;
        saw_push4 = 0; saw_complete13_pop_push4 = 0;
        saw_meta_tail_wrap = 0; saw_complete_tail_wrap = 0;
        saw_k1 = 0; saw_k2 = 0; saw_k3 = 0; saw_k4 = 0;
        saw_k4_full = 0; saw_k4_partial = 0; saw_k4_no_share = 0;
        saw_request_stall = 0; saw_response_stall = 0;
        saw_output_stall = 0; saw_context_reuse = 0;
        saw_seam = 0; saw_seam_command = 0; saw_seam_output = 0;
        saw_seam_command_output = 0; saw_zero_next_wait = 0;
        seed = 32'h54c4_2026; seed = $urandom(seed);
        if (!$value$plusargs("LEDGER=%s", ledger_path))
            ledger_path = "m54_handshake_ledger.log";
        ledger_fd = $fopen(ledger_path, "w");
        if (ledger_fd == 0) $fatal(1, "M67 cannot open ledger");

        repeat (5) @(posedge clk_core);
        @(negedge clk_core); rst_core = 0;

        // Occupy all sixteen real 4-bit context IDs with zero groups.
        output_ready = 1;
        weight_request_ready = 1;
        automatic_responses = 1;
        for (int index = 0; index < 16; index++) begin
            add_mask[0] = '0; sub_mask[0] = '0;
            send_descriptor(add_mask[0], sub_mask[0], 0, ctx[index], tag_dummy);
        end
        if (context_occupancy != 16)
            $fatal(1, "M67 failed ctx16 occupancy: %0d", context_occupancy);
        launch_group(1, pack4(ctx[0], 0, 0, 0));
        launch_group(2, pack4(ctx[1], ctx[2], 0, 0));
        launch_group(3, pack4(ctx[3], ctx[4], ctx[5], 0));
        launch_group(4, pack4(ctx[6], ctx[7], ctx[8], ctx[9]));
        launch_group(4, pack4(ctx[10], ctx[11], ctx[12], ctx[13]));
        launch_group(2, pack4(ctx[14], ctx[15], 0, 0));
        wait_for_all_outputs();

        // Lowest-free allocator must reuse context ID zero after release.
        add_mask[0] = '0; sub_mask[0] = '0;
        send_descriptor(add_mask[0], sub_mask[0], 0,
                        first_reused_context, tag_dummy);
        if (first_reused_context != 0)
            $fatal(1, "M67 lowest-free context reuse failed got=%0d",
                   first_reused_context);
        saw_context_reuse = 1;
        launch_group(1, pack4(first_reused_context, 0, 0, 0));
        wait_for_all_outputs();

        // K4 fully shared row: one read per bank updates all four contexts.
        for (int slot = 0; slot < 4; slot++) begin
            add_mask[slot] = '0; sub_mask[slot] = '0;
            for (int bank = 0; bank < BANKS; bank++)
                add_mask[slot][1*BANKS + bank] = 1'b1;
            send_descriptor(add_mask[slot], sub_mask[slot], 0,
                            gctx[slot], tag_dummy);
        end
        launch_group(4, pack4(gctx[0], gctx[1], gctx[2], gctx[3]));
        wait_for_all_outputs();

        // K4 partial share; include independent subtract and bypass paths.
        for (int slot = 0; slot < 4; slot++) begin
            add_mask[slot] = '0; sub_mask[slot] = '0;
            for (int bank = 0; bank < BANKS; bank++) begin
                if (slot == 0) add_mask[slot][2*BANKS + bank] = 1;
                else if (slot == 1 && bank < 4)
                    sub_mask[slot][2*BANKS + bank] = 1;
                else if (slot == 1)
                    sub_mask[slot][3*BANKS + bank] = 1;
                else if (slot == 2 && (bank % 2) == 0)
                    add_mask[slot][2*BANKS + bank] = 1;
                else if (slot == 2)
                    add_mask[slot][4*BANKS + bank] = 1;
                else if (bank < 2)
                    sub_mask[slot][2*BANKS + bank] = 1;
                else
                    sub_mask[slot][5*BANKS + bank] = 1;
            end
            send_descriptor(add_mask[slot], sub_mask[slot], 0,
                            gctx[slot], tag_dummy);
        end
        launch_group(4, pack4(gctx[0], gctx[1], gctx[2], gctx[3]));
        wait_for_all_outputs();

        // K4 no-share cycle: each destination owns two disjoint banks.
        for (int slot = 0; slot < 4; slot++) begin
            add_mask[slot] = '0; sub_mask[slot] = '0;
            add_mask[slot][6*BANKS + slot] = 1;
            sub_mask[slot][6*BANKS + slot + 4] = 1;
            send_descriptor(add_mask[slot], sub_mask[slot], 0,
                            gctx[slot], tag_dummy);
        end
        launch_group(4, pack4(gctx[0], gctx[1], gctx[2], gctx[3]));
        wait_for_all_outputs();

        // Explicit nonzero K1, K2 and K3 request coverage.
        for (int count = 1; count <= 3; count++) begin
            for (int slot = 0; slot < count; slot++) begin
                add_mask[slot] = '0; sub_mask[slot] = '0;
                add_mask[slot][(7+slot)*BANKS + slot] = 1;
                send_descriptor(add_mask[slot], sub_mask[slot], 0,
                                gctx[slot], tag_dummy);
            end
            launch_group(count, pack4(gctx[0], gctx[1], gctx[2], 0));
            wait_for_all_outputs();
        end

        // Nonzero K2/K3 fully shared and partially shared relations.
        for (int count = 2; count <= 3; count++) begin
            for (int slot = 0; slot < count; slot++) begin
                add_mask[slot] = '0; sub_mask[slot] = '0;
                for (int bank = 0; bank < BANKS; bank++)
                    add_mask[slot][21*BANKS + bank] = 1;
                send_descriptor(add_mask[slot], sub_mask[slot], 0,
                                gctx[slot], tag_dummy);
            end
            launch_group(count, pack4(gctx[0], gctx[1], gctx[2], 0));
            wait_for_all_outputs();
            for (int slot = 0; slot < count; slot++) begin
                add_mask[slot] = '0; sub_mask[slot] = '0;
                for (int bank = 0; bank < BANKS; bank++) begin
                    if (slot == 0 || bank < (5-slot))
                        add_mask[slot][22*BANKS + bank] = 1;
                    else
                        sub_mask[slot][(23+slot)*BANKS + bank] = 1;
                end
                send_descriptor(add_mask[slot], sub_mask[slot], 0,
                                gctx[slot], tag_dummy);
            end
            launch_group(count, pack4(gctx[0], gctx[1], gctx[2], 0));
            wait_for_all_outputs();
        end

        // Real response-metadata FIFO: enqueue 16, stall, then drain while
        // simultaneous dequeue/enqueue wraps the 4-bit tail.
        automatic_responses = 0;
        random_backpressure = 0;
        weight_request_ready = 1;
        output_ready = 1;
        add_mask[0] = '1; sub_mask[0] = '0;
        send_descriptor(add_mask[0], sub_mask[0], 0, c0, tag_dummy);
        launch_group(1, pack4(c0, 0, 0, 0));
        timeout = 0;
        while (response_metadata_occupancy != 16 && timeout < 2000) begin
            @(posedge clk_core); timeout++;
        end
        if (response_metadata_occupancy != 16 || weight_request_valid)
            $fatal(1, "M67 metadata saturation failed occ=%0d req=%0d",
                   response_metadata_occupancy, weight_request_valid);
        automatic_responses = 1;
        random_response_gaps = 1;
        wait_for_all_outputs();

        // Random legal request/response/output stalls.
        automatic_responses = 1;
        random_response_gaps = 1;
        random_backpressure = 1;
        for (int slot = 0; slot < 4; slot++) begin
            add_mask[slot] = '0; sub_mask[slot] = '0;
            for (int row = 10; row < 18; row++)
                add_mask[slot][row*BANKS + ((slot+row) % BANKS)] = 1;
            send_descriptor(add_mask[slot], sub_mask[slot], 0,
                            gctx[slot], tag_dummy);
        end
        launch_group(4, pack4(gctx[0], gctx[1], gctx[2], gctx[3]));
        wait_for_all_outputs();
        random_backpressure = 0;
        weight_request_ready = 1;
        output_ready = 1;

        // Deterministic four-way seam pressure: an older zero-source output
        // pops while group A's final response retires, nonzero group B launches
        // and an unrelated command allocates a context on that same edge.
        automatic_responses = 0;
        output_ready = 0;
        add_mask[0] = '0; sub_mask[0] = '0;
        send_descriptor(add_mask[0], sub_mask[0], 0,
                        seam_zero_ctx, tag_dummy);
        launch_group(1, pack4(seam_zero_ctx, 0, 0, 0));
        add_mask[0] = '0; sub_mask[0] = '0;
        add_mask[0][18*BANKS] = 1;
        send_descriptor(add_mask[0], sub_mask[0], 0,
                        seam_old_ctx, tag_dummy);
        launch_group(1, pack4(seam_old_ctx, 0, 0, 0));
        timeout = 0;
        while (response_metadata_occupancy != 1 && timeout < 2000) begin
            @(posedge clk_core); timeout++;
        end
        if (response_metadata_occupancy != 1)
            $fatal(1, "M67 seam old response metadata not ready");
        add_mask[0] = '0; sub_mask[0] = '0;
        add_mask[0][19*BANKS + 1] = 1;
        send_descriptor(add_mask[0], sub_mask[0], 0,
                        seam_next_ctx, tag_dummy);
        manual_rsp_index = rsp_read;
        fork
            begin
                launch_group(1, pack4(seam_next_ctx, 0, 0, 0));
            end
            begin
                wait (launch_valid && !launch_ready
                      && response_metadata_occupancy == 1);
                @(negedge clk_core);
                make_seed(0, seam_seed);
                command_tag = legal_tag_next;
                command_add_bits = '0;
                command_subtract_bits = '0;
                command_seed_acc = seam_seed;
                command_valid = 1;
                weight_response_tag = rsp_tag_q[manual_rsp_index];
                weight_response_context_count = rsp_count_q[manual_rsp_index];
                weight_response_contexts = rsp_contexts_q[manual_rsp_index];
                weight_response_bank_valid = rsp_bank_valid_q[manual_rsp_index];
                weight_response_data = rsp_data_q[manual_rsp_index];
                weight_response_valid = 1;
                output_ready = 1;
                @(posedge clk_core);
                if (!(response_accept && launch_accept && command_accept
                        && output_accept))
                    $fatal(1, "M67 four-way seam concurrency missing rsp=%0d launch=%0d command=%0d output=%0d",
                           response_accept, launch_accept, command_accept,
                           output_accept);
                seam_command_ctx = command_accept_context;
                @(negedge clk_core);
                weight_response_valid = 0;
                command_valid = 0;
                legal_tag_next = legal_tag_next + 1;
                rsp_read = rsp_read + 1;
                // Keep the automatic response driver disabled through a full
                // edge after the manual dequeue.  Otherwise its negedge
                // consumed-response path can race this update and skip B.
                @(posedge clk_core);
                @(negedge clk_core);
                automatic_responses = 1;
            end
        join
        wait_for_all_outputs();
        launch_group(1, pack4(seam_command_ctx, 0, 0, 0));
        wait_for_all_outputs();

        // A zero-source next group cannot use the seam because it would require
        // a second completion push.  It must remain valid and accept later.
        automatic_responses = 0;
        add_mask[0] = '0; sub_mask[0] = '0;
        add_mask[0][20*BANKS + 2] = 1;
        send_descriptor(add_mask[0], sub_mask[0], 0,
                        zero_wait_old_ctx, tag_dummy);
        launch_group(1, pack4(zero_wait_old_ctx, 0, 0, 0));
        timeout = 0;
        while (response_metadata_occupancy != 1 && timeout < 2000) begin
            @(posedge clk_core); timeout++;
        end
        if (response_metadata_occupancy != 1)
            $fatal(1, "M67 zero-wait old response metadata not ready");
        add_mask[0] = '0; sub_mask[0] = '0;
        send_descriptor(add_mask[0], sub_mask[0], 0,
                        zero_wait_next_ctx, tag_dummy);
        fork
            begin
                launch_group(1, pack4(zero_wait_next_ctx, 0, 0, 0));
            end
            begin
                wait (launch_valid && !launch_ready);
                automatic_responses = 1;
            end
        join
        wait_for_all_outputs();

        // Fill complete FIFO to 13 with zero-source atomic groups.
        output_ready = 0;
        for (int group = 0; group < 3; group++) begin
            for (int slot = 0; slot < 4; slot++) begin
                add_mask[slot] = '0; sub_mask[slot] = '0;
                send_descriptor(add_mask[slot], sub_mask[slot], 0,
                                gctx[slot], tag_dummy);
            end
            launch_group(4, pack4(gctx[0], gctx[1], gctx[2], gctx[3]));
        end
        add_mask[0] = '0; sub_mask[0] = '0;
        send_descriptor(add_mask[0], sub_mask[0], 0, c0, tag_dummy);
        launch_group(1, pack4(c0, 0, 0, 0));
        if (complete_occupancy != 13)
            $fatal(1, "M67 complete FIFO expected13 got=%0d",
                   complete_occupancy);

        // Final K4 response needs four credits; at occupancy13 it stalls
        // until a same-cycle pop supplies the fourth, then atomically push4.
        automatic_responses = 1;
        random_response_gaps = 0;
        for (int slot = 0; slot < 4; slot++) begin
            add_mask[slot] = '0; sub_mask[slot] = '0;
            add_mask[slot][20*BANKS + slot] = 1;
            send_descriptor(add_mask[slot], sub_mask[slot], 0,
                            gctx[slot], tag_dummy);
        end
        launch_group(4, pack4(gctx[0], gctx[1], gctx[2], gctx[3]));
        timeout = 0;
        while (!(weight_response_valid && !weight_response_ready)
                && timeout < 2000) begin
            @(posedge clk_core); timeout++;
        end
        if (timeout == 2000)
            $fatal(1, "M67 final K4 response did not stall at occupancy13");
        @(negedge clk_core); output_ready = 1;
        @(posedge clk_core); #1;
        if (complete_occupancy != 16)
            $fatal(1, "M67 pop+push4 expected complete16 got=%0d",
                   complete_occupancy);
        wait_for_all_outputs();

        if (!saw_context16 || !saw_meta16 || !saw_complete16 || !saw_push4
                || !saw_complete13_pop_push4 || !saw_meta_tail_wrap
                || !saw_complete_tail_wrap || !saw_k1 || !saw_k2
                || !saw_k3 || !saw_k4 || !saw_k4_full || !saw_k4_partial
                || !saw_k4_no_share || !saw_request_stall
                || !saw_response_stall || !saw_output_stall
                || !saw_context_reuse || !saw_seam || !saw_seam_command
                || !saw_seam_output || !saw_seam_command_output
                || !saw_zero_next_wait)
            $fatal(1, "M67 legal coverage missing c16=%0d m16=%0d q16=%0d p4=%0d q13pp4=%0d mtw=%0d qtw=%0d k=%0d%0d%0d%0d rel=%0d%0d%0d stalls=%0d%0d%0d reuse=%0d",
                   saw_context16, saw_meta16, saw_complete16, saw_push4,
                   saw_complete13_pop_push4, saw_meta_tail_wrap,
                   saw_complete_tail_wrap, saw_k1, saw_k2, saw_k3, saw_k4,
                   saw_k4_full, saw_k4_partial, saw_k4_no_share,
                   saw_request_stall, saw_response_stall, saw_output_stall,
                   saw_context_reuse);

        $display("M67_SEAM_COVER seam=%0d command=%0d output=%0d command_output=%0d zero_wait=%0d",
                 saw_seam, saw_seam_command, saw_seam_output,
                 saw_seam_command_output, saw_zero_next_wait);

        $fdisplay(ledger_fd, "END commands=%0d outputs=%0d",
                  legal_tag_next, outputs_seen);
        sealed_commands = legal_tag_next;
        sealed_outputs = outputs_seen;
        sealed_requests = rsp_write;
        $fclose(ledger_fd);
        ledger_enable = 0;

        // Reset under a held request stall.
        reset_dut();
        add_mask[0] = '0; sub_mask[0] = '0;
        add_mask[0][40] = 1;
        send_descriptor(add_mask[0], sub_mask[0], 0, c0, tag_dummy);
        weight_request_ready = 0;
        launch_group(1, pack4(c0, 0, 0, 0));
        do @(posedge clk_core); while (!weight_request_valid);
        repeat (2) @(posedge clk_core);
        reset_dut();
        if (busy || protocol_error) $fatal(1, "M67 request-stall reset failed");

        // Reset under a held output stall.
        add_mask[0] = '0; sub_mask[0] = '0;
        send_descriptor(add_mask[0], sub_mask[0], 0, c0, tag_dummy);
        output_ready = 0;
        launch_group(1, pack4(c0, 0, 0, 0));
        do @(posedge clk_core); while (!output_valid);
        reset_dut();
        if (busy || protocol_error) $fatal(1, "M67 output-stall reset failed");

        // Unexpected response with empty metadata.
        @(negedge clk_core);
        weight_response_valid = 1;
        weight_response_tag = '1;
        weight_response_context_count = 1;
        weight_response_contexts = 0;
        weight_response_bank_valid = 1;
        @(posedge clk_core); @(negedge clk_core); weight_response_valid = 0;
        require_fail_closed("unexpected_response");

        // Duplicate context in K4 launch.
        reset_dut();
        for (int slot = 0; slot < 4; slot++) begin
            add_mask[slot] = '0; sub_mask[slot] = '0;
            send_descriptor(add_mask[slot], sub_mask[slot], 0,
                            gctx[slot], tag_dummy);
        end
        launch_context_count = 4;
        launch_contexts = pack4(gctx[0], gctx[1], gctx[1], gctx[3]);
        @(negedge clk_core); launch_valid = 1;
        @(posedge clk_core); @(negedge clk_core); launch_valid = 0;
        require_fail_closed("duplicate_context_launch");

        // Stale request tag mismatch.
        reset_dut();
        prepare_one_request(c0, tag_dummy);
        @(negedge clk_core);
        weight_response_tag = rsp_tag_q[0] - 1'b1;
        weight_response_context_count = rsp_count_q[0];
        weight_response_contexts = rsp_contexts_q[0];
        weight_response_bank_valid = rsp_bank_valid_q[0];
        weight_response_data = rsp_data_q[0];
        weight_response_valid = 1;
        @(posedge clk_core); @(negedge clk_core); weight_response_valid = 0;
        require_fail_closed("stale_response_tag");

        // Response context/count mismatch.
        reset_dut();
        prepare_one_request(c0, tag_dummy);
        @(negedge clk_core);
        weight_response_tag = rsp_tag_q[0];
        weight_response_context_count = rsp_count_q[0] + 1'b1;
        weight_response_contexts = rsp_contexts_q[0] ^ 16'h0010;
        weight_response_bank_valid = rsp_bank_valid_q[0];
        weight_response_data = rsp_data_q[0];
        weight_response_valid = 1;
        @(posedge clk_core); @(negedge clk_core); weight_response_valid = 0;
        require_fail_closed("response_context_count_mismatch");

        // Response bank-valid mismatch.
        reset_dut();
        prepare_one_request(c0, tag_dummy);
        @(negedge clk_core);
        weight_response_tag = rsp_tag_q[0];
        weight_response_context_count = rsp_count_q[0];
        weight_response_contexts = rsp_contexts_q[0];
        weight_response_bank_valid = rsp_bank_valid_q[0] ^ 8'h80;
        weight_response_data = rsp_data_q[0];
        weight_response_valid = 1;
        @(posedge clk_core); @(negedge clk_core); weight_response_valid = 0;
        require_fail_closed("response_bank_mismatch");

        // Overlapping add/sub command masks.
        reset_dut();
        add_mask[0] = '0; sub_mask[0] = '0;
        add_mask[0][17] = 1; sub_mask[0][17] = 1;
        command_tag = '1; command_add_bits = add_mask[0];
        command_subtract_bits = sub_mask[0]; command_seed_acc = 0;
        do @(negedge clk_core); while (!command_ready);
        command_valid = 1; @(posedge clk_core);
        @(negedge clk_core); command_valid = 0;
        require_fail_closed("overlapping_masks");

        // Positive signed19 overflow: max seed subtracts -128 source zero.
        reset_dut();
        automatic_responses = 1; weight_request_ready = 1; output_ready = 1;
        add_mask[0] = '0; sub_mask[0] = '0; sub_mask[0][0] = 1;
        send_descriptor(add_mask[0], sub_mask[0], 1, c0, tag_dummy);
        launch_group(1, pack4(c0, 0, 0, 0));
        do @(posedge clk_core); while (!protocol_error);
        require_fail_closed("positive_overflow");

        // Negative signed19 overflow: min seed adds -128 source zero.
        reset_dut();
        automatic_responses = 1; weight_request_ready = 1; output_ready = 1;
        add_mask[0] = '0; sub_mask[0] = '0; add_mask[0][0] = 1;
        send_descriptor(add_mask[0], sub_mask[0], -1, c0, tag_dummy);
        launch_group(1, pack4(c0, 0, 0, 0));
        do @(posedge clk_core); while (!protocol_error);
        require_fail_closed("negative_overflow");

`ifdef SVA_RUNTIME_ENABLED
        $display("M67_SVA_BOUND=1");
`else
        $display("M67_SVA_BOUND=0");
`endif
        $display("M67_ATTACKS reset_request_stall=1 reset_output_stall=1 unexpected_response=1 duplicate_context_launch=1 stale_response_tag=1 response_context_count_mismatch=1 response_bank_mismatch=1 overlapping_masks=1 positive_overflow=1 negative_overflow=1");
        $display("PASS M67 K4_CTX16_ATOMIC_UNION commands=%0d outputs=%0d groups=%0d requests=%0d context16=%0d meta16=%0d complete16=%0d push4=%0d pop13push4=%0d",
                 sealed_commands, sealed_outputs, sealed_groups,
                 sealed_requests, saw_context16, saw_meta16,
                 saw_complete16, saw_push4, saw_complete13_pop_push4);
        $finish;
    end
endmodule

`default_nettype wire
