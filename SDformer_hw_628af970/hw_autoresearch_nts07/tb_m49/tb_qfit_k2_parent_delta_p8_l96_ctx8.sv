`timescale 1ns/1ps
`default_nettype none

module tb_qfit_k2_parent_delta_p8_l96_ctx8;
    localparam int TILE_BITS = 256;
    localparam int BANKS = 8;
    localparam int LANES = 96;
    localparam int ACC_W = 19;
    localparam int TAG_W = 48;
    localparam int MAX_TAGS = 128;
    localparam int MAX_REQS = 512;

    logic clk_core, rst_core;
    logic command_valid, command_ready, command_accept;
    logic [TAG_W-1:0] command_tag;
    logic [TILE_BITS-1:0] command_add_bits, command_subtract_bits;
    logic [LANES*ACC_W-1:0] command_seed_acc;
    logic [2:0] command_accept_context;
    logic launch_valid, launch_ready, launch_context1_valid, launch_accept;
    logic [2:0] launch_context0, launch_context1;
    logic weight_request_valid, weight_request_ready, weight_request_last;
    logic [BANKS-1:0] weight_request_bank_valid;
    logic [BANKS*5-1:0] weight_request_bank_addr;
    logic [2:0] weight_request_context0, weight_request_context1;
    logic weight_request_context1_valid;
    logic [BANKS-1:0] weight_request_context0_valid;
    logic [BANKS-1:0] weight_request_context0_subtract;
    logic [BANKS-1:0] weight_request_context1_valid_by_bank;
    logic [BANKS-1:0] weight_request_context1_subtract;
    logic request_accept;
    logic weight_response_valid, weight_response_ready, response_accept;
    logic [2:0] weight_response_context0, weight_response_context1;
    logic weight_response_context1_valid;
    logic [BANKS-1:0] weight_response_bank_valid;
    logic [BANKS*LANES*8-1:0] weight_response_data;
    logic output_valid, output_ready, output_accept;
    logic [TAG_W-1:0] output_tag;
    logic [8:0] output_source_count;
    logic [LANES*ACC_W-1:0] output_acc;
    logic protocol_error, busy, group_active;
    logic [3:0] context_occupancy;
    logic [4:0] response_metadata_occupancy, complete_occupancy;

    logic signed [ACC_W-1:0] expected_acc [0:MAX_TAGS-1][0:LANES-1];
    integer expected_count [0:MAX_TAGS-1];
    integer context_tag [0:7];
    integer expected_order [0:MAX_TAGS-1];
    integer expected_order_count, outputs_seen, legal_tag_next;
    integer sealed_legal_tags, sealed_legal_outputs, sealed_legal_requests;
    integer cycle_count, ledger_fd;
    string ledger_path;
    logic ledger_enable;

    logic [2:0] rsp_ctx0_q [0:MAX_REQS-1];
    logic rsp_use1_q [0:MAX_REQS-1];
    logic [2:0] rsp_ctx1_q [0:MAX_REQS-1];
    logic [7:0] rsp_bank_valid_q [0:MAX_REQS-1];
    logic [BANKS*LANES*8-1:0] rsp_data_q [0:MAX_REQS-1];
    integer rsp_write, rsp_read;
    logic automatic_responses, random_response_gaps;
    logic response_consumed_since_negedge;
    logic random_backpressure;
    logic saw_context8, saw_meta16, saw_complete16;
    logic saw_meta_full_pop_push, saw_complete_pop_push2;
    logic saw_request_stall, saw_response_stall, saw_output_stall;
    logic [2:0] captured_req_ctx0, captured_req_ctx1;
    logic captured_req_use1;
    logic [7:0] captured_req_bank_valid;
    logic [BANKS*LANES*8-1:0] captured_req_data;

    qfit_k2_parent_delta_p8_l96_ctx8 dut (.*);

    function automatic logic signed [7:0] model_weight(
        input integer source, input integer lane);
        integer raw;
        begin
            if (source < BANKS)
                model_weight = -8'sd128;
            else begin
                raw = ((source * 37 + lane * 13 + 19) % 255) - 127;
                model_weight = raw[7:0];
            end
        end
    endfunction

    function automatic integer popcount256(input logic [255:0] value);
        integer result;
        begin
            result = 0;
            for (int bit_index = 0; bit_index < 256; bit_index++)
                result += value[bit_index];
            popcount256 = result;
        end
    endfunction

    task automatic make_mask(
        input integer pattern,
        output logic [255:0] add_mask,
        output logic [255:0] sub_mask);
        begin
            add_mask = '0;
            sub_mask = '0;
            case (pattern)
                0: begin end
                1: for (int source = 0; source < 40; source++)
                       add_mask[source] = 1'b1;
                2: for (int source = 0; source < 40; source++)
                       sub_mask[source] = 1'b1;
                3: begin
                    add_mask[0] = 1; add_mask[8] = 1;
                    add_mask[1] = 1; sub_mask[9] = 1;
                    add_mask[18] = 1;
                end
                4: begin
                    sub_mask[0] = 1; add_mask[16] = 1;
                    add_mask[2] = 1; add_mask[10] = 1;
                    sub_mask[27] = 1;
                end
                5: begin
                    add_mask[0] = 1; sub_mask[8] = 1; add_mask[16] = 1;
                end
                6: begin
                    add_mask[24] = 1; sub_mask[32] = 1; add_mask[40] = 1;
                end
                7: for (int source = 0; source < 256; source++) begin
                    if ((source % 3) == 0) add_mask[source] = 1'b1;
                    else if ((source % 3) == 1) sub_mask[source] = 1'b1;
                end
                8: for (int source = 0; source < 256; source++)
                       add_mask[source] = 1'b1;
                9: begin sub_mask[0] = 1'b1; end
                10: begin add_mask[0] = 1'b1; end
                default: begin
                    for (int source = 0; source < 256; source++) begin
                        if (((source * 17 + pattern * 11) % 29) == 0)
                            add_mask[source] = 1'b1;
                        else if (((source * 13 + pattern * 7) % 31) == 0)
                            sub_mask[source] = 1'b1;
                    end
                end
            endcase
        end
    endtask

    task automatic send_descriptor(
        input logic [255:0] add_mask,
        input logic [255:0] sub_mask,
        input integer seed_mode,
        output logic [2:0] accepted_context,
        output integer accepted_tag);
        integer seed, total;
        begin
            accepted_tag = legal_tag_next;
            legal_tag_next = legal_tag_next + 1;
            command_tag = TAG_W'(accepted_tag);
            command_add_bits = add_mask;
            command_subtract_bits = sub_mask;
            command_seed_acc = '0;
            expected_count[accepted_tag] = popcount256(add_mask | sub_mask);
            for (int lane = 0; lane < LANES; lane++) begin
                if (seed_mode == 1) seed = (1 << (ACC_W-1)) - 1;
                else if (seed_mode == -1) seed = -(1 << (ACC_W-1));
                else seed = ((accepted_tag * 17 + lane * 5) % 101) - 50;
                command_seed_acc[lane*ACC_W +: ACC_W] = ACC_W'(seed);
                total = seed;
                for (int source = 0; source < 256; source++) begin
                    if (add_mask[source])
                        total += $signed(model_weight(source, lane));
                    if (sub_mask[source])
                        total -= $signed(model_weight(source, lane));
                end
                expected_acc[accepted_tag][lane] = ACC_W'(total);
            end
            do @(negedge clk_core); while (!command_ready);
            command_valid = 1'b1;
            @(posedge clk_core);
            if (!command_accept) $fatal(1, "M49 command handshake lost");
            accepted_context = command_accept_context;
            context_tag[accepted_context] = accepted_tag;
            @(negedge clk_core);
            command_valid = 1'b0;
        end
    endtask

    task automatic launch_group(
        input logic [2:0] context0,
        input logic use_context1,
        input logic [2:0] context1);
        begin
            launch_context0 = context0;
            launch_context1_valid = use_context1;
            launch_context1 = context1;
            do @(negedge clk_core); while (!launch_ready);
            launch_valid = 1'b1;
            @(posedge clk_core);
            if (!launch_accept) $fatal(1, "M49 launch handshake lost");
            expected_order[expected_order_count] = context_tag[context0];
            expected_order_count = expected_order_count + 1;
            if (use_context1) begin
                expected_order[expected_order_count] = context_tag[context1];
                expected_order_count = expected_order_count + 1;
            end
            @(negedge clk_core);
            launch_valid = 1'b0;
        end
    endtask

    task automatic wait_for_all_outputs;
        integer timeout;
        begin
            timeout = 0;
            while (outputs_seen != expected_order_count && timeout < 200000) begin
                @(posedge clk_core);
                timeout++;
            end
            if (outputs_seen != expected_order_count)
                $fatal(1, "M49 output timeout got=%0d exp=%0d",
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
            rsp_write = 0;
            rsp_read = 0;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
        end
    endtask

    task automatic require_fail_closed(input string attack);
        begin
            @(negedge clk_core);
            if (!protocol_error)
                $fatal(1, "M49 attack did not fault: %s", attack);
            repeat (2) @(posedge clk_core);
            @(negedge clk_core);
            if (!protocol_error || command_ready || launch_ready
                    || weight_request_valid || weight_response_ready
                    || output_valid)
                $fatal(1, "M49 fault not sticky/fail-closed: %s", attack);
        end
    endtask

    always #1.5 clk_core = ~clk_core;

    // Capture every accepted memory request into an ordered response queue.
    always @(posedge clk_core) begin
        response_consumed_since_negedge <= response_accept;
        if (!rst_core && request_accept) begin
            if (rsp_write >= MAX_REQS) $fatal(1, "M49 response model overflow");
            rsp_ctx0_q[rsp_write] <= weight_request_context0;
            rsp_use1_q[rsp_write] <= weight_request_context1_valid;
            rsp_ctx1_q[rsp_write] <= weight_request_context1;
            rsp_bank_valid_q[rsp_write] <= weight_request_bank_valid;
            for (int bank = 0; bank < BANKS; bank++) begin
                integer source;
                source = weight_request_bank_addr[bank*5 +: 5] * BANKS + bank;
                for (int lane = 0; lane < LANES; lane++)
                    rsp_data_q[rsp_write][(bank*LANES+lane)*8 +: 8]
                        <= model_weight(source, lane);
            end
            captured_req_ctx0 <= weight_request_context0;
            captured_req_use1 <= weight_request_context1_valid;
            captured_req_ctx1 <= weight_request_context1;
            captured_req_bank_valid <= weight_request_bank_valid;
            for (int bank = 0; bank < BANKS; bank++) begin
                integer captured_source;
                captured_source = weight_request_bank_addr[bank*5 +: 5]
                    * BANKS + bank;
                for (int lane = 0; lane < LANES; lane++)
                    captured_req_data[(bank*LANES+lane)*8 +: 8]
                        <= model_weight(captured_source, lane);
            end
            rsp_write <= rsp_write + 1;
        end
    end

    // Response payload remains stable until the DUT accepts it.
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
                weight_response_context0 = rsp_ctx0_q[rsp_read];
                weight_response_context1_valid = rsp_use1_q[rsp_read];
                weight_response_context1 = rsp_ctx1_q[rsp_read];
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

    always @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (context_occupancy == 8) saw_context8 <= 1'b1;
            if (response_metadata_occupancy == 16) saw_meta16 <= 1'b1;
            if (complete_occupancy == 16) saw_complete16 <= 1'b1;
            if (response_metadata_occupancy == 16
                    && request_accept && response_accept)
                saw_meta_full_pop_push <= 1'b1;
            if (complete_occupancy == 15 && output_accept
                    && dut.complete_push_count == 2)
                saw_complete_pop_push2 <= 1'b1;
            if (weight_request_valid && !weight_request_ready)
                saw_request_stall <= 1'b1;
            if (weight_response_valid && !weight_response_ready)
                saw_response_stall <= 1'b1;
            if (output_valid && !output_ready)
                saw_output_stall <= 1'b1;

            if (ledger_enable && command_accept)
                $fdisplay(ledger_fd, "C %0d %0d %0h %0h %0h %0h",
                    cycle_count, command_accept_context, command_tag,
                    command_add_bits, command_subtract_bits, command_seed_acc);
            if (ledger_enable && launch_accept)
                $fdisplay(ledger_fd, "L %0d %0d %0d %0d",
                    cycle_count, launch_context0,
                    launch_context1_valid, launch_context1);
            if (ledger_enable && request_accept)
                $fdisplay(ledger_fd,
                    "R %0d %0d %0d %0d %0h %0h %0h %0h %0h %0h %0d",
                    cycle_count, weight_request_context0,
                    weight_request_context1_valid, weight_request_context1,
                    weight_request_bank_valid, weight_request_bank_addr,
                    weight_request_context0_valid,
                    weight_request_context0_subtract,
                    weight_request_context1_valid_by_bank,
                    weight_request_context1_subtract, weight_request_last);
            if (ledger_enable && output_accept)
                $fdisplay(ledger_fd, "O %0d %0h %0d %0h",
                    cycle_count, output_tag, output_source_count, output_acc);
            if (ledger_enable && (command_accept || launch_accept
                    || request_accept || output_accept))
                $fflush(ledger_fd);

            if (ledger_enable && protocol_error)
                $fatal(1, "M49 unexpected protocol_error during legal phase cycle=%0d",
                       cycle_count);

            if (output_accept && ledger_enable) begin
                integer tag;
                tag = output_tag;
                if (outputs_seen >= expected_order_count
                        || tag != expected_order[outputs_seen])
                    $fatal(1, "M49 output order mismatch index=%0d got=%0d exp=%0d",
                           outputs_seen, tag, expected_order[outputs_seen]);
                if (output_source_count !== expected_count[tag][8:0])
                    $fatal(1, "M49 count mismatch tag=%0d got=%0d exp=%0d",
                           tag, output_source_count, expected_count[tag]);
                for (int lane = 0; lane < LANES; lane++)
                    if ($signed(output_acc[lane*ACC_W +: ACC_W])
                            !== expected_acc[tag][lane])
                        $fatal(1, "M49 accumulator mismatch tag=%0d lane=%0d got=%0d exp=%0d",
                               tag, lane,
                               $signed(output_acc[lane*ACC_W +: ACC_W]),
                               expected_acc[tag][lane]);
                outputs_seen <= outputs_seen + 1;
            end
        end
    end

    initial begin
        logic [255:0] add_mask, sub_mask;
        logic [2:0] ctx [0:7];
        logic [2:0] c0, c1;
        integer tag_dummy;
        integer timeout;
        integer seed;

        clk_core = 0; rst_core = 1;
        command_valid = 0; command_tag = 0;
        command_add_bits = 0; command_subtract_bits = 0;
        command_seed_acc = 0;
        launch_valid = 0; launch_context0 = 0;
        launch_context1_valid = 0; launch_context1 = 0;
        weight_request_ready = 0;
        weight_response_valid = 0; weight_response_context0 = 0;
        weight_response_context1_valid = 0; weight_response_context1 = 0;
        weight_response_bank_valid = 0; weight_response_data = 0;
        output_ready = 0;
        expected_order_count = 0; outputs_seen = 0; legal_tag_next = 0;
        cycle_count = 0; rsp_write = 0; rsp_read = 0;
        automatic_responses = 0; random_response_gaps = 0;
        response_consumed_since_negedge = 0;
        random_backpressure = 0; ledger_enable = 1;
        saw_context8 = 0; saw_meta16 = 0; saw_complete16 = 0;
        saw_meta_full_pop_push = 0; saw_complete_pop_push2 = 0;
        saw_request_stall = 0; saw_response_stall = 0; saw_output_stall = 0;
        sealed_legal_tags = 0; sealed_legal_outputs = 0;
        sealed_legal_requests = 0;
        seed = 32'h49c8_2026;
        seed = $urandom(seed);
        if (!$value$plusargs("LEDGER=%s", ledger_path))
            ledger_path = "m49_handshake_ledger.log";
        ledger_fd = $fopen(ledger_path, "w");
        if (ledger_fd == 0) $fatal(1, "M49 cannot open ledger");

        repeat (5) @(posedge clk_core);
        @(negedge clk_core); rst_core = 0;

        // Eight resident contexts, then K1 zero, exact fully shared K2,
        // partial K2, same-bank/different-row K2, and a long-burst K1.
        output_ready = 0;
        for (int index = 0; index < 8; index++) begin
            make_mask(index, add_mask, sub_mask);
            send_descriptor(add_mask, sub_mask, 0, ctx[index], tag_dummy);
        end
        if (context_occupancy != 8)
            $fatal(1, "M49 did not fill eight contexts: %0d", context_occupancy);
        automatic_responses = 1;
        random_response_gaps = 1;
        random_backpressure = 1;
        launch_group(ctx[0], 0, 0);
        launch_group(ctx[1], 1, ctx[2]);
        launch_group(ctx[3], 1, ctx[4]);
        launch_group(ctx[5], 1, ctx[6]);
        launch_group(ctx[7], 0, 0);
        wait_for_all_outputs();

        // Fill the ordered metadata FIFO to 16 and force a full pop/push.
        random_backpressure = 0;
        weight_request_ready = 1;
        output_ready = 1;
        automatic_responses = 0;
        make_mask(8, add_mask, sub_mask);
        send_descriptor(add_mask, sub_mask, 0, c0, tag_dummy);
        send_descriptor(add_mask, sub_mask, 0, c1, tag_dummy);
        launch_group(c0, 1, c1);
        timeout = 0;
        while (response_metadata_occupancy != 16 && timeout < 1000) begin
            @(posedge clk_core); timeout++;
        end
        if (response_metadata_occupancy != 16 || weight_request_valid)
            $fatal(1, "M49 metadata FIFO saturation failed occ=%0d valid=%0d",
                   response_metadata_occupancy, weight_request_valid);
        automatic_responses = 1;
        random_response_gaps = 0;
        wait_for_all_outputs();

        // Fill complete FIFO with atomic zero-source K2 groups.
        output_ready = 0;
        automatic_responses = 1;
        for (int pair = 0; pair < 8; pair++) begin
            make_mask(0, add_mask, sub_mask);
            send_descriptor(add_mask, sub_mask, 0, c0, tag_dummy);
            send_descriptor(add_mask, sub_mask, 0, c1, tag_dummy);
            launch_group(c0, 1, c1);
        end
        if (complete_occupancy != 16)
            $fatal(1, "M49 complete FIFO did not fill: %0d", complete_occupancy);
        make_mask(3, add_mask, sub_mask);
        send_descriptor(add_mask, sub_mask, 0, c0, tag_dummy);
        make_mask(4, add_mask, sub_mask);
        send_descriptor(add_mask, sub_mask, 0, c1, tag_dummy);
        launch_group(c0, 1, c1);
        timeout = 0;
        while (!(weight_response_valid && !weight_response_ready)
                && timeout < 1000) begin
            @(posedge clk_core); timeout++;
        end
        if (timeout == 1000)
            $fatal(1, "M49 final response did not stall on full complete FIFO");
        output_ready = 1;
        wait_for_all_outputs();
        if (!saw_context8 || !saw_meta16 || !saw_complete16
                || !saw_meta_full_pop_push || !saw_complete_pop_push2
                || !saw_request_stall || !saw_response_stall
                || !saw_output_stall)
            $fatal(1, "M49 legal coverage missing c8=%0d m16=%0d q16=%0d mpp=%0d cpp2=%0d rs=%0d rps=%0d os=%0d",
                   saw_context8, saw_meta16, saw_complete16,
                   saw_meta_full_pop_push, saw_complete_pop_push2,
                   saw_request_stall, saw_response_stall, saw_output_stall);

        $fdisplay(ledger_fd, "END legal_tags=%0d outputs=%0d",
                  legal_tag_next, outputs_seen);
        sealed_legal_tags = legal_tag_next;
        sealed_legal_outputs = outputs_seen;
        sealed_legal_requests = rsp_write;
        $fclose(ledger_fd);
        ledger_enable = 0;

        // Reset while a request is held under backpressure.
        reset_dut();
        make_mask(3, add_mask, sub_mask);
        send_descriptor(add_mask, sub_mask, 0, c0, tag_dummy);
        weight_request_ready = 0;
        launch_group(c0, 0, 0);
        do @(posedge clk_core); while (!weight_request_valid);
        repeat (2) @(posedge clk_core);
        reset_dut();
        if (busy || protocol_error) $fatal(1, "M49 request-stall reset failed");

        // Reset while a completed vector is held at the output.
        make_mask(0, add_mask, sub_mask);
        send_descriptor(add_mask, sub_mask, 0, c0, tag_dummy);
        output_ready = 0;
        launch_group(c0, 0, 0);
        do @(posedge clk_core); while (!output_valid);
        reset_dut();
        if (busy || protocol_error) $fatal(1, "M49 output-stall reset failed");

        // Unexpected response with empty metadata.
        @(negedge clk_core);
        weight_response_valid = 1;
        weight_response_context0 = 0;
        weight_response_context1_valid = 0;
        weight_response_bank_valid = 1;
        @(posedge clk_core);
        @(negedge clk_core);
        weight_response_valid = 0;
        require_fail_closed("unexpected_response");

        // Same context named twice in a K2 launch.
        reset_dut();
        make_mask(0, add_mask, sub_mask);
        send_descriptor(add_mask, sub_mask, 0, c0, tag_dummy);
        launch_context0 = c0; launch_context1_valid = 1; launch_context1 = c0;
        @(negedge clk_core); launch_valid = 1;
        @(posedge clk_core); @(negedge clk_core); launch_valid = 0;
        require_fail_closed("duplicate_launch_pair");

        // Relaunching a context after its atomic zero completion is illegal.
        reset_dut();
        output_ready = 0;
        make_mask(0, add_mask, sub_mask);
        send_descriptor(add_mask, sub_mask, 0, c0, tag_dummy);
        launch_group(c0, 0, 0);
        launch_context0 = c0; launch_context1_valid = 0;
        @(negedge clk_core); launch_valid = 1;
        @(posedge clk_core); @(negedge clk_core); launch_valid = 0;
        require_fail_closed("duplicate_relaunch_released_context");

        // Live response context mismatch.
        reset_dut();
        weight_request_ready = 1;
        make_mask(3, add_mask, sub_mask);
        send_descriptor(add_mask, sub_mask, 0, c0, tag_dummy);
        launch_group(c0, 0, 0);
        do @(posedge clk_core); while (response_metadata_occupancy == 0);
        @(negedge clk_core);
        weight_response_context0 = captured_req_ctx0 ^ 3'b001;
        weight_response_context1_valid = captured_req_use1;
        weight_response_context1 = captured_req_ctx1;
        weight_response_bank_valid = captured_req_bank_valid;
        weight_response_data = captured_req_data;
        weight_response_valid = 1;
        @(posedge clk_core); @(negedge clk_core); weight_response_valid = 0;
        require_fail_closed("response_context_mismatch");

        // K2 second response-context mismatch is independently checked.
        reset_dut();
        weight_request_ready = 1;
        make_mask(3, add_mask, sub_mask);
        send_descriptor(add_mask, sub_mask, 0, c0, tag_dummy);
        make_mask(4, add_mask, sub_mask);
        send_descriptor(add_mask, sub_mask, 0, c1, tag_dummy);
        launch_group(c0, 1, c1);
        do @(posedge clk_core); while (response_metadata_occupancy == 0);
        @(negedge clk_core);
        weight_response_context0 = captured_req_ctx0;
        weight_response_context1_valid = captured_req_use1;
        weight_response_context1 = captured_req_ctx1 ^ 3'b001;
        weight_response_bank_valid = captured_req_bank_valid;
        weight_response_data = captured_req_data;
        weight_response_valid = 1;
        @(posedge clk_core); @(negedge clk_core); weight_response_valid = 0;
        require_fail_closed("response_context1_mismatch");

        // Live response bank-valid mismatch.
        reset_dut();
        weight_request_ready = 1;
        make_mask(3, add_mask, sub_mask);
        send_descriptor(add_mask, sub_mask, 0, c0, tag_dummy);
        launch_group(c0, 0, 0);
        do @(posedge clk_core); while (response_metadata_occupancy == 0);
        @(negedge clk_core);
        weight_response_context0 = captured_req_ctx0;
        weight_response_context1_valid = captured_req_use1;
        weight_response_context1 = captured_req_ctx1;
        weight_response_bank_valid = captured_req_bank_valid ^ 8'h80;
        weight_response_data = captured_req_data;
        weight_response_valid = 1;
        @(posedge clk_core); @(negedge clk_core); weight_response_valid = 0;
        require_fail_closed("response_bank_mismatch");

        // Overlapping signed masks.
        reset_dut();
        add_mask = 0; sub_mask = 0; add_mask[17] = 1; sub_mask[17] = 1;
        command_tag = '1; command_add_bits = add_mask;
        command_subtract_bits = sub_mask; command_seed_acc = 0;
        do @(negedge clk_core); while (!command_ready);
        command_valid = 1; @(posedge clk_core);
        @(negedge clk_core); command_valid = 0;
        require_fail_closed("overlapping_masks");

        // Positive overflow: maximum seed subtracts a -128 row.
        reset_dut();
        weight_request_ready = 1; output_ready = 1;
        automatic_responses = 1; random_response_gaps = 0;
        make_mask(9, add_mask, sub_mask);
        send_descriptor(add_mask, sub_mask, 1, c0, tag_dummy);
        launch_group(c0, 0, 0);
        do @(posedge clk_core); while (!protocol_error);
        require_fail_closed("positive_overflow");

        // Negative overflow: minimum seed adds a -128 row.
        reset_dut();
        weight_request_ready = 1; output_ready = 1;
        automatic_responses = 1; random_response_gaps = 0;
        make_mask(10, add_mask, sub_mask);
        send_descriptor(add_mask, sub_mask, -1, c0, tag_dummy);
        launch_group(c0, 0, 0);
        do @(posedge clk_core); while (!protocol_error);
        require_fail_closed("negative_overflow");

`ifdef SVA_RUNTIME_ENABLED
        $display("M49_SVA_BOUND=1");
`else
        $display("M49_SVA_BOUND=0");
`endif
        $display("M49_ATTACKS reset_request_stall=1 reset_output_stall=1 unexpected_response=1 duplicate_launch_pair=1 duplicate_relaunch=1 response_context0_mismatch=1 response_context1_mismatch=1 response_bank_mismatch=1 overlapping_masks=1 positive_overflow=1 negative_overflow=1");
        $display("PASS M49 K2_CTX8_ATOMIC_DUAL_ENQUEUE legal_tags=%0d outputs=%0d requests=%0d context8=%0d meta16=%0d complete16=%0d",
                 sealed_legal_tags, sealed_legal_outputs,
                 sealed_legal_requests,
                 saw_context8, saw_meta16, saw_complete16);
        $finish;
    end
endmodule

`default_nettype wire
