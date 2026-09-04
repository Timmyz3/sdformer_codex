`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off BLKSEQ */
/* verilator lint_off UNUSEDSIGNAL */

module tb_gatestack_dctf32_bank_executor;
    localparam int BANK_ID = 1;
    localparam int TOKENS = 18;
    localparam int OUT_TILE = 32;
    localparam int GATE_W = 9;
    localparam int WEIGHT_W = 8;
    localparam int PRODUCT_W = 17;
    localparam int GROUP_TAG_W = 12;
    localparam int CMD_SEQUENCE_W = 8;
    localparam int ISSUE_SEQ_W = 6;
    localparam int INPUT_CH_W = 7;
    localparam int LANE_ID_W = 5;
    localparam int TOKEN_ID_W = 5;
    localparam int OUTPUT_TILE_W = 6;
    localparam int LOGICAL_SUPERTILE_W = 4;
    localparam int EPOCH_W = 3;

    logic clk_core;
    logic rst_core;
    logic flush;
    logic clear_error;
    logic cmd_valid;
    logic cmd_ready;
    logic [GROUP_TAG_W-1:0] cmd_group_tag;
    logic [CMD_SEQUENCE_W-1:0] cmd_sequence;
    logic [ISSUE_SEQ_W-1:0] cmd_term_issue_seq;
    logic cmd_term_first;
    logic cmd_term_last;
    logic cmd_head_last;
    logic [INPUT_CH_W-1:0] cmd_input_channel;
    logic [GATE_W-1:0] cmd_gate_code;
    logic [LANE_ID_W-1:0] cmd_lane_id;
    logic [TOKEN_ID_W-1:0] cmd_destination_token;
    logic [LOGICAL_SUPERTILE_W-1:0] logical_supertile;
    logic weight_req_valid;
    logic weight_req_ready;
    logic [GROUP_TAG_W-1:0] weight_req_tag;
    logic [INPUT_CH_W-1:0] weight_req_input_channel;
    logic [OUTPUT_TILE_W-1:0] weight_req_output_tile;
    logic [EPOCH_W-1:0] weight_req_epoch;
    logic weight_rsp_valid;
    logic weight_rsp_ready;
    logic [GROUP_TAG_W-1:0] weight_rsp_tag;
    logic [INPUT_CH_W-1:0] weight_rsp_input_channel;
    logic [OUTPUT_TILE_W-1:0] weight_rsp_output_tile;
    logic [EPOCH_W-1:0] weight_rsp_epoch;
    logic [(OUT_TILE*WEIGHT_W)-1:0] weight_rsp_weights;
    logic [1:0] acc_update_valid;
    logic [1:0] acc_update_ready;
    logic [(2*TOKEN_ID_W)-1:0] acc_update_token_ids;
    logic [GROUP_TAG_W-1:0] acc_update_tag;
    logic [(OUT_TILE*PRODUCT_W)-1:0] acc_update_values;
    logic term_done;
    logic [GROUP_TAG_W-1:0] term_done_group_tag;
    logic [ISSUE_SEQ_W-1:0] term_done_issue_seq;
    logic term_done_head_last;
    logic protocol_error;
    logic [31:0] count_stale_weight_responses;
    logic [EPOCH_W-1:0] canceled_epoch;
    logic [EPOCH_W-1:0] replacement_epoch;

    integer cycle_count;
    integer weight_request_count;
    integer update_count;
    integer term_done_count;
    integer expected_weight_base;
    logic multi_destination_seen;
    logic single_destination_seen;
    logic even_token_seen;
    logic odd_token_seen;
    logic weight_backpressure_seen;
    logic accumulator_backpressure_seen;
    logic wrong_response_seen;
    logic metadata_error_seen;
    logic midterm_flush_seen;
    logic sticky_error_seen;

    gatestack_dctf32_bank_executor #(
        .BANK_ID(BANK_ID),
        .TOKENS(TOKENS),
        .OUT_TILE(OUT_TILE),
        .GATE_W(GATE_W),
        .WEIGHT_W(WEIGHT_W),
        .PRODUCT_W(PRODUCT_W),
        .GROUP_TAG_W(GROUP_TAG_W),
        .CMD_SEQUENCE_W(CMD_SEQUENCE_W),
        .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .INPUT_CH_W(INPUT_CH_W),
        .LANE_ID_W(LANE_ID_W),
        .TOKEN_ID_W(TOKEN_ID_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .LOGICAL_SUPERTILE_W(LOGICAL_SUPERTILE_W),
        .EPOCH_W(EPOCH_W)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    task automatic apply_reset;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            flush = 1'b0;
            clear_error = 1'b0;
            cmd_valid = 1'b0;
            weight_req_ready = 1'b0;
            weight_rsp_valid = 1'b0;
            acc_update_ready = '0;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
        end
    endtask

    task automatic set_command(
        input logic [GROUP_TAG_W-1:0] tag_value,
        input logic [CMD_SEQUENCE_W-1:0] sequence_value,
        input logic [ISSUE_SEQ_W-1:0] issue_value,
        input logic first_value,
        input logic last_value,
        input logic head_last_value,
        input logic [INPUT_CH_W-1:0] channel_value,
        input logic [GATE_W-1:0] gate_value,
        input logic [LANE_ID_W-1:0] lane_value,
        input logic [TOKEN_ID_W-1:0] token_value,
        input logic [LOGICAL_SUPERTILE_W-1:0] supertile_value
    );
        begin
            @(negedge clk_core);
            cmd_group_tag = tag_value;
            cmd_sequence = sequence_value;
            cmd_term_issue_seq = issue_value;
            cmd_term_first = first_value;
            cmd_term_last = last_value;
            cmd_head_last = head_last_value;
            cmd_input_channel = channel_value;
            cmd_gate_code = gate_value;
            cmd_lane_id = lane_value;
            cmd_destination_token = token_value;
            logical_supertile = supertile_value;
            cmd_valid = 1'b1;
        end
    endtask

    task automatic wait_command_accept;
        begin
            do @(posedge clk_core); while (!cmd_ready);
            @(negedge clk_core);
            cmd_valid = 1'b0;
        end
    endtask

    task automatic drive_command(
        input logic [GROUP_TAG_W-1:0] tag_value,
        input logic [CMD_SEQUENCE_W-1:0] sequence_value,
        input logic [ISSUE_SEQ_W-1:0] issue_value,
        input logic first_value,
        input logic last_value,
        input logic head_last_value,
        input logic [INPUT_CH_W-1:0] channel_value,
        input logic [GATE_W-1:0] gate_value,
        input logic [LANE_ID_W-1:0] lane_value,
        input logic [TOKEN_ID_W-1:0] token_value,
        input logic [LOGICAL_SUPERTILE_W-1:0] supertile_value
    );
        begin
            set_command(tag_value, sequence_value, issue_value, first_value,
                        last_value, head_last_value, channel_value, gate_value,
                        lane_value, token_value, supertile_value);
            wait_command_accept();
        end
    endtask

    task automatic service_weight(
        input integer request_stall_cycles,
        input integer response_stall_cycles,
        input integer weight_base,
        input logic inject_wrong_response
    );
        integer lane;
        logic [GROUP_TAG_W-1:0] captured_tag;
        logic [INPUT_CH_W-1:0] captured_channel;
        logic [OUTPUT_TILE_W-1:0] captured_tile;
        logic [EPOCH_W-1:0] captured_epoch;
        begin
            expected_weight_base = weight_base;
            weight_req_ready = 1'b0;
            repeat (request_stall_cycles) begin
                @(posedge clk_core);
                if (weight_req_valid)
                    weight_backpressure_seen = 1;
            end
            @(negedge clk_core);
            weight_req_ready = 1'b1;
            do @(posedge clk_core); while (!weight_req_valid);
            captured_tag = weight_req_tag;
            captured_channel = weight_req_input_channel;
            captured_tile = weight_req_output_tile;
            captured_epoch = weight_req_epoch;
            if (captured_tile != OUTPUT_TILE_W'(3 * logical_supertile + BANK_ID))
                $fatal(1, "physical tile mapping mismatch got=%0d", captured_tile);
            @(negedge clk_core);
            weight_req_ready = 1'b0;
            repeat (response_stall_cycles) @(posedge clk_core);

            if (inject_wrong_response) begin
                @(negedge clk_core);
                weight_rsp_tag = captured_tag + 1'b1;
                weight_rsp_input_channel = captured_channel;
                weight_rsp_output_tile = captured_tile;
                weight_rsp_epoch = captured_epoch;
                weight_rsp_valid = 1'b1;
                @(posedge clk_core);
                if (weight_rsp_ready)
                    $fatal(1, "wrong weight response was accepted");
                @(negedge clk_core);
                weight_rsp_valid = 1'b0;
                repeat (2) @(posedge clk_core);
                if (!protocol_error)
                    $fatal(1, "wrong weight response did not set protocol_error");
                wrong_response_seen = 1;
            end

            @(negedge clk_core);
            weight_rsp_tag = captured_tag;
            weight_rsp_input_channel = captured_channel;
            weight_rsp_output_tile = captured_tile;
            weight_rsp_epoch = captured_epoch;
            for (lane = 0; lane < OUT_TILE; lane = lane + 1)
                weight_rsp_weights[(lane*WEIGHT_W) +: WEIGHT_W] =
                    WEIGHT_W'(weight_base + lane);
            weight_rsp_valid = 1'b1;
            do @(posedge clk_core); while (!weight_rsp_ready);
            @(negedge clk_core);
            weight_rsp_valid = 1'b0;
        end
    endtask

    task automatic release_accumulator(
        input integer bank,
        input integer stall_cycles
    );
        begin
            while (acc_update_valid == 2'b00)
                @(posedge clk_core);
            repeat (stall_cycles) begin
                @(posedge clk_core);
                if (acc_update_valid != 2'b00)
                    accumulator_backpressure_seen = 1;
            end
            @(negedge clk_core);
            acc_update_ready[bank] = 1'b1;
            do @(posedge clk_core); while (!acc_update_valid[bank]);
            @(negedge clk_core);
            acc_update_ready[bank] = 1'b0;
        end
    endtask

    task automatic expect_protocol_error;
        begin
            repeat (2) @(posedge clk_core);
            if (!protocol_error)
                $fatal(1, "expected sticky protocol_error");
            metadata_error_seen = 1;
        end
    endtask

    always @(posedge clk_core) begin : p_scoreboard
        integer lane;
        integer signed observed_product;
        integer signed expected_product;
        integer selected_bank;
        if (rst_core) begin
            cycle_count = 0;
            weight_request_count = 0;
            update_count = 0;
            term_done_count = 0;
        end else begin
            cycle_count = cycle_count + 1;
            if (weight_req_valid && weight_req_ready)
                weight_request_count = weight_request_count + 1;
            if (flush) begin
                if (cmd_ready || weight_req_valid || weight_rsp_ready ||
                    acc_update_valid != 2'b00 || term_done ||
                    term_done_head_last)
                    $fatal(1, "flush failed to mask executor outputs");
            end
            if (acc_update_valid != 2'b00 &&
                acc_update_valid != 2'b01 && acc_update_valid != 2'b10)
                $fatal(1, "accumulator update is not onehot0");
            if (|(acc_update_valid & acc_update_ready)) begin
                selected_bank = acc_update_valid[1] ? 1 : 0;
                if (!cmd_valid || !cmd_ready)
                    $fatal(1, "accumulator update accepted without command");
                if (selected_bank != 32'(cmd_destination_token[0]))
                    $fatal(1, "token parity bank mismatch");
                if (acc_update_token_ids[
                        (selected_bank*TOKEN_ID_W) +: TOKEN_ID_W] !=
                    cmd_destination_token)
                    $fatal(1, "packed update token mismatch");
                if (acc_update_tag != cmd_group_tag)
                    $fatal(1, "accumulator update tag mismatch");
                for (lane = 0; lane < OUT_TILE; lane = lane + 1) begin
                    observed_product = 32'($signed(acc_update_values[
                        (lane*PRODUCT_W) +: PRODUCT_W]));
                    expected_product = 32'(cmd_gate_code) *
                                       (expected_weight_base + lane);
                    if (observed_product != expected_product)
                        $fatal(1, "product mismatch lane=%0d got=%0d exp=%0d",
                               lane, observed_product, expected_product);
                end
                if (cmd_destination_token[0])
                    odd_token_seen = 1;
                else
                    even_token_seen = 1;
                update_count = update_count + 1;
                if (term_done != cmd_term_last)
                    $fatal(1, "term_done timing mismatch");
                if (term_done_head_last !=
                    (cmd_term_last && cmd_head_last))
                    $fatal(1, "term_done head_last timing mismatch");
            end
            if (cmd_valid && cmd_ready !=
                (|(acc_update_valid & acc_update_ready)))
                $fatal(1, "command consumed without matching Acc update");
            if (term_done) begin
                if (!cmd_valid || !cmd_ready || !cmd_term_last ||
                    term_done_group_tag != cmd_group_tag ||
                    term_done_issue_seq != cmd_term_issue_seq)
                    $fatal(1, "term completion metadata mismatch");
                term_done_count = term_done_count + 1;
            end
            if (term_done_head_last && !term_done)
                $fatal(1, "head_last completion without term_done");
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        flush = 1'b0;
        clear_error = 1'b0;
        cmd_valid = 1'b0;
        cmd_group_tag = '0;
        cmd_sequence = '0;
        cmd_term_issue_seq = '0;
        cmd_term_first = 1'b0;
        cmd_term_last = 1'b0;
        cmd_head_last = 1'b0;
        cmd_input_channel = '0;
        cmd_gate_code = '0;
        cmd_lane_id = '0;
        cmd_destination_token = '0;
        logical_supertile = '0;
        weight_req_ready = 1'b0;
        weight_rsp_valid = 1'b0;
        weight_rsp_tag = '0;
        weight_rsp_input_channel = '0;
        weight_rsp_output_tile = '0;
        weight_rsp_epoch = '0;
        weight_rsp_weights = '0;
        acc_update_ready = '0;
        expected_weight_base = -16;
        multi_destination_seen = 0;
        single_destination_seen = 0;
        even_token_seen = 0;
        odd_token_seen = 0;
        weight_backpressure_seen = 0;
        accumulator_backpressure_seen = 0;
        wrong_response_seen = 0;
        metadata_error_seen = 0;
        midterm_flush_seen = 0;
        sticky_error_seen = 0;

        repeat (4) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        // One three-destination term: one request, wrong-then-correct response,
        // even/odd routing, weight stall, Acc stall, and head-last completion.
        fork
            drive_command(12'h451, 8'd0, 6'd11, 1'b1, 1'b0, 1'b0,
                          7'd23, 9'd3, 5'd7, 5'd2, 4'd3);
            service_weight(3, 2, -16, 1'b1);
            release_accumulator(0, 3);
        join
        if (weight_request_count != 1 || update_count != 1)
            $fatal(1, "first destination accounting mismatch");

        fork
            drive_command(12'h451, 8'd1, 6'd11, 1'b0, 1'b0, 1'b0,
                          7'd23, 9'd3, 5'd7, 5'd3, 4'd3);
            release_accumulator(1, 2);
        join
        fork
            drive_command(12'h451, 8'd2, 6'd11, 1'b0, 1'b1, 1'b1,
                          7'd23, 9'd3, 5'd7, 5'd4, 4'd3);
            release_accumulator(0, 0);
        join
        if (weight_request_count != 1 || update_count != 3 ||
            term_done_count != 1)
            $fatal(1, "multi-destination product reuse mismatch");
        multi_destination_seen = 1;

        // Reset clears sticky error; a one-command odd-token term is legal.
        apply_reset();
        fork
            drive_command(12'h522, 8'd0, 6'd12, 1'b1, 1'b1, 1'b0,
                          7'd19, 9'd5, 5'd2, 5'd5, 4'd2);
            service_weight(0, 1, -8, 1'b0);
            release_accumulator(1, 0);
        join
        if (protocol_error || weight_request_count != 1 || update_count != 1 ||
            term_done_count != 1 || term_done_head_last)
            $fatal(1, "single-destination execution mismatch");
        single_destination_seen = 1;

        // Missing first and malformed head-last are rejected without updates.
        apply_reset();
        set_command(12'h601, 8'd0, 6'd1, 1'b0, 1'b1, 1'b0,
                    7'd4, 9'd2, 5'd1, 5'd1, 4'd1);
        expect_protocol_error();
        if (cmd_ready || acc_update_valid != 2'b00 || weight_req_valid)
            $fatal(1, "missing-first command escaped rejection");
        @(negedge clk_core);
        cmd_valid = 1'b0;

        apply_reset();
        set_command(12'h602, 8'd0, 6'd2, 1'b1, 1'b0, 1'b1,
                    7'd5, 9'd2, 5'd1, 5'd2, 4'd1);
        expect_protocol_error();
        if (cmd_ready || acc_update_valid != 2'b00 || weight_req_valid)
            $fatal(1, "malformed head-last command escaped rejection");
        @(negedge clk_core);
        cmd_valid = 1'b0;

        // After a valid first destination, corrupt all continuation identity
        // fields and sequence. The command must remain unconsumed until flush.
        apply_reset();
        fork
            drive_command(12'h633, 8'd0, 6'd9, 1'b1, 1'b0, 1'b0,
                          7'd31, 9'd4, 5'd6, 5'd6, 4'd2);
            service_weight(0, 0, -12, 1'b0);
            release_accumulator(0, 0);
        join
        set_command(12'h634, 8'd7, 6'd10, 1'b1, 1'b0, 1'b0,
                    7'd32, 9'd6, 5'd7, 5'd7, 4'd3);
        expect_protocol_error();
        if (cmd_ready || acc_update_valid != 2'b00 || weight_request_count != 1)
            $fatal(1, "bad continuation was consumed or relaunched weight");
        @(negedge clk_core);
        flush = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        cmd_valid = 1'b0;
        flush = 1'b0;
        if (!protocol_error)
            $fatal(1, "flush incorrectly cleared sticky protocol_error");
        sticky_error_seen = 1;
        @(negedge clk_core);
        clear_error = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        clear_error = 1'b0;
        if (protocol_error)
            $fatal(1, "clear_error did not clear executor sticky error");

        // Flush a request, relaunch the exact same SRAM identity under a new
        // epoch, then deliver the canceled response before the replacement.
        apply_reset();
        set_command(12'h711, 8'd0, 6'd15, 1'b1, 1'b0, 1'b0,
                    7'd42, 9'd7, 5'd4, 5'd8, 4'd3);
        @(negedge clk_core);
        weight_req_ready = 1'b1;
        do @(posedge clk_core); while (!weight_req_valid);
        if (weight_req_output_tile != 6'd10)
            $fatal(1, "mid-term physical tile mismatch");
        canceled_epoch = weight_req_epoch;
        @(negedge clk_core);
        weight_req_ready = 1'b0;
        flush = 1'b1;
        #1;
        if (cmd_ready || weight_req_valid || weight_rsp_ready ||
            acc_update_valid != 2'b00 || term_done)
            $fatal(1, "flush output masking was not combinational");
        @(posedge clk_core);
        @(negedge clk_core);
        flush = 1'b0;
        cmd_valid = 1'b0;
        midterm_flush_seen = 1;

        set_command(12'h711, 8'd19, 6'd16, 1'b1, 1'b1, 1'b1,
                    7'd42, 9'd2, 5'd5, 5'd9, 4'd3);
        @(negedge clk_core);
        weight_req_ready = 1'b1;
        do @(posedge clk_core); while (!weight_req_valid);
        replacement_epoch = weight_req_epoch;
        if (weight_req_tag != 12'h711 ||
            weight_req_input_channel != 7'd42 ||
            weight_req_output_tile != 6'd10 ||
            replacement_epoch == canceled_epoch)
            $fatal(1, "replacement request identity/epoch mismatch");
        @(negedge clk_core);
        weight_req_ready = 1'b0;

        // The stale response is otherwise bit-for-bit identical to the new
        // request. Epoch alone must cause a ready/drop transaction.
        weight_rsp_tag = 12'h711;
        weight_rsp_input_channel = 7'd42;
        weight_rsp_output_tile = 6'd10;
        weight_rsp_epoch = canceled_epoch;
        weight_rsp_valid = 1'b1;
        @(posedge clk_core);
        if (!weight_rsp_ready || acc_update_valid != 2'b00 || term_done)
            $fatal(1, "stale response was not atomically dropped");
        @(negedge clk_core);
        weight_rsp_valid = 1'b0;
        if (count_stale_weight_responses != 1 || protocol_error)
            $fatal(1, "stale response audit/error accounting mismatch");

        expected_weight_base = -4;
        @(negedge clk_core);
        weight_rsp_tag = 12'h711;
        weight_rsp_input_channel = 7'd42;
        weight_rsp_output_tile = 6'd10;
        weight_rsp_epoch = replacement_epoch;
        for (integer lane = 0; lane < OUT_TILE; lane = lane + 1)
            weight_rsp_weights[(lane*WEIGHT_W) +: WEIGHT_W] =
                WEIGHT_W'(-4 + lane);
        weight_rsp_valid = 1'b1;
        do @(posedge clk_core); while (!weight_rsp_ready);
        @(negedge clk_core);
        weight_rsp_valid = 1'b0;
        acc_update_ready[1] = 1'b1;
        do @(posedge clk_core); while (!cmd_ready);
        @(negedge clk_core);
        cmd_valid = 1'b0;
        acc_update_ready[1] = 1'b0;
        if (update_count != 1 || term_done_count != 1 ||
            protocol_error || count_stale_weight_responses != 1)
            $fatal(1, "post-flush recovery mismatch");

        if (!multi_destination_seen || !single_destination_seen ||
            !even_token_seen || !odd_token_seen ||
            !weight_backpressure_seen || !accumulator_backpressure_seen ||
            !wrong_response_seen || !metadata_error_seen ||
            !midterm_flush_seen || !sticky_error_seen)
            $fatal(1, "required coverage marker missing");

        $display("PASS DCTF32 BANK EXECUTOR requests=%0d updates=%0d done=%0d stale_rsp=%0d coverage=multi,single,parity,weight_bp,acc_bp,wrong_rsp,metadata,flush,stale_epoch",
                 weight_request_count, update_count, term_done_count,
                 count_stale_weight_responses);
        $finish;
    end

    initial begin
        repeat (5000) @(posedge clk_core);
        $fatal(1, "DCTF32 bank executor TB timeout");
    end
endmodule

/* verilator lint_on UNUSEDSIGNAL */
/* verilator lint_on BLKSEQ */

`default_nettype wire
