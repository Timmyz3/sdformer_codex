`timescale 1ns/1ps
`default_nettype none

module tb_et3_native_slice;
    localparam int HEAD_DIM = 4;
    localparam int OUT_DIM = 4;
    localparam int MAX_DEST = 16;
    localparam int DEST_W = 4;
    localparam int OUT_ID_W = 2;

    logic clk_core;
    logic rst_core;
    logic flush;

    logic weight_load_valid;
    logic weight_load_ready;
    logic [1:0] weight_load_lane;
    logic [OUT_ID_W-1:0] weight_load_output;
    logic signed [7:0] weight_load_value;
    logic weight_load_last;
    logic run_start;
    logic run_active;

    logic source_valid;
    logic source_ready;
    logic [7:0] source_group_tag;
    logic source_mode_multiset;
    logic [8:0] source_gate_code;
    logic [1:0] source_lane_id;
    logic [2:0] source_multiplicity;
    logic [DEST_W-1:0] source_destination;
    logic source_head_last;
    logic group_close_valid;
    logic group_close_ready;
    logic [7:0] group_close_tag;

    logic acc_write_ready;
    logic acc_read_valid;
    logic [DEST_W-1:0] acc_read_destination;
    logic [OUT_ID_W-1:0] acc_read_output;
    logic acc_read_data_valid;
    logic signed [31:0] acc_read_data;

    logic group_done;
    logic protocol_error;
    logic trace_cmd_valid;
    logic trace_cmd_ready;
    logic [7:0] trace_cmd_group_tag;
    logic trace_cmd_mode_multiset;
    logic [8:0] trace_cmd_gate_code;
    logic [1:0] trace_cmd_lane_id;
    logic [2:0] trace_cmd_multiplicity;
    logic [DEST_W-1:0] trace_cmd_destination;
    logic trace_cmd_term_first;
    logic trace_cmd_term_last;
    logic trace_cmd_head_last;
    logic trace_cmd_fallback;

    logic [31:0] count_source_items;
    logic [31:0] count_directory_entries;
    logic [31:0] count_fallback_items;
    logic [31:0] count_typed_terms;
    logic [31:0] count_destination_beats;
    logic [31:0] count_partial_drains;
    logic [31:0] count_product_computes;
    logic [31:0] count_native_commands;
    logic [31:0] count_explode_baseline_commands;
    logic [31:0] count_fallback_terms;
    logic [31:0] count_set_terms;
    logic [31:0] count_multiset_terms;

    integer signed weights [0:HEAD_DIM-1][0:OUT_DIM-1];
    integer signed golden [0:MAX_DEST-1][0:OUT_DIM-1];
    integer cycle_count;
    integer done_count;
    integer stall_count;
    integer term_first_count;
    integer term_last_count;
    integer fallback_beat_count;
    integer error_count;

    logic stalled_q;
    logic [30:0] stalled_payload_q;

    et3_native_slice_top #(
        .KEY_CAP(2),
        .SEG_DEPTH(2),
        .FALLBACK_DEPTH(1),
        .HEAD_DIM(HEAD_DIM),
        .OUT_DIM(OUT_DIM),
        .MAX_DEST(MAX_DEST),
        .TAG_W(8),
        .GATE_W(9),
        .LANE_W(2),
        .MULT_W(3),
        .DEST_W(DEST_W),
        .WEIGHT_W(8),
        .ACC_W(32)
    ) dut (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .flush(flush),
        .weight_load_valid(weight_load_valid),
        .weight_load_ready(weight_load_ready),
        .weight_load_lane(weight_load_lane),
        .weight_load_output(weight_load_output),
        .weight_load_value(weight_load_value),
        .weight_load_last(weight_load_last),
        .run_start(run_start),
        .run_active(run_active),
        .source_valid(source_valid),
        .source_ready(source_ready),
        .source_group_tag(source_group_tag),
        .source_mode_multiset(source_mode_multiset),
        .source_gate_code(source_gate_code),
        .source_lane_id(source_lane_id),
        .source_multiplicity(source_multiplicity),
        .source_destination(source_destination),
        .source_head_last(source_head_last),
        .group_close_valid(group_close_valid),
        .group_close_ready(group_close_ready),
        .group_close_tag(group_close_tag),
        .acc_write_ready(acc_write_ready),
        .acc_read_valid(acc_read_valid),
        .acc_read_destination(acc_read_destination),
        .acc_read_output(acc_read_output),
        .acc_read_data_valid(acc_read_data_valid),
        .acc_read_data(acc_read_data),
        .group_done(group_done),
        .protocol_error(protocol_error),
        .trace_cmd_valid(trace_cmd_valid),
        .trace_cmd_ready(trace_cmd_ready),
        .trace_cmd_group_tag(trace_cmd_group_tag),
        .trace_cmd_mode_multiset(trace_cmd_mode_multiset),
        .trace_cmd_gate_code(trace_cmd_gate_code),
        .trace_cmd_lane_id(trace_cmd_lane_id),
        .trace_cmd_multiplicity(trace_cmd_multiplicity),
        .trace_cmd_destination(trace_cmd_destination),
        .trace_cmd_term_first(trace_cmd_term_first),
        .trace_cmd_term_last(trace_cmd_term_last),
        .trace_cmd_head_last(trace_cmd_head_last),
        .trace_cmd_fallback(trace_cmd_fallback),
        .count_source_items(count_source_items),
        .count_directory_entries(count_directory_entries),
        .count_fallback_items(count_fallback_items),
        .count_typed_terms(count_typed_terms),
        .count_destination_beats(count_destination_beats),
        .count_partial_drains(count_partial_drains),
        .count_product_computes(count_product_computes),
        .count_native_commands(count_native_commands),
        .count_explode_baseline_commands(
            count_explode_baseline_commands
        ),
        .count_fallback_terms(count_fallback_terms),
        .count_set_terms(count_set_terms),
        .count_multiset_terms(count_multiset_terms)
    );

`ifdef VERILATOR
    et3_native_slice_assertions #(
        .TAG_W(8),
        .GATE_W(9),
        .LANE_W(2),
        .MULT_W(3),
        .DEST_W(DEST_W)
    ) u_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .flush(flush),
        .run_active(run_active),
        .group_done(group_done),
        .source_valid(source_valid),
        .source_ready(source_ready),
        .source_group_tag(source_group_tag),
        .source_mode_multiset(source_mode_multiset),
        .source_gate_code(source_gate_code),
        .source_lane_id(source_lane_id),
        .source_multiplicity(source_multiplicity),
        .source_destination(source_destination),
        .source_head_last(source_head_last),
        .group_close_valid(group_close_valid),
        .group_close_ready(group_close_ready),
        .group_close_tag(group_close_tag),
        .trace_cmd_valid(trace_cmd_valid),
        .trace_cmd_ready(trace_cmd_ready),
        .trace_cmd_group_tag(trace_cmd_group_tag),
        .trace_cmd_mode_multiset(trace_cmd_mode_multiset),
        .trace_cmd_gate_code(trace_cmd_gate_code),
        .trace_cmd_lane_id(trace_cmd_lane_id),
        .trace_cmd_multiplicity(trace_cmd_multiplicity),
        .trace_cmd_destination(trace_cmd_destination),
        .trace_cmd_term_first(trace_cmd_term_first),
        .trace_cmd_term_last(trace_cmd_term_last),
        .trace_cmd_head_last(trace_cmd_head_last),
        .trace_cmd_fallback(trace_cmd_fallback),
        .count_source_items(count_source_items),
        .count_destination_beats(count_destination_beats)
    );
`endif

    /* verilator lint_off BLKSEQ */
    always #5 clk_core = ~clk_core;
    /* verilator lint_on BLKSEQ */

    initial begin
        #200000;
        $fatal(1, "global simulation timeout");
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            acc_write_ready <= 1'b1;
            done_count <= 0;
            stall_count <= 0;
            term_first_count <= 0;
            term_last_count <= 0;
            fallback_beat_count <= 0;
            stalled_q <= 1'b0;
            stalled_payload_q <= '0;
        end else begin
            cycle_count <= cycle_count + 1;
            acc_write_ready <= (cycle_count[1:0] != 2'b01);
            if (group_done) begin
                done_count <= done_count + 1;
            end
            if (trace_cmd_valid && !trace_cmd_ready) begin
                stall_count <= stall_count + 1;
            end
            if (trace_cmd_valid && trace_cmd_ready) begin
                if (trace_cmd_term_first) begin
                    term_first_count <= term_first_count + 1;
                end
                if (trace_cmd_term_last) begin
                    term_last_count <= term_last_count + 1;
                end
                if (trace_cmd_fallback) begin
                    fallback_beat_count <= fallback_beat_count + 1;
                end
            end

            if (stalled_q) begin
                if (!trace_cmd_valid) begin
                    $fatal(1, "cmd_valid dropped under backpressure");
                end
                if ({
                    trace_cmd_group_tag,
                    trace_cmd_mode_multiset,
                    trace_cmd_gate_code,
                    trace_cmd_lane_id,
                    trace_cmd_multiplicity,
                    trace_cmd_destination,
                    trace_cmd_term_first,
                    trace_cmd_term_last,
                    trace_cmd_head_last,
                    trace_cmd_fallback
                } != stalled_payload_q) begin
                    $fatal(1, "command payload changed under backpressure");
                end
            end
            stalled_q <= trace_cmd_valid && !trace_cmd_ready;
            stalled_payload_q <= {
                trace_cmd_group_tag,
                trace_cmd_mode_multiset,
                trace_cmd_gate_code,
                trace_cmd_lane_id,
                trace_cmd_multiplicity,
                trace_cmd_destination,
                trace_cmd_term_first,
                trace_cmd_term_last,
                trace_cmd_head_last,
                trace_cmd_fallback
            };
        end
    end

    task automatic load_weight(
        input logic [1:0] lane,
        input logic [OUT_ID_W-1:0] output_lane,
        input logic signed [7:0] value,
        input logic last
    );
        begin
            @(negedge clk_core);
            if (!weight_load_ready) begin
                $fatal(1, "weight loader was not ready");
            end
            weight_load_valid = 1'b1;
            weight_load_lane = lane;
            weight_load_output = output_lane;
            weight_load_value = value;
            weight_load_last = last;
            @(negedge clk_core);
            weight_load_valid = 1'b0;
            weight_load_last = 1'b0;
        end
    endtask

    task automatic send_item(
        input logic [7:0] tag,
        input logic mode_multiset,
        input integer gate_code,
        input integer lane,
        input integer multiplicity,
        input integer destination,
        input logic head_last
    );
        integer timeout;
        logic accepted;
        begin
            @(negedge clk_core);
            source_valid = 1'b1;
            source_group_tag = tag;
            source_mode_multiset = mode_multiset;
            source_gate_code = gate_code[8:0];
            source_lane_id = lane[1:0];
            source_multiplicity = multiplicity[2:0];
            source_destination = destination[DEST_W-1:0];
            source_head_last = head_last;
            timeout = 0;
            accepted = 1'b0;
            while (!accepted && (timeout < 100)) begin
                @(posedge clk_core);
                if (source_ready) begin
                    accepted = 1'b1;
                end
                timeout = timeout + 1;
            end
            if (!accepted) begin
                $fatal(1, "source timeout tag=%0h mode=%0d gate=%0d lane=%0d mult=%0d dest=%0d run=%0d cmd_v=%0d cmd_r=%0d",
                       tag, mode_multiset, gate_code, lane, multiplicity,
                       destination, run_active, trace_cmd_valid,
                       trace_cmd_ready);
            end
            for (int output_lane = 0; output_lane < OUT_DIM;
                 output_lane++) begin
                golden[destination][output_lane] =
                    golden[destination][output_lane] +
                    multiplicity * gate_code *
                    weights[lane][output_lane];
            end
            @(negedge clk_core);
            source_valid = 1'b0;
            source_head_last = 1'b0;
        end
    endtask

    task automatic wait_done_count(input integer target);
        integer timeout;
        begin
            timeout = 0;
            while ((done_count < target) && (timeout < 500)) begin
                @(negedge clk_core);
                timeout = timeout + 1;
            end
            if (done_count < target) begin
                $error("timeout waiting for group_done=%0d", target);
                error_count = error_count + 1;
            end
        end
    endtask

    task automatic start_group;
        begin
            @(negedge clk_core);
            run_start = 1'b1;
            @(negedge clk_core);
            run_start = 1'b0;
            if (!run_active) begin
                $fatal(1, "run_active did not assert");
            end
        end
    endtask

    task automatic close_empty_group(input logic [7:0] tag);
        integer timeout;
        logic accepted;
        begin
            @(negedge clk_core);
            group_close_valid = 1'b1;
            group_close_tag = tag;
            timeout = 0;
            accepted = 1'b0;
            while (!accepted && (timeout < 100)) begin
                @(posedge clk_core);
                if (group_close_ready) begin
                    accepted = 1'b1;
                end
                timeout = timeout + 1;
            end
            if (!accepted) begin
                $fatal(1, "empty group close timeout tag=%0h", tag);
            end
            @(negedge clk_core);
            group_close_valid = 1'b0;
        end
    endtask

    task automatic send_item_with_concurrent_close(
        input logic [7:0] tag,
        input logic mode_multiset,
        input logic [8:0] gate_code,
        input logic [1:0] lane,
        input logic [2:0] multiplicity,
        input logic [DEST_W-1:0] destination
    );
        integer timeout;
        logic source_accepted;
        logic close_accepted;
        begin
            @(negedge clk_core);
            source_valid = 1'b1;
            source_group_tag = tag;
            source_mode_multiset = mode_multiset;
            source_gate_code = gate_code;
            source_lane_id = lane;
            source_multiplicity = multiplicity;
            source_destination = destination;
            source_head_last = 1'b0;
            group_close_valid = 1'b1;
            group_close_tag = tag;
            timeout = 0;
            source_accepted = 1'b0;
            close_accepted = 1'b0;
            while (!source_accepted && (timeout < 100)) begin
                @(posedge clk_core);
                if (source_ready) begin
                    source_accepted = 1'b1;
                end
                timeout = timeout + 1;
            end
            if (!source_accepted) begin
                $fatal(1, "source/close arbitration did not accept source");
            end
            for (int output_lane = 0; output_lane < OUT_DIM;
                 output_lane++) begin
                golden[destination][output_lane] =
                    golden[destination][output_lane] +
                    multiplicity * gate_code *
                    weights[lane][output_lane];
            end
            @(negedge clk_core);
            source_valid = 1'b0;
            timeout = 0;
            while (!close_accepted && (timeout < 100)) begin
                @(posedge clk_core);
                if (group_close_ready) begin
                    close_accepted = 1'b1;
                end
                timeout = timeout + 1;
            end
            if (!close_accepted) begin
                $fatal(1, "source/close arbitration did not drain close");
            end
            @(negedge clk_core);
            group_close_valid = 1'b0;
        end
    endtask

    task automatic clear_golden;
        begin
            for (int destination = 0; destination < MAX_DEST;
                 destination++) begin
                for (int output_lane = 0; output_lane < OUT_DIM;
                     output_lane++) begin
                    golden[destination][output_lane] = 0;
                end
            end
        end
    endtask

    task automatic check_accumulators;
        integer signed observed;
        begin
            for (int destination = 0; destination < MAX_DEST;
                 destination++) begin
                for (int output_lane = 0; output_lane < OUT_DIM;
                     output_lane++) begin
                    @(negedge clk_core);
                    acc_read_valid = 1'b1;
                    acc_read_destination = destination[DEST_W-1:0];
                    acc_read_output = output_lane[OUT_ID_W-1:0];
                    #1;
                    observed = $signed(acc_read_data);
                    if (!acc_read_data_valid ||
                        (observed != golden[destination][output_lane])) begin
                        $error(
                            "acc mismatch dest=%0d out=%0d got=%0d exp=%0d",
                            destination,
                            output_lane,
                            observed,
                            golden[destination][output_lane]
                        );
                        error_count = error_count + 1;
                    end
                end
            end
            @(negedge clk_core);
            acc_read_valid = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        flush = 1'b0;
        weight_load_valid = 1'b0;
        weight_load_lane = '0;
        weight_load_output = '0;
        weight_load_value = '0;
        weight_load_last = 1'b0;
        run_start = 1'b0;
        source_valid = 1'b0;
        source_group_tag = '0;
        source_mode_multiset = 1'b0;
        source_gate_code = '0;
        source_lane_id = '0;
        source_multiplicity = '0;
        source_destination = '0;
        source_head_last = 1'b0;
        group_close_valid = 1'b0;
        group_close_tag = '0;
        acc_write_ready = 1'b1;
        acc_read_valid = 1'b0;
        acc_read_destination = '0;
        acc_read_output = '0;
        error_count = 0;

        weights[0][0] = 1;
        weights[0][1] = -2;
        weights[0][2] = 3;
        weights[0][3] = -4;
        weights[1][0] = 2;
        weights[1][1] = 1;
        weights[1][2] = -1;
        weights[1][3] = 3;
        weights[2][0] = -1;
        weights[2][1] = 2;
        weights[2][2] = 4;
        weights[2][3] = 1;
        weights[3][0] = 3;
        weights[3][1] = -3;
        weights[3][2] = 2;
        weights[3][3] = -2;

        clear_golden();

        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;

        for (int lane = 0; lane < HEAD_DIM; lane++) begin
            for (int output_lane = 0; output_lane < OUT_DIM;
                 output_lane++) begin
                load_weight(
                    2'(lane),
                    OUT_ID_W'(output_lane),
                    8'(weights[lane][output_lane]),
                    (lane == HEAD_DIM - 1) &&
                    (output_lane == OUT_DIM - 1)
                );
            end
        end

        start_group();

        // Motion SET group. The third destination creates a second segment;
        // the second key fills fallback. The held final third key forces a
        // partial drain, then resumes without dropping the source item.
        send_item(8'h11, 1'b0, 3, 0, 1, 0, 1'b0);
        send_item(8'h11, 1'b0, 3, 0, 1, 2, 1'b0);
        send_item(8'h11, 1'b0, 3, 0, 1, 4, 1'b0);
        send_item(8'h11, 1'b0, 2, 1, 1, 1, 1'b0);
        send_item(8'h11, 1'b0, 1, 3, 1, 8, 1'b1);
        wait_done_count(1);
        check_accumulators();
        if (count_product_computes != 4 ||
            count_native_commands != 5 ||
            count_explode_baseline_commands != 5 ||
            count_fallback_terms != 1 ||
            count_set_terms != 4 ||
            count_multiset_terms != 0) begin
            $error("Motion epoch counters mismatch");
            error_count = error_count + 1;
        end

        // Local5 MULTISET group. Multiplicity is consumed natively rather
        // than exploded into repeated commands. Destinations deliberately
        // overlap the prior Motion group to verify epoch clearing.
        clear_golden();
        start_group();
        send_item(8'h22, 1'b1, 4, 2, 2, 0, 1'b0);
        send_item(8'h22, 1'b1, 4, 2, 2, 2, 1'b0);
        send_item(8'h22, 1'b1, 4, 2, 2, 4, 1'b0);
        send_item(8'h22, 1'b1, 4, 2, 2, 6, 1'b0);
        send_item(8'h22, 1'b1, 1, 3, 3, 1, 1'b0);
        send_item(8'h22, 1'b1, 2, 1, 4, 8, 1'b1);
        wait_done_count(2);
        repeat (3) @(negedge clk_core);

        check_accumulators();
        if (count_product_computes != 4 ||
            count_native_commands != 6 ||
            count_explode_baseline_commands != 15 ||
            count_fallback_terms != 1 ||
            count_set_terms != 0 ||
            count_multiset_terms != 4) begin
            $error("Local5 epoch counters mismatch");
            error_count = error_count + 1;
        end

        // Empty windows/heads are common. A close event must complete a
        // zero-term epoch without inventing a source item.
        clear_golden();
        start_group();
        close_empty_group(8'h33);
        wait_done_count(3);
        check_accumulators();

        // Independent valid inputs may arrive together. Source wins the first
        // handshake; close remains pending and commits on the next cycle.
        clear_golden();
        start_group();
        send_item_with_concurrent_close(
            8'h44,
            1'b1,
            9'(2),
            2'(0),
            3'(2),
            DEST_W'(0)
        );
        wait_done_count(4);
        check_accumulators();
        if (count_product_computes != 1 ||
            count_native_commands != 1 ||
            count_explode_baseline_commands != 2 ||
            count_multiset_terms != 1) begin
            $error("source/close arbitration epoch counters mismatch");
            error_count = error_count + 1;
        end

        if (protocol_error) begin
            $error("unexpected protocol_error");
            error_count = error_count + 1;
        end
        if (stall_count == 0) begin
            $error("backpressure was not exercised");
            error_count = error_count + 1;
        end
        if (count_source_items != 12 ||
            count_destination_beats != 12) begin
            $error(
                "destination count mismatch source=%0d dir=%0d",
                count_source_items,
                count_destination_beats
            );
            error_count = error_count + 1;
        end
        if (count_directory_entries != 7 ||
            count_fallback_items != 2 ||
            count_partial_drains != 2) begin
            $error(
                "directory/fallback mismatch entries=%0d items=%0d drains=%0d",
                count_directory_entries,
                count_fallback_items,
                count_partial_drains
            );
            error_count = error_count + 1;
        end
        if (count_typed_terms != 9 ||
            term_first_count != 9 ||
            term_last_count != 9) begin
            $error(
                "term mismatch dir=%0d first=%0d last=%0d",
                count_typed_terms,
                term_first_count,
                term_last_count
            );
            error_count = error_count + 1;
        end
        if (fallback_beat_count != 2) begin
            $error("fallback beat mismatch got=%0d", fallback_beat_count);
            error_count = error_count + 1;
        end
        if (done_count != 4) begin
            $error("group_done count mismatch got=%0d", done_count);
            error_count = error_count + 1;
        end

        if (error_count == 0) begin
            $display("PASS ET3 native slice: source=12 terms=9 partial_drains=2 fallback=2 groups=4 stalls=%0d",
                     stall_count);
        end else begin
            $fatal(1, "FAIL ET3 native slice errors=%0d", error_count);
        end
        $finish;
    end

endmodule

`default_nettype wire
