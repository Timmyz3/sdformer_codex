`timescale 1ns/1ps
`default_nettype none

module tb_et3_native_m_queue_baseline;
    localparam int OUT_DIM = 4;
    localparam int MAX_DEST = 16;

    logic clk_core;
    logic rst_core;
    logic flush;
    logic weight_load_valid;
    logic weight_load_ready;
    logic [1:0] weight_load_lane;
    logic [1:0] weight_load_output;
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
    logic [3:0] source_destination;
    logic source_head_last;
    logic group_close_valid;
    logic group_close_ready;
    logic [7:0] group_close_tag;
    logic acc_write_ready;
    logic acc_read_valid;
    logic [3:0] acc_read_destination;
    logic [1:0] acc_read_output;
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
    logic [3:0] trace_cmd_destination;
    logic trace_cmd_head_last;
    logic [31:0] count_source_items;
    logic [31:0] count_queue_commands;
    logic [31:0] count_close_markers;
    logic [31:0] count_product_computes;
    logic [31:0] count_native_commands;
    logic [31:0] count_explode_commands;
    logic [31:0] count_fallback_terms;
    logic [31:0] count_set_terms;
    logic [31:0] count_multiset_terms;
    logic [1:0] max_fifo_occupancy;

    integer signed weights [0:3][0:3];
    integer signed golden [0:MAX_DEST-1][0:OUT_DIM-1];
    integer done_count;
    integer error_count;
    integer cycle_count;
    integer stall_count;
    integer full_pop_push_count;
    logic force_acc_stall;

    et3_native_m_queue_baseline #(
        .FIFO_DEPTH(2),
        .HEAD_DIM(4),
        .OUT_DIM(OUT_DIM),
        .MAX_DEST(MAX_DEST),
        .TAG_W(8),
        .GATE_W(9),
        .LANE_W(2),
        .MULT_W(3),
        .DEST_W(4)
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
        .trace_cmd_head_last(trace_cmd_head_last),
        .count_source_items(count_source_items),
        .count_queue_commands(count_queue_commands),
        .count_close_markers(count_close_markers),
        .count_product_computes(count_product_computes),
        .count_native_commands(count_native_commands),
        .count_explode_commands(count_explode_commands),
        .count_fallback_terms(count_fallback_terms),
        .count_set_terms(count_set_terms),
        .count_multiset_terms(count_multiset_terms),
        .max_fifo_occupancy(max_fifo_occupancy)
    );

`ifdef VERILATOR
    et3_native_slice_assertions #(
        .TAG_W(8),
        .GATE_W(9),
        .LANE_W(2),
        .MULT_W(3),
        .DEST_W(4)
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
        .trace_cmd_term_first(1'b1),
        .trace_cmd_term_last(1'b1),
        .trace_cmd_head_last(trace_cmd_head_last),
        .trace_cmd_fallback(1'b0),
        .count_source_items(count_source_items),
        .count_destination_beats(count_queue_commands)
    );
`endif

    /* verilator lint_off BLKSEQ */
    always #5 clk_core = ~clk_core;
    /* verilator lint_on BLKSEQ */

    initial begin
        #100000;
        $fatal(1, "native-m baseline timeout");
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            done_count <= 0;
            stall_count <= 0;
            full_pop_push_count <= 0;
            acc_write_ready <= 1'b1;
        end else begin
            cycle_count <= cycle_count + 1;
            acc_write_ready <= force_acc_stall
                             ? 1'b0
                             : cycle_count[1:0] != 2'b01;
            if (group_done) begin
                done_count <= done_count + 1;
            end
            if (trace_cmd_valid && !trace_cmd_ready) begin
                stall_count <= stall_count + 1;
            end
            if (32'(dut.fifo_count_q) == 2 &&
                dut.queue_pop && dut.queue_push) begin
                full_pop_push_count <= full_pop_push_count + 1;
            end
        end
    end

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

    task automatic load_weight(
        input logic [1:0] lane,
        input logic [1:0] output_lane,
        input logic signed [7:0] value,
        input logic last
    );
        begin
            @(negedge clk_core);
            if (!weight_load_ready) begin
                $fatal(1, "weight loader not ready");
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

    task automatic send_item(
        input logic [8:0] gate,
        input logic [1:0] lane,
        input logic [2:0] multiplicity,
        input logic [3:0] destination,
        input logic last
    );
        logic accepted;
        integer timeout;
        begin
            @(negedge clk_core);
            source_valid = 1'b1;
            source_group_tag = 8'h22;
            source_mode_multiset = 1'b1;
            source_gate_code = gate;
            source_lane_id = lane;
            source_multiplicity = multiplicity;
            source_destination = destination;
            source_head_last = last;
            accepted = 1'b0;
            timeout = 0;
            while (!accepted && timeout < 100) begin
                @(posedge clk_core);
                if (source_ready) begin
                    accepted = 1'b1;
                end
                timeout = timeout + 1;
            end
            if (!accepted) begin
                $fatal(1, "source handshake timeout");
            end
            for (int output_lane = 0; output_lane < OUT_DIM;
                 output_lane++) begin
                golden[destination][output_lane] =
                    golden[destination][output_lane] +
                    multiplicity * gate * weights[lane][output_lane];
            end
            @(negedge clk_core);
            source_valid = 1'b0;
            source_head_last = 1'b0;
        end
    endtask

    task automatic wait_done(input integer target);
        integer timeout;
        begin
            timeout = 0;
            while (done_count < target && timeout < 300) begin
                @(negedge clk_core);
                timeout = timeout + 1;
            end
            if (done_count < target) begin
                $fatal(1, "group_done timeout");
            end
        end
    endtask

    task automatic check_acc;
        integer signed observed;
        begin
            for (int destination = 0; destination < MAX_DEST;
                 destination++) begin
                for (int output_lane = 0; output_lane < OUT_DIM;
                     output_lane++) begin
                    @(negedge clk_core);
                    acc_read_valid = 1'b1;
                    acc_read_destination = destination[3:0];
                    acc_read_output = output_lane[1:0];
                    #1;
                    observed = $signed(acc_read_data);
                    if (!acc_read_data_valid ||
                        observed != golden[destination][output_lane]) begin
                        $error("baseline acc mismatch d=%0d o=%0d got=%0d exp=%0d",
                               destination, output_lane, observed,
                               golden[destination][output_lane]);
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
        force_acc_stall = 1'b0;

        weights[0][0] = 1;  weights[0][1] = -2;
        weights[0][2] = 3;  weights[0][3] = -4;
        weights[1][0] = 2;  weights[1][1] = 1;
        weights[1][2] = -1; weights[1][3] = 3;
        weights[2][0] = -1; weights[2][1] = 2;
        weights[2][2] = 4;  weights[2][3] = 1;
        weights[3][0] = 3;  weights[3][1] = -3;
        weights[3][2] = 2;  weights[3][3] = -2;
        clear_golden();

        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;
        for (int lane = 0; lane < 4; lane++) begin
            for (int output_lane = 0; output_lane < 4;
                 output_lane++) begin
                load_weight(
                    2'(lane),
                    2'(output_lane),
                    8'(weights[lane][output_lane]),
                    lane == 3 && output_lane == 3
                );
            end
        end

        // P1 coverage closure: fill the depth-2 FIFO while the executor is
        // blocked, hold a third item at valid, then release the executor. The
        // third push must be accepted in the same cycle as the full-FIFO pop.
        clear_golden();
        start_group();
        force_acc_stall = 1'b1;
        send_item(9'(3), 2'(0), 3'(1), 4'(0), 1'b0);
        send_item(9'(5), 2'(1), 3'(1), 4'(1), 1'b0);
        fork
            send_item(9'(7), 2'(2), 3'(1), 4'(2), 1'b1);
            begin
                while (!source_valid ||
                       32'(dut.fifo_count_q) != 2) begin
                    @(negedge clk_core);
                end
                @(negedge clk_core);
                force_acc_stall = 1'b0;
            end
        join
        wait_done(1);
        if (full_pop_push_count == 0) begin
            $error("full FIFO simultaneous pop/push path not covered");
            error_count = error_count + 1;
        end

        start_group();
        clear_golden();
        send_item(9'(4), 2'(2), 3'(2), 4'(0), 1'b0);
        send_item(9'(4), 2'(2), 3'(2), 4'(2), 1'b0);
        send_item(9'(4), 2'(2), 3'(2), 4'(4), 1'b0);
        send_item(9'(4), 2'(2), 3'(2), 4'(6), 1'b0);
        send_item(9'(1), 2'(3), 3'(3), 4'(1), 1'b0);
        send_item(9'(2), 2'(1), 3'(4), 4'(8), 1'b1);
        wait_done(2);
        check_acc();

        if (protocol_error ||
            count_source_items != 6 ||
            count_queue_commands != 6 ||
            count_product_computes != 6 ||
            count_native_commands != 6 ||
            count_explode_commands != 15 ||
            count_fallback_terms != 0 ||
            count_set_terms != 0 ||
            count_multiset_terms != 6 ||
            max_fifo_occupancy == 0 ||
            stall_count == 0) begin
            $error("native-m baseline counter/protocol mismatch");
            error_count = error_count + 1;
        end

        clear_golden();
        start_group();
        @(negedge clk_core);
        group_close_valid = 1'b1;
        group_close_tag = 8'h33;
        while (!group_close_ready) begin
            @(negedge clk_core);
        end
        @(negedge clk_core);
        group_close_valid = 1'b0;
        wait_done(3);
        check_acc();
        if (count_close_markers != 1) begin
            $error("empty close marker count mismatch");
            error_count = error_count + 1;
        end

        if (error_count == 0) begin
            $display("PASS native-m baseline: items=6 products=6 explode=15 groups=3 stalls=%0d full_pop_push=%0d",
                     stall_count, full_pop_push_count);
        end else begin
            $fatal(1, "FAIL native-m baseline errors=%0d", error_count);
        end
        $finish;
    end

endmodule

`default_nettype wire
