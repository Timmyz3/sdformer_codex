`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_replay_mux;
    logic clk_core;
    logic rst_core;
    logic route_start_valid;
    logic route_start_ready;
    logic [1:0] route_start_select;
    logic route_active;
    logic [1:0] route_active_select;
    logic [2:0] source_term_valid;
    logic [2:0] source_term_ready;
    logic [26:0] source_term_gate_code;
    logic [14:0] source_term_lane_id;
    logic [23:0] source_term_destination_count;
    logic [2:0] source_term_head_last;
    logic [2:0] source_event_valid;
    logic [2:0] source_event_ready;
    logic [26:0] source_event_gate_code;
    logic [14:0] source_event_lane_id;
    logic [11:0] source_event_token_valid;
    logic [95:0] source_event_token_ids;
    logic [8:0] source_event_count;
    logic [2:0] source_event_term_first;
    logic [2:0] source_event_term_last;
    logic [2:0] source_event_head_last;
    logic [2:0] source_done_valid;
    logic [2:0] source_done_ready;
    logic [95:0] source_done_tag;
    logic [2:0] source_done_error;
    logic term_valid;
    logic term_ready;
    logic [8:0] term_gate_code;
    logic [4:0] term_lane_id;
    logic [7:0] term_destination_count;
    logic [12:0] term_issue_seq;
    logic term_head_last;
    logic event_valid;
    logic event_ready;
    logic [8:0] event_gate_code;
    logic [4:0] event_lane_id;
    logic [3:0] event_token_valid;
    logic [31:0] event_token_ids;
    logic [2:0] event_count;
    logic [12:0] event_issue_seq;
    logic event_term_first;
    logic event_term_last;
    logic event_head_last;
    logic done_valid;
    logic done_ready;
    logic [31:0] done_tag;
    logic done_error;
    logic protocol_error;
    logic [31:0] count_completed_heads;
    logic [95:0] count_route_heads;

    gatestack_replay_mux dut (.*);
    always #5 clk_core <= ~clk_core;

    task automatic start_route(input int source);
        begin
            @(negedge clk_core);
            route_start_select = 2'(source);
            route_start_valid = 1'b1;
            do @(posedge clk_core); while (!route_start_ready);
            @(negedge clk_core);
            route_start_valid = 1'b0;
            if (!route_active || route_active_select != 2'(source)) begin
                $fatal(1, "route did not lock source=%0d", source);
            end
        end
    endtask

    task automatic drive_route(input int source);
        int terms;
        logic [8:0] gate;
        logic [4:0] lane;
        logic [7:0] token;
        logic [31:0] tag;
        begin
            terms = source + 1;
            tag = 32'h9000_0000 + 32'(source);
            start_route(source);
            for (int index = 0; index < terms; index = index + 1) begin
                gate = 9'(source * 16 + index + 1);
                lane = 5'(source * 4 + index);
                token = 8'(source * 20 + index);
                @(negedge clk_core);
                source_term_gate_code[(source*9) +: 9] = gate;
                source_term_lane_id[(source*5) +: 5] = lane;
                source_term_destination_count[(source*8) +: 8] = 8'd1;
                source_term_head_last[source] = index == terms - 1;
                source_term_valid = 3'b111;
                term_ready = 1'b0;
                @(posedge clk_core);
                if (!term_valid || source_term_ready != 0) begin
                    $fatal(1, "term stall routing mismatch source=%0d", source);
                end
                @(negedge clk_core);
                term_ready = 1'b1;
                @(posedge clk_core);
                if (!term_valid || !source_term_ready[source] ||
                    (source_term_ready & ~(3'b001 << source)) != 0 ||
                    term_gate_code != gate || term_lane_id != lane ||
                    term_destination_count != 1 ||
                    term_issue_seq != 13'(index) ||
                    term_head_last != (index == terms - 1)) begin
                    $fatal(1, "term route mismatch source=%0d index=%0d",
                           source, index);
                end
                @(negedge clk_core);
                source_term_valid = '0;
                term_ready = 1'b0;

                source_event_gate_code[(source*9) +: 9] = gate;
                source_event_lane_id[(source*5) +: 5] = lane;
                source_event_token_valid[(source*4) +: 4] = 4'b0001;
                source_event_token_ids[(source*32) +: 32] = {24'd0, token};
                source_event_count[(source*3) +: 3] = 3'd1;
                source_event_term_first[source] = 1'b1;
                source_event_term_last[source] = 1'b1;
                source_event_head_last[source] = index == terms - 1;
                source_event_valid = 3'b111;
                event_ready = 1'b0;
                @(posedge clk_core);
                if (!event_valid || source_event_ready != 0) begin
                    $fatal(1, "event stall routing mismatch source=%0d", source);
                end
                @(negedge clk_core);
                event_ready = 1'b1;
                @(posedge clk_core);
                if (!event_valid || !source_event_ready[source] ||
                    (source_event_ready & ~(3'b001 << source)) != 0 ||
                    event_gate_code != gate || event_lane_id != lane ||
                    event_token_valid != 4'b0001 ||
                    event_token_ids[7:0] != token ||
                    event_token_ids[31:8] != 0 || event_count != 1 ||
                    event_issue_seq != 13'(index) ||
                    !event_term_first || !event_term_last ||
                    event_head_last != (index == terms - 1)) begin
                    $fatal(1, "event route mismatch source=%0d index=%0d",
                           source, index);
                end
                @(negedge clk_core);
                source_event_valid = '0;
                event_ready = 1'b0;
            end

            source_done_tag[(source*32) +: 32] = tag;
            source_done_error[source] = 1'b0;
            source_done_valid = 3'b111;
            done_ready = 1'b0;
            @(posedge clk_core);
            if (!done_valid || source_done_ready != 0 || done_tag != tag ||
                done_error) begin
                $fatal(1, "done stall mismatch source=%0d", source);
            end
            @(negedge clk_core);
            done_ready = 1'b1;
            @(posedge clk_core);
            if (!done_valid || !source_done_ready[source] ||
                (source_done_ready & ~(3'b001 << source)) != 0 ||
                done_tag != tag || done_error) begin
                $fatal(1, "done route mismatch source=%0d", source);
            end
            @(negedge clk_core);
            source_done_valid = '0;
            done_ready = 1'b0;
            if (route_active) $fatal(1, "route did not release source=%0d", source);
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        route_start_valid = 1'b0;
        route_start_select = '0;
        source_term_valid = '0;
        source_term_gate_code = '0;
        source_term_lane_id = '0;
        source_term_destination_count = '0;
        source_term_head_last = '0;
        source_event_valid = '0;
        source_event_gate_code = '0;
        source_event_lane_id = '0;
        source_event_token_valid = '0;
        source_event_token_ids = '0;
        source_event_count = '0;
        source_event_term_first = '0;
        source_event_term_last = '0;
        source_event_head_last = '0;
        source_done_valid = '0;
        source_done_tag = '0;
        source_done_error = '0;
        term_ready = 1'b0;
        event_ready = 1'b0;
        done_ready = 1'b0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        drive_route(0);
        drive_route(1);
        drive_route(2);

        if (protocol_error || count_completed_heads != 3 ||
            count_route_heads[31:0] != 1 ||
            count_route_heads[63:32] != 1 ||
            count_route_heads[95:64] != 1) begin
            $fatal(1, "replay mux counters/error mismatch");
        end
        $display("PASS: replay mux completed=%0d routes={%0d,%0d,%0d}",
                 count_completed_heads, count_route_heads[31:0],
                 count_route_heads[63:32], count_route_heads[95:64]);
        $finish;
    end

    initial begin
        repeat (2000) @(posedge clk_core);
        $fatal(1, "replay mux TB timeout");
    end
endmodule

`default_nettype wire
