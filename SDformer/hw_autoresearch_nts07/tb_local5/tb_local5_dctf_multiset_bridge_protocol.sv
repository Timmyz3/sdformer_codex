`timescale 1ns/1ps
`default_nettype none

module tb_local5_dctf_multiset_bridge_protocol;
    logic clk_core;
    logic rst_core;

    logic term_valid;
    logic term_ready;
    logic [15:0] term_tag;
    logic [7:0] term_dest_id;
    logic [4:0] term_lane;
    logic [8:0] term_gate;
    logic [2:0] term_mult;
    logic term_last;
    logic term_head_last;

    logic cmd_valid;
    logic cmd_ready;
    logic [15:0] cmd_group_tag;
    logic [15:0] cmd_sequence;
    logic [8:0] cmd_gate_code;
    logic [4:0] cmd_lane_id;
    logic [7:0] cmd_destination_token;
    logic [2:0] cmd_multiplicity;
    logic [12:0] cmd_term_issue_seq;
    logic cmd_term_first;
    logic cmd_term_last;
    logic cmd_head_last;
    logic protocol_error;
    logic [31:0] count_cmds;
    logic [31:0] count_exploded;

    local5_dctf_multiset_bridge #(.EXPLODE(1'b1)) dut (.*);

    always #5 clk_core = ~clk_core;

    int beat_index;
    int errors;
    int exp_issue [0:5];
    int exp_first [0:5];
    int exp_last [0:5];
    int exp_head [0:5];
    int exp_dest [0:5];
    int exp_lane [0:5];
    int exp_gate [0:5];
    int zero_bubble_accepts;
    logic stalled_q;
    logic [72:0] stalled_payload_q;

    task automatic send_term(
        input int dest,
        input int lane,
        input int gate,
        input int mult,
        input bit destination_last,
        input bit head_last
    );
        @(negedge clk_core);
        term_valid = 1'b1;
        term_tag = 16'h1234;
        term_dest_id = 8'(dest);
        term_lane = 5'(lane);
        term_gate = 9'(gate);
        term_mult = 3'(mult);
        term_last = destination_last;
        term_head_last = head_last;
        while (!term_ready) @(negedge clk_core);
        @(negedge clk_core);
        term_valid = 1'b0;
    endtask

    // Exercise output stability and backpressure with a deterministic pattern.
    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cmd_ready <= 1'b0;
        end else begin
            cmd_ready <= ((beat_index + $time / 10) % 3) != 1;
        end
    end

    always_ff @(posedge clk_core) begin
        if (!rst_core && cmd_valid && cmd_ready) begin
            if (beat_index >= 6) begin
                $error("unexpected extra beat");
                errors <= errors + 1;
            end else begin
                if (cmd_group_tag !== 16'h1234 ||
                    cmd_sequence !== 16'(beat_index) ||
                    cmd_term_issue_seq !== 13'(exp_issue[beat_index]) ||
                    cmd_term_first !== exp_first[beat_index] ||
                    cmd_term_last !== exp_last[beat_index] ||
                    cmd_head_last !== exp_head[beat_index] ||
                    cmd_destination_token !== 8'(exp_dest[beat_index]) ||
                    cmd_lane_id !== 5'(exp_lane[beat_index]) ||
                    cmd_gate_code !== 9'(exp_gate[beat_index]) ||
                    cmd_multiplicity !== 3'd1) begin
                    $error("beat %0d mismatch seq=%0d issue=%0d f/l/h=%0b%0b%0b",
                           beat_index, cmd_sequence, cmd_term_issue_seq,
                           cmd_term_first, cmd_term_last, cmd_head_last);
                    errors <= errors + 1;
                end
                beat_index <= beat_index + 1;
            end
        end
        if (!rst_core && cmd_valid && cmd_ready && cmd_term_last &&
            term_valid && term_ready)
            zero_bubble_accepts <= zero_bubble_accepts + 1;
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            stalled_q <= 1'b0;
            stalled_payload_q <= '0;
        end else begin
            if (stalled_q) begin
                if (!cmd_valid ||
                    {cmd_group_tag, cmd_sequence, cmd_gate_code, cmd_lane_id,
                     cmd_destination_token, cmd_multiplicity,
                     cmd_term_issue_seq, cmd_term_first, cmd_term_last,
                     cmd_head_last} !== stalled_payload_q)
                    $fatal(1, "output changed while stalled");
            end
            stalled_q <= cmd_valid && !cmd_ready;
            if (cmd_valid && !cmd_ready)
                stalled_payload_q <=
                    {cmd_group_tag, cmd_sequence, cmd_gate_code, cmd_lane_id,
                     cmd_destination_token, cmd_multiplicity,
                     cmd_term_issue_seq, cmd_term_first, cmd_term_last,
                     cmd_head_last};
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        term_valid = 1'b0;
        term_tag = '0;
        term_dest_id = '0;
        term_lane = '0;
        term_gate = '0;
        term_mult = '0;
        term_last = 1'b0;
        term_head_last = 1'b0;
        beat_index = 0;
        errors = 0;
        zero_bubble_accepts = 0;
        stalled_q = 1'b0;
        stalled_payload_q = '0;

        // Term 0: multiplicity 3, not the final MFEP item of destination.
        exp_issue[0] = 0; exp_first[0] = 1; exp_last[0] = 0; exp_head[0] = 0;
        exp_issue[1] = 0; exp_first[1] = 0; exp_last[1] = 0; exp_head[1] = 0;
        exp_issue[2] = 0; exp_first[2] = 0; exp_last[2] = 1; exp_head[2] = 0;
        for (int i = 0; i < 3; i++) begin
            exp_dest[i] = 4; exp_lane[i] = 2; exp_gate[i] = 17;
        end

        // Term 1: final MFEP item of a destination, but not final head item.
        exp_issue[3] = 1; exp_first[3] = 1; exp_last[3] = 1; exp_head[3] = 0;
        exp_dest[3] = 4; exp_lane[3] = 7; exp_gate[3] = 23;

        // Term 2: final MFEP and head item, multiplicity 2.
        exp_issue[4] = 2; exp_first[4] = 1; exp_last[4] = 0; exp_head[4] = 0;
        exp_issue[5] = 2; exp_first[5] = 0; exp_last[5] = 1; exp_head[5] = 1;
        for (int i = 4; i < 6; i++) begin
            exp_dest[i] = 9; exp_lane[i] = 11; exp_gate[i] = 31;
        end

        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        send_term(4, 2, 17, 3, 1'b0, 1'b0);
        send_term(4, 7, 23, 1, 1'b1, 1'b0);
        send_term(9, 11, 31, 2, 1'b1, 1'b1);

        wait (beat_index == 6);
        repeat (4) @(posedge clk_core);
        if (protocol_error || count_cmds != 6 || count_exploded != 3 ||
            zero_bubble_accepts == 0)
            errors++;
        if (errors != 0)
            $fatal(1, "FAIL errors=%0d cmds=%0d exploded=%0d",
                   errors, count_cmds, count_exploded);
        $display("PASS tb_local5_dctf_multiset_bridge_protocol beats=%0d zero_bubble=%0d",
                 beat_index, zero_bubble_accepts);
        $finish;
    end

    initial begin
        #200000;
        $fatal(1, "TIMEOUT");
    end
endmodule

`default_nettype wire
