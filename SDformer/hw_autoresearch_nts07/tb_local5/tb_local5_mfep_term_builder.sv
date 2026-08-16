`timescale 1ns/1ps
`default_nettype none

module tb_local5_mfep_term_builder;
    localparam int HEAD_DIM = 32;
    localparam int N_CAND   = 5;

    logic clk_core, rst_core;
    logic dest_valid, dest_ready;
    logic [15:0] dest_tag;
    logic [7:0] dest_id;
    logic edge_valid, edge_ready;
    logic [2:0] edge_dir;
    logic [8:0] edge_gate_q17;
    logic [31:0] edge_k_bits;
    logic edge_last;
    logic term_valid, term_ready;
    logic [15:0] term_tag;
    logic [7:0] term_dest_id;
    logic [4:0] term_lane;
    logic [8:0] term_gate_q17;
    logic [2:0] term_multiplicity;
    logic term_last;
    logic dest_done_valid, dest_done_ready;
    logic [15:0] dest_done_tag;
    logic protocol_error;
    logic [15:0] count_edges, count_terms, count_naive_products;

    local5_mfep_term_builder dut (.*);

    always #5 clk_core = ~clk_core;

    // Golden multiplicity map: for each (lane,gate) count
    int golden_mult [0:31][0:511]; // gate 0..511 enough for 9b
    int golden_gate_order [0:4];
    int golden_n_uniq;
    int golden_terms;
    int golden_naive;
    int expected_lane [0:159];
    int expected_gate [0:159];
    int expected_mult [0:159];
    int expected_count;
    int expected_index;

    task automatic clear_golden;
        for (int l = 0; l < 32; l++)
            for (int g = 0; g < 512; g++)
                golden_mult[l][g] = 0;
        golden_terms = 0;
        golden_naive = 0;
        golden_n_uniq = 0;
        expected_count = 0;
        expected_index = 0;
    endtask

    task automatic add_edge(input logic [8:0] gate, input logic [31:0] k);
        bit found;
        found = 0;
        if (gate != 0) begin
            for (int u = 0; u < golden_n_uniq; u++)
                if (golden_gate_order[u] == gate) found = 1;
            if (!found) begin
                golden_gate_order[golden_n_uniq] = gate;
                golden_n_uniq++;
            end
        end
        for (int l = 0; l < 32; l++) begin
            if (k[l] && gate != 0) begin
                if (golden_mult[l][gate] == 0) golden_terms++;
                golden_mult[l][gate]++;
                golden_naive++;
            end
        end
    endtask

    task automatic prepare_expected_order;
        expected_count = 0;
        expected_index = 0;
        // Frozen producer contract: lane-major, then gate in first valid
        // candidate occurrence order.
        for (int l = 0; l < 32; l++) begin
            for (int u = 0; u < golden_n_uniq; u++) begin
                if (golden_mult[l][golden_gate_order[u]] > 0) begin
                    expected_lane[expected_count] = l;
                    expected_gate[expected_count] = golden_gate_order[u];
                    expected_mult[expected_count] =
                        golden_mult[l][golden_gate_order[u]];
                    expected_count++;
                end
            end
        end
    endtask

    task automatic send_destination(
        input logic [15:0] tag,
        input logic [7:0] destination
    );
        @(negedge clk_core);
        dest_valid = 1;
        dest_tag = tag;
        dest_id = destination;
        do @(posedge clk_core); while (!dest_ready);
        @(negedge clk_core);
        dest_valid = 0;
    endtask

    task automatic send_edge(
        input logic [2:0] direction,
        input logic [8:0] gate,
        input logic [31:0] k,
        input logic last
    );
        @(negedge clk_core);
        edge_valid = 1;
        edge_dir = direction;
        edge_gate_q17 = gate;
        edge_k_bits = k;
        edge_last = last;
        do @(posedge clk_core); while (!edge_ready);
        @(negedge clk_core);
        edge_valid = 0;
    endtask

    int errors;
    int checked_terms;
    int test_phase;

    initial begin
        clk_core = 0;
        rst_core = 1;
        dest_valid = 0;
        edge_valid = 0;
        term_ready = 0;
        dest_done_ready = 0;
        errors = 0;
        checked_terms = 0;
        test_phase = 0;
        repeat (4) @(posedge clk_core);
        rst_core = 0;

        // Case 1: two edges same gate on overlapping lanes -> mult=2
        clear_golden;
        begin
            logic [8:0] g0, g1;
            logic [31:0] k0, k1;
            g0 = 9'd128;
            g1 = 9'd128;
            k0 = 32'h0000_00FF;
            k1 = 32'h0000_0F0F;
            add_edge(g0, k0);
            add_edge(g1, k1);
            prepare_expected_order();

            test_phase = 11;
            send_destination(16'h1, 8'h3);
            test_phase = 12;
            send_edge(3'd0, g0, k0, 1'b0);
            test_phase = 13;
            send_edge(3'd1, g1, k1, 1'b1);
            @(negedge clk_core);
            term_ready = 1;

            test_phase = 14;
            while (!dest_done_valid) begin
                @(posedge clk_core);
                if (term_valid && term_ready) begin
                    if (expected_index >= expected_count ||
                        term_lane !== 5'(expected_lane[expected_index]) ||
                        term_gate_q17 !== 9'(expected_gate[expected_index]) ||
                        term_multiplicity !==
                            3'(expected_mult[expected_index])) begin
                        $error("producer order mismatch idx=%0d got={lane=%0d gate=%0d mult=%0d} exp={lane=%0d gate=%0d mult=%0d}",
                               expected_index, term_lane, term_gate_q17,
                               term_multiplicity,
                               expected_lane[expected_index],
                               expected_gate[expected_index],
                               expected_mult[expected_index]);
                        errors++;
                    end
                    if (term_last !==
                        (expected_index == expected_count - 1)) begin
                        $error("term_last order mismatch idx=%0d count=%0d",
                               expected_index, expected_count);
                        errors++;
                    end
                    expected_index++;
                    if (term_multiplicity !== 3'(golden_mult[term_lane][term_gate_q17])) begin
                        $error("mult mismatch lane=%0d gate=%0d got=%0d exp=%0d",
                               term_lane, term_gate_q17, term_multiplicity,
                               golden_mult[term_lane][term_gate_q17]);
                        errors++;
                    end
                    if (golden_mult[term_lane][term_gate_q17] == 0) begin
                        $error("unexpected term");
                        errors++;
                    end
                    golden_mult[term_lane][term_gate_q17] = -1; // mark seen
                    checked_terms++;
                end
            end
            @(negedge clk_core);
            dest_done_ready = 1;
            test_phase = 15;
            @(posedge clk_core);
            @(negedge clk_core);
            dest_done_ready = 0;
            term_ready = 0;
            if (expected_index != expected_count) begin
                $error("producer order count mismatch got=%0d exp=%0d",
                       expected_index, expected_count);
                errors++;
            end
            // ensure all expected terms seen
            for (int l = 0; l < 32; l++)
                for (int g = 0; g < 512; g++)
                    if (golden_mult[l][g] > 0) begin
                        $error("missing term lane=%0d gate=%0d mult=%0d", l, g, golden_mult[l][g]);
                        errors++;
                    end
            if (count_naive_products !== 16'(golden_naive)) begin
                $error("naive count got=%0d exp=%0d", count_naive_products, golden_naive);
                errors++;
            end
            @(posedge clk_core);
        end

        // Case 2: distinct gates, zero-gate skipped
        clear_golden;
        begin
            logic [8:0] ga, gb, gz;
            logic [31:0] ka, kb, kz;
            ga = 9'd64; gb = 9'd200; gz = 9'd0;
            ka = 32'hFFFF_0000;
            kb = 32'h0000_FFFF;
            kz = 32'hFFFF_FFFF;
            add_edge(ga, ka);
            add_edge(gb, kb);
            prepare_expected_order();
            // zero gate not added to golden products

            test_phase = 21;
            send_destination(16'h2, 8'h7);
            test_phase = 22;
            send_edge(3'd0, ga, ka, 1'b0);
            test_phase = 23;
            send_edge(3'd1, gz, kz, 1'b0);
            test_phase = 24;
            send_edge(3'd2, gb, kb, 1'b1);
            @(negedge clk_core);
            term_ready = 1;

            test_phase = 25;
            while (!dest_done_valid) begin
                @(posedge clk_core);
                if (term_valid && term_ready) begin
                    if (expected_index >= expected_count ||
                        term_lane !== 5'(expected_lane[expected_index]) ||
                        term_gate_q17 !== 9'(expected_gate[expected_index]) ||
                        term_multiplicity !==
                            3'(expected_mult[expected_index])) begin
                        $error("c2 producer order mismatch idx=%0d", expected_index);
                        errors++;
                    end
                    if (term_last !==
                        (expected_index == expected_count - 1)) begin
                        $error("c2 term_last order mismatch idx=%0d",
                               expected_index);
                        errors++;
                    end
                    expected_index++;
                    if (term_multiplicity !== 3'(golden_mult[term_lane][term_gate_q17])) begin
                        $error("c2 mult mismatch");
                        errors++;
                    end
                    golden_mult[term_lane][term_gate_q17] = -1;
                    checked_terms++;
                end
            end
            @(negedge clk_core);
            dest_done_ready = 1;
            test_phase = 26;
            @(posedge clk_core);
            @(negedge clk_core);
            dest_done_ready = 0;
            term_ready = 0;
            if (expected_index != expected_count) begin
                $error("c2 producer order count mismatch got=%0d exp=%0d",
                       expected_index, expected_count);
                errors++;
            end
            for (int l = 0; l < 32; l++)
                for (int g = 0; g < 512; g++)
                    if (golden_mult[l][g] > 0) begin
                        $error("c2 missing term");
                        errors++;
                    end
            @(posedge clk_core);
        end

        if (protocol_error) begin
            $error("protocol_error set");
            errors++;
        end
        if (errors != 0) $fatal(1, "FAIL errors=%0d", errors);
        $display("PASS tb_local5_mfep_term_builder checked_terms=%0d", checked_terms);
        $finish;
    end

    initial begin
        #200000;
        $fatal(1, "TIMEOUT phase=%0d state=%0d dest_valid=%b dest_ready=%b edge_valid=%b edge_ready=%b term_valid=%b done=%b n_edges=%0d n_uniq=%0d lane=%0d gate_idx=%0d",
               test_phase, dut.state_q, dest_valid, dest_ready,
               edge_valid, edge_ready, term_valid,
               dest_done_valid, dut.n_edges_q, dut.n_uniq_q,
               dut.lane_q, dut.ug_idx_q);
    end
endmodule

`default_nettype wire
