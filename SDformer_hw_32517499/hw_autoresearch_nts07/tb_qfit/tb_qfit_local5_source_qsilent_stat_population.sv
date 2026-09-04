`timescale 1ns/1ps
`default_nettype none

module tb_qfit_local5_source_qsilent_stat_population;
    localparam int GROUPS = 100;
    localparam int TOKENS = 450;
    localparam int TOTAL = GROUPS * TOKENS;
    localparam int TOKEN_W = 9;
    localparam int PLANE_TOKENS = 225;
    localparam int SIDE = 15;

    logic clk = 1'b0;
    logic rst;
    logic in_valid;
    logic in_ready;
    logic [TOKEN_W-1:0] in_source;
    logic [3:0] in_source_y;
    logic [3:0] in_source_x;
    logic [31:0] in_source_k;
    logic [4:0] in_consumer_mask;
    logic out_valid;
    logic out_ready;
    logic [TOKEN_W-1:0] out_source;
    logic signed [15:0] out_score;
    logic [4:0] out_consumer_valid;
    logic [5*TOKEN_W-1:0] out_destination;
    logic [14:0] out_bank;

    logic [31:0] q_mem [0:TOTAL-1];
    logic [159:0] k_mem [0:TOTAL-1];
    logic [4:0] valid_mem [0:TOTAL-1];
    logic [79:0] score_mem [0:TOTAL-1];
    logic [31:0] lfsr;
    integer checked_edges;
    integer checked_sources;
    integer group_idx;
    integer source_idx;
    logic [4:0] expected_mask_q;
    logic consumed_q;

    always #1 clk = ~clk;

    qfit_local5_source_qsilent_stat_router dut (
        .clk_core(clk),
        .rst_core(rst),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_source_token(in_source),
        .in_source_y(in_source_y),
        .in_source_x(in_source_x),
        .in_source_k(in_source_k),
        .in_consumer_qzero_mask(in_consumer_mask),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_source_token(out_source),
        .out_score_q7(out_score),
        .out_consumer_valid(out_consumer_valid),
        .out_destination(out_destination),
        .out_destination_bank(out_bank)
    );

    function automatic integer destination_for_role(
        input integer source_idx_arg,
        input integer role
    );
        integer within_idx;
        integer x;
        integer y;
        begin
            within_idx = source_idx_arg % PLANE_TOKENS;
            x = within_idx % SIDE;
            y = within_idx / SIDE;
            destination_for_role = -1;
            case (role)
                0: destination_for_role = source_idx_arg;
                1: if (y < SIDE-1) destination_for_role = source_idx_arg + SIDE;
                2: if (y > 0) destination_for_role = source_idx_arg - SIDE;
                3: if (x < SIDE-1) destination_for_role = source_idx_arg + 1;
                4: if (x > 0) destination_for_role = source_idx_arg - 1;
                default: destination_for_role = -1;
            endcase
        end
    endfunction

    task automatic build_consumer_mask(
        input integer group,
        input integer source_idx_arg,
        output logic [4:0] mask
    );
        integer destination;
        integer flat;
        begin
            mask = '0;
            for (int role = 0; role < 5; role = role + 1) begin
                destination = destination_for_role(source_idx_arg, role);
                if (destination >= 0) begin
                    flat = group * TOKENS + destination;
                    if ((q_mem[flat] == 32'd0) && valid_mem[flat][role])
                        mask[role] = 1'b1;
                end
            end
        end
    endtask

    task automatic check_output(
        input integer group,
        input integer source_idx_arg,
        input logic [4:0] expected_mask
    );
        integer destination;
        integer flat;
        logic [15:0] expected_score;
        begin
            if (out_source !== TOKEN_W'(source_idx_arg))
                $fatal(1, "source mismatch group=%0d expected=%0d actual=%0d",
                    group, source_idx_arg, out_source);
            if (out_consumer_valid !== expected_mask)
                $fatal(1, "mask mismatch group=%0d source=%0d expected=%b actual=%b",
                    group, source_idx_arg, expected_mask, out_consumer_valid);
            for (int role = 0; role < 5; role = role + 1) begin
                destination = destination_for_role(source_idx_arg, role);
                if (expected_mask[role]) begin
                    flat = group * TOKENS + destination;
                    if (out_destination[role*TOKEN_W +: TOKEN_W]
                        !== TOKEN_W'(destination))
                        $fatal(1, "destination mismatch g=%0d s=%0d role=%0d",
                            group, source_idx_arg, role);
                    expected_score = score_mem[flat][role*16 +: 16];
                    if (out_score !== $signed(expected_score))
                        $fatal(1,
                            "score mismatch g=%0d s=%0d d=%0d role=%0d exp=%h got=%h",
                            group, source_idx_arg, destination, role,
                            expected_score, out_score);
                    checked_edges = checked_edges + 1;
                end
            end
            checked_sources = checked_sources + 1;
        end
    endtask

    initial begin
        $readmemh("tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_v1_20260813/input_q.memh", q_mem);
        $readmemh("tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_v1_20260813/input_candidate_k.memh", k_mem);
        $readmemh("tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_v1_20260813/input_valid.memh", valid_mem);
        $readmemh("tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_v1_20260813/expected_scores.memh", score_mem);

        rst = 1'b1;
        in_valid = 1'b0;
        in_source = '0;
        in_source_y = '0;
        in_source_x = '0;
        in_source_k = '0;
        in_consumer_mask = '0;
        out_ready = 1'b0;
        lfsr = 32'h5a17_3c91;
        checked_edges = 0;
        checked_sources = 0;
        repeat (5) @(negedge clk);
        rst = 1'b0;

        for (group_idx = 0; group_idx < GROUPS; group_idx = group_idx + 1) begin
            for (source_idx = 0; source_idx < TOKENS;
                 source_idx = source_idx + 1) begin
                build_consumer_mask(group_idx, source_idx, expected_mask_q);
                @(negedge clk);
                in_source = TOKEN_W'(source_idx);
                in_source_y = 4'((source_idx % PLANE_TOKENS) / SIDE);
                in_source_x = 4'((source_idx % PLANE_TOKENS) % SIDE);
                in_source_k = k_mem[group_idx*TOKENS + source_idx][31:0];
                in_consumer_mask = expected_mask_q;
                in_valid = 1'b1;
                out_ready = lfsr[0] | lfsr[3];
                do begin
                    @(posedge clk);
                    lfsr = {lfsr[30:0],
                        lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
                end while (!in_ready);
                @(negedge clk);
                in_valid = 1'b0;

                consumed_q = 1'b0;
                while (!consumed_q) begin
                    out_ready = lfsr[0] | lfsr[3];
                    @(posedge clk);
                    lfsr = {lfsr[30:0],
                        lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
                    if (out_valid && out_ready) begin
                        check_output(group_idx, source_idx, expected_mask_q);
                        consumed_q = 1'b1;
                    end
                end
            end
        end

        @(negedge clk);
        out_ready = 1'b1;
        if (checked_sources != TOTAL)
            $fatal(1, "source count mismatch expected=%0d actual=%0d",
                TOTAL, checked_sources);
        if (checked_edges != 190575)
            $fatal(1, "edge count mismatch expected=190575 actual=%0d",
                checked_edges);
        $display("SOURCE_QSILENT_POPULATION sources=%0d edges=%0d mismatch=0",
            checked_sources, checked_edges);
        $display("PASS tb_qfit_local5_source_qsilent_stat_population");
        $finish;
    end
endmodule

`default_nettype wire
