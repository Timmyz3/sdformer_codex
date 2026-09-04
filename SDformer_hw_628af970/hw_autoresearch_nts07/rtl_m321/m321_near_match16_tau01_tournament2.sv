module m321_near_match16_tau01_tournament2 (
    input  logic         core_clk,
    input  logic         reset_n,

    input  logic         in_valid,
    output logic         in_ready,
    input  logic [15:0]  in_pattern,
    input  logic [255:0] in_centers_flat,
    input  logic [1:0]   in_tau,

    output logic         out_valid,
    input  logic         out_ready,
    output logic [15:0]  out_original_pattern,
    output logic [15:0]  out_selected_pattern,
    output logic [4:0]   out_selected_distance,
    output logic [4:0]   out_population,
    output logic [1:0]   out_tau,
    output logic         out_snapped,
    output logic         out_exact_hit,
    output logic         out_positive_distance
);

    // A winner is {Hamming distance, packed center}.  pick_best is
    // associative under the total order (distance, unsigned center), so the
    // balanced tree is bit-exact with M311's serial recurrence.
    function automatic logic [20:0] pick_best(
        input logic [20:0] left,
        input logic [20:0] right
    );
        begin
            if ((right[20:16] < left[20:16]) ||
                    ((right[20:16] == left[20:16]) &&
                     (right[15:0] < left[15:0])))
                pick_best = right;
            else
                pick_best = left;
        end
    endfunction

    function automatic logic [4:0] popcount16(input logic [15:0] value);
        integer index;
        logic [4:0] count;
        begin
            count = '0;
            for (index = 0; index < 16; index = index + 1)
                count = count + value[index];
            popcount16 = count;
        end
    endfunction

    logic [20:0] candidate_d [0:15];
    logic [20:0] local_pair_d [0:7];
    logic [20:0] group_winner_d [0:3];
    logic [4:0]  population_d;

    integer candidate_index;
    integer pair_index;
    integer group_index;
    always_comb begin
        for (candidate_index = 0; candidate_index < 16;
                candidate_index = candidate_index + 1) begin
            candidate_d[candidate_index][15:0] =
                in_centers_flat[candidate_index * 16 +: 16];
            candidate_d[candidate_index][20:16] = popcount16(
                in_pattern ^ in_centers_flat[candidate_index * 16 +: 16]);
        end
        for (pair_index = 0; pair_index < 8;
                pair_index = pair_index + 1)
            local_pair_d[pair_index] = pick_best(
                candidate_d[2 * pair_index],
                candidate_d[2 * pair_index + 1]);
        for (group_index = 0; group_index < 4;
                group_index = group_index + 1)
            group_winner_d[group_index] = pick_best(
                local_pair_d[2 * group_index],
                local_pair_d[2 * group_index + 1]);
        population_d = popcount16(in_pattern);
    end

    logic        stage0_valid_q;
    logic [15:0] stage0_original_q;
    logic [15:0] stage0_center_q [0:3];
    logic [4:0]  stage0_distance_q [0:3];
    logic [4:0]  stage0_population_q;
    logic [1:0]  stage0_tau_q;

    logic [20:0] global_left_d;
    logic [20:0] global_right_d;
    logic [20:0] global_winner_d;
    logic        snap_d;

    always_comb begin
        global_left_d = pick_best(
            {stage0_distance_q[0], stage0_center_q[0]},
            {stage0_distance_q[1], stage0_center_q[1]});
        global_right_d = pick_best(
            {stage0_distance_q[2], stage0_center_q[2]},
            {stage0_distance_q[3], stage0_center_q[3]});
        global_winner_d = pick_best(global_left_d, global_right_d);
        snap_d = (stage0_population_q >= 2) &&
                 (global_winner_d[20:16] <= {3'b000, stage0_tau_q});
    end

    logic stage1_ready;
    logic stage0_ready;
    assign stage1_ready = !out_valid || out_ready;
    assign stage0_ready = !stage0_valid_q || stage1_ready;
    assign in_ready = stage0_ready;

    integer register_index;
    always_ff @(posedge core_clk or negedge reset_n) begin
        if (!reset_n) begin
            stage0_valid_q <= 1'b0;
            stage0_original_q <= '0;
            stage0_population_q <= '0;
            stage0_tau_q <= '0;
            for (register_index = 0; register_index < 4;
                    register_index = register_index + 1) begin
                stage0_center_q[register_index] <= '0;
                stage0_distance_q[register_index] <= '0;
            end
            out_valid <= 1'b0;
            out_original_pattern <= '0;
            out_selected_pattern <= '0;
            out_selected_distance <= '0;
            out_population <= '0;
            out_tau <= '0;
            out_snapped <= 1'b0;
            out_exact_hit <= 1'b0;
            out_positive_distance <= 1'b0;
        end else begin
            if (stage1_ready) begin
                out_valid <= stage0_valid_q;
                if (stage0_valid_q) begin
                    out_original_pattern <= stage0_original_q;
                    out_selected_pattern <= snap_d ?
                        global_winner_d[15:0] : stage0_original_q;
                    out_selected_distance <= global_winner_d[20:16];
                    out_population <= stage0_population_q;
                    out_tau <= stage0_tau_q;
                    out_snapped <= snap_d;
                    out_exact_hit <= snap_d &&
                        (global_winner_d[20:16] == 0);
                    out_positive_distance <= snap_d &&
                        (global_winner_d[20:16] != 0);
                end
            end
            if (stage0_ready) begin
                stage0_valid_q <= in_valid;
                if (in_valid) begin
                    stage0_original_q <= in_pattern;
                    stage0_population_q <= population_d;
                    stage0_tau_q <= in_tau;
                    for (register_index = 0; register_index < 4;
                            register_index = register_index + 1) begin
                        stage0_center_q[register_index] <=
                            group_winner_d[register_index][15:0];
                        stage0_distance_q[register_index] <=
                            group_winner_d[register_index][20:16];
                    end
                end
            end
        end
    end

endmodule
