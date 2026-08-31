module m363_banked_q128_exact_signed_residual_matcher (
    input  logic          core_clk,
    input  logic          reset_n,
    input  logic          cfg_valid,
    output logic          cfg_ready,
    input  logic [2:0]    cfg_group,
    input  logic [255:0]  cfg_patterns_flat,
    input  logic          cfg_commit,
    output logic          cfg_active,
    output logic          cfg_protocol_error,
    input  logic          in_valid,
    output logic          in_ready,
    input  logic [15:0]   in_original_pattern,
    output logic          out_valid,
    input  logic          out_ready,
    output logic [15:0]   out_original_pattern,
    output logic [15:0]   out_best_center,
    output logic [6:0]    out_best_center_id,
    output logic [4:0]    out_best_distance,
    output logic [4:0]    out_population,
    output logic          out_use_pwp,
    output logic          out_fallback_bit_sparse,
    output logic [15:0]   out_plus_mask,
    output logic [15:0]   out_minus_mask
);

    localparam integer PATTERNS = 128;
    localparam integer STAGE0_GROUPS = 32;
    localparam integer STAGE1_GROUPS = 8;
    localparam integer STAGE2_GROUPS = 2;

    // Winner layout: {distance[4:0], center_id[6:0], center[15:0]}.
    // Lowest center ID is the frozen tie policy used by M348/M356.
    function automatic logic [27:0] pick_best(
        input logic [27:0] left,
        input logic [27:0] right
    );
        begin
            if ((right[27:23] < left[27:23]) ||
                    ((right[27:23] == left[27:23]) &&
                     (right[22:16] < left[22:16])))
                pick_best = right;
            else
                pick_best = left;
        end
    endfunction

    function automatic logic [4:0] popcount16(input logic [15:0] value);
        integer bit_index;
        logic [4:0] count;
        begin
            count = '0;
            for (bit_index = 0; bit_index < 16; bit_index = bit_index + 1)
                count = count + value[bit_index];
            popcount16 = count;
        end
    endfunction

    logic [15:0] pattern_q [0:PATTERNS-1];
    logic [3:0] cfg_next_group_q;

    logic [27:0] candidate_d [0:PATTERNS-1];
    logic [27:0] stage0_pair_d [0:63];
    logic [27:0] stage0_group_d [0:STAGE0_GROUPS-1];
    logic [27:0] stage0_winner_q [0:STAGE0_GROUPS-1];
    logic        stage0_valid_q;
    logic [15:0] stage0_original_q;
    logic [4:0]  stage0_population_q;

    logic [27:0] stage1_pair_d [0:15];
    logic [27:0] stage1_group_d [0:STAGE1_GROUPS-1];
    logic [27:0] stage1_winner_q [0:STAGE1_GROUPS-1];
    logic        stage1_valid_q;
    logic [15:0] stage1_original_q;
    logic [4:0]  stage1_population_q;

    logic [27:0] stage2_pair_d [0:3];
    logic [27:0] stage2_group_d [0:STAGE2_GROUPS-1];
    logic [27:0] stage2_winner_q [0:STAGE2_GROUPS-1];
    logic        stage2_valid_q;
    logic [15:0] stage2_original_q;
    logic [4:0]  stage2_population_q;

    logic stage3_ready_d;
    logic stage2_ready_d;
    logic stage1_ready_d;
    logic stage0_ready_d;
    logic pipeline_empty_d;
    logic use_pwp_d;

    integer candidate_index;
    integer pair_index;
    integer group_index;
    always_comb begin
        for (candidate_index = 0; candidate_index < PATTERNS;
                candidate_index = candidate_index + 1) begin
            candidate_d[candidate_index][15:0] = pattern_q[candidate_index];
            candidate_d[candidate_index][22:16] = candidate_index[6:0];
            candidate_d[candidate_index][27:23] = popcount16(
                in_original_pattern ^ pattern_q[candidate_index]);
        end
        for (pair_index = 0; pair_index < 64;
                pair_index = pair_index + 1)
            stage0_pair_d[pair_index] = pick_best(
                candidate_d[2 * pair_index],
                candidate_d[2 * pair_index + 1]);
        for (group_index = 0; group_index < STAGE0_GROUPS;
                group_index = group_index + 1)
            stage0_group_d[group_index] = pick_best(
                stage0_pair_d[2 * group_index],
                stage0_pair_d[2 * group_index + 1]);

        for (pair_index = 0; pair_index < 16;
                pair_index = pair_index + 1)
            stage1_pair_d[pair_index] = pick_best(
                stage0_winner_q[2 * pair_index],
                stage0_winner_q[2 * pair_index + 1]);
        for (group_index = 0; group_index < STAGE1_GROUPS;
                group_index = group_index + 1)
            stage1_group_d[group_index] = pick_best(
                stage1_pair_d[2 * group_index],
                stage1_pair_d[2 * group_index + 1]);

        for (pair_index = 0; pair_index < 4;
                pair_index = pair_index + 1)
            stage2_pair_d[pair_index] = pick_best(
                stage1_winner_q[2 * pair_index],
                stage1_winner_q[2 * pair_index + 1]);
        for (group_index = 0; group_index < STAGE2_GROUPS;
                group_index = group_index + 1)
            stage2_group_d[group_index] = pick_best(
                stage2_pair_d[2 * group_index],
                stage2_pair_d[2 * group_index + 1]);
    end

    assign stage3_ready_d = !out_valid || out_ready;
    assign stage2_ready_d = !stage2_valid_q || stage3_ready_d;
    assign stage1_ready_d = !stage1_valid_q || stage2_ready_d;
    assign stage0_ready_d = !stage0_valid_q || stage1_ready_d;
    assign pipeline_empty_d = !stage0_valid_q && !stage1_valid_q &&
                              !stage2_valid_q && !out_valid;

    // Error quarantine is intrinsic rather than an output-only mask: after a
    // bad configuration beat neither configuration nor input can handshake,
    // and cfg_active cannot revive until reset.
    assign cfg_ready = pipeline_empty_d && !cfg_protocol_error;
    assign in_ready = cfg_active && !cfg_protocol_error && !cfg_valid &&
                      stage0_ready_d;

    assign use_pwp_d = ({1'b0, out_best_distance} + 6'd1) <
                       {1'b0, out_population};
    assign out_use_pwp = out_valid && use_pwp_d;
    assign out_fallback_bit_sparse = out_valid && !use_pwp_d;
    assign out_plus_mask = use_pwp_d ?
        (out_original_pattern & ~out_best_center) : out_original_pattern;
    assign out_minus_mask = use_pwp_d ?
        (out_best_center & ~out_original_pattern) : 16'h0000;

    integer reset_index;
    integer cfg_lane;
    integer register_index;
    always_ff @(posedge core_clk or negedge reset_n) begin
        if (!reset_n) begin
            cfg_active <= 1'b0;
            cfg_protocol_error <= 1'b0;
            cfg_next_group_q <= '0;
            stage0_valid_q <= 1'b0;
            stage1_valid_q <= 1'b0;
            stage2_valid_q <= 1'b0;
            out_valid <= 1'b0;
            stage0_original_q <= '0;
            stage1_original_q <= '0;
            stage2_original_q <= '0;
            out_original_pattern <= '0;
            stage0_population_q <= '0;
            stage1_population_q <= '0;
            stage2_population_q <= '0;
            out_population <= '0;
            out_best_center <= '0;
            out_best_center_id <= '0;
            out_best_distance <= '0;
            for (reset_index = 0; reset_index < PATTERNS;
                    reset_index = reset_index + 1)
                pattern_q[reset_index] <= '0;
            for (reset_index = 0; reset_index < STAGE0_GROUPS;
                    reset_index = reset_index + 1)
                stage0_winner_q[reset_index] <= '0;
            for (reset_index = 0; reset_index < STAGE1_GROUPS;
                    reset_index = reset_index + 1)
                stage1_winner_q[reset_index] <= '0;
            for (reset_index = 0; reset_index < STAGE2_GROUPS;
                    reset_index = reset_index + 1)
                stage2_winner_q[reset_index] <= '0;
        end else begin
            if (cfg_valid && cfg_ready) begin
                if (cfg_group != cfg_next_group_q[2:0] ||
                        cfg_next_group_q >= 8 ||
                        (cfg_group != 7 && cfg_commit) ||
                        (cfg_group == 7 && !cfg_commit)) begin
                    cfg_protocol_error <= 1'b1;
                    cfg_active <= 1'b0;
                    cfg_next_group_q <= '0;
                end else begin
                    for (cfg_lane = 0; cfg_lane < 16;
                            cfg_lane = cfg_lane + 1)
                        pattern_q[cfg_group * 16 + cfg_lane] <=
                            cfg_patterns_flat[cfg_lane * 16 +: 16];
                    if (cfg_group == 0)
                        cfg_active <= 1'b0;
                    if (cfg_group == 7) begin
                        cfg_active <= 1'b1;
                        cfg_next_group_q <= '0;
                    end else begin
                        cfg_next_group_q <= cfg_next_group_q + 1'b1;
                    end
                end
            end

            if (stage3_ready_d) begin
                out_valid <= stage2_valid_q;
                if (stage2_valid_q) begin
                    out_original_pattern <= stage2_original_q;
                    out_population <= stage2_population_q;
                    if ((stage2_winner_q[1][27:23] <
                            stage2_winner_q[0][27:23]) ||
                            ((stage2_winner_q[1][27:23] ==
                              stage2_winner_q[0][27:23]) &&
                             (stage2_winner_q[1][22:16] <
                              stage2_winner_q[0][22:16]))) begin
                        out_best_center <= stage2_winner_q[1][15:0];
                        out_best_center_id <= stage2_winner_q[1][22:16];
                        out_best_distance <= stage2_winner_q[1][27:23];
                    end else begin
                        out_best_center <= stage2_winner_q[0][15:0];
                        out_best_center_id <= stage2_winner_q[0][22:16];
                        out_best_distance <= stage2_winner_q[0][27:23];
                    end
                end
            end
            if (stage2_ready_d) begin
                stage2_valid_q <= stage1_valid_q;
                if (stage1_valid_q) begin
                    stage2_original_q <= stage1_original_q;
                    stage2_population_q <= stage1_population_q;
                    for (register_index = 0;
                            register_index < STAGE2_GROUPS;
                            register_index = register_index + 1)
                        stage2_winner_q[register_index] <=
                            stage2_group_d[register_index];
                end
            end
            if (stage1_ready_d) begin
                stage1_valid_q <= stage0_valid_q;
                if (stage0_valid_q) begin
                    stage1_original_q <= stage0_original_q;
                    stage1_population_q <= stage0_population_q;
                    for (register_index = 0;
                            register_index < STAGE1_GROUPS;
                            register_index = register_index + 1)
                        stage1_winner_q[register_index] <=
                            stage1_group_d[register_index];
                end
            end
            if (stage0_ready_d) begin
                stage0_valid_q <= in_valid && in_ready;
                if (in_valid && in_ready) begin
                    stage0_original_q <= in_original_pattern;
                    stage0_population_q <= popcount16(in_original_pattern);
                    for (register_index = 0;
                            register_index < STAGE0_GROUPS;
                            register_index = register_index + 1)
                        stage0_winner_q[register_index] <=
                            stage0_group_d[register_index];
                end
            end
        end
    end

endmodule
