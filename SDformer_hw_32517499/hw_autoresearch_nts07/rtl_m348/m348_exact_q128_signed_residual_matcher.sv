module m348_exact_q128_signed_residual_matcher (
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

    localparam integer STAGES = 128;

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

    logic [15:0] pattern_q [0:STAGES-1];
    logic [3:0]  cfg_next_group_q;

    logic        valid_q [0:STAGES-1];
    logic [15:0] original_q [0:STAGES-1];
    logic [15:0] best_center_q [0:STAGES-1];
    logic [6:0]  best_id_q [0:STAGES-1];
    logic [4:0]  best_distance_q [0:STAGES-1];
    logic [4:0]  population_q [0:STAGES-1];

    logic [4:0] input_distance_d;
    logic [4:0] stage_distance_d [1:STAGES-1];
    logic pipeline_nonempty_d;
    logic advance_d;
    logic use_pwp_d;

    assign input_distance_d = popcount16(
        in_original_pattern ^ pattern_q[0]);

    genvar stage_gen;
    generate
        for (stage_gen = 1; stage_gen < STAGES;
                stage_gen = stage_gen + 1) begin : g_stage_distance
            always_comb begin
                stage_distance_d[stage_gen] = popcount16(
                    original_q[stage_gen-1] ^ pattern_q[stage_gen]);
            end
        end
    endgenerate

    integer reduce_index;
    always_comb begin
        pipeline_nonempty_d = 1'b0;
        for (reduce_index = 0; reduce_index < STAGES;
                reduce_index = reduce_index + 1)
            pipeline_nonempty_d = pipeline_nonempty_d | valid_q[reduce_index];
    end

    assign out_valid = valid_q[STAGES-1];
    assign advance_d = !out_valid || out_ready;
    assign cfg_ready = !pipeline_nonempty_d;
    assign in_ready = cfg_active && !cfg_protocol_error &&
                      !cfg_valid && advance_d;

    assign out_original_pattern = original_q[STAGES-1];
    assign out_best_center = best_center_q[STAGES-1];
    assign out_best_center_id = best_id_q[STAGES-1];
    assign out_best_distance = best_distance_q[STAGES-1];
    assign out_population = population_q[STAGES-1];
    assign use_pwp_d = ({1'b0, best_distance_q[STAGES-1]} + 6'd1) <
                       {1'b0, population_q[STAGES-1]};
    assign out_use_pwp = out_valid && use_pwp_d;
    assign out_fallback_bit_sparse = out_valid && !use_pwp_d;
    assign out_plus_mask = use_pwp_d ?
        (original_q[STAGES-1] & ~best_center_q[STAGES-1]) :
        original_q[STAGES-1];
    assign out_minus_mask = use_pwp_d ?
        (best_center_q[STAGES-1] & ~original_q[STAGES-1]) : 16'h0000;

    integer cfg_lane;
    integer reset_index;
    integer stage_index;
    always_ff @(posedge core_clk or negedge reset_n) begin
        if (!reset_n) begin
            cfg_active <= 1'b0;
            cfg_protocol_error <= 1'b0;
            cfg_next_group_q <= '0;
            for (reset_index = 0; reset_index < STAGES;
                    reset_index = reset_index + 1) begin
                pattern_q[reset_index] <= '0;
                valid_q[reset_index] <= 1'b0;
                original_q[reset_index] <= '0;
                best_center_q[reset_index] <= '0;
                best_id_q[reset_index] <= '0;
                best_distance_q[reset_index] <= '0;
                population_q[reset_index] <= '0;
            end
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

            if (advance_d) begin
                for (stage_index = STAGES-1; stage_index > 0;
                        stage_index = stage_index - 1) begin
                    valid_q[stage_index] <= valid_q[stage_index-1];
                    if (valid_q[stage_index-1]) begin
                        original_q[stage_index] <= original_q[stage_index-1];
                        population_q[stage_index] <= population_q[stage_index-1];
                        if (stage_distance_d[stage_index] <
                                best_distance_q[stage_index-1]) begin
                            best_center_q[stage_index] <= pattern_q[stage_index];
                            best_id_q[stage_index] <= stage_index[6:0];
                            best_distance_q[stage_index] <=
                                stage_distance_d[stage_index];
                        end else begin
                            best_center_q[stage_index] <=
                                best_center_q[stage_index-1];
                            best_id_q[stage_index] <= best_id_q[stage_index-1];
                            best_distance_q[stage_index] <=
                                best_distance_q[stage_index-1];
                        end
                    end
                end
                valid_q[0] <= in_valid && in_ready;
                if (in_valid && in_ready) begin
                    original_q[0] <= in_original_pattern;
                    population_q[0] <= popcount16(in_original_pattern);
                    best_center_q[0] <= pattern_q[0];
                    best_id_q[0] <= 7'd0;
                    best_distance_q[0] <= input_distance_d;
                end
            end
        end
    end

endmodule
