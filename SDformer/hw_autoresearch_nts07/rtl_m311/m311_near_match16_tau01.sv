module m311_near_match16_tau01 (
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

    logic [15:0] best_center_d;
    logic [4:0]  best_distance_d;
    logic [4:0]  population_d;
    logic         snap_d;

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

    integer center_index;
    logic [15:0] candidate_center;
    logic [4:0] candidate_distance;
    always_comb begin
        best_center_d = in_centers_flat[15:0];
        best_distance_d = popcount16(in_pattern ^ in_centers_flat[15:0]);
        for (center_index = 1; center_index < 16;
                center_index = center_index + 1) begin
            candidate_center = in_centers_flat[center_index * 16 +: 16];
            candidate_distance = popcount16(in_pattern ^ candidate_center);
            if ((candidate_distance < best_distance_d) ||
                    ((candidate_distance == best_distance_d) &&
                     (candidate_center < best_center_d))) begin
                best_center_d = candidate_center;
                best_distance_d = candidate_distance;
            end
        end
        population_d = popcount16(in_pattern);
        snap_d = (population_d >= 2) && (best_distance_d <= {3'b000, in_tau});
    end

    assign in_ready = !out_valid || out_ready;

    always_ff @(posedge core_clk or negedge reset_n) begin
        if (!reset_n) begin
            out_valid <= 1'b0;
            out_original_pattern <= '0;
            out_selected_pattern <= '0;
            out_selected_distance <= '0;
            out_population <= '0;
            out_tau <= '0;
            out_snapped <= 1'b0;
            out_exact_hit <= 1'b0;
            out_positive_distance <= 1'b0;
        end else if (in_ready) begin
            out_valid <= in_valid;
            if (in_valid) begin
                out_original_pattern <= in_pattern;
                out_selected_pattern <= snap_d ? best_center_d : in_pattern;
                out_selected_distance <= best_distance_d;
                out_population <= population_d;
                out_tau <= in_tau;
                out_snapped <= snap_d;
                out_exact_hit <= snap_d && (best_distance_d == 0);
                out_positive_distance <= snap_d && (best_distance_d != 0);
            end
        end
    end

endmodule
