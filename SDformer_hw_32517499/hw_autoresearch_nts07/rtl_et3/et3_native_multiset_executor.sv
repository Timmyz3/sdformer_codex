`timescale 1ns/1ps
`default_nettype none

// Native bank-local SET/MULTISET executor.
// A term computes multiplicity*gate*weight once and reuses the product for all
// destination beats in the segment. Motion is the exact multiplicity=1 mode.
module et3_native_multiset_executor #(
    parameter int HEAD_DIM = 4,
    parameter int OUT_DIM = 4,
    parameter int MAX_DEST = 16,
    parameter int TAG_W = 16,
    parameter int GATE_W = 9,
    parameter int LANE_W = 5,
    parameter int MULT_W = 3,
    parameter int DEST_W = 8,
    parameter int WEIGHT_W = 8,
    parameter int PRODUCT_W = GATE_W + MULT_W + WEIGHT_W,
    parameter int ACC_W = 32,
    parameter int COUNTER_W = 32,
    parameter int LANE_ADDR_W = (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM),
    parameter int DEST_ADDR_W = (MAX_DEST <= 1) ? 1 : $clog2(MAX_DEST),
    parameter int OUT_ID_W = (OUT_DIM <= 1) ? 1 : $clog2(OUT_DIM)
) (
    input  logic                             clk_core,
    input  logic                             rst_core,
    input  logic                             flush,

    input  logic                             weight_load_valid,
    output logic                             weight_load_ready,
    input  logic [LANE_W-1:0]                weight_load_lane,
    input  logic [OUT_ID_W-1:0]              weight_load_output,
    input  logic signed [WEIGHT_W-1:0]       weight_load_value,
    input  logic                             weight_load_last,

    input  logic                             run_start,
    output logic                             run_active,
    input  logic                             empty_group_commit,

    input  logic                             cmd_valid,
    output logic                             cmd_ready,
    input  logic [TAG_W-1:0]                 cmd_group_tag,
    input  logic                             cmd_mode_multiset,
    input  logic [GATE_W-1:0]                cmd_gate_code,
    input  logic [LANE_W-1:0]                cmd_lane_id,
    input  logic [MULT_W-1:0]                cmd_multiplicity,
    input  logic [DEST_W-1:0]                cmd_destination,
    input  logic                             cmd_term_first,
    input  logic                             cmd_term_last,
    input  logic                             cmd_head_last,
    input  logic                             cmd_fallback,
    input  logic                             acc_write_ready,

    input  logic                             acc_read_valid,
    input  logic [DEST_W-1:0]                acc_read_destination,
    input  logic [OUT_ID_W-1:0]              acc_read_output,
    output logic                             acc_read_data_valid,
    output logic signed [ACC_W-1:0]          acc_read_data,

    output logic                             group_done,
    output logic                             protocol_error,
    output logic [COUNTER_W-1:0]             count_product_computes,
    output logic [COUNTER_W-1:0]             count_native_commands,
    output logic [COUNTER_W-1:0]             count_explode_baseline_commands,
    output logic [COUNTER_W-1:0]             count_fallback_terms,
    output logic [COUNTER_W-1:0]             count_set_terms,
    output logic [COUNTER_W-1:0]             count_multiset_terms
);
    logic signed [WEIGHT_W-1:0] weight_q [0:HEAD_DIM-1][0:OUT_DIM-1];
    logic signed [ACC_W-1:0] acc_q [0:MAX_DEST-1][0:OUT_DIM-1];
    logic signed [PRODUCT_W-1:0] product_q [0:OUT_DIM-1];

    logic weights_loaded_q;
    logic term_active_q;
    logic [TAG_W-1:0] term_tag_q;
    logic term_mode_q;
    logic [GATE_W-1:0] term_gate_q;
    logic [LANE_W-1:0] term_lane_q;
    logic [MULT_W-1:0] term_mult_q;
    logic cmd_identity_ok;
    logic cmd_base_ok;
    logic cmd_contract_ok;
    logic cmd_fire;
    logic signed [PRODUCT_W-1:0] first_product [0:OUT_DIM-1];

    function automatic logic signed [PRODUCT_W-1:0] make_product(
        input logic [GATE_W-1:0] gate_value,
        input logic [MULT_W-1:0] mult_value,
        input logic signed [WEIGHT_W-1:0] weight_value
    );
        logic signed [GATE_W:0] gate_positive;
        logic signed [MULT_W:0] mult_positive;
        begin
            gate_positive = $signed({1'b0, gate_value});
            mult_positive = $signed({1'b0, mult_value});
            make_product = PRODUCT_W'(
                gate_positive * mult_positive * weight_value
            );
        end
    endfunction

    assign weight_load_ready = !run_active;
    assign cmd_identity_ok =
        (cmd_group_tag == term_tag_q) &&
        (cmd_mode_multiset == term_mode_q) &&
        (cmd_gate_code == term_gate_q) &&
        (cmd_lane_id == term_lane_q) &&
        (cmd_multiplicity == term_mult_q);
    assign cmd_base_ok =
        (cmd_gate_code != '0) &&
        (cmd_multiplicity != '0) &&
        (32'(cmd_multiplicity) <= 5) &&
        (cmd_mode_multiset || (cmd_multiplicity == MULT_W'(1))) &&
        (32'(cmd_lane_id) < HEAD_DIM) &&
        (32'(cmd_destination) < MAX_DEST) &&
        (!cmd_head_last || cmd_term_last);
    assign cmd_contract_ok = cmd_base_ok &&
        ((!term_active_q && cmd_term_first) ||
         (term_active_q && !cmd_term_first && cmd_identity_ok));
    assign cmd_ready = run_active && acc_write_ready && cmd_contract_ok;
    assign cmd_fire = cmd_valid && cmd_ready;

    always_comb begin
        for (int output_lane = 0; output_lane < OUT_DIM;
             output_lane++) begin
            first_product[output_lane] = make_product(
                cmd_gate_code,
                cmd_multiplicity,
                weight_q[LANE_ADDR_W'(cmd_lane_id)][output_lane]
            );
        end
    end

    assign acc_read_data_valid = acc_read_valid;
    assign acc_read_data =
        ((32'(acc_read_destination) < MAX_DEST) &&
         (32'(acc_read_output) < OUT_DIM)) ?
        acc_q[DEST_ADDR_W'(acc_read_destination)][acc_read_output] : '0;

    always_ff @(posedge clk_core) begin
        if (rst_core || flush) begin
            run_active <= 1'b0;
            weights_loaded_q <= 1'b0;
            term_active_q <= 1'b0;
            term_tag_q <= '0;
            term_mode_q <= 1'b0;
            term_gate_q <= '0;
            term_lane_q <= '0;
            term_mult_q <= '0;
            group_done <= 1'b0;
            protocol_error <= 1'b0;
            count_product_computes <= '0;
            count_native_commands <= '0;
            count_explode_baseline_commands <= '0;
            count_fallback_terms <= '0;
            count_set_terms <= '0;
            count_multiset_terms <= '0;
            for (int lane = 0; lane < HEAD_DIM; lane++) begin
                for (int output_lane = 0; output_lane < OUT_DIM;
                     output_lane++) begin
                    weight_q[lane][output_lane] <= '0;
                end
            end
            for (int destination = 0; destination < MAX_DEST;
                 destination++) begin
                for (int output_lane = 0; output_lane < OUT_DIM;
                     output_lane++) begin
                    acc_q[destination][output_lane] <= '0;
                end
            end
            for (int output_lane = 0; output_lane < OUT_DIM;
                 output_lane++) begin
                product_q[output_lane] <= '0;
            end
        end else begin
            group_done <= 1'b0;

            if (weight_load_valid && weight_load_ready) begin
                if ((32'(weight_load_lane) >= HEAD_DIM) ||
                    (32'(weight_load_output) >= OUT_DIM)) begin
                    protocol_error <= 1'b1;
                end else begin
                    weight_q[LANE_ADDR_W'(weight_load_lane)][
                        weight_load_output
                    ] <=
                        weight_load_value;
                    if (weight_load_last) begin
                        weights_loaded_q <= 1'b1;
                    end
                end
            end

            if (run_start) begin
                if (!weights_loaded_q || run_active) begin
                    protocol_error <= 1'b1;
                end else begin
                    run_active <= 1'b1;
                    term_active_q <= 1'b0;
                    count_product_computes <= '0;
                    count_native_commands <= '0;
                    count_explode_baseline_commands <= '0;
                    count_fallback_terms <= '0;
                    count_set_terms <= '0;
                    count_multiset_terms <= '0;
                    for (int destination = 0; destination < MAX_DEST;
                         destination++) begin
                        for (int output_lane = 0; output_lane < OUT_DIM;
                             output_lane++) begin
                            acc_q[destination][output_lane] <= '0;
                        end
                    end
                end
            end

            if (empty_group_commit) begin
                if (!run_active || term_active_q || cmd_valid) begin
                    protocol_error <= 1'b1;
                end else begin
                    run_active <= 1'b0;
                    group_done <= 1'b1;
                end
            end

            if (cmd_valid && run_active && !cmd_contract_ok) begin
                protocol_error <= 1'b1;
            end

            if (cmd_fire) begin
                count_native_commands <= count_native_commands + 1'b1;
                count_explode_baseline_commands <=
                    count_explode_baseline_commands +
                    COUNTER_W'(cmd_multiplicity);
                if (cmd_fallback && cmd_term_first) begin
                    count_fallback_terms <= count_fallback_terms + 1'b1;
                end
                if (cmd_term_first) begin
                    count_product_computes <=
                        count_product_computes + 1'b1;
                    if (cmd_mode_multiset) begin
                        count_multiset_terms <= count_multiset_terms + 1'b1;
                    end else begin
                        count_set_terms <= count_set_terms + 1'b1;
                    end
                    term_tag_q <= cmd_group_tag;
                    term_mode_q <= cmd_mode_multiset;
                    term_gate_q <= cmd_gate_code;
                    term_lane_q <= cmd_lane_id;
                    term_mult_q <= cmd_multiplicity;
                    for (int output_lane = 0; output_lane < OUT_DIM;
                         output_lane++) begin
                        product_q[output_lane] <=
                            first_product[output_lane];
                    end
                end

                for (int output_lane = 0; output_lane < OUT_DIM;
                     output_lane++) begin
                    if (cmd_term_first) begin
                        acc_q[DEST_ADDR_W'(cmd_destination)][output_lane] <=
                            acc_q[
                                DEST_ADDR_W'(cmd_destination)
                            ][output_lane] +
                            ACC_W'(first_product[output_lane]);
                    end else begin
                        acc_q[DEST_ADDR_W'(cmd_destination)][output_lane] <=
                            acc_q[
                                DEST_ADDR_W'(cmd_destination)
                            ][output_lane] +
                            ACC_W'(product_q[output_lane]);
                    end
                end

                term_active_q <= !cmd_term_last;
                if (cmd_head_last) begin
                    run_active <= 1'b0;
                    group_done <= 1'b1;
                end
            end
        end
    end

endmodule

`default_nettype wire
