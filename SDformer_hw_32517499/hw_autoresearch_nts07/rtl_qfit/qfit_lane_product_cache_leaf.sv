`timescale 1ns/1ps
`default_nettype none

module qfit_lane_product_cache_leaf #(
    parameter int LANES = 32,
    parameter int WAYS = 4,
    parameter bit NO_REPLACE = 1'b0,
    parameter int OUT_DIM = 4,
    parameter int GATE_W = 9,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter int PLANE_W = 1,
    parameter int Y_W = 4,
    parameter int X_W = 4,
    parameter int DEST_MASK_W = 5,
    parameter int LANE_W = (LANES <= 1) ? 1 : $clog2(LANES),
    parameter int OUT_W = (OUT_DIM <= 1) ? 1 : $clog2(OUT_DIM)
) (
    input  logic clk_core,
    input  logic rst_core,

    input  logic weight_valid,
    output logic weight_ready,
    input  logic [LANE_W-1:0] weight_lane,
    input  logic [OUT_W-1:0] weight_out,
    input  logic signed [W_W-1:0] weight_data,
    input  logic weight_last,

    input  logic epoch_start_valid,
    output logic epoch_start_ready,
    input  logic epoch_close_valid,
    output logic epoch_close_ready,
    output logic epoch_active,
    output logic epoch_done,

    input  logic in_valid,
    output logic in_ready,
    input  logic [LANE_W-1:0] in_lane,
    input  logic [GATE_W-1:0] in_gate,
    input  logic [PLANE_W-1:0] in_source_plane,
    input  logic [Y_W-1:0] in_source_y,
    input  logic [X_W-1:0] in_source_x,
    input  logic [DEST_MASK_W-1:0] in_destination_mask,
    input  logic in_window_last,

    output logic out_valid,
    input  logic out_ready,
    output logic [LANE_W-1:0] out_lane,
    output logic [GATE_W-1:0] out_gate,
    output logic [PLANE_W-1:0] out_source_plane,
    output logic [Y_W-1:0] out_source_y,
    output logic [X_W-1:0] out_source_x,
    output logic [DEST_MASK_W-1:0] out_destination_mask,
    output logic out_window_last,
    output logic [OUT_DIM*ACC_W-1:0] out_product,

    output logic protocol_error,
    output logic [31:0] perf_accepted_terms,
    output logic [31:0] perf_cache_hits,
    output logic [31:0] perf_cache_misses,
    output logic [31:0] perf_tag_compares,
    output logic [31:0] perf_lru_writes,
    output logic [31:0] perf_product_reads,
    output logic [31:0] perf_product_writes,
    output logic [31:0] perf_product_starts,
    output logic [31:0] perf_weight_reads,
    output logic [31:0] perf_output_stalls
);
    localparam int PRODUCT_ELEM_W = GATE_W + W_W;
    localparam int CACHE_PRODUCT_W = OUT_DIM * PRODUCT_ELEM_W;
    localparam int WAY_W = (WAYS <= 1) ? 1 : $clog2(WAYS);
    localparam int AGE_W = (WAYS <= 1) ? 1 : $clog2(WAYS);

    logic signed [W_W-1:0] weight_q [0:LANES-1][0:OUT_DIM-1];
    logic weights_loaded_q;
    logic [LANE_W-1:0] weight_expected_lane_q;
    logic [OUT_W-1:0] weight_expected_out_q;
    logic epoch_active_q;
    logic closing_q;
    logic protocol_error_q;

    logic cache_valid_q [0:LANES-1][0:WAYS-1];
    logic [GATE_W-1:0] cache_gate_q [0:LANES-1][0:WAYS-1];
    logic [AGE_W-1:0] cache_age_q [0:LANES-1][0:WAYS-1];
    logic [CACHE_PRODUCT_W-1:0] product_bank_read_data [0:WAYS-1];
    logic [WAYS-1:0] product_bank_read_valid;
    logic [WAYS-1:0] product_bank_access;
    logic [WAYS-1:0] product_bank_write;

    logic out_valid_q;
    logic out_hit_q;
    logic [WAY_W-1:0] out_hit_way_q;
    logic [LANE_W-1:0] out_lane_q;
    logic [GATE_W-1:0] out_gate_q;
    logic [PLANE_W-1:0] out_plane_q;
    logic [Y_W-1:0] out_y_q;
    logic [X_W-1:0] out_x_q;
    logic [DEST_MASK_W-1:0] out_mask_q;
    logic out_window_last_q;
    logic [CACHE_PRODUCT_W-1:0] out_product_q;

    logic input_contract_valid;
    logic lookup_hit;
    logic free_found;
    logic [WAY_W-1:0] hit_way;
    logic [WAY_W-1:0] free_way;
    logic [WAY_W-1:0] victim_way;
    logic [AGE_W-1:0] victim_age;
    logic [WAY_W-1:0] target_way;
    logic [AGE_W-1:0] target_old_age;
    logic [CACHE_PRODUCT_W-1:0] computed_product;
    logic signed [W_W-1:0] isolated_weight [0:OUT_DIM-1];
    logic signed [PRODUCT_ELEM_W-1:0] narrow_product [0:OUT_DIM-1];
    logic multiply_enable;
    logic cache_insert;
    logic weight_contract_valid;
    logic weight_fire;
    logic in_fire;
    logic out_fire;
    logic epoch_start_fire;
    logic epoch_close_fire;

    logic [31:0] perf_accepted_q;
    logic [31:0] perf_hits_q;
    logic [31:0] perf_misses_q;
    logic [31:0] perf_compares_q;
    logic [31:0] perf_lru_q;
    logic [31:0] perf_product_reads_q;
    logic [31:0] perf_product_writes_q;
    logic [31:0] perf_product_starts_q;
    logic [31:0] perf_weight_reads_q;
    logic [31:0] perf_output_stalls_q;

    function automatic logic [OUT_DIM*ACC_W-1:0] expand_product(
        input logic [CACHE_PRODUCT_W-1:0] compact
    );
        logic signed [PRODUCT_ELEM_W-1:0] element;
        begin
            expand_product = '0;
            for (integer out = 0; out < OUT_DIM; out = out + 1) begin
                element =
                    compact[out*PRODUCT_ELEM_W +: PRODUCT_ELEM_W];
                expand_product[out*ACC_W +: ACC_W] =
                    {{(ACC_W-PRODUCT_ELEM_W){element[PRODUCT_ELEM_W-1]}},
                     element};
            end
        end
    endfunction

    always_comb begin
        input_contract_valid =
            32'(in_lane) < LANES
            && in_gate != '0
            && in_destination_mask != '0;

        lookup_hit = 1'b0;
        free_found = 1'b0;
        hit_way = '0;
        free_way = '0;
        victim_way = '0;
        victim_age = '0;
        if (32'(in_lane) < LANES) begin
            for (integer way = 0; way < WAYS; way = way + 1) begin
                if (
                    !lookup_hit
                    && cache_valid_q[in_lane][way]
                    && cache_gate_q[in_lane][way] == in_gate
                ) begin
                    lookup_hit = 1'b1;
                    hit_way = WAY_W'(way);
                end
                if (!free_found && !cache_valid_q[in_lane][way]) begin
                    free_found = 1'b1;
                    free_way = WAY_W'(way);
                end
                if (
                    !NO_REPLACE
                    &&
                    cache_valid_q[in_lane][way]
                    && cache_age_q[in_lane][way] >= victim_age
                ) begin
                    victim_way = WAY_W'(way);
                    victim_age = cache_age_q[in_lane][way];
                end
            end
        end
        target_way = lookup_hit
            ? hit_way
            : (free_found ? free_way : victim_way);
        target_old_age = lookup_hit && !NO_REPLACE
            ? cache_age_q[in_lane][hit_way]
            : '0;

    end

    generate
        for (genvar out = 0; out < OUT_DIM; out = out + 1) begin : g_product
            assign isolated_weight[out] = multiply_enable
                ? weight_q[in_lane][out]
                : '0;
            qfit_narrow_gate_weight_mul #(
                .GATE_W(GATE_W),
                .W_W(W_W),
                .PRODUCT_W(PRODUCT_ELEM_W)
            ) u_product_mul (
                .enable(multiply_enable),
                .gate(in_gate),
                .weight(isolated_weight[out]),
                .product(narrow_product[out])
            );
            assign computed_product[
                out*PRODUCT_ELEM_W +: PRODUCT_ELEM_W
            ] = narrow_product[out];
        end
    endgenerate

    assign weight_ready =
        !epoch_active_q
        && !out_valid_q;
    assign weight_fire = weight_valid && weight_ready;
    assign weight_contract_valid =
        32'(weight_lane) < LANES
        && 32'(weight_out) < OUT_DIM
        && weight_lane == weight_expected_lane_q
        && weight_out == weight_expected_out_q
        && weight_last
            == (
                weight_expected_lane_q == LANE_W'(LANES - 1)
                && weight_expected_out_q == OUT_W'(OUT_DIM - 1)
            );
    assign epoch_start_ready =
        !epoch_active_q
        && weights_loaded_q
        && !out_valid_q
        && !weight_valid;
    assign epoch_start_fire = epoch_start_valid && epoch_start_ready;
    assign epoch_close_ready =
        epoch_active_q && !closing_q && !in_valid;
    assign epoch_close_fire = epoch_close_valid && epoch_close_ready;
    assign epoch_active = epoch_active_q;

    assign in_ready =
        epoch_active_q
        && !closing_q
        && (!out_valid_q || out_ready);
    assign in_fire = in_valid && in_ready;
    assign out_fire = out_valid_q && out_ready;
    assign multiply_enable =
        in_fire && input_contract_valid && !lookup_hit;
    assign cache_insert = !lookup_hit && (free_found || !NO_REPLACE);

    assign out_valid = out_valid_q;
    assign out_lane = out_lane_q;
    assign out_gate = out_gate_q;
    assign out_source_plane = out_plane_q;
    assign out_source_y = out_y_q;
    assign out_source_x = out_x_q;
    assign out_destination_mask = out_mask_q;
    assign out_window_last = out_window_last_q;
    assign out_product = expand_product(
        out_hit_q
            ? product_bank_read_data[out_hit_way_q]
            : out_product_q
    );

    assign protocol_error = protocol_error_q;
    assign perf_accepted_terms = perf_accepted_q;
    assign perf_cache_hits = perf_hits_q;
    assign perf_cache_misses = perf_misses_q;
    assign perf_tag_compares = perf_compares_q;
    assign perf_lru_writes = perf_lru_q;
    assign perf_product_reads = perf_product_reads_q;
    assign perf_product_writes = perf_product_writes_q;
    assign perf_product_starts = perf_product_starts_q;
    assign perf_weight_reads = perf_weight_reads_q;
    assign perf_output_stalls = perf_output_stalls_q;

    generate
        for (genvar way = 0; way < WAYS; way = way + 1) begin : g_product_bank
            assign product_bank_access[way] =
                in_fire
                && input_contract_valid
                && (
                    (lookup_hit && hit_way == WAY_W'(way))
                    || (cache_insert && target_way == WAY_W'(way))
                );
            assign product_bank_write[way] =
                product_bank_access[way] && !lookup_hit;

            qfit_sync_1rw_bank #(
                .DATA_W(CACHE_PRODUCT_W),
                .DEPTH(LANES)
            ) u_product_bank (
                .clk_core(clk_core),
                .rst_core(rst_core),
                .mem_en(product_bank_access[way]),
                .mem_write(product_bank_write[way]),
                .mem_addr(in_lane),
                .mem_write_data(computed_product),
                .mem_read_valid(product_bank_read_valid[way]),
                .mem_read_data(product_bank_read_data[way])
            );
        end
    endgenerate

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            weights_loaded_q <= 1'b0;
            weight_expected_lane_q <= '0;
            weight_expected_out_q <= '0;
            epoch_active_q <= 1'b0;
            closing_q <= 1'b0;
            epoch_done <= 1'b0;
            protocol_error_q <= 1'b0;
            out_valid_q <= 1'b0;
            out_hit_q <= 1'b0;
            out_hit_way_q <= '0;
            out_lane_q <= '0;
            out_gate_q <= '0;
            out_plane_q <= '0;
            out_y_q <= '0;
            out_x_q <= '0;
            out_mask_q <= '0;
            out_window_last_q <= 1'b0;
            out_product_q <= '0;
            perf_accepted_q <= '0;
            perf_hits_q <= '0;
            perf_misses_q <= '0;
            perf_compares_q <= '0;
            perf_lru_q <= '0;
            perf_product_reads_q <= '0;
            perf_product_writes_q <= '0;
            perf_product_starts_q <= '0;
            perf_weight_reads_q <= '0;
            perf_output_stalls_q <= '0;
            for (integer lane = 0; lane < LANES; lane = lane + 1) begin
                for (integer out = 0; out < OUT_DIM; out = out + 1)
                    weight_q[lane][out] <= '0;
                for (integer way = 0; way < WAYS; way = way + 1) begin
                    cache_valid_q[lane][way] <= 1'b0;
                    cache_gate_q[lane][way] <= '0;
                    if (!NO_REPLACE)
                        cache_age_q[lane][way] <= '0;
                end
            end
        end else begin
            epoch_done <= 1'b0;

            if (weight_fire) begin
                if (!weight_contract_valid) begin
                    protocol_error_q <= 1'b1;
                    weights_loaded_q <= 1'b0;
                    weight_expected_lane_q <= '0;
                    weight_expected_out_q <= '0;
                end else begin
                    weight_q[weight_lane][weight_out] <= weight_data;
                    if (weights_loaded_q)
                        weights_loaded_q <= 1'b0;
                    if (weight_last) begin
                        weights_loaded_q <= 1'b1;
                        weight_expected_lane_q <= '0;
                        weight_expected_out_q <= '0;
                    end else if (
                        weight_expected_out_q == OUT_W'(OUT_DIM - 1)
                    ) begin
                        weight_expected_lane_q <=
                            weight_expected_lane_q + 1'b1;
                        weight_expected_out_q <= '0;
                    end else begin
                        weight_expected_out_q <=
                            weight_expected_out_q + 1'b1;
                    end
                end
            end

            if (epoch_start_fire) begin
                epoch_active_q <= 1'b1;
                closing_q <= 1'b0;
                protocol_error_q <= 1'b0;
                perf_accepted_q <= '0;
                perf_hits_q <= '0;
                perf_misses_q <= '0;
                perf_compares_q <= '0;
                perf_lru_q <= '0;
                perf_product_reads_q <= '0;
                perf_product_writes_q <= '0;
                perf_product_starts_q <= '0;
                perf_weight_reads_q <= '0;
                perf_output_stalls_q <= '0;
                for (integer lane = 0; lane < LANES; lane = lane + 1)
                    for (integer way = 0; way < WAYS; way = way + 1) begin
                        cache_valid_q[lane][way] <= 1'b0;
                        if (!NO_REPLACE)
                            cache_age_q[lane][way] <= '0;
                    end
            end

            if (out_valid_q && !out_ready)
                perf_output_stalls_q <= perf_output_stalls_q + 1'b1;
            if (out_fire)
                out_valid_q <= 1'b0;

            if (in_fire && !input_contract_valid) begin
                protocol_error_q <= 1'b1;
            end else if (in_fire) begin
                logic [31:0] lru_writes;
                lru_writes = '0;
                out_valid_q <= 1'b1;
                out_lane_q <= in_lane;
                out_gate_q <= in_gate;
                out_plane_q <= in_source_plane;
                out_y_q <= in_source_y;
                out_x_q <= in_source_x;
                out_mask_q <= in_destination_mask;
                out_window_last_q <= in_window_last;
                out_hit_q <= lookup_hit;
                out_hit_way_q <= hit_way;
                perf_accepted_q <= perf_accepted_q + 1'b1;
                perf_compares_q <= perf_compares_q + 32'(WAYS);

                if (lookup_hit) begin
                    perf_hits_q <= perf_hits_q + 1'b1;
                    perf_product_reads_q <=
                        perf_product_reads_q + 1'b1;
                end else begin
                    out_product_q <= computed_product;
                    if (cache_insert) begin
                        cache_valid_q[in_lane][target_way] <= 1'b1;
                        cache_gate_q[in_lane][target_way] <= in_gate;
                    end
                    perf_misses_q <= perf_misses_q + 1'b1;
                    if (cache_insert)
                        perf_product_writes_q <=
                            perf_product_writes_q + 1'b1;
                    perf_product_starts_q <=
                        perf_product_starts_q + 1'b1;
                    perf_weight_reads_q <=
                        perf_weight_reads_q + 32'(OUT_DIM);
                end

                if (!NO_REPLACE) begin
                    for (integer way = 0; way < WAYS; way = way + 1) begin
                        if (WAY_W'(way) == target_way) begin
                            if (
                                cache_age_q[in_lane][way] != '0
                                || !cache_valid_q[in_lane][way]
                            )
                                lru_writes = lru_writes + 1'b1;
                            cache_age_q[in_lane][way] <= '0;
                        end else if (
                            cache_valid_q[in_lane][way]
                            && (
                                (!lookup_hit
                                    && cache_age_q[in_lane][way]
                                        < AGE_W'(WAYS - 1))
                                || (lookup_hit
                                    && cache_age_q[in_lane][way]
                                        < target_old_age)
                            )
                        ) begin
                            cache_age_q[in_lane][way] <=
                                cache_age_q[in_lane][way] + 1'b1;
                            lru_writes = lru_writes + 1'b1;
                        end
                    end
                end
                perf_lru_q <= perf_lru_q + lru_writes;
            end

            if (epoch_close_fire)
                closing_q <= 1'b1;

            if (
                epoch_active_q
                && closing_q
                && !out_valid_q
            ) begin
                epoch_active_q <= 1'b0;
                closing_q <= 1'b0;
                epoch_done <= 1'b1;
            end
        end
    end

    initial begin
        if (
            LANES < 2
            || WAYS < 2
            || OUT_DIM < 1
            || ACC_W < PRODUCT_ELEM_W
        )
            $fatal(1, "product cache参数非法");
    end
endmodule

`default_nettype wire
