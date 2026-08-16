`timescale 1ns/1ps
`default_nettype none

module tb_qfit_dual_color_index_equivalence;
    localparam int HEIGHT = 15;
    localparam int WIDTH = 15;
    localparam int TIME_PLANES = 2;
    localparam int SOURCE_ID_W = $clog2(HEIGHT * WIDTH * TIME_PLANES);

    logic clk_core;
    logic rst_core;
    logic build_start;
    logic build_seal;
    logic in_valid;
    logic in_plane;
    logic [3:0] in_destination_y;
    logic [3:0] in_destination_x;
    logic [4:0] in_active_candidate_mask;
    logic out_ready;
    logic old_in_ready;
    logic old_build_active;
    logic old_build_done;
    logic old_out_valid;
    logic [SOURCE_ID_W-1:0] old_source_id;
    logic old_source_plane;
    logic [3:0] old_source_y;
    logic [3:0] old_source_x;
    logic old_out_last;
    logic old_error;
    logic [31:0] old_input;
    logic [31:0] old_unique;
    logic [31:0] old_duplicate;
    logic [31:0] old_conflict;
    logic new_in_ready;
    logic new_build_active;
    logic new_build_done;
    logic new_out_valid;
    logic [SOURCE_ID_W-1:0] new_source_id;
    logic new_source_plane;
    logic [3:0] new_source_y;
    logic [3:0] new_source_x;
    logic new_out_last;
    logic new_error;
    logic [31:0] new_input;
    logic [31:0] new_unique;
    logic [31:0] new_duplicate;
    logic [31:0] new_conflict;
    logic [31:0] new_word_probes;
    integer cycle_count;
    integer output_count;

    qfit_dual_color_active_source_index #(
        .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES)
    ) u_old (
        .clk_core(clk_core), .rst_core(rst_core),
        .build_start(build_start), .build_seal(build_seal),
        .build_active(old_build_active), .build_done(old_build_done),
        .in_valid(in_valid), .in_ready(old_in_ready),
        .in_plane(in_plane),
        .in_destination_y(in_destination_y),
        .in_destination_x(in_destination_x),
        .in_active_candidate_mask(in_active_candidate_mask),
        .out_valid(old_out_valid), .out_ready(out_ready),
        .out_source_id(old_source_id),
        .out_source_plane(old_source_plane),
        .out_source_y(old_source_y), .out_source_x(old_source_x),
        .out_last(old_out_last), .protocol_error(old_error),
        .perf_input_candidates(old_input),
        .perf_unique_sources(old_unique),
        .perf_duplicate_sets(old_duplicate),
        .perf_bank_conflicts(old_conflict)
    );

    qfit_dual_color_word_skipper_index #(
        .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES)
    ) u_new (
        .clk_core(clk_core), .rst_core(rst_core),
        .build_start(build_start), .build_seal(build_seal),
        .build_active(new_build_active), .build_done(new_build_done),
        .in_valid(in_valid), .in_ready(new_in_ready),
        .in_plane(in_plane),
        .in_destination_y(in_destination_y),
        .in_destination_x(in_destination_x),
        .in_active_candidate_mask(in_active_candidate_mask),
        .out_valid(new_out_valid), .out_ready(out_ready),
        .out_source_id(new_source_id),
        .out_source_plane(new_source_plane),
        .out_source_y(new_source_y), .out_source_x(new_source_x),
        .out_last(new_out_last), .protocol_error(new_error),
        .perf_input_candidates(new_input),
        .perf_unique_sources(new_unique),
        .perf_duplicate_sets(new_duplicate),
        .perf_bank_conflicts(new_conflict),
        .perf_word_probes(new_word_probes)
    );

    always #5 clk_core = ~clk_core;

    task automatic send_destination(
        input integer plane,
        input integer y,
        input integer x
    );
        logic [4:0] geometry;
        begin
            geometry = 5'b00001;
            if (y > 0) geometry[1] = 1'b1;
            if (y < HEIGHT - 1) geometry[2] = 1'b1;
            if (x > 0) geometry[3] = 1'b1;
            if (x < WIDTH - 1) geometry[4] = 1'b1;
            @(negedge clk_core);
            while (!old_in_ready || !new_in_ready) @(negedge clk_core);
            in_plane = plane[0];
            in_destination_y = y[3:0];
            in_destination_x = x[3:0];
            in_active_candidate_mask = geometry & $urandom;
            in_valid = 1'b1;
            @(negedge clk_core);
            in_valid = 1'b0;
        end
    endtask

    always @(negedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            out_ready <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            out_ready <= (cycle_count % 7) != 3;
        end
    end

    always @(posedge clk_core) begin
        if (!rst_core) begin
            if (
                old_in_ready != new_in_ready
                || old_build_active != new_build_active
                || old_build_done != new_build_done
                || old_out_valid != new_out_valid
            )
                $fatal(1, "control equivalence mismatch");
            if (old_out_valid && (
                old_source_id != new_source_id
                || old_source_plane != new_source_plane
                || old_source_y != new_source_y
                || old_source_x != new_source_x
                || old_out_last != new_out_last
            ))
                $fatal(1, "payload equivalence mismatch old=%0d new=%0d",
                    old_source_id, new_source_id);
            if (old_out_valid && out_ready)
                output_count = output_count + 1;
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        build_start = 1'b0;
        build_seal = 1'b0;
        in_valid = 1'b0;
        in_plane = 1'b0;
        in_destination_y = '0;
        in_destination_x = '0;
        in_active_candidate_mask = '0;
        out_ready = 1'b0;
        cycle_count = 0;
        output_count = 0;
        repeat (3) @(negedge clk_core);
        rst_core = 1'b0;
        @(negedge clk_core);
        build_start = 1'b1;
        @(negedge clk_core);
        build_start = 1'b0;
        for (integer plane = 0; plane < TIME_PLANES; plane = plane + 1)
            for (integer y = 0; y < HEIGHT; y = y + 1)
                for (integer x = 0; x < WIDTH; x = x + 1)
                    send_destination(plane, y, x);
        @(negedge clk_core);
        build_seal = 1'b1;
        @(negedge clk_core);
        build_seal = 1'b0;
        wait (old_build_done && new_build_done);
        repeat (2) @(negedge clk_core);
        if (
            old_error || new_error
            || old_input != new_input
            || old_unique != new_unique
            || old_duplicate != new_duplicate
            || old_conflict != new_conflict
            || output_count != old_unique
        )
            $fatal(1, "final equivalence/counter mismatch");
        $display(
            "PASS T450 full-depth/word-skipper equivalence unique=%0d probes=%0d",
            old_unique,
            new_word_probes
        );
        $finish;
    end
endmodule

`default_nettype wire
