`default_nettype none

module h67_real_weight_projection2_monitor #(
    parameter int MAX_ROWS = 256,
    parameter int HEAD_DIM = 32
) (
    input logic clk,
    input logic rst_core,
    input integer row_tag,
    input integer stage_tag,
    input integer block_tag,
    input integer head_tag,
    input logic fixed_start,
    input logic fixed_done,
    input logic fixed_out_valid,
    input logic fixed_out_last,
    input logic [31:0] fixed_out_k,
    input logic [8:0] fixed_out_gate,
    input logic rqtb_start,
    input logic rqtb_done,
    input logic rqtb_out_valid,
    input logic rqtb_out_last,
    input logic [31:0] rqtb_out_k,
    input logic [8:0] rqtb_out_gate,
    input logic common_out_ready
);
    logic signed [7:0] weight0_mem [0:MAX_ROWS*HEAD_DIM-1];
    logic signed [7:0] weight1_mem [0:MAX_ROWS*HEAD_DIM-1];
    integer signed expected0_mem [0:MAX_ROWS-1];
    integer signed expected1_mem [0:MAX_ROWS-1];
    integer expected_stage [0:MAX_ROWS-1];
    integer expected_block [0:MAX_ROWS-1];
    integer expected_head [0:MAX_ROWS-1];
    logic [HEAD_DIM*8-1:0] weight0_flat;
    logic [HEAD_DIM*8-1:0] weight1_flat;

    logic fixed_result_valid;
    logic rqtb_result_valid;
    logic signed [31:0] fixed_result0;
    logic signed [31:0] fixed_result1;
    logic signed [31:0] rqtb_result0;
    logic signed [31:0] rqtb_result1;
    logic fixed_result_seen_q;
    logic rqtb_result_seen_q;
    logic signed [31:0] fixed_result0_q;
    logic signed [31:0] fixed_result1_q;
    logic signed [31:0] rqtb_result0_q;
    logic signed [31:0] rqtb_result1_q;
    integer active_row_q;
    integer fixed_result_row_q;
    integer rqtb_result_row_q;
    integer last_compared_row_q;
    integer fd;
    integer file_rows;
    integer file_channels;
    integer scan_count;
    integer row;
    integer lane;
    integer flat_lane;
    integer row_id;
    integer checks;
    integer signed weight0_read;
    integer signed weight1_read;
    string vector_path;
    wire fixed_result_available = fixed_result_seen_q || fixed_result_valid;
    wire rqtb_result_available = rqtb_result_seen_q || rqtb_result_valid;
    wire signed [31:0] fixed_compare0 =
        fixed_result_valid ? fixed_result0 : fixed_result0_q;
    wire signed [31:0] fixed_compare1 =
        fixed_result_valid ? fixed_result1 : fixed_result1_q;
    wire signed [31:0] rqtb_compare0 =
        rqtb_result_valid ? rqtb_result0 : rqtb_result0_q;
    wire signed [31:0] rqtb_compare1 =
        rqtb_result_valid ? rqtb_result1 : rqtb_result1_q;
    wire signed [31:0] fixed_compare_row =
        fixed_result_valid ? active_row_q : fixed_result_row_q;
    wire signed [31:0] rqtb_compare_row =
        rqtb_result_valid ? active_row_q : rqtb_result_row_q;

    always_comb begin
        weight0_flat = '0;
        weight1_flat = '0;
        if (active_row_q >= 0 && active_row_q < file_rows) begin
            for (flat_lane = 0; flat_lane < HEAD_DIM;
                 flat_lane = flat_lane + 1) begin
                weight0_flat[flat_lane*8 +: 8] =
                    weight0_mem[active_row_q*HEAD_DIM + flat_lane];
                weight1_flat[flat_lane*8 +: 8] =
                    weight1_mem[active_row_q*HEAD_DIM + flat_lane];
            end
        end
    end

    h67_gated_k_projection2_acc u_fixed_projection (
        .clk, .rst_core, .row_start(fixed_start), .row_done(fixed_done),
        .in_fire(fixed_out_valid && common_out_ready),
        .in_last(fixed_out_last), .in_k_bits(fixed_out_k),
        .in_gate_q17(fixed_out_gate), .weight0_flat, .weight1_flat,
        .result_valid(fixed_result_valid),
        .result0_acc32(fixed_result0), .result1_acc32(fixed_result1)
    );

    h67_gated_k_projection2_acc u_rqtb_projection (
        .clk, .rst_core, .row_start(rqtb_start), .row_done(rqtb_done),
        .in_fire(rqtb_out_valid && common_out_ready),
        .in_last(rqtb_out_last), .in_k_bits(rqtb_out_k),
        .in_gate_q17(rqtb_out_gate), .weight0_flat, .weight1_flat,
        .result_valid(rqtb_result_valid),
        .result0_acc32(rqtb_result0), .result1_acc32(rqtb_result1)
    );

    initial begin
        checks = 0;
        file_rows = 0;
        file_channels = 0;
        if (!$value$plusargs("REALW=%s", vector_path))
            $fatal(1, "missing +REALW=<path>");
        fd = $fopen(vector_path, "r");
        if (fd == 0)
            $fatal(1, "cannot open real-weight vectors: %s", vector_path);
        scan_count = $fscanf(fd, "%d %d", file_rows, file_channels);
        if (scan_count != 2 || file_rows <= 0 || file_rows > MAX_ROWS
            || file_channels != 2)
            $fatal(1, "invalid real-weight header rows=%0d channels=%0d",
                file_rows, file_channels);
        for (row = 0; row < file_rows; row = row + 1) begin
            scan_count = $fscanf(fd, "%d %d %d %d %d %d",
                row_id, expected_stage[row], expected_block[row],
                expected_head[row], expected0_mem[row], expected1_mem[row]);
            if (scan_count != 6 || row_id != row)
                $fatal(1, "invalid real-weight row header row=%0d", row);
            for (lane = 0; lane < HEAD_DIM; lane = lane + 1) begin
                scan_count = $fscanf(fd, "%d %d",
                    weight0_read, weight1_read);
                if (scan_count != 2)
                    $fatal(1, "invalid real-weight payload row=%0d lane=%0d",
                        row, lane);
                weight0_mem[row*HEAD_DIM + lane] = weight0_read[7:0];
                weight1_mem[row*HEAD_DIM + lane] = weight1_read[7:0];
            end
        end
        $fclose(fd);
    end

    always @(posedge clk) begin
        if (rst_core) begin
            fixed_result_seen_q <= 1'b0;
            rqtb_result_seen_q <= 1'b0;
            fixed_result0_q <= '0;
            fixed_result1_q <= '0;
            rqtb_result0_q <= '0;
            rqtb_result1_q <= '0;
            active_row_q <= 0;
            fixed_result_row_q <= -1;
            rqtb_result_row_q <= -1;
            last_compared_row_q <= -1;
        end else begin
            if (fixed_start || rqtb_start) begin
                if (row_tag < 0 || row_tag >= file_rows)
                    $fatal(1, "real-weight row_tag out of range: %0d", row_tag);
                if (stage_tag != expected_stage[row_tag]
                    || block_tag != expected_block[row_tag]
                    || head_tag != expected_head[row_tag])
                    $fatal(1, "real-weight identity mismatch row=%0d", row_tag);
                active_row_q <= row_tag;
            end
            if (fixed_result_available && rqtb_result_available) begin
                if (fixed_compare_row != rqtb_compare_row
                    || fixed_compare_row != last_compared_row_q + 1)
                $fatal(1, "real-weight result row ordering mismatch fixed=%0d rqtb=%0d last=%0d",
                    fixed_compare_row, rqtb_compare_row, last_compared_row_q);
                if ($signed(fixed_compare0) != expected0_mem[fixed_compare_row]
                    || $signed(fixed_compare1) != expected1_mem[fixed_compare_row]
                    || $signed(rqtb_compare0) != expected0_mem[fixed_compare_row]
                    || $signed(rqtb_compare1) != expected1_mem[fixed_compare_row])
                $fatal(1, "real-weight Acc32 mismatch row=%0d expected=%0d,%0d fixed=%0d,%0d rqtb=%0d,%0d",
                    fixed_compare_row, expected0_mem[fixed_compare_row],
                    expected1_mem[fixed_compare_row],
                    $signed(fixed_compare0), $signed(fixed_compare1),
                    $signed(rqtb_compare0), $signed(rqtb_compare1));
                checks = checks + 1;
                last_compared_row_q <= fixed_compare_row;
                fixed_result_seen_q <= 1'b0;
                rqtb_result_seen_q <= 1'b0;
                $display("REALW_ROW row=%0d stage=%0d block=%0d head=%0d expected0=%0d expected1=%0d fixed0=%0d fixed1=%0d rqtb0=%0d rqtb1=%0d",
                    fixed_compare_row, expected_stage[fixed_compare_row],
                    expected_block[fixed_compare_row], expected_head[fixed_compare_row],
                    expected0_mem[fixed_compare_row], expected1_mem[fixed_compare_row],
                    $signed(fixed_compare0), $signed(fixed_compare1),
                    $signed(rqtb_compare0), $signed(rqtb_compare1));
            end else begin
                if (fixed_result_valid) begin
                    fixed_result_seen_q <= 1'b1;
                    fixed_result_row_q <= active_row_q;
                    fixed_result0_q <= fixed_result0;
                    fixed_result1_q <= fixed_result1;
                end
                if (rqtb_result_valid) begin
                    rqtb_result_seen_q <= 1'b1;
                    rqtb_result_row_q <= active_row_q;
                    rqtb_result0_q <= rqtb_result0;
                    rqtb_result1_q <= rqtb_result1;
                end
            end
        end
    end
endmodule

`default_nettype wire
