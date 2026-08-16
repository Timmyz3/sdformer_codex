`timescale 1ns/1ps
`default_nettype none

// Passive G0 calibration monitor for the existing online Direct path.
// The log is intentionally line-oriented so Icarus and Verilator produce the
// same evidence without DPI, UVM, or changes to the monitored RTL.
module qfit_local5_erep_direct_monitor_v4 #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int OUT_DIM = 2,
    parameter int GATE_W = 9,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int SOURCE_ID_W =
        (HEIGHT * WIDTH * TIME_PLANES <= 1)
        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES),
    parameter int LANE_W = (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM),
    parameter int OUT_W = (OUT_DIM <= 1) ? 1 : $clog2(OUT_DIM),
    parameter int BANK_DEPTH = TIME_PLANES * HEIGHT * ((WIDTH + 4) / 5),
    parameter int BANK_ADDR_W =
        (BANK_DEPTH <= 1) ? 1 : $clog2(BANK_DEPTH),
    parameter int VEC_W = OUT_DIM * ACC_W
) (
    input logic clk_core,
    input logic rst_core,

    input logic weight_valid,
    input logic weight_ready,
    input logic [LANE_W-1:0] weight_lane,
    input logic [OUT_W-1:0] weight_out,
    input logic signed [W_W-1:0] weight_data,
    input logic weight_last,

    input logic projection_start,
    input logic projection_close,
    input logic projection_close_ready,
    input logic projection_busy,
    input logic projection_done,
    input logic relation_start,
    input logic relation_seal,
    input logic relation_active,
    input logic relation_done,

    input logic relation_valid,
    input logic relation_ready,
    input logic [PLANE_W-1:0] relation_plane,
    input logic [Y_W-1:0] relation_destination_y,
    input logic [X_W-1:0] relation_destination_x,
    input logic [4:0] relation_candidate_valid,
    input logic [4:0] relation_active_candidate_mask,
    input logic [HEAD_DIM-1:0] relation_k_self,
    input logic [5*GATE_W-1:0] relation_direction_gates,

    input logic relation_read_fire,
    input logic [SOURCE_ID_W-1:0] relation_read_source_id,
    input logic [PLANE_W-1:0] relation_read_plane,
    input logic [Y_W-1:0] relation_read_y,
    input logic [X_W-1:0] relation_read_x,
    input logic relation_read_last,

    input logic descriptor_valid,
    input logic descriptor_ready,
    input logic [SOURCE_ID_W-1:0] descriptor_source_id,
    input logic [PLANE_W-1:0] descriptor_plane,
    input logic [Y_W-1:0] descriptor_y,
    input logic [X_W-1:0] descriptor_x,
    input logic [HEAD_DIM-1:0] descriptor_k,
    input logic [5*GATE_W-1:0] descriptor_gates,
    input logic [4:0] descriptor_mask,
    input logic descriptor_last,

    input logic fifo_enqueue,
    input logic fifo_dequeue,
    input logic [1:0] fifo_count,
    input logic [SOURCE_ID_W-1:0] fifo_head_source_id,
    input logic [PLANE_W-1:0] fifo_head_plane,
    input logic [Y_W-1:0] fifo_head_y,
    input logic [X_W-1:0] fifo_head_x,

    input logic term_valid,
    input logic term_ready,
    input logic [SOURCE_ID_W-1:0] term_source_id,
    input logic [PLANE_W-1:0] term_source_plane,
    input logic [Y_W-1:0] term_source_y,
    input logic [X_W-1:0] term_source_x,
    input logic [LANE_W-1:0] term_lane,
    input logic [GATE_W-1:0] term_gate,
    input logic [4:0] term_destination_mask,
    input logic term_last,
    input logic term_source_last,

    input logic [4:0] bank_update_enable,
    input logic [4:0] bank_update_ready,
    input logic [5*BANK_ADDR_W-1:0] bank_update_addr,
    input logic [VEC_W-1:0] bank_update_delta,
    input logic [4:0] bank_command_valid,
    input logic [4:0] bank_command_write,
    input logic [5*BANK_ADDR_W-1:0] bank_command_addr,
    input logic [5*VEC_W-1:0] bank_command_write_data,

    input logic read_valid,
    input logic read_ready,
    input logic [PLANE_W-1:0] read_plane,
    input logic [Y_W-1:0] read_y,
    input logic [X_W-1:0] read_x,
    input logic [OUT_W-1:0] read_out,
    input logic read_data_valid,
    input logic signed [ACC_W-1:0] read_data,
    input logic protocol_error
);
    localparam int TOTAL_SOURCES = HEIGHT * WIDTH * TIME_PLANES;
    localparam int PH_PREPARE = 0;
    localparam int PH_FILL = 1;
    localparam int PH_EXECUTE = 2;
    localparam int PH_DRAIN = 3;
    localparam int PH_READOUT = 4;

    integer trace_enable;
    integer active_q;
    integer cycle_q;
    integer window_q;
    integer phase_q;
    integer relation_count_q;
    integer done_seen_q;
    integer pending_read_source_q;
    integer pending_read_out_q;
    logic descriptor_stall_q;
    logic [SOURCE_ID_W-1:0] stalled_descriptor_source_q;
    logic [PLANE_W-1:0] stalled_descriptor_plane_q;
    logic [Y_W-1:0] stalled_descriptor_y_q;
    logic [X_W-1:0] stalled_descriptor_x_q;
    logic [HEAD_DIM-1:0] stalled_descriptor_k_q;
    logic [5*GATE_W-1:0] stalled_descriptor_gates_q;
    logic [4:0] stalled_descriptor_mask_q;
    logic stalled_descriptor_last_q;
    logic term_stall_q;
    logic [SOURCE_ID_W-1:0] stalled_term_source_q;
    logic [PLANE_W-1:0] stalled_term_plane_q;
    logic [Y_W-1:0] stalled_term_y_q;
    logic [X_W-1:0] stalled_term_x_q;
    logic [LANE_W-1:0] stalled_term_lane_q;
    logic [GATE_W-1:0] stalled_term_gate_q;
    logic [4:0] stalled_term_mask_q;
    logic stalled_term_last_q;
    logic stalled_term_source_last_q;

    function automatic integer flatten_source(
        input logic [PLANE_W-1:0] plane,
        input logic [Y_W-1:0] y,
        input logic [X_W-1:0] x
    );
        flatten_source = integer'(plane) * HEIGHT * WIDTH
                       + integer'(y) * WIDTH + integer'(x);
    endfunction

    initial begin
        trace_enable = $test$plusargs("EREP_TRACE_V4");
        window_q = -1;
    end

    always @(posedge clk_core) begin : monitor_direct
        integer event_cycle;
        integer event_window;
        integer bank;
        integer occupancy_post;
        integer relation_source;

        if (rst_core) begin
            active_q <= 0;
            cycle_q <= 0;
            phase_q <= PH_PREPARE;
            relation_count_q <= 0;
            done_seen_q <= 0;
            pending_read_source_q <= -1;
            pending_read_out_q <= -1;
            descriptor_stall_q <= 1'b0;
            term_stall_q <= 1'b0;
        end else begin
            event_cycle = projection_start ? 0
                        : (active_q ? cycle_q + 1 : cycle_q);
            event_window = projection_start ? window_q + 1 : window_q;
            relation_source = flatten_source(
                relation_plane,
                relation_destination_y,
                relation_destination_x
            );

            if (projection_start) begin
                active_q <= 1;
                cycle_q <= 0;
                window_q <= window_q + 1;
                phase_q <= PH_PREPARE;
                relation_count_q <= 0;
                done_seen_q <= 0;
                pending_read_source_q <= -1;
                pending_read_out_q <= -1;
                if (trace_enable) begin
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=direct_online event=phase_boundary kind=PREPARE_BEGIN cycle=0 window=%0d phase=%0d time=%0t scope=%m", event_window, PH_PREPARE, $time);
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=direct_online event=context_prepare resource=context_prepare_1rw kind=context_prepare cycle=0 window=%0d phase=%0d valid=1 ready=1 fire=1 identity=window_%0d time=%0t scope=%m", event_window, PH_PREPARE, event_window, $time);
                end
            end else if (active_q) begin
                cycle_q <= cycle_q + 1;
                if (phase_q == PH_PREPARE) begin
                    phase_q <= PH_FILL;
                    if (trace_enable)
                        $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=direct_online event=phase_boundary kind=FILL_BEGIN cycle=%0d window=%0d phase=%0d time=%0t scope=%m", event_cycle + 1, event_window, PH_FILL, $time);
                end
            end

            if ((active_q || projection_start) && trace_enable) begin
                $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=direct_online event=cycle_snapshot resource=pipeline kind=state cycle=%0d window=%0d phase=%0d valid=%0h ready=%0h fire=%0h fifo_occupancy=%0d projection_busy=%0d projection_done=%0d relation_active=%0d relation_done=%0d protocol_error=%0d time=%0t scope=%m",
                    event_cycle, event_window, phase_q,
                    {read_valid, projection_close, term_valid, descriptor_valid, relation_valid},
                    {read_ready, projection_close_ready, term_ready, descriptor_ready, relation_ready},
                    {read_valid && read_ready,
                     projection_close && projection_close_ready,
                     term_valid && term_ready,
                     descriptor_valid && descriptor_ready,
                     relation_valid && relation_ready},
                    fifo_count, projection_busy, projection_done,
                    relation_active, relation_done, protocol_error, $time);
            end

            if (weight_valid && weight_ready && trace_enable)
                $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=direct_online event=weight_accept resource=weight_service kind=weight cycle=%0d window=%0d phase=%0d valid=1 ready=1 fire=1 lane=%0d out=%0d data=%0d last=%0d time=%0t scope=%m", event_cycle, event_window, phase_q, weight_lane, weight_out, weight_data, weight_last, $time);

            if (relation_valid && relation_ready) begin
                relation_count_q <= relation_count_q + 1;
                if (trace_enable)
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=direct_online event=relation_accept resource=relation_workspace_1rw kind=relation_write cycle=%0d window=%0d phase=%0d valid=1 ready=1 fire=1 source_id=%0d plane=%0d y=%0d x=%0d candidate_valid=%0h active_mask=%0h k=%0h gates=%0h time=%0t scope=%m", event_cycle, event_window, phase_q, relation_source, relation_plane, relation_destination_y, relation_destination_x, relation_candidate_valid, relation_active_candidate_mask, relation_k_self, relation_direction_gates, $time);
            end

            if (relation_read_fire && trace_enable)
                $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=direct_online event=relation_read resource=relation_workspace_1rw kind=relation_read cycle=%0d window=%0d phase=%0d valid=1 ready=1 fire=1 source_id=%0d plane=%0d y=%0d x=%0d last=%0d time=%0t scope=%m", event_cycle, event_window, phase_q, relation_read_source_id, relation_read_plane, relation_read_y, relation_read_x, relation_read_last, $time);

            occupancy_post = integer'(fifo_count)
                           + (fifo_enqueue ? 1 : 0)
                           - (fifo_dequeue ? 1 : 0);
            if (fifo_enqueue && trace_enable)
                $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=direct_online event=fifo_enqueue resource=fifo2_enq kind=fifo_enqueue cycle=%0d window=%0d phase=%0d valid=%0d ready=%0d fire=1 source_id=%0d plane=%0d y=%0d x=%0d k=%0h gates=%0h mask=%0h last=%0d occupancy_pre=%0d occupancy_post=%0d time=%0t scope=%m", event_cycle, event_window, phase_q, descriptor_valid, descriptor_ready, descriptor_source_id, descriptor_plane, descriptor_y, descriptor_x, descriptor_k, descriptor_gates, descriptor_mask, descriptor_last, fifo_count, occupancy_post, $time);
            if (fifo_dequeue && trace_enable)
                $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=direct_online event=fifo_dequeue resource=fifo2_deq kind=fifo_dequeue cycle=%0d window=%0d phase=%0d valid=1 ready=1 fire=1 source_id=%0d plane=%0d y=%0d x=%0d occupancy_pre=%0d occupancy_post=%0d time=%0t scope=%m", event_cycle, event_window, phase_q, fifo_head_source_id, fifo_head_plane, fifo_head_y, fifo_head_x, fifo_count, occupancy_post, $time);

            if (term_valid && term_ready) begin
                if (trace_enable)
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=direct_online event=term_accept resource=execute_lane kind=term cycle=%0d window=%0d phase=%0d valid=1 ready=1 fire=1 source_id=%0d plane=%0d y=%0d x=%0d lane=%0d gate=%0d destination_mask=%0h last=%0d source_last=%0d delta=%0h time=%0t scope=%m", event_cycle, event_window, phase_q, term_source_id, term_source_plane, term_source_y, term_source_x, term_lane, term_gate, term_destination_mask, term_last, term_source_last, bank_update_delta, $time);
                for (bank = 0; bank < 5; bank = bank + 1) begin
                    if (bank_update_enable[bank] && trace_enable)
                        $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=direct_online event=acc_update_accept resource=acc_bank_%0d_1rw kind=acc_write cycle=%0d window=%0d phase=%0d valid=1 ready=%0d fire=1 bank=%0d source_id=%0d lane=%0d gate=%0d address=%0d delta=%0h time=%0t scope=%m", bank, event_cycle, event_window, phase_q, bank_update_ready[bank], bank, term_source_id, term_lane, term_gate, bank_update_addr[bank*BANK_ADDR_W +: BANK_ADDR_W], bank_update_delta, $time);
                end
            end

            for (bank = 0; bank < 5; bank = bank + 1) begin
                if (bank_command_valid[bank] && trace_enable)
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=direct_online event=acc_physical_command resource=acc_bank_%0d_1rw kind=%0s cycle=%0d window=%0d phase=%0d valid=1 ready=1 fire=1 bank=%0d address=%0d data=%0h time=%0t scope=%m", bank, bank_command_write[bank] ? "physical_write" : "physical_read", event_cycle, event_window, phase_q, bank, bank_command_addr[bank*BANK_ADDR_W +: BANK_ADDR_W], bank_command_write_data[bank*VEC_W +: VEC_W], $time);
            end

            if (read_valid && read_ready) begin
                pending_read_source_q <= flatten_source(read_plane, read_y, read_x);
                pending_read_out_q <= integer'(read_out);
                if (trace_enable)
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=direct_online event=drain_read_accept resource=drain_read_1rw kind=drain_read cycle=%0d window=%0d phase=%0d valid=1 ready=1 fire=1 source_id=%0d plane=%0d y=%0d x=%0d out=%0d time=%0t scope=%m", event_cycle, event_window, phase_q, flatten_source(read_plane, read_y, read_x), read_plane, read_y, read_x, read_out, $time);
            end
            if (read_data_valid && trace_enable)
                $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=direct_online event=drain_read_response resource=drain_read_response kind=final cycle=%0d window=%0d phase=%0d valid=1 ready=1 fire=1 source_id=%0d out=%0d data=%0d time=%0t scope=%m", event_cycle, event_window, phase_q, pending_read_source_q, pending_read_out_q, read_data, $time);

            if (relation_seal) begin
                phase_q <= PH_EXECUTE;
                if (relation_count_q != TOTAL_SOURCES)
                    $fatal(1, "EREP v4 Direct relation count mismatch got=%0d expected=%0d", relation_count_q, TOTAL_SOURCES);
                if (trace_enable)
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=direct_online event=phase_boundary kind=EXECUTE_BEGIN cycle=%0d window=%0d phase=%0d relation_records=%0d time=%0t scope=%m", event_cycle + 1, event_window, PH_EXECUTE, relation_count_q, $time);
            end
            if (projection_close && projection_close_ready) begin
                phase_q <= PH_DRAIN;
                if (fifo_count != 0)
                    $fatal(1, "EREP v4 Direct close with non-empty FIFO2");
                if (trace_enable)
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=direct_online event=phase_boundary kind=DRAIN_BEGIN cycle=%0d window=%0d phase=%0d time=%0t scope=%m", event_cycle + 1, event_window, PH_DRAIN, $time);
            end
            if (projection_done && !done_seen_q) begin
                done_seen_q <= 1;
                phase_q <= PH_READOUT;
                if (trace_enable)
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=direct_online event=phase_boundary kind=COMPUTE_DONE cycle=%0d window=%0d phase=%0d time=%0t scope=%m", event_cycle, event_window, PH_READOUT, $time);
            end

            if (fifo_count > 2)
                $fatal(1, "EREP v4 FIFO2 occupancy exceeded two");
            if (fifo_dequeue && fifo_count == 0 && !fifo_enqueue)
                $fatal(1, "EREP v4 FIFO2 underflow");
            if (protocol_error)
                $fatal(1, "EREP v4 Direct monitor observed protocol_error");

            if (descriptor_valid && !descriptor_ready) begin
                if (descriptor_stall_q && (
                    descriptor_source_id != stalled_descriptor_source_q
                    || descriptor_plane != stalled_descriptor_plane_q
                    || descriptor_y != stalled_descriptor_y_q
                    || descriptor_x != stalled_descriptor_x_q
                    || descriptor_k != stalled_descriptor_k_q
                    || descriptor_gates != stalled_descriptor_gates_q
                    || descriptor_mask != stalled_descriptor_mask_q
                    || descriptor_last != stalled_descriptor_last_q
                ))
                    $fatal(1, "EREP v4 descriptor changed under backpressure");
                if (trace_enable)
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=direct_online event=stall_observation resource=descriptor kind=backpressure cycle=%0d window=%0d phase=%0d valid=1 ready=0 fire=0 source_id=%0d plane=%0d y=%0d x=%0d lane=0 gate=0 destination_mask=%0h last=%0d source_last=0 time=%0t scope=%m", event_cycle, event_window, phase_q, descriptor_source_id, descriptor_plane, descriptor_y, descriptor_x, descriptor_mask, descriptor_last, $time);
                descriptor_stall_q <= 1'b1;
                stalled_descriptor_source_q <= descriptor_source_id;
                stalled_descriptor_plane_q <= descriptor_plane;
                stalled_descriptor_y_q <= descriptor_y;
                stalled_descriptor_x_q <= descriptor_x;
                stalled_descriptor_k_q <= descriptor_k;
                stalled_descriptor_gates_q <= descriptor_gates;
                stalled_descriptor_mask_q <= descriptor_mask;
                stalled_descriptor_last_q <= descriptor_last;
            end else begin
                descriptor_stall_q <= 1'b0;
            end

            if (term_valid && !term_ready) begin
                if (term_stall_q && (
                    term_source_id != stalled_term_source_q
                    || term_source_plane != stalled_term_plane_q
                    || term_source_y != stalled_term_y_q
                    || term_source_x != stalled_term_x_q
                    || term_lane != stalled_term_lane_q
                    || term_gate != stalled_term_gate_q
                    || term_destination_mask != stalled_term_mask_q
                    || term_last != stalled_term_last_q
                    || term_source_last != stalled_term_source_last_q
                ))
                    $fatal(1, "EREP v4 term changed under backpressure");
                if (trace_enable)
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=direct_online event=stall_observation resource=execute_lane kind=backpressure cycle=%0d window=%0d phase=%0d valid=1 ready=0 fire=0 source_id=%0d plane=%0d y=%0d x=%0d lane=%0d gate=%0d destination_mask=%0h last=%0d source_last=%0d time=%0t scope=%m", event_cycle, event_window, phase_q, term_source_id, term_source_plane, term_source_y, term_source_x, term_lane, term_gate, term_destination_mask, term_last, term_source_last, $time);
                term_stall_q <= 1'b1;
                stalled_term_source_q <= term_source_id;
                stalled_term_plane_q <= term_source_plane;
                stalled_term_y_q <= term_source_y;
                stalled_term_x_q <= term_source_x;
                stalled_term_lane_q <= term_lane;
                stalled_term_gate_q <= term_gate;
                stalled_term_mask_q <= term_destination_mask;
                stalled_term_last_q <= term_last;
                stalled_term_source_last_q <= term_source_last;
            end else begin
                term_stall_q <= 1'b0;
            end
        end
    end
endmodule


// Passive monitor for the existing TCFM5 top configured with its legal 1RW
// backend. It records the same five logical updates and physical bank commands
// as the Direct monitor, plus the explicit prepare/drain state boundaries.
module qfit_local5_erep_tcfm5_monitor_v4 #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int OUT_DIM = 2,
    parameter int GATE_W = 9,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int LANE_W = (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM),
    parameter int OUT_W = (OUT_DIM <= 1) ? 1 : $clog2(OUT_DIM),
    parameter int BANK_DEPTH = TIME_PLANES * HEIGHT * ((WIDTH + 4) / 5),
    parameter int BANK_ADDR_W =
        (BANK_DEPTH <= 1) ? 1 : $clog2(BANK_DEPTH),
    parameter int VEC_W = OUT_DIM * ACC_W
) (
    input logic clk_core,
    input logic rst_core,
    input logic weight_valid,
    input logic weight_ready,
    input logic [LANE_W-1:0] weight_lane,
    input logic [OUT_W-1:0] weight_out,
    input logic signed [W_W-1:0] weight_data,
    input logic weight_last,
    input logic run_start,
    input logic run_start_accepted,
    input logic run_busy,
    input logic run_done,
    input logic [2:0] state,
    input logic term_valid,
    input logic term_ready,
    input logic term_commit,
    input logic [PLANE_W-1:0] term_source_plane,
    input logic [Y_W-1:0] term_source_y,
    input logic [X_W-1:0] term_source_x,
    input logic [LANE_W-1:0] term_lane,
    input logic [GATE_W-1:0] term_gate,
    input logic [4:0] term_destination_mask,
    input logic term_window_last,
    input logic window_close,
    input logic window_close_ready,
    input logic [4:0] bank_update_enable,
    input logic [4:0] bank_update_ready,
    input logic [5*BANK_ADDR_W-1:0] bank_update_addr,
    input logic [VEC_W-1:0] bank_update_delta,
    input logic [4:0] bank_command_valid,
    input logic [4:0] bank_command_write,
    input logic [5*BANK_ADDR_W-1:0] bank_command_addr,
    input logic [5*VEC_W-1:0] bank_command_write_data,
    input logic read_valid,
    input logic read_ready,
    input logic [PLANE_W-1:0] read_plane,
    input logic [Y_W-1:0] read_y,
    input logic [X_W-1:0] read_x,
    input logic [OUT_W-1:0] read_out,
    input logic read_data_valid,
    input logic signed [ACC_W-1:0] read_data,
    input logic vector_read_valid,
    input logic vector_read_ready,
    input logic [PLANE_W-1:0] vector_read_plane,
    input logic [Y_W-1:0] vector_read_y,
    input logic [X_W-1:0] vector_read_x,
    input logic vector_read_data_valid,
    input logic [VEC_W-1:0] vector_read_data,
    input logic protocol_error
);
    localparam int ST_RUN = 2;
    localparam int ST_DRAIN = 3;
    localparam int ST_DONE = 4;
    localparam int PH_PREPARE = 0;
    localparam int PH_EXECUTE = 2;
    localparam int PH_DRAIN = 3;
    localparam int PH_READOUT = 4;

    integer trace_enable;
    integer active_q;
    integer cycle_q;
    integer window_q;
    integer phase_q;
    logic [2:0] previous_state_q;
    logic term_stall_q;
    logic [PLANE_W-1:0] stalled_plane_q;
    logic [Y_W-1:0] stalled_y_q;
    logic [X_W-1:0] stalled_x_q;
    logic [LANE_W-1:0] stalled_lane_q;
    logic [GATE_W-1:0] stalled_gate_q;
    logic [4:0] stalled_mask_q;
    logic stalled_commit_q;
    logic stalled_window_last_q;
    integer pending_read_source_q;
    integer pending_read_out_q;
    integer pending_vector_source_q;

    function automatic integer flatten_source(
        input logic [PLANE_W-1:0] plane,
        input logic [Y_W-1:0] y,
        input logic [X_W-1:0] x
    );
        flatten_source = integer'(plane) * HEIGHT * WIDTH
                       + integer'(y) * WIDTH + integer'(x);
    endfunction

    initial begin
        trace_enable = $test$plusargs("EREP_TRACE_V4");
        window_q = -1;
    end

    always @(posedge clk_core) begin : monitor_tcfm5
        integer event_cycle;
        integer event_window;
        integer bank;

        if (rst_core) begin
            active_q <= 0;
            cycle_q <= 0;
            phase_q <= PH_PREPARE;
            previous_state_q <= state;
            term_stall_q <= 1'b0;
            pending_read_source_q <= -1;
            pending_read_out_q <= -1;
            pending_vector_source_q <= -1;
        end else begin
            event_cycle = run_start_accepted ? 0
                        : (active_q ? cycle_q + 1 : cycle_q);
            event_window = run_start_accepted ? window_q + 1 : window_q;
            previous_state_q <= state;

            if (run_start_accepted) begin
                active_q <= 1;
                cycle_q <= 0;
                window_q <= window_q + 1;
                phase_q <= PH_PREPARE;
                if (trace_enable) begin
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=tcfm5_1rw event=phase_boundary kind=PREPARE_BEGIN cycle=0 window=%0d phase=%0d time=%0t scope=%m", event_window, PH_PREPARE, $time);
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=tcfm5_1rw event=context_prepare resource=context_prepare_1rw kind=context_prepare cycle=0 window=%0d phase=%0d valid=1 ready=1 fire=1 identity=window_%0d time=%0t scope=%m", event_window, PH_PREPARE, event_window, $time);
                end
            end else if (active_q) begin
                cycle_q <= cycle_q + 1;
            end

            if ((active_q || run_start_accepted) && trace_enable)
                $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=tcfm5_1rw event=cycle_snapshot resource=pipeline kind=state cycle=%0d window=%0d phase=%0d state=%0d valid=%0h ready=%0h fire=%0h run_busy=%0d run_done=%0d protocol_error=%0d time=%0t scope=%m", event_cycle, event_window, phase_q, state, {vector_read_valid, read_valid, window_close, term_valid}, {vector_read_ready, read_ready, window_close_ready, term_ready}, {vector_read_valid && vector_read_ready, read_valid && read_ready, window_close && window_close_ready, term_valid && term_ready}, run_busy, run_done, protocol_error, $time);

            if (weight_valid && weight_ready && trace_enable)
                $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=tcfm5_1rw event=weight_accept resource=weight_service kind=weight cycle=%0d window=%0d phase=%0d valid=1 ready=1 fire=1 lane=%0d out=%0d data=%0d last=%0d time=%0t scope=%m", event_cycle, event_window, phase_q, weight_lane, weight_out, weight_data, weight_last, $time);

            if (state == ST_RUN && previous_state_q != ST_RUN) begin
                phase_q <= PH_EXECUTE;
                if (trace_enable)
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=tcfm5_1rw event=phase_boundary kind=EXECUTE_BEGIN cycle=%0d window=%0d phase=%0d time=%0t scope=%m", event_cycle, event_window, PH_EXECUTE, $time);
            end
            if (state == ST_DRAIN && previous_state_q != ST_DRAIN) begin
                phase_q <= PH_DRAIN;
                if (trace_enable)
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=tcfm5_1rw event=phase_boundary kind=DRAIN_BEGIN cycle=%0d window=%0d phase=%0d time=%0t scope=%m", event_cycle, event_window, PH_DRAIN, $time);
            end
            if (state == ST_DONE && previous_state_q != ST_DONE) begin
                phase_q <= PH_READOUT;
                if (trace_enable)
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=tcfm5_1rw event=phase_boundary kind=COMPUTE_DONE cycle=%0d window=%0d phase=%0d time=%0t scope=%m", event_cycle, event_window, PH_READOUT, $time);
            end

            if (term_valid && term_ready) begin
                if (trace_enable)
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=tcfm5_1rw event=term_accept resource=execute_lane kind=term cycle=%0d window=%0d phase=%0d valid=1 ready=1 fire=1 commit=%0d source_id=%0d plane=%0d y=%0d x=%0d lane=%0d gate=%0d destination_mask=%0h last=%0d delta=%0h time=%0t scope=%m", event_cycle, event_window, phase_q, term_commit, flatten_source(term_source_plane, term_source_y, term_source_x), term_source_plane, term_source_y, term_source_x, term_lane, term_gate, term_destination_mask, term_window_last, bank_update_delta, $time);
                if (term_commit) begin
                    for (bank = 0; bank < 5; bank = bank + 1) begin
                        if (bank_update_enable[bank] && trace_enable)
                            $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=tcfm5_1rw event=acc_update_accept resource=acc_bank_%0d_1rw kind=acc_write cycle=%0d window=%0d phase=%0d valid=1 ready=%0d fire=1 bank=%0d source_id=%0d lane=%0d gate=%0d address=%0d delta=%0h time=%0t scope=%m", bank, event_cycle, event_window, phase_q, bank_update_ready[bank], bank, flatten_source(term_source_plane, term_source_y, term_source_x), term_lane, term_gate, bank_update_addr[bank*BANK_ADDR_W +: BANK_ADDR_W], bank_update_delta, $time);
                    end
                end
            end

            for (bank = 0; bank < 5; bank = bank + 1) begin
                if (bank_command_valid[bank] && trace_enable)
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=tcfm5_1rw event=acc_physical_command resource=acc_bank_%0d_1rw kind=%0s cycle=%0d window=%0d phase=%0d valid=1 ready=1 fire=1 bank=%0d address=%0d data=%0h time=%0t scope=%m", bank, bank_command_write[bank] ? "physical_write" : "physical_read", event_cycle, event_window, phase_q, bank, bank_command_addr[bank*BANK_ADDR_W +: BANK_ADDR_W], bank_command_write_data[bank*VEC_W +: VEC_W], $time);
            end

            if (read_valid && read_ready) begin
                pending_read_source_q <= flatten_source(read_plane, read_y, read_x);
                pending_read_out_q <= integer'(read_out);
                if (trace_enable)
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=tcfm5_1rw event=drain_read_accept resource=drain_read_1rw kind=drain_read cycle=%0d window=%0d phase=%0d valid=1 ready=1 fire=1 source_id=%0d plane=%0d y=%0d x=%0d out=%0d time=%0t scope=%m", event_cycle, event_window, phase_q, flatten_source(read_plane, read_y, read_x), read_plane, read_y, read_x, read_out, $time);
            end
            if (vector_read_valid && vector_read_ready) begin
                pending_vector_source_q <= flatten_source(
                    vector_read_plane,
                    vector_read_y,
                    vector_read_x
                );
                if (trace_enable)
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=tcfm5_1rw event=vector_read_accept resource=drain_read_1rw kind=drain_read cycle=%0d window=%0d phase=%0d valid=1 ready=1 fire=1 source_id=%0d plane=%0d y=%0d x=%0d time=%0t scope=%m", event_cycle, event_window, phase_q, flatten_source(vector_read_plane, vector_read_y, vector_read_x), vector_read_plane, vector_read_y, vector_read_x, $time);
            end
            if (read_data_valid && trace_enable)
                $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=tcfm5_1rw event=drain_read_response resource=drain_read_response kind=final cycle=%0d window=%0d phase=%0d valid=1 ready=1 fire=1 source_id=%0d out=%0d data=%0d time=%0t scope=%m", event_cycle, event_window, phase_q, pending_read_source_q, pending_read_out_q, read_data, $time);
            if (vector_read_data_valid && trace_enable)
                $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=tcfm5_1rw event=vector_read_response resource=drain_read_response kind=final_vector cycle=%0d window=%0d phase=%0d valid=1 ready=1 fire=1 source_id=%0d data=%0h time=%0t scope=%m", event_cycle, event_window, phase_q, pending_vector_source_q, vector_read_data, $time);

            if (run_start && !run_start_accepted && trace_enable)
                $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=tcfm5_1rw event=run_start_reject resource=context_prepare_1rw kind=protocol cycle=%0d window=%0d phase=%0d valid=1 ready=0 fire=0 time=%0t scope=%m", event_cycle, event_window, phase_q, $time);
            if (protocol_error)
                $fatal(1, "EREP v4 TCFM5 monitor observed protocol_error");

            if (term_valid && !term_ready) begin
                if (term_stall_q && (
                    term_source_plane != stalled_plane_q
                    || term_source_y != stalled_y_q
                    || term_source_x != stalled_x_q
                    || term_lane != stalled_lane_q
                    || term_gate != stalled_gate_q
                    || term_destination_mask != stalled_mask_q
                    || term_commit != stalled_commit_q
                    || term_window_last != stalled_window_last_q
                ))
                    $fatal(1, "EREP v4 TCFM5 term changed under backpressure");
                if (trace_enable)
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=tcfm5_1rw event=stall_observation resource=execute_lane kind=backpressure cycle=%0d window=%0d phase=%0d valid=1 ready=0 fire=0 commit=%0d source_id=%0d plane=%0d y=%0d x=%0d lane=%0d gate=%0d destination_mask=%0h last=%0d time=%0t scope=%m", event_cycle, event_window, phase_q, term_commit, flatten_source(term_source_plane, term_source_y, term_source_x), term_source_plane, term_source_y, term_source_x, term_lane, term_gate, term_destination_mask, term_window_last, $time);
                term_stall_q <= 1'b1;
                stalled_plane_q <= term_source_plane;
                stalled_y_q <= term_source_y;
                stalled_x_q <= term_source_x;
                stalled_lane_q <= term_lane;
                stalled_gate_q <= term_gate;
                stalled_mask_q <= term_destination_mask;
                stalled_commit_q <= term_commit;
                stalled_window_last_q <= term_window_last;
            end else begin
                term_stall_q <= 1'b0;
            end
        end
    end
endmodule


module qfit_local5_erep_serializer_monitor_v4 #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int OUT_DIM = 2,
    parameter int ACC_W = 32,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int OUT_W = (OUT_DIM <= 1) ? 1 : $clog2(OUT_DIM)
) (
    input logic clk_core,
    input logic rst_core,
    input logic in_valid,
    input logic in_ready,
    input logic [PLANE_W-1:0] in_plane,
    input logic [Y_W-1:0] in_y,
    input logic [X_W-1:0] in_x,
    input logic [OUT_DIM*ACC_W-1:0] in_data,
    input logic in_last,
    input logic out_valid,
    input logic out_ready,
    input logic [PLANE_W-1:0] out_plane,
    input logic [Y_W-1:0] out_y,
    input logic [X_W-1:0] out_x,
    input logic [OUT_W-1:0] out_index,
    input logic signed [ACC_W-1:0] out_data,
    input logic out_last
);
    integer trace_enable;
    integer cycle_q;
    integer window_q;
    integer active_q;
    logic stalled_q;
    logic [PLANE_W-1:0] stalled_plane_q;
    logic [Y_W-1:0] stalled_y_q;
    logic [X_W-1:0] stalled_x_q;
    logic [OUT_W-1:0] stalled_index_q;
    logic signed [ACC_W-1:0] stalled_data_q;
    logic stalled_last_q;

    function automatic integer flatten_source(
        input logic [PLANE_W-1:0] plane,
        input logic [Y_W-1:0] y,
        input logic [X_W-1:0] x
    );
        flatten_source = integer'(plane) * HEIGHT * WIDTH
                       + integer'(y) * WIDTH + integer'(x);
    endfunction

    initial begin
        trace_enable = $test$plusargs("EREP_TRACE_V4");
        window_q = -1;
    end

    always @(posedge clk_core) begin
        if (rst_core) begin
            cycle_q <= 0;
            active_q <= 0;
            stalled_q <= 1'b0;
        end else begin
            cycle_q <= cycle_q + 1;
            if (in_valid && in_ready && !active_q) begin
                active_q <= 1;
                window_q <= window_q + 1;
            end
            if (in_valid && in_ready && trace_enable)
                $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=tcfm5_1rw event=serializer_input resource=vector_serializer kind=vector_accept cycle=%0d window=%0d phase=4 valid=1 ready=1 fire=1 source_id=%0d plane=%0d y=%0d x=%0d data=%0h last=%0d time=%0t scope=%m", cycle_q, active_q ? window_q : window_q + 1, flatten_source(in_plane, in_y, in_x), in_plane, in_y, in_x, in_data, in_last, $time);
            if (out_valid && out_ready && trace_enable)
                $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=tcfm5_1rw event=serializer_output resource=vector_serializer kind=final cycle=%0d window=%0d phase=4 valid=1 ready=1 fire=1 source_id=%0d plane=%0d y=%0d x=%0d out=%0d data=%0d last=%0d time=%0t scope=%m", cycle_q, window_q, flatten_source(out_plane, out_y, out_x), out_plane, out_y, out_x, out_index, out_data, out_last, $time);
            if (out_valid && out_ready && out_last)
                active_q <= 0;

            if (out_valid && !out_ready) begin
                if (stalled_q && (
                    out_plane != stalled_plane_q
                    || out_y != stalled_y_q
                    || out_x != stalled_x_q
                    || out_index != stalled_index_q
                    || out_data != stalled_data_q
                    || out_last != stalled_last_q
                ))
                    $fatal(1, "EREP v4 serializer output changed under backpressure");
                if (trace_enable)
                    $display("EREP_V4 schema=local5_erep_raw_trace_v4 candidate=tcfm5_1rw event=stall_observation resource=vector_serializer kind=backpressure cycle=%0d window=%0d phase=4 valid=1 ready=0 fire=0 source_id=%0d plane=%0d y=%0d x=%0d out=%0d data=%0d last=%0d time=%0t scope=%m", cycle_q, window_q, flatten_source(out_plane, out_y, out_x), out_plane, out_y, out_x, out_index, out_data, out_last, $time);
                stalled_q <= 1'b1;
                stalled_plane_q <= out_plane;
                stalled_y_q <= out_y;
                stalled_x_q <= out_x;
                stalled_index_q <= out_index;
                stalled_data_q <= out_data;
                stalled_last_q <= out_last;
            end else begin
                stalled_q <= 1'b0;
            end
        end
    end
endmodule


// Hierarchy-wide monitor binding is retired. The calibration wrapper must
// instantiate monitors explicitly and expose the audited internal signals.
`ifdef QFIT_EREP_BIND_V4
    QFIT_EREP_BIND_V4_IS_RETIRED__USE_EXPLICIT_MONITOR_PORTS invalid_bind_mode();
`endif
`default_nettype wire
