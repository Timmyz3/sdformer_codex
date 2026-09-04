`timescale 1ns/1ps
`default_nettype none

// Geometry-Ahead Source-Resident accumulator bank with two contexts.
// The active slot accepts exact vector additions while the spare slot uses a
// single-port backing SRAM for writeback/prefetch.
module qfit_gasr2c_acc_bank #(
    parameter int DEPTH = 90,
    parameter int OUT_DIM = 2,
    parameter int ACC_W = 32,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter int VEC_W = OUT_DIM * ACC_W,
    parameter int MEMORY_IMPL = 0
) (
    input  logic                   clk_core,
    input  logic                   rst_core,
    input  logic                   run_start,

    input  logic                   prepare_valid,
    output logic                   prepare_ready,
    input  logic [ADDR_W-1:0]      prepare_addr,
    input  logic                   activate_valid,
    output logic                   activate_ready,
    input  logic [ADDR_W-1:0]      activate_addr,

    input  logic                   update_valid,
    output logic                   update_ready,
    input  logic [ADDR_W-1:0]      update_addr,
    input  logic [VEC_W-1:0]       update_delta,

    input  logic                   flush_valid,
    output logic                   flush_ready,
    output logic                   flush_done,

    input  logic                   read_valid,
    output logic                   read_ready,
    input  logic [ADDR_W-1:0]      read_addr,
    output logic                   read_data_valid,
    output logic [VEC_W-1:0]       read_data,

    output logic                   protocol_error,
    output logic [31:0]            perf_updates,
    output logic [31:0]            perf_prepare_hits,
    output logic [31:0]            perf_prepare_misses,
    output logic [31:0]            perf_sram_reads,
    output logic [31:0]            perf_sram_writes
);
    typedef enum logic [2:0] {
        ST_IDLE = 3'd0,
        ST_AFTER_EVICT = 3'd1,
        ST_WAIT_PREP_READ = 3'd2,
        ST_FLUSH_SECOND = 3'd3,
        ST_WAIT_READBACK = 3'd4
    } state_t;

    state_t state_q;
    logic [DEPTH-1:0] backing_valid_q;
    logic slot_valid_q [0:1];
    logic slot_dirty_q [0:1];
    logic [ADDR_W-1:0] slot_addr_q [0:1];
    logic [VEC_W-1:0] slot_data_q [0:1];
    logic active_valid_q;
    logic active_sel_q;
    logic prepared_valid_q;
    logic prepared_sel_q;
    logic [ADDR_W-1:0] prepared_addr_q;
    logic [ADDR_W-1:0] pending_target_addr_q;
    logic pending_victim_sel_q;
    logic flush_second_sel_q;
    logic window_flushed_q;
    logic protocol_error_q;
    logic [31:0] updates_q;
    logic [31:0] prepare_hits_q;
    logic [31:0] prepare_misses_q;
    logic [31:0] reads_q;
    logic [31:0] writes_q;

    logic prepare_addr_valid;
    logic activate_addr_valid;
    logic update_addr_valid;
    logic read_addr_valid;
    logic prepare_hit0;
    logic prepare_hit1;
    logic prepare_hit;
    logic prepare_hit_sel;
    logic prepare_victim_sel;
    logic prepare_victim_dirty;
    logic update_fire;
    logic activate_fire;
    logic read_fire;

    logic memory_command_valid;
    logic memory_command_write;
    logic [ADDR_W-1:0] memory_command_addr;
    logic [VEC_W-1:0] memory_command_write_data;
    logic memory_read_data_valid;
    logic [VEC_W-1:0] memory_read_data;

    function automatic logic [VEC_W-1:0] vector_add(
        input logic [VEC_W-1:0] lhs,
        input logic [VEC_W-1:0] rhs
    );
        logic [VEC_W-1:0] result;
        begin
            result = '0;
            for (integer out = 0; out < OUT_DIM; out++)
                result[out*ACC_W +: ACC_W] = ACC_W'(
                    signed'(lhs[out*ACC_W +: ACC_W])
                    + signed'(rhs[out*ACC_W +: ACC_W])
                );
            vector_add = result;
        end
    endfunction

    assign prepare_addr_valid = 32'(prepare_addr) < DEPTH;
    assign activate_addr_valid = 32'(activate_addr) < DEPTH;
    assign update_addr_valid = 32'(update_addr) < DEPTH;
    assign read_addr_valid = 32'(read_addr) < DEPTH;
    assign prepare_hit0 = prepare_addr_valid && slot_valid_q[0]
                        && slot_addr_q[0] == prepare_addr;
    assign prepare_hit1 = prepare_addr_valid && slot_valid_q[1]
                        && slot_addr_q[1] == prepare_addr;
    assign prepare_hit = prepare_hit0 || prepare_hit1;
    assign prepare_hit_sel = prepare_hit1;
    assign prepare_victim_sel = active_valid_q
        ? !active_sel_q
        : (slot_valid_q[0] && !slot_valid_q[1]);
    assign prepare_victim_dirty = slot_valid_q[prepare_victim_sel]
                                && slot_dirty_q[prepare_victim_sel];

    assign prepare_ready = prepared_valid_q
                         && prepared_addr_q == prepare_addr;
    assign activate_ready = state_q == ST_IDLE && prepared_valid_q
                          && activate_addr_valid
                          && prepared_addr_q == activate_addr;
    assign activate_fire = activate_valid && activate_ready;
    assign update_ready = active_valid_q && update_addr_valid
                        && slot_addr_q[active_sel_q] == update_addr
                        && !window_flushed_q
                        && state_q != ST_FLUSH_SECOND
                        && state_q != ST_WAIT_READBACK;
    assign update_fire = update_valid && update_ready;
    assign flush_ready = state_q == ST_IDLE && !prepared_valid_q
                       && !prepare_valid && !update_valid;
    assign read_ready = state_q == ST_IDLE && window_flushed_q
                     && !prepare_valid && !flush_valid && read_addr_valid;
    assign read_fire = read_valid && read_ready;
    assign protocol_error = protocol_error_q;
    assign perf_updates = updates_q;
    assign perf_prepare_hits = prepare_hits_q;
    assign perf_prepare_misses = prepare_misses_q;
    assign perf_sram_reads = reads_q;
    assign perf_sram_writes = writes_q;

    always_comb begin
        memory_command_valid = 1'b0;
        memory_command_write = 1'b0;
        memory_command_addr = '0;
        memory_command_write_data = '0;

        if (state_q == ST_IDLE && flush_valid && flush_ready) begin
            if (slot_valid_q[0] && slot_dirty_q[0]) begin
                memory_command_valid = 1'b1;
                memory_command_write = 1'b1;
                memory_command_addr = slot_addr_q[0];
                memory_command_write_data = slot_data_q[0];
            end else if (slot_valid_q[1] && slot_dirty_q[1]) begin
                memory_command_valid = 1'b1;
                memory_command_write = 1'b1;
                memory_command_addr = slot_addr_q[1];
                memory_command_write_data = slot_data_q[1];
            end
        end else if (
            state_q == ST_IDLE && prepare_valid && !prepared_valid_q
            && prepare_addr_valid && !prepare_hit
        ) begin
            memory_command_addr = prepare_victim_dirty
                ? slot_addr_q[prepare_victim_sel] : prepare_addr;
            memory_command_write = prepare_victim_dirty;
            memory_command_write_data = slot_data_q[prepare_victim_sel];
            memory_command_valid = prepare_victim_dirty
                                || backing_valid_q[prepare_addr];
        end else if (state_q == ST_AFTER_EVICT) begin
            memory_command_valid = backing_valid_q[pending_target_addr_q];
            memory_command_addr = pending_target_addr_q;
        end else if (state_q == ST_FLUSH_SECOND) begin
            memory_command_valid = 1'b1;
            memory_command_write = 1'b1;
            memory_command_addr = slot_addr_q[flush_second_sel_q];
            memory_command_write_data = slot_data_q[flush_second_sel_q];
        end else if (
            state_q == ST_IDLE && read_fire && backing_valid_q[read_addr]
        ) begin
            memory_command_valid = 1'b1;
            memory_command_addr = read_addr;
        end
    end

    qfit_single_port_acc_memory #(
        .DEPTH(DEPTH), .VEC_W(VEC_W), .ADDR_W(ADDR_W),
        .MEMORY_IMPL(MEMORY_IMPL)
    ) u_memory (
        .clk_core(clk_core), .rst_core(rst_core),
        .command_valid(memory_command_valid),
        .command_write(memory_command_write),
        .command_addr(memory_command_addr),
        .command_write_data(memory_command_write_data),
        .read_data_valid(memory_read_data_valid),
        .read_data(memory_read_data)
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            backing_valid_q <= '0;
            active_valid_q <= 1'b0;
            active_sel_q <= 1'b0;
            prepared_valid_q <= 1'b0;
            prepared_sel_q <= 1'b0;
            prepared_addr_q <= '0;
            pending_target_addr_q <= '0;
            pending_victim_sel_q <= 1'b0;
            flush_second_sel_q <= 1'b0;
            window_flushed_q <= 1'b0;
            flush_done <= 1'b0;
            read_data_valid <= 1'b0;
            read_data <= '0;
            protocol_error_q <= 1'b0;
            updates_q <= '0;
            prepare_hits_q <= '0;
            prepare_misses_q <= '0;
            reads_q <= '0;
            writes_q <= '0;
            for (integer slot = 0; slot < 2; slot++) begin
                slot_valid_q[slot] <= 1'b0;
                slot_dirty_q[slot] <= 1'b0;
                slot_addr_q[slot] <= '0;
                slot_data_q[slot] <= '0;
            end
        end else if (run_start) begin
            state_q <= ST_IDLE;
            backing_valid_q <= '0;
            active_valid_q <= 1'b0;
            prepared_valid_q <= 1'b0;
            window_flushed_q <= 1'b0;
            flush_done <= 1'b0;
            read_data_valid <= 1'b0;
            protocol_error_q <= 1'b0;
            updates_q <= '0;
            prepare_hits_q <= '0;
            prepare_misses_q <= '0;
            reads_q <= '0;
            writes_q <= '0;
            for (integer slot = 0; slot < 2; slot++) begin
                slot_valid_q[slot] <= 1'b0;
                slot_dirty_q[slot] <= 1'b0;
                slot_addr_q[slot] <= '0;
                slot_data_q[slot] <= '0;
            end
        end else begin
            flush_done <= 1'b0;
            read_data_valid <= 1'b0;
            if (prepare_valid && !prepare_addr_valid)
                protocol_error_q <= 1'b1;
            if (
                activate_valid && prepared_valid_q
                && (!activate_addr_valid || activate_addr != prepared_addr_q)
            )
                protocol_error_q <= 1'b1;
            if (read_valid && !read_addr_valid)
                protocol_error_q <= 1'b1;

            if (update_fire) begin
                slot_data_q[active_sel_q] <= vector_add(
                    slot_data_q[active_sel_q], update_delta
                );
                slot_dirty_q[active_sel_q] <= 1'b1;
                updates_q <= updates_q + 1'b1;
            end
            if (activate_fire) begin
                active_valid_q <= 1'b1;
                active_sel_q <= prepared_sel_q;
                prepared_valid_q <= 1'b0;
            end

            case (state_q)
                ST_IDLE: begin
                    if (flush_valid && flush_ready) begin
                        if (slot_valid_q[0] && slot_dirty_q[0]) begin
                            backing_valid_q[slot_addr_q[0]] <= 1'b1;
                            slot_dirty_q[0] <= 1'b0;
                            writes_q <= writes_q + 1'b1;
                            if (slot_valid_q[1] && slot_dirty_q[1]) begin
                                flush_second_sel_q <= 1'b1;
                                state_q <= ST_FLUSH_SECOND;
                            end else begin
                                window_flushed_q <= 1'b1;
                                flush_done <= 1'b1;
                            end
                        end else if (slot_valid_q[1] && slot_dirty_q[1]) begin
                            backing_valid_q[slot_addr_q[1]] <= 1'b1;
                            slot_dirty_q[1] <= 1'b0;
                            writes_q <= writes_q + 1'b1;
                            window_flushed_q <= 1'b1;
                            flush_done <= 1'b1;
                        end else begin
                            window_flushed_q <= 1'b1;
                            flush_done <= 1'b1;
                        end
                    end else if (prepare_valid && !prepared_valid_q) begin
                        if (prepare_hit) begin
                            prepared_valid_q <= 1'b1;
                            prepared_sel_q <= prepare_hit_sel;
                            prepared_addr_q <= prepare_addr;
                            prepare_hits_q <= prepare_hits_q + 1'b1;
                        end else if (prepare_addr_valid) begin
                            prepare_misses_q <= prepare_misses_q + 1'b1;
                            pending_target_addr_q <= prepare_addr;
                            pending_victim_sel_q <= prepare_victim_sel;
                            if (prepare_victim_dirty) begin
                                backing_valid_q[
                                    slot_addr_q[prepare_victim_sel]
                                ] <= 1'b1;
                                slot_dirty_q[prepare_victim_sel] <= 1'b0;
                                writes_q <= writes_q + 1'b1;
                                state_q <= ST_AFTER_EVICT;
                            end else if (backing_valid_q[prepare_addr]) begin
                                reads_q <= reads_q + 1'b1;
                                state_q <= ST_WAIT_PREP_READ;
                            end else begin
                                slot_valid_q[prepare_victim_sel] <= 1'b1;
                                slot_dirty_q[prepare_victim_sel] <= 1'b0;
                                slot_addr_q[prepare_victim_sel] <= prepare_addr;
                                slot_data_q[prepare_victim_sel] <= '0;
                                prepared_valid_q <= 1'b1;
                                prepared_sel_q <= prepare_victim_sel;
                                prepared_addr_q <= prepare_addr;
                            end
                        end
                    end else if (read_fire) begin
                        if (backing_valid_q[read_addr]) begin
                            reads_q <= reads_q + 1'b1;
                            state_q <= ST_WAIT_READBACK;
                        end else begin
                            read_data <= '0;
                            read_data_valid <= 1'b1;
                        end
                    end
                end

                ST_AFTER_EVICT: begin
                    if (backing_valid_q[pending_target_addr_q]) begin
                        reads_q <= reads_q + 1'b1;
                        state_q <= ST_WAIT_PREP_READ;
                    end else begin
                        slot_valid_q[pending_victim_sel_q] <= 1'b1;
                        slot_dirty_q[pending_victim_sel_q] <= 1'b0;
                        slot_addr_q[pending_victim_sel_q]
                            <= pending_target_addr_q;
                        slot_data_q[pending_victim_sel_q] <= '0;
                        prepared_valid_q <= 1'b1;
                        prepared_sel_q <= pending_victim_sel_q;
                        prepared_addr_q <= pending_target_addr_q;
                        state_q <= ST_IDLE;
                    end
                end

                ST_WAIT_PREP_READ: begin
                    if (memory_read_data_valid) begin
                        slot_valid_q[pending_victim_sel_q] <= 1'b1;
                        slot_dirty_q[pending_victim_sel_q] <= 1'b0;
                        slot_addr_q[pending_victim_sel_q]
                            <= pending_target_addr_q;
                        slot_data_q[pending_victim_sel_q]
                            <= memory_read_data;
                        prepared_valid_q <= 1'b1;
                        prepared_sel_q <= pending_victim_sel_q;
                        prepared_addr_q <= pending_target_addr_q;
                        state_q <= ST_IDLE;
                    end
                end

                ST_FLUSH_SECOND: begin
                    backing_valid_q[slot_addr_q[flush_second_sel_q]] <= 1'b1;
                    slot_dirty_q[flush_second_sel_q] <= 1'b0;
                    writes_q <= writes_q + 1'b1;
                    window_flushed_q <= 1'b1;
                    flush_done <= 1'b1;
                    state_q <= ST_IDLE;
                end

                ST_WAIT_READBACK: begin
                    if (memory_read_data_valid) begin
                        read_data <= memory_read_data;
                        read_data_valid <= 1'b1;
                        state_q <= ST_IDLE;
                    end
                end

                default: state_q <= ST_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
