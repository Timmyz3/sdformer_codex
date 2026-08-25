`timescale 1ns/1ps
`default_nettype none

// Wide-only temporal destination state for the integrated M4 engine.
//
// M4 always updates the full 96-lane vector (all six 16-lane banks) for one
// {context, base_tile}.  Keeping epoch/domain/tag/step metadata independently
// in every bank therefore stores six identical copies and builds six parallel
// admission comparators.  This engine retains the six data SRAM banks but owns
// one coherent metadata record per row.  The legacy dual-granularity engine is
// retained for clients that genuinely require independent narrow-bank traffic.
module qfit_wide_temporal_state_engine #(
    parameter int CONTEXTS = 4,
    parameter int BASE_TILES = 32,
    parameter int BANKS = 6,
    parameter int LANES_PER_BANK = 16,
    parameter int ACC_W = 32,
    parameter int TAG_W = 32,
    parameter int EPOCH_W = 16,
    parameter int DOMAIN_W = 32,
    parameter int STEP_W = 4,
    parameter int LEN_W = 4,
    parameter int CTX_W = (CONTEXTS <= 1) ? 1 : $clog2(CONTEXTS),
    parameter int BASE_TILE_W = (BASE_TILES <= 1) ? 1 : $clog2(BASE_TILES),
    parameter int ROWS = CONTEXTS * BASE_TILES,
    parameter int ROW_ADDR_W = (ROWS <= 1) ? 1 : $clog2(ROWS),
    parameter int BANK_ACC_BITS = LANES_PER_BANK * ACC_W,
    parameter int WIDE_ACC_BITS = BANKS * BANK_ACC_BITS
) (
    input  logic                         clk_core,
    input  logic                         por_core,
    input  logic                         rst_core,
    input  logic [DOMAIN_W-1:0]          active_domain,
    output logic                         domain_fence_ready,
    output logic                         domain_fence_error,

    input  logic                         request_valid,
    output logic                         request_ready,
    input  logic [CTX_W-1:0]             request_context,
    input  logic [BASE_TILE_W-1:0]       request_base_tile,
    input  logic [EPOCH_W-1:0]           request_epoch,
    input  logic [DOMAIN_W-1:0]          request_domain,
    input  logic [STEP_W-1:0]            request_temporal_step,
    input  logic [LEN_W-1:0]             request_temporal_length,
    input  logic                         request_temporal_first,
    input  logic                         request_temporal_last,
    input  logic                         request_use_motion,
    input  logic [TAG_W-1:0]             request_tag,
    input  logic [WIDE_ACC_BITS-1:0]     request_acc,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [CTX_W-1:0]             output_context,
    output logic [BASE_TILE_W-1:0]       output_base_tile,
    output logic [EPOCH_W-1:0]           output_epoch,
    output logic [DOMAIN_W-1:0]          output_domain,
    output logic [STEP_W-1:0]            output_temporal_step,
    output logic [LEN_W-1:0]             output_temporal_length,
    output logic                         output_temporal_first,
    output logic                         output_temporal_last,
    output logic                         output_used_motion,
    output logic [TAG_W-1:0]             output_tag,
    output logic [WIDE_ACC_BITS-1:0]     output_current_acc,

    output logic                         rmw_busy,
    output logic                         protocol_error
);
    logic [EPOCH_W-1:0] state_epoch_q [0:CONTEXTS-1][0:BASE_TILES-1];
    logic [DOMAIN_W-1:0] state_domain_q [0:CONTEXTS-1][0:BASE_TILES-1];
    logic [STEP_W-1:0] next_step_q [0:CONTEXTS-1][0:BASE_TILES-1];
    logic [LEN_W-1:0] sequence_length_q [0:CONTEXTS-1][0:BASE_TILES-1];
    logic [TAG_W-1:0] sequence_tag_q [0:CONTEXTS-1][0:BASE_TILES-1];
    logic epoch_initialized_q [0:CONTEXTS-1][0:BASE_TILES-1];
    logic state_valid_q [0:CONTEXTS-1][0:BASE_TILES-1];
    logic sequence_open_q [0:CONTEXTS-1][0:BASE_TILES-1];

    logic domain_seen_q, domain_armed_q, domain_rearm_pending_q;
    logic domain_fault_q;
    logic [DOMAIN_W-1:0] last_domain_q;

    logic [BANKS-1:0] bank_enable, bank_write_enable;
    logic [ROW_ADDR_W-1:0] bank_address [0:BANKS-1];
    logic [BANK_ACC_BITS-1:0] bank_write_data [0:BANKS-1];
    logic [BANK_ACC_BITS-1:0] bank_read_data [0:BANKS-1];

    logic request_index_valid, request_length_admitted, request_admitted;
    logic request_fire, output_available;
    logic rmw_pending_q, rmw_commit;
    logic [CTX_W-1:0] rmw_context_q;
    logic [BASE_TILE_W-1:0] rmw_base_tile_q;
    logic [EPOCH_W-1:0] rmw_epoch_q;
    logic [DOMAIN_W-1:0] rmw_domain_q;
    logic [STEP_W-1:0] rmw_step_q;
    logic [LEN_W-1:0] rmw_length_q;
    logic rmw_first_q, rmw_last_q;
    logic [TAG_W-1:0] rmw_tag_q;
    logic [WIDE_ACC_BITS-1:0] rmw_delta_q, rmw_result;

    initial begin
        if (CONTEXTS < 1 || BASE_TILES < 1 || BANKS < 1 ||
                LANES_PER_BANK < 1 || ACC_W < 2)
            $error("wide temporal-state geometry must be positive");
    end

    function automatic logic serial_epoch_fresh(
        input logic initialized,
        input logic [DOMAIN_W-1:0] resident_domain,
        input logic [EPOCH_W-1:0] resident_epoch,
        input logic [DOMAIN_W-1:0] candidate_domain,
        input logic [EPOCH_W-1:0] candidate_epoch
    );
        logic [EPOCH_W-1:0] delta;
        begin
            delta = candidate_epoch - resident_epoch;
            serial_epoch_fresh = !initialized ||
                ((resident_domain == candidate_domain) &&
                 (delta != '0) && !delta[EPOCH_W-1]);
        end
    endfunction

    function automatic logic [ROW_ADDR_W-1:0] row_address(
        input logic [CTX_W-1:0] row_context,
        input logic [BASE_TILE_W-1:0] base_tile
    );
        row_address = ROW_ADDR_W'($unsigned(row_context) * BASE_TILES +
                                  $unsigned(base_tile));
    endfunction

    for (genvar bank = 0; bank < BANKS; bank = bank + 1) begin : g_state_bank
        qfit_sync_1rw_acc_bank #(
            .DEPTH(ROWS), .DATA_W(BANK_ACC_BITS), .ADDR_W(ROW_ADDR_W)
        ) u_bank (
            .clk_core,
            .enable(bank_enable[bank]),
            .write_enable(bank_write_enable[bank]),
            .address(bank_address[bank]),
            .write_data(bank_write_data[bank]),
            .read_data(bank_read_data[bank])
        );
    end

    always_ff @(posedge clk_core) begin
        if (por_core) begin
            domain_seen_q <= 1'b0;
            domain_armed_q <= 1'b0;
            domain_rearm_pending_q <= 1'b1;
            domain_fault_q <= 1'b0;
            last_domain_q <= '0;
        end else if (rst_core) begin
            domain_armed_q <= 1'b0;
            domain_rearm_pending_q <= 1'b1;
        end else begin
            if (domain_armed_q && active_domain != last_domain_q) begin
                domain_armed_q <= 1'b0;
                domain_fault_q <= 1'b1;
            end
            if (!domain_armed_q && domain_rearm_pending_q &&
                    (!domain_seen_q || active_domain != last_domain_q)) begin
                domain_seen_q <= 1'b1;
                domain_armed_q <= 1'b1;
                domain_rearm_pending_q <= 1'b0;
                domain_fault_q <= 1'b0;
                last_domain_q <= active_domain;
            end
        end
    end

    assign domain_fence_ready = !por_core && !rst_core &&
                                !domain_fault_q && domain_armed_q &&
                                active_domain == last_domain_q;
    assign domain_fence_error = !por_core && !rst_core &&
        (domain_fault_q ||
         (domain_rearm_pending_q && domain_seen_q &&
          active_domain == last_domain_q) ||
         (domain_armed_q && active_domain != last_domain_q));

    assign request_index_valid = ($unsigned(request_context) < CONTEXTS) &&
                                 ($unsigned(request_base_tile) < BASE_TILES);
    assign request_length_admitted =
        (request_temporal_length == LEN_W'(2)) ||
        (request_temporal_length == LEN_W'(10));

    always_comb begin
        request_admitted = domain_fence_ready && request_index_valid &&
                           request_length_admitted;
        if (request_index_valid) begin
            if (request_temporal_first) begin
                request_admitted = request_admitted &&
                    !request_temporal_last && !request_use_motion &&
                    (request_domain == active_domain) &&
                    (request_temporal_step == '0) &&
                    !sequence_open_q[request_context][request_base_tile] &&
                    serial_epoch_fresh(
                        epoch_initialized_q[request_context][request_base_tile],
                        state_domain_q[request_context][request_base_tile],
                        state_epoch_q[request_context][request_base_tile],
                        request_domain, request_epoch);
            end else begin
                request_admitted = request_admitted &&
                    state_valid_q[request_context][request_base_tile] &&
                    sequence_open_q[request_context][request_base_tile] &&
                    (request_domain == active_domain) &&
                    (state_domain_q[request_context][request_base_tile] ==
                        request_domain) &&
                    (state_epoch_q[request_context][request_base_tile] ==
                        request_epoch) &&
                    (sequence_tag_q[request_context][request_base_tile] ==
                        request_tag) &&
                    (sequence_length_q[request_context][request_base_tile] ==
                        request_temporal_length) &&
                    (next_step_q[request_context][request_base_tile] ==
                        request_temporal_step) &&
                    (request_temporal_last ==
                        (request_temporal_step ==
                         request_temporal_length - 1'b1));
            end
        end
    end

    assign output_available = !output_valid || output_ready;
    assign request_ready = request_valid && request_admitted &&
                           !rmw_pending_q &&
                           (request_use_motion || output_available);
    assign request_fire = request_valid && request_ready;
    assign protocol_error = !por_core && !rst_core && request_valid &&
                            !request_admitted;
    assign rmw_busy = rmw_pending_q;
    assign rmw_commit = !por_core && !rst_core && rmw_pending_q &&
                        output_available;

    always_comb begin
        rmw_result = '0;
        for (int bank = 0; bank < BANKS; bank = bank + 1) begin
            for (int lane = 0; lane < LANES_PER_BANK; lane = lane + 1) begin
                rmw_result[((bank*LANES_PER_BANK+lane)*ACC_W) +: ACC_W] =
                    $signed(bank_read_data[bank][lane*ACC_W +: ACC_W]) +
                    $signed(rmw_delta_q[
                        (bank*LANES_PER_BANK+lane)*ACC_W +: ACC_W]);
            end
        end
    end

    always_comb begin
        for (int bank = 0; bank < BANKS; bank = bank + 1) begin
            bank_enable[bank] = 1'b0;
            bank_write_enable[bank] = 1'b0;
            bank_address[bank] = '0;
            bank_write_data[bank] = '0;
            if (request_fire) begin
                bank_enable[bank] = 1'b1;
                bank_write_enable[bank] = !request_use_motion;
                bank_address[bank] = row_address(
                    request_context, request_base_tile);
                bank_write_data[bank] =
                    request_acc[bank*BANK_ACC_BITS +: BANK_ACC_BITS];
            end else if (rmw_commit) begin
                bank_enable[bank] = 1'b1;
                bank_write_enable[bank] = 1'b1;
                bank_address[bank] = row_address(
                    rmw_context_q, rmw_base_tile_q);
                bank_write_data[bank] =
                    rmw_result[bank*BANK_ACC_BITS +: BANK_ACC_BITS];
            end
        end
    end

    always_ff @(posedge clk_core) begin
        if (por_core || rst_core) begin
            output_valid <= 1'b0;
            output_context <= '0;
            output_base_tile <= '0;
            output_epoch <= '0;
            output_domain <= '0;
            output_temporal_step <= '0;
            output_temporal_length <= '0;
            output_temporal_first <= 1'b0;
            output_temporal_last <= 1'b0;
            output_used_motion <= 1'b0;
            output_tag <= '0;
            output_current_acc <= '0;
            rmw_pending_q <= 1'b0;
            rmw_context_q <= '0;
            rmw_base_tile_q <= '0;
            rmw_epoch_q <= '0;
            rmw_domain_q <= '0;
            rmw_step_q <= '0;
            rmw_length_q <= '0;
            rmw_first_q <= 1'b0;
            rmw_last_q <= 1'b0;
            rmw_tag_q <= '0;
            rmw_delta_q <= '0;
            for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
                for (int tile = 0; tile < BASE_TILES; tile = tile + 1) begin
                    state_epoch_q[ctx][tile] <= '0;
                    state_domain_q[ctx][tile] <= '0;
                    next_step_q[ctx][tile] <= '0;
                    sequence_length_q[ctx][tile] <= '0;
                    sequence_tag_q[ctx][tile] <= '0;
                    epoch_initialized_q[ctx][tile] <= 1'b0;
                    state_valid_q[ctx][tile] <= 1'b0;
                    sequence_open_q[ctx][tile] <= 1'b0;
                end
            end
        end else begin
            if (output_valid && output_ready)
                output_valid <= 1'b0;

            if (request_fire && request_use_motion) begin
                rmw_pending_q <= 1'b1;
                rmw_context_q <= request_context;
                rmw_base_tile_q <= request_base_tile;
                rmw_epoch_q <= request_epoch;
                rmw_domain_q <= request_domain;
                rmw_step_q <= request_temporal_step;
                rmw_length_q <= request_temporal_length;
                rmw_first_q <= request_temporal_first;
                rmw_last_q <= request_temporal_last;
                rmw_tag_q <= request_tag;
                rmw_delta_q <= request_acc;
            end

            if (request_fire && !request_use_motion) begin
                output_valid <= 1'b1;
                output_context <= request_context;
                output_base_tile <= request_base_tile;
                output_epoch <= request_epoch;
                output_domain <= request_domain;
                output_temporal_step <= request_temporal_step;
                output_temporal_length <= request_temporal_length;
                output_temporal_first <= request_temporal_first;
                output_temporal_last <= request_temporal_last;
                output_used_motion <= 1'b0;
                output_tag <= request_tag;
                output_current_acc <= request_acc;
                state_valid_q[request_context][request_base_tile] <= 1'b1;
                sequence_open_q[request_context][request_base_tile] <=
                    !request_temporal_last;
                next_step_q[request_context][request_base_tile] <=
                    request_temporal_step + 1'b1;
                if (request_temporal_first) begin
                    state_epoch_q[request_context][request_base_tile] <=
                        request_epoch;
                    state_domain_q[request_context][request_base_tile] <=
                        request_domain;
                    sequence_length_q[request_context][request_base_tile] <=
                        request_temporal_length;
                    sequence_tag_q[request_context][request_base_tile] <=
                        request_tag;
                    epoch_initialized_q[request_context][request_base_tile] <=
                        1'b1;
                end
            end

            if (rmw_commit) begin
                rmw_pending_q <= 1'b0;
                output_valid <= 1'b1;
                output_context <= rmw_context_q;
                output_base_tile <= rmw_base_tile_q;
                output_epoch <= rmw_epoch_q;
                output_domain <= rmw_domain_q;
                output_temporal_step <= rmw_step_q;
                output_temporal_length <= rmw_length_q;
                output_temporal_first <= rmw_first_q;
                output_temporal_last <= rmw_last_q;
                output_used_motion <= 1'b1;
                output_tag <= rmw_tag_q;
                output_current_acc <= rmw_result;
                state_valid_q[rmw_context_q][rmw_base_tile_q] <= 1'b1;
                sequence_open_q[rmw_context_q][rmw_base_tile_q] <=
                    !rmw_last_q;
                next_step_q[rmw_context_q][rmw_base_tile_q] <=
                    rmw_step_q + 1'b1;
                if (rmw_first_q) begin
                    state_epoch_q[rmw_context_q][rmw_base_tile_q] <=
                        rmw_epoch_q;
                    state_domain_q[rmw_context_q][rmw_base_tile_q] <=
                        rmw_domain_q;
                    sequence_length_q[rmw_context_q][rmw_base_tile_q] <=
                        rmw_length_q;
                    sequence_tag_q[rmw_context_q][rmw_base_tile_q] <=
                        rmw_tag_q;
                    epoch_initialized_q[rmw_context_q][rmw_base_tile_q] <=
                        1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
