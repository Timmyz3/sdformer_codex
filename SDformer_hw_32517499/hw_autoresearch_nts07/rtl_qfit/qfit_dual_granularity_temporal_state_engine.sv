`timescale 1ns/1ps
`default_nettype none

// M9.1 shared Local/Motion temporal-state fabric with an SRAM-realistic
// protocol.  A wide transaction reserves all 16-lane banks atomically; a
// narrow ATLIF transaction reserves one bank.  Absolute Local refreshes write
// directly, while Motion deltas execute a synchronous read-modify-write.
module qfit_dual_granularity_temporal_state_engine #(
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
    parameter int BANK_W = (BANKS <= 1) ? 1 : $clog2(BANKS),
    parameter int ROWS = CONTEXTS * BASE_TILES,
    parameter int ROW_ADDR_W = (ROWS <= 1) ? 1 : $clog2(ROWS),
    parameter int BANK_ACC_BITS = LANES_PER_BANK * ACC_W,
    parameter int WIDE_ACC_BITS = BANKS * BANK_ACC_BITS
) (
    input  logic                         clk_core,
    // POR initializes the always-on domain watermark and automatically arms
    // the first observed domain after release.  rst_core is a recoverable
    // functional reset and must not erase the previous-domain watermark.
    input  logic                         por_core,
    input  logic                         rst_core,
    input  logic [DOMAIN_W-1:0]          active_domain,
    output logic                         domain_fence_ready,
    output logic                         domain_fence_error,

    input  logic                         wide_valid,
    output logic                         wide_ready,
    input  logic [CTX_W-1:0]             wide_context,
    input  logic [BASE_TILE_W-1:0]       wide_base_tile,
    input  logic [EPOCH_W-1:0]           wide_epoch,
    input  logic [DOMAIN_W-1:0]          wide_domain,
    input  logic [STEP_W-1:0]            wide_temporal_step,
    input  logic [LEN_W-1:0]             wide_temporal_length,
    input  logic                         wide_temporal_first,
    input  logic                         wide_temporal_last,
    input  logic                         wide_use_motion,
    input  logic [TAG_W-1:0]             wide_tag,
    input  logic [WIDE_ACC_BITS-1:0]     wide_acc,

    input  logic                         narrow_valid,
    output logic                         narrow_ready,
    input  logic [CTX_W-1:0]             narrow_context,
    input  logic [BASE_TILE_W-1:0]       narrow_base_tile,
    input  logic [BANK_W-1:0]            narrow_bank,
    input  logic [EPOCH_W-1:0]           narrow_epoch,
    input  logic [DOMAIN_W-1:0]          narrow_domain,
    input  logic [STEP_W-1:0]            narrow_temporal_step,
    input  logic [LEN_W-1:0]             narrow_temporal_length,
    input  logic                         narrow_temporal_first,
    input  logic                         narrow_temporal_last,
    input  logic                         narrow_use_motion,
    input  logic [TAG_W-1:0]             narrow_tag,
    input  logic [BANK_ACC_BITS-1:0]     narrow_acc,

    // An abort invalidates exactly the selected resident banks while keeping
    // their epoch watermark.  A wide abort uses an all-one bank mask.
    input  logic                         abort_valid,
    output logic                         abort_ready,
    input  logic [CTX_W-1:0]             abort_context,
    input  logic [BASE_TILE_W-1:0]       abort_base_tile,
    input  logic [BANKS-1:0]             abort_bank_mask,
    input  logic [EPOCH_W-1:0]           abort_epoch,
    input  logic [DOMAIN_W-1:0]          abort_domain,
    input  logic [TAG_W-1:0]             abort_tag,
    output logic                         abort_error,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic                         output_is_wide,
    output logic [CTX_W-1:0]             output_context,
    output logic [BASE_TILE_W-1:0]       output_base_tile,
    output logic [BANKS-1:0]             output_bank_mask,
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
    output logic                         wide_protocol_error,
    output logic                         narrow_protocol_error
);
    logic [EPOCH_W-1:0] state_epoch_q
        [0:CONTEXTS-1][0:BASE_TILES-1][0:BANKS-1];
    logic [DOMAIN_W-1:0] state_domain_q
        [0:CONTEXTS-1][0:BASE_TILES-1][0:BANKS-1];
    logic [STEP_W-1:0] next_step_q
        [0:CONTEXTS-1][0:BASE_TILES-1][0:BANKS-1];
    logic [LEN_W-1:0] sequence_length_q
        [0:CONTEXTS-1][0:BASE_TILES-1][0:BANKS-1];
    logic [TAG_W-1:0] sequence_tag_q
        [0:CONTEXTS-1][0:BASE_TILES-1][0:BANKS-1];
    logic epoch_initialized_q
        [0:CONTEXTS-1][0:BASE_TILES-1][0:BANKS-1];
    logic state_valid_q
        [0:CONTEXTS-1][0:BASE_TILES-1][0:BANKS-1];
    logic sequence_open_q
        [0:CONTEXTS-1][0:BASE_TILES-1][0:BANKS-1];

    logic domain_seen_q;
    logic domain_armed_q;
    logic domain_rearm_pending_q;
    logic domain_fault_q;
    logic [DOMAIN_W-1:0] last_domain_q;

    logic [BANKS-1:0] bank_enable;
    logic [BANKS-1:0] bank_write_enable;
    logic [ROW_ADDR_W-1:0] bank_address [0:BANKS-1];
    logic [BANK_ACC_BITS-1:0] bank_write_data [0:BANKS-1];
    logic [BANK_ACC_BITS-1:0] bank_read_data [0:BANKS-1];

    logic wide_index_valid;
    logic narrow_index_valid;
    logic abort_index_valid;
    logic wide_length_admitted;
    logic narrow_length_admitted;
    logic wide_admitted;
    logic narrow_admitted;
    logic abort_admitted;
    logic wide_bank_admitted [0:BANKS-1];
    logic abort_bank_admitted [0:BANKS-1];
    logic wide_eligible;
    logic narrow_eligible;
    logic abort_eligible;
    logic grant_wide;
    logic grant_narrow;
    logic grant_abort;
    logic wide_fire;
    logic narrow_fire;
    logic abort_fire;
    logic last_grant_wide_q;

    logic output_valid_q;
    logic output_available;
    logic abort_output_conflict;

    logic rmw_pending_q;
    logic rmw_commit;
    logic rmw_is_wide_q;
    logic [CTX_W-1:0] rmw_context_q;
    logic [BASE_TILE_W-1:0] rmw_base_tile_q;
    logic [BANKS-1:0] rmw_bank_mask_q;
    logic [EPOCH_W-1:0] rmw_epoch_q;
    logic [DOMAIN_W-1:0] rmw_domain_q;
    logic [STEP_W-1:0] rmw_step_q;
    logic [LEN_W-1:0] rmw_length_q;
    logic rmw_first_q;
    logic rmw_last_q;
    logic [TAG_W-1:0] rmw_tag_q;
    logic [WIDE_ACC_BITS-1:0] rmw_delta_q;
    logic [WIDE_ACC_BITS-1:0] rmw_result;
    logic [WIDE_ACC_BITS-1:0] narrow_acc_wide;

    function automatic logic serial_epoch_fresh(
        input logic initialized,
        input logic [DOMAIN_W-1:0] resident_domain,
        input logic [EPOCH_W-1:0] resident_epoch,
        input logic [DOMAIN_W-1:0] request_domain,
        input logic [EPOCH_W-1:0] request_epoch
    );
        logic [EPOCH_W-1:0] delta;
        begin
            delta = request_epoch - resident_epoch;
            serial_epoch_fresh = !initialized ||
                ((resident_domain == request_domain) &&
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

    initial begin
        if (CONTEXTS < 1 || BASE_TILES < 1 || BANKS < 1 ||
                LANES_PER_BANK < 1)
            $error("temporal-state geometry must be positive");
        if (STEP_W < 4 || LEN_W < 4)
            $error("STEP_W and LEN_W must represent T10");
        if (EPOCH_W < 2)
            $error("EPOCH_W must support serial-number freshness");
    end

    for (genvar bank = 0; bank < BANKS; bank = bank + 1) begin : g_state_bank
        qfit_sync_1rw_acc_bank #(
            .DEPTH(ROWS),
            .DATA_W(BANK_ACC_BITS),
            .ADDR_W(ROW_ADDR_W)
        ) u_acc_bank (
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

    assign wide_index_valid = ($unsigned(wide_context) < CONTEXTS) &&
                              ($unsigned(wide_base_tile) < BASE_TILES);
    assign narrow_index_valid = ($unsigned(narrow_context) < CONTEXTS) &&
                                ($unsigned(narrow_base_tile) < BASE_TILES) &&
                                ($unsigned(narrow_bank) < BANKS);
    assign abort_index_valid = ($unsigned(abort_context) < CONTEXTS) &&
                               ($unsigned(abort_base_tile) < BASE_TILES);
    assign wide_length_admitted = (wide_temporal_length == LEN_W'(2)) ||
                                  (wide_temporal_length == LEN_W'(10));
    assign narrow_length_admitted = (narrow_temporal_length == LEN_W'(2)) ||
                                    (narrow_temporal_length == LEN_W'(10));

    always_comb begin
        wide_admitted = domain_fence_ready && wide_index_valid &&
                        wide_length_admitted;
        for (int bank = 0; bank < BANKS; bank = bank + 1) begin
            wide_bank_admitted[bank] = 1'b0;
            if (wide_index_valid) begin
                if (wide_temporal_first) begin
                    wide_bank_admitted[bank] = !wide_temporal_last &&
                        !wide_use_motion && (wide_domain == active_domain) &&
                        (wide_temporal_step == '0) &&
                        !sequence_open_q[wide_context][wide_base_tile][bank] &&
                        serial_epoch_fresh(
                            epoch_initialized_q[wide_context][wide_base_tile][bank],
                            state_domain_q[wide_context][wide_base_tile][bank],
                            state_epoch_q[wide_context][wide_base_tile][bank],
                            wide_domain, wide_epoch);
                end else begin
                    wide_bank_admitted[bank] =
                        state_valid_q[wide_context][wide_base_tile][bank] &&
                        sequence_open_q[wide_context][wide_base_tile][bank] &&
                        (wide_domain == active_domain) &&
                        (state_domain_q[wide_context][wide_base_tile][bank] == wide_domain) &&
                        (state_epoch_q[wide_context][wide_base_tile][bank] == wide_epoch) &&
                        (sequence_tag_q[wide_context][wide_base_tile][bank] == wide_tag) &&
                        (sequence_length_q[wide_context][wide_base_tile][bank] ==
                            wide_temporal_length) &&
                        (next_step_q[wide_context][wide_base_tile][bank] ==
                            wide_temporal_step) &&
                        (wide_temporal_last ==
                            (wide_temporal_step == (wide_temporal_length - 1'b1)));
                end
            end
            wide_admitted = wide_admitted && wide_bank_admitted[bank];
        end

        narrow_admitted = domain_fence_ready && narrow_index_valid &&
                          narrow_length_admitted;
        if (narrow_index_valid) begin
            if (narrow_temporal_first) begin
                narrow_admitted = narrow_admitted && !narrow_temporal_last &&
                    !narrow_use_motion && (narrow_domain == active_domain) &&
                    (narrow_temporal_step == '0) &&
                    !sequence_open_q[narrow_context][narrow_base_tile][narrow_bank] &&
                    serial_epoch_fresh(
                        epoch_initialized_q[narrow_context][narrow_base_tile][narrow_bank],
                        state_domain_q[narrow_context][narrow_base_tile][narrow_bank],
                        state_epoch_q[narrow_context][narrow_base_tile][narrow_bank],
                        narrow_domain, narrow_epoch);
            end else begin
                narrow_admitted = narrow_admitted &&
                    state_valid_q[narrow_context][narrow_base_tile][narrow_bank] &&
                    sequence_open_q[narrow_context][narrow_base_tile][narrow_bank] &&
                    (narrow_domain == active_domain) &&
                    (state_domain_q[narrow_context][narrow_base_tile][narrow_bank] ==
                        narrow_domain) &&
                    (state_epoch_q[narrow_context][narrow_base_tile][narrow_bank] ==
                        narrow_epoch) &&
                    (sequence_tag_q[narrow_context][narrow_base_tile][narrow_bank] ==
                        narrow_tag) &&
                    (sequence_length_q[narrow_context][narrow_base_tile][narrow_bank] ==
                        narrow_temporal_length) &&
                    (next_step_q[narrow_context][narrow_base_tile][narrow_bank] ==
                        narrow_temporal_step) &&
                    (narrow_temporal_last ==
                        (narrow_temporal_step == (narrow_temporal_length - 1'b1)));
            end
        end

        abort_admitted = domain_fence_ready && abort_index_valid &&
                         (abort_bank_mask != '0) &&
                         (abort_domain == active_domain);
        for (int bank = 0; bank < BANKS; bank = bank + 1) begin
            abort_bank_admitted[bank] = 1'b1;
            if (abort_bank_mask[bank]) begin
                abort_bank_admitted[bank] = abort_index_valid &&
                    state_valid_q[abort_context][abort_base_tile][bank] &&
                    sequence_open_q[abort_context][abort_base_tile][bank] &&
                    (state_domain_q[abort_context][abort_base_tile][bank] ==
                        abort_domain) &&
                    (state_epoch_q[abort_context][abort_base_tile][bank] ==
                        abort_epoch) &&
                    (sequence_tag_q[abort_context][abort_base_tile][bank] ==
                        abort_tag);
            end
            abort_admitted = abort_admitted && abort_bank_admitted[bank];
        end
    end

    assign output_available = !output_valid_q || output_ready;
    assign abort_output_conflict = output_valid_q && !output_ready &&
        (output_context == abort_context) &&
        (output_base_tile == abort_base_tile) &&
        (output_epoch == abort_epoch) && (output_domain == abort_domain) &&
        (output_tag == abort_tag) &&
        (|(output_bank_mask & abort_bank_mask));

    assign wide_eligible = wide_valid && wide_admitted && !rmw_pending_q &&
                           (wide_use_motion || output_available);
    assign narrow_eligible = narrow_valid && narrow_admitted && !rmw_pending_q &&
                             (narrow_use_motion || output_available);
    assign abort_eligible = abort_valid && abort_admitted && !rmw_pending_q &&
                            !abort_output_conflict;
    assign grant_abort = abort_eligible;
    assign grant_wide = !grant_abort && wide_eligible &&
                        (!narrow_eligible || !last_grant_wide_q);
    assign grant_narrow = !grant_abort && narrow_eligible && !grant_wide;
    assign wide_ready = grant_wide;
    assign narrow_ready = grant_narrow;
    assign abort_ready = grant_abort;
    assign wide_fire = wide_valid && wide_ready;
    assign narrow_fire = narrow_valid && narrow_ready;
    assign abort_fire = abort_valid && abort_ready;
    assign wide_protocol_error = !por_core && !rst_core && wide_valid &&
                                 !wide_admitted;
    assign narrow_protocol_error = !por_core && !rst_core && narrow_valid &&
                                   !narrow_admitted;
    assign abort_error = !por_core && !rst_core && abort_valid &&
                         (!abort_admitted || abort_output_conflict);
    assign output_valid = output_valid_q;
    assign rmw_busy = rmw_pending_q;
    // A reset may arrive between the synchronous read and its RMW writeback.
    // Gate the combinational memory command directly; waiting for the reset
    // edge to clear rmw_pending_q would permit one stale write on that edge.
    assign rmw_commit = !por_core && !rst_core && rmw_pending_q &&
                        output_available;

    always_comb begin
        narrow_acc_wide = '0;
        if ($unsigned(narrow_bank) < BANKS)
            narrow_acc_wide[(narrow_bank*BANK_ACC_BITS) +: BANK_ACC_BITS] =
                narrow_acc;
        rmw_result = '0;
        for (int bank = 0; bank < BANKS; bank = bank + 1) begin
            for (int lane = 0; lane < LANES_PER_BANK; lane = lane + 1) begin
                if (rmw_bank_mask_q[bank]) begin
                    rmw_result[((bank*LANES_PER_BANK+lane)*ACC_W) +: ACC_W] =
                        $signed(bank_read_data[bank][(lane*ACC_W) +: ACC_W]) +
                        $signed(rmw_delta_q[
                            ((bank*LANES_PER_BANK+lane)*ACC_W) +: ACC_W]);
                end
            end
        end
    end

    always_comb begin
        for (int bank = 0; bank < BANKS; bank = bank + 1) begin
            bank_enable[bank] = 1'b0;
            bank_write_enable[bank] = 1'b0;
            bank_address[bank] = '0;
            bank_write_data[bank] = '0;
            if (wide_fire) begin
                bank_enable[bank] = 1'b1;
                bank_write_enable[bank] = !wide_use_motion;
                bank_address[bank] = row_address(wide_context, wide_base_tile);
                bank_write_data[bank] =
                    wide_acc[(bank*BANK_ACC_BITS) +: BANK_ACC_BITS];
            end else if (narrow_fire && bank == $unsigned(narrow_bank)) begin
                bank_enable[bank] = 1'b1;
                bank_write_enable[bank] = !narrow_use_motion;
                bank_address[bank] = row_address(narrow_context, narrow_base_tile);
                bank_write_data[bank] = narrow_acc;
            end else if (rmw_commit && rmw_bank_mask_q[bank]) begin
                bank_enable[bank] = 1'b1;
                bank_write_enable[bank] = 1'b1;
                bank_address[bank] = row_address(rmw_context_q, rmw_base_tile_q);
                bank_write_data[bank] =
                    rmw_result[(bank*BANK_ACC_BITS) +: BANK_ACC_BITS];
            end
        end
    end

    always_ff @(posedge clk_core) begin
        if (por_core || rst_core) begin
            output_valid_q <= 1'b0;
            output_is_wide <= 1'b0;
            output_context <= '0;
            output_base_tile <= '0;
            output_bank_mask <= '0;
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
            rmw_is_wide_q <= 1'b0;
            rmw_context_q <= '0;
            rmw_base_tile_q <= '0;
            rmw_bank_mask_q <= '0;
            rmw_epoch_q <= '0;
            rmw_domain_q <= '0;
            rmw_step_q <= '0;
            rmw_length_q <= '0;
            rmw_first_q <= 1'b0;
            rmw_last_q <= 1'b0;
            rmw_tag_q <= '0;
            rmw_delta_q <= '0;
            last_grant_wide_q <= 1'b0;
            for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
                for (int tile = 0; tile < BASE_TILES; tile = tile + 1) begin
                    for (int bank = 0; bank < BANKS; bank = bank + 1) begin
                        state_epoch_q[ctx][tile][bank] <= '0;
                        state_domain_q[ctx][tile][bank] <= '0;
                        next_step_q[ctx][tile][bank] <= '0;
                        sequence_length_q[ctx][tile][bank] <= '0;
                        sequence_tag_q[ctx][tile][bank] <= '0;
                        epoch_initialized_q[ctx][tile][bank] <= 1'b0;
                        state_valid_q[ctx][tile][bank] <= 1'b0;
                        sequence_open_q[ctx][tile][bank] <= 1'b0;
                    end
                end
            end
        end else begin
            if (output_valid_q && output_ready)
                output_valid_q <= 1'b0;

            if (abort_fire) begin
                for (int bank = 0; bank < BANKS; bank = bank + 1) begin
                    if (abort_bank_mask[bank]) begin
                        state_valid_q[abort_context][abort_base_tile][bank] <= 1'b0;
                        sequence_open_q[abort_context][abort_base_tile][bank] <= 1'b0;
                    end
                end
            end

            if (wide_fire || narrow_fire)
                last_grant_wide_q <= wide_fire;

            if (wide_fire && wide_use_motion) begin
                rmw_pending_q <= 1'b1;
                rmw_is_wide_q <= 1'b1;
                rmw_context_q <= wide_context;
                rmw_base_tile_q <= wide_base_tile;
                rmw_bank_mask_q <= {BANKS{1'b1}};
                rmw_epoch_q <= wide_epoch;
                rmw_domain_q <= wide_domain;
                rmw_step_q <= wide_temporal_step;
                rmw_length_q <= wide_temporal_length;
                rmw_first_q <= wide_temporal_first;
                rmw_last_q <= wide_temporal_last;
                rmw_tag_q <= wide_tag;
                rmw_delta_q <= wide_acc;
            end else if (narrow_fire && narrow_use_motion) begin
                rmw_pending_q <= 1'b1;
                rmw_is_wide_q <= 1'b0;
                rmw_context_q <= narrow_context;
                rmw_base_tile_q <= narrow_base_tile;
                rmw_bank_mask_q <= BANKS'(1'b1) << narrow_bank;
                rmw_epoch_q <= narrow_epoch;
                rmw_domain_q <= narrow_domain;
                rmw_step_q <= narrow_temporal_step;
                rmw_length_q <= narrow_temporal_length;
                rmw_first_q <= narrow_temporal_first;
                rmw_last_q <= narrow_temporal_last;
                rmw_tag_q <= narrow_tag;
                rmw_delta_q <= narrow_acc_wide;
            end

            if (wide_fire && !wide_use_motion) begin
                output_valid_q <= 1'b1;
                output_is_wide <= 1'b1;
                output_context <= wide_context;
                output_base_tile <= wide_base_tile;
                output_bank_mask <= {BANKS{1'b1}};
                output_epoch <= wide_epoch;
                output_domain <= wide_domain;
                output_temporal_step <= wide_temporal_step;
                output_temporal_length <= wide_temporal_length;
                output_temporal_first <= wide_temporal_first;
                output_temporal_last <= wide_temporal_last;
                output_used_motion <= 1'b0;
                output_tag <= wide_tag;
                output_current_acc <= wide_acc;
                for (int bank = 0; bank < BANKS; bank = bank + 1) begin
                    state_valid_q[wide_context][wide_base_tile][bank] <= 1'b1;
                    sequence_open_q[wide_context][wide_base_tile][bank] <=
                        !wide_temporal_last;
                    next_step_q[wide_context][wide_base_tile][bank] <=
                        wide_temporal_step + 1'b1;
                    if (wide_temporal_first) begin
                        state_epoch_q[wide_context][wide_base_tile][bank] <= wide_epoch;
                        state_domain_q[wide_context][wide_base_tile][bank] <= wide_domain;
                        sequence_length_q[wide_context][wide_base_tile][bank] <=
                            wide_temporal_length;
                        sequence_tag_q[wide_context][wide_base_tile][bank] <= wide_tag;
                        epoch_initialized_q[wide_context][wide_base_tile][bank] <= 1'b1;
                    end
                end
            end else if (narrow_fire && !narrow_use_motion) begin
                output_valid_q <= 1'b1;
                output_is_wide <= 1'b0;
                output_context <= narrow_context;
                output_base_tile <= narrow_base_tile;
                output_bank_mask <= BANKS'(1'b1) << narrow_bank;
                output_epoch <= narrow_epoch;
                output_domain <= narrow_domain;
                output_temporal_step <= narrow_temporal_step;
                output_temporal_length <= narrow_temporal_length;
                output_temporal_first <= narrow_temporal_first;
                output_temporal_last <= narrow_temporal_last;
                output_used_motion <= 1'b0;
                output_tag <= narrow_tag;
                output_current_acc[(narrow_bank*BANK_ACC_BITS) +: BANK_ACC_BITS]
                    <= narrow_acc;
                state_valid_q[narrow_context][narrow_base_tile][narrow_bank] <= 1'b1;
                sequence_open_q[narrow_context][narrow_base_tile][narrow_bank] <=
                    !narrow_temporal_last;
                next_step_q[narrow_context][narrow_base_tile][narrow_bank] <=
                    narrow_temporal_step + 1'b1;
                if (narrow_temporal_first) begin
                    state_epoch_q[narrow_context][narrow_base_tile][narrow_bank] <=
                        narrow_epoch;
                    state_domain_q[narrow_context][narrow_base_tile][narrow_bank] <=
                        narrow_domain;
                    sequence_length_q[narrow_context][narrow_base_tile][narrow_bank] <=
                        narrow_temporal_length;
                    sequence_tag_q[narrow_context][narrow_base_tile][narrow_bank] <=
                        narrow_tag;
                    epoch_initialized_q[narrow_context][narrow_base_tile][narrow_bank] <=
                        1'b1;
                end
            end

            if (rmw_commit) begin
                rmw_pending_q <= 1'b0;
                output_valid_q <= 1'b1;
                output_is_wide <= rmw_is_wide_q;
                output_context <= rmw_context_q;
                output_base_tile <= rmw_base_tile_q;
                output_bank_mask <= rmw_bank_mask_q;
                output_epoch <= rmw_epoch_q;
                output_domain <= rmw_domain_q;
                output_temporal_step <= rmw_step_q;
                output_temporal_length <= rmw_length_q;
                output_temporal_first <= rmw_first_q;
                output_temporal_last <= rmw_last_q;
                output_used_motion <= 1'b1;
                output_tag <= rmw_tag_q;
                for (int bank = 0; bank < BANKS; bank = bank + 1) begin
                    if (rmw_bank_mask_q[bank]) begin
                        output_current_acc[(bank*BANK_ACC_BITS) +: BANK_ACC_BITS]
                            <= rmw_result[(bank*BANK_ACC_BITS) +: BANK_ACC_BITS];
                        state_valid_q[rmw_context_q][rmw_base_tile_q][bank] <= 1'b1;
                        sequence_open_q[rmw_context_q][rmw_base_tile_q][bank] <=
                            !rmw_last_q;
                        next_step_q[rmw_context_q][rmw_base_tile_q][bank] <=
                            rmw_step_q + 1'b1;
                        if (rmw_first_q) begin
                            state_epoch_q[rmw_context_q][rmw_base_tile_q][bank] <=
                                rmw_epoch_q;
                            state_domain_q[rmw_context_q][rmw_base_tile_q][bank] <=
                                rmw_domain_q;
                            sequence_length_q[rmw_context_q][rmw_base_tile_q][bank] <=
                                rmw_length_q;
                            sequence_tag_q[rmw_context_q][rmw_base_tile_q][bank] <=
                                rmw_tag_q;
                            epoch_initialized_q[rmw_context_q][rmw_base_tile_q][bank]
                                <= 1'b1;
                        end
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
