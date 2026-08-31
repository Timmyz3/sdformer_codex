`timescale 1ns/1ps
`default_nettype none

// M523 D8: exact K3/S2/P1/OP1 ConvTranspose2d descriptor FIFO and
// cross-event bundled-AER packer.
//
// Each source event is accepted atomically as four, six, or nine exact taps.
// The 18-entry descriptor FIFO decouples expansion from an eight-lane
// descriptor transport.  A bundle may cross an event boundary only when the previous event
// is not stream-last and the successor has identical tag/time.  Every lane
// carries its own source and kernel/destination tuple plus an event-boundary
// marker; a tag/time boundary or stream-last marker flushes a partial bundle.
// Inserted zeros are never materialized.
//
// This is descriptor-only support.  It does not implement M218's eight
// weight-bank contract: no flattened (source-channel,kernel-index) weight key,
// bank-conflict deferral, or stored-weight identity is present.  Therefore it
// does not admit direct C2 integration or decoder/system speedup.
module m523_c2d_k8_polyphase_tap_bundler #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int COORD_BITS = 10,
    parameter int TIME_BITS = 4,
    parameter int BUNDLE_LANES = 8,
    parameter int FIFO_DEPTH = 18
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         event_valid,
    output logic                         event_ready,
    input  logic [TAG_BITS-1:0]          event_tag,
    input  logic [TIME_BITS-1:0]         event_time,
    input  logic [CHANNEL_BITS-1:0]      event_source_channel,
    input  logic [COORD_BITS-1:0]        event_source_y,
    input  logic [COORD_BITS-1:0]        event_source_x,
    input  logic [COORD_BITS-1:0]        event_input_height,
    input  logic [COORD_BITS-1:0]        event_input_width,
    input  logic                         event_last,
    output logic                         event_accept,

    output logic                         bundle_valid,
    input  logic                         bundle_ready,
    output logic [TAG_BITS-1:0]          bundle_tag,
    output logic [TIME_BITS-1:0]         bundle_time,
    output logic [3:0]                   bundle_count,
    output logic [BUNDLE_LANES-1:0]      tap_lane_valid,
    output logic [BUNDLE_LANES-1:0]      tap_event_last,
    output logic [CHANNEL_BITS-1:0]      tap_source_channel [0:BUNDLE_LANES-1],
    output logic [COORD_BITS-1:0]        tap_source_y [0:BUNDLE_LANES-1],
    output logic [COORD_BITS-1:0]        tap_source_x [0:BUNDLE_LANES-1],
    output logic [1:0]                   tap_kernel_y [0:BUNDLE_LANES-1],
    output logic [1:0]                   tap_kernel_x [0:BUNDLE_LANES-1],
    output logic [3:0]                   tap_kernel_index [0:BUNDLE_LANES-1],
    output logic [COORD_BITS-1:0]        tap_destination_y [0:BUNDLE_LANES-1],
    output logic [COORD_BITS-1:0]        tap_destination_x [0:BUNDLE_LANES-1],
    output logic [1:0]                   tap_phase_bank [0:BUNDLE_LANES-1],
    output logic                         bundle_last_for_event,
    output logic                         bundle_stream_last,
    output logic                         bundle_accept,

    output logic                         protocol_error,
    output logic                         busy,
    output logic [31:0]                  debug_event_count,
    output logic [31:0]                  debug_bundle_count,
    output logic [31:0]                  debug_tap_count,
    output logic [5:0]                   debug_fifo_count
);
    localparam int PTR_BITS = $clog2(FIFO_DEPTH);
    localparam int COUNT_BITS = $clog2(FIFO_DEPTH + 1);
    localparam bit PARAMETERS_LEGAL = BUNDLE_LANES == 8
        && FIFO_DEPTH == 18 && TAG_BITS >= 1 && CHANNEL_BITS >= 1
        && COORD_BITS >= 2 && TIME_BITS >= 1;

    logic fault_q;
    logic [PTR_BITS-1:0] head_q, tail_q;
    logic [COUNT_BITS-1:0] count_q;
    logic [31:0] event_count_q, bundle_count_q, tap_count_q;

    logic [TAG_BITS-1:0] fifo_tag [0:FIFO_DEPTH-1];
    logic [TIME_BITS-1:0] fifo_time [0:FIFO_DEPTH-1];
    logic [CHANNEL_BITS-1:0] fifo_source_channel [0:FIFO_DEPTH-1];
    logic [COORD_BITS-1:0] fifo_source_y [0:FIFO_DEPTH-1];
    logic [COORD_BITS-1:0] fifo_source_x [0:FIFO_DEPTH-1];
    logic [1:0] fifo_kernel_y [0:FIFO_DEPTH-1];
    logic [1:0] fifo_kernel_x [0:FIFO_DEPTH-1];
    logic [3:0] fifo_kernel_index [0:FIFO_DEPTH-1];
    logic [COORD_BITS-1:0] fifo_destination_y [0:FIFO_DEPTH-1];
    logic [COORD_BITS-1:0] fifo_destination_x [0:FIFO_DEPTH-1];
    logic [1:0] fifo_phase_bank [0:FIFO_DEPTH-1];
    logic fifo_event_last [0:FIFO_DEPTH-1];
    logic fifo_stream_last [0:FIFO_DEPTH-1];

    logic event_legal;
    logic [8:0] event_tap_mask;
    logic [4:0] event_tap_count;
    logic [COUNT_BITS:0] available_slots;
    logic bundle_flush_allowed;

    function automatic logic [1:0] slot_ky(input logic [3:0] slot);
        case (slot)
            0, 1, 4: slot_ky = 0;
            6, 7, 8: slot_ky = 1;
            default: slot_ky = 2;
        endcase
    endfunction

    function automatic logic [1:0] slot_kx(input logic [3:0] slot);
        case (slot)
            0, 2, 6: slot_kx = 0;
            4, 5, 8: slot_kx = 1;
            default: slot_kx = 2;
        endcase
    endfunction

    function automatic integer ring_index(
        input integer base, input integer offset);
        integer value;
        begin
            value = base + offset;
            if (value >= FIFO_DEPTH)
                value = value - FIFO_DEPTH;
            ring_index = value;
        end
    endfunction

    function automatic logic [COORD_BITS-1:0] event_destination(
        input logic [COORD_BITS-1:0] source,
        input logic [1:0] kernel);
        logic [COORD_BITS:0] doubled;
        begin
            doubled = {1'b0, source} << 1;
            case (kernel)
                0: event_destination = doubled[COORD_BITS-1:0] - 1'b1;
                1: event_destination = doubled[COORD_BITS-1:0];
                default:
                    event_destination = doubled[COORD_BITS-1:0] + 1'b1;
            endcase
        end
    endfunction

    generate
        if (!PARAMETERS_LEGAL) begin : g_illegal_parameters
            initial $fatal(1, "M523 requires BUNDLE_LANES=8 FIFO_DEPTH=18 and legal widths");
        end
    endgenerate

    always_comb begin
        event_legal = PARAMETERS_LEGAL
            && event_input_height != 0 && event_input_width != 0
            && (!event_input_height[COORD_BITS-1]
                || event_input_height[COORD_BITS-2:0] == 0)
            && (!event_input_width[COORD_BITS-1]
                || event_input_width[COORD_BITS-2:0] == 0)
            && event_source_y < event_input_height
            && event_source_x < event_input_width;

        // M514 phase-major order: 4 odd/odd, 2 odd/even,
        // 2 even/odd, and 1 even/even tap.
        event_tap_mask[0] = event_source_y != 0 && event_source_x != 0;
        event_tap_mask[1] = event_source_y != 0;
        event_tap_mask[2] = event_source_x != 0;
        event_tap_mask[3] = 1'b1;
        event_tap_mask[4] = event_source_y != 0;
        event_tap_mask[5] = 1'b1;
        event_tap_mask[6] = event_source_x != 0;
        event_tap_mask[7] = 1'b1;
        event_tap_mask[8] = 1'b1;
        event_tap_count = $countones(event_tap_mask);
    end

    always_comb begin : bundle_selection
        integer index_value;
        integer next_index_value;
        integer selected_count;
        logic selection_open;

        bundle_tag = 0;
        bundle_time = 0;
        bundle_count = 0;
        tap_lane_valid = 0;
        tap_event_last = 0;
        bundle_last_for_event = 1'b0;
        bundle_stream_last = 1'b0;
        bundle_flush_allowed = 1'b0;
        for (int lane = 0; lane < BUNDLE_LANES; lane++) begin
            tap_source_channel[lane] = 0;
            tap_source_y[lane] = 0;
            tap_source_x[lane] = 0;
            tap_kernel_y[lane] = 0;
            tap_kernel_x[lane] = 0;
            tap_kernel_index[lane] = 0;
            tap_destination_y[lane] = 0;
            tap_destination_x[lane] = 0;
            tap_phase_bank[lane] = 0;
        end

        selected_count = 0;
        selection_open = 1'b1;
        for (int lane = 0; lane < BUNDLE_LANES; lane++) begin
            if (selection_open && lane < count_q) begin
                index_value = ring_index(head_q, lane);
                tap_lane_valid[lane] = 1'b1;
                tap_event_last[lane] = fifo_event_last[index_value];
                tap_source_channel[lane] = fifo_source_channel[index_value];
                tap_source_y[lane] = fifo_source_y[index_value];
                tap_source_x[lane] = fifo_source_x[index_value];
                tap_kernel_y[lane] = fifo_kernel_y[index_value];
                tap_kernel_x[lane] = fifo_kernel_x[index_value];
                tap_kernel_index[lane] = fifo_kernel_index[index_value];
                tap_destination_y[lane] = fifo_destination_y[index_value];
                tap_destination_x[lane] = fifo_destination_x[index_value];
                tap_phase_bank[lane] = fifo_phase_bank[index_value];
                if (lane == 0) begin
                    bundle_tag = fifo_tag[index_value];
                    bundle_time = fifo_time[index_value];
                end
                selected_count = selected_count + 1;
                bundle_last_for_event = fifo_event_last[index_value];
                bundle_stream_last = fifo_stream_last[index_value];

                if (fifo_event_last[index_value]) begin
                    if (fifo_stream_last[index_value]) begin
                        bundle_flush_allowed = 1'b1;
                        selection_open = 1'b0;
                    end else if ((lane + 1) >= count_q) begin
                        // A nonterminal event tail waits for its successor so
                        // that it can be packed.  A sticky fault is the only
                        // legal reason to flush when no successor can arrive.
                        bundle_flush_allowed = fault_q;
                        selection_open = 1'b0;
                    end else begin
                        next_index_value = ring_index(head_q, lane + 1);
                        if (fifo_tag[next_index_value]
                                != fifo_tag[index_value]
                                || fifo_time[next_index_value]
                                != fifo_time[index_value]) begin
                            bundle_flush_allowed = 1'b1;
                            selection_open = 1'b0;
                        end
                    end
                end
            end
        end
        bundle_count = selected_count[3:0];
    end

    assign bundle_valid = count_q != 0
        && (bundle_count == BUNDLE_LANES || bundle_flush_allowed);
    assign bundle_accept = bundle_valid && bundle_ready;
    assign available_slots = FIFO_DEPTH - count_q
        + (bundle_accept ? bundle_count : 0);
    assign event_ready = !fault_q && event_legal
        && available_slots >= event_tap_count;
    assign event_accept = event_valid && event_ready;
    assign protocol_error = fault_q;
    assign busy = count_q != 0;
    assign debug_event_count = event_count_q;
    assign debug_bundle_count = bundle_count_q;
    assign debug_tap_count = tap_count_q;
    assign debug_fifo_count = count_q;

    always_ff @(posedge clk_core) begin : fifo_update
        integer fill;
        integer write_index;
        logic [1:0] ky;
        logic [1:0] kx;
        logic [COORD_BITS-1:0] dy;
        logic [COORD_BITS-1:0] dx;
        if (rst_core) begin
            fault_q <= 1'b0;
            head_q <= 0;
            tail_q <= 0;
            count_q <= 0;
            event_count_q <= 0;
            bundle_count_q <= 0;
            tap_count_q <= 0;
        end else begin
            if (event_valid && !event_legal)
                fault_q <= 1'b1;

            case ({event_accept, bundle_accept})
                2'b10: count_q <= count_q + event_tap_count;
                2'b01: count_q <= count_q - bundle_count;
                2'b11: count_q <= count_q - bundle_count
                    + event_tap_count;
                default: count_q <= count_q;
            endcase

            if (bundle_accept) begin
                head_q <= ring_index(head_q, bundle_count);
                bundle_count_q <= bundle_count_q + 1'b1;
                tap_count_q <= tap_count_q + bundle_count;
            end

            if (event_accept) begin
                fill = 0;
                for (int slot = 0; slot < 9; slot++) begin
                    if (event_tap_mask[slot]) begin
                        write_index = ring_index(tail_q, fill);
                        ky = slot_ky(slot[3:0]);
                        kx = slot_kx(slot[3:0]);
                        dy = event_destination(event_source_y, ky);
                        dx = event_destination(event_source_x, kx);
                        fifo_tag[write_index] <= event_tag;
                        fifo_time[write_index] <= event_time;
                        fifo_source_channel[write_index]
                            <= event_source_channel;
                        fifo_source_y[write_index] <= event_source_y;
                        fifo_source_x[write_index] <= event_source_x;
                        fifo_kernel_y[write_index] <= ky;
                        fifo_kernel_x[write_index] <= kx;
                        fifo_kernel_index[write_index] <= (ky * 3) + kx;
                        fifo_destination_y[write_index] <= dy;
                        fifo_destination_x[write_index] <= dx;
                        fifo_phase_bank[write_index] <= {dy[0], dx[0]};
                        fifo_event_last[write_index]
                            <= fill == event_tap_count - 1;
                        fifo_stream_last[write_index]
                            <= event_last && fill == event_tap_count - 1;
                        fill = fill + 1;
                    end
                end
                tail_q <= ring_index(tail_q, event_tap_count);
                event_count_q <= event_count_q + 1'b1;
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk_core) begin
        if (!rst_core) begin
            assert (count_q <= FIFO_DEPTH);
            if (event_accept)
                assert (available_slots >= event_tap_count);
            if (bundle_valid) begin
                assert (bundle_count >= 1
                    && bundle_count <= BUNDLE_LANES);
                assert ($countones(tap_lane_valid) == bundle_count);
                assert (!bundle_stream_last || bundle_last_for_event);
                for (int lane = 0; lane < BUNDLE_LANES; lane++) begin
                    if (tap_lane_valid[lane]) begin
                        assert (tap_phase_bank[lane] == {
                            tap_destination_y[lane][0],
                            tap_destination_x[lane][0]});
                    end
                end
            end
        end
    end
`endif
endmodule

`default_nettype wire
