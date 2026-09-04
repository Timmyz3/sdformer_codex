`timescale 1ns/1ps
`default_nettype none

// M514 C2-D: exact K3/S2/P1/OP1 ConvTranspose2d address expansion.
//
// The module consumes one binary ATLIF source event and emits only the legal
// source-tap destinations.  It never materializes an inserted-zero image.
// Taps are emitted in phase-major order (4/2/2/1 for interior sources), so the
// two destination parity bits can directly select four psum banks.  This is a
// completeness adapter for the existing C2 signed-source fabric; it does not
// claim a speedup over a strong bit-sparse polyphase baseline.  As with any
// data-dependent-ready interface, the producer must drive and hold the event
// payload independently of event_ready while event_valid is asserted.
module m514_c2_convtranspose_k3s2_polyphase_address_mapper #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int COORD_BITS = 10,
    parameter int TIME_BITS = 4
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

    output logic                         tap_valid,
    input  logic                         tap_ready,
    output logic [TAG_BITS-1:0]          tap_tag,
    output logic [TIME_BITS-1:0]         tap_time,
    output logic [CHANNEL_BITS-1:0]      tap_source_channel,
    output logic [COORD_BITS-1:0]        tap_source_y,
    output logic [COORD_BITS-1:0]        tap_source_x,
    output logic [1:0]                   tap_kernel_y,
    output logic [1:0]                   tap_kernel_x,
    output logic [3:0]                   tap_kernel_index,
    output logic [COORD_BITS-1:0]        tap_destination_y,
    output logic [COORD_BITS-1:0]        tap_destination_x,
    output logic [1:0]                   tap_phase_bank,
    output logic                         tap_last_for_event,
    output logic                         tap_stream_last,
    output logic                         tap_accept,

    output logic                         protocol_error,
    output logic                         busy
);
    logic fault_q, busy_q;
    logic [8:0] pending_q;
    logic [TAG_BITS-1:0] tag_q;
    logic [TIME_BITS-1:0] time_q;
    logic [CHANNEL_BITS-1:0] source_channel_q;
    logic [COORD_BITS-1:0] source_y_q, source_x_q;
    logic [COORD_BITS-1:0] input_height_q, input_width_q;
    logic stream_last_q;

    logic event_legal, event_capacity;
    logic [8:0] event_tap_mask;
    logic [3:0] selected_slot;
    logic [8:0] selected_onehot, pending_after_accept;
    logic selected_found;
    logic [1:0] selected_ky, selected_kx;
    logic [COORD_BITS:0] doubled_y, doubled_x;

    always_comb begin
        event_legal = event_input_height != 0 && event_input_width != 0
            && (!event_input_height[COORD_BITS-1]
                || event_input_height[COORD_BITS-2:0] == 0)
            && (!event_input_width[COORD_BITS-1]
                || event_input_width[COORD_BITS-2:0] == 0)
            && event_source_y < event_input_height
            && event_source_x < event_input_width;

        // Phase-major slot order:
        //   odd/odd: (0,0), (0,2), (2,0), (2,2)
        //   odd/even: (0,1), (2,1)
        //   even/odd: (1,0), (1,2)
        //   even/even: (1,1)
        // ky=0 and kx=0 are clipped only at the top and left boundaries.
        event_tap_mask[0] = event_source_y != 0 && event_source_x != 0;
        event_tap_mask[1] = event_source_y != 0;
        event_tap_mask[2] = event_source_x != 0;
        event_tap_mask[3] = 1'b1;
        event_tap_mask[4] = event_source_y != 0;
        event_tap_mask[5] = 1'b1;
        event_tap_mask[6] = event_source_x != 0;
        event_tap_mask[7] = 1'b1;
        event_tap_mask[8] = 1'b1;
    end

    always_comb begin
        selected_slot = 0;
        selected_onehot = 0;
        selected_found = 0;
        for (int slot = 0; slot < 9; slot++) begin
            if (!selected_found && pending_q[slot]) begin
                selected_slot = slot[3:0];
                selected_onehot[slot] = 1'b1;
                selected_found = 1'b1;
            end
        end
        case (selected_slot)
            0: begin selected_ky = 0; selected_kx = 0; end
            1: begin selected_ky = 0; selected_kx = 2; end
            2: begin selected_ky = 2; selected_kx = 0; end
            3: begin selected_ky = 2; selected_kx = 2; end
            4: begin selected_ky = 0; selected_kx = 1; end
            5: begin selected_ky = 2; selected_kx = 1; end
            6: begin selected_ky = 1; selected_kx = 0; end
            7: begin selected_ky = 1; selected_kx = 2; end
            default: begin selected_ky = 1; selected_kx = 1; end
        endcase
    end

    assign pending_after_accept = pending_q & ~selected_onehot;
    // A protocol attack locks out every successor event, but it must never
    // retract a previously advertised tap.  Accepted work drains to the
    // event boundary under the ordinary ready/valid contract.
    assign tap_valid = busy_q && selected_found;
    assign tap_accept = tap_valid && tap_ready;
    assign tap_last_for_event = tap_valid && pending_after_accept == 0;
    assign event_capacity = !busy_q || (tap_accept && tap_last_for_event);
    assign event_ready = !fault_q && event_capacity && event_legal;
    assign event_accept = event_valid && event_ready;

    assign doubled_y = {1'b0, source_y_q} << 1;
    assign doubled_x = {1'b0, source_x_q} << 1;
    always_comb begin
        case (selected_ky)
            0: tap_destination_y = doubled_y[COORD_BITS-1:0] - 1'b1;
            1: tap_destination_y = doubled_y[COORD_BITS-1:0];
            default: tap_destination_y =
                doubled_y[COORD_BITS-1:0] + 1'b1;
        endcase
        case (selected_kx)
            0: tap_destination_x = doubled_x[COORD_BITS-1:0] - 1'b1;
            1: tap_destination_x = doubled_x[COORD_BITS-1:0];
            default: tap_destination_x =
                doubled_x[COORD_BITS-1:0] + 1'b1;
        endcase
    end

    assign tap_tag = tag_q;
    assign tap_time = time_q;
    assign tap_source_channel = source_channel_q;
    assign tap_source_y = source_y_q;
    assign tap_source_x = source_x_q;
    assign tap_kernel_y = selected_ky;
    assign tap_kernel_x = selected_kx;
    assign tap_kernel_index = (selected_ky * 3) + selected_kx;
    assign tap_phase_bank = {tap_destination_y[0], tap_destination_x[0]};
    assign tap_stream_last = tap_last_for_event && stream_last_q;
    assign protocol_error = fault_q;
    assign busy = busy_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            fault_q <= 1'b0;
            busy_q <= 1'b0;
            pending_q <= 0;
            tag_q <= 0;
            time_q <= 0;
            source_channel_q <= 0;
            source_y_q <= 0;
            source_x_q <= 0;
            input_height_q <= 0;
            input_width_q <= 0;
            stream_last_q <= 1'b0;
        end else begin
            if (event_valid && !event_legal)
                fault_q <= 1'b1;

            if (tap_accept) begin
                pending_q <= pending_after_accept;
                if (tap_last_for_event)
                    busy_q <= 1'b0;
            end

            // A successor event may replace a retiring event on the same
            // edge, removing inter-event bubbles without reordering taps.
            if (event_accept) begin
                busy_q <= 1'b1;
                pending_q <= event_tap_mask;
                tag_q <= event_tag;
                time_q <= event_time;
                source_channel_q <= event_source_channel;
                source_y_q <= event_source_y;
                source_x_q <= event_source_x;
                input_height_q <= event_input_height;
                input_width_q <= event_input_width;
                stream_last_q <= event_last;
            end
        end
    end

`ifndef SYNTHESIS
    initial begin
        if (COORD_BITS < 2 || TAG_BITS < 1 || CHANNEL_BITS < 1
                || TIME_BITS < 1)
            $fatal(1, "M514 illegal parameterization");
    end

    always_ff @(posedge clk_core) begin
        if (!rst_core && tap_valid) begin
            assert (pending_q != 0);
            assert ($onehot(selected_onehot));
            assert ((selected_onehot & pending_q) == selected_onehot);
            assert ({1'b0, tap_destination_y}
                    < ({1'b0, input_height_q} << 1));
            assert ({1'b0, tap_destination_x}
                    < ({1'b0, input_width_q} << 1));
            assert (tap_phase_bank ==
                    {tap_destination_y[0], tap_destination_x[0]});
            assert (tap_kernel_index == tap_kernel_y * 3 + tap_kernel_x);
        end
    end
`endif
endmodule

`default_nettype wire
