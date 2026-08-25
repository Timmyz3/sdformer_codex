`timescale 1ns/1ps
`default_nettype none

module m166_q8_prefolded_rank3_left_atlif_backend_assertions #(
    parameter int TAG_BITS = 16
) (
    input logic                    clk_core,
    input logic                    rst_core,
    input logic                    config_valid,
    input logic                    config_ready,
    input logic                    config_accept,
    input logic                    rank_valid,
    input logic                    rank_ready,
    input logic                    rank_accept,
    input logic                    event_valid,
    input logic                    event_ready,
    input logic [TAG_BITS-1:0]     event_tag,
    input logic                    event_channel_last,
    input logic [2:0]              event_beat,
    input logic [31:0]             event_bits,
    input logic                    event_accept,
    input logic                    configured,
    input logic                    protocol_error,
    input logic                    busy,
    input logic                    service_active_internal,
    input logic [2:0]              service_phase_internal,
    input logic [1:0]              input_count_internal,
    input logic [4:0]              output_count_internal,
    input logic                    input_push_internal,
    input logic                    input_release_internal,
    input logic                    output_push_internal
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_config_accept_definition:
        assert property (config_accept == (config_valid && config_ready));
    ap_rank_accept_definition:
        assert property (rank_accept == (rank_valid && rank_ready));
    ap_event_accept_definition:
        assert property (event_accept == (event_valid && event_ready));
    ap_fault_sticky:
        assert property (protocol_error |=> protocol_error);
    ap_fault_closes_inputs:
        assert property (protocol_error |=> !config_ready && !rank_ready);
    ap_event_stable_under_stall:
        assert property (event_valid && !event_ready
            |=> event_valid && $stable({event_tag, event_channel_last,
                event_beat, event_bits}));
    ap_input_count_bounded:
        assert property (input_count_internal <= 2);
    ap_output_count_bounded:
        assert property (output_count_internal <= 16);
    ap_service_requires_owned_input:
        assert property (service_active_internal
            |-> input_count_internal != 0 && configured);
    ap_push_matches_service:
        assert property (output_push_internal == service_active_internal);
    ap_nonfinal_phase_advances:
        assert property (service_active_internal && service_phase_internal < 4
            |=> service_active_internal
                && service_phase_internal == $past(service_phase_internal) + 1'b1);
    ap_release_only_on_phase_four:
        assert property (input_release_internal
            |-> service_active_internal && service_phase_internal == 4);
    ap_busy_for_owned_work:
        assert property (service_active_internal || input_count_internal != 0
            || output_count_internal != 0 |-> busy);

    cp_unstalled_five_cycle_tile:
        cover property (output_push_internal && service_phase_internal == 0
            ##1 output_push_internal && service_phase_internal == 1
            ##1 output_push_internal && service_phase_internal == 2
            ##1 output_push_internal && service_phase_internal == 3
            ##1 output_push_internal && service_phase_internal == 4);
    cp_back_to_back_five_cycle_tiles:
        cover property (output_push_internal && service_phase_internal == 4
            ##1 output_push_internal && service_phase_internal == 0);
    cp_input_push_release_same_cycle:
        cover property (input_push_internal && input_release_internal);
    cp_full_owned_input_fifo:
        cover property (service_active_internal && input_count_internal == 2);
    cp_event_stall_then_accept:
        cover property (event_valid && !event_ready ##1 event_valid && event_ready);
    cp_mixed_event_word:
        cover property (event_valid && event_bits != 0 && event_bits != 32'hffff_ffff);
    cp_fault_after_configuration:
        cover property (configured && protocol_error);
endmodule

`default_nettype wire
