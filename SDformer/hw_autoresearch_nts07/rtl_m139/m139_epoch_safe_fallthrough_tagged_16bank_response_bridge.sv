`timescale 1ns/1ps
`default_nettype none

// Reset-epoch-safe wrapper around the unchanged M137 normal data path.  The
// bridge cannot leave recovery until the macro wrapper completes a fresh
// four-phase flush acknowledgement (sampled low, high, then low).  Old drain
// traffic is discarded before acknowledgement; a completion collision or a
// response after completion fails closed.
module m139_epoch_safe_fallthrough_tagged_16bank_response_bridge (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         request_valid,
    output logic                         request_ready,
    input  logic [11:0]                  logical_base_word,
    input  logic                         request_start,
    input  logic                         request_last,
    input  logic [3:0]                   request_width,
    input  logic [31:0]                  request_tag,
    output logic                         request_accept,

    output logic                         macro_flush_req,
    input  logic                         macro_flush_ack,
    output logic                         macro_request_valid,
    output logic [127:0]                 macro_bank_row_addresses,
    output logic [15:0]                  macro_request_token,
    input  logic                         macro_response_valid,
    input  logic [15:0]                  macro_response_token,
    input  logic [511:0]                 macro_bank_words,

    output logic                         response_valid,
    input  logic                         response_ready,
    output logic [511:0]                 response_logical_words,
    output logic                         response_start,
    output logic                         response_last,
    output logic [3:0]                   response_width,
    output logic [31:0]                  response_tag,
    output logic [15:0]                  response_token,
    output logic                         response_accept,

    output logic                         protocol_error,
    output logic                         recovery_active,
    output logic                         pending_response,
    output logic [1:0]                   buffered_responses,
    output logic                         busy
);
    typedef enum logic [1:0] {
        REC_WAIT_ACK_LOW  = 2'd0,
        REC_WAIT_ACK_HIGH = 2'd1,
        REC_WAIT_ACK_DROP = 2'd2,
        REC_RUN           = 2'd3
    } recovery_state_e;

    recovery_state_e recovery_state_q;
    logic wrapper_fault_q;
    logic wrapper_fault_event;
    logic wrapper_quarantine;
    logic normal_run;
    logic bridge_rst;

    logic core_request_ready;
    logic core_request_accept;
    logic core_macro_request_valid;
    logic [127:0] core_macro_bank_row_addresses;
    logic [15:0] core_macro_request_token;
    logic core_response_valid;
    logic [511:0] core_response_logical_words;
    logic core_response_start;
    logic core_response_last;
    logic [3:0] core_response_width;
    logic [31:0] core_response_tag;
    logic [15:0] core_response_token;
    logic core_response_accept;
    logic core_protocol_error;
    logic core_pending_response;
    logic [1:0] core_buffered_responses;
    logic core_busy;

    assign wrapper_fault_event =
        (recovery_state_q == REC_WAIT_ACK_HIGH
         && macro_flush_ack && macro_response_valid)
        || (recovery_state_q == REC_WAIT_ACK_DROP
            && macro_response_valid)
        || (recovery_state_q == REC_RUN && macro_flush_ack);
    assign wrapper_quarantine = wrapper_fault_q || wrapper_fault_event;
    assign normal_run = recovery_state_q == REC_RUN && !wrapper_quarantine;
    assign bridge_rst = rst_core || !normal_run;

    assign macro_flush_req = rst_core
                           || recovery_state_q == REC_WAIT_ACK_LOW
                           || recovery_state_q == REC_WAIT_ACK_HIGH;
    assign recovery_active = !rst_core && recovery_state_q != REC_RUN;

    m137_fallthrough_tagged_16bank_response_bridge bridge (
        .clk_core(clk_core),
        .rst_core(bridge_rst),
        .request_valid(request_valid && normal_run),
        .request_ready(core_request_ready),
        .logical_base_word(logical_base_word),
        .request_start(request_start),
        .request_last(request_last),
        .request_width(request_width),
        .request_tag(request_tag),
        .request_accept(core_request_accept),
        .macro_request_valid(core_macro_request_valid),
        .macro_bank_row_addresses(core_macro_bank_row_addresses),
        .macro_request_token(core_macro_request_token),
        .macro_response_valid(macro_response_valid && normal_run),
        .macro_response_token(macro_response_token),
        .macro_bank_words(macro_bank_words),
        .response_valid(core_response_valid),
        .response_ready(response_ready && normal_run),
        .response_logical_words(core_response_logical_words),
        .response_start(core_response_start),
        .response_last(core_response_last),
        .response_width(core_response_width),
        .response_tag(core_response_tag),
        .response_token(core_response_token),
        .response_accept(core_response_accept),
        .protocol_error(core_protocol_error),
        .pending_response(core_pending_response),
        .buffered_responses(core_buffered_responses),
        .busy(core_busy)
    );

    assign request_ready = normal_run && core_request_ready;
    assign request_accept = normal_run && core_request_accept;
    assign macro_request_valid = normal_run && core_macro_request_valid;
    assign macro_bank_row_addresses = macro_request_valid
                                    ? core_macro_bank_row_addresses : '0;
    assign macro_request_token = macro_request_valid
                               ? core_macro_request_token : '0;

    assign response_valid = normal_run && core_response_valid;
    assign response_logical_words = response_valid
                                  ? core_response_logical_words : '0;
    assign response_start = response_valid && core_response_start;
    assign response_last = response_valid && core_response_last;
    assign response_width = response_valid ? core_response_width : '0;
    assign response_tag = response_valid ? core_response_tag : '0;
    assign response_token = response_valid ? core_response_token : '0;
    assign response_accept = normal_run && core_response_accept;

    assign protocol_error = !rst_core
                          && (wrapper_quarantine
                              || (normal_run && core_protocol_error));
    assign pending_response = normal_run && core_pending_response;
    assign buffered_responses = normal_run ? core_buffered_responses : '0;
    assign busy = !rst_core && (!normal_run || core_busy);

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            recovery_state_q <= REC_WAIT_ACK_LOW;
            wrapper_fault_q <= 1'b0;
        end else begin
            if (wrapper_fault_event)
                wrapper_fault_q <= 1'b1;
            if (!wrapper_fault_q && !wrapper_fault_event) begin
                case (recovery_state_q)
                    REC_WAIT_ACK_LOW: begin
                        if (!macro_flush_ack)
                            recovery_state_q <= REC_WAIT_ACK_HIGH;
                    end
                    REC_WAIT_ACK_HIGH: begin
                        if (macro_flush_ack && !macro_response_valid)
                            recovery_state_q <= REC_WAIT_ACK_DROP;
                    end
                    REC_WAIT_ACK_DROP: begin
                        if (!macro_flush_ack && !macro_response_valid)
                            recovery_state_q <= REC_RUN;
                    end
                    default: recovery_state_q <= REC_RUN;
                endcase
            end
        end
    end
endmodule

`default_nettype wire
