`timescale 1ns/1ps
`default_nettype none

module qfit_local_banked_multisource_engine_assertions #(
    parameter int ISSUE_WIDTH = 4,
    parameter int BANK_ADDR_W = 6,
    parameter int TAG_W = 32,
    parameter int COUNT_W = 9,
    parameter int OUT_LANES = 16,
    parameter int ACC_W = 32
) (
    input logic clk_core,
    input logic rst_core,
    input logic command_valid,
    input logic command_ready,
    input logic weight_request_valid,
    input logic weight_request_ready,
    input logic [ISSUE_WIDTH-1:0] weight_request_bank_valid,
    input logic [ISSUE_WIDTH*BANK_ADDR_W-1:0] weight_request_bank_addr,
    input logic weight_request_last,
    input logic weight_response_valid,
    input logic weight_response_ready,
    input logic output_valid,
    input logic output_ready,
    input logic [TAG_W-1:0] output_tag,
    input logic [COUNT_W-1:0] output_source_count,
    input logic [OUT_LANES*ACC_W-1:0] output_acc,
    input logic protocol_error
);
    property request_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
            weight_request_valid && !weight_request_ready |=>
                weight_request_valid && $stable({weight_request_bank_valid,
                    weight_request_bank_addr, weight_request_last});
    endproperty
    assert property (request_stable_under_stall);

    property output_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
            output_valid && !output_ready |=> output_valid
                && $stable({output_tag, output_source_count, output_acc});
    endproperty
    assert property (output_stable_under_stall);

    property request_has_source;
        @(posedge clk_core) disable iff (rst_core)
            weight_request_valid |-> |weight_request_bank_valid;
    endproperty
    assert property (request_has_source);

    property last_requires_request;
        @(posedge clk_core) disable iff (rst_core)
            weight_request_last |-> weight_request_valid;
    endproperty
    assert property (last_requires_request);

    property unsolicited_response_faults;
        @(posedge clk_core) disable iff (rst_core)
            weight_response_valid && !weight_response_ready |=> protocol_error;
    endproperty
    assert property (unsolicited_response_faults);

    property reset_clears_visible_state;
        @(posedge clk_core)
            rst_core |=> !protocol_error && !output_valid && !weight_request_valid;
    endproperty
    assert property (reset_clears_visible_state);

    property fault_is_sticky_and_closed;
        @(posedge clk_core) disable iff (rst_core)
            protocol_error |=> protocol_error && !command_ready && !output_valid;
    endproperty
    assert property (fault_is_sticky_and_closed);

    cover property (@(posedge clk_core) disable iff (rst_core)
        weight_request_valid && weight_request_ready
        && weight_response_valid && weight_response_ready);
    cover property (@(posedge clk_core) disable iff (rst_core)
        output_valid && !output_ready);

endmodule

bind qfit_local_banked_multisource_engine
    qfit_local_banked_multisource_engine_assertions #(
        .ISSUE_WIDTH(ISSUE_WIDTH), .BANK_ADDR_W(BANK_ADDR_W), .TAG_W(TAG_W),
        .COUNT_W(COUNT_W), .OUT_LANES(OUT_LANES), .ACC_W(ACC_W)
    ) u_qfit_local_banked_multisource_engine_assertions (
        .clk_core, .rst_core, .command_valid, .command_ready,
        .weight_request_valid, .weight_request_ready,
        .weight_request_bank_valid, .weight_request_bank_addr, .weight_request_last,
        .weight_response_valid, .weight_response_ready,
        .output_valid, .output_ready, .output_tag, .output_source_count, .output_acc,
        .protocol_error
    );

`default_nettype wire
