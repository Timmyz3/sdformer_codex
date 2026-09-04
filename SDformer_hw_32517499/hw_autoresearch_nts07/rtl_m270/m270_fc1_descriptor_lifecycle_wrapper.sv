`timescale 1ns/1ps
`default_nettype none

// M270 is the fail-closed correction of the M262 small-width FC1 lifecycle.
// It deliberately uses a small eight-output-lane datapath, but every storage
// interaction is an explicit ready/valid port: factor metadata, INT8 weights,
// Acc19 read/write, final commit and abort.  Only one descriptor is in flight;
// this milestone establishes fail-closed transaction semantics rather than a
// full-width throughput implementation.
module m270_fc1_descriptor_lifecycle_wrapper #(
    parameter int LANES=8,
    parameter int CONTEXTS=8,
    parameter int ACC_BITS=19,
    parameter int TAG_BITS=24,
    parameter int EPOCH_BITS=16,
    parameter int DESC_BITS=12,
    parameter int FACTOR_ADDR_BITS=20
)(
    input  logic clk_core,
    input  logic rst_core,

    input  logic header_valid,
    output logic header_ready,
    output logic header_accept,
    input  logic[TAG_BITS-1:0] header_tag,
    input  logic[EPOCH_BITS-1:0] header_epoch,
    input  logic[1:0] header_mode,
    input  logic[DESC_BITS-1:0] header_descriptor_count,
    input  logic[FACTOR_ADDR_BITS-1:0] header_factor_base,

    output logic factor_req_valid,
    input  logic factor_req_ready,
    output logic factor_req_accept,
    output logic[TAG_BITS-1:0] factor_req_tag,
    output logic[EPOCH_BITS-1:0] factor_req_epoch,
    output logic[DESC_BITS-1:0] factor_req_descriptor,
    output logic[FACTOR_ADDR_BITS-1:0] factor_req_addr,
    input  logic factor_rsp_valid,
    output logic factor_rsp_ready,
    output logic factor_rsp_accept,
    input  logic[TAG_BITS-1:0] factor_rsp_tag,
    input  logic[EPOCH_BITS-1:0] factor_rsp_epoch,
    input  logic[DESC_BITS-1:0] factor_rsp_descriptor,
    input  logic[FACTOR_ADDR_BITS-1:0] factor_rsp_addr,
    input  logic[8:0] factor_rsp_source,
    input  logic[CONTEXTS-1:0] factor_rsp_context_mask,
    input  logic[CONTEXTS-1:0] factor_rsp_sign_mask,
    input  logic factor_rsp_zero,
    input  logic factor_rsp_last,

    output logic weight_req_valid,
    input  logic weight_req_ready,
    output logic weight_req_accept,
    output logic[TAG_BITS-1:0] weight_req_tag,
    output logic[EPOCH_BITS-1:0] weight_req_epoch,
    output logic[DESC_BITS-1:0] weight_req_descriptor,
    output logic[8:0] weight_req_source,
    input  logic weight_rsp_valid,
    output logic weight_rsp_ready,
    output logic weight_rsp_accept,
    input  logic[TAG_BITS-1:0] weight_rsp_tag,
    input  logic[EPOCH_BITS-1:0] weight_rsp_epoch,
    input  logic[DESC_BITS-1:0] weight_rsp_descriptor,
    input  logic[8:0] weight_rsp_source,
    input  logic[LANES*8-1:0] weight_rsp_data,

    output logic acc_read_req_valid,
    input  logic acc_read_req_ready,
    output logic acc_read_req_accept,
    output logic[TAG_BITS-1:0] acc_read_req_tag,
    output logic[EPOCH_BITS-1:0] acc_read_req_epoch,
    output logic[DESC_BITS-1:0] acc_read_req_descriptor,
    output logic[$clog2(CONTEXTS)-1:0] acc_read_req_context,
    output logic acc_read_req_commit,
    input  logic acc_read_rsp_valid,
    output logic acc_read_rsp_ready,
    output logic acc_read_rsp_accept,
    input  logic[TAG_BITS-1:0] acc_read_rsp_tag,
    input  logic[EPOCH_BITS-1:0] acc_read_rsp_epoch,
    input  logic[DESC_BITS-1:0] acc_read_rsp_descriptor,
    input  logic[$clog2(CONTEXTS)-1:0] acc_read_rsp_context,
    input  logic acc_read_rsp_commit,
    input  logic[LANES*ACC_BITS-1:0] acc_read_rsp_data,

    output logic acc_write_valid,
    input  logic acc_write_ready,
    output logic acc_write_accept,
    output logic[TAG_BITS-1:0] acc_write_tag,
    output logic[EPOCH_BITS-1:0] acc_write_epoch,
    output logic[DESC_BITS-1:0] acc_write_descriptor,
    output logic[$clog2(CONTEXTS)-1:0] acc_write_context,
    output logic acc_write_update,
    output logic[LANES*ACC_BITS-1:0] acc_write_data,

    output logic commit_valid,
    input  logic commit_ready,
    output logic commit_accept,
    output logic[TAG_BITS-1:0] commit_tag,
    output logic[EPOCH_BITS-1:0] commit_epoch,
    output logic[$clog2(CONTEXTS)-1:0] commit_context,
    output logic commit_last,
    output logic[LANES*ACC_BITS-1:0] commit_data,

    output logic abort_valid,
    input  logic abort_ready,
    output logic abort_accept,
    output logic[TAG_BITS-1:0] abort_tag,
    output logic[EPOCH_BITS-1:0] abort_epoch,
    output logic[3:0] abort_reason,

    output logic done_valid,
    input  logic done_ready,
    output logic done_accept,
    output logic[TAG_BITS-1:0] done_tag,
    output logic[EPOCH_BITS-1:0] done_epoch,
    output logic done_empty_bypass,

    output logic descriptor_retire_valid,
    output logic[DESC_BITS-1:0] descriptor_retire_index,
    output logic[15:0] descriptor_retire_cycles,
    output logic protocol_error,
    output logic numeric_overflow,
    output logic busy,
    output logic[31:0] debug_tile_cycles,
    output logic[31:0] debug_factor_request_count,
    output logic[31:0] debug_weight_request_count,
    output logic[31:0] debug_acc_read_count,
    output logic[31:0] debug_acc_write_count,
    output logic[31:0] debug_commit_count,
    output logic[31:0] debug_empty_bypass_count,
    output logic[31:0] debug_abort_count
);
    localparam int CTX_BITS=$clog2(CONTEXTS);
    localparam logic[1:0] MODE_DENSE=2'd0;
    localparam logic[1:0] MODE_BIT_SPARSE=2'd1;
    localparam logic[1:0] MODE_FACTORIZED=2'd2;
    localparam logic[3:0] ABORT_PROTOCOL=4'd1;
    localparam logic[3:0] ABORT_FACTOR=4'd2;
    localparam logic[3:0] ABORT_WEIGHT=4'd3;
    localparam logic[3:0] ABORT_ACC=4'd4;
    localparam logic[3:0] ABORT_OVERFLOW=4'd5;
    localparam bit PARAMETERS_LEGAL=LANES==8&&CONTEXTS==8&&ACC_BITS==19
        &&TAG_BITS==24&&EPOCH_BITS==16&&DESC_BITS==12;

    typedef enum logic[4:0] {
        ST_IDLE, ST_INIT_WRITE, ST_FACTOR_REQ, ST_FACTOR_WAIT,
        ST_WEIGHT_REQ, ST_WEIGHT_WAIT, ST_ACC_READ_REQ,
        ST_ACC_READ_WAIT, ST_ACC_WRITE, ST_COMMIT_READ_REQ,
        ST_COMMIT_READ_WAIT, ST_COMMIT_SEND, ST_DONE, ST_ABORT, ST_FAULT
    } state_t;
    state_t state_q;

    logic[TAG_BITS-1:0] tag_q;
    logic[EPOCH_BITS-1:0] epoch_q;
    logic[1:0] mode_q;
    logic[DESC_BITS-1:0] descriptor_count_q,descriptor_index_q;
    logic[FACTOR_ADDR_BITS-1:0] factor_base_q;
    logic[CTX_BITS-1:0] init_context_q,active_context,commit_context_q;
    logic[8:0] source_q;
    logic[CONTEXTS-1:0] pending_context_q,sign_mask_q;
    logic factor_zero_q;
    logic[LANES*8-1:0] weight_q;
    logic[LANES*ACC_BITS-1:0] acc_read_q,update_data;
    logic[15:0] descriptor_cycles_q;
    logic[31:0] tile_cycles_q,factor_request_count_q,weight_request_count_q;
    logic[31:0] acc_read_count_q,acc_write_count_q,commit_count_q;
    logic[31:0] empty_bypass_count_q,abort_count_q;
    logic fault_q,overflow_seen_q;
    logic[3:0] abort_reason_q;
    logic factor_identity_legal,factor_shape_legal,weight_identity_legal;
    logic acc_identity_legal,header_shape_legal,header_address_legal;
    logic[FACTOR_ADDR_BITS:0] header_last_addr_ext;
    logic illegal_header,illegal_factor_response,illegal_weight_response;
    logic illegal_acc_response,overflow_this_cycle,fault_event;
    logic[3:0] fault_reason;
    logic descriptor_state;

    generate if(!PARAMETERS_LEGAL)begin:g_bad_parameters
        initial $fatal(1,"M262 frozen small-width geometry drift");
    end endgenerate

    always_comb begin:select_context
        active_context='0;
        for(int ctx=CONTEXTS-1;ctx>=0;ctx--)
            if(pending_context_q[ctx])active_context=ctx[CTX_BITS-1:0];
    end

    always_comb begin:numeric_update
        update_data=acc_read_q;
        overflow_this_cycle=1'b0;
        if(state_q==ST_ACC_WRITE&&!fault_q)begin
            for(int lane=0;lane<LANES;lane++)begin
                logic signed[ACC_BITS-1:0] current_value,weight_value;
                logic signed[ACC_BITS:0] sum_ext;
                current_value=acc_read_q[lane*ACC_BITS+:ACC_BITS];
                weight_value={{(ACC_BITS-8){weight_q[lane*8+7]}},
                              weight_q[lane*8+:8]};
                if(factor_zero_q)sum_ext=$signed(current_value);
                else if(sign_mask_q[active_context])
                    sum_ext=$signed(current_value)-$signed(weight_value);
                else sum_ext=$signed(current_value)+$signed(weight_value);
                update_data[lane*ACC_BITS+:ACC_BITS]=sum_ext[ACC_BITS-1:0];
                if(sum_ext[ACC_BITS]!=sum_ext[ACC_BITS-1])
                    overflow_this_cycle=1'b1;
            end
        end
    end

    always_comb begin:legality
        header_last_addr_ext={1'b0,header_factor_base};
        if(header_descriptor_count!=0)
            header_last_addr_ext={1'b0,header_factor_base}
                +header_descriptor_count-1'b1;
        header_address_legal=header_descriptor_count==0
            ||!header_last_addr_ext[FACTOR_ADDR_BITS];
        header_shape_legal=header_mode<=MODE_FACTORIZED
            &&header_descriptor_count<=3072&&header_address_legal;
        factor_identity_legal=state_q==ST_FACTOR_WAIT
            &&factor_rsp_tag==tag_q&&factor_rsp_epoch==epoch_q
            &&factor_rsp_descriptor==descriptor_index_q
            &&factor_rsp_addr==factor_base_q+descriptor_index_q;
        factor_shape_legal=factor_rsp_source<384
            &&factor_rsp_context_mask!=0
            &&!(|(factor_rsp_sign_mask&~factor_rsp_context_mask))
            &&factor_rsp_last==(descriptor_index_q==descriptor_count_q-1'b1)
            &&((mode_q==MODE_DENSE)
                ?$onehot(factor_rsp_context_mask)
                :((mode_q==MODE_BIT_SPARSE)
                    ?($onehot(factor_rsp_context_mask)&&!factor_rsp_zero)
                    :(!factor_rsp_zero)));
        weight_identity_legal=state_q==ST_WEIGHT_WAIT
            &&weight_rsp_tag==tag_q&&weight_rsp_epoch==epoch_q
            &&weight_rsp_descriptor==descriptor_index_q
            &&weight_rsp_source==source_q;
        acc_identity_legal=(state_q==ST_ACC_READ_WAIT
                ||state_q==ST_COMMIT_READ_WAIT)
            &&acc_read_rsp_tag==tag_q&&acc_read_rsp_epoch==epoch_q
            &&acc_read_rsp_descriptor==descriptor_index_q
            &&acc_read_rsp_context==((state_q==ST_ACC_READ_WAIT)
                                     ?active_context:commit_context_q)
            &&acc_read_rsp_commit==(state_q==ST_COMMIT_READ_WAIT);
        // An invalid header is consumed as a fail-closed protocol fault even
        // though header_ready remains low.  This avoids permanent IDLE
        // backpressure and preserves its tag/epoch on the abort channel.
        illegal_header=header_valid
            &&(state_q!=ST_IDLE||!header_shape_legal);
        illegal_factor_response=factor_rsp_valid
            &&!(factor_identity_legal&&factor_shape_legal);
        illegal_weight_response=weight_rsp_valid&&!weight_identity_legal;
        illegal_acc_response=acc_read_rsp_valid&&!acc_identity_legal;
        // Once abort is visible, the original stale responder may legally keep
        // valid asserted while ready is low.  Do not let that held response
        // starve the abort handshake or retrigger a different reason.
        fault_event=state_q!=ST_ABORT&&state_q!=ST_FAULT
            &&(illegal_header||illegal_factor_response
               ||illegal_weight_response||illegal_acc_response
               ||overflow_this_cycle);
        fault_reason=ABORT_PROTOCOL;
        if(illegal_factor_response)fault_reason=ABORT_FACTOR;
        else if(illegal_weight_response)fault_reason=ABORT_WEIGHT;
        else if(illegal_acc_response)fault_reason=ABORT_ACC;
        else if(overflow_this_cycle)fault_reason=ABORT_OVERFLOW;
    end

    always_comb begin:ports
        header_ready=state_q==ST_IDLE&&!fault_q
            &&header_shape_legal
            &&(!header_valid||header_descriptor_count!=0||done_ready);
        header_accept=header_valid&&header_ready;

        factor_req_valid=state_q==ST_FACTOR_REQ&&!fault_q;
        factor_req_tag=tag_q;factor_req_epoch=epoch_q;
        factor_req_descriptor=descriptor_index_q;
        factor_req_addr=factor_base_q+descriptor_index_q;
        factor_req_accept=factor_req_valid&&factor_req_ready;
        factor_rsp_ready=state_q==ST_FACTOR_WAIT&&!fault_q
            &&!illegal_factor_response;
        factor_rsp_accept=factor_rsp_valid&&factor_rsp_ready;

        weight_req_valid=state_q==ST_WEIGHT_REQ&&!fault_q;
        weight_req_tag=tag_q;weight_req_epoch=epoch_q;
        weight_req_descriptor=descriptor_index_q;weight_req_source=source_q;
        weight_req_accept=weight_req_valid&&weight_req_ready;
        weight_rsp_ready=state_q==ST_WEIGHT_WAIT&&!fault_q
            &&!illegal_weight_response;
        weight_rsp_accept=weight_rsp_valid&&weight_rsp_ready;

        acc_read_req_valid=(state_q==ST_ACC_READ_REQ
                            ||state_q==ST_COMMIT_READ_REQ)&&!fault_q;
        acc_read_req_tag=tag_q;acc_read_req_epoch=epoch_q;
        acc_read_req_descriptor=descriptor_index_q;
        acc_read_req_context=(state_q==ST_COMMIT_READ_REQ)
            ?commit_context_q:active_context;
        acc_read_req_commit=state_q==ST_COMMIT_READ_REQ;
        acc_read_req_accept=acc_read_req_valid&&acc_read_req_ready;
        acc_read_rsp_ready=(state_q==ST_ACC_READ_WAIT
                            ||state_q==ST_COMMIT_READ_WAIT)
            &&!fault_q&&!illegal_acc_response;
        acc_read_rsp_accept=acc_read_rsp_valid&&acc_read_rsp_ready;

        acc_write_valid=(state_q==ST_INIT_WRITE
                         ||state_q==ST_ACC_WRITE)
            &&!fault_q&&!overflow_this_cycle;
        acc_write_tag=tag_q;acc_write_epoch=epoch_q;
        acc_write_descriptor=(state_q==ST_INIT_WRITE)?'0:descriptor_index_q;
        acc_write_context=(state_q==ST_INIT_WRITE)?init_context_q:active_context;
        acc_write_update=state_q==ST_ACC_WRITE;
        acc_write_data=(state_q==ST_INIT_WRITE)?'0:update_data;
        acc_write_accept=acc_write_valid&&acc_write_ready;

        commit_valid=state_q==ST_COMMIT_SEND&&!fault_q;
        commit_tag=tag_q;commit_epoch=epoch_q;commit_context=commit_context_q;
        commit_last=commit_context_q==CONTEXTS-1;
        commit_data=acc_read_q;commit_accept=commit_valid&&commit_ready;

        abort_valid=state_q==ST_ABORT;
        abort_tag=tag_q;abort_epoch=epoch_q;abort_reason=abort_reason_q;
        abort_accept=abort_valid&&abort_ready;

        done_valid=(state_q==ST_DONE&&!fault_q)
            ||(state_q==ST_IDLE&&!fault_q&&header_valid
               &&header_shape_legal&&header_descriptor_count==0);
        done_tag=(state_q==ST_DONE)?tag_q:header_tag;
        done_epoch=(state_q==ST_DONE)?epoch_q:header_epoch;
        done_empty_bypass=state_q==ST_IDLE;
        done_accept=done_valid&&done_ready;

        protocol_error=fault_q||fault_event;
        numeric_overflow=overflow_seen_q||overflow_this_cycle;
        busy=state_q!=ST_IDLE;
        debug_tile_cycles=tile_cycles_q;
        debug_factor_request_count=factor_request_count_q;
        debug_weight_request_count=weight_request_count_q;
        debug_acc_read_count=acc_read_count_q;
        debug_acc_write_count=acc_write_count_q;
        debug_commit_count=commit_count_q;
        debug_empty_bypass_count=empty_bypass_count_q;
        debug_abort_count=abort_count_q;
        descriptor_state=state_q==ST_FACTOR_REQ||state_q==ST_FACTOR_WAIT
            ||state_q==ST_WEIGHT_REQ||state_q==ST_WEIGHT_WAIT
            ||state_q==ST_ACC_READ_REQ||state_q==ST_ACC_READ_WAIT
            ||state_q==ST_ACC_WRITE;
    end

    always_ff @(posedge clk_core)begin
        if(rst_core)begin
            state_q<=ST_IDLE;tag_q<='0;epoch_q<='0;mode_q<=MODE_DENSE;
            descriptor_count_q<='0;descriptor_index_q<='0;factor_base_q<='0;
            init_context_q<='0;commit_context_q<='0;source_q<='0;
            pending_context_q<='0;sign_mask_q<='0;factor_zero_q<=1'b0;
            weight_q<='0;acc_read_q<='0;descriptor_cycles_q<='0;
            tile_cycles_q<='0;factor_request_count_q<='0;
            weight_request_count_q<='0;acc_read_count_q<='0;
            acc_write_count_q<='0;commit_count_q<='0;
            empty_bypass_count_q<='0;abort_count_q<='0;
            descriptor_retire_valid<=1'b0;descriptor_retire_index<='0;
            descriptor_retire_cycles<='0;fault_q<=1'b0;
            overflow_seen_q<=1'b0;abort_reason_q<='0;
        end else begin
            descriptor_retire_valid<=1'b0;
            if(state_q!=ST_IDLE&&state_q!=ST_FAULT)
                tile_cycles_q<=tile_cycles_q+1'b1;
            if(descriptor_state)descriptor_cycles_q<=descriptor_cycles_q+1'b1;
            if(fault_event)begin
                if(illegal_header&&state_q==ST_IDLE)begin
                    tag_q<=header_tag;
                    epoch_q<=header_epoch;
                end
                fault_q<=1'b1;
                if(overflow_this_cycle)overflow_seen_q<=1'b1;
                abort_reason_q<=fault_reason;
                state_q<=ST_ABORT;
            end else begin
                case(state_q)
                    ST_IDLE:if(header_accept)begin
                        if(header_descriptor_count==0)begin
                            empty_bypass_count_q<=empty_bypass_count_q+1'b1;
                            tile_cycles_q<=1;
                        end else begin
                            tag_q<=header_tag;epoch_q<=header_epoch;
                            mode_q<=header_mode;
                            descriptor_count_q<=header_descriptor_count;
                            descriptor_index_q<='0;factor_base_q<=header_factor_base;
                            init_context_q<='0;commit_context_q<='0;
                            descriptor_cycles_q<='0;tile_cycles_q<=1;
                            factor_request_count_q<='0;weight_request_count_q<='0;
                            acc_read_count_q<='0;acc_write_count_q<='0;
                            commit_count_q<='0;state_q<=ST_INIT_WRITE;
                        end
                    end
                    ST_INIT_WRITE:if(acc_write_accept)begin
                        acc_write_count_q<=acc_write_count_q+1'b1;
                        if(init_context_q==CONTEXTS-1)begin
                            descriptor_cycles_q<='0;state_q<=ST_FACTOR_REQ;
                        end else init_context_q<=init_context_q+1'b1;
                    end
                    ST_FACTOR_REQ:if(factor_req_accept)begin
                        factor_request_count_q<=factor_request_count_q+1'b1;
                        state_q<=ST_FACTOR_WAIT;
                    end
                    ST_FACTOR_WAIT:if(factor_rsp_accept)begin
                        source_q<=factor_rsp_source;
                        pending_context_q<=factor_rsp_context_mask;
                        sign_mask_q<=factor_rsp_sign_mask;
                        factor_zero_q<=factor_rsp_zero;
                        state_q<=ST_WEIGHT_REQ;
                    end
                    ST_WEIGHT_REQ:if(weight_req_accept)begin
                        weight_request_count_q<=weight_request_count_q+1'b1;
                        state_q<=ST_WEIGHT_WAIT;
                    end
                    ST_WEIGHT_WAIT:if(weight_rsp_accept)begin
                        weight_q<=weight_rsp_data;state_q<=ST_ACC_READ_REQ;
                    end
                    ST_ACC_READ_REQ:if(acc_read_req_accept)begin
                        acc_read_count_q<=acc_read_count_q+1'b1;
                        state_q<=ST_ACC_READ_WAIT;
                    end
                    ST_ACC_READ_WAIT:if(acc_read_rsp_accept)begin
                        acc_read_q<=acc_read_rsp_data;state_q<=ST_ACC_WRITE;
                    end
                    ST_ACC_WRITE:if(acc_write_accept)begin
                        logic[CONTEXTS-1:0] remaining;
                        remaining=pending_context_q;
                        remaining[active_context]=1'b0;
                        acc_write_count_q<=acc_write_count_q+1'b1;
                        pending_context_q<=remaining;
                        if(remaining!=0)state_q<=ST_ACC_READ_REQ;
                        else begin
                            descriptor_retire_valid<=1'b1;
                            descriptor_retire_index<=descriptor_index_q;
                            descriptor_retire_cycles<=descriptor_cycles_q+1'b1;
                            descriptor_cycles_q<='0;
                            if(descriptor_index_q==descriptor_count_q-1'b1)begin
                                commit_context_q<='0;state_q<=ST_COMMIT_READ_REQ;
                            end else begin
                                descriptor_index_q<=descriptor_index_q+1'b1;
                                state_q<=ST_FACTOR_REQ;
                            end
                        end
                    end
                    ST_COMMIT_READ_REQ:if(acc_read_req_accept)begin
                        acc_read_count_q<=acc_read_count_q+1'b1;
                        state_q<=ST_COMMIT_READ_WAIT;
                    end
                    ST_COMMIT_READ_WAIT:if(acc_read_rsp_accept)begin
                        acc_read_q<=acc_read_rsp_data;state_q<=ST_COMMIT_SEND;
                    end
                    ST_COMMIT_SEND:if(commit_accept)begin
                        commit_count_q<=commit_count_q+1'b1;
                        if(commit_context_q==CONTEXTS-1)state_q<=ST_DONE;
                        else begin commit_context_q<=commit_context_q+1'b1;
                            state_q<=ST_COMMIT_READ_REQ;end
                    end
                    ST_DONE:if(done_accept)state_q<=ST_IDLE;
                    ST_ABORT:if(abort_accept)begin
                        abort_count_q<=abort_count_q+1'b1;state_q<=ST_FAULT;
                    end
                    default:state_q<=ST_FAULT;
                endcase
            end
        end
    end
endmodule
`default_nettype wire
