// 2 output tiles x 4 source-ID PEs, P4/F1, C384. Source-ready endpoint.
// All source scanning, W arbitration, return ownership, private compaction,
// NR4 buffering, S accumulation, barriers and T10 consumers are RTL.
// External interfaces are four 64-bit source banks and two W8 ports/tile.
// Each memory port permits ONE transaction. Responses must be held until ready.
module gp_slice (
    input logic clk,rst_n,
    input logic cfg_uop_valid,
    input logic [7:0] cfg_uop_address,
    input logic [15:0] cfg_uop_data,
    input logic cfg_tau_valid,cfg_tau_tile,
    input logic [3:0] cfg_tau_time,
    input logic signed [23:0] cfg_tau_data,
    input logic start_valid,
    output logic start_ready,
    input logic start_reduce,
    input logic start_intersection,
    input logic [2:0] start_rank,
    input logic [55:0] start_decode,
    input logic [8:0] start_program_length,
    input logic [63:0] start_theta_payload,
    input logic [15:0] start_tag,
    output logic busy,done_valid,
    input logic done_ready,
    output logic [3:0] src_req_valid,
    input logic [3:0] src_req_ready,
    output logic [27:0] src_req_address, // k*7: 64-bit local word 0..71
    input logic [3:0] src_rsp_valid,
    output logic [3:0] src_rsp_ready,
    input logic [255:0] src_rsp_data,
    output logic [3:0] w_req_valid,
    input logic [3:0] w_req_ready,
    output logic [35:0] w_req_address, // (tile*2+port)*9: local W8 address 0..383
    output logic [43:0] w_req_full_address, // four11-bit byte addresses; legacy port is low9
    input logic [3:0] w_rsp_valid,
    output logic [3:0] w_rsp_ready,
    input logic [31:0] w_rsp_data,
    output logic out_valid,
    input logic out_ready,
    output logic [1:0] out_source_id,out_position,
    output logic [19:0] out_gates, // tile*10+t
    output logic [63:0] out_theta_payload,
    output logic [15:0] out_tag,
    output logic overflow,
    output logic [7:0] dbg_syn_done,dbg_psn,dbg_packet,dbg_zero_weight,dbg_commit,
    output logic [3:0] dbg_consumer_go,
    output logic [15:0] dbg_nr_occupancy,
    output logic [23:0] dbg_packet_sources,
    output logic [31:0] dbg_members,
    output logic [39:0] dbg_s_address,
    output logic [119:0] dbg_s_value,
    output logic [63:0] dbg_pc,
    output logic [3:0] dbg_w_pending,
    output logic [7:0] dbg_w_owner,
    output logic [35:0] dbg_source_index,
    output logic [11:0] dbg_w_kind,
    output logic [7:0] dbg_compare,dbg_advance_weight,dbg_advance_nrv
);
    typedef enum logic [2:0] {SCAN, SRC_REQ, SRC_WAIT, BRIDGE, END_SOURCE, WAIT_FIN} source_phase_t;
    typedef enum logic [1:0] {PORT_EMPTY, PORT_REQ, PORT_WAIT} port_phase_t;
    typedef enum logic [2:0] {COUNT_LO,COUNT_HI,INDEX_LO,INDEX_HI,HEAD_READY,WEIGHT_END} head_phase_t;
    localparam logic [2:0] READ_VALUE=0,READ_COUNT_LO=1,READ_COUNT_HI=2,
                           READ_INDEX_LO=3,READ_INDEX_HI=4;
    source_phase_t source_phase [0:3];
    port_phase_t port_phase [0:3];
    logic [8:0] source_index [0:3];
    logic [6:0] loaded_words [0:3];
    logic [127:0] source_window [0:3];
    logic [11:0] bridge_codes [0:3];
    logic [1:0] assigned [0:3],received [0:3],end_sent [0:3];
    logic [1:0] port_owner [0:3];
    logic [10:0] port_address [0:3];
    logic [11:0] port_codes [0:3];
    logic [2:0] port_kind [0:3];
    head_phase_t head_phase [0:7];
    logic [8:0] weight_count [0:7],weight_pointer [0:7],weight_index [0:7];
    logic [7:0] weight_inflight,weight_request;
    logic [10:0] weight_address [0:7];
    logic [2:0] weight_kind [0:7];
    logic intersection_q;
    logic [1:0] rr [0:1];
    logic [3:0] alloc_valid;
    logic [1:0] alloc_owner [0:3];
    logic [3:0] chosen [0:1];
    logic found;
    integer candidate;
    logic [11:0] scan_codes [0:3];
    logic [3:0] scan_live;
    integer word_needed [0:3],extract_shift [0:3];
    logic start_fire;
    logic [3:0] consumer_started;
    logic [63:0] theta_q;
    logic [15:0] tag_q;
    logic [55:0] decode_q;

    // Four program replicas, one per source ID, each broadcast to both tiles.
    // These are register arrays here; depth and multi-read cost are explicit.
    logic [15:0] program_mem [0:3][0:255];
    logic signed [23:0] tau_mem [0:1][0:9];
    logic [7:0] pe_pc [0:7];
    logic [3:0] pe_tau_address [0:7];
    logic [7:0] pe_row_valid,pe_row_ready,pe_end_valid,pe_end_ready;
    logic [11:0] pe_row_codes [0:7];
    logic signed [7:0] pe_row_weight [0:7];
    logic [7:0] pe_out_valid,pe_out_ready,pe_finished,pe_overflow;
    logic [1:0] pe_out_position [0:7];
    logic [9:0] pe_out_gates [0:7];
    logic [1:0] output_owner,output_rr;
    logic output_locked,output_found;
    logic [1:0] output_choice;

    assign start_ready=!busy && !done_valid;
    assign start_fire=start_valid && start_ready;
    assign overflow=|pe_overflow;
    assign out_valid=output_locked;
    assign out_source_id=output_owner;
    assign out_position=pe_out_position[int'(output_owner)];
    assign out_gates={pe_out_gates[4+int'(output_owner)],pe_out_gates[int'(output_owner)]};
    assign out_theta_payload=theta_q;
    assign out_tag=tag_q;

    always_comb begin
        src_req_valid=0;src_rsp_ready=0;src_req_address=0;
        scan_live=0;dbg_source_index=0;
        for (int k=0;k<4;k++) begin
            word_needed[k]=(int'(source_index[k])*12+11)/64;
            extract_shift[k]=int'(source_index[k])*12-(int'(loaded_words[k])-2)*64;
            scan_codes[k]=12'(source_window[k]>>extract_shift[k]);
            for (int p=0;p<4;p++)
                if (|decode_q[int'(scan_codes[k][p*3 +: 3])*7 +: 7]) scan_live[k]=1;
            src_req_valid[k]=busy && source_phase[k]==SRC_REQ;
            src_req_address[k*7 +: 7]=loaded_words[k];
            src_rsp_ready[k]=busy && source_phase[k]==SRC_WAIT;
            dbg_source_index[k*9 +: 9]=source_index[k];
        end
    end
    // GustavSNN VII-B: sorted NRV row index versus sorted nonzero-W index.
    // Static format in the SAME two byte-wide W ports: uint16 count, then
    // count repetitions of {uint16 column_index, int8 value}. Only a matching
    // index requests the value byte. A head/pointer is private to every PE;
    // there is no free bitmap or preloaded index list in a configuration input.
    always_comb begin
        weight_request=0;dbg_compare=0;dbg_advance_weight=0;dbg_advance_nrv=0;
        for (int m=0;m<2;m++) for (int k=0;k<4;k++) begin
            weight_address[m*4+k]=0;weight_kind[m*4+k]=READ_VALUE;
            if (busy && source_phase[k]==BRIDGE && !assigned[k][m] && !weight_inflight[m*4+k]) begin
                if (!intersection_q) begin
                    weight_request[m*4+k]=1;weight_address[m*4+k]={2'd0,source_index[k]};
                end else begin
                    case (head_phase[m*4+k])
                        COUNT_LO: begin
                            weight_request[m*4+k]=1;weight_address[m*4+k]=0;weight_kind[m*4+k]=READ_COUNT_LO;
                        end
                        COUNT_HI: begin
                            weight_request[m*4+k]=1;weight_address[m*4+k]=1;weight_kind[m*4+k]=READ_COUNT_HI;
                        end
                        INDEX_LO: begin
                            weight_request[m*4+k]=1;weight_address[m*4+k]=11'(2+3*int'(weight_pointer[m*4+k]));
                            weight_kind[m*4+k]=READ_INDEX_LO;
                        end
                        INDEX_HI: begin
                            weight_request[m*4+k]=1;weight_address[m*4+k]=11'(3+3*int'(weight_pointer[m*4+k]));
                            weight_kind[m*4+k]=READ_INDEX_HI;
                        end
                        HEAD_READY: begin
                            dbg_compare[m*4+k]=1;
                            if (weight_index[m*4+k]<source_index[k]) dbg_advance_weight[m*4+k]=1;
                            else if (weight_index[m*4+k]>source_index[k]) dbg_advance_nrv[m*4+k]=1;
                            else begin
                                weight_request[m*4+k]=1;weight_address[m*4+k]=11'(4+3*int'(weight_pointer[m*4+k]));
                            end
                        end
                        WEIGHT_END: dbg_advance_nrv[m*4+k]=1;
                        default: begin end
                    endcase
                end
            end
        end
    end
    // Two independent physical read ports per tile, static four-ID round robin.
    // Ports retain ownership and source metadata through response backpressure.
    // One extra W8 memory response register per port is part of the contract.
    always_comb begin
        alloc_valid=0;found=0;candidate=0;
        for (int m=0;m<2;m++) begin
            chosen[m]=0;
            for (int p=0;p<2;p++) begin
                alloc_owner[m*2+p]=0;found=0;
                for (int j=0;j<4;j++) begin
                    candidate=(int'(rr[m])+j)%4;
                    if (!found && busy && port_phase[m*2+p]==PORT_EMPTY
                        && weight_request[m*4+candidate]
                        && !chosen[m][candidate]) begin
                        alloc_valid[m*2+p]=1;alloc_owner[m*2+p]=2'(candidate);
                        chosen[m][candidate]=1;found=1;
                    end
                end
            end
        end
        w_req_valid=0;w_req_address=0;w_req_full_address=0;w_rsp_ready=0;
        dbg_w_pending=0;dbg_w_owner=0;dbg_w_kind=0;
        pe_row_valid=0;pe_end_valid=0;
        for (int i=0;i<8;i++) begin pe_row_codes[i]=0;pe_row_weight[i]=0;end
        for (int q=0;q<4;q++) begin
            w_req_valid[q]=busy && port_phase[q]==PORT_REQ;
            w_req_address[q*9 +: 9]=port_address[q][8:0];
            w_req_full_address[q*11 +: 11]=port_address[q];
            dbg_w_kind[q*3 +: 3]=port_kind[q];
            dbg_w_pending[q]=port_phase[q]!=PORT_EMPTY;
            dbg_w_owner[q*2 +: 2]=port_owner[q];
            if (busy && port_phase[q]==PORT_WAIT) begin
                if (port_kind[q]==READ_VALUE) begin
                    pe_row_valid[(q/2)*4+int'(port_owner[q])]=w_rsp_valid[q];
                    pe_row_codes[(q/2)*4+int'(port_owner[q])]=port_codes[q];
                    pe_row_weight[(q/2)*4+int'(port_owner[q])]=w_rsp_data[q*8 +: 8];
                    w_rsp_ready[q]=pe_row_ready[(q/2)*4+int'(port_owner[q])];
                end else w_rsp_ready[q]=1;
            end
        end
        for (int m=0;m<2;m++) for (int k=0;k<4;k++)
            pe_end_valid[m*4+k]=busy && source_phase[k]==END_SOURCE && !end_sent[k][m];
    end
    always_comb begin
        dbg_consumer_go=0;pe_out_ready=0;dbg_pc=0;
        for (int k=0;k<4;k++)
            dbg_consumer_go[k]=busy && !consumer_started[k] && dbg_syn_done[k] && dbg_syn_done[4+k];
        for (int i=0;i<8;i++) dbg_pc[i*8 +: 8]=pe_pc[i];
        output_found=0;output_choice=0;
        for (int j=0;j<4;j++)
            if (!output_found && pe_out_valid[(int'(output_rr)+j)%4]
                && pe_out_valid[4+(int'(output_rr)+j)%4]) begin
                output_choice=2'((int'(output_rr)+j)%4);output_found=1;
            end
        if (output_locked && out_ready) begin
            pe_out_ready[int'(output_owner)]=1;pe_out_ready[4+int'(output_owner)]=1;
        end
    end
    generate
        for (genvar m=0;m<2;m++) begin:g_tile
            for (genvar k=0;k<4;k++) begin:g_pe
                gp_slice_pe pe (
                    .clk(clk),.rst_n(rst_n),.start(start_fire),
                    .reduce_mode(start_reduce),.rank(start_rank),.decode(start_decode),
                    .program_length(start_program_length),
                    .row_valid(pe_row_valid[m*4+k]),.row_ready(pe_row_ready[m*4+k]),
                    .row_codes(pe_row_codes[m*4+k]),.row_weight(pe_row_weight[m*4+k]),
                    .end_valid(pe_end_valid[m*4+k]),.end_ready(pe_end_ready[m*4+k]),
                    .syn_done(dbg_syn_done[m*4+k]),.consumer_go(dbg_consumer_go[k]),
                    .program_address(pe_pc[m*4+k]),.program_data(program_mem[k][pe_pc[k]]),
                    .tau_address(pe_tau_address[m*4+k]),.tau_data(tau_mem[m][pe_tau_address[m*4+k]]),
                    .out_valid(pe_out_valid[m*4+k]),.out_ready(pe_out_ready[m*4+k]),
                    .out_position(pe_out_position[m*4+k]),.out_gates(pe_out_gates[m*4+k]),
                    .finished(pe_finished[m*4+k]),.overflow(pe_overflow[m*4+k]),
                    .dbg_occupancy(dbg_nr_occupancy[(m*4+k)*2 +: 2]),
                    .dbg_packet(dbg_packet[m*4+k]),.dbg_packet_sources(dbg_packet_sources[(m*4+k)*3 +: 3]),
                    .dbg_zero_weight(dbg_zero_weight[m*4+k]),.dbg_members(dbg_members[(m*4+k)*4 +: 4]),
                    .dbg_commit(dbg_commit[m*4+k]),.dbg_s_address(dbg_s_address[(m*4+k)*5 +: 5]),
                    .dbg_s_value(dbg_s_value[(m*4+k)*15 +: 15]),.dbg_psn(dbg_psn[m*4+k])
                );
            end
        end
    endgenerate
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy<=0;done_valid<=0;consumer_started<=0;theta_q<=0;tag_q<=0;decode_q<=0;
            intersection_q<=0;weight_inflight<=0;
            output_owner<=0;output_rr<=0;output_locked<=0;rr[0]<=0;rr[1]<=0;
            for (int k=0;k<4;k++) begin
                source_phase[k]<=WAIT_FIN;source_index[k]<=0;loaded_words[k]<=0;
                source_window[k]<=0;bridge_codes[k]<=0;assigned[k]<=0;received[k]<=0;end_sent[k]<=0;
                port_phase[k]<=PORT_EMPTY;port_owner[k]<=0;port_address[k]<=0;port_codes[k]<=0;port_kind[k]<=READ_VALUE;
            end
            for (int i=0;i<8;i++) begin
                head_phase[i]<=COUNT_LO;weight_count[i]<=0;weight_pointer[i]<=0;weight_index[i]<=0;
            end
        end else begin
            if (!busy && !done_valid) begin
                if (cfg_uop_valid) for (int k=0;k<4;k++) program_mem[k][cfg_uop_address]<=cfg_uop_data;
                if (cfg_tau_valid) tau_mem[cfg_tau_tile][cfg_tau_time]<=cfg_tau_data;
            end
            if (done_valid && done_ready) done_valid<=0;
            if (start_fire) begin
                busy<=1;consumer_started<=0;theta_q<=start_theta_payload;tag_q<=start_tag;decode_q<=start_decode;
                intersection_q<=start_intersection;weight_inflight<=0;
                output_owner<=0;output_rr<=0;output_locked<=0;rr[0]<=0;rr[1]<=0;
                for (int k=0;k<4;k++) begin
                    source_phase[k]<=SCAN;source_index[k]<=0;loaded_words[k]<=0;source_window[k]<=0;
                    assigned[k]<=0;received[k]<=0;end_sent[k]<=0;port_phase[k]<=PORT_EMPTY;
                end
                for (int i=0;i<8;i++) begin
                    head_phase[i]<=COUNT_LO;weight_count[i]<=0;weight_pointer[i]<=0;weight_index[i]<=0;
                end
            end else if (busy) begin
                for (int k=0;k<4;k++) begin
                    case (source_phase[k])
                        SCAN: begin
                            if (source_index[k]==384) source_phase[k]<=END_SOURCE;
                            else if (int'(loaded_words[k])<=word_needed[k]) source_phase[k]<=SRC_REQ;
                            else if (scan_live[k]) begin
                                bridge_codes[k]<=scan_codes[k];source_phase[k]<=BRIDGE;
                            end else source_index[k]<=source_index[k]+1'b1;
                        end
                        SRC_REQ: if (src_req_ready[k]) source_phase[k]<=SRC_WAIT;
                        SRC_WAIT: if (src_rsp_valid[k]) begin
                            source_window[k]<={src_rsp_data[k*64 +: 64],source_window[k][127:64]};
                            loaded_words[k]<=loaded_words[k]+1'b1;source_phase[k]<=SCAN;
                        end
                        BRIDGE: if (received[k]==2'b11) begin
                            assigned[k]<=0;received[k]<=0;source_index[k]<=source_index[k]+1'b1;
                            source_phase[k]<=SCAN;
                        end
                        END_SOURCE: begin
                            for (int m=0;m<2;m++) if (pe_end_ready[m*4+k]) end_sent[k][m]<=1;
                            if (end_sent[k]==2'b11) source_phase[k]<=WAIT_FIN;
                        end
                        default: begin end
                    endcase
                    if (dbg_consumer_go[k]) consumer_started[k]<=1;
                    for (int m=0;m<2;m++) begin
                        if (dbg_advance_nrv[m*4+k]) begin assigned[k][m]<=1;received[k][m]<=1;end
                        if (dbg_advance_weight[m*4+k]) begin
                            if (weight_pointer[m*4+k]+9'd1==weight_count[m*4+k]) head_phase[m*4+k]<=WEIGHT_END;
                            else begin
                                weight_pointer[m*4+k]<=weight_pointer[m*4+k]+1'b1;head_phase[m*4+k]<=INDEX_LO;
                            end
                        end
                    end
                end
                for (int q=0;q<4;q++) begin
                    case (port_phase[q])
                        PORT_EMPTY: if (alloc_valid[q]) begin
                            port_owner[q]<=alloc_owner[q];port_address[q]<=weight_address[(q/2)*4+int'(alloc_owner[q])];
                            port_kind[q]<=weight_kind[(q/2)*4+int'(alloc_owner[q])];
                            port_codes[q]<=bridge_codes[alloc_owner[q]];port_phase[q]<=PORT_REQ;
                            weight_inflight[(q/2)*4+int'(alloc_owner[q])]<=1;
                            if (weight_kind[(q/2)*4+int'(alloc_owner[q])]==READ_VALUE) assigned[alloc_owner[q]][q/2]<=1;
                            rr[q/2]<=alloc_owner[q]+1'b1;
                        end
                        PORT_REQ: if (w_req_ready[q]) port_phase[q]<=PORT_WAIT;
                        PORT_WAIT: if (w_rsp_valid[q] && w_rsp_ready[q]) begin
                            weight_inflight[(q/2)*4+int'(port_owner[q])]<=0;port_phase[q]<=PORT_EMPTY;
                            case (port_kind[q])
                                READ_COUNT_LO: begin
                                    weight_count[(q/2)*4+int'(port_owner[q])][7:0]<=w_rsp_data[q*8 +: 8];
                                    head_phase[(q/2)*4+int'(port_owner[q])]<=COUNT_HI;
                                end
                                READ_COUNT_HI: begin
                                    weight_count[(q/2)*4+int'(port_owner[q])][8]<=w_rsp_data[q*8];
                                    head_phase[(q/2)*4+int'(port_owner[q])]<=
                                        {w_rsp_data[q*8],weight_count[(q/2)*4+int'(port_owner[q])][7:0]}==0 ? WEIGHT_END:INDEX_LO;
                                end
                                READ_INDEX_LO: begin
                                    weight_index[(q/2)*4+int'(port_owner[q])][7:0]<=w_rsp_data[q*8 +: 8];
                                    head_phase[(q/2)*4+int'(port_owner[q])]<=INDEX_HI;
                                end
                                READ_INDEX_HI: begin
                                    weight_index[(q/2)*4+int'(port_owner[q])][8]<=w_rsp_data[q*8];
                                    head_phase[(q/2)*4+int'(port_owner[q])]<=HEAD_READY;
                                end
                                default: begin
                                    received[port_owner[q]][q/2]<=1;
                                    if (intersection_q) begin
                                        if (weight_pointer[(q/2)*4+int'(port_owner[q])]+9'd1==weight_count[(q/2)*4+int'(port_owner[q])])
                                            head_phase[(q/2)*4+int'(port_owner[q])]<=WEIGHT_END;
                                        else begin
                                            weight_pointer[(q/2)*4+int'(port_owner[q])]<=weight_pointer[(q/2)*4+int'(port_owner[q])]+1'b1;
                                            head_phase[(q/2)*4+int'(port_owner[q])]<=INDEX_LO;
                                        end
                                    end
                                end
                            endcase
                        end
                        default: begin end
                    endcase
                end
                if (!output_locked && output_found) begin output_locked<=1;output_owner<=output_choice;end
                else if (output_locked && out_ready) begin output_locked<=0;output_rr<=output_owner+1'b1;end
                if (&pe_finished) begin busy<=0;done_valid<=1;end
            end
        end
    end
endmodule
