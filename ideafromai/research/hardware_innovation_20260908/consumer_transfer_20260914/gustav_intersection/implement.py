from pathlib import Path
H=Path(__file__).resolve().parent;B=H.parents[1];OLD=B/'psn/rtl/gp_slice'
s=(OLD/'gp_slice.sv').read_text()
def sub(a,b):
 global s
 assert a in s,a[:100];s=s.replace(a,b)
sub('input logic start_intersection,','input logic start_intersection,\n    input logic [1:0] start_frontend,')
sub('output logic [7:0] dbg_compare,dbg_advance_weight,dbg_advance_nrv','output logic [7:0] dbg_compare,dbg_advance_weight,dbg_advance_nrv,\n    output logic [1:0] dbg_shared_ready,dbg_dense_bypass')
sub('READ_INDEX_LO=3,READ_INDEX_HI=4;','READ_INDEX_LO=3,READ_INDEX_HI=4,READ_SHARED_META=5;')
sub('logic intersection_q;','''logic intersection_q;
    logic [1:0] frontend_q;
    logic [383:0] weight_bitmap [0:1];
    logic [8:0] shared_nnz [0:1];
    logic [5:0] meta_issued [0:1],meta_received [0:1];
    logic [1:0] shared_ready,dense_bypass;
    logic [3:0] alloc_meta;
    logic [5:0] alloc_meta_address [0:3];
    integer meta_limit [0:1],meta_allocations [0:1],meta_returns [0:1];
    logic [3:0] source_advance;
    assign dbg_shared_ready=shared_ready;
    assign dbg_dense_bypass=dense_bypass;
    always_comb begin
        source_advance=0;
        for(int m=0;m<2;m++)begin
            dense_bypass[m]=(frontend_q==3 && meta_received[m]>=2 && shared_nnz[m]>=334);
            meta_limit[m]=(frontend_q==2)?48:((meta_received[m]<2||dense_bypass[m])?2:50);
            shared_ready[m]=(frontend_q<2)||(int'(meta_received[m])==meta_limit[m]);
            meta_returns[m]=0;
            for(int p=0;p<2;p++)if(port_phase[m*2+p]==PORT_WAIT && port_kind[m*2+p]==READ_SHARED_META
                && w_rsp_valid[m*2+p] && w_rsp_ready[m*2+p])meta_returns[m]=meta_returns[m]+1;
        end
        for(int k=0;k<4;k++)begin
            source_advance[k]=(source_phase[k]==BRIDGE && received[k]==2'b11)
                ||(source_phase[k]==SCAN && source_index[k]<384 && int'(loaded_words[k])>word_needed[k]
                    && (&shared_ready) && !scan_live[k]);
        end
    end''')
sub('''if (!intersection_q) begin
                    weight_request[m*4+k]=1;weight_address[m*4+k]={2'd0,source_index[k]};
                end else begin''','''if(frontend_q>=2)begin
                    if(shared_ready[m])begin
                        if(dense_bypass[m])begin
                            weight_request[m*4+k]=1;weight_address[m*4+k]=11'(2+int'(source_index[k]));
                        end else if(weight_bitmap[m][source_index[k]])begin
                            weight_request[m*4+k]=1;
                            weight_address[m*4+k]=11'((frontend_q==2?48:50)+int'(weight_pointer[m*4+k]));
                        end else dbg_advance_nrv[m*4+k]=1;
                    end
                end else if (!intersection_q) begin
                    weight_request[m*4+k]=1;weight_address[m*4+k]={2'd0,source_index[k]};
                end else begin''')
sub('''alloc_valid=0;found=0;candidate=0;
        for (int m=0;m<2;m++) begin
            chosen[m]=0;''','''alloc_valid=0;alloc_meta=0;found=0;candidate=0;
        for(int q=0;q<4;q++)alloc_meta_address[q]=0;
        for (int m=0;m<2;m++) begin
            chosen[m]=0;meta_allocations[m]=0;''')
sub('''alloc_owner[m*2+p]=0;found=0;
                for (int j=0;j<4;j++) begin''','''alloc_owner[m*2+p]=0;found=0;
                if(busy && frontend_q>=2 && !shared_ready[m] && port_phase[m*2+p]==PORT_EMPTY
                    && int'(meta_issued[m])+meta_allocations[m]<meta_limit[m])begin
                    alloc_valid[m*2+p]=1;alloc_meta[m*2+p]=1;found=1;
                    alloc_meta_address[m*2+p]=6'(int'(meta_issued[m])+meta_allocations[m]);
                    meta_allocations[m]=meta_allocations[m]+1;
                end
                for (int j=0;j<4;j++) begin''')
sub('''intersection_q<=0;weight_inflight<=0;''','''intersection_q<=0;frontend_q<=0;weight_inflight<=0;
            for(int m=0;m<2;m++)begin weight_bitmap[m]<=0;shared_nnz[m]<=0;meta_issued[m]<=0;meta_received[m]<=0;end''')
sub('''intersection_q<=start_intersection;weight_inflight<=0;''','''intersection_q<=start_frontend<2 && start_intersection;
                frontend_q<=start_frontend>=2?start_frontend:{1'b0,start_intersection};weight_inflight<=0;
                for(int m=0;m<2;m++)begin weight_bitmap[m]<=0;shared_nnz[m]<=0;meta_issued[m]<=0;meta_received[m]<=0;end''')
sub('''end else if (busy) begin
                for (int k=0;k<4;k++) begin''','''end else if (busy) begin
                for(int m=0;m<2;m++)begin
                    meta_issued[m]<=meta_issued[m]+6'(meta_allocations[m]);
                    meta_received[m]<=meta_received[m]+6'(meta_returns[m]);
                end
                for (int k=0;k<4;k++) begin
                    if(frontend_q>=2 && source_advance[k])for(int m=0;m<2;m++)
                        if(weight_bitmap[m][source_index[k]])weight_pointer[m*4+k]<=weight_pointer[m*4+k]+1'b1;''')
sub('''else if (scan_live[k]) begin''','''else if(!(&shared_ready))begin end
                            else if (scan_live[k]) begin''')
sub('''port_owner[q]<=alloc_owner[q];port_address[q]<=weight_address[(q/2)*4+int'(alloc_owner[q])];''','''port_owner[q]<=alloc_owner[q];
                            if(alloc_meta[q])begin
                                port_address[q]<={5'd0,alloc_meta_address[q]};port_kind[q]<=READ_SHARED_META;
                                port_codes[q]<=0;port_phase[q]<=PORT_REQ;
                            end else begin
                            port_address[q]<=weight_address[(q/2)*4+int'(alloc_owner[q])];''')
sub('''rr[q/2]<=alloc_owner[q]+1'b1;
                        end''','''rr[q/2]<=alloc_owner[q]+1'b1;
                            end
                        end''')
sub('''weight_inflight[(q/2)*4+int'(port_owner[q])]<=0;port_phase[q]<=PORT_EMPTY;
                            case (port_kind[q])''','''if(port_kind[q]!=READ_SHARED_META)weight_inflight[(q/2)*4+int'(port_owner[q])]<=0;
                            port_phase[q]<=PORT_EMPTY;
                            case (port_kind[q])
                                READ_SHARED_META:begin
                                    if(frontend_q==3 && port_address[q]<2)begin
                                        if(port_address[q]==0)shared_nnz[q/2][7:0]<=w_rsp_data[q*8+:8];
                                        else shared_nnz[q/2][8]<=w_rsp_data[q*8];
                                    end else weight_bitmap[q/2][(int'(port_address[q])-(frontend_q==3?2:0))*8+:8]<=w_rsp_data[q*8+:8];
                                end''')
(H/'gp_slice.sv').write_text(s)
print('isolated top generated; original PE referenced directly')
