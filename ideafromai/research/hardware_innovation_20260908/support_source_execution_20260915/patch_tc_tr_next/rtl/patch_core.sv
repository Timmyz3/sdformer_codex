module patch_core(
 input logic clk,rst_n,start_valid,output logic start_ready,input logic start_tc,start_av,input logic[15:0] start_group,
 output logic req_valid,input logic req_ready,output logic[13:0] req_addr,
 input logic rsp_valid,output logic rsp_ready,input logic[127:0] rsp_data,
 output logic out_valid,input logic out_ready,output logic[5:0] out_row,output logic[95:0] out_gate,
 output logic done_valid,input logic done_ready,
 output logic z_write_valid,output logic[8:0] z_write_row,output logic[127:0] z_write_data,
 output logic scratch_write_valid,output logic[8:0] scratch_write_row,output logic[383:0] scratch_write_data,
 output logic phase_valid,output logic[1:0] phase_kind,
 output logic final_valid,output logic[5:0] final_row,output logic[3:0] final_hblock,output logic[383:0] final_u,output logic[7:0] final_gate,
 output logic tc_valid,output logic[4:0] tc_compact,tc_global,
 output logic[31:0] count_source_words,count_U_words,count_V_words,count_A_words,count_tau_words,count_config_words,
 output logic[31:0] count_U_updates,count_V_updates,count_A_updates,count_A_scalar_mac,count_U_cache_hits,count_zero_source,
 output logic[7:0] debug_state,output logic[2:0] debug_phase,output logic[6:0] debug_nlatent
);
 typedef enum logic[7:0] {IDLE,MREQ,MWAIT,D_REQ,D_GOT,M_REQ,M_GOT,A_REQ,A_GOT,TC_SCAN,Z_CLEAR,SOURCE,S_GOT,U_NEXT,U_GATHER,U_GOT,U_POS,U_READ,U_ADD,Z_DONE,
 VC_START,VC_NEXT,VC_GOT,V_NEW,V_RANK,V_READ,V_MAC,Y_STORE,Y_DONE,Q_DONE,A_NEW,A_READ,A_MAC,A_STORE,TAU_REQ,TAU_GOT,FINAL,DRAIN,DONE} state_t;
 state_t st,ret;logic tc,av,a_is_q;logic[2:0] region,stage;
 logic[127:0] mem_hold,source_cache,u_cache;logic source_valid,u_valid;logic[13:0] source_tag,u_tag;
 logic[23:0] mask;logic[4:0] tc_map[24];logic signed[15:0] a[100];logic[383:0] tau_hold;
 logic signed[15:0] zmem[8][480];logic signed[47:0] scratch[8][480];logic[63:0] vcache[96];logic[95:0] gates[40];
 logic[8:0] z_read_addr[8],scratch_read_addr[8];logic[7:0] z_read_enable,scratch_read_enable;
 logic signed[15:0] z_read_data[8];logic signed[47:0] scratch_read_data[8];
 logic signed[47:0] acc[8],alu_b[8],alu_result[8],mul_result[8];logic signed[31:0] operand[8];logic signed[15:0] mac_a;
 logic signed[7:0] u_hold[8];logic[7:0] u_got;logic[39:0] source_bits,gate_pending;
 integer rank,nlatent,nblocks,cfg_i,scan_tc,ntc,clear_row,k,rb,pos,hb,r_cursor,v_r,op_t,op_p,source_t,tau_part,out_i;
 integer next_live,next_pos,need_lane,want_u_addr;logic[7:0] lane_live;logic any_u,any_operand,any_v;
 function automatic integer global_rank(input integer r);
   return tc?int'(tc_map[r/4])*4+r%4:r;
 endfunction
 function automatic logic live_rank(input integer r);
   return r<nlatent&&(tc||mask[r/4]);
 endfunction
 task automatic fetch(input integer addr,input state_t back);
   begin req_addr<=14'(addr);ret<=back;st<=MREQ;end
 endtask
 task automatic next_a_term;
   begin if(source_t==9)st<=A_STORE;else begin source_t<=source_t+1;st<=A_READ;end end
 endtask
 task automatic next_output;
   begin
     if(op_p<3)begin op_p<=op_p+1;pos<=(op_p+1)*10+op_t;st<=av?V_NEW:A_NEW;end
     else if(op_t<9)begin op_t<=op_t+1;op_p<=0;pos<=op_t+1;tau_part<=0;st<=TAU_REQ;end
     else if(hb<11)begin hb<=hb+1;op_t<=0;op_p<=0;pos<=0;tau_part<=0;st<=av?VC_START:TAU_REQ;end
     else begin out_i<=0;stage<=4;st<=DRAIN;end
   end
 endtask
 always_comb begin
   start_ready=st==IDLE;done_valid=st==DONE;req_valid=st==MREQ;rsp_ready=st==MWAIT;
   debug_state=st;debug_phase=stage;debug_nlatent=7'(nlatent);
   next_live=-1;for(integer j=95;j>=0;j--)if(j>=r_cursor&&live_rank(j))next_live=j;
   next_pos=-1;for(integer j=39;j>=0;j--)if(gate_pending[j])next_pos=j;
   lane_live=0;need_lane=-1;any_u=0;want_u_addr=0;
   for(integer l=7;l>=0;l--)begin
     lane_live[l]=live_rank(rb*8+l);
     if(lane_live[l]&&!u_got[l])need_lane=l;
     if(lane_live[l]&&u_hold[l]!=0)any_u=1;
   end
   if(need_lane>=0)want_u_addr=512+k*(rank/16)+global_rank(rb*8+need_lane)/16;
   mac_a=a[(a_is_q?pos%10:op_t)*10+source_t];any_operand=0;any_v=0;
   for(integer l=0;l<8;l++)begin
     logic[7:0] code;logic subtract;logic[47:0] add_operand;logic[48:0] carry;
     code=vcache[v_r][l*8+:8];
     if(operand[l]!=0)any_operand=1;if(code[5])any_v=1;
     // The sole eight signed16 x signed32 products. U and V use the same ALUs.
     mul_result[l]=$signed(mac_a)*$signed(operand[l]);alu_b[l]=mul_result[l];
     if(st==U_ADD)alu_b[l]=48'($signed(u_hold[l]));
     if(st==V_MAC)alu_b[l]=code[5]?(48'($signed(operand[0]))<<<code[3:0]):48'sd0;
     // One explicit 48-bit adder per lane, with XOR-controlled subtraction
     // and its single carry-in. No separate plus/minus datapaths to share.
     subtract=st==V_MAC&&code[4];add_operand=alu_b[l]^{48{subtract}};carry[0]=subtract;
     for(integer bit_id=0;bit_id<48;bit_id++)begin
       alu_result[l][bit_id]=acc[l][bit_id]^add_operand[bit_id]^carry[bit_id];
       carry[bit_id+1]=(acc[l][bit_id]&add_operand[bit_id])|((acc[l][bit_id]^add_operand[bit_id])&carry[bit_id]);
     end
   end
   // Exactly one read address/data mux per physical bank. The three
   // producer/consumer states are mutually exclusive; no inferred extra ports.
   z_read_enable=0;scratch_read_enable=0;
   for(integer l=0;l<8;l++)begin
     z_read_addr[l]=0;scratch_read_addr[l]=0;
     if(st==U_READ)begin z_read_enable[l]=1;z_read_addr[l]=9'(pos*nblocks+rb);end
     if(st==A_READ&&a_is_q)begin z_read_enable[l]=1;z_read_addr[l]=9'((pos/10*10+source_t)*nblocks+rb);end
     if(st==A_READ&&!a_is_q)begin scratch_read_enable[l]=1;scratch_read_addr[l]=9'((op_p*10+source_t)*12+hb);end
     if(st==V_READ&&v_r%8==l)begin
       if(av)begin scratch_read_enable[l]=1;scratch_read_addr[l]=9'(pos*nblocks+v_r/8);end
       else begin z_read_enable[l]=1;z_read_addr[l]=9'(pos*nblocks+v_r/8);end
     end
     z_read_data[l]=z_read_enable[l]?zmem[l][z_read_addr[l]]:16'sd0;
     scratch_read_data[l]=scratch_read_enable[l]?scratch[l][scratch_read_addr[l]]:48'sd0;
   end
   z_write_valid=st==Z_CLEAR||st==U_ADD;z_write_row=9'(st==Z_CLEAR?clear_row:pos*nblocks+rb);z_write_data=0;
   for(integer l=0;l<8;l++)if(st==U_ADD)z_write_data[l*16+:16]=alu_result[l][15:0];
   scratch_write_valid=st==Y_STORE||(st==A_STORE&&a_is_q);scratch_write_row=9'(st==Y_STORE?pos*12+hb:pos*nblocks+rb);scratch_write_data=0;
   for(integer l=0;l<8;l++)scratch_write_data[l*48+:48]=acc[l];
   phase_valid=st==Z_DONE||st==Y_DONE||st==Q_DONE;phase_kind=st==Z_DONE?1:st==Y_DONE?2:3;
   final_valid=st==FINAL||(st==A_STORE&&!a_is_q);final_row=6'(pos);final_hblock=4'(hb);final_u=0;final_gate=0;
   for(integer l=0;l<8;l++)begin final_u[l*48+:48]=acc[l];final_gate[l]=$signed(acc[l])>=$signed(tau_hold[l*48+:48]);end
   tc_valid=st==TC_SCAN&&mask[scan_tc];tc_compact=5'(ntc);tc_global=5'(scan_tc);
   out_valid=st==DRAIN;out_row=6'(out_i);out_gate=gates[out_i];
 end
 always_ff @(posedge clk or negedge rst_n)begin
   if(!rst_n)begin
     st<=IDLE;ret<=IDLE;req_addr<=0;tc<=0;av<=0;a_is_q<=0;region<=0;stage<=0;
     source_valid<=0;u_valid<=0;rank<=32;nlatent<=32;nblocks<=4;cfg_i<=0;scan_tc<=0;ntc<=0;clear_row<=0;k<=0;rb<=0;pos<=0;hb<=0;r_cursor<=0;v_r<=0;op_t<=0;op_p<=0;source_t<=0;tau_part<=0;out_i<=0;mask<=0;u_got<=0;gate_pending<=0;source_bits<=0;
     count_source_words<=0;count_U_words<=0;count_V_words<=0;count_A_words<=0;count_tau_words<=0;count_config_words<=0;
     count_U_updates<=0;count_V_updates<=0;count_A_updates<=0;count_A_scalar_mac<=0;count_U_cache_hits<=0;count_zero_source<=0;
   end else begin
     if(z_write_valid)for(integer l=0;l<8;l++)zmem[l][z_write_row]<=$signed(z_write_data[l*16+:16]);
     if(scratch_write_valid)for(integer l=0;l<8;l++)scratch[l][scratch_write_row]<=$signed(scratch_write_data[l*48+:48]);
     if(final_valid)gates[pos][hb*8+:8]<=final_gate;
     if(req_valid&&req_ready)begin
       if(req_addr>=8192)count_source_words<=count_source_words+1;
       else if(req_addr>=6000)count_V_words<=count_V_words+1;
       else if(req_addr>=512)count_U_words<=count_U_words+1;
       else if(req_addr>=16)count_tau_words<=count_tau_words+1;
       else if(req_addr>=3)count_A_words<=count_A_words+1;
       else count_config_words<=count_config_words+1;
     end
     case(st)
       IDLE:if(start_valid)begin
         st<=D_REQ;tc<=start_tc;av<=start_av;stage<=0;source_valid<=0;u_valid<=0;region<=0;
         for(integer g=0;g<8;g++)if(int'(start_group)>=g*2400)region<=3'(g);
         count_source_words<=0;count_U_words<=0;count_V_words<=0;count_A_words<=0;count_tau_words<=0;count_config_words<=0;
         count_U_updates<=0;count_V_updates<=0;count_A_updates<=0;count_A_scalar_mac<=0;count_U_cache_hits<=0;count_zero_source<=0;
       end
       MREQ:if(req_ready)st<=MWAIT;
       MWAIT:if(rsp_valid)begin mem_hold<=rsp_data;st<=ret;end
       D_REQ:fetch(0,D_GOT);
       D_GOT:begin rank<=int'(mem_hold[7:0]);assert(mem_hold[7:0]==32||mem_hold[7:0]==96)else $fatal(1,"rank");st<=M_REQ;end
       M_REQ:fetch(1+int'(region)/4,M_GOT);
       M_GOT:begin mask<=mem_hold[(int'(region)%4)*32+:24];cfg_i<=0;st<=A_REQ;end
       A_REQ:fetch(3+cfg_i,A_GOT);
       A_GOT:begin
         for(integer j=0;j<8;j++)if(cfg_i*8+j<100)a[cfg_i*8+j]<=$signed(mem_hold[j*16+:16]);
         if(cfg_i==12)begin
           scan_tc<=0;ntc<=0;
           if(tc)st<=TC_SCAN;
           else begin nlatent<=rank;nblocks<=rank/8;clear_row<=0;stage<=1;st<=Z_CLEAR;end
         end else begin cfg_i<=cfg_i+1;st<=A_REQ;end
       end
       TC_SCAN:begin
         if(mask[scan_tc])begin tc_map[ntc]<=5'(scan_tc);ntc<=ntc+1;end
         if(scan_tc==rank/4-1)begin
           integer total;total=ntc+int'(mask[scan_tc]);assert(total>0&&total%2==0)else $fatal(1,"fixed model TC admission");
           nlatent<=tc?total*4:rank;nblocks<=tc?total/2:rank/8;clear_row<=0;stage<=1;st<=Z_CLEAR;
         end else scan_tc<=scan_tc+1;
       end
       Z_CLEAR:if(clear_row==40*nblocks-1)begin k<=0;st<=SOURCE;end else clear_row<=clear_row+1;
       SOURCE:begin
         if(k==864)st<=Z_DONE;
         else if(source_valid&&source_tag==14'(8192+k/3))begin
           source_bits<=source_cache[(k%3)*40+:40];rb<=0;st<=U_NEXT;
           if(source_cache[(k%3)*40+:40]==0)count_zero_source<=count_zero_source+1;
         end else fetch(8192+k/3,S_GOT);
       end
       S_GOT:begin source_cache<=mem_hold;source_tag<=req_addr;source_valid<=1;st<=SOURCE;end
       U_NEXT:begin
         if(source_bits==0||rb==nblocks)begin k<=k+1;st<=SOURCE;end
         else if(lane_live==0)rb<=rb+1;
         else begin u_got<=~lane_live;for(integer l=0;l<8;l++)u_hold[l]<=0;st<=U_GATHER;end
       end
       U_GATHER:begin
         if(need_lane<0)begin if(!any_u)begin rb<=rb+1;st<=U_NEXT;end else begin gate_pending<=source_bits;st<=U_POS;end end
         else if(u_valid&&u_tag==14'(want_u_addr))begin
           count_U_cache_hits<=count_U_cache_hits+1;
           for(integer l=0;l<8;l++)if(lane_live[l]&&!u_got[l]&&512+k*(rank/16)+global_rank(rb*8+l)/16==want_u_addr)begin
             u_hold[l]<=$signed(u_cache[(global_rank(rb*8+l)%16)*8+:8]);u_got[l]<=1;
           end
         end else fetch(want_u_addr,U_GOT);
       end
       U_GOT:begin u_cache<=mem_hold;u_tag<=req_addr;u_valid<=1;st<=U_GATHER;end
       U_POS:if(next_pos<0)begin rb<=rb+1;st<=U_NEXT;end else begin pos<=next_pos;st<=U_READ;end
       U_READ:begin for(integer l=0;l<8;l++)acc[l]<=48'($signed(z_read_data[l]));st<=U_ADD;end
       U_ADD:begin
         for(integer l=0;l<8;l++)assert(alu_result[l]>=-32768&&alu_result[l]<=32767)else $fatal(1,"Z16 overflow");
         count_U_updates<=count_U_updates+1;gate_pending[pos]<=0;st<=U_POS;
       end
       Z_DONE:begin
         hb<=0;pos<=0;rb<=0;op_t<=0;op_p<=0;r_cursor<=0;
         if(av)begin a_is_q<=1;stage<=3;st<=A_NEW;end else begin a_is_q<=0;stage<=2;st<=VC_START;end
       end
       VC_START:begin r_cursor<=0;st<=VC_NEXT;end
       VC_NEXT:if(next_live<0)begin
         pos<=0;op_t<=0;op_p<=0;tau_part<=0;st<=av?TAU_REQ:V_NEW;
       end else begin v_r<=next_live;fetch(6000+global_rank(next_live)*12+hb,VC_GOT);end
       VC_GOT:begin for(integer l=0;l<8;l++)vcache[v_r][l*8+:8]<={2'b0,mem_hold[l*16+:6]};r_cursor<=v_r+1;st<=VC_NEXT;end
       V_NEW:begin for(integer l=0;l<8;l++)acc[l]<=0;r_cursor<=0;st<=V_RANK;end
       V_RANK:if(next_live<0)st<=av?FINAL:Y_STORE;else begin v_r<=next_live;st<=V_READ;end
       V_READ:begin
         logic signed[47:0] scalar;
         scalar=av?$signed(scratch_read_data[v_r%8]):48'($signed(z_read_data[v_r%8]));
         assert(scalar>=-48'sd2147483648&&scalar<=48'sd2147483647)else $fatal(1,"V input32");operand[0]<=scalar[31:0];
         if(scalar==0||!any_v)begin r_cursor<=v_r+1;st<=V_RANK;end else st<=V_MAC;
       end
       V_MAC:begin for(integer l=0;l<8;l++)acc[l]<=alu_result[l];count_V_updates<=count_V_updates+1;r_cursor<=v_r+1;st<=V_RANK;end
       Y_STORE:begin
         for(integer l=0;l<8;l++)assert(acc[l]>=-48'sd2147483648&&acc[l]<=48'sd2147483647)else $fatal(1,"Y32");
         if(pos<39)begin pos<=pos+1;st<=V_NEW;end else if(hb<11)begin hb<=hb+1;st<=VC_START;end else st<=Y_DONE;
       end
       Y_DONE:begin stage<=3;a_is_q<=0;hb<=0;op_t<=0;op_p<=0;pos<=0;tau_part<=0;st<=TAU_REQ;end
       Q_DONE:begin stage<=2;hb<=0;pos<=0;st<=VC_START;end
       A_NEW:begin
         source_t<=0;for(integer l=0;l<8;l++)acc[l]<=0;
         st<=a_is_q&&lane_live==0?A_STORE:A_READ;
       end
       A_READ:begin
         logic any_nonzero;any_nonzero=0;
         for(integer l=0;l<8;l++)begin
           logic signed[47:0] x;
           x=a_is_q?48'($signed(z_read_data[l])):$signed(scratch_read_data[l]);
           assert(x>=-48'sd2147483648&&x<=48'sd2147483647)else $fatal(1,"A input32");operand[l]<=x[31:0];if(x!=0)any_nonzero=1;
         end
         if(!any_nonzero||mac_a==0)next_a_term();else st<=A_MAC;
       end
       A_MAC:begin
         integer n;n=0;for(integer l=0;l<8;l++)begin acc[l]<=alu_result[l];if(operand[l]!=0)n++;end
         count_A_updates<=count_A_updates+1;count_A_scalar_mac<=count_A_scalar_mac+32'(n);next_a_term();
       end
       A_STORE:begin
         if(a_is_q)begin
           for(integer l=0;l<8;l++)assert(acc[l]>=-48'sd2147483648&&acc[l]<=48'sd2147483647)else $fatal(1,"Q32");
           if(pos<39)begin pos<=pos+1;st<=A_NEW;end else if(rb<nblocks-1)begin rb<=rb+1;pos<=0;st<=A_NEW;end else st<=Q_DONE;
         end else next_output();
       end
       TAU_REQ:fetch(16+(hb*10+op_t)*3+tau_part,TAU_GOT);
       TAU_GOT:begin tau_hold[tau_part*128+:128]<=mem_hold;
         if(tau_part==2)st<=av?V_NEW:A_NEW;else begin tau_part<=tau_part+1;st<=TAU_REQ;end
       end
       FINAL:next_output();
       DRAIN:if(out_ready)begin if(out_i==39)st<=DONE;else out_i<=out_i+1;end
       DONE:if(done_ready)st<=IDLE;
       default:st<=IDLE;
     endcase
   end
 end
endmodule
