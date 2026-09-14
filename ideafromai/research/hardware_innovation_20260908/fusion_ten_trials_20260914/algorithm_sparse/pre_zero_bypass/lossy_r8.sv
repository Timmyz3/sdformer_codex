module lossy_r8(
 input logic clk,reset_n,cfg_valid,
 input logic [3:0] cfg_kind,input logic [10:0] cfg_addr,input logic [255:0] cfg_data,
 input logic start,input logic [3:0] mode,input logic source_allow,weight_allow,
 output logic result_valid,input logic result_ready,
 output logic [8:0] result_addr,output logic [255:0] result_data,output logic done,
 output logic [31:0] cycles,source_words,weight_words,second_weight_words,local_source_reads,
 output logic [31:0] z_vector_reads,z_scalar_reads,z_writes,first_issues,dual_updates,
 output logic [31:0] psum_reads,psum_writes,mac_issues,
 output logic [31:0] source_stalls,weight_stalls,output_stalls,
 output logic [31:0] encoder_cycles,prototype_reads,held_vectors,
 output logic [5:0] debug_state
);
 typedef enum logic [5:0] {IDLE,ZCLEAR,L_START,L_LOAD,L_GATHER,CHECK,QREAD,TIMESEL,
 ZREAD,ZADD,KNEXT,ZSCAN,VLOAD,POSLOAD,BASE_MAC,STORE,DRAIN_READ,DRAIN_SEND,FINISH,EREAD,EDIFF,EABS,ERED1,ERED2,ERED3,ESCORE,EMAX,EWRITE,PROTO_READ} state_t;
 state_t state;logic [3:0] mode_q;
 logic [9:0] source_mem[0:1535],local_source[0:15],src_masks[0:3];
 logic signed [2:0] q_mem[0:7][0:863],q_hold[0:7];
 logic signed [15:0] v_mem[0:7][0:95],qblock[0:7][0:7];
 logic k_live[0:863],v_live[0:95];
 logic [25:0] z_mem[0:7][0:19],z_hold[0:7];
 logic signed [31:0] p_mem[0:7][0:479],acc[0:7];
 logic [7:0] position_live[0:39],rank_live,block_live,remaining,scan_mask;
 logic [39:0] pending;logic signed [15:0] oy,ox;
 integer k,fp,zrow,og,qfill,row,load_xy,selected_time;
 logic [10:0] source_addr;logic [4:0] read_zrow;
 logic in_bounds,cfg_v_live,active_lo,active_hi,read_half;
 logic [2:0] selected_rank;
 logic signed [13:0] scalar;
 logic signed [15:0] multiply_coefficient[0:7];
 logic signed [31:0] product[0:7],lhs[0:7],rhs[0:7],add_y[0:7];

 // All modes receive the union resources: one encoder state bank and static K4 table.
 logic signed [12:0] current_z[0:7],reference_z[0:7],codebook[0:3][0:7];
 logic signed [13:0] difference[0:7];
 logic signed [31:0] work[0:7];
 logic [2:0] score_shift[0:7];
 logic [31:0] rank_threshold[0:7],time_rank_threshold[0:7],group_threshold[0:8],time_threshold;
 logic signed [31:0] prototype_mem[0:7][0:47];
 logic [1:0] codes[0:39],enc_code,best_code;
 logic [2:0] residual_rank[0:39];
 logic signed [13:0] residual[0:39];
 logic refresh[0:39],encode_residual,drop_or_hold;
 logic [31:0] best_score;
 logic [3:0] current_nnz;
 logic [2:0] max_rank;
 logic signed [12:0] approximate[0:7];
 logic changed,delta_fits,select_delta;
 logic [3:0] delta_nnz,approx_nnz;
 logic use_delta[0:39];
 logic signed [12:0] encoded_z[0:7];
 logic alu_cin[0:7];
 task clear_counters;
 begin
 cycles<=0;source_words<=0;weight_words<=0;second_weight_words<=0;local_source_reads<=0;
 z_vector_reads<=0;z_scalar_reads<=0;z_writes<=0;first_issues<=0;dual_updates<=0;
 psum_reads<=0;psum_writes<=0;mac_issues<=0;
 source_stalls<=0;weight_stalls<=0;output_stalls<=0;encoder_cycles<=0;prototype_reads<=0;held_vectors<=0;
 end
 endtask

 always_comb begin
 source_addr=11'((k/9)*16+load_xy);
 in_bounds=(int'(oy)+load_xy/4>=0 && int'(oy)+load_xy/4<240 &&
 int'(ox)+load_xy%4>=0 && int'(ox)+load_xy%4<320);
 cfg_v_live=0;for(integer i=0;i<8;i=i+1)cfg_v_live=cfg_v_live||(cfg_data[i*32+:16]!=0);
 selected_time=0;for(integer i=39;i>=0;i=i-1)if(pending[i])selected_time=i;
 selected_rank=0;for(integer i=7;i>=0;i=i-1)if(remaining[i])selected_rank=3'(i);
 read_zrow=5'((fp/20)*10+fp%10);read_half=((fp/10)%2)!=0;
 scan_mask=0;
 if(state==ZSCAN)for(integer i=0;i<8;i=i+1)
 scan_mask[i]=read_half?(z_mem[i][read_zrow][25:13]!=0):(z_mem[i][read_zrow][12:0]!=0);
 scalar=read_half?14'($signed(z_mem[selected_rank][read_zrow][25:13])):14'($signed(z_mem[selected_rank][read_zrow][12:0]));
 if(state==ZSCAN && (mode_q==3 || mode_q==4)) begin
  scan_mask=0;if(residual[fp]!=0)scan_mask[residual_rank[fp]]=1;
 end
 if(mode_q==3 || mode_q==4)scalar=residual[fp];
 max_rank=0;for(integer i=1;i<8;i=i+1)if(work[i]>work[max_rank])max_rank=3'(i);
 current_nnz=0;changed=0;
 for(integer i=0;i<8;i=i+1)begin
  if(current_z[i]!=0)current_nnz=current_nnz+1;
  approximate[i]=current_z[i];
  if(mode_q==1 && drop_or_hold)approximate[i]=0;
  if(mode_q==2 && $unsigned(work[i]>>score_shift[i])<=rank_threshold[i])approximate[i]=0;
  if(mode_q==3 || mode_q==4)approximate[i]=(i==int'(max_rank))?current_z[i]:codebook[enc_code][i];
  if((mode_q==5 || mode_q==8) && fp%10!=0 && drop_or_hold)approximate[i]=reference_z[i];
  if((mode_q==6 || mode_q==7) && fp%10!=0 && $unsigned(work[i]>>score_shift[i])<=time_rank_threshold[i])approximate[i]=reference_z[i];
  changed=changed || approximate[i]!=reference_z[i];
 end
 delta_nnz=0;approx_nnz=0;delta_fits=1;
 for(integer i=0;i<8;i=i+1)begin
  if(approximate[i]!=0)approx_nnz=approx_nnz+1;
  if(approximate[i]!=reference_z[i])begin
   delta_nnz=delta_nnz+1;
   if(difference[i]>14'sd4095 || difference[i]<-14'sd4096)delta_fits=0;
  end
 end
 select_delta=(mode_q==7 || mode_q==8) && fp%10!=0 && delta_fits && delta_nnz<approx_nnz;
 for(integer i=0;i<8;i=i+1)encoded_z[i]=select_delta?((approximate[i]!=reference_z[i])?difference[i][12:0]:13'd0):approximate[i];
 for(integer l=0;l<8;l=l+1)begin
 multiply_coefficient[l]=qblock[selected_rank][l];
 product[l]=$signed(multiply_coefficient[l])*$signed(scalar);
 lhs[l]=acc[l];rhs[l]=product[l];alu_cin[l]=0;
 if(state==EDIFF)begin
 lhs[l]=32'(current_z[l]);
 if(mode_q==3 || mode_q==4)rhs[l]=~32'(codebook[enc_code][l]);
 else if((mode_q==5 || mode_q==8) || (mode_q==6 || mode_q==7))rhs[l]=~32'(reference_z[l]);
 else rhs[l]=-32'sd1;
 alu_cin[l]=1;
 end
 if(state==EABS)begin
 lhs[l]=0;rhs[l]=difference[l][13]?~32'(difference[l]):32'(difference[l]);alu_cin[l]=difference[l][13];
 end
 if(state==ERED1)begin lhs[l]=0;rhs[l]=0;if(l<4)begin lhs[l]=work[2*l];rhs[l]=work[2*l+1];end end
 if(state==ERED2)begin lhs[l]=0;rhs[l]=0;if(l<2)begin lhs[l]=work[2*l];rhs[l]=work[2*l+1];end end
 if(state==ERED3)begin lhs[l]=0;rhs[l]=0;if(l==0)begin lhs[l]=work[0];rhs[l]=work[1];end end
 if(state==ZADD)begin
 lhs[l]={6'b0,z_hold[l]};
 rhs[l]={6'b0,(active_hi?{{10{q_hold[l][2]}},q_hold[l]}:13'd0),
                    (active_lo?{{10{q_hold[l][2]}},q_hold[l]}:13'd0)};
 end
 end
 result_valid=(state==DRAIN_SEND);result_addr=9'(row);debug_state=state;
 end
 genvar l,b;
 generate for(l=0;l<8;l=l+1)begin:ALU
 for(b=0;b<32;b=b+1)begin:BIT
 wire cin;
 if(b==0)assign cin=alu_cin[l];
 else if(b==13)assign cin=(state==ZADD)?1'b0:BIT[b-1].CARRY.cout;
 else assign cin=BIT[b-1].CARRY.cout;
 assign add_y[l][b]=lhs[l][b]^rhs[l][b]^cin;
 if(b<31)begin:CARRY
 wire cout;assign cout=(lhs[l][b]&rhs[l][b])|((lhs[l][b]^rhs[l][b])&cin);
 end
 end
 end endgenerate

 always_ff @(posedge clk)begin
 if(!reset_n)begin
 state<=IDLE;mode_q<=0;done<=0;oy<=0;ox<=0;k<=0;fp<=0;zrow<=0;og<=0;qfill<=0;row<=0;load_xy<=0;
 enc_code<=0;best_code<=0;best_score<=0;encode_residual<=0;drop_or_hold<=0;rank_live<=0;block_live<=0;remaining<=0;pending<=0;result_data<=0;active_lo<=0;active_hi<=0;clear_counters();
 for(integer i=0;i<4;i=i+1)src_masks[i]<=0;
 for(integer i=0;i<8;i=i+1)begin q_hold[i]<=0;z_hold[i]<=0;acc[i]<=0;end
 end else begin
 done<=0;
 if(cfg_valid&&state==IDLE)case(cfg_kind)
 0:source_mem[cfg_addr[10:0]]<=cfg_data[9:0];
 3:begin oy<=cfg_data[15:0];ox<=cfg_data[31:16];end
 4:for(integer i=0;i<8;i=i+1)q_mem[i][cfg_addr[9:0]]<=cfg_data[i*32+:3];
 5:begin
 for(integer i=0;i<8;i=i+1)v_mem[i][cfg_addr[6:0]]<=cfg_data[i*32+:16];
 v_live[cfg_addr[6:0]]<=cfg_v_live;
 end
 6:k_live[cfg_addr[9:0]]<=cfg_data[0];
 7:begin
  if(cfg_addr==0)for(integer i=0;i<8;i=i+1)score_shift[i]<=cfg_data[i*32+:3];
  if(cfg_addr==1)for(integer i=0;i<8;i=i+1)rank_threshold[i]<=cfg_data[i*32+:32];
  if(cfg_addr==2)for(integer i=0;i<8;i=i+1)time_rank_threshold[i]<=cfg_data[i*32+:32];
  if(cfg_addr==3)time_threshold<=cfg_data[31:0];
  if(cfg_addr>=4 && cfg_addr<=12)group_threshold[cfg_addr-4]<=cfg_data[31:0];
 end
 8:for(integer i=0;i<8;i=i+1)codebook[cfg_addr[1:0]][i]<=cfg_data[i*32+:13];
 9:for(integer i=0;i<8;i=i+1)prototype_mem[i][cfg_addr[5:0]]<=cfg_data[i*32+:32];
 default:begin end
 endcase
 if(state!=IDLE)cycles<=cycles+1;
 if(state>=EREAD && state<=EWRITE)encoder_cycles<=encoder_cycles+1;
 case(state)
 IDLE:if(start)begin
 mode_q<=mode;zrow<=0;rank_live<=0;block_live<=0;state<=ZCLEAR;clear_counters();
 end
 ZCLEAR:begin
 for(integer i=0;i<8;i=i+1)z_mem[i][zrow]<=0;
 z_writes<=z_writes+1;
 if(zrow==19)begin k<=0;state<=L_START;end else zrow<=zrow+1;
 end
 L_START:begin load_xy<=0;state<=L_LOAD;end
 L_LOAD:if(!in_bounds||source_allow)begin
 local_source[load_xy]<=in_bounds?source_mem[source_addr]:10'd0;
 if(in_bounds)source_words<=source_words+1;
 if(load_xy==15)state<=L_GATHER;else load_xy<=load_xy+1;
 end else source_stalls<=source_stalls+1;
 L_GATHER:if(!k_live[k])state<=KNEXT;
 else begin
 for(integer i=0;i<4;i=i+1)src_masks[i]<=local_source[(i/2+(k%9)/3)*4+i%2+k%3];
 local_source_reads<=local_source_reads+1;state<=CHECK;
 end
 CHECK:begin
 if(1'b1)pending<={20'd0,(src_masks[2]|src_masks[3]),(src_masks[0]|src_masks[1])};
 else pending<={src_masks[3],src_masks[2],src_masks[1],src_masks[0]};
 state<=((src_masks[0]|src_masks[1]|src_masks[2]|src_masks[3])==0)?KNEXT:QREAD;
 end
 QREAD:if(weight_allow)begin
 for(integer i=0;i<8;i=i+1)q_hold[i]<=q_mem[i][k];
 weight_words<=weight_words+1;state<=TIMESEL;
 end else weight_stalls<=weight_stalls+1;
 TIMESEL:if(pending!=0)begin
 if(1'b1)begin
 zrow<=selected_time;
 active_lo<=src_masks[(selected_time/10)*2][selected_time%10];
 active_hi<=src_masks[(selected_time/10)*2+1][selected_time%10];
 end else begin
 zrow<=(selected_time/20)*10+selected_time%10;
 active_lo<=((selected_time/10)%2)==0;active_hi<=((selected_time/10)%2)==1;
 end
 state<=ZREAD;
 end else state<=KNEXT;
 ZREAD:begin
 for(integer i=0;i<8;i=i+1)z_hold[i]<=z_mem[i][zrow];
 z_vector_reads<=z_vector_reads+1;state<=ZADD;
 end
 ZADD:begin
 for(integer i=0;i<8;i=i+1)z_mem[i][zrow]<=add_y[i][25:0];
 first_issues<=first_issues+1;z_writes<=z_writes+1;
 if(active_lo&&active_hi)dual_updates<=dual_updates+1;
 pending<=pending&(pending-40'd1);state<=TIMESEL;
 end
 KNEXT:if(k==863)begin fp<=0;state<=(mode_q==0)?ZSCAN:EREAD;end
 else begin k<=k+1;state<=(k%9==8)?L_START:L_GATHER;end
 EREAD:begin
 for(integer i=0;i<8;i=i+1)begin
 z_hold[i]<=z_mem[i][read_zrow];
 current_z[i]<=read_half?$signed(z_mem[i][read_zrow][25:13]):$signed(z_mem[i][read_zrow][12:0]);
 end
 z_vector_reads<=z_vector_reads+1;enc_code<=0;best_code<=0;best_score<=32'hffffffff;
 encode_residual<=(mode_q==4);drop_or_hold<=0;
 state<=(((mode_q==5 || mode_q==8) || (mode_q==6 || mode_q==7)) && fp%10==0)?EWRITE:EDIFF;
 end
 EDIFF:begin
 for(integer i=0;i<8;i=i+1)difference[i]<=add_y[i][13:0];state<=EABS;
 end
 EABS:begin
 for(integer i=0;i<8;i=i+1)work[i]<=add_y[i]<<<score_shift[i];
 if(mode_q==2 || (mode_q==6 || mode_q==7))state<=EWRITE;
 else if(encode_residual)state<=EMAX;
 else state<=ERED1;
 end
 ERED1:begin for(integer i=0;i<4;i=i+1)work[i]<=add_y[i];state<=ERED2;end
 ERED2:begin for(integer i=0;i<2;i=i+1)work[i]<=add_y[i];state<=ERED3;end
 ERED3:begin work[0]<=add_y[0];state<=ESCORE;end
 ESCORE:begin
 if(mode_q==3)begin
 if($unsigned(work[0])<best_score)begin best_score<=32'(work[0]);best_code<=enc_code;end
 if(enc_code==3)begin
 enc_code<=($unsigned(work[0])<best_score)?enc_code:best_code;
 encode_residual<=1;state<=EDIFF;
 end else begin enc_code<=enc_code+1;state<=EDIFF;end
 end else begin
 drop_or_hold<=(mode_q==1)?($unsigned(work[0])<=group_threshold[current_nnz]):($unsigned(work[0])<=time_threshold);
 state<=EWRITE;
 end
 end
 EMAX:begin
 codes[fp]<=enc_code;residual_rank[fp]<=max_rank;residual[fp]<=difference[max_rank];state<=EWRITE;
 end
 EWRITE:begin
 for(integer i=0;i<8;i=i+1)begin
 z_mem[i][read_zrow]<=read_half?{encoded_z[i],z_hold[i][12:0]}:{z_hold[i][25:13],encoded_z[i]};
 reference_z[i]<=approximate[i];
 end
 use_delta[fp]<=select_delta;
 refresh[fp]<=((mode_q!=5 && mode_q!=6 && mode_q!=7 && mode_q!=8) || fp%10==0 || changed);
 z_writes<=z_writes+1;
 if(fp==39)begin fp<=0;state<=ZSCAN;end else begin fp<=fp+1;state<=EREAD;end
 end
 ZSCAN:begin
 z_vector_reads<=z_vector_reads+1;position_live[fp]<=scan_mask;rank_live<=rank_live|scan_mask;
 if(fp==39)begin og<=0;qfill<=0;state<=VLOAD;end else fp<=fp+1;
 end
 VLOAD:begin
 if(!rank_live[qfill]||!v_live[og*8+qfill]||weight_allow)begin
 block_live[qfill]<=rank_live[qfill]&&v_live[og*8+qfill];
 for(integer i=0;i<8;i=i+1)qblock[qfill][i]<=(rank_live[qfill]&&v_live[og*8+qfill])?v_mem[i][og*8+qfill]:16'sd0;
 if(rank_live[qfill]&&v_live[og*8+qfill])begin weight_words<=weight_words+1;second_weight_words<=second_weight_words+1;end
 if(qfill==7)begin fp<=0;state<=POSLOAD;end else qfill<=qfill+1;
 end else weight_stalls<=weight_stalls+1;
 end
 POSLOAD:begin
 remaining<=position_live[fp]&block_live;
 if(((mode_q==5 || mode_q==8) || (mode_q==6 || mode_q==7)) && !refresh[fp])begin state<=STORE;held_vectors<=held_vectors+1;end
 else begin
 if(!((mode_q==7 || mode_q==8) && use_delta[fp]))for(integer i=0;i<8;i=i+1)acc[i]<=0;
 if(mode_q==3 && codes[fp]!=0)state<=PROTO_READ;
 else state<=((position_live[fp]&block_live)==0)?STORE:BASE_MAC;
 end
 end
 PROTO_READ:if(weight_allow)begin
 for(integer i=0;i<8;i=i+1)acc[i]<=prototype_mem[i][og*4+int'(codes[fp])];
 weight_words<=weight_words+1;prototype_reads<=prototype_reads+1;
 state<=(remaining==0)?STORE:BASE_MAC;
 end else weight_stalls<=weight_stalls+1;
 BASE_MAC:begin
 for(integer i=0;i<8;i=i+1)acc[i]<=add_y[i];
 mac_issues<=mac_issues+1;z_scalar_reads<=z_scalar_reads+1;
 remaining<=remaining&(remaining-8'd1);
 if((remaining&(remaining-8'd1))==0)state<=STORE;
 end
 STORE:begin
 for(integer i=0;i<8;i=i+1)p_mem[i][og*40+fp]<=acc[i];
 psum_writes<=psum_writes+1;
 if(fp<39)begin fp<=fp+1;state<=POSLOAD;end
 else if(og<11)begin og<=og+1;qfill<=0;state<=VLOAD;end
 else begin row<=0;state<=DRAIN_READ;end
 end
 DRAIN_READ:begin
 for(integer i=0;i<8;i=i+1)result_data[i*32+:32]<=p_mem[i][row];
 psum_reads<=psum_reads+1;state<=DRAIN_SEND;
 end
 DRAIN_SEND:if(result_ready)begin
 if(row==479)state<=FINISH;else begin row<=row+1;state<=DRAIN_READ;end
 end else output_stalls<=output_stalls+1;
 FINISH:begin done<=1;state<=IDLE;end
 default:state<=IDLE;
 endcase
 end
 end
endmodule
