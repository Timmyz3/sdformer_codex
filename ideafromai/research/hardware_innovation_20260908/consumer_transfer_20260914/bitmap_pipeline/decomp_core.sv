module decomp_core(
 input logic clk,reset_n,cfg_valid,
 input logic [2:0] cfg_kind,input logic [10:0] cfg_addr,input logic [255:0] cfg_data,
 input logic start,input logic [3:0] mode,input logic source_allow,weight_allow,
 output logic result_valid,input logic result_ready,
 output logic [8:0] result_addr,output logic [255:0] result_data,output logic done,
 output logic [31:0] cycles,source_words,weight_words,second_weight_words,local_source_reads,
 output logic [31:0] z_vector_reads,z_scalar_reads,z_writes,first_issues,dual_updates,
 output logic [31:0] psum_reads,psum_writes,mac_issues,
 output logic [31:0] source_stalls,weight_stalls,output_stalls,
 output logic [31:0] aux_reads,aux_writes,aux_issues,aux_weight_words,aux_events,
 output logic [31:0] bitmap_native_reads,bitmap_native_issues,
 output logic [5:0] debug_state
);
 typedef enum logic [5:0] {IDLE,ZCLEAR,L_START,L_LOAD,L_GATHER,CHECK,QREAD,TIMESEL,
 ZREAD,ZADD,KNEXT,ZSCAN,VLOAD,POSLOAD,BASE_MAC,STORE,DRAIN_READ,DRAIN_SEND,FINISH,BM_PACK,BM_POS,BM_SCAN,BM_QREAD,BM_POP,BM_STORE,BM_PIPE,BM_NATIVE_QREAD,BM_NATIVE_ADD} state_t;
 state_t state;logic [3:0] mode_q;
 logic [15:0] bitmap[0:39][0:53],bp_q[0:2][0:7][0:53],bm_hold,bpq_hold[0:7];
 logic bp_live[0:2][0:53];logic [2:0] cfg_plane_live;
 integer bm_block,bm_plane;
 logic [53:0] bm_live[0:39],bm_pending,bm_after;
 integer first_live,next_live;
 logic pipe_valid;logic [1:0] pipe_plane,execute_plane;
 logic pop_execute;integer onehot_index;logic bitmap_onehot;
 logic [9:0] q_read_addr;logic q_read_enable,bp_read_enable;
 logic signed [2:0] q_read_data[0:7];logic [15:0] bp_read_data[0:7],bitmap_read_word;

 logic signed [31:0] bm_acc[0:7];logic [4:0] pc[0:7];logic sub_alu;
 function automatic [4:0] pop16(input logic [15:0] a);
 logic [1:0] x[0:7];logic [2:0] y[0:3];logic [3:0] z[0:1];
 begin
 for(integer i=0;i<8;i=i+1)x[i]={1'b0,a[2*i]}+{1'b0,a[2*i+1]};
 for(integer i=0;i<4;i=i+1)y[i]={1'b0,x[2*i]}+{1'b0,x[2*i+1]};
 for(integer i=0;i<2;i=i+1)z[i]={1'b0,y[2*i]}+{1'b0,y[2*i+1]};
 pop16={1'b0,z[0]}+{1'b0,z[1]};
 end endfunction
 task bm_next;
 begin
 if((mode_q==13||mode_q==10))begin
  bm_pending<=bm_after;
  if(bm_after==0)state<=BM_STORE;
  else begin bm_block<=next_live;state<=BM_SCAN;end
 end else if(bm_block==53)state<=BM_STORE;else begin bm_block<=bm_block+1;state<=BM_SCAN;end
 end endtask
 task plane_next;
 begin
 if(bm_plane==2)bm_next();else begin bm_plane<=bm_plane+1;state<=BM_QREAD;end
 end endtask
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
 logic signed [12:0] scalar;
 logic signed [18:0] multiply_coefficient[0:7];
 logic signed [31:0] product[0:7],lhs[0:7],rhs[0:7],add_y[0:7];

 task clear_counters;
 begin
 bitmap_native_reads<=0;bitmap_native_issues<=0;aux_reads<=0;aux_writes<=0;aux_issues<=0;aux_weight_words<=0;aux_events<=0;
 cycles<=0;source_words<=0;weight_words<=0;second_weight_words<=0;local_source_reads<=0;
 z_vector_reads<=0;z_scalar_reads<=0;z_writes<=0;first_issues<=0;dual_updates<=0;
 psum_reads<=0;psum_writes<=0;mac_issues<=0;
 source_stalls<=0;weight_stalls<=0;output_stalls<=0;
 end
 endtask

 always_comb begin
 source_addr=11'((k/9)*16+load_xy);
 in_bounds=(int'(oy)+load_xy/4>=0 && int'(oy)+load_xy/4<240 &&
 int'(ox)+load_xy%4>=0 && int'(ox)+load_xy%4<320);
 cfg_plane_live=0;
 for(integer b0=0;b0<3;b0=b0+1)for(integer r0=0;r0<8;r0=r0+1)cfg_plane_live[b0]=cfg_plane_live[b0]|cfg_data[r0*32+b0];
 onehot_index=0;for(integer j=15;j>=0;j=j-1)if(bm_hold[j])onehot_index=j;
 q_read_enable=(state==QREAD||state==BM_NATIVE_QREAD);
 q_read_addr=(state==BM_NATIVE_QREAD)?10'(bm_block*16+onehot_index):10'(k);
 bp_read_enable=(state==BM_QREAD||state==BM_PIPE)&&(bm_plane<3);
 bitmap_read_word=(state==BM_SCAN)?bitmap[fp][bm_block]:16'd0;
 bitmap_onehot=(bitmap_read_word!=0)&&((bitmap_read_word&(bitmap_read_word-16'd1))==0);

 pop_execute=(state==BM_POP)||(state==BM_PIPE&&pipe_valid);
 execute_plane=(state==BM_PIPE)?pipe_plane:2'(bm_plane);
 sub_alu=pop_execute&&execute_plane==2;
 bm_after=bm_pending&(bm_pending-54'd1);first_live=0;next_live=0;
 for(integer j=53;j>=0;j=j-1)begin
  if(bm_live[fp][j])first_live=j;
  if(bm_after[j])next_live=j;
 end
 cfg_v_live=0;for(integer i=0;i<8;i=i+1)cfg_v_live=cfg_v_live||(cfg_data[i*32+:16]!=0);
 selected_time=0;for(integer i=39;i>=0;i=i-1)if(pending[i])selected_time=i;
 selected_rank=0;for(integer i=7;i>=0;i=i-1)if(remaining[i])selected_rank=3'(i);
 read_zrow=5'((fp/20)*10+fp%10);read_half=((fp/10)%2)!=0;
 scan_mask=0;
 if(state==ZSCAN)for(integer i=0;i<8;i=i+1)
 scan_mask[i]=read_half?(z_mem[i][read_zrow][25:13]!=0):(z_mem[i][read_zrow][12:0]!=0);
 scalar=read_half?$signed(z_mem[selected_rank][read_zrow][25:13]):$signed(z_mem[selected_rank][read_zrow][12:0]);
 for(integer l=0;l<8;l=l+1)begin
 q_read_data[l]=q_read_enable?q_mem[l][q_read_addr]:3'sd0;
 bp_read_data[l]=bp_read_enable?bp_q[bm_plane][l][bm_block]:16'd0;
 multiply_coefficient[l]={{3{qblock[selected_rank][l][15]}},qblock[selected_rank][l]};
 pc[l]=pop16(bm_hold&bpq_hold[l]);
 product[l]=$signed(multiply_coefficient[l])*$signed(scalar);
 lhs[l]=acc[l];rhs[l]=product[l];
 if(pop_execute)begin lhs[l]=bm_acc[l];rhs[l]=$signed({27'd0,pc[l]})<<<execute_plane;end
 if(state==BM_NATIVE_ADD)begin lhs[l]=bm_acc[l];rhs[l]={{29{q_hold[l][2]}},q_hold[l]};end
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
 if(b==0)assign cin=sub_alu;
 else if(b==13)assign cin=(state==ZADD)?1'b0:BIT[b-1].CARRY.cout;
 else assign cin=BIT[b-1].CARRY.cout;
 assign add_y[l][b]=lhs[l][b]^(rhs[l][b]^sub_alu)^cin;
 if(b<31)begin:CARRY
 wire cout;assign cout=(lhs[l][b]&(rhs[l][b]^sub_alu))|((lhs[l][b]^(rhs[l][b]^sub_alu))&cin);
 end
 end
 end endgenerate

 always_ff @(posedge clk)begin
 if(!reset_n)begin
 bm_block<=0;bm_plane<=0;bm_hold<=0;bm_pending<=0;pipe_valid<=0;pipe_plane<=0;
 state<=IDLE;mode_q<=14;done<=0;oy<=0;ox<=0;k<=0;fp<=0;zrow<=0;og<=0;qfill<=0;row<=0;load_xy<=0;
 rank_live<=0;block_live<=0;remaining<=0;pending<=0;result_data<=0;active_lo<=0;active_hi<=0;clear_counters();
 for(integer i=0;i<4;i=i+1)src_masks[i]<=0;
 for(integer i=0;i<8;i=i+1)begin q_hold[i]<=0;z_hold[i]<=0;acc[i]<=0;bm_acc[i]<=0;bpq_hold[i]<=0;end
 end else begin
 done<=0;
 if(cfg_valid&&state==IDLE)case(cfg_kind)
 0:source_mem[cfg_addr[10:0]]<=cfg_data[9:0];
 3:begin oy<=cfg_data[15:0];ox<=cfg_data[31:16];end
 4:begin
 for(integer i=0;i<8;i=i+1)begin
 q_mem[i][cfg_addr[9:0]]<=cfg_data[i*32+:3];
 for(integer b0=0;b0<3;b0=b0+1)bp_q[b0][i][cfg_addr[9:4]][cfg_addr[3:0]]<=cfg_data[i*32+b0];
 end
 for(integer b0=0;b0<3;b0=b0+1)
 if(cfg_addr[3:0]==0)bp_live[b0][cfg_addr[9:4]]<=cfg_plane_live[b0];
 else bp_live[b0][cfg_addr[9:4]]<=bp_live[b0][cfg_addr[9:4]]|cfg_plane_live[b0];
 end
 5:begin
 for(integer i=0;i<8;i=i+1)v_mem[i][cfg_addr[6:0]]<=cfg_data[i*32+:16];
 v_live[cfg_addr[6:0]]<=cfg_v_live;
 end
 6:k_live[cfg_addr[9:0]]<=cfg_data[0];
 default:begin end
 endcase
 if(state!=IDLE)cycles<=cycles+1;
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
 L_GATHER:if(mode_q!=14)begin
 for(integer i=0;i<4;i=i+1)src_masks[i]<=k_live[k]?local_source[(i/2+(k%9)/3)*4+i%2+k%3]:10'd0;
 if(k_live[k])local_source_reads<=local_source_reads+1;
 if(mode_q==15)state<=BM_PACK;
 else begin
  // Use the original forty bit-write banks in the native gather cycle.
  for(integer i=0;i<40;i=i+1)begin
   bitmap[i][k/16][k%16]<=k_live[k]&&local_source[(i/20+(k%9)/3)*4+(i/10)%2+k%3][i%10];
   if((mode_q==13||mode_q==10))begin
    if(k%16==0)bm_live[i][k/16]<=k_live[k]&&local_source[(i/20+(k%9)/3)*4+(i/10)%2+k%3][i%10];
    else bm_live[i][k/16]<=bm_live[i][k/16]||(k_live[k]&&local_source[(i/20+(k%9)/3)*4+(i/10)%2+k%3][i%10]);
   end
  end
  aux_writes<=aux_writes+1;state<=KNEXT;
 end
 end else if(!k_live[k])state<=KNEXT;
 else begin
 for(integer i=0;i<4;i=i+1)src_masks[i]<=local_source[(i/2+(k%9)/3)*4+i%2+k%3];
 local_source_reads<=local_source_reads+1;state<=CHECK;
 end
 BM_PACK:begin
 for(integer i=0;i<40;i=i+1)bitmap[i][k/16][k%16]<=src_masks[i/10][i%10];
 aux_writes<=aux_writes+1;state<=KNEXT;
 end
 BM_POS:begin
 for(integer i=0;i<8;i=i+1)bm_acc[i]<=0;
 if((mode_q==13||mode_q==10))begin
 bm_pending<=bm_live[fp];bm_block<=first_live;state<=(bm_live[fp]==0)?BM_STORE:BM_SCAN;
 end else begin bm_block<=0;state<=BM_SCAN;end
 end
 BM_SCAN:begin
 bm_hold<=bitmap_read_word;aux_reads<=aux_reads+1;
 if(bitmap_read_word==0)bm_next();else begin bm_plane<=0;pipe_valid<=0;state<=(mode_q==10&&bitmap_onehot)?BM_NATIVE_QREAD:((mode_q==12||(mode_q==13||mode_q==10))?BM_PIPE:BM_QREAD);end
 end
 BM_QREAD:if(!bp_live[bm_plane][bm_block])plane_next();
 else if(weight_allow)begin
 for(integer i=0;i<8;i=i+1)bpq_hold[i]<=bp_read_data[i];
 weight_words<=weight_words+1;aux_weight_words<=aux_weight_words+1;state<=BM_POP;
 end else weight_stalls<=weight_stalls+1;
 BM_POP:begin
 for(integer i=0;i<8;i=i+1)bm_acc[i]<=add_y[i];
 first_issues<=first_issues+1;aux_issues<=aux_issues+1;plane_next();
 end
 BM_PIPE:begin
  // One plane read can overlap one older pop/ALU issue. The original
  // bpq_hold is a single response register; no second memory read is added.
  if(pipe_valid)begin
   for(integer i=0;i<8;i=i+1)bm_acc[i]<=add_y[i];
   first_issues<=first_issues+1;aux_issues<=aux_issues+1;
  end
  pipe_valid<=0;
  if(bm_plane<3)begin
   if(!bp_live[bm_plane][bm_block])bm_plane<=bm_plane+1;
   else if(weight_allow)begin
    for(integer i=0;i<8;i=i+1)bpq_hold[i]<=bp_read_data[i];
    pipe_plane<=2'(bm_plane);pipe_valid<=1;bm_plane<=bm_plane+1;
    weight_words<=weight_words+1;aux_weight_words<=aux_weight_words+1;
    if(pipe_valid)aux_events<=aux_events+1;
   end else weight_stalls<=weight_stalls+1;
  end else bm_next();
 end
 BM_NATIVE_QREAD:if(weight_allow)begin
 for(integer i=0;i<8;i=i+1)q_hold[i]<=q_read_data[i];
 weight_words<=weight_words+1;bitmap_native_reads<=bitmap_native_reads+1;state<=BM_NATIVE_ADD;
 end else weight_stalls<=weight_stalls+1;
 BM_NATIVE_ADD:begin
 for(integer i=0;i<8;i=i+1)bm_acc[i]<=add_y[i];
 first_issues<=first_issues+1;bitmap_native_issues<=bitmap_native_issues+1;bm_next();
 end
 BM_STORE:begin
 for(integer i=0;i<8;i=i+1)
 if(read_half)z_mem[i][read_zrow][25:13]<=bm_acc[i][12:0];else z_mem[i][read_zrow][12:0]<=bm_acc[i][12:0];
 z_writes<=z_writes+1;
 if(fp==39)begin fp<=0;state<=ZSCAN;end else begin fp<=fp+1;state<=BM_POS;end
 end
 CHECK:begin
 pending<={20'd0,(src_masks[2]|src_masks[3]),(src_masks[0]|src_masks[1])};
 state<=((src_masks[0]|src_masks[1]|src_masks[2]|src_masks[3])==0)?KNEXT:QREAD;
 end
 QREAD:if(weight_allow)begin
 for(integer i=0;i<8;i=i+1)q_hold[i]<=q_read_data[i];
 weight_words<=weight_words+1;state<=TIMESEL;
 end else weight_stalls<=weight_stalls+1;
 TIMESEL:if(pending!=0)begin
 begin
 zrow<=selected_time;
 active_lo<=src_masks[(selected_time/10)*2][selected_time%10];
 active_hi<=src_masks[(selected_time/10)*2+1][selected_time%10];
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
 KNEXT:if(k==863)begin fp<=0;state<=(mode_q!=14)?BM_POS:ZSCAN;end
 else begin k<=k+1;state<=(k%9==8)?L_START:L_GATHER;end
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
 for(integer i=0;i<8;i=i+1)acc[i]<=0;
 remaining<=position_live[fp]&block_live;
 state<=((position_live[fp]&block_live)==0)?STORE:BASE_MAC;
 end
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
