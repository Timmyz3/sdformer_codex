module spatial_core(
 input logic clk,reset_n,cfg_valid,start,
 input logic [2:0] cfg_kind,input logic [10:0] cfg_addr,input logic [255:0] cfg_data,
 input logic source_allow,weight_allow,result_ready,
 output logic result_valid,done,output logic [8:0] result_addr,output logic [255:0] result_data,
 output logic [31:0] cycles,source_words,q1_words,q2_words,local_gathers,q1_issues,q2_issues,
 output logic [31:0] z_vector_reads,z_scalar_reads,z_writes,psum_reads,psum_writes,cache_writes,
 output logic [31:0] source_stalls,weight_stalls,output_stalls,count_constructs,count2_fields,count_nonzero_fields,
 output logic [5:0] debug_state,
 output logic z_monitor_valid,z_monitor_stripe,output logic [5:0] z_monitor_addr,
 output logic [255:0] z_monitor_data
);
 typedef enum logic [5:0] {IDLE,ZCLEAR,L_START,L_LOAD,L_GATHER,CHECK,QREAD,TIMESEL,
 ZREAD,ZADD,KNEXT,ZSCAN,VLOAD,POSLOAD,BASE_MAC,STORE,DRAIN_READ,DRAIN_SEND,FINISH} state_t;
 state_t state;logic stripe;
 logic [9:0] source_mem[0:1535],local_source[0:15],src_masks[0:7];
 logic [9:0] count_nonzero[0:7],count_two[0:7]; // combinational wires, no count SRAM
 logic signed [7:0] q1_mem[0:7][0:575],q1_hold[0:7];
 logic signed [12:0] q2_mem[0:7][0:383],qcache[0:7][0:15];
 logic q1_live[0:575],q2_live[0:383];
 logic [31:0] z_mem[0:7][0:39],z_hold[0:7],z_read_word[0:7];
 logic signed [31:0] p_mem[0:7][0:479],acc[0:7];
 logic [7:0] position_live[0:79],rank_live,scan_low,scan_high;
 logic [15:0] block_live,remaining,next_support;
 logic [39:0] pending;
 integer k,load_xy,zrow,fp,og,qfill,row,selected_time,selected_term;
 integer output_p,source_p,z_addr,weight_addr;logic [8:0] p_addr;
 logic signed [15:0] origin_y,origin_x;
 logic [1:0] active_low,active_high;
 logic in_bounds,z_read_enable,p_read_enable,p_write_enable;
 logic [2:0] selected_rank;logic selected_half;
 logic signed [15:0] z_scalar;
 logic signed [18:0] multiply_lhs[0:7];logic signed [12:0] multiply_rhs[0:7];
 logic signed [31:0] product[0:7],lhs[0:7],rhs[0:7],add_y[0:7];
 logic [31:0] p_read_data[0:7],p_write_data[0:7];
 logic signed [7:0] q1_data[0:7];logic signed [12:0] q2_data[0:7];
 logic cfg_q1_live,cfg_q2_live;
 task clear_counters;
 begin
 cycles<=0;source_words<=0;q1_words<=0;q2_words<=0;local_gathers<=0;q1_issues<=0;q2_issues<=0;
 z_vector_reads<=0;z_scalar_reads<=0;z_writes<=0;psum_reads<=0;psum_writes<=0;cache_writes<=0;
 source_stalls<=0;weight_stalls<=0;output_stalls<=0;count_constructs<=0;count2_fields<=0;count_nonzero_fields<=0;
 end endtask
 always_comb begin
  in_bounds=(int'(origin_y)+load_xy/4>=0&&int'(origin_y)+load_xy/4<240&&
             int'(origin_x)+load_xy%4>=0&&int'(origin_x)+load_xy%4<320);
  cfg_q1_live=0;cfg_q2_live=0;
  for(integer l=0;l<8;l=l+1)begin
   cfg_q1_live=cfg_q1_live||(cfg_data[l*32+:8]!=0);
   cfg_q2_live=cfg_q2_live||(cfg_data[l*32+:13]!=0);
  end
  count_nonzero[3]=0;count_nonzero[7]=0;count_two[3]=0;count_two[7]=0;
  for(integer y=0;y<2;y=y+1)for(integer x=0;x<3;x=x+1)begin
   count_nonzero[y*4+x]=src_masks[y*4+x]|src_masks[y*4+x+1];
   count_two[y*4+x]=src_masks[y*4+x]&src_masks[y*4+x+1];
  end
  selected_time=0;for(integer t=39;t>=0;t=t-1)if(pending[t])selected_time=t;
  selected_term=0;for(integer t=15;t>=0;t=t-1)if(remaining[t])selected_term=t;
  selected_rank=3'(selected_term/2);
  output_p=fp/10;
  source_p=(output_p/2)*4+output_p%2+selected_term%2;
  selected_half=(source_p%2)!=0;
  z_addr=(state==BASE_MAC)?(source_p/2)*10+fp%10:zrow;
  z_read_enable=(state==ZREAD||state==ZSCAN||state==BASE_MAC);
  p_addr=(state==DRAIN_READ)?9'(row):9'(og*40+fp);
  p_read_enable=(state==DRAIN_READ)||(state==POSLOAD&&stripe);
  p_write_enable=(state==STORE);
  weight_addr=state==QREAD?int'(stripe)*288+k:int'(stripe)*192+og*16+qfill;
  next_support=0;
  for(integer r=0;r<8;r=r+1)for(integer x=0;x<2;x=x+1)
   next_support[r*2+x]=block_live[r*2+x]&&position_live[((output_p/2)*4+output_p%2+x)*10+fp%10][r];
  scan_low=0;scan_high=0;
  for(integer l=0;l<8;l=l+1)begin
   z_read_word[l]=(z_read_enable&&(state!=BASE_MAC||selected_rank==3'(l)))?z_mem[l][z_addr]:32'd0;
   scan_low[l]=z_read_word[l][15:0]!=0;scan_high[l]=z_read_word[l][31:16]!=0;
   p_read_data[l]=p_read_enable?p_mem[l][p_addr]:32'd0;
   p_write_data[l]=acc[l];
   q1_data[l]=(state==QREAD&&weight_allow)?q1_mem[l][weight_addr]:8'sd0;
   q2_data[l]=(state==VLOAD&&weight_allow&&rank_live[qfill/2]&&q2_live[weight_addr])?q2_mem[l][weight_addr]:13'sd0;
  end
  z_scalar=selected_half?$signed(z_read_word[selected_rank][31:16]):$signed(z_read_word[selected_rank][15:0]);
  for(integer l=0;l<8;l=l+1)begin
   multiply_lhs[l]=(state==BASE_MAC)?{{3{z_scalar[15]}},z_scalar}:19'sd0;
   multiply_rhs[l]=(state==BASE_MAC)?qcache[l][selected_term]:13'sd0;
   product[l]=$signed(multiply_lhs[l])*$signed(multiply_rhs[l]);
   lhs[l]=acc[l];rhs[l]=product[l];
   if(state==ZADD)begin
    lhs[l]=z_hold[l];
    rhs[l]={(active_high==2?{{7{q1_hold[l][7]}},q1_hold[l],1'b0}:
                         active_high==1?{{8{q1_hold[l][7]}},q1_hold[l]}:16'd0),
            (active_low==2?{{7{q1_hold[l][7]}},q1_hold[l],1'b0}:
                         active_low==1?{{8{q1_hold[l][7]}},q1_hold[l]}:16'd0)};
   end
  end
  result_valid=state==DRAIN_SEND;result_addr=9'(row);debug_state=state;
  z_monitor_valid=state==ZSCAN;z_monitor_stripe=stripe;z_monitor_addr=6'(zrow);
  for(integer l=0;l<8;l=l+1)z_monitor_data[l*32+:32]=z_read_word[l];
 end
 genvar l,b;
 generate for(l=0;l<8;l=l+1)begin:ALU
  for(b=0;b<32;b=b+1)begin:BIT
   wire cin;
   if(b==0)assign cin=1'b0;
   else if(b==16)assign cin=state==ZADD?1'b0:BIT[b-1].CARRY.cout;
   else assign cin=BIT[b-1].CARRY.cout;
   assign add_y[l][b]=lhs[l][b]^rhs[l][b]^cin;
   if(b<31)begin:CARRY
    wire cout;assign cout=(lhs[l][b]&rhs[l][b])|((lhs[l][b]^rhs[l][b])&cin);
   end
  end
 end endgenerate
 always_ff @(posedge clk)begin
  if(!reset_n)begin
   state<=IDLE;stripe<=0;done<=0;k<=0;load_xy<=0;zrow<=0;fp<=0;og<=0;qfill<=0;row<=0;
   origin_y<=0;origin_x<=0;pending<=0;rank_live<=0;block_live<=0;remaining<=0;
   active_low<=0;active_high<=0;result_data<=0;clear_counters();
   for(integer i=0;i<8;i=i+1)begin src_masks[i]<=0;q1_hold[i]<=0;z_hold[i]<=0;acc[i]<=0;end
  end else begin
   done<=0;
   if(p_write_enable)for(integer i=0;i<8;i=i+1)p_mem[i][p_addr]<=p_write_data[i];
   if(cfg_valid&&state==IDLE)case(cfg_kind)
    0:source_mem[cfg_addr[10:0]]<=cfg_data[9:0];
    3:begin origin_y<=cfg_data[15:0];origin_x<=cfg_data[31:16];end
    4:begin
     for(integer i=0;i<8;i=i+1)q1_mem[i][cfg_addr[9:0]]<=cfg_data[i*32+:8];
     q1_live[cfg_addr[9:0]]<=cfg_q1_live;
    end
    5:begin
     for(integer i=0;i<8;i=i+1)q2_mem[i][cfg_addr[8:0]]<=cfg_data[i*32+:13];
     q2_live[cfg_addr[8:0]]<=cfg_q2_live;
    end
    default:begin end
   endcase
   if(state!=IDLE)cycles<=cycles+1;
   case(state)
    IDLE:if(start)begin stripe<=0;zrow<=0;rank_live<=0;block_live<=0;state<=ZCLEAR;clear_counters();end
    ZCLEAR:begin
     for(integer i=0;i<8;i=i+1)z_mem[i][zrow]<=0;
     z_writes<=z_writes+1;
     if(zrow==39)begin k<=0;state<=L_START;end else zrow<=zrow+1;
    end
    L_START:begin load_xy<=0;state<=L_LOAD;end
    L_LOAD:if(!in_bounds||source_allow)begin
     local_source[load_xy]<=in_bounds?source_mem[(k/3)*16+load_xy]:10'd0;
     if(in_bounds)source_words<=source_words+1;
     if(load_xy==15)state<=L_GATHER;else load_xy<=load_xy+1;
    end else source_stalls<=source_stalls+1;
    L_GATHER:if(!q1_live[int'(stripe)*288+k])state<=KNEXT;
    else begin
     for(integer i=0;i<8;i=i+1)src_masks[i]<=local_source[(i/4+k%3)*4+i%4];
     local_gathers<=local_gathers+1;state<=CHECK;
    end
    CHECK:begin
     count_constructs<=count_constructs+1;
     for(integer i=0;i<4;i=i+1)pending[i*10+:10]<=count_nonzero[i*2]|count_nonzero[i*2+1];
     if((src_masks[0]|src_masks[1]|src_masks[2]|src_masks[3]|src_masks[4]|src_masks[5]|src_masks[6]|src_masks[7])==0)state<=KNEXT;
     else state<=QREAD;
    end
    QREAD:if(weight_allow)begin
     for(integer i=0;i<8;i=i+1)q1_hold[i]<=q1_data[i];
     q1_words<=q1_words+1;state<=TIMESEL;
    end else weight_stalls<=weight_stalls+1;
    TIMESEL:if(pending!=0)begin
     zrow<=selected_time;active_low<=count_two[2*(selected_time/10)][selected_time%10]?2:count_nonzero[2*(selected_time/10)][selected_time%10]?1:0;
     active_high<=count_two[2*(selected_time/10)+1][selected_time%10]?2:count_nonzero[2*(selected_time/10)+1][selected_time%10]?1:0;state<=ZREAD;
    end else state<=KNEXT;
    ZREAD:begin
     for(integer i=0;i<8;i=i+1)z_hold[i]<=z_read_word[i];
     z_vector_reads<=z_vector_reads+1;state<=ZADD;
    end
    ZADD:begin
     count2_fields<=count2_fields+32'(active_low==2)+32'(active_high==2);
     count_nonzero_fields<=count_nonzero_fields+32'(active_low!=0)+32'(active_high!=0);
     for(integer i=0;i<8;i=i+1)z_mem[i][zrow]<=add_y[i];
     q1_issues<=q1_issues+1;z_writes<=z_writes+1;pending<=pending&(pending-40'd1);state<=TIMESEL;
    end
    KNEXT:if(k==287)begin zrow<=0;state<=ZSCAN;end
    else begin k<=k+1;state<=(k%3==2)?L_START:L_GATHER;end
    ZSCAN:begin
     position_live[(zrow/10)*20+zrow%10]<=scan_low;
     position_live[(zrow/10)*20+10+zrow%10]<=scan_high;
     rank_live<=rank_live|scan_low|scan_high;z_vector_reads<=z_vector_reads+1;
     if(zrow==39)begin og<=0;qfill<=0;state<=VLOAD;end else zrow<=zrow+1;
    end
    VLOAD:if(!rank_live[qfill/2]||!q2_live[weight_addr]||weight_allow)begin
     block_live[qfill]<=rank_live[qfill/2]&&q2_live[weight_addr];
     for(integer i=0;i<8;i=i+1)qcache[i][qfill]<=q2_data[i];
     cache_writes<=cache_writes+1;
     if(rank_live[qfill/2]&&q2_live[weight_addr])q2_words<=q2_words+1;
     if(qfill==15)begin fp<=0;state<=POSLOAD;end else qfill<=qfill+1;
    end else weight_stalls<=weight_stalls+1;
    POSLOAD:begin
     for(integer i=0;i<8;i=i+1)acc[i]<=stripe?$signed(p_read_data[i]):32'sd0;
     if(stripe)psum_reads<=psum_reads+1;
     remaining<=next_support;state<=(next_support==0)?STORE:BASE_MAC;
    end
    BASE_MAC:begin
     for(integer i=0;i<8;i=i+1)acc[i]<=add_y[i];
     q2_issues<=q2_issues+1;z_scalar_reads<=z_scalar_reads+1;remaining<=remaining&(remaining-16'd1);
     if((remaining&(remaining-16'd1))==0)state<=STORE;
    end
    STORE:begin
     psum_writes<=psum_writes+1;
     if(fp<39)begin fp<=fp+1;state<=POSLOAD;end
     else if(og<11)begin og<=og+1;qfill<=0;state<=VLOAD;end
     else if(!stripe)begin stripe<=1;zrow<=0;rank_live<=0;state<=ZCLEAR;end
     else begin row<=0;state<=DRAIN_READ;end
    end
    DRAIN_READ:begin
     for(integer i=0;i<8;i=i+1)result_data[i*32+:32]<=p_read_data[i];
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
 `ifdef VERILATOR
 integer checked_stores;
 always_ff @(posedge clk)begin
  if(!reset_n)checked_stores<=0;
  else begin
   if(state==IDLE&&start)checked_stores<=0;
   if(state==STORE)begin
    if(int'(stripe)*480+og*40+fp!=checked_stores)$fatal(1,"psum ordering");
    checked_stores<=checked_stores+1;
   end
   if(state==DRAIN_READ&&checked_stores!=960)$fatal(1,"drain before both stripes");
   if(p_read_enable&&p_write_enable)$fatal(1,"psum double access");
   if(z_read_enable&&(z_addr<0||z_addr>=40))$fatal(1,"z address");
   if((state==QREAD&&(weight_addr<0||weight_addr>=576))||(state==VLOAD&&(weight_addr<0||weight_addr>=384)))$fatal(1,"weight address");
   if(state==ZADD)begin
    if(active_low>2||active_high>2)$fatal(1,"invalid source count");
    for(integer i=0;i<8;i=i+1)begin
     if(32'($signed(z_hold[i][15:0]))+32'($signed(rhs[i][15:0]))<-32768||
        32'($signed(z_hold[i][15:0]))+32'($signed(rhs[i][15:0]))>32767)$fatal(1,"low E16 overflow");
     if(32'($signed(z_hold[i][31:16]))+32'($signed(rhs[i][31:16]))<-32768||
        32'($signed(z_hold[i][31:16]))+32'($signed(rhs[i][31:16]))>32767)$fatal(1,"high E16 overflow");
    end
   end
   if(state==BASE_MAC)for(integer i=0;i<8;i=i+1)
    if(64'($signed(lhs[i]))+64'($signed(rhs[i]))!=64'($signed(add_y[i])))$fatal(1,"Q2 signed32 prefix overflow");
  end
 end
 `endif
endmodule
