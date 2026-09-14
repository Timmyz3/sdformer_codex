module spatial_core(
 input logic clk,reset_n,cfg_valid,start,mode,
 input logic [2:0] cfg_kind,input logic [10:0] cfg_addr,input logic [255:0] cfg_data,
 input logic source_allow,weight_allow,result_ready,
 output logic result_valid,done,output logic [8:0] result_addr,output logic [255:0] result_data,
 output logic [31:0] cycles,source_words,q1_words,q2_words,local_gathers,q1_issues,q2_issues,
 output logic [31:0] z_vector_reads,z_scalar_reads,z_writes,psum_reads,psum_writes,cache_writes,
 output logic [31:0] source_stalls,weight_stalls,output_stalls,
 output logic [31:0] transform_issues,transform_reads,transform_writes,reconstruction_issues,stripe_add_issues,cache_reads,exact_halves,
 output logic d_monitor_valid,output logic [5:0] d_monitor_addr,output logic [255:0] d_monitor_data,
 output logic [5:0] debug_state,
 output logic z_monitor_valid,z_monitor_stripe,output logic [5:0] z_monitor_addr,
 output logic [255:0] z_monitor_data
);
 typedef enum logic [5:0] {IDLE,ZCLEAR,L_START,L_LOAD,L_GATHER,CHECK,QREAD,TIMESEL,
 ZREAD,ZADD,KNEXT,ZSCAN,VLOAD,POSLOAD,BASE_MAC,STORE,DRAIN_READ,DRAIN_SEND,FINISH,XDREAD0,XDREAD1,XD0,XD1,XD2,W_INV0A,W_PREAD0,W_PADD0,W_STORE0,W_INV1A,W_PREAD1,W_PADD1,W_STORE1} state_t;
 state_t state;logic stripe,mode_q;
 logic [31:0] transform_tail[0:7];logic signed [31:0] m_aux[0:1][0:7];
 integer tx,transform_addr;logic sub_alu;logic [7:0] transform_support;
 logic [9:0] source_mem[0:1535],local_source[0:15],src_masks[0:7];
 logic signed [7:0] q1_mem[0:7][0:575],q1_hold[0:7];
 logic signed [12:0] q2_mem[0:7][0:575],qcache[0:7][0:23];
 logic q1_live[0:575],q2_live[0:575];
 logic [31:0] z_mem[0:7][0:39],z_hold[0:7],z_read_word[0:7];
 logic signed [31:0] p_mem[0:7][0:479],acc[0:7];
 logic [7:0] position_live[0:79],rank_live,scan_low,scan_high;
 logic [23:0] block_live,remaining,next_support;
 logic [39:0] pending;
 integer k,load_xy,zrow,fp,og,qfill,row,selected_time,selected_term;
 integer output_p,source_p,z_addr,weight_addr;logic [8:0] p_addr;
 logic signed [15:0] origin_y,origin_x;
 logic in_bounds,active_low,active_high,z_read_enable,p_read_enable,p_write_enable;
 logic [2:0] selected_rank;logic selected_half;
 logic signed [15:0] z_scalar;
 logic signed [18:0] multiply_lhs[0:7];logic signed [12:0] multiply_rhs[0:7];
 logic signed [31:0] product[0:7],lhs[0:7],rhs[0:7],add_y[0:7];
 logic [31:0] p_read_data[0:7],p_write_data[0:7];
 logic signed [7:0] q1_data[0:7];logic signed [12:0] q2_data[0:7];
 logic cfg_q1_live,cfg_q2_live;
 task clear_counters;
 begin
 transform_issues<=0;transform_reads<=0;transform_writes<=0;reconstruction_issues<=0;stripe_add_issues<=0;cache_reads<=0;exact_halves<=0;
 cycles<=0;source_words<=0;q1_words<=0;q2_words<=0;local_gathers<=0;q1_issues<=0;q2_issues<=0;
 z_vector_reads<=0;z_scalar_reads<=0;z_writes<=0;psum_reads<=0;psum_writes<=0;cache_writes<=0;
 source_stalls<=0;weight_stalls<=0;output_stalls<=0;
 end endtask
 always_comb begin
  in_bounds=(int'(origin_y)+load_xy/4>=0&&int'(origin_y)+load_xy/4<240&&
             int'(origin_x)+load_xy%4>=0&&int'(origin_x)+load_xy%4<320);
  cfg_q1_live=0;cfg_q2_live=0;
  for(integer l=0;l<8;l=l+1)begin
   cfg_q1_live=cfg_q1_live||(cfg_data[l*32+:8]!=0);
   cfg_q2_live=cfg_q2_live||(cfg_data[l*32+:13]!=0);
  end
  selected_time=0;for(integer t=39;t>=0;t=t-1)if(pending[t])selected_time=t;
  selected_term=0;for(integer t=23;t>=0;t=t-1)if(remaining[t])selected_term=t;
  selected_rank=3'(selected_term/3);
  output_p=fp/10;
  source_p=mode_q?(fp/10)*4+selected_term%3:(output_p/2)*4+output_p%2+selected_term%3;
  selected_half=(source_p%2)!=0;
  transform_addr=(tx/10)*20+tx%10;
  z_addr=(state==XDREAD0)?transform_addr:(state==XDREAD1)?transform_addr+10:(state==BASE_MAC)?(source_p/2)*10+fp%10:zrow;
  z_read_enable=(state==ZREAD||state==ZSCAN||state==BASE_MAC||state==XDREAD0||state==XDREAD1);
  p_addr=(state==DRAIN_READ)?9'(row):mode_q?9'(og*40+(fp/10)*20+fp%10+((state==W_PREAD1||state==W_PADD1||state==W_STORE1)?10:0)):9'(og*40+fp);
  p_read_enable=(state==DRAIN_READ)||(state==POSLOAD&&stripe&&!mode_q)||state==W_PREAD0||state==W_PREAD1;
  p_write_enable=(state==STORE||state==W_STORE0||state==W_STORE1);
  weight_addr=state==QREAD?int'(stripe)*288+k:int'(stripe)*288+og*24+qfill;
  next_support=0;
  if(mode_q)for(integer r=0;r<8;r=r+1)for(integer x=0;x<3;x=x+1)
   next_support[r*3+x]=block_live[r*3+x]&&position_live[((fp/10)*4+x)*10+fp%10][r];
  else for(integer r=0;r<8;r=r+1)for(integer x=0;x<3;x=x+1)
   next_support[r*3+x]=block_live[r*3+x]&&position_live[((output_p/2)*4+output_p%2+x)*10+fp%10][r];
  scan_low=0;scan_high=0;
  for(integer l=0;l<8;l=l+1)begin
   z_read_word[l]=(z_read_enable&&(state!=BASE_MAC||selected_rank==3'(l)))?z_mem[l][z_addr]:32'd0;
   scan_low[l]=z_read_word[l][14:0]!=0;scan_high[l]=z_read_word[l][29:15]!=0;
   p_read_data[l]=p_read_enable?p_mem[l][p_addr]:32'd0;
   p_write_data[l]=acc[l];
   q1_data[l]=(state==QREAD&&weight_allow)?q1_mem[l][weight_addr]:8'sd0;
   q2_data[l]=(state==VLOAD&&weight_allow&&rank_live[qfill/3]&&q2_live[weight_addr])?q2_mem[l][weight_addr]:13'sd0;
  end
  z_scalar=mode_q?(selected_half?$signed(z_read_word[selected_rank][31:16]):$signed(z_read_word[selected_rank][15:0])):
            (selected_half?16'($signed(z_read_word[selected_rank][29:15])):16'($signed(z_read_word[selected_rank][14:0])));
  sub_alu=(state==XD0||state==XD2||state==W_INV1A);
  for(integer l=0;l<8;l=l+1)begin
   multiply_lhs[l]=(state==BASE_MAC)?{{3{z_scalar[15]}},z_scalar}:19'sd0;
   multiply_rhs[l]=(state==BASE_MAC)?qcache[l][selected_term]:13'sd0;
   product[l]=$signed(multiply_lhs[l])*$signed(multiply_rhs[l]);
   lhs[l]=acc[l];rhs[l]=product[l];
   if(state==BASE_MAC&&mode_q&&selected_term%3!=0)lhs[l]=m_aux[selected_term%3-1][l];
   case(state)
    XD0:begin lhs[l]=32'($signed(z_hold[l][14:0]));rhs[l]=32'($signed(transform_tail[l][14:0]));end
    XD1:begin lhs[l]=32'($signed(z_hold[l][29:15]));rhs[l]=32'($signed(transform_tail[l][14:0]));end
    XD2:begin lhs[l]=32'($signed(z_hold[l][29:15]));rhs[l]=32'($signed(transform_tail[l][29:15]));end
    W_INV0A:rhs[l]=m_aux[0][l];
    W_INV1A:begin lhs[l]=m_aux[0][l];rhs[l]=m_aux[1][l];end
    W_PADD0,W_PADD1:rhs[l]=$signed(z_hold[l]);
    default:begin end
   endcase
   if(state==ZADD)begin
    lhs[l]=z_hold[l];
    rhs[l]={2'd0,(active_high?{{7{q1_hold[l][7]}},q1_hold[l]}:15'd0),
                    (active_low?{{7{q1_hold[l][7]}},q1_hold[l]}:15'd0)};
   end
  end
  transform_support=0;
  for(integer i=0;i<8;i=i+1)transform_support[i]=(add_y[i][15:0]!=0);
  d_monitor_valid=state==XD1||state==XD2;d_monitor_addr=6'(transform_addr+((state==XD2)?10:0));
  for(integer i=0;i<8;i=i+1)d_monitor_data[i*32+:32]=(state==XD2)?{16'd0,add_y[i][15:0]}:{add_y[i][15:0],acc[i][15:0]};
  result_valid=state==DRAIN_SEND;result_addr=9'(row);debug_state=state;
  z_monitor_valid=state==ZSCAN;z_monitor_stripe=stripe;z_monitor_addr=6'(zrow);
  for(integer l=0;l<8;l=l+1)z_monitor_data[l*32+:32]=z_read_word[l];
 end
 genvar l,b;
 generate for(l=0;l<8;l=l+1)begin:ALU
  for(b=0;b<32;b=b+1)begin:BIT
   wire cin;
   if(b==0)assign cin=sub_alu;
   else if(b==15)assign cin=state==ZADD?1'b0:BIT[b-1].CARRY.cout;
   else assign cin=BIT[b-1].CARRY.cout;
   assign add_y[l][b]=lhs[l][b]^(rhs[l][b]^sub_alu)^cin;
   if(b<31)begin:CARRY
    wire cout;assign cout=(lhs[l][b]&(rhs[l][b]^sub_alu))|((lhs[l][b]^(rhs[l][b]^sub_alu))&cin);
   end
  end
 end endgenerate
 always_ff @(posedge clk)begin
  if(!reset_n)begin
   state<=IDLE;stripe<=0;mode_q<=0;tx<=0;done<=0;k<=0;load_xy<=0;zrow<=0;fp<=0;og<=0;qfill<=0;row<=0;
   origin_y<=0;origin_x<=0;pending<=0;rank_live<=0;block_live<=0;remaining<=0;
   active_low<=0;active_high<=0;result_data<=0;clear_counters();
   for(integer i=0;i<8;i=i+1)begin transform_tail[i]<=0;for(integer m=0;m<2;m=m+1)m_aux[m][i]<=0;src_masks[i]<=0;q1_hold[i]<=0;z_hold[i]<=0;acc[i]<=0;end
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
     for(integer i=0;i<8;i=i+1)q2_mem[i][cfg_addr[9:0]]<=cfg_data[i*32+:13];
     q2_live[cfg_addr[9:0]]<=cfg_q2_live;
    end
    default:begin end
   endcase
   if(state!=IDLE)cycles<=cycles+1;
   case(state)
    IDLE:if(start)begin mode_q<=mode;stripe<=0;zrow<=0;rank_live<=0;block_live<=0;state<=ZCLEAR;clear_counters();end
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
     for(integer i=0;i<4;i=i+1)pending[i*10+:10]<=src_masks[i*2]|src_masks[i*2+1];
     if((src_masks[0]|src_masks[1]|src_masks[2]|src_masks[3]|src_masks[4]|src_masks[5]|src_masks[6]|src_masks[7])==0)state<=KNEXT;
     else state<=QREAD;
    end
    QREAD:if(weight_allow)begin
     for(integer i=0;i<8;i=i+1)q1_hold[i]<=q1_data[i];
     q1_words<=q1_words+1;state<=TIMESEL;
    end else weight_stalls<=weight_stalls+1;
    TIMESEL:if(pending!=0)begin
     zrow<=selected_time;active_low<=src_masks[2*(selected_time/10)][selected_time%10];
     active_high<=src_masks[2*(selected_time/10)+1][selected_time%10];state<=ZREAD;
    end else state<=KNEXT;
    ZREAD:begin
     for(integer i=0;i<8;i=i+1)z_hold[i]<=z_read_word[i];
     z_vector_reads<=z_vector_reads+1;state<=ZADD;
    end
    ZADD:begin
     for(integer i=0;i<8;i=i+1)z_mem[i][zrow]<={2'd0,add_y[i][29:0]};
     q1_issues<=q1_issues+1;z_writes<=z_writes+1;pending<=pending&(pending-40'd1);state<=TIMESEL;
    end
    KNEXT:if(k==287)begin zrow<=0;state<=ZSCAN;end
    else begin k<=k+1;state<=(k%3==2)?L_START:L_GATHER;end
    ZSCAN:begin
     position_live[(zrow/10)*20+zrow%10]<=scan_low;
     position_live[(zrow/10)*20+10+zrow%10]<=scan_high;
     rank_live<=rank_live|scan_low|scan_high;z_vector_reads<=z_vector_reads+1;
     if(zrow==39)begin og<=0;qfill<=0;if(mode_q)begin tx<=0;rank_live<=0;state<=XDREAD0;end else state<=VLOAD;end else zrow<=zrow+1;
    end
    VLOAD:if(!rank_live[qfill/3]||!q2_live[weight_addr]||weight_allow)begin
     block_live[qfill]<=rank_live[qfill/3]&&q2_live[weight_addr];
     for(integer i=0;i<8;i=i+1)qcache[i][qfill]<=q2_data[i];
     cache_writes<=cache_writes+1;
     if(rank_live[qfill/3]&&q2_live[weight_addr])q2_words<=q2_words+1;
     if(qfill==23)begin fp<=0;state<=POSLOAD;end else qfill<=qfill+1;
    end else weight_stalls<=weight_stalls+1;
    POSLOAD:begin
     for(integer i=0;i<8;i=i+1)begin
      acc[i]<=(stripe&&!mode_q)?$signed(p_read_data[i]):32'sd0;
      if(mode_q)for(integer m=0;m<2;m=m+1)m_aux[m][i]<=0;
     end
     if(stripe&&!mode_q)psum_reads<=psum_reads+1;
     remaining<=next_support;state<=(next_support==0)?(mode_q?W_INV0A:STORE):BASE_MAC;
    end
    BASE_MAC:begin
     for(integer i=0;i<8;i=i+1)begin
      if(mode_q&&selected_term%3!=0)m_aux[selected_term%3-1][i]<=add_y[i];else acc[i]<=add_y[i];
     end
     q2_issues<=q2_issues+1;cache_reads<=cache_reads+1;z_scalar_reads<=z_scalar_reads+1;remaining<=remaining&(remaining-24'd1);
     if((remaining&(remaining-24'd1))==0)state<=mode_q?W_INV0A:STORE;
    end
    STORE:begin
     psum_writes<=psum_writes+1;
     if(fp<39)begin fp<=fp+1;state<=POSLOAD;end
     else if(og<11)begin og<=og+1;qfill<=0;state<=VLOAD;end
     else if(!stripe)begin stripe<=1;zrow<=0;rank_live<=0;state<=ZCLEAR;end
     else begin row<=0;state<=DRAIN_READ;end
    end

    XDREAD0:begin
     for(integer i=0;i<8;i=i+1)z_hold[i]<=z_read_word[i];
     transform_reads<=transform_reads+1;z_vector_reads<=z_vector_reads+1;state<=XDREAD1;
    end
    XDREAD1:begin
     for(integer i=0;i<8;i=i+1)transform_tail[i]<=z_read_word[i];
     transform_reads<=transform_reads+1;z_vector_reads<=z_vector_reads+1;state<=XD0;
    end
    XD0,XD1,XD2:begin
     transform_issues<=transform_issues+1;rank_live<=rank_live|transform_support;
     if(state==XD0)begin
      for(integer i=0;i<8;i=i+1)acc[i]<=add_y[i];
      position_live[(tx/10)*40+tx%10]<=transform_support;state<=XD1;
     end else begin
      for(integer i=0;i<8;i=i+1)z_mem[i][transform_addr+((state==XD2)?10:0)]<=
       (state==XD2)?{16'd0,add_y[i][15:0]}:{add_y[i][15:0],acc[i][15:0]};
      transform_writes<=transform_writes+1;z_writes<=z_writes+1;
      position_live[(tx/10)*40+((state==XD2)?20:10)+tx%10]<=transform_support;
      if(state==XD1)state<=XD2;
      else begin
       position_live[(tx/10)*40+30+tx%10]<=0;
       if(tx==19)begin og<=0;qfill<=0;state<=VLOAD;end
       else begin tx<=tx+1;state<=XDREAD0;end
      end
     end
    end
    W_INV0A,W_INV1A:begin
     for(integer i=0;i<8;i=i+1)acc[i]<=add_y[i];
     reconstruction_issues<=reconstruction_issues+1;
     if(state==W_INV0A)state<=stripe?W_PREAD0:W_STORE0;
     else state<=stripe?W_PREAD1:W_STORE1;
    end
    W_PREAD0,W_PREAD1:begin
     for(integer i=0;i<8;i=i+1)z_hold[i]<=p_read_data[i];
     psum_reads<=psum_reads+1;state<=(state==W_PREAD0)?W_PADD0:W_PADD1;
    end
    W_PADD0,W_PADD1:begin
     for(integer i=0;i<8;i=i+1)acc[i]<=add_y[i];
     stripe_add_issues<=stripe_add_issues+1;state<=(state==W_PADD0)?W_STORE0:W_STORE1;
    end
    W_STORE0:begin psum_writes<=psum_writes+1;state<=W_INV1A;end
    W_STORE1:begin
     psum_writes<=psum_writes+1;
     if(fp<19)begin fp<=fp+1;state<=POSLOAD;end
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
 integer checked_stores;logic [479:0] checked_first,checked_second;
 always_ff @(posedge clk)begin
  if(!reset_n)begin checked_stores<=0;checked_first<=0;checked_second<=0;end
  else begin
   if(state==IDLE&&start)begin checked_stores<=0;checked_first<=0;checked_second<=0;end
   if(state==STORE||state==W_STORE0||state==W_STORE1)begin
    if(!mode_q&&int'(stripe)*480+og*40+fp!=checked_stores)$fatal(1,"psum ordering");
    if(stripe)begin
     if(!checked_first[p_addr]||checked_second[p_addr])$fatal(1,"stripe sum ownership");
     checked_second[p_addr]<=1;
    end else begin
     if(checked_first[p_addr])$fatal(1,"first stripe duplicate write");
     checked_first[p_addr]<=1;
    end
    checked_stores<=checked_stores+1;
   end
   if((state==W_PREAD0||state==W_PREAD1)&&!checked_first[p_addr])$fatal(1,"read before first stripe write");
   if(state==XD0||state==XD1||state==XD2)for(integer i=0;i<8;i=i+1)
    if(add_y[i]<-32768||add_y[i]>32767)$fatal(1,"D16 overflow");
   if(state==BASE_MAC||state==W_INV0A||state==W_INV1A||state==W_PADD0||state==W_PADD1)
    for(integer i=0;i<8;i=i+1)begin
     if(!sub_alu&&(64'($signed(lhs[i]))+64'($signed(rhs[i]))!=64'($signed(add_y[i]))))$fatal(1,"add prefix overflow");
     if(sub_alu&&(64'($signed(lhs[i]))-64'($signed(rhs[i]))!=64'($signed(add_y[i]))))$fatal(1,"subtract prefix overflow");
    end
   if(state==DRAIN_READ)for(integer j=0;j<480;j=j+1)if(!checked_second[j])$fatal(1,"missing second stripe row %0d",j);
   if(state==DRAIN_READ&&checked_stores!=960)$fatal(1,"drain before both stripes mode=%0d stores=%0d mask=%h",mode_q,checked_stores,checked_second);
   if(p_read_enable&&p_write_enable)$fatal(1,"psum double access");
   if(z_read_enable&&(z_addr<0||z_addr>=40))$fatal(1,"z address");
   if((state==QREAD&&(weight_addr<0||weight_addr>=576))||(state==VLOAD&&(weight_addr<0||weight_addr>=576)))$fatal(1,"weight address");
   if(state==ZADD)for(integer i=0;i<8;i=i+1)begin
    if(32'($signed(z_hold[i][14:0]))+(active_low?32'($signed(q1_hold[i])):32'sd0)<-16384||
       32'($signed(z_hold[i][14:0]))+(active_low?32'($signed(q1_hold[i])):32'sd0)>16383)$fatal(1,"low Z15 overflow");
    if(32'($signed(z_hold[i][29:15]))+(active_high?32'($signed(q1_hold[i])):32'sd0)<-16384||
       32'($signed(z_hold[i][29:15]))+(active_high?32'($signed(q1_hold[i])):32'sd0)>16383)$fatal(1,"high Z15 overflow");
   end
  end
 end
 `endif
endmodule
