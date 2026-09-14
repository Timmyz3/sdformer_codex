module direct_core(
 input logic clk,reset_n,cfg_valid,start,
 input logic [2:0] cfg_kind,input logic [13:0] cfg_addr,input logic [255:0] cfg_data,
 input logic source_allow,weight_allow,result_ready,
 output logic result_valid,done,output logic [8:0] result_addr,output logic [255:0] result_data,
 output logic [31:0] cycles,source_words,weight_words,local_gathers,add_issues,
 output logic [31:0] psum_reads,psum_writes,psum_clears,
 output logic [31:0] source_stalls,weight_stalls,output_stalls,
 output logic [5:0] debug_state
);
 typedef enum logic [5:0] {IDLE,PCLEAR,L_START,L_LOAD,GATHER,WREAD,PREAD,PADD,KNEXT,DRAIN_READ,DRAIN_SEND,FINISH} state_t;
 state_t state;
 logic [9:0] source_mem[0:1535],local_source[0:15];
 logic signed [31:0] weight_mem[0:7][0:10367],weight_hold[0:7];
 logic weight_live[0:10367];
 logic signed [31:0] p_mem[0:7][0:479],p_hold[0:7],p_read_data[0:7],add_result[0:7];
 logic [39:0] support,pending,gather_mask;
 logic signed [15:0] origin_y,origin_x;
 integer k,og,load_xy,row,selected_time,weight_address;
 logic [8:0] p_address;
 logic in_bounds,cfg_weight_live,p_read_enable,p_write_enable;
 logic signed [31:0] weight_data[0:7];
 always_comb begin
  in_bounds=(int'(origin_y)+load_xy/4>=0&&int'(origin_y)+load_xy/4<240&&
             int'(origin_x)+load_xy%4>=0&&int'(origin_x)+load_xy%4<320);
  gather_mask=0;
  for(integer p=0;p<4;p=p+1)
   gather_mask[p*10+:10]=local_source[(p/2+(k%9)/3)*4+p%2+k%3];
  selected_time=0;for(integer t=39;t>=0;t=t-1)if(pending[t])selected_time=t;
  weight_address=og*864+k;
  p_address=(state==PCLEAR||state==DRAIN_READ)?9'(row):9'(og*40+selected_time);
  p_read_enable=state==PREAD||state==DRAIN_READ;
  p_write_enable=state==PCLEAR||state==PADD;
  cfg_weight_live=0;
  for(integer l=0;l<8;l=l+1)begin
   cfg_weight_live=cfg_weight_live||(cfg_data[l*32+:32]!=0);
   p_read_data[l]=p_read_enable?p_mem[l][p_address]:32'sd0;
   weight_data[l]=(state==WREAD&&support!=0&&weight_live[weight_address]&&weight_allow)?weight_mem[l][weight_address]:32'sd0;
   add_result[l]=p_hold[l]+weight_hold[l];
  end
  result_valid=state==DRAIN_SEND;result_addr=9'(row);debug_state=state;
 end
 task clear_counters;
 begin cycles<=0;source_words<=0;weight_words<=0;local_gathers<=0;add_issues<=0;
  psum_reads<=0;psum_writes<=0;psum_clears<=0;source_stalls<=0;weight_stalls<=0;output_stalls<=0;
 end endtask
 always_ff @(posedge clk)begin
  if(!reset_n)begin
   state<=IDLE;done<=0;k<=0;og<=0;load_xy<=0;row<=0;support<=0;pending<=0;
   origin_y<=0;origin_x<=0;result_data<=0;clear_counters();
   for(integer l=0;l<8;l=l+1)begin p_hold[l]<=0;weight_hold[l]<=0;end
  end else begin
   done<=0;
   if(p_write_enable)for(integer l=0;l<8;l=l+1)p_mem[l][p_address]<=state==PCLEAR?32'sd0:add_result[l];
   if(cfg_valid&&state==IDLE)case(cfg_kind)
    0:source_mem[cfg_addr[10:0]]<=cfg_data[9:0];
    3:begin origin_y<=cfg_data[15:0];origin_x<=cfg_data[31:16];end
    4:begin
     for(integer l=0;l<8;l=l+1)weight_mem[l][cfg_addr]<=cfg_data[l*32+:32];
     weight_live[cfg_addr]<=cfg_weight_live;
    end
    default:begin end
   endcase
   if(state!=IDLE)cycles<=cycles+1;
   case(state)
    IDLE:if(start)begin row<=0;k<=0;og<=0;state<=PCLEAR;clear_counters();end
    PCLEAR:begin
     psum_writes<=psum_writes+1;psum_clears<=psum_clears+1;
     if(row==479)state<=L_START;else row<=row+1;
    end
    L_START:begin load_xy<=0;state<=L_LOAD;end
    L_LOAD:if(!in_bounds||source_allow)begin
     local_source[load_xy]<=in_bounds?source_mem[(k/9)*16+load_xy]:10'd0;
     if(in_bounds)source_words<=source_words+1;
     if(load_xy==15)state<=GATHER;else load_xy<=load_xy+1;
    end else source_stalls<=source_stalls+1;
    GATHER:begin support<=gather_mask;og<=0;local_gathers<=local_gathers+1;state<=WREAD;end
    WREAD:if(support==0)state<=KNEXT;
    else if(!weight_live[weight_address])begin
     if(og==11)state<=KNEXT;else og<=og+1;
    end else if(weight_allow)begin
     for(integer l=0;l<8;l=l+1)weight_hold[l]<=weight_data[l];
     weight_words<=weight_words+1;pending<=support;state<=PREAD;
    end else weight_stalls<=weight_stalls+1;
    PREAD:begin
     for(integer l=0;l<8;l=l+1)p_hold[l]<=p_read_data[l];
     psum_reads<=psum_reads+1;state<=PADD;
    end
    PADD:begin
     add_issues<=add_issues+1;psum_writes<=psum_writes+1;pending<=pending&(pending-40'd1);
     if((pending&(pending-40'd1))!=0)state<=PREAD;
     else if(og<11)begin og<=og+1;state<=WREAD;end
     else state<=KNEXT;
    end
    KNEXT:if(k==863)begin row<=0;state<=DRAIN_READ;end
    else begin k<=k+1;state<=(k%9==8)?L_START:GATHER;end
    DRAIN_READ:begin
     for(integer l=0;l<8;l=l+1)result_data[l*32+:32]<=p_read_data[l];
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
 always_ff @(posedge clk)if(reset_n)begin
  if(p_read_enable&&p_write_enable)$fatal(1,"psum simultaneous read/write");
  if((p_read_enable||p_write_enable)&&p_address>=480)$fatal(1,"psum address");
  if(state==WREAD&&(weight_address<0||weight_address>=10368))$fatal(1,"weight address");
  if(state==PREAD||state==PADD)begin
   if(pending==0)$fatal(1,"empty update");
   if(psum_clears!=480)$fatal(1,"update before complete clear");
  end
  if(state==PADD)for(integer l=0;l<8;l=l+1)
   if(($signed({p_hold[l][31],p_hold[l]})+$signed({weight_hold[l][31],weight_hold[l]}))!=
       $signed({add_result[l][31],add_result[l]}))$fatal(1,"signed32 prefix overflow");
  if(state==DRAIN_READ&&psum_clears!=480)$fatal(1,"drain before clear");
 end
 `endif
endmodule
