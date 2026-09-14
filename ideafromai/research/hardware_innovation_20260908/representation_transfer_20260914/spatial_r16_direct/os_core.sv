module os_core(
 input logic clk,reset_n,cfg_valid,start,
 input logic [2:0] cfg_kind,input logic [13:0] cfg_addr,input logic [255:0] cfg_data,
 input logic source_allow,weight_allow,result_ready,
 output logic result_valid,done,output logic [8:0] result_addr,output logic [255:0] result_data,
 output logic [31:0] cycles,source_words,weight_words,local_gathers,add_issues,
 output logic [31:0] psum_reads,psum_writes,psum_clears,bitmap_reads,bitmap_writes,
 output logic [31:0] source_stalls,weight_stalls,output_stalls,
 output logic [5:0] debug_state
);
 typedef enum logic [5:0] {IDLE,L_START,L_LOAD,BITS,BWRITE,OTSTART,WORD_READ,MAC,STORE,NEXTP,DRAIN_READ,DRAIN_SEND,FINISH} state_t;
 state_t state;
 logic [9:0] source_mem[0:1535],local_source[0:15];
 logic signed [31:0] weight_mem[0:7][0:10367];logic weight_live[0:10367];
 logic signed [31:0] p_mem[0:7][0:479],phase_hold[0:9],weight_data[0:7],add_result[0:7];
 // Same physical 8-bank x 40-row x 32-bit store as factor Z. Each logical
 // 64-bit bitmap row maps to an adjacent bank pair at one common address.
 logic [31:0] bitmap_mem[0:7][0:39],pending;
 logic [26:0] word_live[0:9],remaining_words;
 logic signed [15:0] origin_y,origin_x;
 integer k,output_p,og,timestep,load_xy,commit_group,row,selected_word,selected_bit,word_hold;
 integer weight_address,bitmap_address;logic [8:0] p_address;
 logic in_bounds,cfg_weight_live,p_read_enable,p_write_enable,bitmap_read_enable,bitmap_write_enable;
 always_comb begin
  in_bounds=(int'(origin_y)+load_xy/4>=0&&int'(origin_y)+load_xy/4<240&&
             int'(origin_x)+load_xy%4>=0&&int'(origin_x)+load_xy%4<320);
  selected_word=0;for(integer b=26;b>=0;b=b-1)if(remaining_words[b])selected_word=b;
  selected_bit=0;for(integer b=31;b>=0;b=b-1)if(pending[b])selected_bit=b;
  weight_address=og*864+word_hold*32+selected_bit;
  bitmap_address=(state==BWRITE)?commit_group*27+k/32:(timestep/2)*27+selected_word;
  bitmap_read_enable=state==WORD_READ;bitmap_write_enable=state==BWRITE;
  p_address=state==DRAIN_READ?9'(row):9'(og*40+output_p*10+timestep);
  p_read_enable=state==DRAIN_READ;p_write_enable=state==STORE;
  cfg_weight_live=0;
  for(integer l=0;l<8;l=l+1)begin
   cfg_weight_live=cfg_weight_live||(cfg_data[l*32+:32]!=0);
   weight_data[l]=(state==MAC&&weight_live[weight_address]&&weight_allow)?weight_mem[l][weight_address]:32'sd0;
   add_result[l]=phase_hold[l]+weight_data[l];
  end
  result_valid=state==DRAIN_SEND;result_addr=9'(row);debug_state=state;
 end
 task clear_counters;
 begin cycles<=0;source_words<=0;weight_words<=0;local_gathers<=0;add_issues<=0;
  psum_reads<=0;psum_writes<=0;psum_clears<=0;bitmap_reads<=0;bitmap_writes<=0;
  source_stalls<=0;weight_stalls<=0;output_stalls<=0;
 end endtask
 always_ff @(posedge clk)begin
  if(!reset_n)begin
   state<=IDLE;done<=0;k<=0;output_p<=0;og<=0;timestep<=0;load_xy<=0;commit_group<=0;row<=0;
   origin_y<=0;origin_x<=0;pending<=0;remaining_words<=0;word_hold<=0;result_data<=0;clear_counters();
   for(integer l=0;l<10;l=l+1)phase_hold[l]<=0;
  end else begin
   done<=0;
   if(p_write_enable)for(integer l=0;l<8;l=l+1)p_mem[l][p_address]<=phase_hold[l];
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
    IDLE:if(start)begin output_p<=0;k<=0;state<=L_START;clear_counters();end
    L_START:begin load_xy<=0;state<=L_LOAD;end
    L_LOAD:if(!in_bounds||source_allow)begin
     local_source[load_xy]<=in_bounds?source_mem[(k/9)*16+load_xy]:10'd0;
     if(in_bounds)source_words<=source_words+1;
     if(load_xy==15)state<=BITS;else load_xy<=load_xy+1;
    end else source_stalls<=source_stalls+1;
    BITS:begin
     for(integer t=0;t<10;t=t+1)phase_hold[t][k%32]<=local_source[(output_p/2+(k%9)/3)*4+output_p%2+k%3][t];
     local_gathers<=local_gathers+1;
     if(k%32==31)begin commit_group<=0;state<=BWRITE;end
     else begin k<=k+1;state<=(k%9==8)?L_START:BITS;end
    end
    BWRITE:begin
     for(integer b=0;b<2;b=b+1)begin
      bitmap_mem[(bitmap_address%4)*2+b][bitmap_address/4]<=phase_hold[commit_group*2+b];
      word_live[commit_group*2+b][k/32]<=phase_hold[commit_group*2+b]!=0;
     end
     bitmap_writes<=bitmap_writes+1;
     if(commit_group<4)commit_group<=commit_group+1;
     else if(k==863)begin og<=0;timestep<=0;state<=OTSTART;end
     else begin k<=k+1;state<=(k%9==8)?L_START:BITS;end
    end
    OTSTART:begin
     for(integer l=0;l<8;l=l+1)phase_hold[l]<=0;
     remaining_words<=word_live[timestep];state<=(word_live[timestep]==0)?STORE:WORD_READ;
    end
    WORD_READ:begin
     pending<=bitmap_mem[(bitmap_address%4)*2+timestep%2][bitmap_address/4];word_hold<=selected_word;
     remaining_words<=remaining_words&(remaining_words-27'd1);
     bitmap_reads<=bitmap_reads+1;state<=MAC;
    end
    MAC:if(!weight_live[weight_address]||weight_allow)begin
     if(weight_live[weight_address])begin
      for(integer l=0;l<8;l=l+1)phase_hold[l]<=add_result[l];
      weight_words<=weight_words+1;add_issues<=add_issues+1;
     end
     pending<=pending&(pending-32'd1);
     if((pending&(pending-32'd1))==0)state<=(remaining_words==0)?STORE:WORD_READ;
    end else weight_stalls<=weight_stalls+1;
    STORE:begin
     psum_writes<=psum_writes+1;
     if(timestep<9)begin timestep<=timestep+1;state<=OTSTART;end
     else if(og<11)begin og<=og+1;timestep<=0;state<=OTSTART;end
     else state<=NEXTP;
    end
    NEXTP:if(output_p<3)begin output_p<=output_p+1;k<=0;state<=L_START;end
    else begin row<=0;state<=DRAIN_READ;end
    DRAIN_READ:begin
     for(integer l=0;l<8;l=l+1)result_data[l*32+:32]<=p_mem[l][p_address];
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
  if(bitmap_read_enable&&bitmap_write_enable)$fatal(1,"bitmap simultaneous read/write");
  if((bitmap_read_enable||bitmap_write_enable)&&(bitmap_address<0||bitmap_address>=135))$fatal(1,"bitmap address");
  if(state==WORD_READ&&remaining_words==0)$fatal(1,"empty word request");
  if(state==MAC)begin
   if(pending==0||weight_address<0||weight_address>=10368)$fatal(1,"invalid MAC request");
   if(weight_live[weight_address]&&weight_allow)for(integer l=0;l<8;l=l+1)
    if(($signed({phase_hold[l][31],phase_hold[l]})+$signed({weight_data[l][31],weight_data[l]}))!=
        $signed({add_result[l][31],add_result[l]}))$fatal(1,"signed32 prefix overflow");
  end
  if(state==DRAIN_READ&&psum_writes!=480)$fatal(1,"drain before all output positions");
 end
 `endif
endmodule
