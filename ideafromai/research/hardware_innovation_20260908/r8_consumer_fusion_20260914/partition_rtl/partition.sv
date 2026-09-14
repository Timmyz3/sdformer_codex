module partition(
 input logic clk,reset_n,cfg_valid,
 input logic [2:0] cfg_kind,input logic [10:0] cfg_addr,input logic [255:0] cfg_data,
 input logic start,input logic [3:0] mode,input logic source_allow,weight_allow,
 output logic result_valid,input logic result_ready,
 output logic [8:0] result_addr,output logic [255:0] result_data,output logic done,
 output logic [31:0] cycles,source_words,weight_words,second_weight_words,
 output logic [31:0] z_vector_reads,z_scalar_reads,z_writes,first_issues,
 output logic [31:0] psum_reads,psum_writes,mac_issues,
 output logic [31:0] source_stalls,weight_stalls,output_stalls,
 output logic [31:0] descriptor_writes,descriptor_reads,abs_issues,build_issues,
 output logic [31:0] cache_hits,cache_misses,cache_evictions,cache_epochs,singletons,empty_groups,
 output logic [5:0] debug_state
);
 typedef enum logic [5:0] {IDLE,ZCLEAR,SOURCE,CHECK,QREAD,TIMESEL,ZREAD,ZADD,KNEXT,
 ZSCAN,ABSVAL,DESCGEN,VLOAD,EPOCH,POSLOAD,BASE_MAC,LOOKUP,BUILD,GROUP_MAC,STORE,
 DRAIN_READ,DRAIN_SEND,FINISH} state_t;
 state_t state;logic [3:0] mode_q;
 logic [9:0] source_mem[0:1535];
 logic signed [2:0] q_mem[0:7][0:863],q_hold[0:7];
 logic signed [15:0] v_mem[0:7][0:95],qblock[0:7][0:7];
 logic k_live[0:863],v_live[0:95];
 logic signed [12:0] z_mem[0:7][0:39],z_hold[0:7],z_reg[0:7];
 logic [12:0] amag[0:7];
 logic signed [31:0] p_mem[0:7][0:479],acc[0:7];
 logic [28:0] descriptors[0:319];
 logic [8:0] starts[0:39];logic [3:0] counts[0:39];logic [7:0] position_live[0:39];
 logic [7:0] rank_live,block_live,remaining,build_remaining;
 logic [7:0] cache_valid;logic [15:0] cache_tag[0:7];
 logic signed [18:0] cache_coef[0:7][0:7],coef_hold[0:7];
 logic [2:0] victim;
 logic [9:0] src_masks[0:3];logic [39:0] pending;
 logic signed [15:0] oy,ox;
 integer k,p,fp,og,qfill,row,dtotal,dptr,groups_left;
 integer source_addr,selected_time;logic [2:0] selected_rank,selected_build,leader;
 logic in_bounds,cfg_v_live;
 logic [7:0] scan_mask,desc_group,desc_negative,effective_group,effective_negative;
 logic [28:0] desc_word,desc_hold;logic [15:0] effective_tag;
 logic cache_found;logic [2:0] cache_slot;
 logic [7:0] encode_group,encode_negative;
 logic signed [12:0] scalar;
 logic signed [18:0] multiply_coefficient[0:7];
 logic signed [31:0] product[0:7],lhs[0:7],rhs[0:7],add_y[0:7];
 logic [7:0] subtract;logic [31:0] xor_rhs[0:7];

 task clear_counters;
 begin
 cycles<=0;source_words<=0;weight_words<=0;second_weight_words<=0;
 z_vector_reads<=0;z_scalar_reads<=0;z_writes<=0;first_issues<=0;
 psum_reads<=0;psum_writes<=0;mac_issues<=0;
 source_stalls<=0;weight_stalls<=0;output_stalls<=0;
 descriptor_writes<=0;descriptor_reads<=0;abs_issues<=0;build_issues<=0;
 cache_hits<=0;cache_misses<=0;cache_evictions<=0;cache_epochs<=0;singletons<=0;empty_groups<=0;
 end
 endtask
 task next_group;
 begin
 dptr<=dptr+1;groups_left<=groups_left-1;
 state<=(groups_left==1)?STORE:LOOKUP;
 end
 endtask
 task next_scan;
 begin
 if(fp==39)begin og<=0;qfill<=0;state<=VLOAD;end
 else begin fp<=fp+1;state<=ZSCAN;end
 end
 endtask

 always_comb begin
 source_addr=(k/9)*16+(p/2+(k%9)/3)*4+p%2+k%3;
 in_bounds=(int'(oy)+(source_addr%16)/4>=0 && int'(oy)+(source_addr%16)/4<240 &&
 int'(ox)+source_addr%4>=0 && int'(ox)+source_addr%4<320);
 cfg_v_live=0;for(integer i=0;i<8;i=i+1)cfg_v_live=cfg_v_live||(cfg_data[i*32+:16]!=0);
 selected_time=0;for(integer i=39;i>=0;i=i-1)if(pending[i])selected_time=i;
 selected_rank=0;selected_build=0;
 for(integer i=7;i>=0;i=i-1)begin
 if(remaining[i])selected_rank=3'(i);if(build_remaining[i])selected_build=3'(i);
 end
 scan_mask=0;
 if(state==ZSCAN)for(integer i=0;i<8;i=i+1)scan_mask[i]=(z_mem[i][fp]!=0);
 encode_group=0;encode_negative=0;
 for(integer i=0;i<8;i=i+1)begin
 encode_group[i]=remaining[i]&&(amag[i]==amag[selected_rank]);
 encode_negative[i]=encode_group[i]&&(z_reg[i][12]^z_reg[selected_rank][12]);
 end
 // One descriptor read at LOOKUP; miss construction and MAC use the hold.
 desc_word=desc_hold;if(state==LOOKUP&&dptr>=0&&dptr<320)desc_word=descriptors[dptr];
 desc_group=desc_word[20:13];desc_negative=desc_word[28:21];
 effective_group=desc_group&block_live;effective_negative=desc_negative&effective_group;
 effective_tag={effective_negative,effective_group};
 leader=0;for(integer i=7;i>=0;i=i-1)if(effective_group[i])leader=3'(i);
 cache_found=0;cache_slot=0;
 for(integer i=0;i<8;i=i+1)if(cache_valid[i]&&cache_tag[i]==effective_tag&&!cache_found)begin cache_found=1;cache_slot=3'(i);end
 scalar=$signed(desc_word[12:0]);
 if(state==BASE_MAC)scalar=z_mem[selected_rank][fp];
 subtract=0;
 for(integer l=0;l<8;l=l+1)begin
 multiply_coefficient[l]={{3{qblock[leader][l][15]}},qblock[leader][l]};
 if(state==BASE_MAC)multiply_coefficient[l]={{3{qblock[selected_rank][l][15]}},qblock[selected_rank][l]};
 else if(state==GROUP_MAC)multiply_coefficient[l]=coef_hold[l];
 else if(state==LOOKUP&&cache_found)multiply_coefficient[l]=cache_coef[cache_slot][l];
 product[l]=$signed(multiply_coefficient[l])*$signed(scalar);
 lhs[l]=acc[l];rhs[l]=product[l];
 if(state==ZADD)begin
 lhs[l]={{19{z_hold[l][12]}},z_hold[l]};rhs[l]={{29{q_hold[l][2]}},q_hold[l]};
 end else if(state==ABSVAL)begin
 lhs[l]=0;rhs[l]={{19{z_reg[l][12]}},z_reg[l]};subtract[l]=z_reg[l][12];
 end else if(state==BUILD)begin
 lhs[l]={{13{coef_hold[l][18]}},coef_hold[l]};
 rhs[l]={{16{qblock[selected_build][l][15]}},qblock[selected_build][l]};
 subtract[l]=effective_negative[selected_build];
 end else if(state==LOOKUP)begin
 if((effective_group&(effective_group-8'd1))==0)subtract[l]=effective_negative[leader];
 else if(!cache_found)begin
 lhs[l]=0;rhs[l]={{16{qblock[leader][l][15]}},qblock[leader][l]};subtract[l]=effective_negative[leader];
 end
 end
 end
 result_valid=(state==DRAIN_SEND);result_addr=9'(row);debug_state=state;
 end
 genvar l,b;
 generate for(l=0;l<8;l=l+1)begin:ALU
 assign xor_rhs[l]=rhs[l]^{32{subtract[l]}};
 for(b=0;b<32;b=b+1)begin:BIT
 wire cin;if(b==0)assign cin=subtract[l];else assign cin=BIT[b-1].CARRY.cout;
 assign add_y[l][b]=lhs[l][b]^xor_rhs[l][b]^cin;
 if(b<31)begin:CARRY
 wire cout;assign cout=(lhs[l][b]&xor_rhs[l][b])|((lhs[l][b]^xor_rhs[l][b])&cin);
 end
 end
 end endgenerate

 always_ff @(posedge clk)begin
 if(!reset_n)begin
 state<=IDLE;mode_q<=12;done<=0;oy<=0;ox<=0;k<=0;p<=0;fp<=0;og<=0;qfill<=0;row<=0;
 dtotal<=0;dptr<=0;groups_left<=0;rank_live<=0;block_live<=0;remaining<=0;build_remaining<=0;
 cache_valid<=0;victim<=0;pending<=0;result_data<=0;desc_hold<=0;clear_counters();
 for(integer i=0;i<4;i=i+1)src_masks[i]<=0;
 for(integer i=0;i<8;i=i+1)begin q_hold[i]<=0;z_hold[i]<=0;z_reg[i]<=0;amag[i]<=0;acc[i]<=0;coef_hold[i]<=0;end
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
 default:begin end
 endcase
 if(state!=IDLE)cycles<=cycles+1;
 case(state)
 IDLE:if(start)begin
 mode_q<=mode;fp<=0;rank_live<=0;block_live<=0;cache_valid<=0;dtotal<=0;dptr<=0;
 state<=ZCLEAR;clear_counters();
 end
 ZCLEAR:begin
 for(integer i=0;i<8;i=i+1)z_mem[i][fp]<=0;
 z_writes<=z_writes+1;
 if(fp==39)begin k<=0;p<=0;state<=SOURCE;end else fp<=fp+1;
 end
 SOURCE:if(!k_live[k])state<=KNEXT;
 else if(!in_bounds||source_allow)begin
 src_masks[p]<=in_bounds?source_mem[source_addr]:10'd0;
 if(in_bounds)source_words<=source_words+1;
 if(p==3)state<=CHECK;else p<=p+1;
 end else source_stalls<=source_stalls+1;
 CHECK:begin
 pending<={src_masks[3],src_masks[2],src_masks[1],src_masks[0]};
 state<=((src_masks[0]|src_masks[1]|src_masks[2]|src_masks[3])==0)?KNEXT:QREAD;
 end
 QREAD:if(weight_allow)begin
 for(integer i=0;i<8;i=i+1)q_hold[i]<=q_mem[i][k];
 weight_words<=weight_words+1;state<=TIMESEL;
 end else weight_stalls<=weight_stalls+1;
 TIMESEL:if(pending!=0)begin fp<=selected_time;state<=ZREAD;end else state<=KNEXT;
 ZREAD:begin
 for(integer i=0;i<8;i=i+1)z_hold[i]<=z_mem[i][fp];
 z_vector_reads<=z_vector_reads+1;state<=ZADD;
 end
 ZADD:begin
 for(integer i=0;i<8;i=i+1)z_mem[i][fp]<=add_y[i][12:0];
 first_issues<=first_issues+1;z_writes<=z_writes+1;
 pending<=pending&(pending-40'd1);state<=TIMESEL;
 end
 KNEXT:if(k==863)begin fp<=0;state<=ZSCAN;end
 else begin k<=k+1;p<=0;state<=SOURCE;end
 ZSCAN:begin
 for(integer i=0;i<8;i=i+1)z_reg[i]<=z_mem[i][fp];
 z_vector_reads<=z_vector_reads+1;position_live[fp]<=scan_mask;rank_live<=rank_live|scan_mask;
 starts[fp]<=9'(dtotal);counts[fp]<=0;remaining<=scan_mask;
 if(mode_q==13&&scan_mask!=0)state<=ABSVAL;else next_scan();
 end
 ABSVAL:begin
 for(integer i=0;i<8;i=i+1)amag[i]<=add_y[i][12:0];
 abs_issues<=abs_issues+1;state<=DESCGEN;
 end
 DESCGEN:begin
 descriptors[dtotal]<={encode_negative,encode_group,z_reg[selected_rank]};
 descriptor_writes<=descriptor_writes+1;dtotal<=dtotal+1;counts[fp]<=counts[fp]+4'd1;
 remaining<=remaining&~encode_group;
 if((remaining&~encode_group)==0)next_scan();
 end
 VLOAD:begin
 if(!rank_live[qfill]||!v_live[og*8+qfill]||weight_allow)begin
 block_live[qfill]<=rank_live[qfill]&&v_live[og*8+qfill];
 for(integer i=0;i<8;i=i+1)qblock[qfill][i]<=(rank_live[qfill]&&v_live[og*8+qfill])?v_mem[i][og*8+qfill]:16'sd0;
 if(rank_live[qfill]&&v_live[og*8+qfill])begin weight_words<=weight_words+1;second_weight_words<=second_weight_words+1;end
 if(qfill==7)begin fp<=0;state<=(mode_q==13)?EPOCH:POSLOAD;end else qfill<=qfill+1;
 end else weight_stalls<=weight_stalls+1;
 end
 EPOCH:begin cache_valid<=0;victim<=0;cache_epochs<=cache_epochs+1;state<=POSLOAD;end
 POSLOAD:begin
 for(integer i=0;i<8;i=i+1)acc[i]<=0;
 remaining<=position_live[fp]&block_live;dptr<=int'(starts[fp]);groups_left<=int'(counts[fp]);
 if(mode_q==13)state<=(counts[fp]==0)?STORE:LOOKUP;
 else state<=((position_live[fp]&block_live)==0)?STORE:BASE_MAC;
 end
 BASE_MAC:begin
 for(integer i=0;i<8;i=i+1)acc[i]<=add_y[i];
 mac_issues<=mac_issues+1;z_scalar_reads<=z_scalar_reads+1;
 remaining<=remaining&(remaining-8'd1);
 if((remaining&(remaining-8'd1))==0)state<=STORE;
 end
 LOOKUP:begin
 descriptor_reads<=descriptor_reads+1;
 if(effective_group==0)begin empty_groups<=empty_groups+1;next_group();end
 else if((effective_group&(effective_group-8'd1))==0||cache_found)begin
 for(integer i=0;i<8;i=i+1)acc[i]<=add_y[i];
 mac_issues<=mac_issues+1;
 if((effective_group&(effective_group-8'd1))==0)singletons<=singletons+1;else cache_hits<=cache_hits+1;
 next_group();
 end else begin
 desc_hold<=desc_word;
 for(integer i=0;i<8;i=i+1)coef_hold[i]<=add_y[i][18:0];
 cache_misses<=cache_misses+1;build_issues<=build_issues+1;
 build_remaining<=effective_group&~(8'd1<<leader);state<=BUILD;
 end
 end
 BUILD:begin
 for(integer i=0;i<8;i=i+1)coef_hold[i]<=add_y[i][18:0];
 build_issues<=build_issues+1;build_remaining<=build_remaining&(build_remaining-8'd1);
 if((build_remaining&(build_remaining-8'd1))==0)begin
 for(integer i=0;i<8;i=i+1)cache_coef[victim][i]<=add_y[i][18:0];
 cache_tag[victim]<=effective_tag;cache_valid[victim]<=1;
 if(cache_valid[victim])cache_evictions<=cache_evictions+1;
 victim<=victim+3'd1;state<=GROUP_MAC;
 end
 end
 GROUP_MAC:begin
 for(integer i=0;i<8;i=i+1)acc[i]<=add_y[i];
 mac_issues<=mac_issues+1;next_group();
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
