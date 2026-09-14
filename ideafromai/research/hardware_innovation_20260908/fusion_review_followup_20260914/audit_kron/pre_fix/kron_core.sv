module kron_core(
 input logic clk,reset_n,cfg_valid,
 input logic [3:0] cfg_kind,input logic [10:0] cfg_addr,input logic [255:0] cfg_data,
 input logic start,input logic [3:0] mode,input logic source_allow,weight_allow,
 output logic result_valid,input logic result_ready,
 output logic [8:0] result_addr,output logic [255:0] result_data,output logic done,
 output logic [31:0] cycles,source_words,weight_words,second_weight_words,local_source_reads,
 output logic [31:0] z_vector_reads,z_scalar_reads,z_writes,first_issues,dual_updates,
 output logic [31:0] psum_reads,psum_writes,mac_issues,
 output logic [31:0] source_stalls,weight_stalls,output_stalls,
 output logic [31:0] s_build_cycles,s_mac_issues,s_writes,s_reads,a_words,factor_mac_issues,escape_mac_issues,
 output logic [5:0] debug_state
);
 typedef enum logic [5:0] {IDLE,ZCLEAR,L_START,L_LOAD,L_GATHER,CHECK,QREAD,TIMESEL,
 ZREAD,ZADD,KNEXT,ZSCAN,VLOAD,POSLOAD,BASE_MAC,STORE,DRAIN_READ,DRAIN_SEND,FINISH,SINIT,SMAC,SSTORE,ALOAD,KPOS,KMAC} state_t;
 state_t state;logic [3:0] mode_q;
 logic [9:0] source_mem[0:1535],local_source[0:15],src_masks[0:3];
 logic signed [2:0] q_mem[0:7][0:863],q_hold[0:7];
 logic signed [18:0] v_mem[0:7][0:95],qblock[0:7][0:7];
 logic k_live[0:863],v_live[0:95];
 logic [25:0] z_mem[0:7][0:19],z_hold[0:7];
 logic signed [31:0] p_mem[0:7][0:479],acc[0:7];
 logic [7:0] position_live[0:39],rank_live,block_live,remaining,scan_mask;
 logic [39:0] pending;logic signed [15:0] oy,ox;
 integer k,fp,zrow,og,qfill,row,load_xy,selected_time;
 logic [10:0] source_addr;logic [4:0] read_zrow;
 logic in_bounds,cfg_v_live,active_lo,active_hi,read_half;
 logic [2:0] selected_rank;
 logic signed [12:0] scalar,multiply_scalar;
 logic signed [18:0] multiply_coefficient[0:7];
 logic signed [31:0] product[0:7],lhs[0:7],rhs[0:7],add_y[0:7];

 logic signed [12:0] a_mem[0:11][0:3],a_hold[0:3];
 logic signed [5:0] b_mem[0:7][0:1];
 logic [11:0] escape_groups;
 logic signed [18:0] s_mem[0:7][0:159];
 logic [3:0] s_live[0:39],s_remaining;
 logic [1:0] pair_remaining;
 integer pair_index;
 logic pair_component;
 logic [1:0] selected_pair;
 logic s_nonzero;
 task advance_output;
 begin
 if(fp<39)begin fp<=fp+1;state<=(mode_q==1 && !escape_groups[og])?KPOS:POSLOAD;end
 else if(og<11)begin og<=og+1;qfill<=0;state<=(mode_q==1 && !escape_groups[og+1])?ALOAD:VLOAD;end
 else begin row<=0;state<=DRAIN_READ;end
 end endtask
 task clear_counters;
 begin
 cycles<=0;source_words<=0;weight_words<=0;second_weight_words<=0;local_source_reads<=0;
 z_vector_reads<=0;z_scalar_reads<=0;z_writes<=0;first_issues<=0;dual_updates<=0;
 psum_reads<=0;psum_writes<=0;mac_issues<=0;
 source_stalls<=0;weight_stalls<=0;output_stalls<=0;
 s_build_cycles<=0;s_mac_issues<=0;s_writes<=0;s_reads<=0;a_words<=0;factor_mac_issues<=0;escape_mac_issues<=0;
 end endtask
 always_comb begin
 source_addr=11'((k/9)*16+load_xy);
 in_bounds=(int'(oy)+load_xy/4>=0 && int'(oy)+load_xy/4<240 && int'(ox)+load_xy%4>=0 && int'(ox)+load_xy%4<320);
 cfg_v_live=0;for(integer i=0;i<8;i=i+1)cfg_v_live=cfg_v_live||(cfg_data[i*32+:19]!=0);
 selected_time=0;for(integer i=39;i>=0;i=i-1)if(pending[i])selected_time=i;
 selected_rank=0;for(integer i=7;i>=0;i=i-1)if(remaining[i])selected_rank=3'(i);
 pair_component=!pair_remaining[0];
 if(state==SMAC)selected_rank=3'(pair_index*2+int'(pair_component));
 selected_pair=0;for(integer i=3;i>=0;i=i-1)if(s_remaining[i])selected_pair=2'(i);
 read_zrow=5'((fp/20)*10+fp%10);read_half=((fp/10)%2)!=0;
 scan_mask=0;
 if(state==ZSCAN)for(integer i=0;i<8;i=i+1)
 scan_mask[i]=read_half?(z_mem[i][read_zrow][25:13]!=0):(z_mem[i][read_zrow][12:0]!=0);
 scalar=read_half?$signed(z_mem[selected_rank][read_zrow][25:13]):$signed(z_mem[selected_rank][read_zrow][12:0]);
 multiply_scalar=(state==KMAC)?a_hold[selected_pair]:scalar;
 s_nonzero=0;
 for(integer l=0;l<8;l=l+1)begin
 multiply_coefficient[l]=qblock[selected_rank][l];
 if(state==SMAC)multiply_coefficient[l]=19'(b_mem[l][pair_component]);
 if(state==KMAC)multiply_coefficient[l]=s_mem[l][fp*4+int'(selected_pair)];
 product[l]=$signed(multiply_coefficient[l])*$signed(multiply_scalar);
 lhs[l]=acc[l];rhs[l]=product[l];s_nonzero=s_nonzero||(acc[l]!=0);
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
 if(b==0)assign cin=1'b0;
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
 state<=IDLE;mode_q<=0;done<=0;oy<=0;ox<=0;k<=0;fp<=0;zrow<=0;og<=0;qfill<=0;row<=0;load_xy<=0;pair_index<=0;pair_remaining<=0;s_remaining<=0;escape_groups<=0;
 rank_live<=0;block_live<=0;remaining<=0;pending<=0;result_data<=0;active_lo<=0;active_hi<=0;clear_counters();
 for(integer i=0;i<4;i=i+1)begin src_masks[i]<=0;a_hold[i]<=0;end
 for(integer i=0;i<8;i=i+1)begin q_hold[i]<=0;z_hold[i]<=0;acc[i]<=0;end
 end else begin
 done<=0;
 if(cfg_valid&&state==IDLE)case(cfg_kind)
 0:source_mem[cfg_addr]<=cfg_data[9:0];
 3:begin oy<=cfg_data[15:0];ox<=cfg_data[31:16];end
 4:for(integer i=0;i<8;i=i+1)q_mem[i][cfg_addr[9:0]]<=cfg_data[i*32+:3];
 5:begin
 for(integer i=0;i<8;i=i+1)v_mem[i][cfg_addr[6:0]]<=cfg_data[i*32+:19];
 v_live[cfg_addr[6:0]]<=cfg_v_live;
 end
 6:k_live[cfg_addr[9:0]]<=cfg_data[0];
 8:for(integer i=0;i<4;i=i+1)a_mem[cfg_addr[3:0]][i]<=cfg_data[i*32+:13];
 9:for(integer i=0;i<8;i=i+1)b_mem[i][cfg_addr[0]]<=cfg_data[i*32+:6];
 10:escape_groups<=cfg_data[11:0];
 default:begin end
 endcase
 if(state!=IDLE)cycles<=cycles+1;
 if(state==SINIT||state==SMAC||state==SSTORE)s_build_cycles<=s_build_cycles+1;
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
 KNEXT:if(k==863)begin fp<=0;state<=ZSCAN;end
 else begin k<=k+1;state<=(k%9==8)?L_START:L_GATHER;end
 ZSCAN:begin
 z_vector_reads<=z_vector_reads+1;position_live[fp]<=scan_mask;rank_live<=rank_live|scan_mask;
 if(fp==39)begin og<=0;qfill<=0;fp<=0;pair_index<=0;state<=mode_q==1?SINIT:VLOAD;end else fp<=fp+1;
 end
 SINIT:begin
 for(integer i=0;i<8;i=i+1)acc[i]<=0;
 pair_remaining<=2'(position_live[fp]>>(pair_index*2));
 state<=(((position_live[fp]>>(pair_index*2))&8'd3)!=0)?SMAC:SSTORE;
 end
 SMAC:begin
 for(integer i=0;i<8;i=i+1)acc[i]<=add_y[i];
 s_mac_issues<=s_mac_issues+1;mac_issues<=mac_issues+1;z_scalar_reads<=z_scalar_reads+1;
 pair_remaining<=pair_remaining&(pair_remaining-2'd1);
 if((pair_remaining&(pair_remaining-2'd1))==0)state<=SSTORE;
 end
 SSTORE:begin
 for(integer i=0;i<8;i=i+1)s_mem[i][fp*4+pair_index]<=acc[i][18:0];
 s_writes<=s_writes+1;s_live[fp][pair_index]<=s_nonzero;
 if(pair_index<3)begin pair_index<=pair_index+1;state<=SINIT;end
 else if(fp<39)begin fp<=fp+1;pair_index<=0;state<=SINIT;end
 else begin fp<=0;state<=escape_groups[0]?VLOAD:ALOAD;end
 end
 VLOAD:begin
 if(!rank_live[qfill]||!v_live[og*8+qfill]||weight_allow)begin
 block_live[qfill]<=rank_live[qfill]&&v_live[og*8+qfill];
 for(integer i=0;i<8;i=i+1)qblock[qfill][i]<=(rank_live[qfill]&&v_live[og*8+qfill])?v_mem[i][og*8+qfill]:19'sd0;
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
 if(mode_q==1)escape_mac_issues<=escape_mac_issues+1;
 remaining<=remaining&(remaining-8'd1);
 if((remaining&(remaining-8'd1))==0)state<=STORE;
 end
 ALOAD:if(weight_allow)begin
 for(integer i=0;i<4;i=i+1)a_hold[i]<=a_mem[og][i];
 a_words<=a_words+1;weight_words<=weight_words+1;second_weight_words<=second_weight_words+1;
 fp<=0;state<=KPOS;
 end else weight_stalls<=weight_stalls+1;
 KPOS:begin
 for(integer i=0;i<8;i=i+1)acc[i]<=0;
 for(integer i=0;i<4;i=i+1)s_remaining[i]<=s_live[fp][i]&&(a_hold[i]!=0);
 if((s_live[fp]&{(a_hold[3]!=0),(a_hold[2]!=0),(a_hold[1]!=0),(a_hold[0]!=0)})==0)state<=STORE;else state<=KMAC;
 end
 KMAC:begin
 for(integer i=0;i<8;i=i+1)acc[i]<=add_y[i];
 mac_issues<=mac_issues+1;factor_mac_issues<=factor_mac_issues+1;s_reads<=s_reads+1;
 s_remaining<=s_remaining&(s_remaining-4'd1);
 if((s_remaining&(s_remaining-4'd1))==0)state<=STORE;
 end
 STORE:begin
 for(integer i=0;i<8;i=i+1)p_mem[i][og*40+fp]<=acc[i];
 psum_writes<=psum_writes+1;advance_output();
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
