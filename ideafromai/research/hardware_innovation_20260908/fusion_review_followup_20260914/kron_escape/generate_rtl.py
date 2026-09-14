from pathlib import Path
import re,shutil
H=Path(__file__).resolve().parent;B=H.parents[1];O=B/'fusion_ten_trials_20260914/temporal_direction'
oldcounters='encoder_cycles candidate_trials range_rejects base_reads base_negations reference_writes choice_prev choice_anchor choice_neg1 choice_pos2 choice_neg2 anchor_count'.split()
newcounters='s_build_cycles s_mac_issues s_writes s_reads a_words factor_mac_issues escape_mac_issues'.split()
s=(O/'consumer_stream.sv').read_text()
for n in oldcounters:
 # declarations and leaf wiring are each complete independent lines; resets share lines.
 s=re.sub(r'^ (?:output logic \[63:0\] core_'+n+r',|logic \[31:0\] l_'+n+r';)\n','',s,flags=re.M)
 s=re.sub(r'^  \.'+n+r'\(l_'+n+r'\),\n','',s,flags=re.M)
 s=re.sub(r'core_'+n+r'<=0;','',s)
 s=re.sub(r'^      core_'+n+r'<=.*\n','',s,flags=re.M)
s=s.replace(' output logic [63:0] core_cycles,',''.join(' output logic [63:0] core_'+n+',\n' for n in newcounters)+' output logic [63:0] core_cycles,')
s=s.replace(' logic [31:0] l_cycles;',''.join(' logic [31:0] l_'+n+';\n' for n in newcounters)+' logic [31:0] l_cycles;')
s=s.replace('  .cycles(l_cycles),',''.join('  .'+n+'(l_'+n+'),\n' for n in newcounters)+'  .cycles(l_cycles),')
s=s.replace('core_cycles<=0;',''.join('core_'+n+'<=0;' for n in newcounters)+'core_cycles<=0;')
s=s.replace('      core_cycles<=core_cycles+', ''.join('      core_'+n+'<=core_'+n+"+64'(l_"+n+');\n' for n in newcounters)+'      core_cycles<=core_cycles+')
s=s.replace('logic [2:0] core_cfg_kind','logic [3:0] core_cfg_kind').replace('param_kind<7','param_kind!=7').replace('core_cfg_kind=param_kind[2:0]','core_cfg_kind=param_kind')
s=s.replace('temporal_core leaf','kron_core leaf').replace('(mode>2)','(mode>1)')
s=s.replace('(param_kind==7 && param_row==23)', '(param_kind==7 && param_row==23) || (param_kind==8 && param_row==11) || (param_kind==9 && param_row==1) || param_kind==10')
s=s.replace('4:param_kind<=5;5:param_kind<=6;6:param_kind<=7;', '4:param_kind<=5;5:param_kind<=6;6:param_kind<=7;\n       7:if(mode_q==1)param_kind<=8;else begin resident<=1;state<=SOURCE;end\n       8:param_kind<=9;9:param_kind<=10;')
s=s.replace('logic resident, core_seen','logic resident, factor_resident, core_seen').replace('resident<=0;mode_q','resident<=0;factor_resident<=0;mode_q').replace('else state<=resident?SOURCE:PARAMETERS;',"else if(resident && mode==1 && !factor_resident)begin param_kind<=8;state<=PARAMETERS;end\n     else state<=resident?SOURCE:PARAMETERS;").replace('8:param_kind<=9;9:param_kind<=10;','8:param_kind<=9;9:param_kind<=10;\n       10:begin factor_resident<=1;resident<=1;state<=SOURCE;end')
(H/'consumer_stream.sv').write_text(s)
shutil.copy2(O/'i24_consumer.sv',H/'i24_consumer.sv')
s=(O/'tb.cpp').read_text().replace('param[8];for(int k:{4,5,6,7})','param[11];for(int k:{4,5,6,7,8,9,10})').replace('command?0:1848','command?0:(mode?1863:1848)')
for n in oldcounters:s=s.replace('SHOW(core_'+n+');','')
s=s.replace('SHOW(core_cycles);',''.join('SHOW(core_'+n+');' for n in newcounters)+'SHOW(core_cycles);')
s=s.replace('if(d.raw_monitor_data[l]!=raw.at(raws*8+l))return 20;', 'if(d.raw_monitor_data[l]!=raw.at(raws*8+l)){std::cerr<<"raw mismatch mode="<<mode<<" row="<<raws<<" lane="<<l<<" got="<<int32_t(d.raw_monitor_data[l])<<" expected="<<int32_t(raw.at(raws*8+l))<<"\\n";return 20;}')
(H/'tb.cpp').write_text(s)
# Keep the reviewed complete source/Q1 machinery and original cached-output Q2 control.
s=(O/'temporal_core.sv').read_text()
preamble=s[:s.index(' logic signed [12:0] current_z')]
preamble=preamble.replace('temporal_core','kron_core').replace('input logic [2:0] cfg_kind','input logic [3:0] cfg_kind')
preamble=re.sub(r' output logic \[31:0\] encoder_cycles.*?\n',' output logic [31:0] '+','.join(newcounters)+',\n',preamble)
preamble=preamble.replace('FINISH,EREAD,EDIFF,EVAL,EWRITE,BASE_READ,BASE_NEG,ANCHOR_SAVE','FINISH,SINIT,SMAC,SSTORE,ALOAD,KPOS,KMAC')
preamble=preamble.replace('signed [15:0] v_mem','signed [18:0] v_mem')
preamble=preamble.replace('logic signed [12:0] scalar;', 'logic signed [12:0] scalar,multiply_scalar;')
s=preamble+''' logic signed [12:0] a_mem[0:11][0:3],a_hold[0:3];
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
'''
orig=(O/'temporal_core.sv').read_text()
s+=orig[orig.index(' ZCLEAR:begin'):orig.index(' EREAD:begin')].replace('state<=(mode_q==0)?ZSCAN:EREAD','state<=ZSCAN')
s+=''' ZSCAN:begin
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
'''
s+=orig[orig.index(' DRAIN_READ:begin'):]
(H/'kron_core.sv').write_text(s)
