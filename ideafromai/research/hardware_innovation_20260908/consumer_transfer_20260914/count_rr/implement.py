"""Isolated graft: retain RR/consumer control, add paid count/psum execution."""
from pathlib import Path
import shutil

H=Path(__file__).resolve().parent
B=H.parents[1]
R=B/'transfer_adapt_20260914/rr_borrow/adapt_rr'
P=B/'transfer_adapt_20260914/pair_psum_overlay/temporal_pairing'
AUX='aux_reads aux_writes aux_issues aux_weight_words aux_events metadata_reads count_checks count_bank_reads count_bank_writes'.split()

def replace(s,a,b):
    assert a in s,a
    return s.replace(a,b)

c=(R/'rr_context.sv').read_text()
p=(P/'decomp_core.sv').read_text()
c=c.replace('input logic [3:0] mode','input logic [4:0] mode')
c=c.replace('output logic weight_is_q2,','output logic [1:0] weight_kind,')
c=c.replace('output logic signed [12:0] alu_scalar,','output logic [103:0] alu_scalar,output logic [7:0] alu_correction,')
c=c.replace('output logic [31:0] arbitration_stalls,','input logic [39:0] time_permutation,input logic [5:0] ngroups,\n output logic [31:0] '+','.join(AUX)+',\n output logic [31:0] arbitration_stalls,')
c=c.replace('DRAIN_SEND,FINISH} state_t;','DRAIN_SEND,FINISH,MREAD,C_CHECK,C_ADD,G_START,G_QREAD,G_READ,G_SELECT,G_ZREAD,G_MAC} state_t;')
c=c.replace('state_t state;logic [1:0] mode_q;','state_t state;logic [4:0] mode_q;wire count_mode=(mode_q==20||mode_q==21);')
decl=p[p.index(' logic [31:0] count_hold'):p.index(' logic [9:0] source_mem')]
decl=decl.replace('logic signed [12:0] multiply_scalar[0:7];','')
c=c.replace(' logic repair_needed;',' logic [5:0] class_hold[0:3];logic [3:0] inverse_time;logic [51:0] retire_write[0:7];\n'+decl+' logic repair_needed;')
c=c.replace('cycles<=0;arbitration_stalls', ''.join(x+'<=0;' for x in AUX)+'\n cycles<=0;arbitration_stalls')
c=c.replace('if(mode_q==0)', 'if(mode_q==0||count_mode)')
c=c.replace('mode_q>=2', '(mode_q==2||mode_q==3)')
c=c.replace('z_read_address=(state==ZSCAN || state==BASE_MAC)?read_zrow:4\'(zrow);',"z_read_address=(state==ZSCAN || state==BASE_MAC || state==G_ZREAD || state==G_MAC)?read_zrow:4'(zrow);")
c=c.replace('  QREAD:resource_request', '  QREAD,MREAD,G_QREAD:resource_request')
c=c.replace('  STORE,DRAIN_READ:resource_request=6\'b001000;',"  STORE,DRAIN_READ,G_READ:resource_request=6'b001000;\n  C_CHECK:if(read_banks!=0)resource_request=6'b001000;\n  C_ADD:resource_request=6'b011000;\n  G_ZREAD:resource_request=6'b000100;\n  G_MAC:resource_request=6'b010100;")
c=replace(c,'weight_is_q2=(state==VLOAD);weight_address=weight_is_q2?10\'(og*8+qfill):10\'(k);',"weight_kind=(state==VLOAD)?2'd1:(state==MREAD)?2'd2:(state==G_QREAD)?2'd3:2'd0;\n weight_address=(state==VLOAD)?10'(og*8+qfill):(state==G_QREAD)?10'(group_iter):10'(k);")
c=c.replace('alu_mac=(state==BASE_MAC);','alu_mac=(state==BASE_MAC||state==G_MAC);')
c=c.replace("else if(state==ZADD)alu_format={1'b0,mode_q}+3'd1;", "else if(state==ZADD)alu_format=count_mode?3'd1:(mode_q[2:0]+3'd1);\n else if(state==C_ADD)alu_format=3;\n else if(state==G_MAC&&packed_retire)alu_format=5;")
c=c.replace('state==NORMALIZE_READ ||\n', 'state==NORMALIZE_READ || state==G_ZREAD ||\n')
c=c.replace(' alu_scalar=scalar;', ' alu_correction=correction_carry;')
c=c.replace('  alu_lhs[l*32+:32]=lhs[l];', '''  alu_scalar[l*13+:13]=scalar;
  if(state==C_ADD)begin lhs[l]=count_hold[l];rhs[l]=count_increment[l];end
  if(state==G_MAC)begin
   multiply_coefficient[l]={{16{q_hold[l][2]}},q_hold[l]};
   alu_scalar[l*13+:13]={3'd0,count_scalar[l/2]};
   lhs[l]={{19{z_hold[l][13*(fp/10)+12]}},z_hold[l][13*(fp/10)+:13]};
   if(packed_retire)begin
    multiply_coefficient[l]=$signed({1'b0,count_high[l/2][6:0],3'd0,count_low[l/2][7:0]});
    alu_scalar[l*13+:13]={{10{q_hold[l][2]}},q_hold[l]};
    lhs[l]={6'd0,z_hold[l][26*(fp/20)+:26]};
   end
  end
  alu_lhs[l*32+:32]=lhs[l];''')

comb=p[p.index(' // One physical p_mem'):p.index(' cfg_v_live=0;')]
comb=comb.replace('p_read_enable[i]=count_live', 'p_read_enable[i]=resource_grant&&count_live')
comb=comb.replace('p_write_enable[i]=1;', 'p_write_enable[i]=resource_grant;')
comb=comb.replace('p_read_enable[i]=1;', 'p_read_enable[i]=resource_grant;')
# read_banks used for requests must not depend on grants, avoiding combinational loop.
comb=comb.replace('if((state==C_CHECK||state==G_READ)&&p_read_enable[i])read_banks=read_banks+1;', '''if(state==C_CHECK&&class_hold[i/2]>0&&class_hold[i/2]<=32&&count_live[i/2][(int'(class_hold[i/2])-1)*5+cblock])read_banks=read_banks+1;
  if(state==G_READ&&count_live[i/2][group_iter*5+cblock])read_banks=read_banks+1;''')
extra=''' always_comb begin
 inverse_time=4'(row%10);
 if(mode_q==21)for(integer t=0;t<10;t=t+1)
  if(time_permutation[t*4+:4]==4'(row%10))inverse_time=4'(t);
 has_direct=0;has_count=0;active_groups=0;
 for(integer g=0;g<4;g=g+1)begin
  has_direct=has_direct||(weight_data[g*6+:6]==63);
  has_count=has_count||(weight_data[g*6+:6]>0&&weight_data[g*6+:6]<=32);
  if(class_hold[g]>0&&class_hold[g]<=32)active_groups=active_groups+1;
 end
'''+p[p.index(' source_blocks=0;'):p.index(' // One physical p_mem')]+comb+'''
 correction_carry=0;
 for(integer l=0;l<8;l=l+1)correction_carry[l]=(state==G_MAC&&packed_retire)&&q_hold[l][2]&&(count_low[l/2]!=0);
 end
'''
c=c.replace(' always_ff @(posedge clk)begin', extra+'\n always_ff @(posedge clk)begin',1)
c=c.replace(' always_ff @(posedge clk)begin', ''' always_comb begin
  // Full52bit write from the previously authorized row read; no bit-write memory port.
  for(integer lane=0;lane<8;lane=lane+1)begin
   retire_write[lane]=z_hold[lane];
   if(packed_retire)retire_write[lane][26*(fp/20)+:26]=add_y[lane][25:0];
   else retire_write[lane][13*(fp/10)+:13]=add_y[lane][12:0];
  end
 end
 always_ff @(posedge clk)begin''',1)
c=c.replace('  state<=IDLE;mode_q<=0;', '''  block_pending<=0;for(integer g=0;g<4;g=g+1)begin count_live[g]<=0;class_hold[g]<=0;end
  count_active<=0;cblock<=0;group_iter<=0;group_pending<=0;group_live<=0;
  for(integer i=0;i<8;i=i+1)count_hold[i]<=0;
  state<=IDLE;mode_q<=0;''')
c=c.replace('  done<=0;\n  if(cfg_valid', '  done<=0;\n  for(integer i=0;i<8;i=i+1)if(p_write_enable[i])p_mem[i][p_addr[i]]<=p_write_data[i];\n  if(cfg_valid')
c=c.replace("mode_q<=(mode==1&&!range_ok)?2'd0:mode[1:0];", "mode_q<=(mode==1&&!range_ok)?5'd0:mode;")
c=c.replace('    compute_done<=0;','    compute_done<=0;block_pending<=0;group_live<=0;for(integer g=0;g<4;g=g+1)count_live[g]<=0;')
c=c.replace('for(integer i=0;i<4;i=i+1)src_masks[i]<=local_source[(i/2+(k%9)/3)*4+i%2+k%3];', '''for(integer i=0;i<4;i=i+1)begin
     if(mode_q==21)for(integer t=0;t<10;t=t+1)src_masks[i][t]<=local_source[(i/2+(k%9)/3)*4+i%2+k%3][time_permutation[t*4+:4]];
     else src_masks[i]<=local_source[(i/2+(k%9)/3)*4+i%2+k%3];
    end''')
c=c.replace(')?KNEXT:QREAD;', ')?KNEXT:(count_mode?MREAD:QREAD);')
c=c.replace('q_hold[i]<=weight_data[i*32+:3];','q_hold[i]<=(!count_mode||class_hold[i/2]==63)?weight_data[i*32+:3]:3\'sd0;',1)
c=c.replace('   end else state<=KNEXT;\n   ZREAD:', '''   end else if(count_mode&&count_active)begin
    cblock<=0;for(integer i=4;i>=0;i=i-1)if(block_pending[i])cblock<=i;state<=C_CHECK;
   end else state<=KNEXT;
   ZREAD:''')
c=c.replace('else begin fp<=0;state<=ZSCAN;end\n   end else begin k<=k+1;', 'else begin fp<=0;group_iter<=0;state<=(count_mode&&ngroups!=0)?G_START:ZSCAN;end\n   end else begin k<=k+1;')
c=c.replace('for(integer i=0;i<8;i=i+1)p_mem[i][og*40+fp]<=acc[i];','')
c=c.replace('result_data[i*32+:32]<=p_mem[i][row];', 'result_data[i*32+:32]<=p_read_data[i];')
states=p[p.index(' MREAD:begin'):p.index(' QREAD:if(weight_allow)')]
states=states.replace('k_class[g][k]','weight_data[g*6+:6]')
states+=p[p.index(' C_CHECK:begin'):p.index(' ZSCAN:begin')]
states=states.replace('representative[i][group_iter]', 'weight_data[i*32+:3]')
states=states.replace('z_hold[i]<=z_mem[i][read_zrow];','z_hold[i]<=z_read_bus[i*52+:52];')
states=states.replace('if(packed_retire)z_mem[i][read_zrow]<=add_y[i][25:0];\n else if(ghalf)z_mem[i][read_zrow][25:13]<=add_y[i][12:0];else z_mem[i][read_zrow][12:0]<=add_y[i][12:0];', 'z_mem[i][read_zrow]<=retire_write[i];')
c=c.replace('   ZSCAN:begin',states+'\n   ZSCAN:begin')
checks=p[p.index('`ifdef VERILATOR'):p.index('endmodule')]
checks=checks.replace('if(state==C_ADD&&p_write_enable[i])', 'if(state==C_ADD&&p_write_enable[i])')
checks=checks.replace('if(state==STORE)begin','if(state==STORE&&resource_grant)begin')
checks=checks.replace('if(state==DRAIN_READ&&check_stores!=480)', 'if(state==DRAIN_READ&&resource_grant&&check_stores!=480)')
checks=checks.replace('  for(integer i=0;i<8;i=i+1)begin', '''  if(resource_request!=0&&!resource_grant)begin
   if(p_read_enable!=0||p_write_enable!=0||z_read_bus!=0)$fatal(1,"ungranted memory access");
  end
  for(integer i=0;i<8;i=i+1)begin''')
c=c.replace('endmodule',checks+'endmodule')
(H/'rr_context.sv').write_text(c)

w=(R/'interleave_stream.sv').read_text()
w=w.replace('input logic [3:0] mode,','input logic [4:0] mode,')
w=w.replace('logic [2:0] mode_q;', 'logic [4:0] mode_q;')
w=w.replace('logic resident,','logic class_resident,permutation_resident;logic [39:0] time_permutation;logic [5:0] ngroups;\n logic [23:0] class_mem[0:863];logic [23:0] representative[0:31];\n logic resident,')
w=w.replace('logic [1:0] weight_is_q2;', 'logic [1:0] weight_kind[0:1];')
w=w.replace('logic signed [12:0] child_scalar[0:1];','logic [103:0] child_scalar[0:1];logic [7:0] child_correction[0:1];')
w=w.replace('logic signed [12:0] scalar;', 'logic signed [12:0] scalar[0:7];')
w=w.replace('  scalar=alu_mac?child_scalar[alu_owner]:13\'sd0;shared_weight_data=0;', '  shared_weight_data=0;')
w=w.replace('if(weight_is_q2[weight_owner])shared_weight_data[l*32+:16]=v_mem[l][weight_addr[weight_owner][6:0]];\n    else shared_weight_data[l*32+:3]=q_mem[l][weight_addr[weight_owner]];', '''if(weight_kind[weight_owner]==1)shared_weight_data[l*32+:16]=v_mem[l][weight_addr[weight_owner][6:0]];
    else if(weight_kind[weight_owner]==0)shared_weight_data[l*32+:3]=q_mem[l][weight_addr[weight_owner]];
    else if(weight_kind[weight_owner]==3)shared_weight_data[l*32+:3]=representative[weight_addr[weight_owner][4:0]][l*3+:3];''')
w=w.replace('product[l]=$signed(coefficient[l])*$signed(scalar);', 'scalar[l]=alu_mac?$signed(child_scalar[alu_owner][l*13+:13]):13\'sd0;\n   product[l]=$signed(coefficient[l])*$signed(scalar[l]);')
w=w.replace('   if(proof_active)begin', "   if(alu_mac&&alu_format==5)rhs[l]={6'd0,product[l][23:11],{2{product[l][10]}},product[l][10:0]};\n   if(proof_active)begin",1)
w=w.replace('  end\n end\n genvar gl,gb;', '''  end
  if(((grant[0]&&request[0][1])||(grant[1]&&request[1][1]))&&weight_kind[weight_owner]==2)
   shared_weight_data[23:0]=class_mem[weight_addr[weight_owner]];
 end
 genvar gl,gb;''',1)
w=w.replace("else assign cin=(((gb==13", "else if(gb==13)assign cin=(alu_format==5)?child_correction[alu_owner][gl]:((alu_format==1)?1'b0:BIT[gb-1].CARRY.cout);\n   else assign cin=(((gb==13")
w=w.replace(".mode(mode_q==4?4'd3:{1'd0,mode_q})", ".mode(mode_q==4?5'd3:mode_q)")
w=w.replace('.weight_is_q2(weight_is_q2[c])','.weight_kind(weight_kind[c])')
w=w.replace('.alu_scalar(child_scalar[c]),', '.alu_scalar(child_scalar[c]),.alu_correction(child_correction[c]),')
w=w.replace('  rr_context core(', '  rr_context core(\n   .time_permutation(time_permutation),.ngroups(ngroups),'+''.join('\n   .'+x+'(l_'+x+'[c]),' for x in AUX))
w=w.replace('mode_q<=mode[2:0];','mode_q<=mode;')
w=w.replace('(mode!=1&&mode!=2&&mode!=3&&mode!=4)', '(mode!=0&&mode!=1&&mode!=2&&mode!=3&&mode!=4&&mode!=20&&mode!=21)')
w=w.replace('else state<=resident?SOURCE:PARAMETERS;', '''else if(!resident)state<=PARAMETERS;
     else if(mode>=20&&!class_resident)begin param_kind<=8;state<=PARAMETERS;end
     else if(mode==21&&!permutation_resident)begin param_kind<=11;state<=PARAMETERS;end
     else state<=SOURCE;''')
old='''if(((param_kind==4||param_kind==6)&&param_row==863)||(param_kind==5&&param_row==95)||(param_kind==7&&param_row==23))begin
      param_row<=0;
      case(param_kind)4:param_kind<=5;5:param_kind<=6;6:param_kind<=7;default:begin resident<=1;state<=SOURCE;end endcase
     end else param_row<=param_row+1;'''
new='''if(((param_kind==4||param_kind==6||param_kind==8)&&param_row==863)||(param_kind==5&&param_row==95)||(param_kind==7&&param_row==23)||(param_kind==9&&param_row==31)||param_kind==10||param_kind==11)begin
      param_row<=0;
      case(param_kind)
       4:param_kind<=5;5:param_kind<=6;6:param_kind<=7;
       7:begin resident<=1;if(mode_q>=20)param_kind<=8;else state<=SOURCE;end
       8:param_kind<=9;9:param_kind<=10;
       10:begin class_resident<=1;if(mode_q==21)param_kind<=11;else state<=SOURCE;end
       default:begin permutation_resident<=1;state<=SOURCE;end
      endcase
     end else param_row<=param_row+1;'''
w=replace(w,old,new)
w=w.replace('   if(state==PARAMETERS&&parameter_valid)begin', '''   if(state==PARAMETERS&&parameter_valid)begin
    if(param_kind==8)class_mem[param_row[9:0]]<=parameter_data[23:0];
    if(param_kind==9)for(integer l=0;l<8;l=l+1)representative[param_row[4:0]][l*3+:3]<=parameter_data[l*32+:3];
    if(param_kind==10)ngroups<=parameter_data[5:0];
    if(param_kind==11)time_permutation<=parameter_data[39:0];''')
w=w.replace('resident<=0;', "resident<=0;class_resident<=0;permutation_resident<=0;ngroups<=0;time_permutation<=40'h9876543210;",1)
# Public counters and per-context accumulation use existing leaf_done lifetime.
w=w.replace(' output logic [63:0] core_cycles,', ''.join(' output logic [63:0] core_'+x+',\n' for x in AUX)+' output logic [63:0] core_cycles,')
w=w.replace(' logic [31:0] l_cycles[0:1];', ''.join(' logic [31:0] l_'+x+'[0:1];\n' for x in AUX)+' logic [31:0] l_cycles[0:1];')
w=w.replace('   core_cycles<=0;', ''.join('   core_'+x+'<=0;\n' for x in AUX)+'   core_cycles<=0;')
w=w.replace('    if(|leaf_done)core_cycles', ''.join("    if(|leaf_done)core_"+x+"<=core_"+x+"+(leaf_done[0]?64'(l_"+x+"[0]):64'd0)+(leaf_done[1]?64'(l_"+x+"[1]):64'd0);\n" for x in AUX)+'    if(|leaf_done)core_cycles')
w=w.replace('endmodule', '''`ifdef VERILATOR
 logic [1:0] check_owned;
 always_ff @(posedge clk)begin
  if(!reset_n)check_owned<=0;
  else begin
   for(integer check_ctx=0;check_ctx<2;check_ctx=check_ctx+1)begin
    if(core_start[check_ctx])begin
     if(check_owned[check_ctx])$fatal(1,"restarting context before consumer retirement");
     check_owned[check_ctx]<=1;
    end
    if(check_owned[check_ctx]&&core_cfg_valid[check_ctx])$fatal(1,"reconfiguring live psum owner");
    if(grant[check_ctx]&&!eligible[check_ctx])$fatal(1,"grant without eligibility");
    if(grant[check_ctx]&&request[check_ctx][5]&&!request[check_ctx][2])$fatal(1,"borrow without z permission");
   end
   if(state==RUN&&consumer_done)begin
    if(!check_owned[consumer_select])$fatal(1,"consumer without context ownership");
    check_owned[consumer_select]<=0;
   end
   if(grant==3&&((request[0]&request[1])!=0))$fatal(1,"shared resource double grant");
   if(cons_add_grant&&borrow_active)$fatal(1,"wide chain double use");
   if(proof_active&&((grant[0]&&request[0][4])||(grant[1]&&request[1][4])))$fatal(1,"proof double use");
  end
 end
`endif
endmodule''')
(H/'interleave_stream.sv').write_text(w)
for file in ['wide_phase_alu.sv','i24_consumer.sv','stream_tb.cpp','tb.cpp']:
    shutil.copyfile(R/file,H/file)
for file in ['stream_tb.cpp','tb.cpp']:
    t=(H/file).read_text()
    t=t.replace('int main(int argc', 'double sc_time_stamp(){return 0;}\nint main(int argc')
    t=t.replace('param[8];for(int k:{4,5,6,7})','param[12];for(int k:{4,5,6,7,8,9,10,11})')
    t=t.replace('for(unsigned command=0;', 'bool class_loaded=false,permutation_loaded=false;\n for(unsigned command=0;',1)
    t=t.replace('  d.mode=active_mode;', '''  unsigned expected_static=command?0:1848;
  if(active_mode>=20&&!class_loaded){expected_static+=897;class_loaded=true;}
  if(active_mode==21&&!permutation_loaded){expected_static++;permutation_loaded=true;}
  d.mode=active_mode;''')
    t=t.replace('(active_mode>=3?0:d.core_first_issues)', '((active_mode==3||active_mode==4)?0:d.core_first_issues)')
    t=t.replace('+d.core_normalization_issues)return 36;', '+d.core_normalization_issues+d.core_aux_issues)return 36;')
    t=t.replace('(active_mode>=3?d.core_first_issues:0)', '((active_mode==3||active_mode==4)?d.core_first_issues:0)')
    t=t.replace('d.shared_weight_grants!=d.core_weight_words', 'd.shared_weight_grants!=d.core_weight_words+d.core_metadata_reads')
    t=t.replace('d.shared_psum_grants!=d.core_psum_reads+d.core_psum_writes', 'd.shared_psum_grants!=d.core_psum_reads+d.core_psum_writes+d.core_aux_reads+d.core_aux_writes')
    t=t.replace('param_count!=(command?0:1848)', 'param_count!=expected_static')
    t=t.replace('    SHOW(core_cycles);', '    '+''.join('SHOW(core_'+x+');' for x in AUX)+'\n    SHOW(core_cycles);')
    t=t.replace('return 20;','{std::cerr<<"raw mismatch mode="<<active_mode<<" row="<<raws%480<<" lane="<<l<<" got="<<int32_t(d.raw_monitor_data[l])<<"\\n";return 20;}')
    (H/file).write_text(t)
print('Wrote isolated RR/count consumer RTL')
