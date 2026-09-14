"""Instantiate the frozen complete consumer, changing the arithmetic interface only."""
from pathlib import Path
H=Path(__file__).resolve().parent
B=H.parent.parent/'r8_consumer_fusion_20260914'

c=(B/'packed_rtl/packed_r8.sv').read_text().replace('module packed_r8(', 'module phase_core(')
c=c.replace('input logic start,input logic [3:0] mode,input logic source_allow,weight_allow,', '''input logic start,input logic [3:0] mode,input logic source_allow,weight_allow,
 output logic borrow_req,input logic borrow_grant,
 output logic [511:0] borrow_lhs,borrow_rhs,input logic [415:0] borrow_y,
 output logic [31:0] borrow_waits,''')
c=c.replace('logic [25:0] z_mem[0:7][0:19],z_hold[0:7];','logic [51:0] z_mem[0:7][0:9],z_hold[0:7];')
c=c.replace('logic in_bounds,cfg_v_live,active_lo,active_hi,read_half;', 'logic in_bounds,cfg_v_live,pair_sel; logic [3:0] active;')
c=c.replace('z_vector_reads<=0;z_scalar_reads<=0;z_writes<=0;', 'borrow_waits<=0;z_vector_reads<=0;z_scalar_reads<=0;z_writes<=0;')
c=c.replace("read_zrow=5'((fp/20)*10+fp%10);read_half=((fp/10)%2)!=0;", "read_zrow=4'(fp%10);")
c=c.replace('logic [4:0] read_zrow;', 'logic [3:0] read_zrow;')
c=c.replace('scan_mask[i]=read_half?(z_mem[i][read_zrow][25:13]!=0):(z_mem[i][read_zrow][12:0]!=0);', 'scan_mask[i]=(z_mem[i][read_zrow][13*(fp/10)+:13]!=0);')
c=c.replace('scalar=read_half?$signed(z_mem[selected_rank][read_zrow][25:13]):$signed(z_mem[selected_rank][read_zrow][12:0]);', 'scalar=$signed(z_mem[selected_rank][read_zrow][13*(fp/10)+:13]);\n borrow_req=(state==ZADD && mode_q==3);')
c=c.replace('lhs[l]={6\'b0,z_hold[l]};', "lhs[l]={6'b0,z_hold[l][26*int'(pair_sel)+:26]};")
c=c.replace('active_hi?', "active[2*int'(pair_sel)+1]?").replace('active_lo?', "active[2*int'(pair_sel)]?")
c=c.replace('end\n end\n result_valid=', '''end
 borrow_lhs[l*64+:64]={12'b0,z_hold[l]};
 borrow_rhs[l*64+:64]=64'd0;
 for(integer p=0;p<4;p=p+1)
 borrow_rhs[l*64+p*13+:13]=active[p]?{{10{q_hold[l][2]}},q_hold[l]}:13'd0;
 end
 result_valid=''',1)
c=c.replace('mode_q<=14;', 'mode_q<=2;').replace('active_lo<=0;active_hi<=0;', 'active<=0;pair_sel<=0;')
c=c.replace('if(zrow==19)', 'if(zrow==9)')
c=c.replace("if(mode_q==15)pending<={20'd0,(src_masks[2]|src_masks[3]),(src_masks[0]|src_masks[1])};\n else pending<={src_masks[3],src_masks[2],src_masks[1],src_masks[0]};", "if(mode_q==3)pending<={30'd0,(src_masks[0]|src_masks[1]|src_masks[2]|src_masks[3])};\n else pending<={20'd0,(src_masks[2]|src_masks[3]),(src_masks[0]|src_masks[1])};")
start=c.index(' if(mode_q==15)begin',c.index(' TIMESEL:'))
end=c.index(' state<=ZREAD;',start)
c=c[:start]+''' zrow<=selected_time%10;pair_sel<=(selected_time>=10);
 for(integer p=0;p<4;p=p+1)
 active[p]<=src_masks[p][selected_time%10] && (mode_q==3 || p/2==selected_time/10);
'''+c[end:]
start=c.index(' ZADD:begin')
end=c.index(' KNEXT:',start)
c=c[:start]+''' ZADD:if(mode_q!=3 || borrow_grant)begin
 for(integer i=0;i<8;i=i+1)begin
 if(mode_q==3)z_mem[i][zrow]<=borrow_y[i*52+:52];
 else if(pair_sel)z_mem[i][zrow]<={add_y[i][25:0],z_hold[i][25:0]};
 else z_mem[i][zrow]<={z_hold[i][51:26],add_y[i][25:0]};
 end
 first_issues<=first_issues+1;z_writes<=z_writes+1;
 dual_updates<=dual_updates+32'(active[0])+32'(active[1])+32'(active[2])+32'(active[3])-32'd1;
 pending<=pending&(pending-40'd1);state<=TIMESEL;
 end else borrow_waits<=borrow_waits+1;
'''+c[end:]
c=c.replace('dual_updates','merged_updates')
(H/'phase_core.sv').write_text(c)

c=(B/'consumer_packed/i24_consumer.sv').read_text()
c=c.replace('input logic clk, reset_n, start,', '''input logic clk, reset_n, start,
  output logic add_req,input logic add_grant,
  output logic [511:0] add_lhs_bus,add_rhs_bus,input logic [511:0] add_y_bus,
  output logic [31:0] wide_waits,''')
c=c.replace('raw_ready=(state==GET && !have_p);', 'add_req=(state==ADD_BIAS || state==ADD_IDENTITY);\n    raw_ready=(state==GET && !have_p);')
c=c.replace('add_result[l]=wide_hold[l]+add_rhs[l];','add_lhs_bus[l*64+:64]=wide_hold[l];add_rhs_bus[l*64+:64]=add_rhs[l];\n      add_result[l]=$signed(add_y_bus[l*64+:64]);')
c=c.replace('conversion_issues<=0;conversion_saturations<=0;', 'wide_waits<=0;conversion_issues<=0;conversion_saturations<=0;')
c=c.replace('ADD_BIAS: begin', 'ADD_BIAS: if(add_grant) begin').replace('ADD_IDENTITY: begin','ADD_IDENTITY: if(add_grant) begin')
c=c.replace('state<=ADD_IDENTITY;\n        end', 'state<=ADD_IDENTITY;\n        end else wide_waits<=wide_waits+1;')
c=c.replace('state<=ROUND;\n        end', 'state<=ROUND;\n        end else wide_waits<=wide_waits+1;')
(H/'i24_consumer.sv').write_text(c)

c=(B/'consumer_packed/consumer_stream.sv').read_text()
c=c.replace('input logic compute_source_allow, compute_weight_allow,','input logic compute_source_allow, compute_weight_allow, compute_wide_allow,')
c=c.replace('output logic [63:0] core_cycles,', 'output logic [63:0] core_borrow_waits, consumer_wide_waits,\n output logic [63:0] core_cycles,')
c=c.replace('typedef enum logic [2:0]', '''logic borrow_req,borrow_grant,cons_add_req,cons_add_grant;
 logic [511:0] borrow_lhs,borrow_rhs,cons_lhs,cons_rhs,wide_lhs,wide_rhs,wide_y;
 logic [415:0] borrow_y;
 logic [31:0] l_borrow_waits,c_wide_waits;
 assign cons_add_grant=cons_add_req && compute_wide_allow;
 assign borrow_grant=borrow_req && !cons_add_req && compute_wide_allow;
 assign wide_lhs=cons_add_req?cons_lhs:borrow_lhs;
 assign wide_rhs=cons_add_req?cons_rhs:borrow_rhs;
 wide_phase_alu wide_alu(.split_fields(!cons_add_req),.lhs(wide_lhs),.rhs(wide_rhs),.y(wide_y));
 genvar wl;
 generate for(wl=0;wl<8;wl=wl+1)begin:UNPACK
 assign borrow_y[wl*52+:52]=wide_y[wl*64+:52];
 end endgenerate
 typedef enum logic [2:0]''',1)
c=c.replace('packed_r8 leaf (', 'phase_core leaf (\n  .borrow_req(borrow_req),.borrow_grant(borrow_grant),.borrow_lhs(borrow_lhs),.borrow_rhs(borrow_rhs),.borrow_y(borrow_y),.borrow_waits(l_borrow_waits),')
c=c.replace('i24_consumer consumer (', 'i24_consumer consumer (\n  .add_req(cons_add_req),.add_grant(cons_add_grant),.add_lhs_bus(cons_lhs),.add_rhs_bus(cons_rhs),.add_y_bus(wide_y),.wide_waits(c_wide_waits),')
c=c.replace('mode_q<=14;', 'mode_q<=2;').replace('(mode!=14 && mode!=15)', '(mode!=2 && mode!=3)')
c=c.replace('core_cycles<=0;', 'core_borrow_waits<=0;consumer_wide_waits<=0;core_cycles<=0;')
c=c.replace('core_cycles<=core_cycles+', 'core_borrow_waits<=core_borrow_waits+64\'(l_borrow_waits);\n      core_cycles<=core_cycles+')
c=c.replace('consumer_cycles<=consumer_cycles+', 'consumer_wide_waits<=consumer_wide_waits+64\'(c_wide_waits);\n      consumer_cycles<=consumer_cycles+')
c=c.replace('dual_updates','merged_updates')
(H/'consumer_stream.sv').write_text(c)

for f in ['prepare.py','run.py','run_stream.py','tb.cpp','stream_tb.cpp']:
 c=(B/'consumer_packed'/f).read_text()
 c=c.replace("D=H.parent/'data'", "D=H.parent.parent/'r8_consumer_fusion_20260914/data'")
 c=c.replace("'../packed_rtl/packed_r8.sv'", "'phase_core.sv','wide_phase_alu.sv'")
 c=c.replace('[14,15]','[2,3]').replace('dual_updates','merged_updates')
 c=c.replace('d.compute_weight_allow=1;', 'd.compute_weight_allow=1;d.compute_wide_allow=1;')
 c=c.replace('d.result_ready=(!stall', 'd.compute_wide_allow=(!stall || n%17!=4);\n   d.result_ready=(!stall')
 c=c.replace('SHOW(core_cycles);', 'SHOW(core_borrow_waits);SHOW(consumer_wide_waits);SHOW(core_cycles);')
 (H/f).write_text(c)
print('Wrote explicit shared-adder complete consumer sources; old stage is read-only')
