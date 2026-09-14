from pathlib import Path
H=Path(__file__).resolve().parent;A=H.parent/'spatial_r16_rtl'
s=(A/'spatial_core.sv').read_text()
s=s.replace('output logic [31:0] source_stalls,weight_stalls,output_stalls,','output logic [31:0] source_stalls,weight_stalls,output_stalls,count_constructs,count2_fields,count_nonzero_fields,')
s=s.replace('src_masks[0:7];','src_masks[0:7];\n logic [9:0] count_nonzero[0:7],count_two[0:7]; // combinational wires, no count SRAM')
s=s.replace('q2_mem[0:7][0:575],qcache[0:7][0:23]','q2_mem[0:7][0:383],qcache[0:7][0:15]').replace('q2_live[0:575]','q2_live[0:383]')
s=s.replace('logic [23:0] block_live,remaining,next_support','logic [15:0] block_live,remaining,next_support')
s=s.replace('logic in_bounds,active_low,active_high,z_read_enable','logic [1:0] active_low,active_high;\n logic in_bounds,z_read_enable')
s=s.replace('logic signed [14:0] z_scalar','logic signed [15:0] z_scalar')
s=s.replace('source_stalls<=0;weight_stalls<=0;output_stalls<=0;','source_stalls<=0;weight_stalls<=0;output_stalls<=0;count_constructs<=0;count2_fields<=0;count_nonzero_fields<=0;')
s=s.replace('  selected_time=0;', '''  count_nonzero[3]=0;count_nonzero[7]=0;count_two[3]=0;count_two[7]=0;
  for(integer y=0;y<2;y=y+1)for(integer x=0;x<3;x=x+1)begin
   count_nonzero[y*4+x]=src_masks[y*4+x]|src_masks[y*4+x+1];
   count_two[y*4+x]=src_masks[y*4+x]&src_masks[y*4+x+1];
  end
  selected_time=0;''')
s=s.replace('for(integer t=23;t>=0;t=t-1)if(remaining[t])','for(integer t=15;t>=0;t=t-1)if(remaining[t])')
s=s.replace('selected_term/3','selected_term/2').replace('selected_term%3','selected_term%2')
s=s.replace("int'(stripe)*288+og*24+qfill","int'(stripe)*192+og*16+qfill")
s=s.replace('for(integer x=0;x<3;x=x+1)\n   next_support[r*3+x]=block_live[r*3+x]','for(integer x=0;x<2;x=x+1)\n   next_support[r*2+x]=block_live[r*2+x]')
s=s.replace('qfill/3','qfill/2').replace('qfill==23','qfill==15').replace("remaining-24'd1","remaining-16'd1")
s=s.replace('[29:15]','[31:16]').replace('[14:0]','[15:0]')
s=s.replace('{{4{z_scalar[14]}},z_scalar}','{{3{z_scalar[15]}},z_scalar}')
old="""    rhs[l]={2'd0,(active_high?{{7{q1_hold[l][7]}},q1_hold[l]}:15'd0),
                    (active_low?{{7{q1_hold[l][7]}},q1_hold[l]}:15'd0)};"""
new="""    rhs[l]={(active_high==2?{{7{q1_hold[l][7]}},q1_hold[l],1'b0}:
                         active_high==1?{{8{q1_hold[l][7]}},q1_hold[l]}:16'd0),
            (active_low==2?{{7{q1_hold[l][7]}},q1_hold[l],1'b0}:
                         active_low==1?{{8{q1_hold[l][7]}},q1_hold[l]}:16'd0)};"""
assert old in s;s=s.replace(old,new).replace('else if(b==15)','else if(b==16)')
s=s.replace('pending[i*10+:10]<=src_masks[i*2]|src_masks[i*2+1];','pending[i*10+:10]<=count_nonzero[i*2]|count_nonzero[i*2+1];')
s=s.replace('    CHECK:begin','    CHECK:begin\n     count_constructs<=count_constructs+1;')
s=s.replace('active_low<=src_masks[2*(selected_time/10)][selected_time%10];','active_low<=count_two[2*(selected_time/10)][selected_time%10]?2:count_nonzero[2*(selected_time/10)][selected_time%10]?1:0;')
s=s.replace('active_high<=src_masks[2*(selected_time/10)+1][selected_time%10];','active_high<=count_two[2*(selected_time/10)+1][selected_time%10]?2:count_nonzero[2*(selected_time/10)+1][selected_time%10]?1:0;')
s=s.replace("z_mem[i][zrow]<={2'd0,add_y[i][29:0]};","z_mem[i][zrow]<=add_y[i];")
s=s.replace('    ZADD:begin','    ZADD:begin\n     count2_fields<=count2_fields+32\'(active_low==2)+32\'(active_high==2);\n     count_nonzero_fields<=count_nonzero_fields+32\'(active_low!=0)+32\'(active_high!=0);')
s=s.replace('   if((state==QREAD||state==VLOAD)&&(weight_addr<0||weight_addr>=576))', '   if((state==QREAD&&(weight_addr<0||weight_addr>=576))||(state==VLOAD&&(weight_addr<0||weight_addr>=384)))')
begin=s.index('   if(state==ZADD)for(integer i=0;i<8;i=i+1)begin');end=s.index('\n  end\n end\n `endif',begin)
s=s[:begin]+'''   if(state==ZADD)begin
    if(active_low>2||active_high>2)$fatal(1,"invalid source count");
    for(integer i=0;i<8;i=i+1)begin
     if(32'($signed(z_hold[i][15:0]))+32'($signed(rhs[i][15:0]))<-32768||
        32'($signed(z_hold[i][15:0]))+32'($signed(rhs[i][15:0]))>32767)$fatal(1,"low E16 overflow");
     if(32'($signed(z_hold[i][31:16]))+32'($signed(rhs[i][31:16]))<-32768||
        32'($signed(z_hold[i][31:16]))+32'($signed(rhs[i][31:16]))>32767)$fatal(1,"high E16 overflow");
    end
   end
   if(state==BASE_MAC)for(integer i=0;i<8;i=i+1)
    if(64'($signed(lhs[i]))+64'($signed(rhs[i]))!=64'($signed(add_y[i])))$fatal(1,"Q2 signed32 prefix overflow");'''+s[end:]
s=s.replace('q2_mem[i][cfg_addr[9:0]]','q2_mem[i][cfg_addr[8:0]]').replace('q2_live[cfg_addr[9:0]]','q2_live[cfg_addr[8:0]]')
(H/'spatial_core.sv').write_text(s)
for name in ['spatial_stream.sv','i24_consumer.sv','wide_phase_alu.sv','tb.cpp','stream_tb.cpp']:
 text=(A/name).read_text()
 if name=='spatial_stream.sv':
  text=text.replace('output_stalls, c_cycles','output_stalls, count_constructs, count2_fields, count_nonzero_fields, c_cycles')
  text=text.replace(' .source_words(source_words),',' .source_words(source_words),\n .count_constructs(count_constructs),.count2_fields(count2_fields),.count_nonzero_fields(count_nonzero_fields),')
 if name.endswith('.cpp'):
  text=text.replace('q2.size()!=4608','q2.size()!=3072').replace('a<576;a++){for(int i=0;i<8;i++)data[i]=q2','a<384;a++){for(int i=0;i<8;i++)data[i]=q2')
  if name=='tb.cpp':text=text.replace('field("source_words",d.source_words);','field("source_words",d.source_words);field("count_constructs",d.count_constructs);field("count2_fields",d.count2_fields);field("count_nonzero_fields",d.count_nonzero_fields);')
  else:text=text.replace('F(source_words);','F(source_words);F(count_constructs);F(count2_fields);F(count_nonzero_fields);')
 (H/name).write_text(text)
print('Box2: ordinary core derived, dual16 counts, two-tap Q2, original full consumer')
