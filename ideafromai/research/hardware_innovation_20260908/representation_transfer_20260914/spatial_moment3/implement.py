"""Derive one constrained-FIR moment3 datapath from the locked general core."""
from pathlib import Path
H=Path(__file__).resolve().parent;A=H.parent/'spatial_winograd'
s=(A/'spatial_core.sv').read_text()
s=s.replace('XD0,XD1,XD2,XD3,W_INV0A,W_INV0B,','XD0,XD1,XD2,W_INV0A,').replace('W_INV1A,W_INV1B,','W_INV1A,')
s=s.replace('m_aux[0:2]','m_aux[0:1]').replace('m<3','m<2')
s=s.replace('q2_mem[0:7][0:767],qcache[0:7][0:31]','q2_mem[0:7][0:575],qcache[0:7][0:23]').replace('q2_live[0:767]','q2_live[0:575]')
s=s.replace('logic [31:0] block_live,remaining,next_support','logic [23:0] block_live,remaining,next_support')
s=s.replace('for(integer t=31;t>=0;t=t-1)if(remaining[t])','for(integer t=23;t>=0;t=t-1)if(remaining[t])')
for old,new in [('(mode_q?4:3)','3'),('(mode_q?384:288)','288'),('(mode_q?32:24)','24'),('(mode_q?31:23)','23'),('(mode_q?768:576)','576'),('selected_term%4','selected_term%3'),('remaining-32\'d1','remaining-24\'d1')]:s=s.replace(old,new)
s=s.replace('for(integer x=0;x<4;x=x+1)\n   next_support[r*4+x]=block_live[r*4+x]', 'for(integer x=0;x<3;x=x+1)\n   next_support[r*3+x]=block_live[r*3+x]')
s=s.replace('sub_alu=(state==XD0||state==XD2||state==XD3||state==W_INV1A||state==W_INV1B);','sub_alu=(state==XD0||state==XD2||state==W_INV1A);')
s=s.replace('XD2:begin lhs[l]=32\'($signed(transform_tail[l][14:0]));rhs[l]=32\'($signed(z_hold[l][29:15]));end\n    XD3:begin', 'XD2:begin')
s=s.replace('    W_INV0B:rhs[l]=m_aux[1][l];\n','').replace('    W_INV1B:rhs[l]=m_aux[2][l];\n','')
s=s.replace('d_monitor_valid=state==XD1||state==XD3;d_monitor_addr=6\'(transform_addr+((state==XD3)?10:0));', 'd_monitor_valid=state==XD1||state==XD2;d_monitor_addr=6\'(transform_addr+((state==XD2)?10:0));')
s=s.replace('d_monitor_data[i*32+:32]={add_y[i][15:0],acc[i][15:0]};','d_monitor_data[i*32+:32]=(state==XD2)?{16\'d0,add_y[i][15:0]}:{add_y[i][15:0],acc[i][15:0]};')
begin=s.index('    XD0,XD1,XD2,XD3:begin');end=s.index('    W_PREAD0,W_PREAD1:begin',begin)
s=s[:begin]+'''    XD0,XD1,XD2:begin
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
'''+s[end:]
s=s.replace('   if(state==W_INV0B||state==W_INV1B)for(integer i=0;i<8;i=i+1)\n    if(add_y[i][0])$fatal(1,"non-exact Winograd half");\n','')
s=s.replace('||state==XD3','').replace('||state==W_INV0B','').replace('||state==W_INV1B','')
assert not any(v in s for v in ['XD3','W_INV0B','W_INV1B','m_aux[2]','>>>1','0:767','0:31]'])
(H/'spatial_core.sv').write_text(s)
# Wrappers and harness retain the original consumer endpoints. The only
# harness change is selecting the original three-tap coefficient table.
for name in ['spatial_stream.sv','i24_consumer.sv','wide_phase_alu.sv','tb.cpp','stream_tb.cpp']:
 text=(A/name).read_text()
 if name.endswith('.cpp'):
  text=text.replace('(mode?"/parameters/wq2.hex":"/parameters/q2.hex")','"/parameters/q2.hex"')
  text=text.replace('(mode?6144:4608)','(4608)').replace('(mode?768:576)','(576)')
 (H/name).write_text(text)
print('moment3 derived: original Q2 table576, cache24, 3M, 3D, 2 inverse adds, no half')
