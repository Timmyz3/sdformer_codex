from pathlib import Path
H=Path(__file__).resolve().parent;O=H.parent
s=(O/'decomp_core.sv').read_text()
s=s.replace('state_t state;logic [4:0] mode_q;','state_t state;logic [4:0] mode_q;\n logic [39:0] time_permutation;logic [3:0] inverse_time;')
s=s.replace('mode_q==20','(mode_q==20||mode_q==21)')
s=s.replace('source_addr=11\'', '''// One configured permutation; inverse is combinational, with no second table.
 inverse_time=4'(row%10);
 if(mode_q==21)for(integer t=0;t<10;t=t+1)
  if(time_permutation[t*4+:4]==4'(row%10))inverse_time=4'(t);
 source_addr=11\'''' )
s=s.replace("p_addr[i]=9'(row);","p_addr[i]=9'((row/10)*10+int'(inverse_time));")
s=s.replace('state<=IDLE;mode_q<=14;',"time_permutation<=40'h9876543210;state<=IDLE;mode_q<=14;")
s=s.replace('0:source_mem[cfg_addr[10:0]]<=cfg_data[9:0];','0:source_mem[cfg_addr[10:0]]<=cfg_data[9:0];\n 2:time_permutation<=cfg_data[39:0];')
s=s.replace('for(integer i=0;i<4;i=i+1)src_masks[i]<=local_source[(i/2+(k%9)/3)*4+i%2+k%3];', '''for(integer i=0;i<4;i=i+1)begin
  if(mode_q==21)for(integer t=0;t<10;t=t+1)
   src_masks[i][t]<=local_source[(i/2+(k%9)/3)*4+i%2+k%3][time_permutation[t*4+:4]];
  else src_masks[i]<=local_source[(i/2+(k%9)/3)*4+i%2+k%3];
 end''')
s=s.replace('if(state==IDLE&&start)begin check_retired<=0;check_stores<=0;end', '''if(state==IDLE&&start)begin
   check_retired<=0;check_stores<=0;
   if(mode==21)for(integer t=0;t<10;t=t+1)begin
    if(time_permutation[t*4+:4]>=10)$fatal(1,"permutation range");
    for(integer u=0;u<t;u=u+1)if(time_permutation[t*4+:4]==time_permutation[u*4+:4])$fatal(1,"permutation duplicate");
   end
  end''')
(H/'decomp_core.sv').write_text(s)
cfg=''' if(mode==21){
 // Fixed endpoint-calibrated order: new time index -> original time index.
 const unsigned order[10]={8,2,6,3,9,4,1,5,7,0};uint64_t packed=0;
 for(int t=0;t<10;t++)packed|=uint64_t(order[t])<<(4*t);
 data[0]=uint32_t(packed);data[1]=uint32_t(packed>>32);cfg(2,0);
 }
'''
for file in ['tb.cpp','stream_tb.cpp']:
 t=(O/file).read_text().replace(' d.cfg_valid=0;unsigned checked=0;',cfg+' d.cfg_valid=0;unsigned checked=0;')
 (H/file).write_text(t)
print('generated isolated modes14/20/21; original mode20 files untouched')
