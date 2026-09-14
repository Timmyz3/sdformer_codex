from pathlib import Path
H=Path(__file__).resolve().parent;p=H/'lossy_r8.sv';s=p.read_text()
s=s.replace("scalar=read_half?$signed(z_mem[selected_rank][read_zrow][25:13]):$signed(z_mem[selected_rank][read_zrow][12:0]);","scalar=read_half?14'($signed(z_mem[selected_rank][read_zrow][25:13])):14'($signed(z_mem[selected_rank][read_zrow][12:0]));")
s=s.replace('logic changed;','logic changed,delta_fits,select_delta;\n logic [3:0] delta_nnz,approx_nnz;\n logic use_delta[0:39];\n logic signed [12:0] encoded_z[0:7];')
s=s.replace('mode_q==5','(mode_q==5 || mode_q==8)').replace('mode_q==6','(mode_q==6 || mode_q==7)')
s=s.replace('mode_q!=5 && mode_q!=6','mode_q!=5 && mode_q!=6 && mode_q!=7 && mode_q!=8')
# difference is actual current minus retained approximation from paid EDIFF; rank deadband makes dropped differences zero.
marker=' for(integer l=0;l<8;l=l+1)begin\n multiply_coefficient'
insert=''' delta_nnz=0;approx_nnz=0;delta_fits=1;
 for(integer i=0;i<8;i=i+1)begin
  if(approximate[i]!=0)approx_nnz=approx_nnz+1;
  if(approximate[i]!=reference_z[i])begin
   delta_nnz=delta_nnz+1;
   if(difference[i]>14'sd4095 || difference[i]<-14'sd4096)delta_fits=0;
  end
 end
 select_delta=(mode_q==7 || mode_q==8) && fp%10!=0 && delta_fits && delta_nnz<approx_nnz;
 for(integer i=0;i<8;i=i+1)encoded_z[i]=select_delta?((approximate[i]!=reference_z[i])?difference[i][12:0]:13'd0):approximate[i];
'''
s=s.replace(marker,insert+marker)
s=s.replace('z_mem[i][read_zrow]<=read_half?{approximate[i],z_hold[i][12:0]}:{z_hold[i][25:13],approximate[i]};','z_mem[i][read_zrow]<=read_half?{encoded_z[i],z_hold[i][12:0]}:{z_hold[i][25:13],encoded_z[i]};')
s=s.replace('refresh[fp]<=','use_delta[fp]<=select_delta;\n refresh[fp]<=')
s=s.replace('for(integer i=0;i<8;i=i+1)acc[i]<=0;\n if(mode_q==3', 'if(!((mode_q==7 || mode_q==8) && use_delta[fp]))for(integer i=0;i<8;i=i+1)acc[i]<=0;\n if(mode_q==3')
s=s.replace('if(state==ZSCAN)for(integer i=0;i<8;i=i+1)','if(state==ZSCAN || state==EREAD)for(integer i=0;i<8;i=i+1)')
s=s.replace('state<=(((mode_q==5 || mode_q==8) || (mode_q==6 || mode_q==7)) && fp%10==0)?EWRITE:EDIFF;','if(mode_q>=1 && mode_q<=4 && scan_mask==0)begin\n codes[fp]<=0;residual_rank[fp]<=0;residual[fp]<=0;state<=EWRITE;\n end else state<=(((mode_q==5 || mode_q==8) || (mode_q==6 || mode_q==7)) && fp%10==0)?EWRITE:EDIFF;')
p.write_text(s)
