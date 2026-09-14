from pathlib import Path
H=Path(__file__).resolve().parent;B=H.parents[1];OLD=H.parent/'spatial_r16_rtl'
EXTRA=['transform_issues','transform_reads','transform_writes','reconstruction_issues','stripe_add_issues','cache_reads','exact_halves']
def sub(s,a,b):
 assert a in s,a
 return s.replace(a,b)
s=(OLD/'spatial_core.sv').read_text()
s=sub(s,' input logic clk,reset_n,cfg_valid,start,',' input logic clk,reset_n,cfg_valid,start,mode,')
s=sub(s,' output logic [5:0] debug_state,',' output logic [31:0] '+','.join(EXTRA)+',\n output logic d_monitor_valid,output logic [5:0] d_monitor_addr,output logic [255:0] d_monitor_data,\n output logic [5:0] debug_state,')
s=sub(s,'DRAIN_SEND,FINISH} state_t;','DRAIN_SEND,FINISH,XDREAD0,XDREAD1,XD0,XD1,XD2,XD3,W_INV0A,W_INV0B,W_PREAD0,W_PADD0,W_STORE0,W_INV1A,W_INV1B,W_PREAD1,W_PADD1,W_STORE1} state_t;')
s=sub(s,'state_t state;logic stripe;','state_t state;logic stripe,mode_q;\n logic [31:0] transform_tail[0:7];logic signed [31:0] m_aux[0:2][0:7];\n integer tx,transform_addr;logic sub_alu;logic [7:0] transform_support;')
s=sub(s,'q2_mem[0:7][0:575],qcache[0:7][0:23]','q2_mem[0:7][0:767],qcache[0:7][0:31]')
s=sub(s,'q1_live[0:575],q2_live[0:575]','q1_live[0:575],q2_live[0:767]')
s=sub(s,'logic [23:0] block_live,remaining,next_support;','logic [31:0] block_live,remaining,next_support;')
s=sub(s,'logic signed [14:0] z_scalar;','logic signed [15:0] z_scalar;')
s=sub(s,' cycles<=0;source_words<=0;', ' '+''.join(f'{c}<=0;' for c in EXTRA)+'\n cycles<=0;source_words<=0;')
s=sub(s,'selected_term=0;for(integer t=23;t>=0;t=t-1)','selected_term=0;for(integer t=31;t>=0;t=t-1)')
s=sub(s,"selected_rank=3'(selected_term/3);","selected_rank=3'(selected_term/(mode_q?4:3));")
s=sub(s,'source_p=(output_p/2)*4+output_p%2+selected_term%3;','source_p=mode_q?(fp/10)*4+selected_term%4:(output_p/2)*4+output_p%2+selected_term%3;')
s=sub(s,'  z_addr=(state==BASE_MAC)?(source_p/2)*10+fp%10:zrow;', '''  transform_addr=(tx/10)*20+tx%10;
  z_addr=(state==XDREAD0)?transform_addr:(state==XDREAD1)?transform_addr+10:(state==BASE_MAC)?(source_p/2)*10+fp%10:zrow;''')
s=sub(s,'z_read_enable=(state==ZREAD||state==ZSCAN||state==BASE_MAC);','z_read_enable=(state==ZREAD||state==ZSCAN||state==BASE_MAC||state==XDREAD0||state==XDREAD1);')
s=sub(s,"p_addr=(state==DRAIN_READ)?9'(row):9'(og*40+fp);", """p_addr=(state==DRAIN_READ)?9'(row):mode_q?9'(og*40+(fp/10)*20+fp%10+((state==W_PREAD1||state==W_PADD1||state==W_STORE1)?10:0)):9'(og*40+fp);""")
s=sub(s,'p_read_enable=(state==DRAIN_READ)||(state==POSLOAD&&stripe);','p_read_enable=(state==DRAIN_READ)||(state==POSLOAD&&stripe&&!mode_q)||state==W_PREAD0||state==W_PREAD1;')
s=sub(s,'p_write_enable=(state==STORE);','p_write_enable=(state==STORE||state==W_STORE0||state==W_STORE1);')
s=sub(s,'weight_addr=state==QREAD?int\'(stripe)*288+k:int\'(stripe)*288+og*24+qfill;',"weight_addr=state==QREAD?int'(stripe)*288+k:int'(stripe)*(mode_q?384:288)+og*(mode_q?32:24)+qfill;")
s=sub(s,'  for(integer r=0;r<8;r=r+1)for(integer x=0;x<3;x=x+1)\n   next_support[r*3+x]=block_live[r*3+x]&&position_live[((output_p/2)*4+output_p%2+x)*10+fp%10][r];', '''  if(mode_q)for(integer r=0;r<8;r=r+1)for(integer x=0;x<4;x=x+1)
   next_support[r*4+x]=block_live[r*4+x]&&position_live[((fp/10)*4+x)*10+fp%10][r];
  else for(integer r=0;r<8;r=r+1)for(integer x=0;x<3;x=x+1)
   next_support[r*3+x]=block_live[r*3+x]&&position_live[((output_p/2)*4+output_p%2+x)*10+fp%10][r];''')
s=s.replace('rank_live[qfill/3]','rank_live[qfill/(mode_q?4:3)]')
s=sub(s,'  z_scalar=selected_half?$signed(z_read_word[selected_rank][29:15]):$signed(z_read_word[selected_rank][14:0]);', '''  z_scalar=mode_q?(selected_half?$signed(z_read_word[selected_rank][31:16]):$signed(z_read_word[selected_rank][15:0])):
            (selected_half?16'($signed(z_read_word[selected_rank][29:15])):16'($signed(z_read_word[selected_rank][14:0])));
  sub_alu=(state==XD0||state==XD2||state==XD3||state==W_INV1A||state==W_INV1B);''')
s=sub(s,"multiply_lhs[l]=(state==BASE_MAC)?{{4{z_scalar[14]}},z_scalar}:19'sd0;","multiply_lhs[l]=(state==BASE_MAC)?{{3{z_scalar[15]}},z_scalar}:19'sd0;")
s=sub(s,'   lhs[l]=acc[l];rhs[l]=product[l];','''   lhs[l]=acc[l];rhs[l]=product[l];
   if(state==BASE_MAC&&mode_q&&selected_term%4!=0)lhs[l]=m_aux[selected_term%4-1][l];
   case(state)
    XD0:begin lhs[l]=32'($signed(z_hold[l][14:0]));rhs[l]=32'($signed(transform_tail[l][14:0]));end
    XD1:begin lhs[l]=32'($signed(z_hold[l][29:15]));rhs[l]=32'($signed(transform_tail[l][14:0]));end
    XD2:begin lhs[l]=32'($signed(transform_tail[l][14:0]));rhs[l]=32'($signed(z_hold[l][29:15]));end
    XD3:begin lhs[l]=32'($signed(z_hold[l][29:15]));rhs[l]=32'($signed(transform_tail[l][29:15]));end
    W_INV0A:rhs[l]=m_aux[0][l];
    W_INV0B:rhs[l]=m_aux[1][l];
    W_INV1A:begin lhs[l]=m_aux[0][l];rhs[l]=m_aux[1][l];end
    W_INV1B:rhs[l]=m_aux[2][l];
    W_PADD0,W_PADD1:rhs[l]=$signed(z_hold[l]);
    default:begin end
   endcase''')
s=sub(s,'  result_valid=state==DRAIN_SEND;', '''  transform_support=0;
  for(integer i=0;i<8;i=i+1)transform_support[i]=(add_y[i][15:0]!=0);
  d_monitor_valid=state==XD1||state==XD3;d_monitor_addr=6'(transform_addr+((state==XD3)?10:0));
  for(integer i=0;i<8;i=i+1)d_monitor_data[i*32+:32]={add_y[i][15:0],acc[i][15:0]};
  result_valid=state==DRAIN_SEND;''')
s=sub(s,"if(b==0)assign cin=1'b0;","if(b==0)assign cin=sub_alu;")
s=sub(s,'lhs[l][b]^rhs[l][b]^cin','lhs[l][b]^(rhs[l][b]^sub_alu)^cin')
s=sub(s,'(lhs[l][b]&rhs[l][b])|((lhs[l][b]^rhs[l][b])&cin)','(lhs[l][b]&(rhs[l][b]^sub_alu))|((lhs[l][b]^(rhs[l][b]^sub_alu))&cin)')
s=sub(s,'   state<=IDLE;stripe<=0;', '   state<=IDLE;stripe<=0;mode_q<=0;tx<=0;')
s=sub(s,'   for(integer i=0;i<8;i=i+1)begin src_masks[i]<=0;', '   for(integer i=0;i<8;i=i+1)begin transform_tail[i]<=0;for(integer m=0;m<3;m=m+1)m_aux[m][i]<=0;src_masks[i]<=0;')
s=sub(s,'IDLE:if(start)begin stripe<=0;','IDLE:if(start)begin mode_q<=mode;stripe<=0;')
s=sub(s,'if(zrow==39)begin og<=0;qfill<=0;state<=VLOAD;end else zrow<=zrow+1;', 'if(zrow==39)begin og<=0;qfill<=0;if(mode_q)begin tx<=0;rank_live<=0;state<=XDREAD0;end else state<=VLOAD;end else zrow<=zrow+1;')
s=sub(s,'if(qfill==23)begin fp<=0;state<=POSLOAD;end','if(qfill==(mode_q?31:23))begin fp<=0;state<=POSLOAD;end')
s=sub(s,"     for(integer i=0;i<8;i=i+1)acc[i]<=stripe?$signed(p_read_data[i]):32'sd0;\n     if(stripe)psum_reads<=psum_reads+1;\n     remaining<=next_support;state<=(next_support==0)?STORE:BASE_MAC;", """     for(integer i=0;i<8;i=i+1)begin
      acc[i]<=(stripe&&!mode_q)?$signed(p_read_data[i]):32'sd0;
      if(mode_q)for(integer m=0;m<3;m=m+1)m_aux[m][i]<=0;
     end
     if(stripe&&!mode_q)psum_reads<=psum_reads+1;
     remaining<=next_support;state<=(next_support==0)?(mode_q?W_INV0A:STORE):BASE_MAC;""")
s=sub(s,'     for(integer i=0;i<8;i=i+1)acc[i]<=add_y[i];\n     q2_issues<=q2_issues+1;z_scalar_reads<=z_scalar_reads+1;remaining<=remaining&(remaining-24\'d1);\n     if((remaining&(remaining-24\'d1))==0)state<=STORE;', '''     for(integer i=0;i<8;i=i+1)begin
      if(mode_q&&selected_term%4!=0)m_aux[selected_term%4-1][i]<=add_y[i];else acc[i]<=add_y[i];
     end
     q2_issues<=q2_issues+1;cache_reads<=cache_reads+1;z_scalar_reads<=z_scalar_reads+1;remaining<=remaining&(remaining-32'd1);
     if((remaining&(remaining-32'd1))==0)state<=mode_q?W_INV0A:STORE;''')
newstates='''
    XDREAD0:begin
     for(integer i=0;i<8;i=i+1)z_hold[i]<=z_read_word[i];
     transform_reads<=transform_reads+1;z_vector_reads<=z_vector_reads+1;state<=XDREAD1;
    end
    XDREAD1:begin
     for(integer i=0;i<8;i=i+1)transform_tail[i]<=z_read_word[i];
     transform_reads<=transform_reads+1;z_vector_reads<=z_vector_reads+1;state<=XD0;
    end
    XD0,XD1,XD2,XD3:begin
     transform_issues<=transform_issues+1;rank_live<=rank_live|transform_support;
     if(state==XD0||state==XD2)begin
      for(integer i=0;i<8;i=i+1)acc[i]<=add_y[i];
      position_live[(tx/10)*40+((state==XD2)?20:0)+tx%10]<=transform_support;
      state<=state==XD0?XD1:XD3;
     end else begin
      for(integer i=0;i<8;i=i+1)z_mem[i][transform_addr+((state==XD3)?10:0)]<={add_y[i][15:0],acc[i][15:0]};
      transform_writes<=transform_writes+1;z_writes<=z_writes+1;
      position_live[(tx/10)*40+((state==XD3)?30:10)+tx%10]<=transform_support;
      if(state==XD1)state<=XD2;
      else if(tx==19)begin og<=0;qfill<=0;state<=VLOAD;end
      else begin tx<=tx+1;state<=XDREAD0;end
     end
    end
    W_INV0A,W_INV1A:begin
     for(integer i=0;i<8;i=i+1)acc[i]<=add_y[i];
     reconstruction_issues<=reconstruction_issues+1;state<=(state==W_INV0A)?W_INV0B:W_INV1B;
    end
    W_INV0B,W_INV1B:begin
     for(integer i=0;i<8;i=i+1)acc[i]<=$signed(add_y[i])>>>1;
     reconstruction_issues<=reconstruction_issues+1;exact_halves<=exact_halves+1;
     if(state==W_INV0B)state<=stripe?W_PREAD0:W_STORE0;else state<=stripe?W_PREAD1:W_STORE1;
    end
    W_PREAD0,W_PREAD1:begin
     for(integer i=0;i<8;i=i+1)z_hold[i]<=p_read_data[i];
     psum_reads<=psum_reads+1;state<=(state==W_PREAD0)?W_PADD0:W_PADD1;
    end
    W_PADD0,W_PADD1:begin
     for(integer i=0;i<8;i=i+1)acc[i]<=add_y[i];
     stripe_add_issues<=stripe_add_issues+1;state<=(state==W_PADD0)?W_STORE0:W_STORE1;
    end
    W_STORE0:begin psum_writes<=psum_writes+1;state<=W_INV1A;end
    W_STORE1:begin
     psum_writes<=psum_writes+1;
     if(fp<19)begin fp<=fp+1;state<=POSLOAD;end
     else if(og<11)begin og<=og+1;qfill<=0;state<=VLOAD;end
     else if(!stripe)begin stripe<=1;zrow<=0;rank_live<=0;state<=ZCLEAR;end
     else begin row<=0;state<=DRAIN_READ;end
    end
'''
s=sub(s,'    DRAIN_READ:begin',newstates+'    DRAIN_READ:begin')
s=sub(s,' integer checked_stores;', ' integer checked_stores;logic [479:0] checked_first,checked_second;')
s=sub(s,'if(!reset_n)checked_stores<=0;',"if(!reset_n)begin checked_stores<=0;checked_first<=0;checked_second<=0;end")
s=sub(s,'if(state==IDLE&&start)checked_stores<=0;',"if(state==IDLE&&start)begin checked_stores<=0;checked_first<=0;checked_second<=0;end")
s=sub(s,'   if(state==STORE)begin\n    if(int\'(stripe)*480+og*40+fp!=checked_stores)$fatal(1,"psum ordering");\n    checked_stores<=checked_stores+1;\n   end', '''   if(state==STORE||state==W_STORE0||state==W_STORE1)begin
    if(!mode_q&&int'(stripe)*480+og*40+fp!=checked_stores)$fatal(1,"psum ordering");
    if(stripe)begin
     if(!checked_first[p_addr]||checked_second[p_addr])$fatal(1,"stripe sum ownership");
     checked_second[p_addr]<=1;
    end else begin
     if(checked_first[p_addr])$fatal(1,"first stripe duplicate write");
     checked_first[p_addr]<=1;
    end
    checked_stores<=checked_stores+1;
   end
   if((state==W_PREAD0||state==W_PREAD1)&&!checked_first[p_addr])$fatal(1,"read before first stripe write");
   if(state==W_INV0B||state==W_INV1B)for(integer i=0;i<8;i=i+1)
    if(add_y[i][0])$fatal(1,"non-exact Winograd half");
   if(state==XD0||state==XD1||state==XD2||state==XD3)for(integer i=0;i<8;i=i+1)
    if(add_y[i]<-32768||add_y[i]>32767)$fatal(1,"D16 overflow");
   if(state==BASE_MAC||state==W_INV0A||state==W_INV0B||state==W_INV1A||state==W_INV1B||state==W_PADD0||state==W_PADD1)
    for(integer i=0;i<8;i=i+1)begin
     if(!sub_alu&&(64'($signed(lhs[i]))+64'($signed(rhs[i]))!=64'($signed(add_y[i]))))$fatal(1,"add prefix overflow");
     if(sub_alu&&(64'($signed(lhs[i]))-64'($signed(rhs[i]))!=64'($signed(add_y[i]))))$fatal(1,"subtract prefix overflow");
    end''')
s=sub(s,'if(state==DRAIN_READ&&checked_stores!=960)', 'if(state==DRAIN_READ)for(integer j=0;j<480;j=j+1)if(!checked_second[j])$fatal(1,"missing second stripe row %0d",j);\n   if(state==DRAIN_READ&&checked_stores!=960)')
s=sub(s,'$fatal(1,"drain before both stripes");','$fatal(1,"drain before both stripes mode=%0d stores=%0d mask=%h",mode_q,checked_stores,checked_second);')
s=sub(s,'if((state==QREAD||state==VLOAD)&&(weight_addr<0||weight_addr>=576))','if((state==QREAD&&(weight_addr<0||weight_addr>=576))||(state==VLOAD&&(weight_addr<0||weight_addr>=(mode_q?768:576))))')
(H/'spatial_core.sv').write_text(s)
# Use the original native/gold emitter with the frozen q11 function, not q13 oracles.
p=(OLD/'prepare.py').read_text().replace("F=H.parent/'spatial_r16_integer'","F=H.parent/'spatial_winograd_inputs'")
p=p.replace("hexfile(H/'parameters/q1.hex',q1vec);", "hexfile(H/'parameters/wq2.hex',f['winograd_q2'].reshape(12,8,2,8,4).transpose(2,0,3,4,1).reshape(768,8));\nhexfile(H/'parameters/q1.hex',q1vec);")
p=p.replace(" hexfile(d/'z.hex',packed)", """ hexfile(d/'z.hex',packed)
 dv=np.stack([z[:,:,:,0]-z[:,:,:,2],z[:,:,:,1]+z[:,:,:,2],z[:,:,:,2]-z[:,:,:,1],z[:,:,:,1]-z[:,:,:,3]],axis=3)
 dp=np.zeros((2,40,8),np.uint32)
 for ss in range(2):
  for a in range(40):
   yy,xx=divmod(a//10,2);t=a%10
   dp[ss,a]=(dv[t,ss*8:ss*8+8,yy,xx*2]&65535)|((dv[t,ss*8:ss*8+8,yy,xx*2+1]&65535)<<16)
 hexfile(d/'d.hex',dp)""")
p=p.replace("profiles[str(d)]=profile(ev,z,origin);records.append(str(d))", """profiles[str(d)]=profile(ev,z,origin)
 wm=wv=0
 for ss in range(2):
  dd=dv[:,ss*8:ss*8+8];rl=np.any(dd!=0,axis=(0,2,3))
  for og in range(12):
   live=np.any(f['winograd_q2'][og*8:og*8+8,ss*8:ss*8+8]!=0,axis=0)&rl[:,None]
   wv+=int(live.sum());wm+=int(((dd!=0)&live[None,:,None,:]).sum())
 profiles[str(d)]['winograd_mac']=wm;profiles[str(d)]['winograd_weight']=wv
 records.append(str(d))""")
p+= "\n(H/'short.txt').write_text('\\n'.join(str(H/'fixtures'/f'tile_{t}') for t in [159,160,161,19199,19040])+'\\n')\n"
(H/'prepare.py').write_text(p)
t=(OLD/'tb.cpp').read_text().replace('if(argc!=4)return 2;','if(argc!=5)return 2;').replace('const std::string root=argv[1];int stall=std::stoi(argv[3]);','const std::string root=argv[1];int stall=std::stoi(argv[3]),mode=std::stoi(argv[4]);')
t=t.replace('root+"/parameters/q2.hex"','root+(mode?"/parameters/wq2.hex":"/parameters/q2.hex")').replace('q2.size()!=4608','q2.size()!=unsigned(mode?6144:4608)')
t=t.replace('d.reset_n=0;d.start=0;','d.mode=mode;d.reset_n=0;d.start=0;')
t=t.replace('for(unsigned a=0;a<576;a++){for(int i=0;i<8;i++)data[i]=q2','for(unsigned a=0;a<unsigned(mode?768:576);a++){for(int i=0;i<8;i++)data[i]=q2')
t=t.replace('zgold=readhex(dir+"/z.hex");','zgold=readhex(dir+"/z.hex"),dgold=readhex(dir+"/d.hex");')
t=t.replace('unsigned outputs=0,zoutputs=0;','unsigned outputs=0,zoutputs=0,doutputs=0;bool dseen[80]={};')
t=t.replace('   if(d.result_valid&&d.result_ready){', '''   if(d.d_monitor_valid){
    unsigned a=unsigned(d.z_monitor_stripe)*40+d.d_monitor_addr;if(a>=80||dseen[a])return 30;dseen[a]=true;
    for(int i=0;i<8;i++)if(d.d_monitor_data[i]!=dgold[a*8+i]){std::cerr<<"D mismatch "<<dir<<" row="<<a<<" lane="<<i<<" got="<<d.d_monitor_data[i]<<" exp="<<dgold[a*8+i]<<"\\n";return 31;}
    doutputs++;
   }
   if(d.result_valid&&d.result_ready){''')
t=t.replace('if(outputs!=480||zoutputs!=80)return 12;','if(outputs!=480||zoutputs!=80||doutputs!=unsigned(mode?80:0))return 12;')
t=t.replace('    field("source_words",d.source_words);','    field("mode",mode);field("d_values",doutputs*16);'+''.join(f'field("{x}",d.{x});' for x in EXTRA)+'\n    field("source_words",d.source_words);')
(H/'tb.cpp').write_text(t)
# Reuse the already measured actual FP32 identity/J20/wide/I24 consumer wrapper.
c=(OLD/'implement_consumer.py').read_text()
c=c.replace("consumer=['cycles'",'base += '+repr(EXTRA)+"\nconsumer=['cycles'")
c=c.replace(' input logic clk,reset_n,cfg_valid,start,',' input logic clk,reset_n,cfg_valid,start,mode,')
c=c.replace(' output logic [5:0] debug_state,',' output logic d_monitor_valid,output logic [5:0] d_monitor_addr,output logic [255:0] d_monitor_data,\n output logic [5:0] debug_state,')
c=c.replace('.cfg_valid(cfg_valid&&cfg_kind!=6),.start(start),','.cfg_valid(cfg_valid&&cfg_kind!=6),.start(start),.mode(mode),')
c=c.replace(' .debug_state(debug_state),.z_monitor_valid', ' .d_monitor_valid(d_monitor_valid),.d_monitor_addr(d_monitor_addr),.d_monitor_data(d_monitor_data),\n .debug_state(debug_state),.z_monitor_valid')
c=c.replace("H.parent/'spatial_r16_integer/factors.npz'","H.parent/'spatial_winograd_inputs/factors.npz'")
(H/'implement_consumer.py').write_text(c)
t=(OLD/'stream_tb.cpp').read_text()
t=t.replace('if(argc!=4)return 2;','if(argc!=5)return 2;').replace('int stall=std::stoi(argv[3]);','int stall=std::stoi(argv[3]),mode=std::stoi(argv[4]);')
t=t.replace('root+"/parameters/q2.hex"','root+(mode?"/parameters/wq2.hex":"/parameters/q2.hex")').replace('q2.size()!=4608','q2.size()!=unsigned(mode?6144:4608)')
t=t.replace('d.reset_n=0;d.start=0;','d.mode=mode;d.reset_n=0;d.start=0;')
t=t.replace('for(unsigned a=0;a<576;a++){for(int i=0;i<8;i++)data[i]=q2','for(unsigned a=0;a<unsigned(mode?768:576);a++){for(int i=0;i<8;i++)data[i]=q2')
t=t.replace('zgold=readhex(dir+"/z.hex");','zgold=readhex(dir+"/z.hex"),dgold=readhex(dir+"/d.hex");')
t=t.replace('unsigned outputs=0,rawoutputs=0,','unsigned doutputs=0;bool dseen[80]={};\n  unsigned outputs=0,rawoutputs=0,')
t=t.replace('   if(d.raw_monitor_valid){', '''   if(d.d_monitor_valid){unsigned a=unsigned(d.z_monitor_stripe)*40+d.d_monitor_addr;if(a>=80||dseen[a])return 32;dseen[a]=true;for(int i=0;i<8;i++)if(d.d_monitor_data[i]!=dgold[a*8+i])return 33;doutputs++;}
   if(d.raw_monitor_valid){''')
t=t.replace('if(outputs!=480||rawoutputs!=480','if(doutputs!=unsigned(mode?80:0)||outputs!=480||rawoutputs!=480')
t=t.replace('    F(cycles);','    field("mode",mode);field("d_values",doutputs*16);'+''.join(f'F({x});' for x in EXTRA)+'\n    F(cycles);')
(H/'stream_tb.cpp').write_text(t)
print('Spatial mode0 ordinary / mode1 F(2,3) generated; one ALU/multiplier bank')
