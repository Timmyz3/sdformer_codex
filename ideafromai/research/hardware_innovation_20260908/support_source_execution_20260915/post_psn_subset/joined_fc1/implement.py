from pathlib import Path
P=Path(__file__).resolve().parent
s=(P.parents[2]/'support_lut_execution_20260915/support_fc1.sv').read_text()
def sub(a,b):
 global s
 assert a in s,a[:100]
 s=s.replace(a,b)
sub('module support_fc1(', 'module joined_fc1(')
sub('input logic start_valid,output logic start_ready,input logic[2:0] start_mode,input logic[1:0] start_hblock,','input logic start_valid,output logic start_ready,input logic[1:0] start_mode,input logic[1:0] start_hblock,input logic start_reuse_config,')
sub('output logic[3:0] dbg_state','''output logic[4:0] dbg_state,
 output logic[31:0] dbg_planes,dbg_early,dbg_table_writes,dbg_pack_writes,dbg_pack_reads,
 output logic mon_group_valid,mon_full_u_valid,mon_y_read,mon_bounds,
 output logic[4:0] mon_p,mon_m,mon_exponent,output logic[3:0] mon_hgroup,
 output logic[3839:0] mon_group_u,mon_bound_lo,mon_bound_hi,
 output logic[959:0] mon_tail,mon_tail_delta,
 output logic[8:0] mon_y_addr,output logic[2303:0] mon_y_data,
 output logic mon_table_valid,output logic[5:0] mon_table_addr,output logic[159:0] mon_table_data''')
sub('typedef enum logic[3:0]{IDLE,BREQ,BRSP,SOURCE,FC,PCLEAR,PREAD,PMAC,PDRAIN,POUT,DONE}', 'typedef enum logic[4:0]{IDLE,BREQ,BRSP,SOURCE,FC,PCLEAR,PREAD,PMAC,PDRAIN,POUT,DONE,FTABLE,FPNINIT,FPNPOS,FPNNEG,FPINIT,FYREAD,FGROUP,FPLANE,FSTORE,FOUT}')
sub(' logic[2:0] mode;logic[1:0] hb;', ''' logic[1:0] psn_mode;logic config_valid,lut_valid;logic cert,y_read_enable;
 logic[2:0] mode;logic[1:0] hb;''')
sub(' logic signed[47:0] add_a[96],add_b[96],add_y[96];',''' logic signed[47:0] add_a[96],add_b[96],add_y[96];logic[95:0] first_sub;
 // One shared model; the fused path adds only LUT/holding, never duplicate A/tau/Y.
 logic signed[15:0] lut[2][32][10];
 logic signed[23:0] ybuf[10][96];
 logic[4:0] exponents[12],exp_next[12],m,group_m,code;logic half;
 logic[3:0] hg,col;
 logic signed[47:0] pos[10],neg[10],tail_pos[10],tail_neg[10],v[80];
 logic[79:0] locked,gate,locked_next,gate_next;
 logic[95:0] gatepack[10];
 logic[4:0] lo_code[8],hi_code[8];logic signed[15:0] lookup_lo[80],lookup_hi[80];
 logic signed[47:0] first_a[80],first_b[80],second_b[80],nv[80];logic[79:0] fsub,second_sub;
 logic signed[47:0] bound_a[80],bound_lo[80],bound_hi[80],tail_a[20],tail_b[20],tail_delta[20];
 logic[79:0] lower_hit,upper_hit,lo_less,lo_equal,hi_less,hi_equal;
 integer lsb,previous;
 function automatic logic[4:0] magnitude_bits(input logic signed[23:0] x);
  logic[4:0] e;logic signed[24:0] xx,lim;
  begin e=0;xx=25'(x);for(integer b=0;b<24;b++)begin lim=25'sd1<<<b;if(xx>=lim||xx<=-lim)e=5'(b+1);end return e;end
 endfunction''')
# The one row read mux is shared with the actual FC1 producer, native PSN and fused PSN.
sub('   y_read_addr=st==FC?9\'(active_row):(st==PREAD?9\'(int\'(p)*10+int\'(s)):9\'(int\'(p)*10+int\'(output_t)));', '''   cert=psn_mode==2;
   y_read_enable=(st==FC&&fc_issue)||st==PREAD||st==POUT||st==FYREAD||st==FOUT;
   y_read_addr=st==FC?9'(active_row):((st==PREAD||st==FYREAD)?9'(int'(p)*10+int'(s)):9'(int'(p)*10+int'(output_t)));''')
sub("y_read_data[h]=yvalid[h/12][y_read_addr]?y[y_read_addr][h]:24'sd0;", "y_read_data[h]=y_read_enable&&yvalid[h/12][y_read_addr]?y[y_read_addr][h]:24'sd0;")
# Calculation of the second prefix/bounds uses the common add_y output below;
# all equations are combinational with no additional register boundary.
sub('   for(integer h=0;h<96;h++)begin\n     y_read_data', '''   for(integer h=0;h<96;h++)begin
     y_read_data''')
sub('     add_a[h]=0;add_b[h]=0;', '     add_a[h]=0;add_b[h]=0;first_sub[h]=0;')
sub('     add_y[h]=add_a[h]+add_b[h];', '''     if(h<80 && (st==FTABLE||st==FPNPOS||st==FPNNEG||st==FGROUP||st==FPLANE))begin
       add_a[h]=first_a[h];add_b[h]=first_b[h];first_sub[h]=fsub[h];
     end
     add_y[h]=add_a[h]+(add_b[h]^{48{first_sub[h]}})+48'(first_sub[h]);''')
# Insert fused operands before common first ALU. Y exponent update is put after its read evaluation.
needle='   u_read_addr=pv[3]?pt[3]:output_t;'
pre='''
   group_m=exponents[hg]==0?5'd0:exponents[hg]-1'b1;
   lsb=0;for(integer b=4;b>=0;b--)if(code[b])lsb=b;
   previous=int'(code)&~(1<<lsb);
   for(integer j=0;j<8;j++)begin
    lo_code[j]=0;hi_code[j]=0;
    for(integer k=0;k<5;k++)begin
     lo_code[j][k]=(st==FGROUP)?ybuf[k][int'(hg)*8+j][23]:ybuf[k][int'(hg)*8+j][m];
     hi_code[j][k]=(st==FGROUP)?ybuf[5+k][int'(hg)*8+j][23]:ybuf[5+k][int'(hg)*8+j][m];
    end
   end
   for(integer i=0;i<80;i++)begin
    lookup_lo[i]=0;lookup_hi[i]=0;
    if(st==FTABLE||st==FGROUP||st==FPLANE)begin
     lookup_lo[i]=lut[0][st==FTABLE?5'(previous):lo_code[i%8]][i/8];
     lookup_hi[i]=lut[1][st==FTABLE?5'(previous):hi_code[i%8]][i/8];
    end
   end
   fsub=0;second_sub=0;
   for(integer i=0;i<80;i++)begin
    first_a[i]=0;first_b[i]=0;second_b[i]=0;
    if(st==FTABLE&&i<10&&code!=0)begin
     first_a[i]=half?48'($signed(lookup_hi[i*8])):48'($signed(lookup_lo[i*8]));
     first_b[i]=48'($signed(a[i][int'(half)*5+lsb]));
    end else if(st==FPNPOS&&i<10)begin first_a[i]=pos[i];first_b[i]=a[i][col]>0?48'($signed(a[i][col])):48'sd0;end
    else if(st==FPNNEG&&i<10)begin first_a[i]=neg[i];first_b[i]=a[i][col]<0?48'($signed(a[i][col])):48'sd0;end
    else if(st==FGROUP)begin first_b[i]=48'($signed(lookup_lo[i]));fsub[i]=1;second_b[i]=48'($signed(lookup_hi[i]));second_sub[i]=1;end
    else if(st==FPLANE)begin first_a[i]=v[i]<<<1;first_b[i]=48'($signed(lookup_lo[i]));second_b[i]=48'($signed(lookup_hi[i]));end
   end
'''
sub(needle,pre+needle)
needle='   out_valid=st==POUT;'
post='''
   for(integer g=0;g<12;g++)begin
    exp_next[g]=exponents[g];for(integer j=0;j<8;j++)if(magnitude_bits(y_read_data[g*8+j])>exp_next[g])exp_next[g]=magnitude_bits(y_read_data[g*8+j]);
   end
   mon_bounds=cert&&st==FPLANE;mon_group_valid=st==FSTORE;mon_full_u_valid=psn_mode==1&&st==FSTORE;
   mon_p=p;mon_hgroup=hg;mon_m=m;mon_exponent=exponents[hg];mon_group_u=0;mon_bound_lo=0;mon_bound_hi=0;mon_tail=0;mon_tail_delta=0;
   mon_y_read=y_read_enable;mon_y_addr=y_read_addr;mon_y_data=0;
   for(integer h=0;h<96;h++)mon_y_data[h*24+:24]=y_read_data[h];
   mon_table_valid=st==FTABLE;mon_table_addr={half,code};mon_table_data=0;
   for(integer t=0;t<10;t++)begin
    tail_a[2*t]=0;tail_b[2*t]=0;tail_a[2*t+1]=0;tail_b[2*t+1]=0;
    if(cert&&st==FGROUP)begin
     tail_a[2*t]=pos[t]<<<group_m;tail_b[2*t]=pos[t];tail_a[2*t+1]=neg[t]<<<group_m;tail_b[2*t+1]=neg[t];
    end else if(cert&&st==FPLANE)begin
     tail_a[2*t]=tail_pos[t];tail_b[2*t]=pos[t];tail_a[2*t+1]=tail_neg[t];tail_b[2*t+1]=neg[t];
    end
    tail_delta[2*t]=tail_a[2*t]-tail_b[2*t];tail_delta[2*t+1]=tail_a[2*t+1]-tail_b[2*t+1];
    mon_tail[(2*t)*48+:48]=tail_pos[t];mon_tail[(2*t+1)*48+:48]=tail_neg[t];
    mon_tail_delta[(2*t)*48+:48]=tail_delta[2*t];mon_tail_delta[(2*t+1)*48+:48]=tail_delta[2*t+1];
    mon_table_data[t*16+:16]=code==0?16'd0:16'(add_y[t]);
   end
   locked_next=locked;gate_next=gate;
   for(integer i=0;i<80;i++)begin
    integer t,h;t=i/8;h=int'(hg)*8+i%8;
    nv[i]=add_y[i]+(second_b[i]^{48{second_sub[i]}})+48'(second_sub[i]);
    bound_a[i]=cert?(nv[i]<<<m):nv[i];
    bound_lo[i]=bound_a[i]+(cert?tail_neg[t]:48'sd0);bound_hi[i]=bound_a[i]+(cert?tail_pos[t]:48'sd0);
    lo_less[i]=bound_lo[i]<tau[t][h];lo_equal[i]=bound_lo[i]==tau[t][h];hi_less[i]=bound_hi[i]<tau[t][h];hi_equal[i]=bound_hi[i]==tau[t][h];
    lower_hit[i]=positive[h]?!lo_less[i]:(!lo_less[i]&&!lo_equal[i]);upper_hit[i]=positive[h]?hi_less[i]:(hi_less[i]||hi_equal[i]);
    if(!locked[i])begin
     if(lower_hit[i])begin locked_next[i]=1;gate_next[i]=positive[h];end
     else if(upper_hit[i])begin locked_next[i]=1;gate_next[i]=!positive[h];end
    end
    mon_group_u[i*48+:48]=v[i];mon_bound_lo[i*48+:48]=bound_lo[i];mon_bound_hi[i*48+:48]=bound_hi[i];
   end
'''
sub(needle,post+needle)
sub('   out_valid=st==POUT;', '   out_valid=st==POUT||st==FOUT;')
sub("if(st==POUT)out_y[h*24+:24]=y_read_data[h];", "if(st==POUT||st==FOUT)out_y[h*24+:24]=y_read_data[h];")
sub('           (positive[h]?(u_read_data[h]>=tau[output_t][h]):(u_read_data[h]<=tau[output_t][h]));', '           (positive[h]?(u_read_data[h]>=tau[output_t][h]):(u_read_data[h]<=tau[output_t][h]));\n     if(st==FOUT)out_gate[h]=gatepack[output_t][h];')
sub('     st<=IDLE;mode<=0;hb<=0;', '''     st<=IDLE;psn_mode<=0;config_valid<=0;lut_valid<=0;hg<=0;code<=0;half<=0;col<=0;m<=0;locked<=0;gate<=0;
     dbg_planes<=0;dbg_early<=0;dbg_table_writes<=0;dbg_pack_writes<=0;dbg_pack_reads<=0;
     mode<=0;hb<=0;''')
sub('       st<=BREQ;mode<=start_mode;hb<=start_hblock;section<=0;boot_word<=0;source_row<=0;num_lut<=0;', '''       st<=start_reuse_config&&config_valid&&start_hblock==hb?SOURCE:BREQ;
       if(!(start_reuse_config&&config_valid&&start_hblock==hb))begin config_valid<=0;lut_valid<=0;num_lut<=0;end
       psn_mode<=start_mode;mode<=3'd4;hb<=start_hblock;section<=0;boot_word<=0;source_row<=0;
       dbg_planes<=0;dbg_early<=0;dbg_table_writes<=0;dbg_pack_writes<=0;dbg_pack_reads<=0;''')
sub('if(section==5 || section==4 || (section==3 && mode<3))st<=SOURCE;', 'if(section==5 || section==4 || (section==3 && mode<3))begin st<=SOURCE;config_valid<=1;end')
sub('FC:if(jobs_left==0 && qcount==0 && !active && fv==0 && pending==0)begin p<=0;clear_t<=0;st<=PCLEAR;end', '''FC:if(jobs_left==0 && qcount==0 && !active && fv==0 && pending==0)begin
       p<=0;clear_t<=0;
       if(psn_mode==0)st<=PCLEAR;
       else if(lut_valid)st<=FPINIT;
       else begin half<=0;code<=0;st<=FTABLE;end
      end''')
needle='      DONE:if(done_ready)st<=IDLE;'
newstates='''      FTABLE:begin
       for(integer t=0;t<10;t++)lut[half][code][t]<=code==0?16'sd0:16'(add_y[t]);
       dbg_table_writes<=dbg_table_writes+1;
       if(code==31)begin if(half)st<=FPNINIT;else begin half<=1;code<=0;end end else code<=code+1'b1;
      end
      FPNINIT:begin for(integer t=0;t<10;t++)begin pos[t]<=0;neg[t]<=0;end col<=0;st<=FPNPOS;end
      FPNPOS:begin for(integer t=0;t<10;t++)pos[t]<=add_y[t];st<=FPNNEG;end
      FPNNEG:begin
       for(integer t=0;t<10;t++)neg[t]<=add_y[t];
       if(col==9)begin lut_valid<=1;st<=FPINIT;end else begin col<=col+1'b1;st<=FPNPOS;end
      end
      FPINIT:begin for(integer g=0;g<12;g++)exponents[g]<=0;s<=0;st<=FYREAD;end
      FYREAD:begin
       for(integer h=0;h<96;h++)ybuf[s][h]<=y_read_data[h];
       for(integer g=0;g<12;g++)exponents[g]<=exp_next[g];
       if(s==9)begin hg<=0;st<=FGROUP;end else s<=s+1'b1;
      end
      FGROUP:begin
       for(integer i=0;i<80;i++)begin
        integer t,h;t=i/8;h=int'(hg)*8+i%8;
        locked[i]<=constant_ch[h];gate[i]<=constant_ch[h]?constant_gate[t*96+h]:1'b0;v[i]<=nv[i];
       end
       for(integer t=0;t<10;t++)begin tail_pos[t]<=tail_delta[2*t];tail_neg[t]<=tail_delta[2*t+1];end
       m<=group_m;st<=FPLANE;
      end
      FPLANE:begin
       for(integer i=0;i<80;i++)v[i]<=nv[i];dbg_planes<=dbg_planes+1;
       if(cert||m==0)begin locked<=locked_next;gate<=gate_next;end
       if(cert&&m!=0)for(integer t=0;t<10;t++)begin tail_pos[t]<=$signed(tail_delta[2*t])>>>1;tail_neg[t]<=$signed(tail_delta[2*t+1])>>>1;end
       if(m==0)st<=FSTORE;
       else if(cert&&locked_next[31:0]==32'hffffffff&&locked_next[63:32]==32'hffffffff&&locked_next[79:64]==16'hffff)begin dbg_early<=dbg_early+1;st<=FSTORE;end
       else m<=m-1'b1;
      end
      FSTORE:begin
       // Ten 96bit gate rows, each receives exactly the selected eight h bits.
       // No consumer can read until all 12 group writes for this P are complete.
       for(integer t=0;t<10;t++)gatepack[t][int'(hg)*8+:8]<=gate[t*8+:8];
       dbg_pack_writes<=dbg_pack_writes+1;
       if(hg==11)begin output_t<=0;st<=FOUT;end else begin hg<=hg+1'b1;st<=FGROUP;end
      end
      FOUT:if(out_ready)begin
       dbg_pack_reads<=dbg_pack_reads+1;
       if(output_t==9)begin if(p==31)st<=DONE;else begin p<=p+1'b1;st<=FPINIT;end end
       else output_t<=output_t+1'b1;
      end
'''
sub(needle,newstates+needle)
(P/'joined_fc1.sv').write_text(s)
print('generated joined_fc1.sv')
