from pathlib import Path
p=Path(__file__).resolve().parent
s=(p.parent/'subset_psn.sv').read_text()
s=s.replace('output logic mon_lower,mon_upper,output logic[3839:0] mon_bound,','output logic mon_lower,mon_upper,output logic[3839:0] mon_bound,mon_bound_hi,\n output logic[959:0] mon_tail,mon_tail_delta,')
s=s.replace('logic signed[47:0] v[80],dot_hold[80];','logic signed[47:0] v[80];')
s=s.replace('logic[79:0] locked,gate,lower_hit,lower_gate;', 'logic[79:0] locked,gate;')
s=s.replace('logic signed[47:0] add_a[80],add_b[80],add_y[80];logic[79:0] add_sub;\n logic[79:0] upper_hit,upper_gate,locked_next,gate_next,cmp_less,cmp_equal;', '''// Explicit combinational resource positions, not the parent's 80 time-shared ALUs.
 logic signed[47:0] add_a[80],add_b[80],add_y[80],second_b[80],nv[80];
 logic[79:0] add_sub,second_sub;
 logic signed[47:0] bound_a[80],bound_lo[80],bound_hi[80];
 logic signed[47:0] tail_a[20],tail_b[20],tail_delta[20];
 logic[4:0] group_m;
 logic[79:0] lower_hit,upper_hit,locked_next,gate_next,lo_less,lo_equal,hi_less,hi_equal;''')
s=s.replace('out_u=0;mon_bound=0;mon_y=0;', 'out_u=0;mon_bound=0;mon_bound_hi=0;mon_tail=0;mon_tail_delta=0;mon_y=0;')
s=s.replace('mon_lower=st==LOWER;mon_upper=st==UPPER;', 'mon_lower=st==PLANE_SUM&&cert;mon_upper=st==PLANE_SUM&&cert;\n   group_m=exponents[hgroup]==0?5\'d0:exponents[hgroup]-1\'b1;')
s=s.replace('(st==SIGN_SUM)?', '(st==GROUP)?')
s=s.replace('st==TABLE || st==SIGN_SUM || st==PLANE_SUM', 'st==TABLE || st==GROUP || st==PLANE_SUM')
start=s.index('   upper_hit=0;');end=s.index('\n end\n always_ff',start)
s=s[:start]+'''   for(integer t=0;t<10;t++)begin
     tail_a[2*t]=0;tail_b[2*t]=0;tail_a[2*t+1]=0;tail_b[2*t+1]=0;
     if(cert && st==GROUP)begin
       tail_a[2*t]=pos[t]<<<group_m;tail_b[2*t]=pos[t];
       tail_a[2*t+1]=neg[t]<<<group_m;tail_b[2*t+1]=neg[t];
     end else if(cert && st==PLANE_SUM)begin
       tail_a[2*t]=tail_pos[t];tail_b[2*t]=pos[t];
       tail_a[2*t+1]=tail_neg[t];tail_b[2*t+1]=neg[t];
     end
     // Twenty dedicated tail subtractors, shared across the eight h lanes.
     tail_delta[2*t]=tail_a[2*t]-tail_b[2*t];
     tail_delta[2*t+1]=tail_a[2*t+1]-tail_b[2*t+1];
     mon_tail[(2*t)*48+:48]=tail_pos[t];mon_tail[(2*t+1)*48+:48]=tail_neg[t];
     mon_tail_delta[(2*t)*48+:48]=tail_delta[2*t];mon_tail_delta[(2*t+1)*48+:48]=tail_delta[2*t+1];
   end
   lower_hit=0;upper_hit=0;locked_next=locked;gate_next=gate;add_sub=0;second_sub=0;
   for(integer i=0;i<80;i++)begin
     integer t,h;t=i/8;h=int'(hgroup)*8+i%8;
     add_a[i]=0;add_b[i]=0;second_b[i]=0;
     if(st==TABLE && i<10 && code!=0)begin
       add_a[i]=half?48'($signed(lookup_hi[i*8])):48'($signed(lookup_lo[i*8]));add_b[i]=48'($signed(a[i][int'(half)*5+lsb]));
     end else if(st==PN_POS && i<10)begin add_a[i]=pos[i];add_b[i]=a[i][col]>0?48'($signed(a[i][col])):48'sd0;end
     else if(st==PN_NEG && i<10)begin add_a[i]=neg[i];add_b[i]=a[i][col]<0?48'($signed(a[i][col])):48'sd0;end
     else if(st==GROUP)begin
       add_b[i]=48'($signed(lookup_lo[i]));add_sub[i]=1;
       second_b[i]=48'($signed(lookup_hi[i]));second_sub[i]=1;
     end else if(st==PLANE_SUM)begin
       add_a[i]=v[i]<<<1;add_b[i]=48'($signed(lookup_lo[i]));second_b[i]=48'($signed(lookup_hi[i]));
     end
     // Eighty lanes, two serial 48-bit add/sub positions per lane, one register boundary.
     add_y[i]=add_a[i]+(add_b[i]^{48{add_sub[i]}})+48'(add_sub[i]);
     nv[i]=add_y[i]+(second_b[i]^{48{second_sub[i]}})+48'(second_sub[i]);
     // A further two parallel 48-bit bound adders per lane. They are not time-shared with prefix.
     bound_a[i]=cert?(nv[i]<<<m):nv[i];
     bound_lo[i]=bound_a[i]+(cert?tail_neg[t]:48'sd0);
     bound_hi[i]=bound_a[i]+(cert?tail_pos[t]:48'sd0);
     lo_less[i]=bound_lo[i]<tau[t][h];lo_equal[i]=bound_lo[i]==tau[t][h];
     hi_less[i]=bound_hi[i]<tau[t][h];hi_equal[i]=bound_hi[i]==tau[t][h];
     lower_hit[i]=positive[h]?!lo_less[i]:(!lo_less[i]&&!lo_equal[i]);
     upper_hit[i]=positive[h]?hi_less[i]:(hi_less[i]||hi_equal[i]);
     if(!locked[i])begin
       if(lower_hit[i])begin locked_next[i]=1;gate_next[i]=positive[h];end
       else if(upper_hit[i])begin locked_next[i]=1;gate_next[i]=!positive[h];end
     end
     out_u[i*48+:48]=v[i];mon_bound[i*48+:48]=bound_lo[i];mon_bound_hi[i*48+:48]=bound_hi[i];
     if(i<10)mon_table[i*16+:16]=code==0?16'd0:16'(add_y[i]);
   end
'''+s[end:]
s=s.replace('locked<=0;gate<=0;lower_hit<=0;lower_gate<=0;', 'locked<=0;gate<=0;')
start=s.index('     GROUP:begin');end=s.index('     OUTPUT:if(out_ready)',start)
s=s[:start]+'''     GROUP:begin
       for(integer i=0;i<80;i++)begin
         integer t,h;t=i/8;h=int'(hgroup)*8+i%8;
         locked[i]<=constant_ch[h];gate[i]<=constant_ch[h]?constant_gate[t*96+h]:1'b0;
         v[i]<=nv[i];
       end
       for(integer t=0;t<10;t++)begin tail_pos[t]<=tail_delta[2*t];tail_neg[t]<=tail_delta[2*t+1];end
       m<=group_m;st<=PLANE_SUM;
     end
     PLANE_SUM:begin
       for(integer i=0;i<80;i++)v[i]<=nv[i];dbg_planes<=dbg_planes+1'b1;
       if(cert || m==0)begin locked<=locked_next;gate<=gate_next;end
       if(cert && m!=0)begin
         for(integer t=0;t<10;t++)begin
           tail_pos[t]<=$signed(tail_delta[2*t])>>>1;
           tail_neg[t]<=$signed(tail_delta[2*t+1])>>>1;
         end
       end
       if(m==0)st<=OUTPUT;
       else if(cert && locked_next[31:0]==32'hffffffff && locked_next[63:32]==32'hffffffff && locked_next[79:64]==16'hffff)begin
         dbg_early<=dbg_early+1'b1;st<=OUTPUT;
       end else m<=m-1'b1;
     end
'''+s[end:]
(p/'subset_psn.sv').write_text(s)
s=(p.parent/'tb.cpp').read_text().replace('f("cases.bin",','f(argc>3?argv[3]:"../cases.bin",')
start=s.index('    if(d.mon_lower||d.mon_upper)');end=s.index('    if(d.out_valid && d.out_ready)',start)
s=s[:start]+'''    if(d.mon_lower && d.mon_upper){
      for(int i=0;i<80;i++){
        int h=d.out_hgroup*8+i%8,t=i/8;int64_t ref=c.u[(d.out_p*10+t)*96+h];
        require(signedbits(d.mon_bound,i*48,48)<=ref && signedbits(d.mon_bound_hi,i*48,48)>=ref,"simultaneous certificate bounds "+c.name);bcheck+=2;
      }
      for(int t=0;t<10;t++)for(int sign=0;sign<2;sign++){
        int64_t coefficient=0;for(int s=0;s<10;s++)coefficient+=sign?std::min(int64_t(a[t*10+s]),int64_t(0)):std::max(int64_t(a[t*10+s]),int64_t(0));
        int64_t tail=coefficient*((int64_t(1)<<d.dbg_m)-1),observed=signedbits(d.mon_tail,(2*t+sign)*48,48);
        require(observed==tail,"actual tail recurrence mismatch");
        if(d.dbg_m){int64_t delta=signedbits(d.mon_tail_delta,(2*t+sign)*48,48);require(delta==tail-coefficient && delta%2==0,"tail recurrence even exactness");}
      }
    }
'''+s[end:]
(p/'tb.cpp').write_text(s)
