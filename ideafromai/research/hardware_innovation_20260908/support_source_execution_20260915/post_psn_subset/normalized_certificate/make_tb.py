from pathlib import Path
P=Path(__file__).resolve().parent

def normalized_check(v,c,A,p,hg,m,Y,tau,pos):
 return f'''      for(int i=0;i<80;i++){{
       int t=i/8,h={hg}*8+i%8;int64_t ns=0,ps=0,prefix=0;
       for(int s=0;s<10;s++){{
        ns+=std::min(int64_t({A}[t*10+s]),int64_t(0));ps+=std::max(int64_t({A}[t*10+s]),int64_t(0));
        prefix+=int64_t({A}[t*10+s])*floor_shift(int64_t({Y}[({p}*10+s)*96+h]),{m});
       }}
       int64_t threshold={tau}[t*96+h],delta={pos}[h]?1:0;
       int64_t qn=threshold+ns-delta,qp=threshold+ps-delta;
       int64_t sn=floor_shift(qn,{m}),sp=floor_shift(qp,{m});
       int64_t n=prefix+ns,pp=prefix+ps;
       require(signedbits({v}.mon_norm_n,i*49,49)==n&&signedbits({v}.mon_norm_p,i*49,49)==pp,"normalized actual sums");ncheck+=2;
       require(signedbits({v}.mon_q_n,i*49,49)==qn&&signedbits({v}.mon_q_p,i*49,49)==qp,"normalized registered q49");
       require(signedbits({v}.mon_shift_n,i*49,49)==sn&&signedbits({v}.mon_shift_p,i*49,49)==sp,"normalized signed floor threshold");qcheck+=4;
       int64_t scale=int64_t(1)<<{m};
       int64_t lo=prefix*scale+ns*(scale-1),hi=prefix*scale+ps*(scale-1);
       require((n>sn)==({pos}[h]?lo>=threshold:lo>threshold),"lower predicate equivalence");
       require((pp<=sp)==({pos}[h]?hi<threshold:hi<=threshold),"upper predicate equivalence");predcheck+=2;
      }}
'''
helpers='''int64_t floor_shift(int64_t x,int m){return x>=0?(x>>m):-(((-x)+(int64_t(1)<<m)-1)>>m);}
'''
s=(P.parent/'joined_fc1/tb.cpp').read_text().replace('Vjoined_fc1','Vnormalized_fc1').replace('joined_fc1__DOT__','normalized_fc1__DOT__')
s=s.replace('void run(',helpers+'void run(',1)
s=s.replace('bcheck','ncheck').replace('tailcheck','qcheck').replace('evencheck','predcheck')
start=s.index('  if(v.mon_bounds){');end=s.index('  if(outhold)',start)
block=normalized_check('v','c','A','v.mon_p','v.mon_hgroup','v.mon_m','c.Y','c.tau','c.positive').replace('require(', 'need(')
s=s[:start]+'  if(v.mon_bounds){\n'+block+'  }\n'+s[end:]
s=s.replace('bound_checks','normalized_value_checks').replace('tail_checks','threshold_checks').replace('tail_even_checks','normalized_predicate_checks')
(P/'tb.cpp').write_text(s)
s=(P.parent/'fused_datapath/tb.cpp').read_text().replace('Vsubset_psn','Vnormalized_leaf').replace('subset_psn__DOT__','normalized_leaf__DOT__')
s=s.replace('int main(',helpers+'int main(',1)
s=s.replace('uint64_t ycheck=0,ucheck=0,gcheck=0,bcheck=0,echeck=0;', 'uint64_t ycheck=0,ucheck=0,gcheck=0,ncheck=0,qcheck=0,predcheck=0,echeck=0;')
start=s.index('    if(d.mon_lower && d.mon_upper){');end=s.index('    if(d.out_valid && d.out_ready)',start)
s=s[:start]+'    if(d.mon_lower && d.mon_upper){\n'+normalized_check('d','c','a','d.out_p','d.out_hgroup','d.dbg_m','c.y','c.tau','c.positive')+'    }\n'+s[end:]
s=s.replace('bcheck','ncheck').replace('bound_checks','normalized_value_checks')
s=s.replace('total_bounds+=ncheck;', 'total_bounds+=ncheck;')
needle='<<",\\"exponent_checks\\":"<<echeck'
s=s.replace(needle,'<<",\\"threshold_checks\\":"<<qcheck<<",\\"normalized_predicate_checks\\":"<<predcheck'+needle)
(P/'leaf_tb.cpp').write_text(s)
print('generated joined + Y-leaf TB with actual normalized49 checks, no old-bound monitor')
