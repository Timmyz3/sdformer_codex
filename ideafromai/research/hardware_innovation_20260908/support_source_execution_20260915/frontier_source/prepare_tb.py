from pathlib import Path
p=Path(__file__).resolve().parent
s=(p.parent/'tb_source.cpp').read_text().replace('Vsource_classifier','Vfrontier_source')
s=s.replace('int prefetch,const array<int16_t,100>', 'int prefetch,int frontier,const array<int16_t,100>')
s=s.replace('v.start_prefetch=prefetch;', 'v.start_prefetch=prefetch;v.start_frontier=frontier;')
s=s.replace('array<bool,96>produced{};', 'array<bool,960>produced{};')
s=s.replace('int holdgroup=0,products=0,outs=0,', 'int holdgroup=0,products=0,pairs=0,channels=0,outs=0,')
start=s.index('  if(v.producer_valid){');end=s.index('  if(outhold)',start)
s=s[:start]+'''  if(v.producer_valid){
    array<bool,96>batch_seen{};products++;int active=v.producer_active;need(active!=0,"empty producer batch");
    if(!frontier)need(active==1023,"legacy fullT10 provider mask");
    for(int t=0;t<10;t++)if(active>>t&1){int ch=v.producer_channel[t];need(ch<96&&!produced[ch*10+t],"duplicate (t,c)");produced[ch*10+t]=true;pairs++;batch_seen[ch]=true;
      need(((v.producer_gate>>t)&1)==((gate[ch]>>t)&1),"producer valid gate "+c.name);
      need(sg(v.producer_u,t*48,48)==u[ch*10+t],"producer valid integer U "+c.name);
    }
    int bc=count(batch_seen.begin(),batch_seen.end(),true);need(bc<=4,"batch width");channels+=bc;
  }
'''+s[end:]
s=s.replace('need(products==int(v.count_channels)&&int(v.count_mac)==products*100,"producer accounting");need(xw==products*2', 'need(products==int(v.count_batches)&&channels==int(v.count_channels)&&pairs==int(v.count_pairs)&&int(v.count_mac)==pairs*10,"producer accounting");need(xw==channels*2')
s=s.replace('if(mode==0)need(products==96', 'if(mode==0)need(channels==96').replace('if(mode==1)need(products==64','if(mode==1)need(channels==64')
s=s.replace("<<prefetch<<','<<cyc<<','<<products<<','", "<<prefetch<<','<<frontier<<','<<cyc<<','<<channels<<','<<products<<','<<pairs<<','")
s=s.replace('ifstream f("source.bin"', 'ifstream f("../source.bin"')
s=s.replace('ofstream csv("source_cycles.csv");', 'ofstream csv(argc>2?argv[2]:"frontier_cycles.csv");')
s=s.replace('prefetch,cycles,channels,scalar_mac', 'prefetch,frontier,cycles,channels,batches,pairs,scalar_mac')
start=s.index('  for(int o=0;o<2;o++)');end=s.index('  cout<<"PASS "',start)
s=s[:start]+'''  for(int m=1;m<4;m++)for(int b=0;b<2;b++)for(int frontier=0;frontier<2;frontier++){
    if(m==1&&frontier)continue;
    run(c,profiles[1],1,m,b,1,1,frontier,A,tau,D,csv);
  }
'''+s[end:]
(p/'tb.cpp').write_text(s)
