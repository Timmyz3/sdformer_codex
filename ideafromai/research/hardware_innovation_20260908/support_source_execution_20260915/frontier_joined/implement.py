"""Create isolated integration from the reviewed joined-chain harness."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'joined_chain'
s=(OLD/'joined_core.sv').read_text()
s=s.replace('input logic[1:0] start_mode,','input logic start_frontier,input logic[1:0] start_mode,')
s=s.replace('output logic[6:0] producer_channel,','output logic[6:0] producer_channel[10],output logic[9:0] producer_active,')
s=s.replace('state_t st;logic[1:0]', 'state_t st;logic frontier;logic[1:0]')
s=s.replace('source_classifier source_core(', 'frontier_source source_core(')
s=s.replace('.start_pack32(1\'b1),', '.start_frontier(frontier),.start_resident(1\'b1),.start_pack32(1\'b1),')
s=s.replace('.producer_valid,.producer_channel,', '.producer_valid,.producer_active,.producer_channel,')
s=s.replace('.count_channels(schannels),', '.count_channels(schannels),.count_batches(),.count_pairs(),.count_cache_hits(),.count_refetches(),.count_peak_slots(),')
s=s.replace('st<=IDLE;mode<=0;', 'st<=IDLE;frontier<=0;mode<=0;')
s=s.replace('mode<=start_mode;dedup<=', 'frontier<=start_frontier;mode<=start_mode;dedup<=')
(HERE/'joined_core.sv').write_text(s)
core=(HERE.parent/'frontier_source/frontier_source.sv').read_text()
core=core.replace("192+int'(boot)+(boot==30&&mode==3?1:0)","192+int'(boot)")
(HERE/'frontier_source.sv').write_text(core)

s=(OLD/'tb.cpp').read_text()
s=s.replace('bool projected){\n Memory m;', 'bool projected,int mode){\n Memory m;')
s=s.replace('p.roots[i]);', 'p.roots[(i<6&&mode==3)?i+6:i]);')
s=s.replace('bool dedup,ofstream&csv)', 'bool dedup,int frontier,ofstream&csv)')
s=s.replace('memory(p,c,projected)', 'memory(p,c,projected,mode)')
s=s.replace('produced(3072)', 'produced(30720)')
s=s.replace('v.start_mode=mode;', 'v.start_frontier=frontier;v.start_mode=mode;')
a=s.index('  if(v.producer_valid){')
b=s.index('  if(v.code_valid)',a)
s=s[:a]+'''  if(v.producer_valid){
   int pt=v.producer_p;need(pt<32&&v.producer_active,"active producer");
   for(int t=0;t<10;t++)if((v.producer_active>>t)&1){
    int ch=v.producer_channel[t],i=(pt*96+ch)*10+t;
    need(ch<96&&!produced[i],"duplicate (P,c,t) producer");produced[i]=1;nproduced++;
    need(sg(v.producer_u,t*48,48)==o.source_u[i]&&((v.producer_gate>>t)&1)==o.raw[(pt*10+t)*96+ch],"partial source actual MAC/gate mismatch");
   }
  }
'''+s[b:]
s=s.replace('nproduced==int(v.count_source_channels)&&int(v.count_source_mac)==nproduced*100', 'int(v.count_source_mac)==nproduced*10')
s=s.replace('nproduced==2048', 'nproduced==20480')
s=s.replace("for(int n:states)csv<<','<<n;csv<<','<<(mode==0?0:dedup?4:2)<<'\\n';", "for(int n:states)csv<<','<<n;csv<<','<<(mode==0?0:dedup?4:2)<<','<<frontier<<','<<v.count_source_channels<<'\\n';")
s=s.replace(',source_channels,source_scalar_mac', ',source_produced_pairs,source_scalar_mac')
s=s.replace('csv<<",backend_mode\\n";', 'csv<<",backend_mode,frontier,source_channel_refs\\n";')
s=s.replace('for(int bp=0;bp<2;bp++){run(v,p,cases[ci],o,projected,function_name,mode,bp,pass,nh,dedup,csv);commands++;}', 'for(int frontier=0;frontier<(mode<2?1:2);frontier++)for(int bp=0;bp<2;bp++){run(v,p,cases[ci],o,projected,function_name,mode,bp,pass,nh,dedup,frontier,csv);commands++;}')
(HERE/'tb.cpp').write_text(s)
s=(OLD/'prepare.py').read_text()
s=s.replace("ap.add_argument('--adapt',action='store_true');", "ap.add_argument('--adapt',action='store_true');ap.add_argument('--expanded',action='store_true');")
s=s.replace("src=np.load(SOURCE/'source_cases.npz')", "src=np.load(SOURCE/('source_class_adapt/expanded_sources/source_cases.npz' if args.expanded else 'source_cases.npz'))")
s=s.replace('for i in range(2):', "for i in range(len(src['X_q16'])):")
s=s.replace("stem='inputs_adapt' if args.adapt else 'inputs'", "stem=('inputs_adapt' if args.adapt else 'inputs')+('_expanded' if args.expanded else '')")
s=s.replace("'first two training frames, same 32 sampled positions; not held out'", "'training frames, fixed 32 sampled positions; not held out'")
s=s.replace("'source_npz':str(SOURCE/'source_cases.npz')", "'source_npz':str(SOURCE/('source_class_adapt/expanded_sources/source_cases.npz' if args.expanded else 'source_cases.npz'))")
s=s.replace("'mac_units':", "'source_X_holding_bytes':128,'source_graph_cache_bytes':128,'root_bank_neutral':True,'mac_units':")
(HERE/'prepare.py').write_text(s)
