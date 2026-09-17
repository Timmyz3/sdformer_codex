from pathlib import Path
p=Path(__file__).resolve().parent
s=(p.parents[2]/'support_lut_execution_20260915/tb.cpp').read_text()
s=s[:s.index('void run(')].replace('Vsupport_fc1','Vjoined_fc1')
s+='''
void run(Vjoined_fc1& v,const Case& c,const vector<uint8_t>&D,const vector<int16_t>&A,int mode,int bp,int warm,ofstream& out,const string& series,int expected_table=-1){
 if(expected_table<0)expected_table=(mode&&!warm)?64:0;
 auto mem=memory(c,D,A,4);
 struct Pending{bool live=false;int due=0,addr=0;};array<Pending,8> pending;
 array<bool,8> reqhold{};array<int,8> prevaddr{};
 bool started=false,outhold=false,source_offer=false;int source=0,outputs=0,firstsource=-1,lastout=-1;
 int words=0,coeff=0,config=0,outstall=0,reqstall=0,rspstall=0,sourcestall=0;
 array<uint64_t,21> sc{};array<bool,320> seen{};
 vector<uint32_t> heldY(72),heldU(144),heldG(3);int heldrow=0;
 uint64_t ycheck=0,ucheck=0,gcheck=0,bcheck=0,echeck=0,tcheck=0,tailcheck=0,evencheck=0,yreadcycles=0,packreadcycles=0;
 uint64_t ywritecycles=0,yreadbanks=0,ywritebanks=0,memreadcheck=0,memwritecheck=0;
 vector<int64_t> committed(30720),issued(30720);
 struct Write{int row,mask;array<int64_t,96> value;};vector<Write> writequeue;size_t writehead=0;
 v.start_mode=mode;v.start_hblock=c.hb;v.start_reuse_config=warm;
 need(v.start_ready,"start not idle");
 for(int cyc=0;cyc<200000;cyc++){
  v.clk=0;v.start_valid=!started;
  if(!source_offer&&source<320&&v.dbg_state==3&&(!bp||cyc%7!=2))source_offer=true;
  v.source_valid=source_offer;for(int w=0;w<3;w++)v.source_data[w]=0;
  if(source<320)for(int ch=0;ch<96;ch++)v.source_data[ch/32]|=uint32_t(c.S[source*96+ch])<<(ch%32);
  v.out_ready=!bp||cyc%11>2;v.done_ready=!bp||(cyc%17!=3&&cyc%17!=4);v.mem_req_ready=0;v.mem_rsp_valid=0;
  for(int b=0;b<8;b++){
   if(!bp||(cyc+3*b)%9>1)v.mem_req_ready|=1<<b;
   if(pending[b].live&&pending[b].due<=cyc){v.mem_rsp_valid|=1<<b;for(int w=0;w<4;w++)v.mem_rsp_data[b][w]=mem[pending[b].addr][w];}
  }
  v.eval();int st=v.dbg_state;need(st<21,"state");sc[st]++;
  if(v.start_valid&&v.start_ready)started=true;
  if(v.source_valid&&!v.source_ready)sourcestall++;
  if(v.source_valid&&v.source_ready){if(firstsource<0)firstsource=cyc;source++;source_offer=false;}
  for(int b=0;b<8;b++){
   bool rv=(v.mem_req_valid>>b)&1,rr=(v.mem_req_ready>>b)&1;
   if(reqhold[b])need(rv&&v.mem_req_addr[b]==prevaddr[b],"request changed while stalled");
   reqhold[b]=rv&&!rr;prevaddr[b]=v.mem_req_addr[b];if(rv&&!rr)reqstall++;
   if((v.mem_rsp_valid>>b)&1 && !((v.mem_rsp_ready>>b)&1))rspstall++;
   bool rsp=((v.mem_rsp_valid&v.mem_rsp_ready)>>b)&1;
   if(rsp){need(pending[b].live,"unsolicited response");pending[b].live=false;}
   if(rv&&rr){need(!pending[b].live,"bank outstanding >1");int addr=v.mem_req_addr[b];need(addr%8==b,"bank mapping");
    pending[b]={true,cyc+1+(bp?(cyc+2*b)%5:0),addr};words++;if(st==4)coeff++;else config++;
   }
  }
  if(v.mon_y_read){
   yreadcycles++;int row=v.mon_y_addr;
   for(int b=0;b<8;b++)yreadbanks+=bits(v.joined_fc1__DOT__yvalid[b],row,1);
   for(int h=0;h<96;h++){need(signedbits(v.mon_y_data,h*24,24)==committed[row*96+h],"actual shared memory read/ownership");memreadcheck++;}
  }
  if(v.joined_fc1__DOT__fv&2){
   ywritecycles++;need(writehead<writequeue.size(),"orphan FC writeback");const Write& w=writequeue[writehead++];
   need(w.row==int(v.joined_fc1__DOT__fr[1])&&w.mask==int(v.joined_fc1__DOT__fb[1]),"FC writeback ownership");
   for(int b=0;b<8;b++)if(w.mask>>b&1){
    ywritebanks++;
    for(int j=0;j<12;j++){int h=b*12+j;uint32_t raw=v.joined_fc1__DOT__fvalue[1][h];int64_t actual=(raw&0x800000)?int64_t(raw)-0x1000000:int64_t(raw);
     need(actual==w.value[h],"actual FC partial Y write value");committed[w.row*96+h]=actual;memwritecheck++;}
   }
  }
  if(v.joined_fc1__DOT__fc_issue){
   Write w;w.row=v.joined_fc1__DOT__active_row;w.mask=v.joined_fc1__DOT__active_banks;
   for(int h=0;h<96;h++){
    if(w.mask>>(h/12)&1)issued[w.row*96+h]+=int16_t(v.joined_fc1__DOT__coeff[h]);
    w.value[h]=issued[w.row*96+h];
   }writequeue.push_back(w);
  }
  if(st==20)packreadcycles++;
  if(st==16){int r=v.mon_y_addr;for(int h=0;h<96;h++){need(signedbits(v.mon_y_data,h*24,24)==c.Y[r*96+h],"fused shared Y read mismatch");ycheck++;}}
  if(v.mon_table_valid){int half=v.mon_table_addr/32,code=v.mon_table_addr%32;for(int t=0;t<10;t++){int ref=0;for(int b=0;b<5;b++)if(code>>b&1)ref+=A[t*10+half*5+b];need(signedbits(v.mon_table_data,t*16,16)==ref,"actual LUT construction");tcheck++;}}
  if(st==17){int exp=0;for(int s=0;s<10;s++)for(int j=0;j<8;j++){int64_t x=abs(int64_t(c.Y[(v.mon_p*10+s)*96+v.mon_hgroup*8+j]));int e=0;while(x){x>>=1;e++;}exp=max(exp,e);}need(v.mon_exponent==exp,"actual Y exponent");echeck++;}
  if(v.mon_group_valid){
   for(int i=0;i<80;i++)need(bits(v.joined_fc1__DOT__locked,i,1),"unlocked group output");
   if(mode==1){need(v.mon_full_u_valid,"missing full U monitor");for(int i=0;i<80;i++){
    int t=i/8,h=v.mon_hgroup*8+i%8;need(signedbits(v.mon_group_u,i*48,48)==c.U[(v.mon_p*10+t)*96+h],"fused full U mismatch");ucheck++;
   }}
  }
  if(v.mon_bounds){
   for(int i=0;i<80;i++){int t=i/8,h=v.mon_hgroup*8+i%8;int64_t u=c.U[(v.mon_p*10+t)*96+h];
    need(signedbits(v.mon_bound_lo,i*48,48)<=u&&signedbits(v.mon_bound_hi,i*48,48)>=u,"certificate bounds");bcheck+=2;}
   for(int t=0;t<10;t++)for(int sn=0;sn<2;sn++){
    int64_t z=0;for(int s=0;s<10;s++)z+=sn?min(int64_t(A[t*10+s]),int64_t(0)):max(int64_t(A[t*10+s]),int64_t(0));
    int64_t tail=z*((int64_t(1)<<v.mon_m)-1);need(signedbits(v.mon_tail,(2*t+sn)*48,48)==tail,"tail recurrence");tailcheck++;
    if(v.mon_m){int64_t delta=signedbits(v.mon_tail_delta,(2*t+sn)*48,48);need(delta==tail-z&&delta%2==0,"tail recurrence exact division");evencheck++;}
   }
  }
  if(outhold){need(v.out_valid&&int(v.out_row)==heldrow,"output valid/row stalled");
   need(equal(heldY.begin(),heldY.end(),v.out_y)&&equal(heldU.begin(),heldU.end(),v.out_u)&&equal(heldG.begin(),heldG.end(),v.out_gate),"output data stalled");}
  outhold=v.out_valid&&!v.out_ready;
  if(outhold){outstall++;heldrow=v.out_row;copy_n(v.out_y,72,heldY.begin());copy_n(v.out_u,144,heldU.begin());copy_n(v.out_gate,3,heldG.begin());}
  if(v.out_valid&&v.out_ready){int r=v.out_row;need(r==outputs&&r<320&&!seen[r],"output row/order duplicate");seen[r]=true;outputs++;lastout=cyc;
   for(int h=0;h<96;h++){
    int i=r*96+h;int64_t y=signedbits(v.out_y,h*24,24);bool gate=bits(v.out_gate,h,1);
    if(y!=c.Y[i]||gate!=bool(c.gold[i]))throw runtime_error(c.name+" mode="+to_string(mode)+" bp="+to_string(bp)+" r="+to_string(r)+" h="+to_string(h)+" Y="+to_string(y)+"/"+to_string(c.Y[i]));
    ycheck++;gcheck++;
    if(mode==0){need(signedbits(v.out_u,h*48,48)==c.U[i],"native U mismatch");ucheck++;}
   }
  }
  bool done=v.done_valid&&v.done_ready;
  if(done){
   need(writehead==writequeue.size(),"unretired FC writes");
   need(source==320&&outputs==320,"incomplete");need(coeff==int(v.dbg_coeff_words),"request count");
   need(config==(warm?0:397),"common configuration word count");
   need(v.dbg_peak_words<=24&&v.dbg_peak_desc<=4,"prefetch limits");
   for(int b=0;b<8;b++)need(!pending[b].live,"done with outstanding memory");
   need(v.dbg_pack_writes==(mode?384:0)&&v.dbg_pack_reads==(mode?320:0),"transpose traffic");
   need(v.dbg_table_writes==expected_table,"cold table count");
   need(gcheck==30720&&ucheck==(mode==2?0:30720)&&ycheck==(mode?61440:30720),"coverage counts");
   uint64_t psn=0;for(int i=5;i<=9;i++)psn+=sc[i];for(int i=15;i<=20;i++)psn+=sc[i];
   uint64_t build=sc[11]+sc[12]+sc[13]+sc[14];
   out<<"{\\"series\\":\\""<<series<<"\\",\\"case\\":\\""<<c.name<<"\\",\\"real\\":"<<c.real<<",\\"mode\\":"<<mode<<",\\"bp\\":"<<bp<<",\\"warm\\":"<<warm<<",\\"cycles\\":"<<cyc+1<<",\\"source_to_gate\\":"<<lastout-firstsource+1
    <<",\\"fc_cycles\\":"<<sc[4]<<",\\"psn_cycles\\":"<<psn<<",\\"build_cycles\\":"<<build<<",\\"boot_cycles\\":"<<sc[1]+sc[2]<<",\\"words\\":"<<words<<",\\"coeff_words\\":"<<coeff<<",\\"config_words\\":"<<config
    <<",\\"updates\\":"<<v.dbg_updates<<",\\"mac\\":"<<v.dbg_mac<<",\\"jobs\\":"<<v.dbg_jobs<<",\\"zero_jobs\\":"<<v.dbg_zero_jobs<<",\\"peak_words\\":"<<v.dbg_peak_words<<",\\"peak_desc\\":"<<v.dbg_peak_desc
    <<",\\"planes\\":"<<v.dbg_planes<<",\\"early_groups\\":"<<v.dbg_early<<",\\"table_writes\\":"<<v.dbg_table_writes<<",\\"pack_writes\\":"<<v.dbg_pack_writes<<",\\"pack_reads\\":"<<v.dbg_pack_reads<<",\\"pack_read_cycles\\":"<<packreadcycles
    <<",\\"y_read_cycles\\":"<<yreadcycles<<",\\"gate_checks\\":"<<gcheck<<",\\"y_checks\\":"<<ycheck<<",\\"u_checks\\":"<<ucheck<<",\\"bound_checks\\":"<<bcheck<<",\\"exponent_checks\\":"<<echeck<<",\\"table_checks\\":"<<tcheck<<",\\"tail_checks\\":"<<tailcheck<<",\\"tail_even_checks\\":"<<evencheck
    <<",\\"out_stall\\":"<<outstall<<",\\"bank_req_stall\\":"<<reqstall<<",\\"bank_rsp_stall\\":"<<rspstall<<",\\"source_stall\\":"<<sourcestall<<",\\"state_cycles\\":[";
   for(int i=0;i<21;i++)out<<(i?",":"")<<sc[i];out<<"]}\\n";out.flush();
  }
  v.clk=1;v.eval();v.clk=0;v.eval();
  if(done)return;
 }
 throw runtime_error("timeout "+c.name+" mode="+to_string(mode)+" state="+to_string(v.dbg_state));
}
int main(int argc,char**argv){try{
 Verilated::commandArgs(argc,argv);ifstream f("../../../support_lut_execution_20260915/cases.bin",ios::binary);uint32_t count;rd(f,&count,1);
 vector<uint8_t>D(1536);vector<int16_t>A(100);rd(f,D.data(),D.size());rd(f,A.data(),A.size());vector<Case> cases;
 for(unsigned i=0;i<count;i++){
  Case c;uint32_t n;rd(f,&c.real,1);rd(f,&c.hb,1);rd(f,&n,1);c.name.resize(n);rd(f,c.name.data(),n);
  c.S.resize(30720);c.W.resize(9216);c.tau.resize(960);c.positive.resize(96);c.constant.resize(96);c.cgate.resize(960);c.Y.resize(30720);c.U.resize(30720);c.gold.resize(30720);
  rd(f,c.S.data(),c.S.size());rd(f,c.W.data(),c.W.size());rd(f,c.tau.data(),c.tau.size());rd(f,c.positive.data(),96);rd(f,c.constant.data(),96);rd(f,c.cgate.data(),960);rd(f,c.Y.data(),30720);rd(f,c.U.data(),30720);rd(f,c.gold.data(),30720);cases.push_back(move(c));
 }
 string series=argc>1?argv[1]:"all";vector<int> selected;
 if(series=="swap")selected={0,1,36};else if(series=="smoke")selected={0,1,32,33,34,35,36};else for(int i=0;i<37;i++)selected.push_back(i);
 ofstream out("results_"+series+".jsonl");
 Vjoined_fc1 v;v.clk=0;v.rst_n=0;v.start_valid=0;v.source_valid=0;v.mem_req_ready=0;v.mem_rsp_valid=0;v.out_ready=0;v.done_ready=0;
 for(int b=0;b<8;b++)for(int w=0;w<4;w++)v.mem_rsp_data[b][w]=0;
 for(int k=0;k<3;k++){v.clk=0;v.eval();v.clk=1;v.eval();}v.clk=0;v.rst_n=1;v.eval();
 int commands=0;
 for(int i:selected){
  const Case& c=cases[i];Case copy=c;check_dense(copy,A);
  if(series=="swap"){
   for(int bp=0;bp<2;bp++){
    run(v,c,D,A,0,bp,0,out,series,0);
    run(v,c,D,A,1,bp,1,out,series,64); // native cold did not build the LUT
    run(v,c,D,A,2,bp,1,out,series,0);
    run(v,c,D,A,0,bp,1,out,series,0);
    run(v,c,D,A,1,bp,0,out,series,64);
    run(v,c,D,A,2,bp,1,out,series,0);commands+=6;
   }
  }else for(int bp=0;bp<2;bp++)for(int mode=0;mode<3;mode++)for(int warm=0;warm<2;warm++){run(v,c,D,A,mode,bp,warm,out,series);commands++;}
  cout<<"PASS "<<i<<" "<<c.name<<endl;
 }
 cout<<"ALL PASS commands="<<commands<<endl;
 }catch(exception&e){cerr<<"FAIL "<<e.what()<<endl;return 1;}return 0;}
'''
s=s.replace('    <<",\\"out_stall\\":"<<outstall','    <<",\\"y_write_cycles\\":"<<ywritecycles<<",\\"y_read_banks\\":"<<yreadbanks<<",\\"y_write_banks\\":"<<ywritebanks<<",\\"memory_read_checks\\":"<<memreadcheck<<",\\"memory_write_checks\\":"<<memwritecheck\n    <<",\\"out_stall\\":"<<outstall')
(p/'tb.cpp').write_text(s)
