"""Static graph export plus X-only source replay; no gate/code DUT inputs."""
import sys,io,struct
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent

def main():
    src=np.load(SOURCE/'source_cases.npz')
    extended=np.load(SOURCE/'source_class_adapt/expanded_sources/source_cases.npz')
    profiles=[np.load(SOURCE/'prefix_tables.npz'),np.load(SOURCE/'prefix_tables_entropy.npz')]
    for key in ['A_q12','threshold_q28','D']:assert np.array_equal(src[key],extended[key])
    assert np.array_equal(src['X_q16'],extended['X_q16'][:2])
    for tag,tablefile in [('zero','zero_tables.npz'),('l2','integer_l2_tables.npz')]:
        tables=np.load(HERE/tablefile)
        head=io.BytesIO();head.write(src['A_q12'].astype('<i2').tobytes());head.write(src['threshold_q28'].astype('<i8').tobytes());head.write(src['D'].astype('u1').tobytes())
        for order,q in enumerate(profiles):
            cl=tables['class_nodes_natural' if order==0 else 'class_nodes_entropy']
            for nd in [q['code_nodes64'],cl]:
                assert len(nd)<=2180;head.write(struct.pack('<I',len(nd)));head.write(nd.astype('<u8').tobytes())
            roots=q['roots'].copy();roots[1]=tables['roots'][order];head.write(roots.astype('<u2').tobytes());head.write(tables['canonical'].astype('u1').tobytes())
            rank=np.full((6,16),15,dtype=np.uint8)
            for g in range(6):rank[g,q[f'variables_g{g}']]=np.arange(len(q[f'variables_g{g}']))
            head.write(rank.tobytes())
        for scope,nframes in [('small',2),('expanded',32)]:
            cases=[]
            for frame in range(nframes):
                for pt in range(32):cases.append((f'train{frame}_p{pt}',1,extended['X_q16'][frame,:,pt,:].T))
            cases.extend([('diagnostic_zero',0,np.zeros((96,10),dtype=np.int32)),
                          ('diagnostic_positive_extreme',0,np.full((96,10),(1<<23)-1,dtype=np.int32)),
                          ('diagnostic_signed_extreme',0,np.tile(np.asarray([-(1<<23),(1<<23)-1]*5,dtype=np.int32),(96,1)))])
            with (HERE/f'source_{tag}_{scope}.bin').open('wb') as f:
                f.write(head.getvalue());f.write(struct.pack('<I',len(cases)))
                for name,real,x in cases:
                    name=name.encode();f.write(struct.pack('<II',real,len(name)));f.write(name);f.write(x.astype('<i4').tobytes())
    source=(SOURCE/'tb_source.cpp').read_text().split('int main(int argc,char**argv)')[0]
    source+=r'''int main(int argc,char**argv){try{
 Verilated::commandArgs(argc,argv);need(argc==3,"input.bin output.csv");ifstream f(argv[1],ios::binary);array<int16_t,100>A;array<int64_t,10>tau;array<uint8_t,1536>D;
 rd(f,A.data(),100);rd(f,tau.data(),10);rd(f,D.data(),1536);array<Profile,2>profiles;
 for(auto&q:profiles){uint32_t n;rd(f,&n,1);q.code.resize(n);rd(f,q.code.data(),n);rd(f,&n,1);q.cl.resize(n);rd(f,q.cl.data(),n);rd(f,q.roots.data(),12);rd(f,q.canonical.data(),96);rd(f,q.rank.data(),96);}
 uint32_t count;rd(f,&count,1);ofstream csv(argv[2]);csv<<"case,real,order,mode,bp,packed32,prefetch,cycles,channels,scalar_mac,words,xwords,graph_words,prefetch_words,graph_hits,req_stall,out_stall";for(int i=0;i<14;i++)csv<<",state"<<i;csv<<'\n';
 for(unsigned i=0;i<count;i++){
  Case c;uint32_t n;rd(f,&c.real,1);rd(f,&n,1);c.name.resize(n);rd(f,c.name.data(),n);rd(f,c.x.data(),960);
  for(int mode=2;mode<4;mode++)for(int bp=0;bp<2;bp++)run(c,profiles[1],1,mode,bp,1,1,A,tau,D,csv);
  if(i%32==31||i+1==count)cout<<"PASS "<<i+1<<'/'<<count<<' '<<c.name<<endl;
 }
 cout<<"ALL STRONG SOURCE X/MAC/CODE/CLASS/BANK CHECKS PASS; commands="<<count*4<<endl;
 }catch(exception&e){cerr<<e.what()<<endl;return 1;}return 0;}
'''
    (HERE/'source_tb.cpp').write_text(source)
    print('Exported zero/L2, two-frame/32-frame X-only inputs; only entropy packed32 prefetch code/class will run.')

if __name__=='__main__':main()
