from pathlib import Path
import json
import numpy as np

H=Path(__file__).resolve().parent
B=H.parents[1]
P=B/'transfer_adapt_20260914/pair_sparse'
OLD=B/'fusion_review_followup_20260914/pair_dictionary'

def readhex(path):
    return np.array([int(x,16) for x in path.read_text().split()],np.uint32).view(np.int32).astype(np.int64)

def writehex(path,array):
    path.write_text(''.join(f'{int(v)&0xffffffff:08x}\n' for v in np.asarray(array).flat))

cases=[]
for c in json.loads((P/'fixtures.json').read_text()):
    path=P/'fixtures'/c['name']
    if not path.exists():path=OLD/'fixtures'/c['name']
    cases.append(dict(name=c['name'],path=str(path)))
q=readhex(OLD/'fixtures/real_0/q1.hex').reshape(864,8)
q2=readhex(OLD/'fixtures/real_0/q2.hex').reshape(12,8,8).transpose(0,2,1).reshape(96,8)
for name in ['long_run','alternating','mixed_runs','negative4_falling','late_endpoint_runs','inverse_time_tag']:
    s=np.full((96,4,4),0b0011111100,np.int64)
    aq=q.copy()
    v=q2.copy()
    if name=='alternating':s[:]=0b0101010101
    if name=='mixed_runs':s[48:]=0b0101010101
    if name=='late_endpoint_runs':s[:48]=0b0101010101
    if name=='negative4_falling':s[:]=0b0000011111;aq[:]=-4
    if name=='inverse_time_tag':
        s[:]=0;aq[:]=0;v[:]=0;v[:,0]=1;ch=0
        for t in range(10):
            for _ in range(t+1):
                s[ch,:,:]=1<<t;aq[ch*9,0]=1;ch+=1
    events=np.empty((864,40),np.int64)
    for k in range(864):
        c,tap=divmod(k,9)
        for p in range(4):
            events[k,p*10:p*10+10]=(s[c,p//2+tap//3,p%2+tap%3]>>np.arange(10))&1
    raw=np.concatenate([events.T@aq@v[g*8:g*8+8].T for g in range(12)])
    p=H/'fixtures'/name;p.mkdir(parents=True,exist_ok=True)
    for f,a in [('source',s),('origin',[11,13]),('q1',aq),('q2',v.reshape(12,8,8).transpose(0,2,1)),('k_live',np.any(aq!=0,axis=1).astype(int)),('gold',raw)]:
        writehex(p/(f+'.hex'),a)
    cases.append(dict(name=name,path=str(p)))
(H/'fixtures.json').write_text(json.dumps(cases,indent=2)+'\n')

# Preserve the mature native core TB's complete configuration/output checks.
s=(OLD/'tb.cpp').read_text()
s=s.replace('if(argc!=4)', 'if(argc!=4&&argc!=5)')
start=s.index(' if(mode==15){')
end=s.index(' d.cfg_valid=0;unsigned checked=0;',start)
s=s[:start]+s[end:]
s=s.replace('  d.start=1;', '  unsigned runmode=command&&argc==5?std::stoul(argv[4]):mode;d.mode=runmode;\n  d.start=1;')
order=json.loads((H/'calibration.json').read_text())['permutation']
packed=sum(v<<(4*t) for t,v in enumerate(order))
s=s.replace('d.cfg_valid=0;unsigned checked=0;', 'd.cfg_valid=0;unsigned checked=0;bool permutation_loaded=false;')
s=s.replace('  d.start=1;', f'''  unsigned command_cfg=command==0?cfg_cycles:0,permutation_config=0;
  if(runmode>=4&&!permutation_loaded){{
   data[0]=uint32_t(0x{packed:x}ULL);data[1]=uint32_t(0x{packed:x}ULL>>32);cfg(7,0);d.cfg_valid=0;
   permutation_loaded=true;command_cfg++;permutation_config=1;
  }}
  d.start=1;''')
s=s.replace('<<(command==0?cfg_cycles:0)', '<<command_cfg<<",\\\"permutation_configuration_cycles\\\":"<<permutation_config')
s=s.replace('<<mode<<','<<runmode<<')
start=s.index('<<",\\\"aux_reads\\\":')
end=s.index('<<",\\\"state_cycles\\\":[";',start)
s=s[:start]+s[end:]
s=s.replace('    for(int i=0;i<64;i++)std::cout', '''    for(int i=0;i<64;i++)std::cout''')
s=s.replace('std::cout<<"]}\\n";break;', '''std::cout<<"]";
#define SHOW(x) std::cout<<",\\\"" #x "\\\":"<<uint64_t(d.x)
    SHOW(direct_issues);SHOW(endpoint_issues);SHOW(selection_issues);SHOW(prefix_issues);SHOW(merge_issues);
    SHOW(selected_direct_columns);SHOW(selected_endpoint_columns);SHOW(direct_pair_work);SHOW(endpoint_pair_work);
    SHOW(prefix_reads);SHOW(merge_reads);SHOW(zclear_writes);SHOW(falling_fields);
    SHOW(preclassified_direct_columns);SHOW(empty_columns_skipped);
    std::cout<<"}\\n";break;''')
(H/'tb.cpp').write_text(s)
print(json.dumps(dict(fixtures=len(cases),old_fixtures_referenced=20,new_small_fixtures=6)))
