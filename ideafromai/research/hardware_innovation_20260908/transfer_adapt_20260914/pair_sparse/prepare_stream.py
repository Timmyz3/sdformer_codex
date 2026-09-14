from pathlib import Path
import importlib.util,json,numpy as np
H=Path(__file__).resolve().parent;B=H.parents[1]
spec=importlib.util.spec_from_file_location('pair_original',B/'fusion_review_followup_20260914/pair_dictionary/prepare.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
old=B/'fusion_review_followup_20260914/pair_dictionary/fixtures/real_0'
q=m.readhex(old/'q1.hex').reshape(864,8)
v=m.readhex(old/'q2.hex').reshape(12,8,8).transpose(0,2,1).reshape(96,8)
source=np.load(B/'r8_consumer_fusion_20260914/data/first_source_words.npy',mmap_mode='r')
gold=np.load(B/'r8_consumer_fusion_20260914/data/raw_p_full.npy',mmap_mode='r')
out=H/'fixtures/stream';out.mkdir(parents=True,exist_ok=True);manifest=[]
for tile in range(128,192):
 oy,ox=2*(tile//160)-1,2*(tile%160)-1;s=np.zeros((96,4,4),np.int64)
 for y in range(4):
  for x in range(4):
   if 0<=oy+y<240 and 0<=ox+x<320:s[:,y,x]=source[:,oy+y,ox+x]
 e=np.zeros((864,40),np.int64)
 for k in range(864):
  c,tap=divmod(k,9)
  for p in range(4):e[k,p*10:p*10+10]=(s[c,p//2+tap//3,p%2+tap%3]>>np.arange(10))&1
 z=e.T@q;raw=np.concatenate([z@v[og*8:og*8+8].T for og in range(12)])
 reference=gold[tile].reshape(10,12,8,4).transpose(1,3,0,2).reshape(480,8)
 assert np.array_equal(raw,reference),tile
 p=out/str(tile);p.mkdir(exist_ok=True)
 for n,a in [('source',s),('origin',[oy,ox]),('gold',raw)]:m.writehex(p/(n+'.hex'),a)
 manifest.append(str(p))
(out/'manifest.txt').write_text('\n'.join(manifest)+'\n')
(H/'stream_input_check.json').write_text(json.dumps(dict(tiles=64,first=128,independent_raw_values=64*3840,matched_full_capture=True,source=str(source.filename),gold=str(gold.filename)),indent=2)+'\n')
# C++ batch harness: each source/config handed to real RTL, no dynamic intermediate/gold scheduling.
s=(H/'tb.cpp').read_text().replace('if(argc!=4)','if(argc!=5)')
s=s.replace('d.cfg_valid=0;unsigned checked=0;', '''d.cfg_valid=0;unsigned checked=0;
 std::ifstream mf(argv[4]);std::vector<std::string> tiles;std::string td;while(mf>>td)tiles.push_back(td);
 if(tiles.empty())return 22;''')
s=s.replace('for(unsigned i=0;i<src.size();i++){data[0]=src[i];cfg(0,i);}\n data[0]=(origin[0]&65535)|((origin[1]&65535)<<16);cfg(3,0);','')
s=s.replace('for(unsigned command=0;command<2;command++){', '''for(unsigned command=0;command<2*tiles.size();command++){
  unsigned static_cycles=command==0?cfg_cycles:0;cfg_cycles=0;
  src=read(tiles[command%tiles.size()]+"/source.hex");origin=read(tiles[command%tiles.size()]+"/origin.hex");gold=read(tiles[command%tiles.size()]+"/gold.hex");
  for(unsigned i=0;i<src.size();i++){data[0]=src[i];cfg(0,i);}
  data[0]=(origin[0]&65535)|((origin[1]&65535)<<16);cfg(3,0);d.cfg_valid=0;
  const unsigned this_config=static_cycles+cfg_cycles;''')
s=s.replace('(command==0?cfg_cycles:0)','this_config').replace('checked==7680','checked==3840*2*tiles.size()')
(H/'stream_tb.cpp').write_text(s)
