#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent; R=P.parents[2]
contract=R/'flow_story_20261021/e3_rne_probe/PREPARED_CONTRACT_V1.json'
c=json.loads(contract.read_text());words=[0]*192
for i,g in enumerate(c['groups']):
 assert g['id']==i and len(g['tiles'])==4
 lists=list(g['tiles'].values());assert all(t==lists[0] for t in lists)
 assert lists[0]==[[j//35,j%35] for j in range(i*151,(i+1)*151)]
 for y,x in lists[0]:
  tile=y*35+x;words[i*32+tile//32]|=1<<(tile%32)
assert all(sum((x>>i)&1 for x in words[m*32:(m+1)*32] for i in range(32))==151 for m in range(6))
(P/'bitmap_words.txt').write_text(''.join('%08x\n'%w for w in words))
(P/'BITMAP_CONTENT_V1.json').write_text(json.dumps({'schema':'r33_bitmap_content_v1','bytes':768,'per_mask_bytes':128,'words_per_mask':32,'complete_tiles':910,'macro_tiles':151,'reserved_statistics_RF_bytes':2048,'words':words,'padding_and_remaining_complete':'all_zero','source_contract_sha256':hashlib.sha256(contract.read_bytes()).hexdigest()},indent=2)+'\n')
paths=[contract,R/'flow_story_20261021/e4_hardware/rne4_supply_v1/rne4_supply.cpp',R/'flow_story_20261021/e4_hardware/rne4_supply_v1/FREEZE_V1.json',R/'flow_story_20261021/e4_hardware/rne4_macro6_v1/FREEZE_V1.json',R/'flow_story_20261021/e4_hardware/rtl/synth_pe.tcl']
(P/'SOURCE_BINDINGS_V1.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
