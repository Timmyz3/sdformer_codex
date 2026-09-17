"""Integrate the active-frontier prefetch adaptation, root-bank neutral."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
SOURCE=PARENT.parent
s=(PARENT/'joined_core.sv').read_text().replace('.start_frontier(frontier),',".start_active_prefetch(1'b1),.start_frontier(frontier),")
(HERE/'joined_core.sv').write_text(s)
s=(SOURCE/'frontier_source/active_prefetch/frontier_source.sv').read_text()
s=s.replace("192+int'(boot)+(boot==30&&mode==3?1:0)","192+int'(boot)")
(HERE/'frontier_source.sv').write_text(s)
s=(PARENT/'tb.cpp').read_text()
s=s.replace('nproduced=0,ncodes=0,', 'nproduced=0,nrefs=0,ncodes=0,')
s=s.replace('int pt=v.producer_p;need(pt<32&&v.producer_active,"active producer");', 'int pt=v.producer_p;need(pt<32&&v.producer_active,"active producer");array<bool,96> batch_refs{};')
s=s.replace('produced[i]=1;nproduced++;', 'produced[i]=1;nproduced++;if(!batch_refs[ch]){batch_refs[ch]=true;nrefs++;}')
s=s.replace('need(int(v.count_source_mac)==nproduced*10,"source MAC accounting");', 'need(int(v.count_source_mac)==nproduced*10&&int(v.count_source_channels)==nrefs,"source MAC/reference accounting");')
(HERE/'tb.cpp').write_text(s)
