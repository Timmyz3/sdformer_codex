from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'active_prefetch'
s=(OLD/'joined_core.sv').read_text().replace('.start_active_prefetch(1\'b1),',".start_reuse_config(point!=0),.start_active_prefetch(1'b1),")
(HERE/'joined_core.sv').write_text(s)
s=(OLD/'frontier_source.sv').read_text()
s=s.replace('input logic start_pack32,','input logic start_reuse_config,input logic start_pack32,')
# The original port line may use a compact declaration with start_mode before it.
if 'input logic start_reuse_config' not in s:
    s=s.replace('input logic start_pack32', 'input logic start_reuse_config,input logic start_pack32')
s=s.replace('st<=BREQ;mode<=start_mode;', 'st<=start_reuse_config?GSTART:BREQ;mode<=start_mode;')
(HERE/'frontier_source.sv').write_text(s)
s=(OLD/'tb.cpp').read_text().replace('scfg==960','scfg==30').replace('scfg==672','scfg==21')
(HERE/'tb.cpp').write_text(s)
