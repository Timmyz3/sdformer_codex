from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'gp_slice.sv').read_text()
def sub(a,b):
 global s
 assert a in s,a[:100];s=s.replace(a,b)
sub('input logic [1:0] start_frontend,','input logic [2:0] start_frontend,')
sub("?start_frontend:{1'b0,start_intersection}","?start_frontend:{2'b0,start_intersection}")
sub('logic [1:0] frontend_q;','logic [2:0] frontend_q;\n    logic meta_enabled;')
sub('output logic [1:0] dbg_shared_ready,dbg_dense_bypass','output logic [1:0] dbg_shared_ready,dbg_dense_bypass,\n    output logic dbg_meta_trigger,output logic [7:0] dbg_rank_catchup')
sub('assign dbg_shared_ready=shared_ready;','''assign dbg_shared_ready=shared_ready;
    assign dbg_meta_trigger=meta_enabled;
    always_comb begin
        meta_enabled=(frontend_q!=4);dbg_rank_catchup=0;
        for(int k=0;k<4;k++)if(source_phase[k]==BRIDGE)meta_enabled=1;
        for(int m=0;m<2;m++)begin
            if(meta_issued[m]!=0)meta_enabled=1;
            for(int k=0;k<4;k++)if(busy && frontend_q==4 && shared_ready[m] && !dense_bypass[m]
                && weight_index[m*4+k]<source_index[k])dbg_rank_catchup[m*4+k]=1;
        end
    end''')
sub('frontend_q==3','frontend_q>=3')
sub('frontend_q>=2 && !shared_ready[m]','frontend_q>=2 && meta_enabled && !shared_ready[m]')
sub('else if(!(&shared_ready))begin end','else if(frontend_q!=4 && !(&shared_ready))begin end')
sub('if(shared_ready[m])begin','if(shared_ready[m] && (frontend_q!=4 || dense_bypass[m] || weight_index[m*4+k]==source_index[k]))begin')
sub('if(frontend_q>=2 && source_advance[k])','if(frontend_q>=2 && frontend_q!=4 && source_advance[k])')
sub('''for (int k=0;k<4;k++) begin
                    if(frontend_q>=2''','''for(int i=0;i<8;i++)if(dbg_rank_catchup[i])begin
                    weight_index[i]<=weight_index[i]+1'b1;
                    if(weight_bitmap[i/4][weight_index[i]])weight_pointer[i]<=weight_pointer[i]+1'b1;
                end
                for (int k=0;k<4;k++) begin
                    if(frontend_q>=2''')
(H/'gp_slice_lazy.sv').write_text(s)
t=(H/'scoreboard.cpp').read_text()
t=t.replace('intersection==3','intersection>=3')
t=t.replace('frontend_load_beats=0;','frontend_load_beats=0,rank_catchup_pe_beats=0;')
t=t.replace('unsigned trace_seen[8]{};bool meta_seen[2][50]{};', 'unsigned trace_seen[8]{};bool meta_seen[2][50]{};bool any_source=false;for(int k=0;k<4;k++)for(int c0=0;c0<384;c0++)any_source|=e.live[k][c0];')
t=t.replace('return intersection==2?48:(intersection>=3?', 'return intersection==4&&!any_source?0:intersection==2?48:(intersection>=3?')
t=t.replace('s.frontend_load_beats+=(d.dbg_shared_ready!=3);','s.frontend_load_beats+=(d.dbg_shared_ready!=3 && (intersection!=4 || d.dbg_meta_trigger));\n        s.rank_catchup_pe_beats+=pc(d.dbg_rank_catchup);')
t=t.replace('for(int intersection=0;intersection<4;intersection++)','for(int intersection=3;intersection<5;intersection++)')
t=t.replace('configuration_beats\\n','configuration_beats\\trank_catchup_pe_beats\\n')
t=t.replace("<<'\\t'<<c.program.size()+20<<'\\n';","<<'\\t'<<c.program.size()+20<<'\\t'<<s.rank_catchup_pe_beats<<'\\n';")
t=t.replace('sixteen configurations','eight lazy comparison configurations')
(H/'lazy_scoreboard.cpp').write_text(t)
print('isolated lazy mode4 generated; modes0-3 source/results retained')
