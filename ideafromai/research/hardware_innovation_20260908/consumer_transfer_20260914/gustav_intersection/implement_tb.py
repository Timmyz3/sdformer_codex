from pathlib import Path
H=Path(__file__).resolve().parent;B=H.parents[1];OLD=B/'psn/rtl/gp_slice'
s=(OLD/'intersection_scoreboard.cpp').read_text()
def sub(a,b):
 global s
 assert a in s,a[:100];s=s.replace(a,b)
sub('dense{},compressed{};','dense{},compressed{},bitmap{},adaptive{};')
sub('''    for(int k=0;k<4;k++) {
        e.source[k].fill''','''    for(int m=0;m<2;m++) {
        e.bitmap[m].fill(0x96);e.adaptive[m].fill(0xb3);
        for(int i=0;i<48;i++)e.bitmap[m][i]=0;
        for(unsigned j=0;j<e.columns[m].size();j++) {
            unsigned d=e.columns[m][j];e.bitmap[m][d/8]|=1u<<(d%8);
            e.bitmap[m][48+j]=uint8_t(c.W[m][d]);
        }
        unsigned count=e.columns[m].size();e.adaptive[m][0]=count&255;e.adaptive[m][1]=count>>8;
        if(count>=334)for(int d=0;d<384;d++)e.adaptive[m][2+d]=uint8_t(c.W[m][d]);
        else for(unsigned i=0;i<48+count;i++)e.adaptive[m][2+i]=e.bitmap[m][i];
    }
    for(int k=0;k<4;k++) {
        e.source[k].fill''')
sub('U w_metadata=0,w_values=0,compare_pe_beats=0,advance_weight=0,advance_nrv=0;','U w_metadata=0,w_values=0,compare_pe_beats=0,advance_weight=0,advance_nrv=0,frontend_load_beats=0;')
sub('clear_io(d);need(d.start_ready,"previous task was not retired");','clear_io(d);need(d.start_ready,"previous task was not retired");rng=0x19283047;')
sub('d.start_intersection=intersection;','d.start_intersection=intersection==1;d.start_frontend=intersection;')
sub('unsigned trace_seen[8]{};','''unsigned trace_seen[8]{};bool meta_seen[2][50]{};
    auto sparse=[&](int m){return intersection==1 || intersection==2 || (intersection==3 && e.columns[m].size()<334);};
    auto metadata_length=[&](int m){return intersection==2?48:(intersection==3?(e.columns[m].size()>=334?2:50):0);};
    auto image_bytes=[&](int m){return intersection==0?384:(intersection==1?2+3*e.columns[m].size():(intersection==2?48+e.columns[m].size():(e.columns[m].size()>=334?386:50+e.columns[m].size())));};
    (void)image_bytes;''')
sub('((cycle+tag)%19>=7 && (rand&7)!=0)','(step%19>=7 && (rand&7)!=0)')
sub('s.compare_pe_beats+=pc(d.dbg_compare);','s.frontend_load_beats+=(d.dbg_shared_ready!=3);\n        s.compare_pe_beats+=pc(d.dbg_compare);')
sub('''                if(intersection) {
                    unsigned id=m*4+k;''','''                if(intersection==1) {
                    unsigned id=m*4+k;''')
sub('''                } else need(kind==0 && addr==source && addr<384,"dense W address/kind/source");
                if(kind==0) {
                    unsigned column=intersection?e.columns[m][(addr-4)/3]:addr;''','''                } else if(intersection==0)need(kind==0 && addr==source && addr<384,"dense W address/kind/source");
                else if(kind==5) {
                    need(addr<unsigned(metadata_length(m)) && !meta_seen[m][addr],"shared metadata actual address/duplicate");
                    meta_seen[m][addr]=true;
                } else {
                    need(kind==0 && ((d.dbg_shared_ready>>m)&1),"value before real metadata returned");
                    unsigned rank=std::lower_bound(e.columns[m].begin(),e.columns[m].end(),source)-e.columns[m].begin();
                    unsigned expected=sparse(m)?(intersection==2?48:50)+rank:2+source;
                    need(addr==expected,"bitmap actual compressed weight address/rank");
                    need(bool((d.dbg_dense_bypass>>m)&1)==(intersection==3 && e.columns[m].size()>=334),"actual header dense bypass");
                }
                if(kind==0) {
                    unsigned column=intersection==1?e.columns[m][(addr-4)/3]:source;''')
sub('need(!intersection || c.W[m][column]!=0,"zero W fetched as compressed value");','need(!sparse(m) || c.W[m][column]!=0,"zero W fetched as compressed value");')
sub('((tag/4)&1)','(c.time&1)')
sub('intersection?e.compressed[m][addr]:e.dense[m][addr]','intersection==0?e.dense[m][addr]:(intersection==1?e.compressed[m][addr]:(intersection==2?e.bitmap[m][addr]:e.adaptive[m][addr]))')
sub('(!intersection || c.W[m][a]!=0)','(!sparse(m) || c.W[m][a]!=0)')
sub('if(intersection)need(trace_seen[m*4+k]','if(intersection==1)need(trace_seen[m*4+k]')
sub('''            need(s.w_reads==s.w_metadata+s.w_values && (!intersection || s.w_zero==0),"W byte/value conservation");''','''            for(int m=0;m<2;m++)for(int a=0;a<metadata_length(m);a++)need(meta_seen[m][a],"missing shared metadata read");
            need(s.w_reads==s.w_metadata+s.w_values,"W byte/value conservation");
            if(intersection==1 || intersection==2)need(s.w_zero==0,"compressed zero W return");''')
sub('intersection<2','intersection<4')
sub('eight configurations','sixteen configurations')
sub('image_bytes_tile1\\n','image_bytes_tile1\\tfrontend_load_beats\\tconfiguration_beats\\n')
sub("<<'\\t'<<(intersection?2+3*e.columns[0].size():384)<<'\\t'<<(intersection?2+3*e.columns[1].size():384)<<'\\n';",'''<<'\\t'<<(intersection==0?384:(intersection==1?2+3*e.columns[0].size():(intersection==2?48+e.columns[0].size():(e.columns[0].size()>=334?386:50+e.columns[0].size()))))
                   <<'\\t'<<(intersection==0?384:(intersection==1?2+3*e.columns[1].size():(intersection==2?48+e.columns[1].size():(e.columns[1].size()>=334?386:50+e.columns[1].size()))))
                   <<'\\t'<<s.frontend_load_beats<<'\\t'<<c.program.size()+20<<'\\n';''')
(H/'scoreboard.cpp').write_text(s)
print('memory-only responder/oracle generated for four frontends')
