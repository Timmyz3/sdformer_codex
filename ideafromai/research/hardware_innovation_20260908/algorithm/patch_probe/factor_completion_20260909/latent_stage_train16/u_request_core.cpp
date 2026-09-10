// Ordered U address census. One current word, discarded between P4/stages.
// No latency, PE, SRAM energy, or end-to-end service claim.
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using u64=uint64_t;
static int pc(u64 x) { return __builtin_popcountll(x); }
template<class T> std::vector<T> read(const char* file, size_t count) {
    std::vector<T> x(count); std::ifstream f(file,std::ios::binary);
    f.read(reinterpret_cast<char*>(x.data()),count*sizeof(T));
    if(!f) throw std::runtime_error(std::string("short input: ")+file);
    return x;
}
struct Count {
    u64 requests=0, useful_coefficients=0, scalar_updates=0, source_TP_updates=0, max_scalar_updates=0;
    std::array<u64,33> useful_hist{};
    Count& operator+=(const Count& x) {
        requests+=x.requests; useful_coefficients+=x.useful_coefficients;
        scalar_updates+=x.scalar_updates; source_TP_updates+=x.source_TP_updates;
        if(x.max_scalar_updates>max_scalar_updates) max_scalar_updates=x.max_scalar_updates;
        for(int j=0;j<33;j++) useful_hist[j]+=x.useful_hist[j];
        return *this;
    }
};
struct Phase {
    std::array<Count,3> counts;
    u64 active_P4=0, source_words_examined=0, weight_mask_reads=0;
    Phase& operator+=(const Phase& x) {
        active_P4+=x.active_P4; source_words_examined+=x.source_words_examined;
        weight_mask_reads+=x.weight_mask_reads;
        for(int b=0;b<3;b++) counts[b]+=x.counts[b]; return *this;
    }
};
struct Current {
    int64_t word=-1; int k=-1, used=0; u64 updates=0,tp=0,kmask=0;
    void flush(Count& c) {
        if(word<0) return;
        tp+=pc(kmask); c.requests++; c.useful_coefficients+=used;
        c.scalar_updates+=updates; c.source_TP_updates+=tp; c.useful_hist[used]++;
        if(updates>c.max_scalar_updates)c.max_scalar_updates=updates;
        word=-1; k=-1; used=0; updates=0;tp=0;kmask=0;
    }
    void use(int64_t address,int newk,u64 events,int n,Count& c) {
        if(address!=word) { flush(c); word=address; }
        if(k!=newk) { tp+=pc(kmask); kmask=0; k=newk; }
        used++; updates+=n; kmask|=events;
    }
};
void write_count(std::ostream& o,const Count& c,int bytes) {
    o<<"{\"requests\":"<<c.requests<<",\"bytes\":"<<c.requests*bytes
     <<",\"useful_coefficients\":"<<c.useful_coefficients<<",\"scalar_updates\":"<<c.scalar_updates
     <<",\"source_TP_word_updates\":"<<c.source_TP_updates<<",\"max_scalar_updates_per_word\":"<<c.max_scalar_updates
     <<",\"useful_coefficients_histogram\":[";
    for(int j=0;j<=bytes;j++) { if(j)o<<','; o<<c.useful_hist[j]; }
    o<<"]}";
}
void write_phase(std::ostream& o,const Phase& p) {
    o<<"{\"active_P4\":"<<p.active_P4<<",\"source_K_words_examined\":"<<p.source_words_examined
     <<",\"weight_nonzero_mask64_reads\":"<<p.weight_mask_reads<<",\"words\":{";
    constexpr int b[3]={2,8,32};
    for(int j=0;j<3;j++) { if(j)o<<',';o<<'"'<<b[j]<<"\":";write_count(o,p.counts[j],b[j]); }
    o<<"}}";
}
int main(int argc,char**argv) {
    if(argc!=9) return 2;
    int G=std::stoi(argv[4]),K=std::stoi(argv[5]),R=std::stoi(argv[6]),shared=std::stoi(argv[7]);
    auto src=read<u64>(argv[1],size_t(G)*K), need=read<u64>(argv[2],size_t(G)*R/2);
    auto weights=read<int8_t>(argv[3],size_t(K)*R);
    std::vector<u64> wnz(K);
    for(int k=0;k<K;k++)for(int r=0;r<R;r++)if(weights[k*R+r])wnz[k]|=u64(1)<<r;
    std::array<Phase,2> totals;
    std::array<std::array<Phase,2>,3> selected;
    const int selected_g[3]={0,G/2,G-1}, widths[3]={2,8,32};
    for(int g=0;g<G;g++)for(int phase=0;phase<2;phase++) {
        int first=phase?shared:0,last=phase?R:shared,nr=last-first;
        int64_t base=phase?int64_t(K)*shared:0;
        Phase local; std::array<Current,3> current;
        const u64* needs=need.data()+size_t(g)*R/2;
        u64 union_need=0;for(int j=first/2;j<last/2;j++)union_need|=needs[j];
        if(union_need) {
            local.active_P4++;
            for(int k=0;k<K;k++) {
                u64 source=src[size_t(g)*K+k]; if(!source)continue;
                local.source_words_examined++;
                if(!(source&union_need))continue;
                local.weight_mask_reads++;
                for(int j=first/2;j<last/2;j++) {
                    u64 events=source&needs[j], pair=(wnz[k]>>(2*j))&3;
                    if(!events||!pair)continue;
                    int n=pc(events);
                    for(int lane=0;lane<2;lane++)if(pair&(1<<lane)) {
                        int r=2*j+lane;
                        int64_t byte=base+int64_t(k)*nr+r-first;
                        for(int b=0;b<3;b++)current[b].use(byte/widths[b],k,events,n,local.counts[b]);
                    }
                }
            }
        }
        for(int b=0;b<3;b++)current[b].flush(local.counts[b]);
        totals[phase]+=local;
        for(int s=0;s<3;s++)if(g==selected_g[s])selected[s][phase]=local;
    }
    Phase sum=totals[0];sum+=totals[1];
    std::ofstream o(argv[8]);
    o<<"{\"shared\":";write_phase(o,totals[0]);o<<",\"tail\":";write_phase(o,totals[1]);
    o<<",\"total\":";write_phase(o,sum);o<<",\"selected_P4\":[";
    for(int s=0;s<3;s++) {if(s)o<<',';o<<"{\"G\":"<<selected_g[s]<<",\"shared\":";
        write_phase(o,selected[s][0]);o<<",\"tail\":";write_phase(o,selected[s][1]);o<<'}';}
    o<<"]}\n";
}
