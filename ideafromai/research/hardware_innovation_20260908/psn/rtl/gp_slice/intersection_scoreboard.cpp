// Memory response models and independent dense-dot oracle only. No NRV packets,
// selected events, W bundles, S values or PSN decisions are supplied to the DUT.
#include "Vgp_slice.h"
#include "verilated.h"
#include <array>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using U=uint64_t;
static U cycle=0;
static uint32_t rng=0x19283047;
static uint32_t random32(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static void need(bool ok,const std::string&why){if(!ok)throw std::runtime_error(why+" at cycle "+std::to_string(cycle));}
template<class T>void read(std::istream&f,T&x){f.read(reinterpret_cast<char*>(&x),sizeof(x));need(bool(f),"case truncated");}
static U field(const uint32_t*x,int bit,int width){
    U v=x[bit/32];if(bit%32+width>32)v|=U(x[bit/32+1])<<32;
    return (v>>(bit%32))&((U(1)<<width)-1);
}
static int sign15(U x){return (x&16384)?int(x)-32768:int(x);}
static int pc(unsigned x){return __builtin_popcount(x);}
struct Case {
    std::string name;uint8_t rank,real,time;
    uint32_t theta[2];uint8_t decode[8];std::vector<uint16_t>program;
    int32_t coeff[10][7],tau[2][10];int8_t W[2][384];uint8_t codes[4][4][384];
    int32_t S[2][4][4][7];uint8_t present[2][4][4][7];uint16_t gates[2][4][4];
};
struct Expected {
    std::array<std::array<U,128>,4> source{};
    std::array<std::array<uint8_t,2048>,2> dense{},compressed{};
    std::vector<unsigned> columns[2];
    struct Request {unsigned address,kind,source;};
    std::vector<Request> sparse_reads[8];
    bool live[4][384]{};
    U events[8]{},commits[8]{};
    std::vector<int> packets[8];
};
static Expected reference(const Case&c){
    Expected e;
    for(int m=0;m<2;m++) {
        e.dense[m].fill(0xa7);e.compressed[m].fill(0x69);
        for(int d=0;d<384;d++) {
            e.dense[m][d]=uint8_t(c.W[m][d]);
            if(c.W[m][d])e.columns[m].push_back(d);
        }
        unsigned count=e.columns[m].size();
        e.compressed[m][0]=count&255;e.compressed[m][1]=count>>8;
        for(unsigned j=0;j<count;j++) {
            unsigned d=e.columns[m][j];
            e.compressed[m][2+3*j]=d&255;e.compressed[m][3+3*j]=d>>8;
            e.compressed[m][4+3*j]=uint8_t(c.W[m][d]);
        }
    }
    for(int k=0;k<4;k++) {
        e.source[k].fill(0xd1a6124380a70c39ULL);
        for(int w=0;w<72;w++)e.source[k][w]=0;
        for(int d=0;d<384;d++)for(int p=0;p<4;p++) {
            unsigned code=c.codes[k][p][d];need(code<8,"bad source code");
            e.live[k][d]|=c.decode[code]!=0;
            for(int b=0;b<3;b++)if(code&(1u<<b)) {
                int bit=d*12+p*3+b;e.source[k][bit/64]|=U(1)<<(bit%64);
            }
        }
    }
    for(int m=0;m<2;m++)for(int k=0;k<4;k++) {
        int id=m*4+k;std::vector<int> retained;
        // Independently construct the two-pointer memory trace from the dense
        // source and W. Only these checks see the oracle; memory responses below
        // depend exclusively on the actual requested byte address.
        unsigned ptr=0;bool header=false,head=false;
        for(unsigned d=0;d<384;d++)if(e.live[k][d]) {
            auto &trace=e.sparse_reads[id];
            if(!header) {trace.push_back({0,1,d});trace.push_back({1,2,d});header=true;}
            while(ptr<e.columns[m].size()) {
                if(!head) {trace.push_back({2+3*ptr,3,d});trace.push_back({3+3*ptr,4,d});head=true;}
                if(e.columns[m][ptr]<d) {ptr++;head=false;continue;}
                if(e.columns[m][ptr]==d) {trace.push_back({4+3*ptr,0,d});ptr++;head=false;}
                break;
            }
        }
        for(int d=0;d<384;d++)if(e.live[k][d] && c.W[m][d])retained.push_back(d);
        for(size_t first=0;first<retained.size();first+=4) {
            int n=std::min<size_t>(4,retained.size()-first);e.packets[id].push_back(n);
            uint32_t union_dest=0;
            for(int a=0;a<n;a++)for(int p=0;p<4;p++) {
                uint32_t mask=c.decode[c.codes[k][p][retained[first+a]]];
                e.events[id]+=pc(mask);union_dest|=mask<<(p*7);
            }
            e.commits[id]+=pc(union_dest);
        }
        for(int p=0;p<4;p++) {
            int64_t s[7]{};bool touched[7]{};
            for(int d=0;d<384;d++)for(int r=0;r<c.rank;r++)if(c.decode[c.codes[k][p][d]]&(1u<<r)) {
                s[r]+=c.W[m][d];touched[r]|=c.W[m][d]!=0;
            }
            for(int r=0;r<7;r++) {
                need(s[r]==c.S[m][k][p][r],"independent C++ vs NumPy S");
                need(touched[r]==bool(c.present[m][k][p][r]),"independent touched");
            }
            unsigned gate=0;
            for(int t=0;t<10;t++) {
                int64_t u=0;for(int r=0;r<c.rank;r++)u+=int64_t(c.coeff[t][r])*s[r];
                if(u>=c.tau[m][t])gate|=1u<<t;
            }
            need(gate==c.gates[m][k][p],"independent C++ vs NumPy gates");
        }
    }
    return e;
}
static void edge(Vgp_slice&d){d.clk=0;d.eval();d.clk=1;d.eval();cycle++;}
static void clear_io(Vgp_slice&d){
    d.start_valid=0;d.cfg_uop_valid=0;d.cfg_tau_valid=0;d.src_req_ready=0;d.src_rsp_valid=0;
    d.w_req_ready=0;d.w_rsp_valid=0;d.w_rsp_data=0;d.out_ready=0;d.done_ready=0;
    for(int w=0;w<8;w++)d.src_rsp_data[w]=0;
}
struct MemoryPort {bool pending=false;U due=0,value=0;unsigned address=0;};
struct Stats {
    U cycles=0,last_gate=0,source_reads=0,w_reads=0,w_zero=0,packets=0,events=0,commits=0;
    U w_metadata=0,w_values=0,compare_pe_beats=0,advance_weight=0,advance_nrv=0;
    U source_req_wait=0,w_req_wait=0,w_response_wait=0,output_wait=0,barrier_wait=0,psn_issues=0;
    unsigned nr_peak=0,w_pending_peak=0,w_accept_peak=0,source_accept_peak=0;
    U source_per_id[4]{},weight_per_tile[2]{},first_done_tile[2]{};
};
static Stats run(Vgp_slice&d,const Case&c,const Expected&e,int intersection,int reduce,int stress,unsigned tag) {
    clear_io(d);need(d.start_ready,"previous task was not retired");
    for(size_t i=0;i<c.program.size();i++) {
        d.cfg_uop_valid=1;d.cfg_uop_address=i;d.cfg_uop_data=c.program[i];edge(d);
    }
    d.cfg_uop_valid=0;
    for(int m=0;m<2;m++)for(int t=0;t<10;t++) {
        d.cfg_tau_valid=1;d.cfg_tau_tile=m;d.cfg_tau_time=t;d.cfg_tau_data=uint32_t(c.tau[m][t])&0xffffff;edge(d);
    }
    d.cfg_tau_valid=0;d.start_reduce=reduce;d.start_intersection=intersection;
    d.start_rank=c.rank;d.start_program_length=c.program.size();
    U table=0;for(int k=0;k<8;k++)table|=U(c.decode[k])<<(k*7);
    d.start_decode=table;d.start_theta_payload=U(c.theta[0])|(U(c.theta[1])<<32);d.start_tag=tag;
    d.start_valid=1;edge(d);d.start_valid=0;U begin=cycle;
    std::array<MemoryPort,4>src{},weight{};Stats s;
    unsigned src_word[4]{};bool w_seen[2][4][384]{};unsigned packet_seen[8]{};
    unsigned trace_seen[8]{};
    U events[8]{},commits[8]{};int last_S[8][28]{};bool last_valid[8][28]{};
    bool s_checked[8]{},gate_seen[4][4]{},consumer_seen[4]{};
    U syn_at[8]{};unsigned gates=0,done_hold=0;bool done_seen=false;
    unsigned old_src_hold=0,old_w_hold=0,old_kind=0,old_w_owner=0;U old_w_addr=0;uint32_t old_src_addr=0;
    bool output_hold=false;U old_theta=0;uint32_t old_gates=0;unsigned old_owner=0,old_p=0,old_tag=0;
    for(unsigned step=0;step<200000;step++) {
        d.clk=0;d.src_req_ready=0;d.src_rsp_valid=0;d.w_req_ready=0;d.w_rsp_valid=0;d.w_rsp_data=0;
        uint32_t rand=random32();
        for(int k=0;k<4;k++) {
            if(!src[k].pending && (!stress || ((rand>>(k*3))&7)!=0))d.src_req_ready|=1u<<k;
            if(src[k].pending && src[k].due<=cycle) {
                d.src_rsp_valid|=1u<<k;d.src_rsp_data[k*2]=src[k].value;d.src_rsp_data[k*2+1]=src[k].value>>32;
            }
            if(!weight[k].pending && (!stress || ((rand>>(k*4+8))&3)!=0))d.w_req_ready|=1u<<k;
            if(weight[k].pending && weight[k].due<=cycle) {
                d.w_rsp_valid|=1u<<k;d.w_rsp_data|=uint32_t(weight[k].value&255)<<(8*k);
            }
        }
        // Bursts longer than one complete output beat, plus deterministic gaps.
        d.out_ready=!stress || ((cycle+tag)%19>=7 && (rand&7)!=0);
        d.done_ready=done_seen && done_hold>=5;d.eval();
        need(!d.overflow,c.name+" arithmetic overflow");
        for(int k=0;k<4;k++) {
            if(old_src_hold&(1u<<k))need((d.src_req_valid&(1u<<k))
                && ((d.src_req_address>>(k*7))&127)==((old_src_addr>>(k*7))&127),"source request changed while stalled");
            if(old_w_hold&(1u<<k))need((d.w_req_valid&(1u<<k))
                && ((d.w_req_full_address>>(k*11))&2047)==((old_w_addr>>(k*11))&2047)
                && ((d.dbg_w_kind>>(k*3))&7)==((old_kind>>(k*3))&7)
                && ((d.dbg_w_owner>>(k*2))&3)==((old_w_owner>>(k*2))&3),"W request changed while stalled");
        }
        if(output_hold)need(d.out_valid && d.out_gates==old_gates && d.out_theta_payload==old_theta
            && d.out_source_id==old_owner && d.out_position==old_p && d.out_tag==old_tag,"output changed under backpressure");
        if(done_seen)need(d.done_valid,"done_valid withdrawn under backpressure");
        old_src_hold=d.src_req_valid&~d.src_req_ready;old_src_addr=d.src_req_address;
        old_w_hold=d.w_req_valid&~d.w_req_ready;old_w_addr=d.w_req_full_address;
        old_kind=d.dbg_w_kind;old_w_owner=d.dbg_w_owner;
        output_hold=d.out_valid&&!d.out_ready;
        old_gates=d.out_gates;old_theta=d.out_theta_payload;old_owner=d.out_source_id;old_p=d.out_position;old_tag=d.out_tag;
        s.source_req_wait+=pc(old_src_hold);s.w_req_wait+=pc(old_w_hold);
        s.w_response_wait+=pc(d.w_rsp_valid&~d.w_rsp_ready);s.output_wait+=output_hold;
        s.compare_pe_beats+=pc(d.dbg_compare);s.advance_weight+=pc(d.dbg_advance_weight);
        s.advance_nrv+=pc(d.dbg_advance_nrv);
        s.w_pending_peak=std::max(s.w_pending_peak,unsigned(pc(d.dbg_w_pending)));
        unsigned w_accept=0,src_accept=0;
        for(int q=0;q<4;q++) {
            if((d.src_req_valid&d.src_req_ready)&(1u<<q)) {
                unsigned addr=(d.src_req_address>>(q*7))&127;
                need(!src[q].pending && addr==src_word[q] && addr<72,"source physical address/order");
                src[q]={true,cycle+1+unsigned(stress?((rand>>(q+1))&3):0),e.source[q][addr],addr};
                src_word[q]++;s.source_reads++;s.source_per_id[q]++;src_accept++;
            }
            if((d.src_rsp_valid&d.src_rsp_ready)&(1u<<q)) {
                need(src[q].pending,"unowned source response");src[q].pending=false;
            }
            if((d.w_req_valid&d.w_req_ready)&(1u<<q)) {
                unsigned addr=(d.w_req_full_address>>(q*11))&2047,k=(d.dbg_w_owner>>(q*2))&3,m=q/2;
                unsigned kind=(d.dbg_w_kind>>(q*3))&7,source=(d.dbg_source_index>>(k*9))&511;
                need(!weight[q].pending,"W port transaction ownership");
                if(intersection) {
                    unsigned id=m*4+k;
                    need(trace_seen[id]<e.sparse_reads[id].size(),"unexpected compressed W request");
                    auto expected=e.sparse_reads[id][trace_seen[id]++];
                    need(addr==expected.address && kind==expected.kind && source==expected.source,
                         c.name+" two-pointer actual address/kind/source trace");
                } else need(kind==0 && addr==source && addr<384,"dense W address/kind/source");
                if(kind==0) {
                    unsigned column=intersection?e.columns[m][(addr-4)/3]:addr;
                    need(column==source && e.live[k][column] && !w_seen[m][k][column],"W value owner/duplicate");
                    need(!intersection || c.W[m][column]!=0,"zero W fetched as compressed value");
                    w_seen[m][k][column]=true;s.w_values++;
                } else s.w_metadata++;
                unsigned delay=stress ? 1+((m==((tag/4)&1))?5:0)+((rand>>(q*3))&3):1;
                weight[q]={true,cycle+delay,intersection?e.compressed[m][addr]:e.dense[m][addr],addr};
                s.w_reads++;s.weight_per_tile[m]++;w_accept++;
            }
            if((d.w_rsp_valid&d.w_rsp_ready)&(1u<<q)) {
                need(weight[q].pending,"unowned W response");weight[q].pending=false;
            }
        }
        s.w_accept_peak=std::max(s.w_accept_peak,w_accept);s.source_accept_peak=std::max(s.source_accept_peak,src_accept);
        s.w_zero+=pc(d.dbg_zero_weight);
        for(int i=0;i<8;i++) {
            int m=i/4,k=i%4;
            unsigned occupancy=(d.dbg_nr_occupancy>>(i*2))&3;
            need(occupancy<=2,"more than two private NR4 packets");s.nr_peak=std::max(s.nr_peak,occupancy);
            if(d.dbg_packet&(1u<<i)) {
                unsigned n=(d.dbg_packet_sources>>(i*3))&7;
                need(packet_seen[i]<e.packets[i].size() && n==unsigned(e.packets[i][packet_seen[i]]),"private W0 compaction/last packet size");
                packet_seen[i]++;s.packets++;
            }
            unsigned members=(d.dbg_members>>(i*4))&15;
            events[i]+=pc(members);s.events+=pc(members);
            if(d.dbg_commit&(1u<<i)) {
                unsigned addr=(d.dbg_s_address>>(i*5))&31;
                need(addr<28,"illegal S address");
                last_S[i][addr]=sign15(field(d.dbg_s_value,i*15,15));last_valid[i][addr]=true;
                commits[i]++;s.commits++;
            }
            if(d.dbg_syn_done&(1u<<i))if(!s_checked[i]) {
                syn_at[i]=cycle;s_checked[i]=true;
                need(events[i]==e.events[i] && commits[i]==e.commits[i]
                     && packet_seen[i]==e.packets[i].size(),"event/commit/packet conservation at S seal");
                for(int p=0;p<4;p++)for(int r=0;r<7;r++) {
                    need(last_S[i][p*7+r]==c.S[m][k][p][r],c.name+" S value at seal");
                    need(last_valid[i][p*7+r]==bool(c.present[m][k][p][r]),"S valid-zero / empty state");
                }
            }
        }
        for(int k=0;k<4;k++) {
            if(d.dbg_consumer_go&(1u<<k)) {
                need(s_checked[k] && s_checked[k+4] && !consumer_seen[k],"two-tile completion barrier");
                consumer_seen[k]=true;s.barrier_wait+=syn_at[k]>syn_at[k+4]?syn_at[k]-syn_at[k+4]:syn_at[k+4]-syn_at[k];
                if(syn_at[k]!=syn_at[k+4])s.first_done_tile[syn_at[k]<syn_at[k+4]?0:1]++;
            }
            bool a=(d.dbg_psn>>k)&1,b=(d.dbg_psn>>(k+4))&1;
            need(a==b,"paired PSN phase diverged");
            if(a) {
                need(consumer_seen[k],"PSN before completed barrier");
                need(((d.dbg_pc>>(k*8))&255)==((d.dbg_pc>>((k+4)*8))&255),"broadcast program PC diverged");
            }
        }
        s.psn_issues+=pc(d.dbg_psn);
        if(d.out_valid && d.out_ready) {
            unsigned k=d.out_source_id,p=d.out_position;
            need(k<4 && p<4 && !gate_seen[k][p] && consumer_seen[k],"gate identity/duplicate");
            need(d.out_gates==(unsigned(c.gates[1][k][p])<<10|c.gates[0][k][p]),c.name+" full T10 gates");
            need(d.out_theta_payload==(U(c.theta[0])|(U(c.theta[1])<<32)) && d.out_tag==tag,"theta payload / task tag changed");
            gate_seen[k][p]=true;gates++;s.last_gate=cycle-begin+1;
        }
        bool completed=d.done_valid&&d.done_ready;
        if(d.done_valid) {done_seen=true;done_hold++;need(gates==16,"done before all accepted gates");}
        edge(d);
        if(completed) {
            s.cycles=cycle-begin;
            for(int k=0;k<4;k++) {
                need(src_word[k]==72 && !src[k].pending && !weight[k].pending,"unfinished memory transaction");
                for(int m=0;m<2;m++) {
                    for(int a=0;a<384;a++)need(w_seen[m][k][a]==(e.live[k][a] && (!intersection || c.W[m][a]!=0)),"W address coverage");
                    if(intersection)need(trace_seen[m*4+k]==e.sparse_reads[m*4+k].size(),"incomplete compressed W trace");
                }
            }
            need(s.w_reads==s.w_metadata+s.w_values && (!intersection || s.w_zero==0),"W byte/value conservation");
            need(s.psn_issues==U(8*4*c.program.size()),"PSN program issue conservation");
            need(d.start_ready && !d.busy && !d.out_valid,"S/output ownership not released");
            return s;
        }
    }
    throw std::runtime_error(c.name+" timeout");
}
int main(int argc,char**argv){
    try {
        Verilated::commandArgs(argc,argv);need(argc==3,"usage cases.bin results.tsv");
        std::ifstream f(argv[1],std::ios::binary);char magic[4];f.read(magic,4);need(std::string(magic,4)=="GPS1","magic");
        uint32_t count;read(f,count);std::ofstream out(argv[2]);
        out<<"case\treal\ttime\tintersection\treduce\tstress\tcycles_through_done_handshake\tlast_gate\tsource_reads\tw_reads\tw_zero\tpackets\tevents\tcommits\tsource_req_wait\tw_req_wait\tw_response_wait\toutput_wait\tbarrier_wait\tpsn_issues\tnr_peak\tw_pending_peak\tw_accept_peak\tsource_accept_peak\tfirst_tile0\tfirst_tile1\tw_metadata_reads\tw_value_reads\tcompare_active_PE_beats\tweight_pointer_advances_without_value\tnrv_skips\tnnz_tile0\tnnz_tile1\timage_bytes_tile0\timage_bytes_tile1\n";
        Vgp_slice d;clear_io(d);d.rst_n=0;for(int i=0;i<3;i++)edge(d);d.rst_n=1;edge(d);
        U tasks=0,total_gate_bits=0,return_wait=0;unsigned nr_peak=0,wpeak=0;U first[2]{};
        for(uint32_t n=0;n<count;n++) {
            Case c;uint16_t len;read(f,len);c.name.resize(len);f.read(&c.name[0],len);
            read(f,c.rank);read(f,c.real);read(f,c.time);read(f,c.theta);read(f,c.decode);
            read(f,len);c.program.resize(len);f.read(reinterpret_cast<char*>(c.program.data()),len*2);
            read(f,c.coeff);read(f,c.tau);read(f,c.W);read(f,c.codes);read(f,c.S);read(f,c.present);read(f,c.gates);
            Expected e=reference(c);
            for(int intersection=0;intersection<2;intersection++)for(int reduce=0;reduce<2;reduce++)for(int stress=0;stress<2;stress++) {
                Stats s=run(d,c,e,intersection,reduce,stress,unsigned(++tasks));total_gate_bits+=320;
                return_wait+=s.w_response_wait;nr_peak=std::max(nr_peak,s.nr_peak);wpeak=std::max(wpeak,s.w_accept_peak);
                first[0]+=s.first_done_tile[0];first[1]+=s.first_done_tile[1];
                out<<c.name<<'\t'<<unsigned(c.real)<<'\t'<<unsigned(c.time)<<'\t'<<intersection<<'\t'<<reduce<<'\t'<<stress
                   <<'\t'<<s.cycles<<'\t'<<s.last_gate<<'\t'<<s.source_reads<<'\t'<<s.w_reads<<'\t'<<s.w_zero
                   <<'\t'<<s.packets<<'\t'<<s.events<<'\t'<<s.commits<<'\t'<<s.source_req_wait<<'\t'<<s.w_req_wait
                   <<'\t'<<s.w_response_wait<<'\t'<<s.output_wait<<'\t'<<s.barrier_wait<<'\t'<<s.psn_issues
                   <<'\t'<<s.nr_peak<<'\t'<<s.w_pending_peak<<'\t'<<s.w_accept_peak<<'\t'<<s.source_accept_peak
                   <<'\t'<<s.first_done_tile[0]<<'\t'<<s.first_done_tile[1]
                   <<'\t'<<s.w_metadata<<'\t'<<s.w_values<<'\t'<<s.compare_pe_beats<<'\t'<<s.advance_weight<<'\t'<<s.advance_nrv
                   <<'\t'<<e.columns[0].size()<<'\t'<<e.columns[1].size()
                   <<'\t'<<(intersection?2+3*e.columns[0].size():384)<<'\t'<<(intersection?2+3*e.columns[1].size():384)<<'\n';
            }
            std::cout<<"PASS "<<c.name<<" eight configurations\n";
        }
        need(nr_peak==2 && return_wait>0 && wpeak==4,"missing full-buffer/return-backpressure/four-port concurrency coverage");
        need(first[0]>0 && first[1]>0,"missing opposite tile completion orders");
        std::cout<<"PASS tasks="<<tasks<<" gate_bits="<<total_gate_bits<<" cycles="<<cycle
                 <<" response_backpressure="<<return_wait<<" nr_peak="<<nr_peak<<" W_accept_peak="<<wpeak<<'\n';
        return 0;
    } catch(const std::exception&e) {std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
