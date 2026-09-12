// Ordinary fixed4x4/H32 native mapping, using only imported Engine operations.
constexpr int GATE_STATE=110720,DIR_HEADER=110592,NATIVE_OUTPUT=126272;
struct NativeCost {
    uint64_t gate_bytes=0,raw_bytes=0,weight_bytes=0,gate_address=0;
    uint64_t directory_decode=0,directory_entries=0,active_adds=0,spill_address=0;
    uint64_t coefficient_slots=0,input_slots=0,directory_slots=0,compute_slots=0,output_slots=0;
};
static std::array<char,8> word(Engine& e,int address) {
    e.tick(address);e.tick();return e.sword;
}
static void spill(Engine& e,int r,float* external) {
    e.state_store(r,NATIVE_OUTPUT);V v=e.gather(NATIVE_OUTPUT);e.transfer5();
    std::memcpy(external,v.data(),32);
}
static void gate_tile(Engine&e,const std::vector<uint8_t>& gate,int oy,int ox,NativeCost& cost) {
    const std::array<char,32> zero{};
    for(int y=0;y<9;++y)for(int x=0;x<9;++x) {
        int sy=2*oy-1+y,sx=2*ox-1+x;e.tick();++cost.gate_address;
        bool valid=sy>=0&&sy<240&&sx>=0&&sx<320;
        for(int j=0;j<192;j+=32) {
            const char* src=zero.data();
            if(valid){src=reinterpret_cast<const char*>(gate.data())+(sy*320+sx)*192+j;e.transfer5();cost.gate_bytes+=32;}
            for(int b=0;b<4;++b)e.tick(-1,GATE_STATE+(y*9+x)*192+j+b*8,src+b*8);
        }
    }
}
static void directories(Engine&e,NativeCost& cost,int block) {
    for(int dy=0;dy<4;++dy)for(int dx=0;dx<4;dx+=2) {
        int group=dy*2+dx/2,n=0;uint32_t live=0;
        for(int h4=0;h4<24;++h4) {
            for(int off=0;off<9;++off) {
                const int ky=off/3,kx=off%3;
                for(int ip=0;ip<2;++ip) {
                    int address=GATE_STATE+((2*dy+ky)*9+2*(dx+ip)+kx)*192+h4*8;
                    auto raw=word(e,address);std::array<uint16_t,4> lanes{};std::memcpy(lanes.data(),raw.data(),8);
                    for(int c=0;c<4;++c)e.load[ip*4+c]=float(lanes[c]);
                }
                // Nine existing RF vectors retain the18 SR64 source words.
                e.op(LOAD,64+off);e.wait(64+off);
            }
            for(int c=0;c<4;++c)for(int off=0;off<9;++off) {
                e.wait(64+off);e.tick();++cost.directory_decode;
                uint32_t a=uint32_t(e.rf[64+off][c]),b=uint32_t(e.rf[64+off][4+c]);
                assert(a<1024&&b<1024);uint32_t mask=a|(b<<10);
                if(mask){std::array<uint32_t,2> item{uint32_t((h4*4+c)*9+off),mask};
                    e.tick(-1,block*55296+group*6912+n*8,reinterpret_cast<const char*>(item.data()));
                    ++n;++cost.directory_entries;live|=mask;}
            }
        }
        assert(n<=864);std::array<uint32_t,2> header{uint32_t(n),live};
        e.tick(-1,DIR_HEADER+block*64+group*8,reinterpret_cast<const char*>(header.data()));
    }
}
static std::vector<float> native(Engine&e,const std::vector<uint8_t>&gate,const std::vector<float>&weights,NativeCost&cost) {
    assert(gate.size()==240*320*96*2&&weights.size()==96*864);
    assert(NATIVE_OUTPUT+32<=131072&&GATE_STATE+15552==NATIVE_OUTPUT);
    std::vector<float> output(192000*96);
    // Complete two directories are retained across all three true H32 loads.
    for(int batch=0;batch<600;++batch) {
        int oy[2],ox[2];
        for(int block=0;block<2;++block) {
            int tile=batch*2+block;oy[block]=(tile/40)*4;ox[block]=(tile%40)*4;
            uint64_t begin=e.t;gate_tile(e,gate,oy[block],ox[block],cost);cost.input_slots+=e.t-begin;
            begin=e.t;directories(e,cost,block);cost.directory_slots+=e.t-begin;
        }
        // Shadow directory bytes are assertion-only reference data, not used
        // for execution. Actual issued reads remain in Engine state SRAM.
        auto directory_reference=e.state;
        for(int h0=0;h0<96;h0+=32) {
            uint64_t begin=e.t;std::vector<float> tile(864*32);
            for(int k=0;k<864;++k)for(int h=0;h<32;++h)tile[k*32+h]=weights[(h0+h)*864+k];
            e.coefficients(tile);cost.weight_bytes+=tile.size()*4;cost.coefficient_slots+=e.t-begin;
            for(int block=0;block<2;++block)for(int group=0;group<8;++group) {
                begin=e.t;auto head=word(e,DIR_HEADER+block*64+group*8);uint32_t n;std::memcpy(&n,head.data(),4);
                for(int r=0;r<80;++r)e.op(ZERO,r);
                for(uint32_t i=0;i<n;++i) {
                    auto raw=word(e,block*55296+group*6912+i*8);std::array<uint32_t,2> item{};std::memcpy(item.data(),raw.data(),8);
                    const auto k=item[0],mask=item[1];assert(k<864);e.tick();
                    for(int hg=0;hg<4;++hg) {
                        e.constant(80,(k*32+hg*8)*4);
                        for(int tp=0;tp<20;++tp)if(mask&(1u<<tp)) {
                            e.op(ADD,tp*4+hg,tp*4+hg,80);++cost.active_adds;
                        }
                    }
                }
                cost.compute_slots+=e.t-begin;begin=e.t;
                int dy=group/2,dx=(group%2)*2;
                for(int ip=0;ip<2;++ip)for(int t=0;t<10;++t)for(int hg=0;hg<4;++hg) {
                    size_t offset=((t*120+oy[block]+dy)*160+ox[block]+dx+ip)*96+h0+hg*8;
                    e.tick();++cost.spill_address;
                    spill(e,(ip*10+t)*4+hg,output.data()+offset);cost.raw_bytes+=32;
                }
                cost.output_slots+=e.t-begin;
            }
            assert(std::memcmp(directory_reference.data(),e.state.data(),110720)==0);
        }
        if((batch+1)%200==0)std::cerr<<"two-block batch "<<batch+1<<"/600 complete at "<<e.t<<" slots\n";
    }
    assert(cost.weight_bytes==1800ull*110592&&cost.raw_bytes==73728000);
    return output;
}

// No Engine calls, no model state/directories reused: an independent complete
// output-position/time/K reference with original source indexing and fmaf.
static std::vector<float> native_reference(const std::vector<uint8_t>&gate,const std::vector<float>&w) {
    std::vector<float> result(192000*96);
    for(int t=0;t<10;++t)for(int y=0;y<120;++y)for(int x=0;x<160;++x) {
        float* out=result.data()+((t*120+y)*160+x)*96;
        for(int c=0;c<96;++c)for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx) {
            int sy=2*y+ky-1,sx=2*x+kx-1;if(sy<0||sy>=240||sx<0||sx>=320)continue;
            uint16_t g;std::memcpy(&g,gate.data()+((sy*320+sx)*96+c)*2,2);
            if(g&(1u<<t)){int k=c*9+ky*3+kx;for(int h=0;h<96;++h)out[h]=std::fma(1.0f,w[h*864+k],out[h]);}
        }
    }
    return result;
}
