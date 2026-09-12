// Arithmetic companion to onepass.cpp; no scheduler, no hardware timing claim.
// Two stripes per moment; paired256 tree; explicit fma and separate mul/add.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
using A=std::array<float,96>;
extern "C" int moments(const float*x,int n,const float*gamma,const float*beta,float eps,float*out) {
    if(n!=192000)return 1;
    std::array<A,2> s{},q{};std::array<std::array<A,2>,10> tree{};unsigned occupied=0;
    for(int p=0;p<n;++p){
        int b=p%2;for(int c=0;c<96;++c){float v=x[p*96+c];s[b][c]+=v;q[b][c]=std::fma(v,v,q[b][c]);}
        if((p+1)%256==0){
            for(int c=0;c<96;++c){s[0][c]+=s[1][c];q[0][c]+=q[1][c];}
            int level=0;while(occupied&(1u<<level)){
                for(int c=0;c<96;++c){s[0][c]+=tree[level][0][c];q[0][c]+=tree[level][1][c];}
                occupied&=~(1u<<level);++level;
            }
            tree[level][0]=s[0];tree[level][1]=q[0];occupied|=1u<<level;
            s={};q={};
        }
    }
    A total{},totalq{};
    for(int level=0;level<10;++level)if(occupied&(1u<<level))for(int c=0;c<96;++c){total[c]+=tree[level][0][c];totalq[c]+=tree[level][1][c];}
    const float reciprocal=1.0f/192000.0f;
    for(int c=0;c<96;++c){
        float mean=total[c]*reciprocal,second=totalq[c]*reciprocal;
        float meansq=mean*mean;float var=second-meansq;
        if(!(var>=0)||!std::isfinite(var))return 2;
        float ve=var+eps,half=ve*0.5f;uint32_t bits;std::memcpy(&bits,&ve,4);
        bits=0x5f3759dfu-(bits>>1);float inv;std::memcpy(&inv,&bits,4);
        for(int it=0;it<3;++it){float square=inv*inv;float rem=std::fma(-half,square,1.5f);inv=inv*rem;}
        float scale=inv*gamma[c],ms=mean*scale,bias=beta[c]-ms;
        out[c]=mean;out[96+c]=var;out[192+c]=inv;out[288+c]=scale;out[384+c]=bias;
    }
    return 0;
}
extern "C" void normalize(const float*x,int n,const float*stats,float*out){
    for(int p=0;p<n;++p)for(int c=0;c<96;++c){float v=x[p*96+c]*stats[288+c];out[p*96+c]=v+stats[384+c];}
}

extern "C" int centered_moments(const float*x,int n,const float*gamma,const float*beta,float eps,float*out) {
    if(n!=192000)return 1;
    A mean{},var{};const float reciprocal=1.0f/192000.0f;
    for(int phase=0;phase<2;++phase){
        std::array<A,4> acc{};std::array<A,10> tree{};unsigned occupied=0;
        for(int p=0;p<n;++p){
            int branch=p%4;
            for(int c=0;c<96;++c){
                if(phase==0)acc[branch][c]+=x[p*96+c];
                else{float centered=x[p*96+c]-mean[c];acc[branch][c]=std::fma(centered,centered,acc[branch][c]);}
            }
            if((p+1)%256==0){
                for(int b=1;b<4;++b)for(int c=0;c<96;++c)acc[0][c]+=acc[b][c];
                int level=0;while(occupied&(1u<<level)){
                    for(int c=0;c<96;++c)acc[0][c]+=tree[level][c];
                    occupied&=~(1u<<level);++level;
                }
                tree[level]=acc[0];occupied|=1u<<level;acc={};
            }
        }
        A total{};for(int level=0;level<10;++level)if(occupied&(1u<<level))for(int c=0;c<96;++c)total[c]+=tree[level][c];
        for(int c=0;c<96;++c)(phase==0?mean[c]:var[c])=total[c]*reciprocal;
    }
    for(int c=0;c<96;++c){
        float ve=var[c]+eps,half=ve*0.5f;uint32_t bits;std::memcpy(&bits,&ve,4);
        bits=0x5f3759dfu-(bits>>1);float inv;std::memcpy(&inv,&bits,4);
        for(int it=0;it<3;++it){float square=inv*inv;float rem=std::fma(-half,square,1.5f);inv=inv*rem;}
        float scale=inv*gamma[c],ms=mean[c]*scale,bias=beta[c]-ms;
        out[c]=mean[c];out[96+c]=var[c];out[192+c]=inv;out[288+c]=scale;out[384+c]=bias;
    }
    return 0;
}
