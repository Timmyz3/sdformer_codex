#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
extern "C" void direct(const uint8_t* a,const int64_t* w,int64_t* y,int M,int K,int N){
  std::fill(y,y+(int64_t)M*N,0);
  for(int m=0;m<M;++m) for(int k=0;k<K;++k) if(a[(int64_t)m*K+k]){
    const int64_t* src=w+(int64_t)k*N;int64_t* dst=y+(int64_t)m*N;
    for(int n=0;n<N;++n)dst[n]+=src[n];
  }
}
extern "C" void forest(const int16_t* parent,const uint16_t* residual,const uint16_t* order,
 const int32_t* perm,const int64_t* w,int64_t* y,int M,int K,int N){
  std::fill(y,y+(int64_t)M*N,0);std::vector<int64_t> local(256*N);int64_t planoff=0;
  for(int mb=0;mb<M;mb+=256){int rows=std::min(256,M-mb);
    for(int kb=0;kb<K;kb+=16){
      for(int ix=0;ix<rows;++ix){int r=order[planoff+ix],p=parent[planoff+r];
        int64_t* dst=local.data()+(int64_t)r*N;
        if(p>=0)std::memcpy(dst,local.data()+(int64_t)p*N,N*sizeof(int64_t));
        else std::fill(dst,dst+N,0);
        unsigned bits=residual[planoff+r];
        while(bits){int b=__builtin_ctz(bits);bits&=bits-1;
          const int64_t* src=w+(int64_t)perm[kb+b]*N;
          for(int n=0;n<N;++n)dst[n]+=src[n];
        }
        int64_t* global=y+(int64_t)(mb+r)*N;
        for(int n=0;n<N;++n)global[n]+=dst[n];
      }
      planoff+=rows;
    }
  }
}
