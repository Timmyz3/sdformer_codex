// Independent numerical reference; never calls the Machine or supplies costs.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
static float tf32(float x){uint32_t u;std::memcpy(&u,&x,4);u=(u+4095+((u>>13)&1))&0xffffe000u;std::memcpy(&x,&u,4);return x;}
extern "C" void preview(const uint8_t*g,int sh,int sw,int sy0,int sx0,int h,int w,int oy,int ox,
    const float*u,const float*v,const float*scale,const float*bias,const float*A,const float*b,float theta,
    float*z,float*raw,float*bn,uint8_t*gate,float*margin) {
    for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
        for(int t=0;t<10;++t) {
            float latent[32]={};
            for(int c=0;c<96;++c)for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx) {
                int sy=oy+y+ky-1,sx=ox+x+kx-1;
                if(sy<0||sy>=240||sx<0||sx>=320)continue;
                if(g[((t*96+c)*sh+sy-sy0)*sw+sx-sx0]) {
                    int k=c*9+ky*3+kx;
                    for(int r=0;r<32;++r)latent[r]=std::fma(1.0f,tf32(u[k*32+r]),latent[r]);
                }
            }
            for(int r=0;r<32;++r)z[((t*32+r)*h+y)*w+x]=latent[r];
            for(int c=0;c<96;++c) {
                float value=0;
                for(int r=0;r<32;++r)value=std::fma(tf32(latent[r]),tf32(v[r*96+c]),value);
                int index=((t*96+c)*h+y)*w+x;raw[index]=value;
                float scaled=value*scale[c];bn[index]=scaled+bias[c];
            }
        }
        for(int c=0;c<96;++c)for(int t=0;t<10;++t) {
            float membrane=0;
            for(int s=0;s<10;++s)if(A[t*10+s]!=0)
                membrane=std::fma(A[t*10+s],bn[((s*96+c)*h+y)*w+x],membrane);
            membrane=membrane+b[t];membrane=membrane-theta;
            int index=((t*96+c)*h+y)*w+x;gate[index]=membrane>=0;margin[index]=membrane;
        }
    }
}
