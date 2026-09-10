// Paper-guided CPTB/NRV synaptic service reference; not the official artifact.
// Timing assumptions and unclosed interfaces are documented by the Python driver.
#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace std;
using I = long long;
constexpr int MT=8,KP=8,NR=4;
constexpr I INF=(1LL<<60);
struct Row { int d; array<uint32_t,7> masks{}; };
struct Sub { int part,r; bool last; int encoding_peak_bytes=0; vector<Row> rows; };
struct Packet { int selected; array<uint32_t,7> masks; };
struct PE {
  bool filling=false;
  int nfill=0, selected=0;
  array<uint32_t,7> umask{},touched{};
  deque<Packet> ready;
  I exec=INF, drain=0, done=INF;
  bool can_fetch() const {
    return filling || int(ready.size())+int(exec!=INF)<2;
  }
  int occupancy() const {return int(filling)+int(ready.size())+int(exec!=INF);}
};
struct Stream {
  int sub=0,row=0, pending_d=-1;
  array<uint32_t,7> pending_masks{};
  I response=INF, control=1;
  bool finished=false, draining=false;
};
struct Stats {
  I beats=0, core_beats=0, groups=0, event_steps=0, jumped=0;
  I local_w_reads=0, zero_w_reads=0, selected=0, packets=0;
  I partial_adds=0,state_writes=0,state_reads=0,initial_writes=0;
  I shared_nrv_reads=0,arbiter_denied_requests=0,bank_backpressure=0;
  I synchronization_wait=0, pe_fetch_blocked=0, pe_exec_idle=0;
  I boundary_words=0, boundary_pe_beats=0;
  I consumer_source_words=0,consumer_output_bits=0,consumer_csd_issues=0,consumer_threshold_inits=0;
  I consumer_PE_service_beats=0,consumer_tau_scalar_read_bytes=0;
  I consumer_restore_loads=0,consumer_restore_add_stores=0;
  I max_buffers=0,max_w_inflight=0,max_source_or_W_inflight=0, peak_nrv_plane_bytes=0;
  I bus_beats=0,bus_input_bytes=0,bus_weight_bytes=0,bus_tau_bytes=0,bus_program_bytes=0,bus_output_bytes=0;
  I input_decode_beats=0,input_dense_rows_scanned=0,local_weight_write_beats=0,local_tau_preload_beats=0;
  I finite_compute_beats=0,source_buffer_peak_bytes=0,source_NR_read_requests=0,finite_wave_count=0;
  I full_output_group_min=INF,full_output_group_max=0;
  I input_rows=0,input_submatrices=0,input_payload_bits=0;
  I group_min=INF,group_max=0;
};
static int pc(uint32_t x) {return __builtin_popcount(x);}

void receive(PE &p, int w, const array<uint32_t,7>& masks, bool last, Stats &s) {
  if(!p.filling) {
    if(!p.can_fetch()) throw runtime_error("response without reserved buffer");
    p.filling=true;p.nfill=0;p.selected=0;p.umask.fill(0);
  }
  if(w!=0) {p.nfill++;for(int r=0;r<7;r++){p.selected+=pc(masks[r]);p.umask[r]|=masks[r];}}
  else s.zero_w_reads++;
  if(p.nfill==NR || last) {
    if(p.nfill) p.ready.push_back({p.selected,p.umask});
    p.filling=false;p.nfill=0;p.selected=0;p.umask.fill(0);
  }
  s.max_buffers=max(s.max_buffers,I(p.occupancy()));
  if(p.occupancy()>2) throw runtime_error("double buffer overflow");
}

using Touch=array<array<array<uint32_t,7>,KP>,MT>;
I simulate_group(const vector<vector<Sub>>& subs,const vector<int8_t>&W,
                 int C,int P,int R,int first,Stats&s,bool count_output_drain,int consumer_per_p,
                 Touch* carry=nullptr,int fetch_latency=1,bool prologue=true,int restore_loads=0,int restore_adds=0) {
  array<Stream,KP> st;
  array<array<PE,KP>,MT> pe;
  I now=(consumer_per_p&&prologue)?15:0;int completed=0,rr=0;
  // Thirty threshold bytes per output-row tile, read through the same two
  // byte-wide W ports before work starts. Thresholds stay in local registers.
  if(consumer_per_p&&prologue)s.consumer_tau_scalar_read_bytes+=MT*30;
  if(carry)for(int m=0;m<MT;m++)for(int k=0;k<KP;k++)pe[m][k].touched=(*carry)[m][k];
  for(auto&q:st)q.control=now+1;
  for(int k=0;k<KP;k++) if(subs[k].empty()) {st[k].finished=true;completed++;}
  while(completed<KP) {
    s.event_steps++;
    // One-cycle memory return. A raw NRV row is broadcast to all same-ID
    // PEs, each of which compacts *nonzero-weight* rows into its NR4 buffer.
    for(int k=0;k<KP;k++) {
      auto &q=st[k]; if(q.finished) continue;
      if(q.response==now) {
        const auto &a=subs[k][q.sub];
        bool last=q.row==int(a.rows.size());
        for(int m=0;m<MT;m++)
          receive(pe[m][k],W[(first+m)*C+q.pending_d],q.pending_masks,last,s);
        q.response=INF;
      }
    }
    for(int m=0;m<MT;m++) for(int k=0;k<KP;k++) {
      auto &p=pe[m][k];
      if(p.exec<=now) {p.exec=INF;p.drain=now+1;}
      if(p.exec==INF && !p.ready.empty()) {
        Packet x=p.ready.front();p.ready.pop_front();
        // One selected spike each beat. A separate potential-update path
        // overlaps column-change flush with the next partial accumulation.
        p.exec=now+x.selected;
        int distinct=0,initial=0;
        for(int r=0;r<7;r++){distinct+=pc(x.masks[r]);initial+=pc(x.masks[r]&~p.touched[r]);p.touched[r]|=x.masks[r];}
        s.selected+=x.selected;s.packets++;
        s.partial_adds+=x.selected-distinct;
        s.state_writes+=distinct;s.initial_writes+=initial;
        s.state_reads+=distinct-initial;
      }
    }
    // Same k-ID retires a complete (position partition, virtual plane)
    // only after every output-row tile has consumed it.
    for(int k=0;k<KP;k++) {
      auto &q=st[k];if(q.finished || q.control>now) continue;
      if(q.draining) {
        q.draining=false;
        q.sub++;q.row=0;
        if(q.sub==int(subs[k].size())) {q.finished=true;completed++;continue;}
        q.control=now+1;
        for(int m=0;m<MT;m++) pe[m][k]=PE{};
        continue;
      }
      const auto &a=subs[k][q.sub];
      if(q.row<int(a.rows.size()) || q.response!=INF) continue;
      bool all=true;
      for(int m=0;m<MT;m++) {
        auto &p=pe[m][k];
        bool done=!p.filling && p.ready.empty() && p.exec==INF && p.drain<=now;
        if(done && p.done==INF) p.done=now;
        if(!done) all=false;
      }
      if(!all) continue;
      for(int m=0;m<MT;m++) s.synchronization_wait+=now-pe[m][k].done;
      if(a.last && count_output_drain) {
        // Noncausal T10 PSN extension: export R*P signed partial values to a ready
        // consumer at one scalar word/PE/beat before reusing these slots.
        // The consumer's compute/BN/FC2 and global transport are NOT modeled.
        q.draining=true;
        if(consumer_per_p) {
          // Two R-word S15 register caches: load first p, then load the next
          // p in parallel with the current p's CSD program. One U24/PE.
          // consumer_per_p = 10 threshold initializations + actual CSD digits.
          I cost=(R+1)+I(P)*consumer_per_p+1;
          q.control=now+cost;
          s.consumer_source_words+=I(MT)*R*P;
          s.consumer_output_bits+=I(MT)*10*P;
          s.consumer_csd_issues+=I(MT)*P*(consumer_per_p-10-restore_loads-restore_adds);
          s.consumer_restore_loads+=I(MT)*P*restore_loads;
          s.consumer_restore_add_stores+=I(MT)*P*restore_adds;
          s.consumer_threshold_inits+=I(MT)*P*10;
          s.consumer_PE_service_beats+=I(MT)*cost;
        } else {
          q.control=now+I(R)*P;
          s.boundary_words+=I(MT)*R*P;s.boundary_pe_beats+=I(MT)*R*P;
        }
      } else {
        q.sub++;q.row=0;
        if(q.sub==int(subs[k].size())) {q.finished=true;completed++;continue;}
        q.control=now+1;
        for(int m=0;m<MT;m++) pe[m][k]=PE{};
      }
    }
    if(completed==KP) break;
    array<bool,KP> eligible{};int demand=0,blocked=0;
    for(int k=0;k<KP;k++) {
      auto &q=st[k];
      if(q.finished||q.draining||q.control>now||q.response!=INF)continue;
      const auto&a=subs[k][q.sub];if(q.row>=int(a.rows.size()))continue;
      bool all=true;
      for(int m=0;m<MT;m++) if(!pe[m][k].can_fetch()) all=false;
      if(all) {eligible[k]=true;demand++;} else blocked++;
    }
    int grants=0,last_grant=-1;array<bool,KP> granted{};
    for(int j=0;j<KP&&grants<2;j++) {
      int k=(rr+j)%KP;if(!eligible[k])continue;
      auto&q=st[k];auto&a=subs[k][q.sub];auto row=a.rows[q.row++];
      q.pending_d=row.d;q.pending_masks=row.masks;q.response=now+fetch_latency;
      // Reserve the same row in all output tiles; no hidden multiported
      // NRV memory or per-tile row-request reordering is assumed.
      for(int m=0;m<MT;m++) if(!pe[m][k].filling) {
        auto&p=pe[m][k];p.filling=true;p.nfill=0;p.selected=0;p.umask.fill(0);
        s.max_buffers=max(s.max_buffers,I(p.occupancy()));
      }
      s.local_w_reads+=MT;s.shared_nrv_reads++;
      if(fetch_latency>1)s.source_NR_read_requests++;
      grants++;granted[k]=true;last_grant=k;
    }
    if(grants) rr=(last_grant+1)%KP;
    s.arbiter_denied_requests+=max(0,demand-2);
    s.bank_backpressure+=blocked;
    int inflight=0,total_inflight=0;
    for(const auto&q:st) if(q.response!=INF) {total_inflight+=MT;if(q.response==now+1)inflight+=MT;}
    s.max_w_inflight=max(s.max_w_inflight,I(inflight));
    s.max_source_or_W_inflight=max(s.max_source_or_W_inflight,I(total_inflight));
    I next=INF;
    for(int k=0;k<KP;k++) {
      const auto&q=st[k];if(q.finished)continue;
      if(q.response>now)next=min(next,q.response);
      if(q.control>now)next=min(next,q.control);
      for(int m=0;m<MT;m++) {
        const auto&p=pe[m][k];
        if(p.exec>now)next=min(next,p.exec);
        if(p.drain>now)next=min(next,p.drain);
      }
    }
    // If a request was denied solely by a busy port, the next arbitration
    // opportunity is one beat later even when no execution event occurs.
    if(demand>grants)next=min(next,now+1);
    if(next==INF||next<=now) throw runtime_error("event timeline stalled");
    s.jumped+=max(I(0),next-now-1);
    now=next;
  }
  if(carry)for(int m=0;m<MT;m++)for(int k=0;k<KP;k++)(*carry)[m][k]=pe[m][k].touched;
  s.group_min=min(s.group_min,now);s.group_max=max(s.group_max,now);
  return now;
}

// Input staging for finite mode. Native P8 codes occupy 3 bytes/source row.
// Reverse read + reverse NRV write is safe even when all 128 rows are live:
// the next output address is at least 8*d, above unread dense input [0,3*d).
Sub staged_sub(const vector<uint8_t>&codes,const vector<uint8_t>&decode,
               int N,int C,int R,int part,int begin,int length) {
  array<uint8_t,1024> bank{};
  for(int i=0;i<length;i++) {
    uint32_t word=0;
    for(int p=0;p<8 && part*8+p<N;p++)word|=uint32_t(codes[(part*8+p)*C+begin+i])<<(3*p);
    for(int byte=0;byte<3;byte++)bank[3*i+byte]=(word>>(8*byte))&255;
  }
  int cursor=1024,peak=length*3;
  for(int i=length-1;i>=0;i--) {
    uint32_t word=uint32_t(bank[3*i])|(uint32_t(bank[3*i+1])<<8)|(uint32_t(bank[3*i+2])<<16);
    uint32_t live=0;
    for(int p=0;p<8 && part*8+p<N;p++) {
      int code=(word>>(3*p))&7;bool any=false;
      for(int r=0;r<R;r++)if(decode[r*8+code])any=true;
      if(any)live|=1u<<p;
    }
    if(!live){peak=max(peak,3*i+1024-cursor);continue;}
    cursor-=8;
    if(cursor<3*i)throw runtime_error("in-place NRV overwrites unread dense codes");
    uint64_t record=uint64_t(begin+i)|(uint64_t(live)<<9)|(uint64_t(word)<<17);
    for(int byte=0;byte<8;byte++)bank[cursor+byte]=(record>>(8*byte))&255;
    peak=max(peak,3*i+1024-cursor);
  }
  Sub a;a.encoding_peak_bytes=peak;a.part=part;a.r=0;a.last=begin+length==C;
  for(int pos=cursor;pos<1024;pos+=8) {
    uint64_t record=0;for(int byte=0;byte<8;byte++)record|=uint64_t(bank[pos+byte])<<(8*byte);
    Row row;row.d=record&511;uint32_t live=(record>>9)&255,word=(record>>17)&0xffffff;
    for(int p=0;p<8&&part*8+p<N;p++) {
      int code=(word>>(3*p))&7;
      if(code!=codes[(part*8+p)*C+row.d])throw runtime_error("in-place code mismatch");
      bool any=false;
      for(int r=0;r<R;r++)if(decode[r*8+code]){row.masks[r]|=1u<<p;any=true;}
      if(any!=bool(live&(1u<<p)))throw runtime_error("NRV mask mismatch");
    }
    if(!a.rows.empty() && row.d<=a.rows.back().d)throw runtime_error("NRV source order changed");
    a.rows.push_back(row);
  }
  return a;
}

I bus_transfer(I bytes,Stats&s) {
  I beats=5*((bytes+31)/32); // complete shared32-byte transactions, each five beats
  s.bus_beats+=beats;return beats;
}

I finite_group(const vector<vector<vector<Sub>>>&waves,const vector<int8_t>&W,
               int N,int C,int H,int P,int R,int first,Stats&s,int consumer_per_p,int restore_loads,int restore_adds) {
  I elapsed=0;
  I wbytes=I(MT)*C,tbytes=I(MT)*30;
  s.bus_weight_bytes+=wbytes;s.bus_tau_bytes+=tbytes;
  elapsed+=bus_transfer(wbytes+tbytes,s);
  // Conservative explicit local fill: one byte write per tile per beat,
  // eight tiles in parallel. No overlap with the shared bus transaction.
  elapsed+=C+30;s.local_weight_write_beats+=C+30;
  elapsed+=15;s.local_tau_preload_beats+=15;
  s.consumer_tau_scalar_read_bytes+=tbytes;
  for(const auto&wave:waves) {
    Touch carry{};int active=0;for(const auto&stream:wave)if(!stream.empty())active++;
    int chunks=(C+127)/128;
    for(int chunk=0;chunk<chunks;chunk++) {
      int length=min(128,C-chunk*128);
      I bytes=I(active)*length*3;
      s.bus_input_bytes+=bytes;elapsed+=bus_transfer(bytes,s);
      // 64-bit 1R1W source-bank. A 16-byte extraction register handles
      // unaligned packed24-bit reads. The reverse pass performs one row per
      // ID per beat and includes zero rows. Two startup/drain beats are paid.
      s.input_decode_beats+=length+2;elapsed+=length+2;
      s.input_dense_rows_scanned+=I(active)*length;
      vector<vector<Sub>>sub(KP);
      for(int k=0;k<KP;k++)if(!wave[k].empty()) {
        sub[k].push_back(wave[k][chunk]);
        s.source_buffer_peak_bytes=max(s.source_buffer_peak_bytes,
                         I(sub[k][0].encoding_peak_bytes));
      }
      // NRV bank read then W read: two-stage one-transaction/ID bridge.
      // Grant reservation fixes the following-cycle W port ownership.
      // The stage intentionally lacks a second NRV lookahead entry.
      I run=simulate_group(sub,W,C,P,R,first,s,true,
                          chunk==chunks-1?consumer_per_p:0,&carry,2,false,restore_loads,restore_adds);
      elapsed+=run;s.finite_compute_beats+=run;
    }
    // All P*T gates held in a common 640-byte bank until this serial bus
    // drain finishes; no hidden wide-partial output or unlimited FIFO.
    I output=(I(active)*MT*P*10+7)/8+8; // one fixed 8-byte wave descriptor
    s.bus_output_bytes+=output;elapsed+=bus_transfer(output,s);s.finite_wave_count++;
  }
  s.full_output_group_min=min(s.full_output_group_min,elapsed);
  s.full_output_group_max=max(s.full_output_group_max,elapsed);
  return elapsed;
}

#ifndef GUSTAV_REFERENCE_LIBRARY
int main(int argc,char**argv) {
  try {
    int P=argc>1?stoi(argv[1]):8;
    int limits=argc>2?stoi(argv[2]):0;
    bool packed=argc>3?stoi(argv[3]):false;
    int consumer_per_p=argc>4?stoi(argv[4]):0;
    bool finite=argc>5?stoi(argv[5]):false;
    int restore_loads=argc>6?stoi(argv[6]):0,restore_adds=argc>7?stoi(argv[7]):0;
    int32_t dim[4];cin.read(reinterpret_cast<char*>(dim),sizeof(dim));
    int N=dim[0],C=dim[1],H=dim[2],R=dim[3];
    if(H%MT||P>32||P<1)throw runtime_error("shape/config");
    vector<int8_t>W(H*C);vector<uint8_t>codes(N*C),decode(R*8);
    cin.read(reinterpret_cast<char*>(W.data()),W.size());
    cin.read(reinterpret_cast<char*>(codes.data()),codes.size());
    cin.read(reinterpret_cast<char*>(decode.data()),decode.size());
    if(!cin)throw runtime_error("input short");
    vector<vector<Sub>>subs(KP);Stats s;
    for(int part=0;part<(N+P-1)/P;part++) for(int r=0;r<(packed?1:R);r++) {
      Sub a;a.part=part;a.r=r;a.last=packed || r==R-1;
      for(int d=0;d<C;d++) {
        Row row;row.d=d;bool any=false;
        for(int p=0;p<P&&part*P+p<N;p++) {
          int code=codes[(part*P+p)*C+d];
          if(packed) {
            for(int j=0;j<R;j++) if(decode[j*8+code]) {
              row.masks[j]|=uint32_t(1)<<p;any=true;
            }
          } else if(decode[r*8+code]) {row.masks[0]|=uint32_t(1)<<p;any=true;}
        }
        if(any)a.rows.push_back(row);
      }
      s.input_rows+=a.rows.size();s.input_submatrices++;
      // packed format has one P-bit live bitmap plus P fixed 3-bit codes.
      // Both packed axes pay exactly the same format, including zero codes.
      I bits=I(a.rows.size())*(9+P*(packed?4:1))+64;
      s.input_payload_bits+=bits;
      s.peak_nrv_plane_bytes=max(s.peak_nrv_plane_bytes,(bits+7)/8);
      subs[part%KP].push_back(move(a));
    }
    int groups=limits?min(limits,H/MT):H/MT;
    if(finite) {
      if(P!=8 || !packed || !consumer_per_p)throw runtime_error("finite scope is packed P8 with CSD consumer");
      // Explicit all-live layout case, independent of the measured sparsity.
      vector<uint8_t>all_live(8*128,1);
      auto check=staged_sub(all_live,decode,8,128,R,0,0,128);
      if(check.rows.size()!=128)throw runtime_error("all-live layout check failed");
      vector<vector<vector<Sub>>>waves;
      int parts=(N+P-1)/P;
      for(int base=0;base<parts;base+=KP) {
        vector<vector<Sub>>wave(KP);
        for(int k=0;k<KP;k++)if(base+k<parts)
          for(int begin=0;begin<C;begin+=128)
            wave[k].push_back(staged_sub(codes,decode,N,C,R,base+k,begin,min(128,C-begin)));
        waves.push_back(move(wave));
      }
      // A single microprogram image is loaded, then broadcast to eight
      // per-ID ROM replicas. Physical ROM capacity is common max, not free.
      I program=(I(consumer_per_p+1)*12+7)/8;
      s.bus_program_bytes+=program;s.beats+=bus_transfer(program,s);
      s.beats+=consumer_per_p+1;s.local_weight_write_beats+=consumer_per_p+1;
      for(int group=0;group<groups;group++) {
        s.beats+=finite_group(waves,W,N,C,H,P,R,group*MT,s,consumer_per_p,restore_loads,restore_adds);s.groups++;
      }
    } else for(int group=0;group<groups;group++) {
      I a=simulate_group(subs,W,C,P,R,group*MT,s,true,consumer_per_p,nullptr,1,true,restore_loads,restore_adds);s.beats+=a;s.groups++;
    }
    // Functional spot-check: actual NR4 selected-weight accumulation vs
    // independent dense integer dot, including signed W and cancellation.
    I numeric=0;
    for(int m: {0,H/3,H-1}) for(int k=0;k<KP;k++) {
      for(int si=0;si<int(subs[k].size());si+=max(1,int(subs[k].size()/7))) {
        const auto&a=subs[k][si];vector<I>got(P*R),want(P*R);
        vector<pair<Row,int>>batch;
        auto flush=[&]() {
          while(true) {
            int key=1000,which=-1,rrr=-1,ppp=-1;
            for(int q=0;q<int(batch.size());q++)for(int r=0;r<7;r++)if(batch[q].first.masks[r]) {
              int p=__builtin_ctz(batch[q].first.masks[r]);
              int c=p*R+r;
              if(c<key){key=c;which=q;rrr=r;ppp=p;}
            }
            if(which<0)break;
            got[rrr*P+ppp]+=batch[which].second;
            batch[which].first.masks[rrr]&=batch[which].first.masks[rrr]-1;
          }
          batch.clear();
        };
        for(const auto&row:a.rows)if(W[m*C+row.d]) {
          batch.push_back({row,W[m*C+row.d]});if(batch.size()==NR)flush();
        }
        flush();
        for(int p=0;p<P&&a.part*P+p<N;p++)for(int d=0;d<C;d++) {
          int code=codes[(a.part*P+p)*C+d];
          if(packed)for(int r=0;r<R;r++)want[r*P+p]+=I(W[m*C+d])*decode[r*8+code];
          else want[p]+=I(W[m*C+d])*decode[a.r*8+code];
        }
        if(got!=want)throw runtime_error("selected-spike sum mismatch");numeric+=P*(packed?R:1);
      }
    }
#define OUT(x) cout << "\"" #x "\":" << s.x << ",";
    cout << "{";
    OUT(beats)OUT(groups)OUT(event_steps)OUT(jumped)OUT(local_w_reads)
    OUT(zero_w_reads)OUT(selected)OUT(packets)OUT(partial_adds)
    OUT(state_writes)OUT(state_reads)OUT(initial_writes)OUT(shared_nrv_reads)
    OUT(arbiter_denied_requests)OUT(bank_backpressure)OUT(synchronization_wait)
    OUT(boundary_words)OUT(boundary_pe_beats)
    OUT(consumer_source_words)OUT(consumer_output_bits)OUT(consumer_csd_issues)
    OUT(consumer_threshold_inits)OUT(consumer_PE_service_beats)OUT(consumer_tau_scalar_read_bytes)
    OUT(consumer_restore_loads)OUT(consumer_restore_add_stores)
    OUT(max_buffers)OUT(max_w_inflight)OUT(max_source_or_W_inflight)
    OUT(bus_beats)OUT(bus_input_bytes)OUT(bus_weight_bytes)OUT(bus_tau_bytes)OUT(bus_program_bytes)OUT(bus_output_bytes)
    OUT(input_decode_beats)OUT(input_dense_rows_scanned)OUT(local_weight_write_beats)OUT(local_tau_preload_beats)
    OUT(finite_compute_beats)OUT(source_buffer_peak_bytes)OUT(source_NR_read_requests)OUT(finite_wave_count)
    OUT(full_output_group_min)OUT(full_output_group_max)
    OUT(input_rows)OUT(input_submatrices)OUT(input_payload_bits)
    OUT(peak_nrv_plane_bytes)OUT(group_min)OUT(group_max)
    cout << "\"numeric_values_checked\":"<<numeric<<"}"<<endl;
  } catch(const exception&e) {cerr<<e.what()<<endl;return 1;}
}
#endif
