// Finite compact-code source banks and sequential output-row strips.
// Reuse the established private W=0 compaction, NR4 and two-buffer definitions.
#define GUSTAV_REFERENCE_LIBRARY
#include "gustavsnn_reference_core.cpp"

struct Source {
  int d=0, word=-1,requested_word=-1;
  unsigned pending_tiles=0;
  I word_return=INF, scan=0, bridge_ready=INF, response=INF, consumer_done=INF;
  bool bridge=false,sealed=false,done=false,consuming=false;
  Row value,pending;
};
struct Extra {
  I source_bank_word_reads=0,dense_rows=0,source_bytes=0,weight_bytes=0,tau_bytes=0;
  I output_bytes=0,bus_service=0,compute_service=0,tau_local_reads=0;
  I weight_metadata_bytes=0;
  I output_slot_wait=0,source_fill_service=0,weight_fill_service=0,output_service=0;
  I source_peak=0,weight_peak=0,gate_slot_peak=0;
};

I run_wave(const vector<uint8_t>&codes,const vector<uint8_t>&decode,
           const vector<int8_t>&W,const vector<uint8_t>&weight_masks,
           int N,int C,int P,int R,int base,int first,
           int begin,int length,int consumer,int restore_loads,int restore_adds,
           bool reduce,int weight_mode,Touch&carry,Stats&s,Extra&e) {
  array<Source,KP>src;
  array<array<PE,KP>,MT>pe;
  vector<bool>block_live((C+15)/16,false);
  for(int d=0;d<C;d++)for(int m=0;m<MT;m++)if(W[(first+m)*C+d])block_live[d/16]=true;
  int completed=0;
  for(int k=0;k<KP;k++) {
    if((base+k)*P>=N){src[k].done=true;completed++;}
    for(int m=0;m<MT;m++)pe[m][k].touched=carry[m][k];
  }
  I now=0;int rr=0;
  while(completed<KP) {
    s.event_steps++;
    for(int k=0;k<KP;k++) {
      auto&q=src[k];if(q.done)continue;
      if(q.word_return==now){q.word=q.requested_word;q.word_return=INF;}
      if(q.response==now) {
        for(int m=0;m<MT;m++)if(q.pending_tiles&(1u<<m))
          receive(pe[m][k],W[(first+m)*C+q.pending.d],q.pending.masks,false,s);
        q.response=INF;
      }
    }
    for(int m=0;m<MT;m++)for(int k=0;k<KP;k++) {
      auto&p=pe[m][k];
      if(p.exec<=now){p.exec=INF;p.drain=now+1;}
      if(p.exec==INF&&!p.ready.empty()) {
        auto x=p.ready.front();p.ready.pop_front();int distinct=0,initial=0;
        for(int r=0;r<R;r++) {
          distinct+=pc(x.masks[r]);initial+=pc(x.masks[r]&~p.touched[r]);p.touched[r]|=x.masks[r];
        }
        // Separate partial and S-update paths, as in the earlier reference.
        p.exec=now+(reduce?distinct:x.selected);
        s.selected+=x.selected;s.packets++;s.partial_adds+=x.selected-distinct;
        s.state_writes+=distinct;s.initial_writes+=initial;s.state_reads+=distinct-initial;
      }
    }
    for(int k=0;k<KP;k++) {
      auto&q=src[k];if(q.done)continue;
      if(q.consuming) {
        if(q.consumer_done<=now){q.done=true;completed++;}
        continue;
      }
      // One source-bank read port and a 128-bit sliding extraction register.
      // At most one decoded bridge plus one outstanding W request per ID.
      if(q.d<length&&!q.bridge&&q.word_return==INF&&q.scan<=now) {
        if(weight_mode==1&&!block_live[(begin+q.d)/16]) {
          q.d=min(length,((begin+q.d)/16+1)*16-begin);
          q.scan=now+1; // one paid control step for a statically empty C16 block
        } else {
        int first_word=(q.d*3*P)/64;
        int last_word=((q.d+1)*3*P-1)/64;
        if(q.word<last_word) {
          q.requested_word=max(q.word+1,first_word);q.word_return=now+1;e.source_bank_word_reads++;
        }
        else {
          Row row;row.d=begin+q.d;bool live=false;
          for(int p=0;p<P&&(base+k)*P+p<N;p++) {
            int code=codes[((base+k)*P+p)*C+row.d];
            for(int r=0;r<R;r++)if(decode[r*8+code]){row.masks[r]|=1u<<p;live=true;}
          }
          q.d++;e.dense_rows++;q.scan=now+1;
          if(live){q.value=row;q.bridge=true;q.bridge_ready=now+1;}
        }
        }
      }
      if(q.d==length&&!q.bridge&&q.response==INF&&!q.sealed) {
        // End-of-C is discovered by scanning all rows, not by a last-live oracle.
        for(int m=0;m<MT;m++) {
          auto&p=pe[m][k];
          if(p.filling) {
            if(p.nfill)p.ready.push_back({p.selected,p.umask});
            p.filling=false;p.nfill=0;p.selected=0;p.umask.fill(0);
          }
        }
        q.sealed=true;
      }
      if(q.sealed) {
        bool all=true;
        for(int m=0;m<MT;m++) {
          auto&p=pe[m][k];bool done=p.ready.empty()&&!p.filling&&p.exec==INF&&p.drain<=now;
          if(done&&p.done==INF)p.done=now;
          if(!done)all=false;
        }
        if(all) {
          for(int m=0;m<MT;m++)s.synchronization_wait+=now-pe[m][k].done;
          int active_p=min(P,N-(base+k)*P);
          I cost=consumer?R+1+I(active_p)*consumer+1:1;
          q.consuming=true;q.consumer_done=now+cost;
          if(consumer) {
            s.consumer_source_words+=I(MT)*R*active_p;
            s.consumer_output_bits+=I(MT)*10*active_p;
            s.consumer_threshold_inits+=I(MT)*10*active_p;
            s.consumer_csd_issues+=I(MT)*active_p*(consumer-10-restore_loads-restore_adds);
            s.consumer_restore_loads+=I(MT)*active_p*restore_loads;
            s.consumer_restore_add_stores+=I(MT)*active_p*restore_adds;
            s.consumer_PE_service_beats+=I(MT)*cost;
          }
        }
      }
    }
    if(completed==KP)break;
    array<bool,KP>eligible{};array<unsigned,KP>needs{};
    array<int,MT>port_use{};int demand=0,grants=0,last=-1;
    for(int k=0;k<KP;k++) {
      auto&q=src[k];
      if(q.done||q.consuming||!q.bridge||q.bridge_ready>now||q.response!=INF)continue;
      // Resident 2:4 metadata is read before W request admission. Each tile
      // has an independent mask lookup for every source ID. These muxes are
      // explicit additional hardware assumptions, not free SRAM ports.
      bool all=true;
      for(int m=0;m<MT;m++) {
        bool need=weight_mode!=2 ||
          (weight_masks[(first+m)*(C/4)+q.value.d/4]&(1u<<(q.value.d%4)));
        if(need){needs[k]|=1u<<m;if(!pe[m][k].can_fetch())all=false;}
      }
      if(all){eligible[k]=true;demand++;}else s.bank_backpressure++;
    }
    for(int j=0;j<KP;j++) {
      int k=(rr+j)%KP;if(!eligible[k])continue;auto&q=src[k];
      bool ports=true;
      for(int m=0;m<MT;m++)if((needs[k]&(1u<<m))&&port_use[m]>=2)ports=false;
      if(!ports)continue;
      q.pending=q.value;q.pending_tiles=needs[k];q.bridge=false;q.response=now+1;
      for(int m=0;m<MT;m++)if(needs[k]&(1u<<m)) {
        port_use[m]++;
        if(!pe[m][k].filling) {
          auto&p=pe[m][k];p.filling=true;p.nfill=0;p.selected=0;p.umask.fill(0);
          s.max_buffers=max(s.max_buffers,I(p.occupancy()));
        }
      }
      s.local_w_reads+=pc(needs[k]);s.shared_nrv_reads++;grants++;last=k;
    }
    if(grants)rr=(last+1)%KP;
    s.arbiter_denied_requests+=max(0,demand-grants);
    I next=INF;
    if(demand>grants)next=now+1;
    for(int k=0;k<KP;k++) {
      auto&q=src[k];if(q.done)continue;
      for(I t:{q.word_return,q.response,q.consumer_done})if(t>now)next=min(next,t);
      if(q.bridge&&q.bridge_ready>now)next=min(next,q.bridge_ready);
      if(q.d<length&&!q.bridge&&!q.consuming&&q.word_return==INF)next=min(next,max(now+1,q.scan));
      for(int m=0;m<MT;m++) {
        auto&p=pe[m][k];if(p.exec>now)next=min(next,p.exec);
        if(p.drain>now)next=min(next,p.drain);
        if(!p.ready.empty()&&p.exec==INF)next=min(next,now+1);
      }
    }
    if(next==INF||next<=now)throw runtime_error("resident timeline stalled");
    s.jumped+=next-now-1;now=next;
  }
  for(int m=0;m<MT;m++)for(int k=0;k<KP;k++)carry[m][k]=pe[m][k].touched;
  e.compute_service+=now;return now;
}

int main(int argc,char**argv) {
  try {
    int P=stoi(argv[1]),F=stoi(argv[2]),consumer=stoi(argv[3]);
    int restore_loads=stoi(argv[4]),restore_adds=stoi(argv[5]);
    bool overlap=stoi(argv[6]),reduce=stoi(argv[7]);
    int weight_mode=argc>8?stoi(argv[8]):0;
    int32_t dim[4];cin.read(reinterpret_cast<char*>(dim),sizeof(dim));
    int N=dim[0],C=dim[1],H=dim[2],R=dim[3];
    vector<int8_t>W(H*C);vector<uint8_t>codes(N*C),decode(R*8);
    cin.read(reinterpret_cast<char*>(W.data()),W.size());
    cin.read(reinterpret_cast<char*>(codes.data()),codes.size());
    cin.read(reinterpret_cast<char*>(decode.data()),decode.size());
    vector<uint8_t>weight_masks(weight_mode==2?H*C/4:0);
    if(weight_mode==2)cin.read(reinterpret_cast<char*>(weight_masks.data()),weight_masks.size());
    if(!cin||P*R>56||H%MT||P>8||F<1||C%16)throw runtime_error("shape/resource contract");
    if(weight_mode==2)for(int h=0;h<H;h++)for(int d=0;d<C;d+=4) {
      unsigned mask=weight_masks[h*(C/4)+d/4];
      if(pc(mask)!=2)throw runtime_error("2:4 metadata must select exactly two slots");
      for(int j=0;j<4;j++)if(W[h*C+d+j]&&!(mask&(1u<<j)))
        throw runtime_error("W outside retained 2:4 mask");
    }
    Stats s;Extra e;I compute=0,bus_free=0;
    deque<I>gate_retire;int slots=8/P;
    auto transfer=[&](I bytes){I service=5*((bytes+31)/32);e.bus_service+=service;return service;};
    auto fill=[&](I bytes,bool weights,I write_service) {
      I start=max(compute,bus_free),service=transfer(bytes);
      // 32-byte shared transactions and explicit finite 64-byte staging.
      // W is statically striped across tiles (four byte writes/tile/burst).
      I finish=start+(overlap?service+(weights?4:2):service+write_service);
      bus_free=start+service;compute=finish;
      if(weights)e.weight_fill_service+=finish-start;else e.source_fill_service+=finish-start;
    };
    auto output=[&](I bytes) {
      while(!gate_retire.empty()&&gate_retire.front()<=compute)gate_retire.pop_front();
      if(int(gate_retire.size())>=slots)throw runtime_error("unreserved output slot");
      I service=transfer(bytes),start=max(bus_free,compute+(overlap?4:0));
      bus_free=start+service+(overlap?0:(bytes+7)/8);gate_retire.push_back(bus_free);
      e.output_service+=service;e.output_bytes+=bytes;
      e.gate_slot_peak=max(e.gate_slot_peak,I(gate_retire.size()));
      if(!overlap)compute=bus_free;
    };
    auto reserve_gate=[&]() {
      while(!gate_retire.empty()&&gate_retire.front()<=compute)gate_retire.pop_front();
      if(int(gate_retire.size())>=slots) {
        e.output_slot_wait+=gate_retire.front()-compute;compute=gate_retire.front();
        gate_retire.pop_front();
      }
    };
    // Common per-layer program, loaded once and broadcast to its replicas.
    I program=(I(consumer+1)*12+7)/8;
    e.bus_service+=5*((program+31)/32);
    compute=5*((program+31)/32)+consumer+1;
    bus_free=compute; // explicit broadcast configuration load, one instruction/cycle
    bool resident=I(C)*P*3<=8192;
    int parts=(N+P-1)/P,chunk=resident?C:256;
    for(int first=0;first<H;first+=MT*F) {
      int rows=min(F,(H-first)/MT);
      int last_tau_row=-1;
      I metadata_per_row=weight_mode==1?(C+127)/128:weight_mode==2?(C/4*3+7)/8:0;
      I local_weights=0,metadata=rows*metadata_per_row;
      for(int f=0;f<rows;f++) {
        if(weight_mode!=1)local_weights+=weight_mode==2?C/2:C;
        else for(int d=0;d<C;d+=16) {
          bool live=false;
          for(int m=0;m<MT;m++)for(int c=d;c<min(d+16,C);c++)
            if(W[(first+f*MT+m)*C+c])live=true;
          if(live)local_weights+=min(16,C-d);
        }
      }
      I local_bytes=local_weights+rows*30+metadata;
      if(local_bytes>8192)throw runtime_error("W strip capacity exceeded");
      e.weight_bytes+=I(MT)*local_weights;e.tau_bytes+=I(MT)*rows*30;
      e.weight_metadata_bytes+=I(MT)*metadata;
      e.weight_peak=max(e.weight_peak,local_bytes);
      fill(I(MT)*local_bytes,true,local_bytes);
      for(int base=0;base<parts;base+=KP) {
        int active=min(KP,parts-base);
        if(resident) {
          I per=(I(C)*P*3+7)/8,bytes=active*per;e.source_bytes+=bytes;
          e.source_peak=max(e.source_peak,per);fill(bytes,false,(per+7)/8);
        }
        for(int f=0;f<rows;f++) {
          reserve_gate();Touch carry{};
          // A single 10x24 threshold register per tile, explicitly reloaded.
          if(last_tau_row!=first+f*MT) {
            compute+=15+(metadata_per_row+1)/2;
            e.tau_local_reads+=I(MT)*30;last_tau_row=first+f*MT;
          }
          for(int begin=0;begin<C;begin+=chunk) {
            int length=min(chunk,C-begin);
            if(!resident) {
              I per=(I(length)*P*3+7)/8,bytes=active*per;e.source_bytes+=bytes;
              e.source_peak=max(e.source_peak,per);fill(bytes,false,(per+7)/8);
            }
            compute+=run_wave(codes,decode,W,weight_masks,N,C,P,R,base,first+f*MT,begin,length,
                        begin+length==C?consumer:0,restore_loads,restore_adds,reduce,weight_mode,carry,s,e);
          }
          output((I(active)*MT*P*10+7)/8+8);
        }
      }
    }
    cout<<"{";
#define SOUT(x) cout<<"\"" #x "\":"<<s.x<<",";
#define EOUT(x) cout<<"\"" #x "\":"<<e.x<<",";
    SOUT(selected)SOUT(local_w_reads)SOUT(zero_w_reads)SOUT(packets)SOUT(partial_adds)
    SOUT(state_writes)SOUT(state_reads)SOUT(initial_writes)SOUT(shared_nrv_reads)
    SOUT(arbiter_denied_requests)SOUT(bank_backpressure)SOUT(synchronization_wait)
    SOUT(consumer_source_words)SOUT(consumer_output_bits)SOUT(consumer_csd_issues)
    SOUT(consumer_threshold_inits)SOUT(consumer_restore_loads)SOUT(consumer_restore_add_stores)
    SOUT(max_buffers)SOUT(event_steps)
    EOUT(source_bank_word_reads)EOUT(dense_rows)EOUT(source_bytes)EOUT(weight_bytes)EOUT(tau_bytes)
    EOUT(weight_metadata_bytes)
    EOUT(output_bytes)EOUT(bus_service)EOUT(compute_service)EOUT(tau_local_reads)
    EOUT(output_slot_wait)EOUT(source_fill_service)EOUT(weight_fill_service)EOUT(output_service)
    EOUT(source_peak)EOUT(weight_peak)EOUT(gate_slot_peak)
    cout<<"\"elapsed\":"<<max(compute,bus_free)<<",\"resident\":"<<int(resident)<<"}"<<endl;
  }catch(const exception&e){cerr<<e.what()<<endl;return 1;}
}
