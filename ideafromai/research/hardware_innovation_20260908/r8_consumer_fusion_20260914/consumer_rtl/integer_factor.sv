module integer_factor (
  input logic clk, reset_n,
  input logic cfg_valid,
  input logic [2:0] cfg_kind,
  input logic [13:0] cfg_addr,
  input logic [255:0] cfg_data,
  input logic start,
  input logic [3:0] mode,
  input logic source_allow, weight_allow,
  output logic result_valid,
  input logic result_ready,
  output logic [8:0] result_addr,
  output logic [255:0] result_data,
  output logic done,
  output logic [31:0] cycles, source_words, weight_words, psum_reads,
  output logic [31:0] psum_writes, sum_issues, update_issues,
  output logic [31:0] source_stalls, weight_stalls, output_stalls,
  output logic [31:0] masked_contexts, zero_contexts, pattern_copies, merge_issues,
  output logic [31:0] z_reads,z_writes,first_issues,mac_issues,second_weight_words
);
  // One 10-bit source read. Expanded coefficients use eight signed21 banks
  // with one common row (168 useful bits within the common 256-bit port).
  // Q1 and Q2 subarrays share that request schedule; no simultaneous reads.
  logic [9:0] source_mem [0:1535];
  logic signed [20:0] weight_mem [0:7][0:10367];
  // N8 x C4 x all 9 spatial taps; one bit per complete structural block.
  logic block_live [0:287];
  // Eight physical psum banks; row=(og*4+p)*10+t. Reads and writes occupy
  // separate states, hence no asynchronous psum read hidden inside an ADD.
  logic signed [31:0] psum_mem [0:7][0:479];
  logic [9:0] mask_a [0:3], mask_b [0:3];
  logic signed [20:0] wa [0:7], wb [0:7];
  logic signed [21:0] wab [0:7];
  logic signed [31:0] psreg [0:7];
  logic signed [31:0] lhs [0:7], rhs [0:7], add_result [0:7];
  logic [3:0] mode_q;
  logic signed [15:0] origin_y, origin_x;
  integer og, cp, tap, p, native_xy, row, time_idx;
  integer source_address, weight_address;
  logic [8:0] state_address;
  logic [9:0] source_a_hold, source_b_hold, pending;
  logic block_needed, any_group, native_dep, any_four;
  logic need_a, need_b, need_sum;
  logic native_in_bounds;
  logic [9:0] current_a, current_b;
  integer selected_time, selected_dest;
  logic [47:0] dest_pending, dest_new;
  // Retained direct-control state: four source words, four signed21 W vectors,
  // signed23 scratch, and support metadata; modes6/7/8 share the union budget.
  // There is no 15-vector sum table and no precomputed support input port.
  logic [9:0] c4_source [0:3];
  logic signed [20:0] c4_weight [0:3][0:7];
  logic signed [22:0] c4_sum [0:7];
  // Direct mode6 retains two signed22 pair-parent vectors. A parent
  // is constructed only when some real T10 member uses both sources.
  logic signed [21:0] pair_parent [0:1][0:7];
  logic signed [31:0] pair_value [0:1][0:7];
  logic [3:0] c4_code [0:9];
  logic [14:0] c4_demand, c4_patterns;
  logic [3:0] c4_needed, c4_weight_pending, c4_remaining;
  logic [9:0] c4_matching_times;
  logic native_mode, c4_mode;
  integer c4_source_idx, c4_pair_idx;
  integer c4_selected_weight, c4_selected_pattern, c4_first;
  logic [1:0] c4_next;
  typedef enum logic [5:0] {
    IDLE, CLEAR, INIT, CONTEXT, READ_A, READ_B, CHECK,
    NATIVE_START, NATIVE_DEST, READ_WA, READ_WB, MAKE_SUM,
    NEXT_TIME, PS_READ, ADD_WRITE, ADVANCE, NATIVE_ADVANCE,
    DRAIN_READ, DRAIN_SEND, FINISH, SKIP_BLOCK, DEST_PICK,
    C4_READ_SOURCE, C4_CHECK, C4_READ_W, C4_PATTERN, C4_BUILD, PAIR_MERGE,
    F_CLEAR, F_SOURCE, F_CHECK, F_QREAD, F_TIME, F_ZREAD, F_ZADD, F_KNEXT,
    F_OUTPUT, F_VREAD, F_SCALAR, F_RPS, F_MAC, F_POSNEXT, F_ACLEAR, F_RNEXT, F_STORE
  } state_t;
  state_t state;
  integer i;

  // Exact rank8 factor chain: Q1 in [-3,3], Q2 signed16. z fits signed13.
  // z is eight banks: first phase reads/writes one common row across all eight;
  // second phase selects one bank and one row. These accesses never overlap.
  logic signed [2:0] q_mem [0:7][0:863];
  logic signed [15:0] v_mem [0:7][0:95];
  logic signed [2:0] q_hold [0:7];
  logic signed [15:0] v_hold [0:7];
  logic signed [12:0] z_mem [0:7][0:39];
  logic signed [12:0] z_hold [0:7], z_scalar;
  logic signed [28:0] factor_product [0:7];
  logic [39:0] f_pending;
  logic k_live [0:863];
  logic [9:0] c4_effective [0:3];
  logic [3:0] dest_needed;
  integer effective_k, selected_k;
  integer fk, fp, fr, f_selected;


  always_comb begin
    f_selected=0;
    for(integer z=39;z>=0;z=z-1) if(f_pending[z]) f_selected=z;
    c4_mode=(mode_q==4 || mode_q==5 || mode_q==6);
    native_mode=mode_q[0] || c4_mode;
    c4_needed=0; c4_demand=0;
    for(integer z=0;z<4;z=z+1) c4_needed[z]=|c4_source[z];
    for(integer t=0;t<10;t=t+1) begin
      c4_code[t]={c4_source[3][t],c4_source[2][t],c4_source[1][t],c4_source[0][t]};
      if(c4_code[t]!=0) c4_demand[int'(c4_code[t])-1]=1'b1;
    end
    c4_selected_weight=0;
    for(integer z=3;z>=0;z=z-1) if(c4_weight_pending[z]) c4_selected_weight=z;
    c4_selected_pattern=1;
    for(integer z=14;z>=0;z=z-1) if(c4_patterns[z]) c4_selected_pattern=z+1;
    c4_first=0; c4_next=0;
    for(integer z=3;z>=0;z=z-1) begin
      if((c4_selected_pattern & (1<<z))!=0) c4_first=z;
      if(c4_remaining[z]) c4_next=2'(z);
    end
    c4_matching_times=0;
    for(integer t=0;t<10;t=t+1)
      c4_matching_times[t]=(int'(c4_code[t])==c4_selected_pattern);
    selected_dest=0;
    for(integer q=47;q>=0;q=q-1) if(dest_pending[q]) selected_dest=q;
    dest_needed=0;effective_k=0;selected_k=0;
    for(integer z=0;z<4;z=z+1) begin
      effective_k=(2*cp+z)*9+(native_xy/4-p/2)*3+native_xy%4-p%2;
      c4_effective[z]=0;
      if(effective_k>=0 && effective_k<864 && k_live[effective_k]) c4_effective[z]=c4_source[z];
      selected_k=(2*cp+z)*9+(native_xy/4-(selected_dest%4)/2)*3+native_xy%4-selected_dest%2;
      if(selected_k>=0 && selected_k<864) dest_needed[z]=(|c4_source[z]) && k_live[selected_k];
    end
    block_needed=block_live[og*24+cp/2];
    any_group=1'b0;
    for (integer g=0;g<12;g=g+1)
      any_group=any_group | block_live[g*24+cp/2];
    dest_new='0;
    for(integer g=0;g<12;g=g+1)
      for(integer q=0;q<4;q=q+1)
        dest_new[g*4+q]=block_live[g*24+cp/2] &&
          (native_xy/4 >= q/2) && (native_xy/4-q/2 < 3) &&
          (native_xy%4 >= q%2) && (native_xy%4-q%2 < 3);
    native_dep=(native_xy/4 >= p/2) && (native_xy/4-p/2 < 3)
            && (native_xy%4 >= p%2) && (native_xy%4-p%2 < 3);
    source_address=(2*cp)*16+(p/2+tap/3)*4+p%2+tap%3;
    if(native_mode) source_address=(2*cp)*16+native_xy;
    if(state==READ_B) source_address=source_address+16;
    if(state==C4_READ_SOURCE) source_address=(2*cp+c4_source_idx)*16+native_xy;
    if(state==F_SOURCE) source_address=(fk/9)*16+(p/2+(fk%9)/3)*4+p%2+fk%3;
    native_in_bounds=(int'(origin_y)+(source_address%16)/4 >= 0)
                  && (int'(origin_y)+(source_address%16)/4 < 240)
                  && (int'(origin_x)+(source_address%16)%4 >= 0)
                  && (int'(origin_x)+(source_address%16)%4 < 320);
    weight_address=og*864+(2*cp)*9+tap;
    if(native_mode)
      weight_address=og*864+(2*cp)*9+(native_xy/4-p/2)*3+native_xy%4-p%2;
    if(state==READ_WB) weight_address=weight_address+9;
    if(state==C4_READ_W) weight_address=weight_address+c4_selected_weight*9;
    current_a=mask_a[p]; current_b=mask_b[p];
    if(native_mode) begin current_a=source_a_hold; current_b=source_b_hold; end
    if(mode_q>=5) begin
      current_a=c4_effective[2*c4_pair_idx]; current_b=c4_effective[2*c4_pair_idx+1];
    end
    any_four=1'b0;
    for(integer z=0;z<4;z=z+1) any_four=any_four | (|(mask_a[z]|mask_b[z]));
    need_a=|current_a;need_b=|current_b;need_sum=|(current_a&current_b);
    if(mode_q==2) begin
      need_a=0;need_b=0;need_sum=0;
      for(integer z=0;z<4;z=z+1) begin
        need_a=need_a | (|mask_a[z]);need_b=need_b | (|mask_b[z]);
        need_sum=need_sum | (|(mask_a[z]&mask_b[z]));
      end
    end
    selected_time=0;
    for(integer t=9;t>=0;t=t-1) if(pending[t]) selected_time=t;
    state_address=9'((og*4+p)*10+time_idx);
    if(mode_q>=7) state_address=9'(og*40+fp);
    result_valid=(state==DRAIN_SEND);
    result_addr=9'(row);
    for(integer lane=0;lane<8;lane=lane+1) begin
      // Parent/W bypass reads only local registers. The weight SRAM still
      // has a single common-row read and psum read remains a separate state.
      for(integer q=0;q<2;q=q+1) begin
        case({c4_effective[2*q][time_idx],c4_effective[2*q+1][time_idx]})
          2'b10: pair_value[q][lane]={{11{c4_weight[2*q][lane][20]}},c4_weight[2*q][lane]};
          2'b01: pair_value[q][lane]={{11{c4_weight[2*q+1][lane][20]}},c4_weight[2*q+1][lane]};
          2'b11: pair_value[q][lane]={{10{pair_parent[q][lane][21]}},pair_parent[q][lane]};
          default: pair_value[q][lane]=32'sd0;
        endcase
      end
      lhs[lane]=psreg[lane]; rhs[lane]=32'sd0;
      if(state==MAKE_SUM) begin
        lhs[lane]={{11{wa[lane][20]}},wa[lane]};
        rhs[lane]={{11{wb[lane][20]}},wb[lane]};
        if(mode_q>=5) begin
          lhs[lane]={{11{c4_weight[2*c4_pair_idx][lane][20]}},c4_weight[2*c4_pair_idx][lane]};
          rhs[lane]={{11{c4_weight[2*c4_pair_idx+1][lane][20]}},c4_weight[2*c4_pair_idx+1][lane]};
        end
      end else if(state==PAIR_MERGE) begin
        lhs[lane]=pair_value[0][lane];rhs[lane]=pair_value[1][lane];
      end else if(state==C4_BUILD) begin
        lhs[lane]={{9{c4_sum[lane][22]}},c4_sum[lane]};
        rhs[lane]={{11{c4_weight[c4_next][lane][20]}},c4_weight[c4_next][lane]};
      end else if(state==ADD_WRITE) begin
        case({current_a[time_idx],current_b[time_idx]})
          2'b10: rhs[lane]={{11{wa[lane][20]}},wa[lane]};
          2'b01: rhs[lane]={{11{wb[lane][20]}},wb[lane]};
          2'b11: rhs[lane]={{10{wab[lane][21]}},wab[lane]};
          default: rhs[lane]=32'sd0;
        endcase
        if(mode_q==4) rhs[lane]={{9{c4_sum[lane][22]}},c4_sum[lane]};
        else if(mode_q>=5) begin
          case({current_a[time_idx],current_b[time_idx]})
            2'b10: rhs[lane]={{11{c4_weight[2*c4_pair_idx][lane][20]}},c4_weight[2*c4_pair_idx][lane]};
            2'b01: rhs[lane]={{11{c4_weight[2*c4_pair_idx+1][lane][20]}},c4_weight[2*c4_pair_idx+1][lane]};
            2'b11: rhs[lane]={{10{pair_parent[c4_pair_idx][lane][21]}},pair_parent[c4_pair_idx][lane]};
            default: rhs[lane]=32'sd0;
          endcase
        end
        if(mode_q==6) begin
          if((c4_effective[0][time_idx]|c4_effective[1][time_idx]) &&
             (c4_effective[2][time_idx]|c4_effective[3][time_idx]))
            rhs[lane]={{9{c4_sum[lane][22]}},c4_sum[lane]};
          else if(c4_effective[0][time_idx]|c4_effective[1][time_idx]) rhs[lane]=pair_value[0][lane];
          else rhs[lane]=pair_value[1][lane];
        end
      end
      factor_product[lane]=$signed(v_hold[lane])*$signed(z_scalar);
      if(state==F_ZADD) begin
        lhs[lane]={{19{z_hold[lane][12]}},z_hold[lane]};
        rhs[lane]={{29{q_hold[lane][2]}},q_hold[lane]};
      end else if(state==F_MAC) begin
        lhs[lane]=psreg[lane];
        rhs[lane]={{3{factor_product[lane][28]}},factor_product[lane]};
      end
      // Exactly eight shared 32-bit datapath adders, including WA+WB.
      add_result[lane]=lhs[lane]+rhs[lane];
    end
  end

  // All schedules share whole-vector demand cancellation. A dead temporal
  // source never causes its separate 128-bit W row to be fetched; WA+WB is
  // computed only if an actual (p,t) consumer needs both products.
  task launch_weights;
    begin
      if(!need_a) for(integer l=0;l<8;l=l+1) wa[l]<=0;
      if(!need_b) for(integer l=0;l<8;l=l+1) wb[l]<=0;
      if(need_a) state<=READ_WA;else state<=READ_WB;
    end
  endtask

  always_ff @(posedge clk) begin
    if(!reset_n) begin
      fk<=0;fp<=0;fr<=0;f_pending<=0;z_scalar<=0;
      for(integer j=0;j<8;j=j+1) begin q_hold[j]<=0;v_hold[j]<=0;z_hold[j]<=0;end
      state<=IDLE; done<=0; mode_q<=0;origin_y<=0;origin_x<=0;
      og<=0;cp<=0;tap<=0;p<=0;native_xy<=0;row<=0;time_idx<=0;
      pending<=0;dest_pending<=0;source_a_hold<=0;source_b_hold<=0;result_data<=0;
      cycles<=0;source_words<=0;weight_words<=0;psum_reads<=0;psum_writes<=0;
      sum_issues<=0;update_issues<=0;source_stalls<=0;weight_stalls<=0;
      output_stalls<=0;masked_contexts<=0;zero_contexts<=0;pattern_copies<=0;merge_issues<=0;z_reads<=0;z_writes<=0;first_issues<=0;mac_issues<=0;second_weight_words<=0;
      c4_source_idx<=0;c4_pair_idx<=0;c4_patterns<=0;c4_weight_pending<=0;c4_remaining<=0;
      for(i=0;i<4;i=i+1) begin
        c4_source[i]<=0;
        for(integer l=0;l<8;l=l+1) c4_weight[i][l]<=0;
      end
      for(i=0;i<8;i=i+1) begin
        c4_sum[i]<=0;pair_parent[0][i]<=0;pair_parent[1][i]<=0;
      end
      for(i=0;i<4;i=i+1) begin mask_a[i]<=0;mask_b[i]<=0;end
      for(i=0;i<8;i=i+1) begin wa[i]<=0;wb[i]<=0;wab[i]<=0;psreg[i]<=0;end
    end else begin
      done<=0;
      // Configuration is exclusively idle. All physical storage and loading
      // permissions are identical in all runtime schedule modes.
      if(cfg_valid && state==IDLE) begin
        case(cfg_kind)
          0: source_mem[cfg_addr[10:0]]<=cfg_data[9:0];
          1: for(i=0;i<8;i=i+1) weight_mem[i][cfg_addr]<=cfg_data[i*32+:21];
          2: block_live[cfg_addr[8:0]]<=cfg_data[0];
          6: k_live[cfg_addr[9:0]]<=cfg_data[0];
          4: for(i=0;i<8;i=i+1) q_mem[i][cfg_addr[9:0]]<=cfg_data[i*32+:3];
          5: for(i=0;i<8;i=i+1) v_mem[i][cfg_addr[6:0]]<=cfg_data[i*32+:16];
          3: begin origin_y<=cfg_data[15:0];origin_x<=cfg_data[31:16];end
        endcase
      end
      if(state!=IDLE) cycles<=cycles+1;
      case(state)
        IDLE: if(start) begin
          mode_q<=mode;state<=CLEAR;row<=0;
          cycles<=0;source_words<=0;weight_words<=0;psum_reads<=0;psum_writes<=0;
          sum_issues<=0;update_issues<=0;source_stalls<=0;weight_stalls<=0;
          output_stalls<=0;masked_contexts<=0;zero_contexts<=0;pattern_copies<=0;merge_issues<=0;z_reads<=0;z_writes<=0;first_issues<=0;mac_issues<=0;second_weight_words<=0;
        end
        CLEAR: begin
          for(i=0;i<8;i=i+1) psum_mem[i][row]<=0;
          psum_writes<=psum_writes+1;
          if(row==479) begin
            if(mode_q>=7) begin fp<=0;state<=F_CLEAR;end else state<=INIT;
          end else row<=row+1;
        end

        F_CLEAR: begin
          z_writes<=z_writes+1;
          for(i=0;i<8;i=i+1) z_mem[i][fp]<=0;
          if(fp==39) begin fk<=0;p<=0;state<=F_SOURCE;end else fp<=fp+1;
        end
        F_SOURCE: if(!k_live[fk]) state<=F_KNEXT;
        else if(!native_in_bounds || source_allow) begin
          mask_a[p]<=native_in_bounds?source_mem[source_address]:10'd0;
          if(native_in_bounds) source_words<=source_words+1;
          if(p==3) state<=F_CHECK;else p<=p+1;
        end else source_stalls<=source_stalls+1;
        F_CHECK: begin
          f_pending<={mask_a[3],mask_a[2],mask_a[1],mask_a[0]};
          if((mask_a[3]|mask_a[2]|mask_a[1]|mask_a[0])==0) state<=F_KNEXT;
          else state<=F_QREAD;
        end
        F_QREAD: if(weight_allow) begin
          for(i=0;i<8;i=i+1) q_hold[i]<=q_mem[i][fk];
          weight_words<=weight_words+1;state<=F_TIME;
        end else weight_stalls<=weight_stalls+1;
        F_TIME: if(f_pending!=0) begin fp<=f_selected;state<=F_ZREAD;end else state<=F_KNEXT;
        F_ZREAD: begin
          for(i=0;i<8;i=i+1) z_hold[i]<=z_mem[i][fp];
          z_reads<=z_reads+1;state<=F_ZADD;
        end
        F_ZADD: begin
          for(i=0;i<8;i=i+1) z_mem[i][fp]<=add_result[i][12:0];
          z_writes<=z_writes+1;first_issues<=first_issues+1;
          f_pending<=f_pending & (f_pending-40'd1);state<=F_TIME;
        end
        F_KNEXT: begin
          if(fk==863) begin og<=0;fp<=0;fr<=0;state<=F_OUTPUT;end
          else begin fk<=fk+1;p<=0;state<=F_SOURCE;end
        end
        F_OUTPUT: if(mode_q==7) state<=F_VREAD;else state<=F_ACLEAR;
        F_ACLEAR: begin
          for(i=0;i<8;i=i+1) psreg[i]<=0;
          fr<=0;state<=F_SCALAR;
        end
        F_VREAD: if(weight_allow) begin
          for(i=0;i<8;i=i+1) v_hold[i]<=v_mem[i][og*8+fr];
          weight_words<=weight_words+1;second_weight_words<=second_weight_words+1;
          state<=(mode_q==7)?F_SCALAR:F_MAC;
        end else weight_stalls<=weight_stalls+1;
        F_SCALAR: begin
          z_scalar<=z_mem[fr][fp];z_reads<=z_reads+1;
          if(z_mem[fr][fp]==0) state<=(mode_q==7)?F_POSNEXT:F_RNEXT;
          else state<=(mode_q==7)?F_RPS:F_VREAD;
        end
        F_RPS: begin
          for(i=0;i<8;i=i+1) psreg[i]<=psum_mem[i][state_address];
          psum_reads<=psum_reads+1;state<=F_MAC;
        end
        F_MAC: begin
          for(i=0;i<8;i=i+1) begin
            if(mode_q==7) psum_mem[i][state_address]<=add_result[i];
            else psreg[i]<=add_result[i];
          end
          mac_issues<=mac_issues+1;
          if(mode_q==7) begin
            psum_writes<=psum_writes+1;update_issues<=update_issues+1;state<=F_POSNEXT;
          end else state<=F_RNEXT;
        end
        F_POSNEXT: begin
          if(fp<39) begin fp<=fp+1;state<=F_SCALAR;end
          else if(fr<7) begin fr<=fr+1;fp<=0;state<=F_VREAD;end
          else if(og<11) begin og<=og+1;fr<=0;fp<=0;state<=F_VREAD;end
          else begin row<=0;state<=DRAIN_READ;end
        end
        F_RNEXT: begin
          if(fr<7) begin fr<=fr+1;state<=F_SCALAR;end else state<=F_STORE;
        end
        F_STORE: begin
          for(i=0;i<8;i=i+1) psum_mem[i][state_address]<=psreg[i];
          psum_writes<=psum_writes+1;update_issues<=update_issues+1;
          if(fp<39) begin fp<=fp+1;state<=F_ACLEAR;end
          else if(og<11) begin og<=og+1;fp<=0;state<=F_ACLEAR;end
          else begin row<=0;state<=DRAIN_READ;end
        end
        INIT: begin
          og<=0;cp<=0;tap<=0;p<=0;native_xy<=0;
          if(native_mode) state<=NATIVE_START; else state<=CONTEXT;
        end
        CONTEXT: begin
          if(!block_needed) begin masked_contexts<=masked_contexts+1;state<=SKIP_BLOCK;end
          else begin if(mode_q==2) p<=0;state<=READ_A;end
        end
        NATIVE_START: begin
          if(!any_group) begin masked_contexts<=masked_contexts+1;state<=SKIP_BLOCK;end
          else if(c4_mode) begin c4_source_idx<=0;state<=C4_READ_SOURCE;end
          else state<=READ_A;
        end
        READ_A: if(!native_in_bounds || source_allow) begin
          if(native_mode) source_a_hold<=native_in_bounds?source_mem[source_address]:10'd0;
          else mask_a[p]<=native_in_bounds?source_mem[source_address]:10'd0;
          if(native_in_bounds) source_words<=source_words+1;
          state<=READ_B;
        end else source_stalls<=source_stalls+1;
        READ_B: if(!native_in_bounds || source_allow) begin
          if(native_mode) source_b_hold<=native_in_bounds?source_mem[source_address]:10'd0;
          else mask_b[p]<=native_in_bounds?source_mem[source_address]:10'd0;
          if(native_in_bounds) source_words<=source_words+1;
          if(mode_q==2 && p<3) begin p<=p+1;state<=READ_A;end
          else state<=CHECK;
        end else source_stalls<=source_stalls+1;
        CHECK: begin
          if(native_mode) begin
            if((source_a_hold|source_b_hold)==0) begin
              zero_contexts<=zero_contexts+1;state<=NATIVE_ADVANCE;
            end else if(mode_q==3) begin dest_pending<=dest_new;state<=DEST_PICK;end
            else begin og<=0;p<=0;state<=NATIVE_DEST;end
          end else if(mode_q==2) begin
            p<=0;
            if(!any_four) begin zero_contexts<=zero_contexts+1;state<=ADVANCE;end
            else launch_weights();
          end else if((current_a|current_b)==0) begin
            zero_contexts<=zero_contexts+1;state<=ADVANCE;
          end else launch_weights();
        end
        DEST_PICK: begin
          if(dest_pending==0) state<=NATIVE_ADVANCE;
          else begin
            og<=selected_dest/4;p<=selected_dest%4;
            dest_pending[selected_dest]<=0;
            if(c4_mode) begin c4_weight_pending<=dest_needed;state<=(dest_needed!=0)?C4_READ_W:ADVANCE;end
            else launch_weights();
          end
        end
        C4_READ_SOURCE: if(!native_in_bounds || source_allow) begin
          c4_source[c4_source_idx]<=native_in_bounds?source_mem[source_address]:10'd0;
          if(native_in_bounds) source_words<=source_words+1;
          if(c4_source_idx==3) state<=C4_CHECK;
          else c4_source_idx<=c4_source_idx+1;
        end else source_stalls<=source_stalls+1;
        C4_CHECK: begin
          if(c4_needed==0) begin zero_contexts<=zero_contexts+1;state<=NATIVE_ADVANCE;end
          else begin dest_pending<=dest_new;state<=DEST_PICK;end
        end
        C4_READ_W: if(weight_allow) begin
          for(i=0;i<8;i=i+1) c4_weight[c4_selected_weight][i]<=weight_mem[i][weight_address];
          weight_words<=weight_words+1;
          c4_weight_pending<=c4_weight_pending & (c4_weight_pending-4'd1);
          if((c4_weight_pending & (c4_weight_pending-4'd1))==0) begin
            if(mode_q==4) begin c4_patterns<=c4_demand;state<=C4_PATTERN;end
            else if(mode_q==6) begin
              // Nonblocking W loads are visible in the following parent
              // build or psum states. Never use stale pair_idx for demand.
              if((c4_effective[0]&c4_effective[1])!=0) begin c4_pair_idx<=0;state<=MAKE_SUM;end
              else if((c4_effective[2]&c4_effective[3])!=0) begin c4_pair_idx<=1;state<=MAKE_SUM;end
              else begin pending<=c4_effective[0]|c4_effective[1]|c4_effective[2]|c4_effective[3];state<=NEXT_TIME;end
            end else if((c4_effective[0]|c4_effective[1])!=0) begin
              c4_pair_idx<=0;pending<=c4_effective[0]|c4_effective[1];
              state<=((c4_effective[0]&c4_effective[1])!=0)?MAKE_SUM:NEXT_TIME;
            end else begin
              c4_pair_idx<=1;pending<=c4_effective[2]|c4_effective[3];
              state<=((c4_effective[2]&c4_effective[3])!=0)?MAKE_SUM:NEXT_TIME;
            end
          end
        end else weight_stalls<=weight_stalls+1;
        C4_PATTERN: begin
          // Legacy mode4 path (not evaluated here); signed23 holds four signed21 W.
          for(i=0;i<8;i=i+1)
            c4_sum[i]<={{2{c4_weight[c4_first][i][20]}},c4_weight[c4_first][i]};
          pattern_copies<=pattern_copies+1;
          c4_patterns[c4_selected_pattern-1]<=0;
          c4_remaining<=4'(c4_selected_pattern) & ~(4'd1<<c4_first);
          pending<=c4_matching_times;
          if((4'(c4_selected_pattern) & ~(4'd1<<c4_first))==0) state<=NEXT_TIME;
          else state<=C4_BUILD;
        end
        C4_BUILD: begin
          for(i=0;i<8;i=i+1) c4_sum[i]<=add_result[i][22:0];
          sum_issues<=sum_issues+1;
          c4_remaining<=c4_remaining & (c4_remaining-4'd1);
          if((c4_remaining & (c4_remaining-4'd1))==0) state<=NEXT_TIME;
        end
        NATIVE_DEST: begin
          if(!block_needed) begin
            masked_contexts<=masked_contexts+1;
            if(og<11) begin og<=og+1;p<=0;end
            else state<=NATIVE_ADVANCE;
          end else if(!native_dep) state<=ADVANCE;
          else launch_weights();
        end
        READ_WA: if(weight_allow) begin
          for(i=0;i<8;i=i+1) wa[i]<=weight_mem[i][weight_address];
          weight_words<=weight_words+1;
          if(need_b) state<=READ_WB;
          else begin pending<=current_a|current_b;state<=NEXT_TIME;end
        end else weight_stalls<=weight_stalls+1;
        READ_WB: if(weight_allow) begin
          for(i=0;i<8;i=i+1) wb[i]<=weight_mem[i][weight_address];
          weight_words<=weight_words+1;
          if(need_sum) state<=MAKE_SUM;
          else begin pending<=current_a|current_b;state<=NEXT_TIME;end
        end else weight_stalls<=weight_stalls+1;
        MAKE_SUM: begin
          for(i=0;i<8;i=i+1) begin
            wab[i]<=add_result[i][21:0];
            if(mode_q>=5) pair_parent[c4_pair_idx][i]<=add_result[i][21:0];
          end
          sum_issues<=sum_issues+1;
          if(mode_q==6) begin
            if(c4_pair_idx==0 && (c4_effective[2]&c4_effective[3])!=0) begin
              c4_pair_idx<=1;state<=MAKE_SUM;
            end else begin pending<=c4_effective[0]|c4_effective[1]|c4_effective[2]|c4_effective[3];state<=NEXT_TIME;end
          end else begin pending<=current_a|current_b;state<=NEXT_TIME;end
        end
        PAIR_MERGE: begin
          for(i=0;i<8;i=i+1) c4_sum[i]<=add_result[i][22:0];
          merge_issues<=merge_issues+1;sum_issues<=sum_issues+1;state<=PS_READ;
        end
        NEXT_TIME: begin
          if(pending!=0) begin
            time_idx<=selected_time;
            if(mode_q==6 && (c4_effective[0][selected_time]|c4_effective[1][selected_time]) &&
                           (c4_effective[2][selected_time]|c4_effective[3][selected_time])) state<=PAIR_MERGE;
            else state<=PS_READ;
          end else if(mode_q==6) state<=ADVANCE;
          else if(mode_q==4) begin
            if(c4_patterns!=0) state<=C4_PATTERN;else state<=ADVANCE;
          end else if(mode_q>=5) begin
            // Same read-end/start permission as mode3: no mandatory pair
            // selection bubble or empty-pair scan is charged to the control.
            if(c4_pair_idx==0 && (c4_effective[2]|c4_effective[3])!=0) begin
              c4_pair_idx<=1;pending<=c4_effective[2]|c4_effective[3];
              state<=((c4_effective[2]&c4_effective[3])!=0)?MAKE_SUM:NEXT_TIME;
            end else state<=ADVANCE;
          end else if(mode_q==2 && p<3) begin
            p<=p+1;pending<=mask_a[p+1]|mask_b[p+1];
          end else state<=ADVANCE;
        end
        PS_READ: begin
          for(i=0;i<8;i=i+1) psreg[i]<=psum_mem[i][state_address];
          psum_reads<=psum_reads+1;state<=ADD_WRITE;
        end
        ADD_WRITE: begin
          for(i=0;i<8;i=i+1) psum_mem[i][state_address]<=add_result[i];
          psum_writes<=psum_writes+1;update_issues<=update_issues+1;
          pending<=pending & (pending-10'd1);state<=NEXT_TIME;
        end
        ADVANCE: begin
          if(mode_q>=3) state<=(dest_pending!=0)?DEST_PICK:NATIVE_ADVANCE;
          else if(native_mode) begin
            if(p<3) begin p<=p+1;state<=NATIVE_DEST;end
            else if(og<11) begin og<=og+1;p<=0;state<=NATIVE_DEST;end
            else state<=NATIVE_ADVANCE;
          end else if(mode_q==2) begin
            p<=0;
            if(tap<8) begin tap<=tap+1;state<=CONTEXT;end
            else if(cp<47) begin cp<=cp+1;tap<=0;state<=CONTEXT;end
            else if(og<11) begin og<=og+1;cp<=0;tap<=0;state<=CONTEXT;end
            else begin row<=0;state<=DRAIN_READ;end
          end else begin
            if(tap<8) begin tap<=tap+1;state<=CONTEXT;end
            else if(cp<47) begin cp<=cp+1;tap<=0;state<=CONTEXT;end
            else if(p<3) begin p<=p+1;cp<=0;tap<=0;state<=CONTEXT;end
            else if(og<11) begin og<=og+1;p<=0;cp<=0;tap<=0;state<=CONTEXT;end
            else begin row<=0;state<=DRAIN_READ;end
          end
        end
        NATIVE_ADVANCE: begin
          if(native_xy<15) begin native_xy<=native_xy+1;state<=NATIVE_START;end
          else if(cp < (c4_mode?46:47)) begin
            cp<=cp+(c4_mode?2:1);native_xy<=0;state<=NATIVE_START;
          end
          else begin row<=0;state<=DRAIN_READ;end
        end
        SKIP_BLOCK: begin
          // Same C4 run-skip permission in all modes. Metadata is consulted
          // before the native source access, not after reading then masking.
          tap<=0;native_xy<=0;
          if(cp/2<23) begin
            cp<=(cp/2+1)*2;
            if(native_mode) state<=NATIVE_START;else state<=CONTEXT;
          end else if(native_mode) begin row<=0;state<=DRAIN_READ;end
          else if(mode_q==0 && p<3) begin p<=p+1;cp<=0;state<=CONTEXT;end
          else if(og<11) begin og<=og+1;p<=0;cp<=0;state<=CONTEXT;end
          else begin row<=0;state<=DRAIN_READ;end
        end
        DRAIN_READ: begin
          for(i=0;i<8;i=i+1) result_data[i*32+:32]<=psum_mem[i][row];
          psum_reads<=psum_reads+1;state<=DRAIN_SEND;
        end
        DRAIN_SEND: if(result_ready) begin
          if(row==479) state<=FINISH;else begin row<=row+1;state<=DRAIN_READ;end
        end else output_stalls<=output_stalls+1;
        FINISH: begin done<=1;state<=IDLE;end
        default: state<=IDLE;
      endcase
    end
  end
endmodule
