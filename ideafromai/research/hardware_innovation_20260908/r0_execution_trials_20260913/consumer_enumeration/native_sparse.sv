module native_sparse (
  input logic clk, reset_n,
  input logic cfg_valid,
  input logic [1:0] cfg_kind,
  input logic [13:0] cfg_addr,
  input logic [127:0] cfg_data,
  input logic start,
  input logic [1:0] mode,
  input logic source_allow, weight_allow,
  output logic result_valid,
  input logic result_ready,
  output logic [8:0] result_addr,
  output logic [255:0] result_data,
  output logic done,
  output logic [31:0] cycles, source_words, weight_words, psum_reads,
  output logic [31:0] psum_writes, sum_issues, update_issues,
  output logic [31:0] source_stalls, weight_stalls, output_stalls,
  output logic [31:0] masked_contexts, zero_contexts
);
  // One 10-bit single-read source bank. Eight independent 16-bit W banks
  // each accept one read of the common row per cycle (128-bit aggregate).
  logic [9:0] source_mem [0:1535];
  logic signed [15:0] weight_mem [0:7][0:10367];
  // N8 x C4 x all 9 spatial taps; one bit per complete structural block.
  logic block_live [0:287];
  // Eight physical psum banks; row=(og*4+p)*10+t. Reads and writes occupy
  // separate states, hence no asynchronous psum read hidden inside an ADD.
  logic signed [31:0] psum_mem [0:7][0:479];
  logic [9:0] mask_a [0:3], mask_b [0:3];
  logic signed [15:0] wa [0:7], wb [0:7];
  logic signed [16:0] wab [0:7];
  logic signed [31:0] psreg [0:7];
  logic signed [31:0] lhs [0:7], rhs [0:7], add_result [0:7];
  logic [1:0] mode_q;
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
  typedef enum logic [4:0] {
    IDLE, CLEAR, INIT, CONTEXT, READ_A, READ_B, CHECK,
    NATIVE_START, NATIVE_DEST, READ_WA, READ_WB, MAKE_SUM,
    NEXT_TIME, PS_READ, ADD_WRITE, ADVANCE, NATIVE_ADVANCE,
    DRAIN_READ, DRAIN_SEND, FINISH, SKIP_BLOCK, DEST_PICK
  } state_t;
  state_t state;
  integer i;

  always_comb begin
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
    selected_dest=0;
    for(integer q=47;q>=0;q=q-1) if(dest_pending[q]) selected_dest=q;
    native_dep=(native_xy/4 >= p/2) && (native_xy/4-p/2 < 3)
            && (native_xy%4 >= p%2) && (native_xy%4-p%2 < 3);
    source_address=(2*cp)*16+(p/2+tap/3)*4+p%2+tap%3;
    if(mode_q[0]) source_address=(2*cp)*16+native_xy;
    if(state==READ_B) source_address=source_address+16;
    native_in_bounds=(int'(origin_y)+(source_address%16)/4 >= 0)
                  && (int'(origin_y)+(source_address%16)/4 < 240)
                  && (int'(origin_x)+(source_address%16)%4 >= 0)
                  && (int'(origin_x)+(source_address%16)%4 < 320);
    weight_address=og*864+(2*cp)*9+tap;
    if(mode_q[0])
      weight_address=og*864+(2*cp)*9+(native_xy/4-p/2)*3+native_xy%4-p%2;
    if(state==READ_WB) weight_address=weight_address+9;
    current_a=mask_a[p]; current_b=mask_b[p];
    if(mode_q[0]) begin current_a=source_a_hold; current_b=source_b_hold; end
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
    result_valid=(state==DRAIN_SEND);
    result_addr=9'(row);
    for(integer lane=0;lane<8;lane=lane+1) begin
      lhs[lane]=psreg[lane]; rhs[lane]=32'sd0;
      if(state==MAKE_SUM) begin
        lhs[lane]={{16{wa[lane][15]}},wa[lane]};
        rhs[lane]={{16{wb[lane][15]}},wb[lane]};
      end else if(state==ADD_WRITE) begin
        case({current_a[time_idx],current_b[time_idx]})
          2'b10: rhs[lane]={{16{wa[lane][15]}},wa[lane]};
          2'b01: rhs[lane]={{16{wb[lane][15]}},wb[lane]};
          2'b11: rhs[lane]={{15{wab[lane][16]}},wab[lane]};
          default: rhs[lane]=32'sd0;
        endcase
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
      state<=IDLE; done<=0; mode_q<=0;origin_y<=0;origin_x<=0;
      og<=0;cp<=0;tap<=0;p<=0;native_xy<=0;row<=0;time_idx<=0;
      pending<=0;dest_pending<=0;source_a_hold<=0;source_b_hold<=0;result_data<=0;
      cycles<=0;source_words<=0;weight_words<=0;psum_reads<=0;psum_writes<=0;
      sum_issues<=0;update_issues<=0;source_stalls<=0;weight_stalls<=0;
      output_stalls<=0;masked_contexts<=0;zero_contexts<=0;
      for(i=0;i<4;i=i+1) begin mask_a[i]<=0;mask_b[i]<=0;end
      for(i=0;i<8;i=i+1) begin wa[i]<=0;wb[i]<=0;wab[i]<=0;psreg[i]<=0;end
    end else begin
      done<=0;
      // Configuration is exclusively idle. All physical storage and loading
      // permissions are identical in the three runtime schedule modes.
      if(cfg_valid && state==IDLE) begin
        case(cfg_kind)
          0: source_mem[cfg_addr[10:0]]<=cfg_data[9:0];
          1: for(i=0;i<8;i=i+1) weight_mem[i][cfg_addr]<=cfg_data[i*16+:16];
          2: block_live[cfg_addr[8:0]]<=cfg_data[0];
          3: begin origin_y<=cfg_data[15:0];origin_x<=cfg_data[31:16];end
        endcase
      end
      if(state!=IDLE) cycles<=cycles+1;
      case(state)
        IDLE: if(start) begin
          mode_q<=mode;state<=CLEAR;row<=0;
          cycles<=0;source_words<=0;weight_words<=0;psum_reads<=0;psum_writes<=0;
          sum_issues<=0;update_issues<=0;source_stalls<=0;weight_stalls<=0;
          output_stalls<=0;masked_contexts<=0;zero_contexts<=0;
        end
        CLEAR: begin
          for(i=0;i<8;i=i+1) psum_mem[i][row]<=0;
          psum_writes<=psum_writes+1;
          if(row==479) state<=INIT; else row<=row+1;
        end
        INIT: begin
          og<=0;cp<=0;tap<=0;p<=0;native_xy<=0;
          if(mode_q[0]) state<=NATIVE_START; else state<=CONTEXT;
        end
        CONTEXT: begin
          if(!block_needed) begin masked_contexts<=masked_contexts+1;state<=SKIP_BLOCK;end
          else begin if(mode_q==2) p<=0;state<=READ_A;end
        end
        NATIVE_START: begin
          if(!any_group) begin masked_contexts<=masked_contexts+1;state<=SKIP_BLOCK;end
          else state<=READ_A;
        end
        READ_A: if(!native_in_bounds || source_allow) begin
          if(mode_q[0]) source_a_hold<=native_in_bounds?source_mem[source_address]:10'd0;
          else mask_a[p]<=native_in_bounds?source_mem[source_address]:10'd0;
          if(native_in_bounds) source_words<=source_words+1;
          state<=READ_B;
        end else source_stalls<=source_stalls+1;
        READ_B: if(!native_in_bounds || source_allow) begin
          if(mode_q[0]) source_b_hold<=native_in_bounds?source_mem[source_address]:10'd0;
          else mask_b[p]<=native_in_bounds?source_mem[source_address]:10'd0;
          if(native_in_bounds) source_words<=source_words+1;
          if(mode_q==2 && p<3) begin p<=p+1;state<=READ_A;end
          else state<=CHECK;
        end else source_stalls<=source_stalls+1;
        CHECK: begin
          if(mode_q[0]) begin
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
            dest_pending[selected_dest]<=0;launch_weights();
          end
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
          for(i=0;i<8;i=i+1) wab[i]<=add_result[i][16:0];
          sum_issues<=sum_issues+1;pending<=current_a|current_b;state<=NEXT_TIME;
        end
        NEXT_TIME: begin
          if(pending!=0) begin time_idx<=selected_time;state<=PS_READ;end
          else if(mode_q==2 && p<3) begin
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
          if(mode_q==3) state<=(dest_pending!=0)?DEST_PICK:NATIVE_ADVANCE;
          else if(mode_q[0]) begin
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
          else if(cp<47) begin cp<=cp+1;native_xy<=0;state<=NATIVE_START;end
          else begin row<=0;state<=DRAIN_READ;end
        end
        SKIP_BLOCK: begin
          // Same C4 run-skip permission in all modes. Metadata is consulted
          // before the native source access, not after reading then masking.
          tap<=0;native_xy<=0;
          if(cp/2<23) begin
            cp<=(cp/2+1)*2;
            if(mode_q[0]) state<=NATIVE_START;else state<=CONTEXT;
          end else if(mode_q[0]) begin row<=0;state<=DRAIN_READ;end
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
