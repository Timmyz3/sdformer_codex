module i24_consumer (
  input logic clk, reset_n, start,
  input logic cfg_valid,
  input logic [4:0] cfg_addr,
  input logic [255:0] cfg_data,
  input logic raw_valid,
  output logic raw_ready,
  input logic [8:0] raw_addr,
  input logic [255:0] raw_data,
  output logic identity_request_valid,
  output logic [8:0] identity_address,
  input logic identity_valid,
  input logic [255:0] identity_data,
  output logic result_valid,
  input logic result_ready,
  output logic [8:0] result_addr,
  output logic [255:0] result_data,
  output logic done, error,
  output logic [31:0] cycles, raw_words, identity_words, coefficient_words,
  output logic [31:0] mul_issues, add_issues, round_issues, output_words,
  output logic [31:0] identity_stalls, raw_wait_cycles, join_wait_cycles, output_stalls, saturations
);
  // One physical common-address 8x32 constant read per cycle. Rows 2*og
  // and 2*og+1 contain a and b. p/id each have one vector buffer.
  logic signed [31:0] coefficient [0:7][0:23];
  logic signed [31:0] p_hold [0:7], j_hold [0:7], a_hold [0:7], b_hold [0:7];
  logic signed [63:0] wide_hold [0:7], product [0:7], add_rhs [0:7], add_result [0:7];
  logic signed [63:0] rounded [0:7];
  logic [3:0] sat_count;
  logic have_p, have_j;
  logic [8:0] row;
  typedef enum logic [3:0] {IDLE, GET, READ_A, READ_B, MULTIPLY, ADD_BIAS,
                            ADD_IDENTITY, ROUND, SEND, FINISH} state_t;
  state_t state;

  // Arithmetic floor quotient plus nonnegative remainder implements RNE
  // correctly for negative ties too. Input is the exact signed64 wide sum.
  function automatic logic signed [63:0] rne26(input logic signed [63:0] value);
    logic signed [63:0] quotient;
    logic increment;
    begin
      quotient=value>>>26;
      increment=(value[25:0]>26'h2000000) ||
                ((value[25:0]==26'h2000000) && quotient[0]);
      rne26=quotient+64'(increment);
    end
  endfunction

  always_comb begin
    raw_ready=(state==GET && !have_p);
    identity_request_valid=(state==GET && !have_j);
    identity_address=row;
    result_valid=(state==SEND);result_addr=row;
    sat_count=0;
    for(integer l=0;l<8;l=l+1) begin
      product[l]=$signed(p_hold[l])*$signed(a_hold[l]);
      add_rhs[l]=64'sd0;
      if(state==ADD_BIAS) add_rhs[l]=$signed({{32{b_hold[l][31]}},b_hold[l]})<<<20;
      if(state==ADD_IDENTITY) add_rhs[l]=$signed({{32{j_hold[l][31]}},j_hold[l]})<<<20;
      add_result[l]=wide_hold[l]+add_rhs[l];
      rounded[l]=rne26(wide_hold[l]);
      if(rounded[l]>64'sd8388607 || rounded[l]<-64'sd8388608) sat_count=sat_count+1;
    end
  end

  always_ff @(posedge clk) begin
    if(!reset_n) begin
      state<=IDLE;row<=0;have_p<=0;have_j<=0;result_data<=0;done<=0;error<=0;
      cycles<=0;raw_words<=0;identity_words<=0;coefficient_words<=0;
      mul_issues<=0;add_issues<=0;round_issues<=0;output_words<=0;
      identity_stalls<=0;raw_wait_cycles<=0;join_wait_cycles<=0;output_stalls<=0;saturations<=0;
      for(integer l=0;l<8;l=l+1) begin
        p_hold[l]<=0;j_hold[l]<=0;a_hold[l]<=0;b_hold[l]<=0;wide_hold[l]<=0;
      end
    end else begin
      done<=0;
      if(cfg_valid && state==IDLE)
        for(integer l=0;l<8;l=l+1) coefficient[l][cfg_addr]<=cfg_data[l*32+:32];
      if(state!=IDLE) cycles<=cycles+1;
      case(state)
        IDLE: if(start) begin
          state<=GET;row<=0;have_p<=0;have_j<=0;error<=0;
          cycles<=0;raw_words<=0;identity_words<=0;coefficient_words<=0;
          mul_issues<=0;add_issues<=0;round_issues<=0;output_words<=0;
          identity_stalls<=0;raw_wait_cycles<=0;join_wait_cycles<=0;output_stalls<=0;saturations<=0;
        end
        GET: begin
          if(!((have_p || raw_valid) && (have_j || identity_valid)))
            join_wait_cycles<=join_wait_cycles+1;
          if(!have_p) begin
            if(raw_valid) begin
              for(integer l=0;l<8;l=l+1) p_hold[l]<=raw_data[l*32+:32];
              have_p<=1;raw_words<=raw_words+1;
              if(raw_addr!=row) error<=1;
            end else raw_wait_cycles<=raw_wait_cycles+1;
          end
          if(!have_j) begin
            if(identity_valid) begin
              for(integer l=0;l<8;l=l+1) j_hold[l]<=identity_data[l*32+:32];
              have_j<=1;identity_words<=identity_words+1;
            end else identity_stalls<=identity_stalls+1;
          end
          // Reuse the same a/b registers across all forty P/T members of
          // one N8 group. Both producer modes receive this ordinary reuse.
          if((have_p || raw_valid) && (have_j || identity_valid))
            state<=((int'(row)%40)==0)?READ_A:MULTIPLY;
        end
        READ_A: begin
          for(integer l=0;l<8;l=l+1) a_hold[l]<=coefficient[l][2*(int'(row)/40)];
          coefficient_words<=coefficient_words+1;state<=READ_B;
        end
        READ_B: begin
          for(integer l=0;l<8;l=l+1) b_hold[l]<=coefficient[l][2*(int'(row)/40)+1];
          coefficient_words<=coefficient_words+1;state<=MULTIPLY;
        end
        MULTIPLY: begin
          for(integer l=0;l<8;l=l+1) wide_hold[l]<=product[l];
          mul_issues<=mul_issues+1;state<=ADD_BIAS;
        end
        ADD_BIAS: begin
          for(integer l=0;l<8;l=l+1) wide_hold[l]<=add_result[l];
          add_issues<=add_issues+1;state<=ADD_IDENTITY;
        end
        ADD_IDENTITY: begin
          for(integer l=0;l<8;l=l+1) wide_hold[l]<=add_result[l];
          add_issues<=add_issues+1;state<=ROUND;
        end
        ROUND: begin
          for(integer l=0;l<8;l=l+1) begin
            if(rounded[l]>64'sd8388607) result_data[l*32+:32]<=32'd8388607;
            else if(rounded[l]<-64'sd8388608) result_data[l*32+:32]<=-32'sd8388608;
            else result_data[l*32+:32]<=rounded[l][31:0];
          end
          saturations<=saturations+32'(sat_count);round_issues<=round_issues+1;state<=SEND;
        end
        SEND: if(result_ready) begin
          output_words<=output_words+1;
          if(row==479) state<=FINISH;
          else begin row<=row+1;have_p<=0;have_j<=0;state<=GET;end
        end else output_stalls<=output_stalls+1;
        FINISH: begin done<=1;state<=IDLE;end
        default: begin error<=1;state<=IDLE;end
      endcase
    end
  end
endmodule
