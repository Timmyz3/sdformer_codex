// E4 derived Q8/Q6/Q4 integer datapath experiment. No SRAM, scale, BN or full accelerator.
// Descriptor is presented with and before the first accepted row-group beat.
module binary_plane_pe (
  input logic clk, rst_n,
  input logic in_valid, input logic row_start,
  input logic [2:0] start_plane, input logic [15:0] k_chunks,
  input logic [127:0] activation,
  input logic [2047:0] weight_plane,
  output logic in_ready,
  output logic out_valid, input logic out_ready,
  output logic [511:0] result,
  output logic [31:0] row_beats,
  output logic [31:0] accepted_beats,
  output logic [2:0] expected_plane,
  output logic [15:0] expected_chunk,
  output logic fault
);
  logic active;
  logic [2:0] plane;
  logic [15:0] chunk, chunks;
  logic [31:0] beats;
  logic signed [31:0] accumulator [0:15];
  logic [2:0] current_plane;
  logic [15:0] current_chunk, current_chunks;
  logic last;
  logic [7:0] pop [0:15];
  logic signed [31:0] delta [0:15];
  logic signed [31:0] next_acc [0:15];

  function automatic [7:0] pop128(input logic [127:0] x);
    logic [1:0] a [0:63];
    logic [2:0] b [0:31];
    logic [3:0] c [0:15];
    logic [4:0] d [0:7];
    logic [5:0] e [0:3];
    logic [6:0] f [0:1];
    integer j;
    begin
      for(j=0;j<64;j=j+1)a[j]={1'b0,x[2*j]}+{1'b0,x[2*j+1]};
      for(j=0;j<32;j=j+1)b[j]={1'b0,a[2*j]}+{1'b0,a[2*j+1]};
      for(j=0;j<16;j=j+1)c[j]={1'b0,b[2*j]}+{1'b0,b[2*j+1]};
      for(j=0;j<8;j=j+1)d[j]={1'b0,c[2*j]}+{1'b0,c[2*j+1]};
      for(j=0;j<4;j=j+1)e[j]={1'b0,d[2*j]}+{1'b0,d[2*j+1]};
      for(j=0;j<2;j=j+1)f[j]={1'b0,e[2*j]}+{1'b0,e[2*j+1]};
      pop128={1'b0,f[0]}+{1'b0,f[1]};
    end
  endfunction
  always_comb begin
    in_ready=(!out_valid || out_ready) && !fault;
    current_plane=active ? plane : start_plane;
    current_chunk=active ? chunk : 16'd0;
    current_chunks=active ? chunks : k_chunks;
    expected_plane=current_plane;
    expected_chunk=current_chunk;
    last=(current_chunk+16'd1==current_chunks) && current_plane==3'd7;
    for(integer o=0;o<16;o=o+1)begin
      pop[o]=pop128(activation & weight_plane[o*128+:128]);
      delta[o]=$signed({24'd0,pop[o]}) <<< current_plane;
      if(current_plane==3'd7)delta[o]=-delta[o];
      next_acc[o]=(active ? accumulator[o] : 32'sd0)+delta[o];
    end
  end
  always_ff @(posedge clk)begin
    if(!rst_n)begin
      active<=0;plane<=0;chunk<=0;chunks<=0;beats<=0;
      out_valid<=0;result<='0;row_beats<=0;accepted_beats<=0;fault<=0;
      for(integer o=0;o<16;o=o+1)accumulator[o]<=0;
    end else begin
      if(out_valid && out_ready)out_valid<=0;
      if(in_valid && in_ready)begin
        if((!active && (!row_start || k_chunks==0 || !(start_plane==0 || start_plane==2 || start_plane==4))) || (active && row_start))begin
          fault<=1;active<=0;out_valid<=0;
        end else begin
          accepted_beats<=accepted_beats+1;
          beats<=active ? beats+1 : 1;
          chunks<=current_chunks;
          for(integer o=0;o<16;o=o+1)accumulator[o]<=next_acc[o];
          if(last)begin
            active<=0;out_valid<=1;
            row_beats<=active ? beats+1 : 1;
            for(integer o=0;o<16;o=o+1)result[o*32+:32]<=next_acc[o];
          end else begin
            active<=1;
            if(current_chunk+16'd1==current_chunks)begin
              chunk<=0;plane<=current_plane+1;
            end else begin chunk<=current_chunk+1;plane<=current_plane;end
          end
        end
      end
    end
  end
endmodule

// Strong ordinary signed8-weight/binary-spike gate-add,16 reductions×16outputs.
// Same 2048-bit weight port and32-bit output accumulators as the binary engine.
module parallel_gateadd_pe (
  input logic clk,rst_n,in_valid,row_start,
  input logic [15:0] k_chunks,
  input logic [15:0] activation,
  input logic [2047:0] weights,
  output logic in_ready,out_valid,input logic out_ready,
  output logic [511:0] result,
  output logic [31:0] row_beats,accepted_beats,
  output logic [15:0] expected_chunk,
  output logic fault
);
  logic active;
  logic [15:0] chunk,chunks,current_chunk,current_chunks;
  logic [31:0] beats;
  logic signed [31:0] accumulator[0:15],delta[0:15],next_acc[0:15];
  logic last;
  function automatic signed [11:0] dot16(input logic[15:0] x,input logic[127:0] w);
    logic signed[7:0] a[0:15];
    logic signed[8:0] b[0:7];
    logic signed[9:0] c[0:3];
    logic signed[10:0] d[0:1];
    integer j;
    begin
      for(j=0;j<16;j=j+1)a[j]=x[j] ? $signed(w[j*8+:8]) : 8'sd0;
      for(j=0;j<8;j=j+1)b[j]=$signed({a[j*2][7],a[j*2]})+$signed({a[j*2+1][7],a[j*2+1]});
      for(j=0;j<4;j=j+1)c[j]=$signed({b[j*2][8],b[j*2]})+$signed({b[j*2+1][8],b[j*2+1]});
      for(j=0;j<2;j=j+1)d[j]=$signed({c[j*2][9],c[j*2]})+$signed({c[j*2+1][9],c[j*2+1]});
      dot16=$signed({d[0][10],d[0]})+$signed({d[1][10],d[1]});
    end
  endfunction
  always_comb begin
    in_ready=(!out_valid || out_ready) && !fault;
    current_chunk=active ? chunk : 0;
    current_chunks=active ? chunks : k_chunks;
    expected_chunk=current_chunk;
    last=current_chunk+16'd1==current_chunks;
    for(integer o=0;o<16;o=o+1)begin
      delta[o]=$signed(dot16(activation,weights[o*128+:128]));
      next_acc[o]=(active ? accumulator[o] : 0)+delta[o];
    end
  end
  always_ff @(posedge clk)begin
    if(!rst_n)begin
      active<=0;chunk<=0;chunks<=0;beats<=0;
      out_valid<=0;result<='0;row_beats<=0;accepted_beats<=0;fault<=0;
      for(integer o=0;o<16;o=o+1)accumulator[o]<=0;
    end else begin
      if(out_valid && out_ready)out_valid<=0;
      if(in_valid && in_ready)begin
        if((!active && (!row_start || k_chunks==0)) || (active && row_start))begin
          fault<=1;active<=0;out_valid<=0;
        end else begin
          accepted_beats<=accepted_beats+1;beats<=active ? beats+1 : 1;chunks<=current_chunks;
          for(integer o=0;o<16;o=o+1)accumulator[o]<=next_acc[o];
          if(last)begin
            active<=0;out_valid<=1;row_beats<=active ? beats+1 : 1;
            for(integer o=0;o<16;o=o+1)result[o*32+:32]<=next_acc[o];
          end else begin active<=1;chunk<=current_chunk+1;end
        end
      end
    end
  end
endmodule

// Simulation-only selection wrapper. Synthesis targets the two modules above separately.
module pe_compare_top(
  input logic clk,rst_n,parallel_mode,in_valid,row_start,
  input logic [2:0] start_plane,
  input logic [15:0] k_chunks,
  input logic [127:0] activation,
  input logic [2047:0] weights,
  output logic in_ready,out_valid,input logic out_ready,
  output logic [511:0] result,
  output logic [31:0] row_beats,accepted_beats,
  output logic [2:0] expected_plane,
  output logic [15:0] expected_chunk,
  output logic fault
);
  wire br,bv,bf,pr,pv,pf;wire[511:0]by,py;wire[31:0]bb,ba,pb,pa;wire[2:0]bp;wire[15:0]bc,pc;
  binary_plane_pe b(clk,rst_n,in_valid&&!parallel_mode,row_start,start_plane,k_chunks,activation,weights,br,bv,out_ready&&!parallel_mode,by,bb,ba,bp,bc,bf);
  parallel_gateadd_pe p(clk,rst_n,in_valid&&parallel_mode,row_start,k_chunks,activation[15:0],weights,pr,pv,out_ready&&parallel_mode,py,pb,pa,pc,pf);
  always_comb begin
    in_ready=parallel_mode?pr:br;out_valid=parallel_mode?pv:bv;result=parallel_mode?py:by;
    row_beats=parallel_mode?pb:bb;accepted_beats=parallel_mode?pa:ba;
    expected_plane=parallel_mode?0:bp;expected_chunk=parallel_mode?pc:bc;fault=parallel_mode?pf:bf;
  end
endmodule
