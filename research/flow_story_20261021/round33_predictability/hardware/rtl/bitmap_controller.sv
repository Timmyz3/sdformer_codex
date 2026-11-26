// Descriptor controller only. Existing descriptor RF is external, one32-bit1RW
// service, one outstanding synchronous read. No PE, predictor, SRAM macro or BN.
module bitmap_ctrl_core #(
  parameter bit DYNAMIC = 1'b1,
  parameter bit FIXED_NONE = 1'b0
)(
  input logic clk, rst_n,
  input logic start_valid, input logic [2:0] action,
  output logic start_ready, input logic finish_valid,
  output logic active, reject_start, reject_finish,
  input logic req_valid, output logic req_ready,
  input logic complete_tile, input logic [9:0] tile_id,
  output logic rsp_valid, input logic rsp_ready,
  output logic [2:0] start_plane,
  output logic mem_req, input logic mem_ready,
  output logic [7:0] mem_addr,
  input logic mem_rvalid, input logic [31:0] mem_rdata,
  output logic fault
);
  typedef enum logic [1:0] {READY, ISSUE, WAIT_READ, RESPONSE} phase_t;
  phase_t phase;
  logic [2:0] held_action;
  logic [4:0] bit_index;
  logic [7:0] read_address;
  logic no_upgrade;
  always_comb begin
    no_upgrade = DYNAMIC ? held_action==3'd6 : FIXED_NONE;
    start_ready = !active && phase==READY && !fault;
    req_ready = active && phase==READY && !fault && !finish_valid;
    rsp_valid = active && phase==RESPONSE && !fault;
    mem_req = active && phase==ISSUE && !fault;
    mem_addr = read_address;
  end
  always_ff @(posedge clk) begin
    if(!rst_n) begin
      phase<=READY;active<=0;held_action<=0;bit_index<=0;
      read_address<=0;start_plane<=0;fault<=0;
      reject_start<=0;reject_finish<=0;
    end else begin
      reject_start<=0;reject_finish<=0;
      if(start_valid) begin
        if(!start_ready)reject_start<=1;
        else if(DYNAMIC && action==3'd7)fault<=1;
        else begin
          active<=1;
          held_action<=DYNAMIC ? action : (FIXED_NONE ? 3'd6 : 3'd0);
        end
      end
      if(finish_valid) begin
        if(active && phase==READY && !req_valid && !fault)active<=0;
        else reject_finish<=1;
      end
      if(mem_rvalid && phase!=WAIT_READ)fault<=1;
      case(phase)
        READY: if(req_valid && req_ready) begin
          if(complete_tile && tile_id>=10'd910)fault<=1;
          else if(!complete_tile || no_upgrade) begin
            start_plane<=complete_tile ? 3'd4 : 3'd0;
            phase<=RESPONSE;
          end else begin
            bit_index<=tile_id[4:0];
            //128B-aligned bitmap: address concatenate replaces general add.
            read_address<=DYNAMIC ? {held_action,tile_id[9:5]} : {3'd0,tile_id[9:5]};
            phase<=ISSUE;
          end
        end
        ISSUE: if(mem_req && mem_ready)phase<=WAIT_READ;
        WAIT_READ: if(mem_rvalid)begin
          start_plane<=mem_rdata[bit_index] ? 3'd0 : 3'd4;
          phase<=RESPONSE;
        end
        RESPONSE: if(rsp_valid && rsp_ready)phase<=READY;
        default: fault<=1;
      endcase
    end
  end
endmodule

`define CTRL_PORTS \
  input logic clk,rst_n,start_valid, input logic [2:0] action, \
  output logic start_ready, input logic finish_valid, \
  output logic active,reject_start,reject_finish, \
  input logic req_valid,output logic req_ready, \
  input logic complete_tile,input logic [9:0] tile_id, \
  output logic rsp_valid,input logic rsp_ready,output logic [2:0] start_plane, \
  output logic mem_req,input logic mem_ready,output logic [7:0] mem_addr, \
  input logic mem_rvalid,input logic [31:0] mem_rdata,output logic fault
`define CTRL_CONNECT \
  .clk(clk),.rst_n(rst_n),.start_valid(start_valid),.action(action), \
  .start_ready(start_ready),.finish_valid(finish_valid),.active(active), \
  .reject_start(reject_start),.reject_finish(reject_finish), \
  .req_valid(req_valid),.req_ready(req_ready),.complete_tile(complete_tile),.tile_id(tile_id), \
  .rsp_valid(rsp_valid),.rsp_ready(rsp_ready),.start_plane(start_plane), \
  .mem_req(mem_req),.mem_ready(mem_ready),.mem_addr(mem_addr), \
  .mem_rvalid(mem_rvalid),.mem_rdata(mem_rdata),.fault(fault)
module bitmap_ctrl_dynamic (`CTRL_PORTS);
  bitmap_ctrl_core #(.DYNAMIC(1),.FIXED_NONE(0)) u (`CTRL_CONNECT);
endmodule
module bitmap_ctrl_static (`CTRL_PORTS);
  bitmap_ctrl_core #(.DYNAMIC(0),.FIXED_NONE(0)) u (`CTRL_CONNECT);
endmodule
module bitmap_ctrl_none (`CTRL_PORTS);
  bitmap_ctrl_core #(.DYNAMIC(0),.FIXED_NONE(1)) u (`CTRL_CONNECT);
endmodule
`undef CTRL_PORTS
`undef CTRL_CONNECT
