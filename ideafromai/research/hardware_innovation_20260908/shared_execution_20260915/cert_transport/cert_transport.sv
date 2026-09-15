// Measured post-Y / known-threshold transport experiment. Adapted from the
// project's Claude T5 arithmetic, with a serialized memory interface, RTL
// exponent/plane construction, shift/sub bounds, and held output retirement.
// Not a complete FC1 / BN / continuous-consumer implementation.
module cert_transport(
    input logic clk, rst_n,
    input logic cmd_valid, output logic cmd_ready,
    input logic [1:0] cmd_mode, // 0 FXfull,1 FXcert,2 BFfull,3 BFcert
    input logic [11:0] cmd_h,
    output logic req_valid, input logic req_ready,
    output logic req_tau, output logic [3:0] req_index,
    output logic [11:0] req_h,
    input logic rsp_valid, output logic rsp_ready,
    input logic [63:0] rsp_data,
    output logic out_valid, input logic out_ready,
    output logic [9:0] out_dec,
    output logic [4:0] out_exp, out_planes,
    output logic [7:0] dbg_state
);
    localparam IDLE=0, REQUEST=1, RESPONSE=2, START=3, PLANE=4, FINISH=5;
    logic [2:0] state;
    logic [1:0] mode;
    logic [4:0] fetch, exponent, m, fed;
    logic signed [23:0] y [0:9];
    logic [63:0] tau [0:9];
    logic signed [15:0] a [0:99];
    logic signed [47:0] pn [0:19];
    logic signed [47:0] vtop [0:9];
    logic [9:0] locked, dec, sign_word, plane_word, lock_next, dec_next;
    logic signed [47:0] nv [0:9], lo [0:9], hi [0:9];
    integer t;
    function automatic [4:0] bit_length_abs(input logic signed [23:0] v);
        logic [24:0] magnitude;
        integer k;
        begin
            magnitude = v[23] ? -$signed({v[23],v}) : {1'b0,v};
            bit_length_abs = 0;
            for (k=0;k<25;k=k+1)
                if (magnitude[k]) bit_length_abs=k[4:0]+5'd1;
        end
    endfunction
    function automatic signed [47:0] dot(input integer tt, input logic [9:0] bits_in);
        integer j;
        logic signed [47:0] sum;
        begin
            sum=0;
            for(j=0;j<10;j=j+1)
                if(bits_in[j]) sum=sum+$signed({{32{a[tt*10+j][15]}},a[tt*10+j]});
            dot=sum;
        end
    endfunction
    always_comb begin
        cmd_ready=(state==IDLE);
        req_valid=(state==REQUEST);
        req_tau=(fetch>=5);
        req_index=(fetch>=5) ? fetch[3:0]-4'd5 : fetch[3:0];
        rsp_ready=(state==RESPONSE);
        out_valid=(state==FINISH);
        out_dec=dec;
        out_exp=exponent;
        out_planes=fed;
        dbg_state={5'd0,state};
        for(integer j=0;j<10;j=j+1) begin
            sign_word[j]=y[j][23];
            plane_word[j]=y[j][m];
        end
        lock_next=locked;
        dec_next=dec;
        for(integer i=0;i<10;i=i+1) begin
            nv[i]=(vtop[i]<<<1)+dot(i,plane_word);
            // N/P * (2^m-1), exactly, with no generic multiplier.
            lo[i]=(nv[i]<<<m)+(pn[2*i+1]<<<m)-pn[2*i+1];
            hi[i]=(nv[i]<<<m)+(pn[2*i]<<<m)-pn[2*i];
            if(!locked[i] && ((lo[i]>=$signed(tau[i][47:0])) ||
                              (hi[i]<$signed(tau[i][47:0])))) begin
                lock_next[i]=1;
                dec_next[i]=(lo[i]>=$signed(tau[i][47:0])) ^ !tau[i][63];
            end
        end
    end
    always_ff @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE; mode<=0; fetch<=0; exponent<=0; m<=0; fed<=0;
            locked<=0; dec<=0; req_h<=0;
            for(t=0;t<10;t=t+1) begin y[t]<=0; tau[t]<=0; vtop[t]<=0; end
        end else begin
            case(state)
                IDLE: if(cmd_valid) begin
                    req_h<=cmd_h; mode<=cmd_mode; fetch<=0; exponent<=0;
                    fed<=0; locked<=0; dec<=0; state<=REQUEST;
                end
                REQUEST: if(req_ready) state<=RESPONSE;
                RESPONSE: if(rsp_valid) begin
                    if(fetch<5) begin
                        // Two signed24 source values per 64-bit transfer.
                        y[2*fetch]<=rsp_data[23:0];
                        y[2*fetch+1]<=rsp_data[47:24];
                        if(bit_length_abs($signed(rsp_data[23:0]))>exponent)
                            exponent<=bit_length_abs($signed(rsp_data[23:0]));
                        if(bit_length_abs($signed(rsp_data[47:24]))>exponent &&
                           bit_length_abs($signed(rsp_data[47:24]))>=
                           bit_length_abs($signed(rsp_data[23:0])))
                            exponent<=bit_length_abs($signed(rsp_data[47:24]));
                    end else tau[fetch-5]<=rsp_data;
                    if(fetch==14) state<=START;
                    else begin fetch<=fetch+1; state<=REQUEST; end
                end
                START: begin
                    // signed24 already has Y>>23=-sign. A fixed-width mode
                    // need not resend bit23 after the sign header.
                    m<=mode[1] ? ((exponent==0)?0:exponent-1) : 22;
                    for(t=0;t<10;t=t+1) vtop[t]<=-dot(t,sign_word);
                    state<=PLANE;
                end
                PLANE: begin
                    for(t=0;t<10;t=t+1) vtop[t]<=nv[t];
                    locked<=lock_next; dec<=dec_next; fed<=fed+1;
                    if(m==0 || (mode[0] && (&lock_next))) state<=FINISH;
                    else m<=m-1;
                end
                FINISH: if(out_ready) state<=IDLE;
                default: state<=IDLE;
            endcase
        end
    end
    reg [8*512-1:0] fa, fp;
    initial begin
        if(!$value$plusargs("a=%s",fa)) $fatal(1,"missing a");
        if(!$value$plusargs("pn=%s",fp)) $fatal(1,"missing pn");
        $readmemh(fa,a);
        $readmemh(fp,pn);
    end
endmodule
