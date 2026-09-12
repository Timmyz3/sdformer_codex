// Common signed48, eight-lane temporal source executor. The program is data:
// both ordinary and lifting use this same module, RF, ROM and port interface.
module temporal_source (
    input logic clk, rst_n,
    input logic cfg_we, input logic [8:0] cfg_addr, input logic [127:0] cfg_data,
    input logic start, input logic [31:0] input_base, output_base,
    input logic [15:0] input_stride,
    output logic idle, done,
    output logic rd_valid, input logic rd_ready, output logic [31:0] rd_addr,
    input logic rsp_valid, input logic [63:0] rsp_data,
    output logic wr_valid, input logic wr_ready,
    output logic [31:0] wr_addr, output logic [63:0] wr_data,
    output logic wb_valid, output logic [6:0] wb_dst,
    output logic [383:0] wb_data
);
    localparam [2:0] NOP=0, LOAD=1, ADD=2, NORM=3, GATE=4, COMMIT=5;
    typedef enum logic [3:0] {IDLE, EXEC, RD_REQ, RD_RSP, LOAD_ISSUE,
        LOAD_WAIT, GATE_WAIT, GATE_COLLECT, COMMIT_WAIT, STORE} state_t;
    state_t state;
    logic [127:0] rom [0:511];
    logic signed [47:0] rf [0:95][0:7];
    logic [8:0] pc;
    logic [31:0] ibase, obase;
    logic [15:0] stride;
    logic [1:0] beat;
    logic store_beat;
    logic [191:0] gather;
    logic [15:0] gate_words [0:7];
    wire [127:0] ins=rom[pc];
    wire [2:0] kind=ins[2:0];
    wire [6:0] dst=ins[9:3], src_a=ins[16:10], src_b=ins[23:17];
    wire [5:0] sh_a=ins[29:24], sh_b=ins[35:30], rne_shift=ins[43:38];
    wire neg_a=ins[36], neg_b=ins[37];
    wire [3:0] time_index=ins[47:44];
    wire signed [47:0] threshold=ins[95:48];
    wire direction_negative=ins[96];
    wire [1:0] constant_code=ins[98:97]; // 0 variable, 1 false, 2 true

    logic p1_valid, p2_valid;
    logic [2:0] p1_kind;
    logic [6:0] p1_dst, p2_dst;
    logic [5:0] p1_shift;
    logic signed [47:0] p1_threshold;
    logic p1_direction;
    logic [1:0] p1_constant;
    logic signed [47:0] p1_a [0:7], p1_b [0:7], p2_value [0:7];
    logic hazard, issue;
    logic [6:0] issue_dst;
    integer lane;

    function automatic logic signed [47:0] normalize24(
        input logic signed [47:0] x, input logic [5:0] shift);
        logic signed [47:0] q;
        logic [47:0] rem_bits, divisor;
        begin
            q=x >>> shift;
            divisor=48'd1 << shift;
            rem_bits=$unsigned(x) & (divisor-48'd1);
            if (shift!=0 && ((rem_bits>(divisor>>1)) ||
                    ((rem_bits==(divisor>>1)) && q[0]))) q=q+48'sd1;
            if (q>48'sd8388607) normalize24=48'sd8388607;
            else if (q< -48'sd8388608) normalize24= -48'sd8388608;
            else normalize24=q;
        end
    endfunction

    always_comb begin
        idle=(state==IDLE);
        rd_valid=(state==RD_REQ);
        rd_addr=ibase+32'(time_index)*32'(stride)+32'(beat)*32'd8;
        wr_valid=(state==STORE);
        wr_addr=obase+(store_beat ? 32'd8 : 32'd0);
        wr_data='0;
        for (integer j=0;j<4;j=j+1)
            wr_data[j*16+:16]=gate_words[j+(store_beat?4:0)];
        wb_valid=p2_valid;
        wb_dst=p2_dst;
        for (integer j=0;j<8;j=j+1) wb_data[j*48+:48]=p2_value[j];
        issue_dst=(kind==GATE) ? 7'd95 : dst;
        hazard=(p1_valid && (p1_dst==issue_dst || p1_dst==src_a ||
                    (kind==ADD && p1_dst==src_b))) ||
                (p2_valid && (p2_dst==issue_dst || p2_dst==src_a ||
                    (kind==ADD && p2_dst==src_b)));
        issue=(state==EXEC && (kind==ADD || kind==NORM || kind==GATE) && !hazard)
            || (state==LOAD_ISSUE && !p1_valid && !p2_valid);
    end

    always_ff @(posedge clk) begin
        if (cfg_we && idle) rom[cfg_addr]<=cfg_data;
        if (!rst_n) begin
            state<=IDLE; pc<='0; done<=0; p1_valid<=0; p2_valid<=0;
            beat<=0; store_beat<=0; gather<='0; ibase<=0; obase<=0; stride<=0;
            for (lane=0;lane<8;lane=lane+1) gate_words[lane]<=0;
        end else begin
            done<=0;
            p2_valid<=p1_valid;
            p2_dst<=p1_dst;
            for (lane=0;lane<8;lane=lane+1) begin
                if (p2_valid) rf[p2_dst][lane]<=p2_value[lane];
                if (p1_valid) begin
                    case (p1_kind)
                        ADD: p2_value[lane]<=p1_a[lane]+p1_b[lane];
                        NORM: p2_value[lane]<=normalize24(p1_a[lane],p1_shift);
                        GATE: begin
                            if (p1_constant!=0) p2_value[lane]<=(p1_constant==2)?48'sd1:48'sd0;
                            else if (p1_direction) p2_value[lane]<=(p1_a[lane]<=p1_threshold)?48'sd1:48'sd0;
                            else p2_value[lane]<=(p1_a[lane]>=p1_threshold)?48'sd1:48'sd0;
                        end
                        default: p2_value[lane]<=p1_a[lane];
                    endcase
                end
            end
            p1_valid<=issue;
            if (issue) begin
                p1_kind<=(state==LOAD_ISSUE)?LOAD:kind;
                p1_dst<=issue_dst; p1_shift<=rne_shift;
                p1_threshold<=threshold; p1_direction<=direction_negative;
                p1_constant<=constant_code;
                for (lane=0;lane<8;lane=lane+1) begin
                    if (state==LOAD_ISSUE) begin
                        p1_a[lane]<={{24{gather[lane*24+23]}},gather[lane*24+:24]};
                        p1_b[lane]<=0;
                    end else begin
                        p1_a[lane]<=neg_a ? -(rf[src_a][lane] <<< sh_a) : (rf[src_a][lane] <<< sh_a);
                        p1_b[lane]<=neg_b ? -(rf[src_b][lane] <<< sh_b) : (rf[src_b][lane] <<< sh_b);
                    end
                end
            end
            case (state)
                IDLE: if (start) begin
                    pc<=0; ibase<=input_base; obase<=output_base; stride<=input_stride;
                    for (lane=0;lane<8;lane=lane+1) gate_words[lane]<=0;
                    state<=EXEC;
                end
                EXEC: begin
                    if (kind==LOAD) begin beat<=0; state<=RD_REQ; end
                    else if (kind==NOP) pc<=pc+9'd1;
                    else if (kind==COMMIT) state<=COMMIT_WAIT;
                    else if (issue) begin
                        if (kind==GATE) state<=GATE_WAIT;
                        else pc<=pc+9'd1;
                    end
                end
                RD_REQ: if (rd_ready) state<=RD_RSP;
                RD_RSP: if (rsp_valid) begin
                    gather[beat*64+:64]<=rsp_data;
                    if (beat==2) state<=LOAD_ISSUE;
                    else begin beat<=beat+2'd1; state<=RD_REQ; end
                end
                LOAD_ISSUE: if (issue) state<=LOAD_WAIT;
                LOAD_WAIT: if (!p1_valid && !p2_valid) begin pc<=pc+9'd1; state<=EXEC; end
                GATE_WAIT: if (!p1_valid && !p2_valid) state<=GATE_COLLECT;
                GATE_COLLECT: begin
                    for (lane=0;lane<8;lane=lane+1) gate_words[lane][time_index]<=rf[95][lane][0];
                    pc<=pc+9'd1; state<=EXEC;
                end
                COMMIT_WAIT: if (!p1_valid && !p2_valid) begin store_beat<=0; state<=STORE; end
                STORE: if (wr_ready) begin
                    if (store_beat) begin done<=1; state<=IDLE; end
                    else store_beat<=1;
                end
                default: state<=IDLE;
            endcase
        end
    end
endmodule
