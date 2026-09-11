// Exact subset relation from a stream of distinct source-column masks.
// ProSparsity parent rule is borrowed from Prosperity (HPCA 2025).
// Counterexample elimination is an existing set-inclusion algorithm.
module column_relation_matcher #(
    parameter integer TOKENS=20,
    parameter integer MAX_K=864,
    parameter integer IW=$clog2(TOKENS),
    parameter integer CW=$clog2(MAX_K+1)
) (
    input logic clk, rst_n,
    input logic start,
    output logic start_ready,
    input logic column_valid,
    output logic column_ready,
    input logic [TOKENS-1:0] column_mask,
    input logic finish,
    output logic parent_valid,
    input logic parent_ready,
    output logic [IW-1:0] child_index,
    output logic has_parent,
    output logic [IW-1:0] parent_index,
    output logic [CW-1:0] child_popcount,
    output logic parent_last,
    input logic residual_valid,
    output logic residual_ready,
    input logic [TOKENS-1:0] residual_mask,
    output logic residual_out_valid,
    input logic residual_out_ready,
    output logic [TOKENS-1:0] residual_out_mask
);
    typedef enum logic [1:0] {IDLE,FILL,SELECT,REPLAY} state_t;
    state_t state;
    logic [TOKENS-1:0] eligible [TOKENS];
    logic [CW-1:0] count [TOKENS];
    logic [IW-1:0] chosen [TOKENS];
    logic [TOKENS-1:0] chosen_valid;
    logic [IW-1:0] cursor;
    logic [CW-1:0] best;
    logic [TOKENS-1:0] filtered;
    always_comb begin
        start_ready=(state==IDLE || state==REPLAY) && !residual_out_valid;
        column_ready=state==FILL;
        parent_valid=state==SELECT;
        child_index=cursor;
        child_popcount=count[cursor];
        parent_last=cursor==IW'(TOKENS-1);
        parent_index='0;
        has_parent=1'b0;
        best='0;
        if(count[cursor]>=2) begin
            for(integer j=0;j<TOKENS;j=j+1) begin
                if(eligible[cursor][j] && count[j]>best &&
                   (count[j]<count[cursor] || j<int'(cursor))) begin
                    best=count[j];
                    parent_index=IW'(j);
                    has_parent=1'b1;
                end
            end
        end
        residual_ready=state==REPLAY && (!residual_out_valid || residual_out_ready) && !start;
        filtered=residual_mask;
        for(integer i=0;i<TOKENS;i=i+1)
            if(chosen_valid[i] && residual_mask[chosen[i]]) filtered[i]=1'b0;
    end
    always_ff @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;
            cursor<='0;
            chosen_valid<='0;
            residual_out_valid<=1'b0;
            residual_out_mask<='0;
            for(integer i=0;i<TOKENS;i=i+1) begin
                eligible[i]<='1;
                count[i]<='0;
                chosen[i]<='0;
            end
        end else begin
            if(residual_out_valid && residual_out_ready) residual_out_valid<=1'b0;
            if(start && start_ready) begin
                state<=FILL;
                cursor<='0;
                chosen_valid<='0;
                for(integer i=0;i<TOKENS;i=i+1) begin
                    eligible[i]<='1;
                    count[i]<='0;
                    chosen[i]<='0;
                end
            end else case(state)
                FILL: begin
                    if(column_valid && column_ready)
                        for(integer i=0;i<TOKENS;i=i+1) begin
                            if(column_mask[i]) count[i]<=count[i]+1'b1;
                            else eligible[i]<=eligible[i] & ~column_mask;
                        end
                    if(finish) begin
                        state<=SELECT;
                        cursor<='0;
                    end
                end
                SELECT: if(parent_valid && parent_ready) begin
                    chosen[cursor]<=parent_index;
                    chosen_valid[cursor]<=has_parent;
                    if(parent_last) state<=REPLAY;
                    else cursor<=cursor+1'b1;
                end
                REPLAY: if(residual_valid && residual_ready) begin
                    residual_out_mask<=filtered;
                    residual_out_valid<=1'b1;
                end
                default: ;
            endcase
        end
    end
endmodule
