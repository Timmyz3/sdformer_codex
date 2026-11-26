// Native binary ProSparsity baseline, M80/K16/H8, C96 or C384.
// Actual raw-source reads, online max-subset matching and legal issue order.
// No offline parents; no free parent value, no external intermediate Y.
// 80 parallel subset comparators + balanced selection trees are NOT free area.
module native_pr (
    input logic clk,rst_n,
    input logic cfg_valid,cfg_kind,
    input logic [15:0] cfg_addr,
    input logic [127:0] cfg_data,
    output logic cfg_ready,
    input logic start,
    input logic [8:0] c_count,
    input logic [1:0] mode,
    input logic src_enable,coef_enable,
    output logic busy,done,
    output logic out_valid,
    input logic out_ready,
    output logic [6:0] out_row,
    output logic [191:0] out_data,
    output logic [31:0] dbg_source_reads,dbg_coef_reads,dbg_add_cycles,dbg_build_cycles,
    output logic [31:0] dbg_match_cycles,dbg_sort_cycles,
    output logic [31:0] dbg_parent_reads,dbg_parent_writes,
    output logic [31:0] dbg_y_reads,dbg_y_writes,dbg_k_tiles
);
    logic [127:0] source_mem [0:239];
    logic [63:0] weight_mem [0:383];
    logic [191:0] y_mem [0:79];
    // Holds one raw 128-channel block for all 80 rows, reused for 8 K16 tiles.
    logic [127:0] source_cache [0:79];
    // One read OR one write per clock. Contains only the CURRENT K16 result.
    logic [191:0] parent_mem [0:79];
    logic [15:0] source_masks [0:79],delta_masks [0:79];
    logic [4:0] popcounts [0:79];
    logic [6:0] parents [0:79];
    logic [79:0] parent_valid,match_remaining,issue_remaining;
    logic signed [23:0] acc [0:7];
    logic [191:0] acc_word,y_hold;
    logic [63:0] read_weight;
    logic read_pending;
    logic [15:0] active_delta;
    logic [8:0] columns;
    logic [4:0] k_tile;
    logic [6:0] load_row,mask_row,active_row,row_id;

    // Binary trees: leaves 128..207, unused leaves padded invalid.
    logic [4:0] match_score [0:255];
    logic [6:0] match_index [0:255];
    logic [11:0] issue_key [0:255];
    logic [6:0] query_row;
    logic query_found;
    logic delta_found;
    logic [3:0] delta_bit;
    logic [191:0] alu_result;
    logic signed [23:0] alu_rhs [0:7];
    typedef enum logic [3:0] {
        IDLE,Y_INIT,SOURCE_LOAD,MASK_BUILD,MATCH,ISSUE,
        PARENT_READ,DELTA_RUN,PUBLISH,Y_ACCUM,NEXT_K,OUT_FETCH,OUT_HOLD
    } phase_t;
    phase_t phase;

    function automatic [4:0] pop16(input logic [15:0] x);
        logic [4:0] s;
        begin
            s=0;
            for(int b=0;b<16;b++)s=s+5'(x[b]);
            return s;
        end
    endfunction

    assign cfg_ready=(phase==IDLE)&&!start;
    assign out_valid=(phase==OUT_HOLD);
    assign out_row=row_id;
    always_comb begin
        query_row=0;query_found=0;
        for(int i=0;i<80;i++)if(!query_found && match_remaining[i])begin
            query_row=7'(i);query_found=1;
        end
        for(int i=0;i<256;i++)begin
            match_score[i]=0;match_index[i]=127;issue_key[i]=12'hfff;
        end
        for(int i=0;i<80;i++)begin
            if(query_found && popcounts[i]!=0
                && ((source_masks[query_row] & source_masks[i])==source_masks[i])
                && !((source_masks[query_row]==source_masks[i]) && i>=int'(query_row)))begin
                match_score[128+i]=popcounts[i];match_index[128+i]=7'(i);
            end
            if(issue_remaining[i])issue_key[128+i]={popcounts[i],7'(i)};
        end
        for(int i=127;i>=1;i--)begin
            if(match_score[2*i]>match_score[2*i+1] ||
              (match_score[2*i]==match_score[2*i+1] && match_index[2*i]<match_index[2*i+1]))begin
                match_score[i]=match_score[2*i];match_index[i]=match_index[2*i];
            end else begin
                match_score[i]=match_score[2*i+1];match_index[i]=match_index[2*i+1];
            end
            issue_key[i]=(issue_key[2*i]<=issue_key[2*i+1])?issue_key[2*i]:issue_key[2*i+1];
        end
        delta_found=0;delta_bit=0;
        for(int b=0;b<16;b++)if(!delta_found && active_delta[b])begin
            delta_found=1;delta_bit=4'(b);
        end
        acc_word=0;alu_result=0;
        for(int h=0;h<8;h++)begin
            acc_word[h*24+:24]=acc[h];
            alu_rhs[h]=(phase==Y_ACCUM)?$signed(y_hold[h*24+:24]):24'($signed(read_weight[h*8+:8]));
            alu_result[h*24+:24]=acc[h]+alu_rhs[h];
        end
    end

    always_ff @(posedge clk)begin
        if(!rst_n)begin
            phase<=IDLE;busy<=0;done<=0;out_data<=0;row_id<=0;
            columns<=0;k_tile<=0;load_row<=0;mask_row<=0;active_row<=0;
            parent_valid<=0;match_remaining<=0;issue_remaining<=0;
            active_delta<=0;read_pending<=0;read_weight<=0;y_hold<=0;
            dbg_source_reads<=0;dbg_coef_reads<=0;dbg_add_cycles<=0;dbg_build_cycles<=0;
            dbg_match_cycles<=0;dbg_sort_cycles<=0;dbg_parent_reads<=0;dbg_parent_writes<=0;
            dbg_y_reads<=0;dbg_y_writes<=0;dbg_k_tiles<=0;
            for(int h=0;h<8;h++)acc[h]<=0;
        end else begin
            done<=0;
            if(cfg_valid && cfg_ready)begin
                if(cfg_kind)weight_mem[cfg_addr]<=cfg_data[63:0];
                else source_mem[cfg_addr]<=cfg_data;
            end
            case(phase)
            IDLE:if(start)begin
                busy<=1;columns<=c_count;k_tile<=0;load_row<=0;row_id<=0;
                read_pending<=0;phase<=Y_INIT;
                dbg_source_reads<=0;dbg_coef_reads<=0;dbg_add_cycles<=0;dbg_build_cycles<=0;
                dbg_match_cycles<=0;dbg_sort_cycles<=0;dbg_parent_reads<=0;dbg_parent_writes<=0;
                dbg_y_reads<=0;dbg_y_writes<=0;dbg_k_tiles<=0;
            end
            Y_INIT:begin
                y_mem[row_id]<=0;dbg_y_writes<=dbg_y_writes+1;
                if(row_id==79)begin row_id<=0;load_row<=0;phase<=SOURCE_LOAD;end
                else row_id<=row_id+1;
            end
            SOURCE_LOAD:if(src_enable)begin
                // Synchronous one-word source service; cache can be used only next cycle.
                source_cache[load_row]<=source_mem[load_row*3+(k_tile>>3)];
                dbg_source_reads<=dbg_source_reads+1;
                if(load_row==79)begin mask_row<=0;phase<=MASK_BUILD;end
                else load_row<=load_row+1;
            end
            MASK_BUILD:begin
                // Eight small popcount16 units, same num_popcnt as official model.
                // Mask/metadata are register banks, not free SRAM 8-read ports.
                for(int j=0;j<8;j++)begin
                    source_masks[mask_row+j]<=source_cache[mask_row+j][k_tile[2:0]*16+:16];
                    delta_masks[mask_row+j]<=source_cache[mask_row+j][k_tile[2:0]*16+:16];
                    popcounts[mask_row+j]<=pop16(source_cache[mask_row+j][k_tile[2:0]*16+:16]);
                    parents[mask_row+j]<=0;parent_valid[mask_row+j]<=0;
                    match_remaining[mask_row+j]<=pop16(source_cache[mask_row+j][k_tile[2:0]*16+:16])>=2;
                    issue_remaining[mask_row+j]<=|source_cache[mask_row+j][k_tile[2:0]*16+:16];
                end
                dbg_build_cycles<=dbg_build_cycles+1;
                if(mask_row==72)begin
                    dbg_k_tiles<=dbg_k_tiles+1;phase<=MATCH;
                end else mask_row<=mask_row+8;
            end
            MATCH:begin
                if(query_found)begin
                    if(match_score[1]!=0)begin
                        parent_valid[query_row]<=1;parents[query_row]<=match_index[1];
                        delta_masks[query_row]<=source_masks[query_row]^source_masks[match_index[1]];
                    end
                    match_remaining[query_row]<=0;
                    dbg_match_cycles<=dbg_match_cycles+1;
                    dbg_build_cycles<=dbg_build_cycles+1;
                end else phase<=ISSUE;
            end
            ISSUE:begin
                if(issue_key[1]!=12'hfff)begin
                    active_row<=issue_key[1][6:0];
                    active_delta<=delta_masks[issue_key[1][6:0]];
                    issue_remaining[issue_key[1][6:0]]<=0;
                    for(int h=0;h<8;h++)acc[h]<=0;
                    read_pending<=0;
                    dbg_sort_cycles<=dbg_sort_cycles+1;
                    dbg_build_cycles<=dbg_build_cycles+1;
                    if(parent_valid[issue_key[1][6:0]])phase<=PARENT_READ;
                    else phase<=DELTA_RUN;
                end else phase<=NEXT_K;
            end
            PARENT_READ:begin
                // Actual synchronous SINGLE read of the partial parent row.
                for(int h=0;h<8;h++)acc[h]<=$signed(parent_mem[parents[active_row]][h*24+:24]);
                dbg_parent_reads<=dbg_parent_reads+1;
                phase<=DELTA_RUN;
            end
            DELTA_RUN:begin
                if(read_pending)begin
                    for(int h=0;h<8;h++)acc[h]<=$signed(alu_result[h*24+:24]);
                    dbg_add_cycles<=dbg_add_cycles+1;
                end
                read_pending<=0;
                if(delta_found && coef_enable)begin
                    read_weight<=weight_mem[k_tile*16+delta_bit];
                    active_delta[delta_bit]<=0;
                    read_pending<=1;dbg_coef_reads<=dbg_coef_reads+1;
                end
                if(!delta_found && !read_pending)phase<=PUBLISH;
            end
            PUBLISH:begin
                // Parent store and persistent-Y read are different single-port arrays.
                parent_mem[active_row]<=acc_word;
                y_hold<=y_mem[active_row];
                dbg_parent_writes<=dbg_parent_writes+1;dbg_y_reads<=dbg_y_reads+1;
                phase<=Y_ACCUM;
            end
            Y_ACCUM:begin
                // Reuse the same eight Acc24 lanes, not a second merge datapath.
                y_mem[active_row]<=alu_result;
                dbg_add_cycles<=dbg_add_cycles+1;dbg_y_writes<=dbg_y_writes+1;
                phase<=ISSUE;
            end
            NEXT_K:begin
                if((k_tile+1)*16>=columns)begin row_id<=0;phase<=OUT_FETCH;end
                else begin
                    k_tile<=k_tile+1;
                    if(k_tile[2:0]==7)begin load_row<=0;phase<=SOURCE_LOAD;end
                    else begin mask_row<=0;phase<=MASK_BUILD;end
                end
            end
            OUT_FETCH:begin
                out_data<=y_mem[row_id];dbg_y_reads<=dbg_y_reads+1;
                phase<=OUT_HOLD;
            end
            OUT_HOLD:if(out_ready)begin
                if(row_id==79)begin busy<=0;done<=1;phase<=IDLE;end
                else begin
                    row_id<=row_id+1;
                    out_data<=y_mem[row_id+1];dbg_y_reads<=dbg_y_reads+1;
                    // Same single read port, synchronous next row on acceptance.
                    // Keep OUT_HOLD so a ready sink accepts one row each cycle.
                end
            end
            default:phase<=IDLE;
            endcase
        end
    end
endmodule
