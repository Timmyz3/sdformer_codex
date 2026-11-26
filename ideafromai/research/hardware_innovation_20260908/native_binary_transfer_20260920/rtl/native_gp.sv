// Gustav-inspired native binary P8 slice; not the full published architecture.
// One synchronous source-word read and one synchronous coefficient-word read
// per cycle. Two NR4 packets overlap coefficient fetch with serial position
// merge execution. Eight 24-bit ALUs update one P8 position per cycle.
module native_gp (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         cfg_valid,
    input  logic         cfg_kind,
    input  logic [15:0]  cfg_addr,
    input  logic [127:0] cfg_data,
    output logic         cfg_ready,
    input  logic         start,
    input  logic [8:0]   c_count,
    input  logic [1:0]   mode,
    input  logic         src_enable,
    input  logic         coef_enable,
    output logic         busy,
    output logic         done,
    output logic         out_valid,
    input  logic         out_ready,
    output logic [6:0]   out_row,
    output logic [191:0] out_data,
    output logic [31:0]  dbg_source_reads,
    output logic [31:0]  dbg_coef_reads,
    output logic [31:0]  dbg_add_cycles,
    output logic [31:0]  dbg_build_cycles
);
    localparam [2:0] IDLE=0, CLEAR_ACC=1, LOAD_SRC=2, MAKE_BITS=3,
                     RUN=4, COMMIT=5, DRAIN=6;
    logic [2:0] state;
    logic [127:0] source_mem [0:239];
    logic [63:0] coefficient_mem [0:383];
    logic [191:0] y_mem [0:79];
    logic [127:0] source_block [0:7];
    logic [191:0] acc [0:7];
    logic [8:0] channels;
    logic [3:0] tick_group;
    logic [1:0] word_index;
    logic [3:0] source_read_index;
    logic [2:0] local_row;
    logic [6:0] drain_next;
    logic [127:0] remaining_channels;

    // A packet has at most four nonzero channel records. One vector of eight
    // W8 values is fetched once per record, then retained for all P8 positions.
    logic [1:0] packet_valid;
    logic [2:0] packet_len [0:1];
    logic [2:0] fetch_count [0:1];
    logic [3:0] weight_ready [0:1];
    logic [8:0] record_channel [0:1][0:3];
    logic [7:0] record_mask [0:1][0:3];
    logic [63:0] record_weight [0:1][0:3];
    logic fill_packet, fetch_packet, execute_packet;

    logic [127:0] source_union;
    logic [127:0] after_build;
    logic [2:0] build_len;
    logic [8:0] build_channel [0:3];
    logic [7:0] build_mask [0:3];
    logic [3:0] merge_position;
    logic [1:0] merge_record;
    logic execute_has_work;
    logic work_after_issue;
    integer i, j, k;
    integer pick;
    integer pos;

    assign cfg_ready = (state == IDLE) && !start;

    // Transposition is real: source_block only becomes valid after eight
    // synchronous source reads. The NRV priority logic consumes this buffer.
    always_comb begin
        source_union = 128'd0;
        for (integer p=0; p<8; p=p+1)
            source_union = source_union | source_block[p];
        for (integer c=0; c<128; c=c+1)
            if ((integer'(word_index)*128+c) >= integer'(channels))
                source_union[c] = 1'b0;
    end

    // Four leading-one selectors produce one NR4 packet per build cycle.
    always_comb begin
        after_build = remaining_channels;
        build_len = 3'd0;
        pick = -1;
        for (integer r=0; r<4; r=r+1) begin
            build_channel[r] = 9'd0;
            build_mask[r] = 8'd0;
            pick = -1;
            for (integer c=0; c<128; c=c+1)
                if (after_build[c] && pick < 0) pick = c;
            if (pick >= 0) begin
                build_channel[r] = {word_index,7'd0} + 9'(pick);
                for (integer p=0; p<8; p=p+1)
                    build_mask[r][p] = source_block[p][pick];
                after_build[pick] = 1'b0;
                build_len = build_len + 3'd1;
            end
        end
    end

    // Minimum position merger. Ties issue one record at a time: combining
    // multiple W vectors in one cycle would exceed the common eight-ALU budget.
    always_comb begin
        merge_position = 4'd8;
        merge_record = 2'd0;
        execute_has_work = 1'b0;
        pos = 8;
        for (integer r=0; r<4; r=r+1) begin
            pos = 8;
            for (integer p=0; p<8; p=p+1)
                if (record_mask[execute_packet][r][p] && pos == 8) pos = p;
            if (r < integer'(packet_len[execute_packet]) && pos < integer'(merge_position)) begin
                merge_position = 4'(pos);
                merge_record = 2'(r);
                execute_has_work = 1'b1;
            end
        end
        work_after_issue = 1'b0;
        for (integer r=0; r<4; r=r+1)
            if (r < integer'(packet_len[execute_packet])) begin
                if (r == integer'(merge_record))
                    work_after_issue = work_after_issue |
                        (|(record_mask[execute_packet][r] & ~(8'd1 << merge_position)));
                else work_after_issue = work_after_issue | (|record_mask[execute_packet][r]);
            end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            out_valid <= 1'b0;
            out_row <= 7'd0;
            out_data <= 192'd0;
            channels <= 9'd0;
            tick_group <= 4'd0;
            word_index <= 2'd0;
            source_read_index <= 4'd0;
            local_row <= 3'd0;
            drain_next <= 7'd0;
            remaining_channels <= 128'd0;
            packet_valid <= 2'd0;
            fill_packet <= 1'b0;
            fetch_packet <= 1'b0;
            execute_packet <= 1'b0;
            dbg_source_reads <= 0;
            dbg_coef_reads <= 0;
            dbg_add_cycles <= 0;
            dbg_build_cycles <= 0;
            for (i=0; i<2; i=i+1) begin
                packet_len[i] <= 0;
                fetch_count[i] <= 0;
                weight_ready[i] <= 0;
                for (j=0; j<4; j=j+1) begin
                    record_channel[i][j] <= 0;
                    record_mask[i][j] <= 0;
                    record_weight[i][j] <= 0;
                end
            end
        end else begin
            done <= 1'b0;
            if (cfg_valid && cfg_ready) begin
                if (cfg_kind) coefficient_mem[cfg_addr[8:0]] <= cfg_data[63:0];
                else source_mem[cfg_addr[7:0]] <= cfg_data;
            end
            case (state)
                IDLE: if (start) begin
                    channels <= c_count;
                    tick_group <= 0;
                    local_row <= 0;
                    word_index <= 0;
                    state <= CLEAR_ACC;
                    busy <= 1'b1;
                    out_valid <= 1'b0;
                    dbg_source_reads <= 0;
                    dbg_coef_reads <= 0;
                    dbg_add_cycles <= 0;
                    dbg_build_cycles <= 0;
                end
                CLEAR_ACC: begin
                    // One P8 row is cleared per cycle; initialization is timed.
                    acc[local_row] <= 0;
                    if (local_row == 7) begin
                        source_read_index <= 0;
                        word_index <= 0;
                        state <= LOAD_SRC;
                    end else local_row <= local_row + 1'b1;
                end
                LOAD_SRC: if (src_enable) begin
                    source_block[source_read_index[2:0]] <=
                        source_mem[(integer'(tick_group)*8+integer'(source_read_index))*3+integer'(word_index)];
                    dbg_source_reads <= dbg_source_reads + 1'b1;
                    if (source_read_index == 7) state <= MAKE_BITS;
                    else source_read_index <= source_read_index + 1'b1;
                end
                MAKE_BITS: begin
                    remaining_channels <= source_union;
                    packet_valid <= 0;
                    fill_packet <= 0;
                    fetch_packet <= 0;
                    execute_packet <= 0;
                    state <= RUN;
                end
                RUN: begin
                    if (remaining_channels != 0 && !packet_valid[fill_packet]) begin
                        packet_valid[fill_packet] <= 1'b1;
                        packet_len[fill_packet] <= build_len;
                        fetch_count[fill_packet] <= 0;
                        weight_ready[fill_packet] <= 0;
                        for (k=0; k<4; k=k+1) begin
                            record_channel[fill_packet][k] <= build_channel[k];
                            record_mask[fill_packet][k] <= build_mask[k];
                        end
                        remaining_channels <= after_build;
                        fill_packet <= ~fill_packet;
                        dbg_build_cycles <= dbg_build_cycles + 1'b1;
                    end
                    // Registered RAM return: execution sees weight_ready only
                    // on the following cycle; no combinational coefficient port.
                    if (coef_enable && packet_valid[fetch_packet] &&
                        fetch_count[fetch_packet] < packet_len[fetch_packet]) begin
                        record_weight[fetch_packet][fetch_count[fetch_packet][1:0]] <=
                            coefficient_mem[record_channel[fetch_packet][fetch_count[fetch_packet][1:0]]];
                        weight_ready[fetch_packet][fetch_count[fetch_packet][1:0]] <= 1'b1;
                        fetch_count[fetch_packet] <= fetch_count[fetch_packet] + 1'b1;
                        dbg_coef_reads <= dbg_coef_reads + 1'b1;
                        if (fetch_count[fetch_packet] + 1'b1 == packet_len[fetch_packet])
                            fetch_packet <= ~fetch_packet;
                    end
                    if (packet_valid[execute_packet]) begin
                        if (execute_has_work) begin
                            if (weight_ready[execute_packet][merge_record[1:0]]) begin
                                for (k=0; k<8; k=k+1)
                                    acc[merge_position[2:0]][k*24 +: 24] <=
                                        $signed(acc[merge_position[2:0]][k*24 +: 24]) +
                                        $signed({{16{record_weight[execute_packet][merge_record[1:0]][k*8+7]}},
                                                 record_weight[execute_packet][merge_record[1:0]][k*8 +: 8]});
                                record_mask[execute_packet][merge_record[1:0]][merge_position[2:0]] <= 1'b0;
                                dbg_add_cycles <= dbg_add_cycles + 1'b1;
                                if (!work_after_issue) begin
                                    packet_valid[execute_packet] <= 1'b0;
                                    execute_packet <= ~execute_packet;
                                end
                            end
                        end else begin
                            packet_valid[execute_packet] <= 1'b0;
                            execute_packet <= ~execute_packet;
                        end
                    end
                    if (remaining_channels == 0 && packet_valid == 0) begin
                        if ((integer'(word_index)+1)*128 < integer'(channels)) begin
                            word_index <= word_index + 1'b1;
                            source_read_index <= 0;
                            state <= LOAD_SRC;
                        end else begin
                            local_row <= 0;
                            state <= COMMIT;
                        end
                    end
                end
                COMMIT: begin
                    // One 192-bit Y write per cycle. No free bulk commit.
                    y_mem[integer'(tick_group)*8+integer'(local_row)] <= acc[local_row];
                    if (local_row == 7) begin
                        if (tick_group == 9) begin
                            drain_next <= 0;
                            out_valid <= 0;
                            state <= DRAIN;
                        end else begin
                            tick_group <= tick_group + 1'b1;
                            local_row <= 0;
                            state <= CLEAR_ACC;
                        end
                    end else local_row <= local_row + 1'b1;
                end
                DRAIN: if (!out_valid || out_ready) begin
                    if (drain_next < 80) begin
                        out_data <= y_mem[drain_next];
                        out_row <= drain_next;
                        out_valid <= 1'b1;
                        drain_next <= drain_next + 1'b1;
                    end else begin
                        out_valid <= 1'b0;
                        busy <= 1'b0;
                        done <= 1'b1;
                        state <= IDLE;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
    // mode is intentionally unused: GP is one specific execution organization.
endmodule
