`timescale 1ns/1ps
`default_nettype none

module tb_qfit_fcsr_relation_memo_top;
    localparam int HEIGHT = 15;
    localparam int WIDTH = 15;
    localparam int TIME_PLANES = 2;
    localparam int TOKENS = HEIGHT * WIDTH;
    localparam int TOTAL = TOKENS * TIME_PLANES;
    localparam int MAX_HEADS = 24;
    localparam int HEAD_W = $clog2(MAX_HEADS);
    localparam int PTR_W = $clog2(513);

    logic clk_core;
    logic rst_core;
    logic window_start;
    logic head_start;
    logic head_ready;
    logic [HEAD_W-1:0] head_index;
    logic plane_start;
    logic plane_id;
    logic in_valid;
    logic in_ready;
    logic [3:0] in_y;
    logic [3:0] in_x;
    logic [4:0] in_candidate_valid;
    logic [31:0] in_k_self;
    logic [44:0] in_direction_gates;
    logic live_valid;
    logic live_ready;
    logic [8:0] live_source_id;
    logic [3:0] live_y;
    logic [3:0] live_x;
    logic [31:0] live_k;
    logic [44:0] live_gates;
    logic [4:0] live_valid_mask;
    logic live_last;
    logic head_done;
    logic head_resident;
    logic head_critical;
    logic head_overflow;
    logic [31:0] head_service_cycles;
    logic [PTR_W-1:0] head_record_count;
    logic replay_start;
    logic replay_cmd_ready;
    logic [HEAD_W-1:0] replay_head_index;
    logic replay_valid;
    logic replay_ready;
    logic [8:0] replay_source_id;
    logic [3:0] replay_y;
    logic [3:0] replay_x;
    logic [31:0] replay_k;
    logic [44:0] replay_gates;
    logic [4:0] replay_valid_mask;
    logic replay_last;
    logic replay_done;
    logic replay_miss;
    logic plane_idle;
    logic protocol_error;
    logic [31:0] perf_speculative_writes;
    logic [31:0] perf_discarded_writes;
    logic [31:0] perf_committed_records;
    logic [31:0] perf_replay_reads;
    logic [31:0] perf_capacity_misses;

    int monitored_head;
    int live_count;
    int active_capture_count;
    int active_sequence [0:511];
    bit live_seen [0:TOTAL-1];
    bit head_done_seen;

    qfit_fcsr_relation_memo_top dut (.*);

    always #5 clk_core = ~clk_core;

    function automatic logic [4:0] candidate_mask(
        input int p,
        input int y,
        input int x
    );
        logic [4:0] mask;
        mask = 5'b00001;
        if (y > 0)
            mask[1] = 1'b1;
        if (y < HEIGHT - 1)
            mask[2] = 1'b1;
        if (x > 0)
            mask[3] = 1'b1;
        if (x < WIDTH - 1)
            mask[4] = 1'b1;
        if (p == 1 && y == 7 && x == 7)
            mask[0] = 1'b0;
        if (p == 1 && y == 8 && x == 7)
            mask[1] = 1'b0;
        if (p == 1 && y == 7 && x == 8)
            mask[3] = 1'b0;
        return mask;
    endfunction

    function automatic logic [8:0] gate_value(
        input int head,
        input int p,
        input int y,
        input int x,
        input int role
    );
        if (head == 0)
            gate_value = 9'd7;
        else
            gate_value = 9'(17 + p * 5 + role);
    endfunction

    function automatic logic [31:0] k_value(
        input int head,
        input int p,
        input int y,
        input int x
    );
        int source_id;
        source_id = p * TOKENS + y * WIDTH + x;
        if (head == 0 && source_id >= 20)
            k_value = '0;
        else
            k_value = 32'h0000_0001;
    endfunction

    function automatic logic [4:0] expected_incoming_mask(
        input int p,
        input int y,
        input int x
    );
        logic [4:0] mask;
        logic [4:0] neighbor;
        mask = '0;
        neighbor = candidate_mask(p, y, x);
        mask[0] = neighbor[0];
        if (y < HEIGHT - 1) begin
            neighbor = candidate_mask(p, y + 1, x);
            mask[1] = neighbor[1];
        end
        if (y > 0) begin
            neighbor = candidate_mask(p, y - 1, x);
            mask[2] = neighbor[2];
        end
        if (x < WIDTH - 1) begin
            neighbor = candidate_mask(p, y, x + 1);
            mask[3] = neighbor[3];
        end
        if (x > 0) begin
            neighbor = candidate_mask(p, y, x - 1);
            mask[4] = neighbor[4];
        end
        return mask;
    endfunction

    task automatic check_descriptor(
        input int head,
        input int source_id,
        input logic [3:0] got_y,
        input logic [3:0] got_x,
        input logic [31:0] got_k,
        input logic [44:0] got_gates,
        input logic [4:0] got_mask
    );
        int p;
        int y;
        int x;
        logic [8:0] expected_gate;
        p = source_id / TOKENS;
        y = (source_id % TOKENS) / WIDTH;
        x = (source_id % TOKENS) % WIDTH;
        if (got_y != 4'(y) || got_x != 4'(x))
            $fatal(1, "descriptor coordinate mismatch source=%0d", source_id);
        if (got_k != k_value(head, p, y, x))
            $fatal(1, "descriptor K mismatch source=%0d", source_id);
        if (got_mask != expected_incoming_mask(p, y, x))
            $fatal(1, "descriptor mask mismatch source=%0d", source_id);
        for (int role = 0; role < 5; role = role + 1) begin
            expected_gate = '0;
            case (role)
                0: expected_gate = gate_value(head, p, y, x, 0);
                1: if (y < HEIGHT - 1)
                    expected_gate = gate_value(head, p, y + 1, x, 1);
                2: if (y > 0)
                    expected_gate = gate_value(head, p, y - 1, x, 2);
                3: if (x < WIDTH - 1)
                    expected_gate = gate_value(head, p, y, x + 1, 3);
                4: if (x > 0)
                    expected_gate = gate_value(head, p, y, x - 1, 4);
            endcase
            if (
                got_mask[role]
                && got_gates[role*9 +: 9] != expected_gate
            )
                $fatal(
                    1,
                    "descriptor gate mismatch source=%0d role=%0d",
                    source_id,
                    role
                );
        end
    endtask

    task automatic drive_plane(input int head, input int p);
        int accepted;
        bit fire;
        while (!plane_idle)
            @(negedge clk_core);
        plane_id = p[0];
        plane_start = 1'b1;
        @(negedge clk_core);
        plane_start = 1'b0;
        accepted = 0;
        in_valid = 1'b1;
        while (accepted < TOKENS) begin
            in_y = 4'(accepted / WIDTH);
            in_x = 4'(accepted % WIDTH);
            in_candidate_valid = candidate_mask(
                p,
                accepted / WIDTH,
                accepted % WIDTH
            );
            in_k_self = k_value(
                head,
                p,
                accepted / WIDTH,
                accepted % WIDTH
            );
            for (int role = 0; role < 5; role = role + 1)
                in_direction_gates[role*9 +: 9] = gate_value(
                    head,
                    p,
                    accepted / WIDTH,
                    accepted % WIDTH,
                    role
                );
            @(posedge clk_core);
            fire = in_ready;
            @(negedge clk_core);
            if (fire)
                accepted = accepted + 1;
        end
        in_valid = 1'b0;
        while (!plane_idle)
            @(negedge clk_core);
    endtask

    task automatic run_head(
        input int head,
        input bit expected_resident,
        input bit expected_critical,
        input int expected_records
    );
        while (!head_ready)
            @(negedge clk_core);
        monitored_head = head;
        live_count = 0;
        active_capture_count = 0;
        head_done_seen = 1'b0;
        for (int source = 0; source < TOTAL; source = source + 1)
            live_seen[source] = 1'b0;
        head_index = HEAD_W'(head);
        head_start = 1'b1;
        @(negedge clk_core);
        head_start = 1'b0;
        for (int p = 0; p < TIME_PLANES; p = p + 1)
            drive_plane(head, p);
        while (!head_done_seen)
            @(negedge clk_core);
        if (live_count != TOTAL)
            $fatal(1, "head %0d live count mismatch", head);
        for (int source = 0; source < TOTAL; source = source + 1)
            if (!live_seen[source])
                $fatal(1, "head %0d missing live source=%0d", head, source);
        if (head_resident != expected_resident)
            $fatal(1, "head %0d resident mismatch", head);
        if (head_critical != expected_critical)
            $fatal(1, "head %0d critical mismatch", head);
        if (head_overflow)
            $fatal(1, "head %0d unexpected overflow", head);
        if (head_record_count != PTR_W'(expected_records))
            $fatal(1, "head %0d record count mismatch", head);
    endtask

    task automatic replay_head(
        input int head,
        input int expected_records,
        input bit expected_miss
    );
        int received;
        bit saw_miss;
        while (!replay_cmd_ready)
            @(negedge clk_core);
        replay_head_index = HEAD_W'(head);
        replay_start = 1'b1;
        @(negedge clk_core);
        replay_start = 1'b0;
        received = 0;
        saw_miss = replay_miss;
        while (!replay_done) begin
            replay_ready = $urandom_range(0, 3) != 0;
            @(posedge clk_core);
            if (replay_miss)
                saw_miss = 1'b1;
            if (replay_valid && replay_ready) begin
                if (replay_source_id != 9'(active_sequence[received]))
                    $fatal(
                        1,
                        "replay order mismatch got=%0d exp=%0d",
                        replay_source_id,
                        active_sequence[received]
                    );
                check_descriptor(
                    head,
                    replay_source_id,
                    replay_y,
                    replay_x,
                    replay_k,
                    replay_gates,
                    replay_valid_mask
                );
                if (replay_last != (received == expected_records - 1))
                    $fatal(1, "replay last mismatch");
                received = received + 1;
            end
            @(negedge clk_core);
        end
        replay_ready = 1'b0;
        if (saw_miss != expected_miss)
            $fatal(1, "replay miss mismatch");
        if (received != expected_records)
            $fatal(1, "replay record count mismatch");
    endtask

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            live_ready <= 1'b0;
            head_done_seen <= 1'b0;
        end else begin
            live_ready <= $urandom_range(0, 4) != 0;
            if (head_done)
                head_done_seen <= 1'b1;
            if (live_valid && live_ready) begin
                if (live_seen[live_source_id])
                    $fatal(1, "duplicate live source=%0d", live_source_id);
                check_descriptor(
                    monitored_head,
                    live_source_id,
                    live_y,
                    live_x,
                    live_k,
                    live_gates,
                    live_valid_mask
                );
                if (live_last != (live_source_id == TOTAL - 1))
                    $fatal(1, "live last mismatch");
                live_seen[live_source_id] <= 1'b1;
                if (live_k != 0) begin
                    active_sequence[active_capture_count] <= live_source_id;
                    active_capture_count <= active_capture_count + 1;
                end
                live_count <= live_count + 1;
            end
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        window_start = 1'b0;
        head_start = 1'b0;
        head_index = '0;
        plane_start = 1'b0;
        plane_id = 1'b0;
        in_valid = 1'b0;
        in_y = '0;
        in_x = '0;
        in_candidate_valid = '0;
        in_k_self = '0;
        in_direction_gates = '0;
        replay_start = 1'b0;
        replay_head_index = '0;
        replay_ready = 1'b0;
        monitored_head = 0;
        live_count = 0;
        active_capture_count = 0;
        head_done_seen = 1'b0;
        for (int source = 0; source < TOTAL; source = source + 1)
            live_seen[source] = 1'b0;
        repeat (5) @(negedge clk_core);
        rst_core = 1'b0;
        window_start = 1'b1;
        @(negedge clk_core);
        window_start = 1'b0;

        run_head(0, 1'b1, 1'b1, 20);
        replay_head(0, 20, 1'b0);
        run_head(1, 1'b0, 1'b0, 0);
        replay_head(1, 0, 1'b1);

        if (protocol_error)
            $fatal(1, "unexpected protocol error");
        if (perf_speculative_writes != 470)
            $fatal(1, "speculative write count mismatch");
        if (perf_discarded_writes != 450)
            $fatal(1, "discard count mismatch");
        if (perf_committed_records != 20)
            $fatal(1, "commit count mismatch");
        if (perf_replay_reads != 20)
            $fatal(1, "replay read count mismatch");
        if (perf_capacity_misses != 0)
            $fatal(1, "unexpected capacity miss");
        $display(
            "PASS FCSR relation memo T450 live=900 committed=%0d replay=%0d",
            perf_committed_records,
            perf_replay_reads
        );
        $finish;
    end
endmodule

`default_nettype wire
