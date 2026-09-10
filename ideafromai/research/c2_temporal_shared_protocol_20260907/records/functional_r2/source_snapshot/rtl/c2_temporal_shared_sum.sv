// Research protocol reference. phi is a signed coefficient INCLUDING theta.
// Single source in flight; no bank concurrency or SRAM timing is implied.
module c2_temporal_shared_sum #(
    parameter integer T = 10,
    parameter integer SLOTS = 8,
    parameter integer LANES = 8,
    parameter integer VALUE_W = 48,
    parameter integer FRAME_W = 16,
    parameter integer INDEX_W = 16
) (
    input  wire clk,
    input  wire rst_n,
    input  wire in_valid,
    output wire in_ready,
    input  wire [T-1:0] in_signature,
    input  wire [LANES*VALUE_W-1:0] in_phi,
    input  wire in_last,
    input  wire [FRAME_W-1:0] in_frame,
    input  wire [INDEX_W-1:0] in_index,
    output wire [T-1:0] out_valid,
    input  wire [T-1:0] out_ready,
    output wire [LANES*VALUE_W-1:0] out_value,
    output wire [FRAME_W-1:0] out_frame,
    output wire done_valid,
    input  wire done_ready,
    output wire [FRAME_W-1:0] done_frame
);
    localparam integer AGE_W = (SLOTS > 1) ? $clog2(SLOTS) : 1;
    localparam [2:0] IDLE=0, PROCESS=1, SCATTER=2, AFTER=3,
                     DRAIN=4, DONE=5;
    localparam [1:0] BYPASS=0, REPLACE=1, FINAL_DRAIN=2;

    reg [2:0] state;
    reg [1:0] action;
    reg [LANES*VALUE_W-1:0] slot_value [0:SLOTS-1];
    reg [T-1:0] slot_tag [0:SLOTS-1];
    reg [AGE_W-1:0] age [0:SLOTS-1];
    reg [SLOTS-1:0] slot_valid;
    reg [LANES*VALUE_W-1:0] src_value, work_value;
    reg [T-1:0] src_tag, pending;
    reg src_last, frame_open;
    reg [FRAME_W-1:0] frame_id;
    reg [INDEX_W-1:0] next_index;
    reg [AGE_W-1:0] held_slot;

    integer hit, free_slot, oldest, oldest_age;
    integer i, lane;
    always @* begin
        hit = -1;
        free_slot = -1;
        oldest = -1;
        oldest_age = -1;
        for (integer k=0; k<SLOTS; k=k+1) begin
            if (slot_valid[k] && slot_tag[k] == src_tag) hit = k;
            if (!slot_valid[k] && free_slot < 0) free_slot = k;
            if (slot_valid[k] && int'(age[k]) > oldest_age) begin
                oldest = k;
                oldest_age = int'(age[k]);
            end
        end
    end

    wire [T-1:0] remaining = pending & ~out_ready;
    wire identity_ok = frame_open ?
        ((in_frame == frame_id) && (in_index == next_index)) :
        (in_index == {INDEX_W{1'b0}});
    assign in_ready = rst_n && state == IDLE && identity_ok;
    assign out_valid = (rst_n && state == SCATTER) ? pending : {T{1'b0}};
    assign out_value = work_value;
    assign out_frame = frame_id;
    assign done_valid = rst_n && state == DONE;
    assign done_frame = frame_id;

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            action <= BYPASS;
            slot_valid <= 0;
            src_value <= 0;
            work_value <= 0;
            src_tag <= 0;
            pending <= 0;
            src_last <= 0;
            frame_open <= 0;
            frame_id <= 0;
            next_index <= 0;
            held_slot <= 0;
            for (i=0; i<SLOTS; i=i+1) begin
                // Invalid values need not be reset in a physical implementation.
                slot_value[i] <= 0;
                slot_tag[i] <= 0;
                age[i] <= 0;
            end
        end else begin
            case (state)
                IDLE: if (in_valid && in_ready) begin
                    if (in_signature != 0) src_value <= in_phi;
                    src_tag <= in_signature;
                    src_last <= in_last;
                    frame_open <= 1;
                    frame_id <= in_frame;
                    next_index <= in_index + {{(INDEX_W-1){1'b0}},1'b1};
                    state <= PROCESS;
                end
                PROCESS: begin
                    if (src_tag == 0) begin
                        state <= AFTER;
                    end else if ((src_tag & (src_tag - {{(T-1){1'b0}},1'b1})) == 0) begin
                        work_value <= src_value;
                        pending <= src_tag;
                        action <= BYPASS;
                        state <= SCATTER;
                    end else if (hit >= 0) begin
                        for (lane=0; lane<LANES; lane=lane+1)
                            slot_value[hit][lane*VALUE_W +: VALUE_W] <=
                                $signed(slot_value[hit][lane*VALUE_W +: VALUE_W]) +
                                $signed(src_value[lane*VALUE_W +: VALUE_W]);
                        for (i=0; i<SLOTS; i=i+1)
                            if (slot_valid[i] && age[i] < age[hit])
                                age[i] <= age[i] + {{(AGE_W-1){1'b0}},1'b1};
                        age[hit] <= 0;
                        state <= AFTER;
                    end else if (free_slot >= 0) begin
                        slot_value[free_slot] <= src_value;
                        slot_tag[free_slot] <= src_tag;
                        slot_valid[free_slot] <= 1;
                        for (i=0; i<SLOTS; i=i+1)
                            if (slot_valid[i])
                                age[i] <= age[i] + {{(AGE_W-1){1'b0}},1'b1};
                        age[free_slot] <= 0;
                        state <= AFTER;
                    end else begin
                        held_slot <= AGE_W'(oldest);
                        work_value <= slot_value[oldest];
                        pending <= slot_tag[oldest];
                        action <= REPLACE;
                        state <= SCATTER;
                    end
                end
                SCATTER: begin
                    pending <= remaining;
                    if (remaining == 0) begin
                        if (action == REPLACE) begin
                            // Reuse only after every victim consumer accepted.
                            slot_value[held_slot] <= src_value;
                            slot_tag[held_slot] <= src_tag;
                            for (i=0; i<SLOTS; i=i+1)
                                if (slot_valid[i] && i != int'(held_slot))
                                    age[i] <= age[i] + {{(AGE_W-1){1'b0}},1'b1};
                            age[held_slot] <= 0;
                            state <= AFTER;
                        end else if (action == FINAL_DRAIN) begin
                            slot_valid[held_slot] <= 0;
                            state <= DRAIN;
                        end else begin
                            state <= AFTER;
                        end
                    end
                end
                AFTER: state <= src_last ? DRAIN : IDLE;
                DRAIN: begin
                    if (oldest < 0) begin
                        state <= DONE;
                    end else begin
                        held_slot <= AGE_W'(oldest);
                        work_value <= slot_value[oldest];
                        pending <= slot_tag[oldest];
                        action <= FINAL_DRAIN;
                        state <= SCATTER;
                    end
                end
                DONE: if (done_ready) begin
                    frame_open <= 0;
                    next_index <= 0;
                    state <= IDLE;
                end
                default: begin
                    state <= IDLE;
                    slot_valid <= 0;
                    pending <= 0;
                    frame_open <= 0;
                    next_index <= 0;
                end
            endcase
        end
    end
endmodule
