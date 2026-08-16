`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_dualtag_replay_lifecycle_manager;
    logic clk_core, rst_core;
    logic session_valid, session_ready, session_context_id;
    logic [4:0] session_head_id;
    logic [31:0] session_payload_tag, session_execution_tag;
    logic session_cache_owned, session_last_output_tile;
    logic decoder_done_valid, decoder_done_ready;
    logic [31:0] decoder_done_payload_tag;
    logic decoder_done_error;
    logic backend_done_valid, backend_done_ready;
    logic [31:0] backend_done_execution_tag;
    logic backend_done_error;
    logic slot_release_valid, slot_release_ready, slot_release_context_id;
    logic [4:0] slot_release_head_id;
    logic cache_release_valid, cache_release_ready, cache_release_context_id;
    logic [4:0] cache_release_head_id;
    logic [31:0] cache_release_payload_tag;
    logic session_done_valid, session_done_ready;
    logic [31:0] session_done_payload_tag, session_done_execution_tag;
    logic session_done_error, protocol_error;
    logic [31:0] count_sessions, count_final_tile_releases;
    logic [31:0] count_cache_releases, count_session_errors;
    int slot_releases, cache_releases;

    gatestack_dualtag_replay_lifecycle_manager dut (.*);
    always #5 clk_core <= ~clk_core;

    always @(posedge clk_core) begin
        if (rst_core) begin
            slot_releases <= 0;
            cache_releases <= 0;
        end else begin
            if (slot_release_valid && slot_release_ready) begin
                if (slot_release_context_id != session_context_id ||
                    slot_release_head_id != session_head_id)
                    $fatal(1, "slot release identity mismatch");
                slot_releases <= slot_releases + 1;
            end
            if (cache_release_valid && cache_release_ready) begin
                if (cache_release_context_id != session_context_id ||
                    cache_release_head_id != session_head_id ||
                    cache_release_payload_tag != session_payload_tag)
                    $fatal(1, "cache release identity mismatch");
                cache_releases <= cache_releases + 1;
            end
        end
    end

    task automatic begin_session(input int id,
                                 input logic cache_owned,
                                 input logic last_tile);
        begin
            @(negedge clk_core);
            session_context_id = id[0];
            session_head_id = 5'(id);
            session_payload_tag = 32'hf600_1000 + 32'(id);
            session_execution_tag = 32'hf6a0_2000 + 32'(id);
            session_cache_owned = cache_owned;
            session_last_output_tile = last_tile;
            session_valid = 1'b1;
            do @(posedge clk_core); while (!session_ready);
            @(negedge clk_core);
            session_valid = 1'b0;
        end
    endtask

    task automatic send_decoder(input int id, input logic bad_tag);
        begin
            @(negedge clk_core);
            decoder_done_payload_tag =
                32'hf600_1000 + 32'(id) + 32'(bad_tag);
            decoder_done_error = 1'b0;
            decoder_done_valid = 1'b1;
            do @(posedge clk_core); while (!decoder_done_ready);
            @(negedge clk_core);
            decoder_done_valid = 1'b0;
        end
    endtask

    task automatic send_backend(input int id, input logic bad_tag);
        begin
            @(negedge clk_core);
            backend_done_execution_tag =
                32'hf6a0_2000 + 32'(id) + 32'(bad_tag);
            backend_done_error = 1'b0;
            backend_done_valid = 1'b1;
            do @(posedge clk_core); while (!backend_done_ready);
            @(negedge clk_core);
            backend_done_valid = 1'b0;
        end
    endtask

    task automatic accept_done(input int id, input logic expected_error);
        begin
            while (!session_done_valid) @(posedge clk_core);
            if (session_done_payload_tag !=
                    32'hf600_1000 + 32'(id) ||
                session_done_execution_tag !=
                    32'hf6a0_2000 + 32'(id) ||
                session_done_error != expected_error)
                $fatal(1, "dualtag lifecycle completion mismatch");
            repeat (2) @(posedge clk_core);
            @(negedge clk_core);
            session_done_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            session_done_ready = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        session_valid = 1'b0;
        session_context_id = 1'b0;
        session_head_id = '0;
        session_payload_tag = '0;
        session_execution_tag = '0;
        session_cache_owned = 1'b0;
        session_last_output_tile = 1'b0;
        decoder_done_valid = 1'b0;
        decoder_done_payload_tag = '0;
        decoder_done_error = 1'b0;
        backend_done_valid = 1'b0;
        backend_done_execution_tag = '0;
        backend_done_error = 1'b0;
        slot_release_ready = 1'b0;
        cache_release_ready = 1'b0;
        session_done_ready = 1'b0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        begin_session(0, 1'b1, 1'b0);
        send_backend(0, 1'b0);
        repeat (2) @(posedge clk_core);
        send_decoder(0, 1'b0);
        accept_done(0, 1'b0);
        if (slot_releases != 0 || cache_releases != 0)
            $fatal(1, "nonfinal dualtag session released resources");

        begin_session(1, 1'b1, 1'b1);
        send_decoder(1, 1'b0);
        repeat (2) @(posedge clk_core);
        send_backend(1, 1'b0);
        while (!slot_release_valid || !cache_release_valid)
            @(posedge clk_core);
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
        cache_release_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        cache_release_ready = 1'b0;
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
        slot_release_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        slot_release_ready = 1'b0;
        accept_done(1, 1'b0);

        begin_session(2, 1'b0, 1'b0);
        send_decoder(2, 1'b1);
        send_backend(2, 1'b0);
        accept_done(2, 1'b1);

        if (!protocol_error || count_sessions != 3 ||
            count_final_tile_releases != 1 || count_cache_releases != 1 ||
            count_session_errors != 1 || slot_releases != 1 ||
            cache_releases != 1)
            $fatal(1, "dualtag lifecycle counters mismatch");
        $display("PASS: dualtag lifecycle sessions=%0d releases=%0d errors=%0d",
                 count_sessions, count_final_tile_releases,
                 count_session_errors);
        $finish;
    end

    initial begin
        repeat (10000) @(posedge clk_core);
        $fatal(1, "dualtag lifecycle timeout");
    end
endmodule

`default_nettype wire
