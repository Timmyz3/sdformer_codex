module tb_m364_banked_q128_independent_hammer;
    localparam integer TOTAL = 3000;
    localparam integer PATTERNS = 128;

    typedef struct packed {
        logic [15:0] original;
        logic [15:0] center;
        logic [6:0]  center_id;
        logic [4:0]  distance;
        logic [4:0]  population;
        logic        use_pwp;
        logic        fallback;
        logic [15:0] plus_mask;
        logic [15:0] minus_mask;
    } match_result_t;

    logic core_clk = 1'b0;

    logic reset363_n = 1'b0;
    logic m363_cfg_valid;
    logic m363_cfg_ready;
    logic [2:0] m363_cfg_group;
    logic [255:0] m363_cfg_patterns_flat;
    logic m363_cfg_commit;
    logic m363_cfg_active;
    logic m363_cfg_protocol_error;
    logic m363_in_valid;
    logic m363_in_ready;
    logic [15:0] m363_in_original_pattern;
    logic m363_out_valid;
    logic m363_out_ready;
    logic [15:0] m363_out_original_pattern;
    logic [15:0] m363_out_best_center;
    logic [6:0] m363_out_best_center_id;
    logic [4:0] m363_out_best_distance;
    logic [4:0] m363_out_population;
    logic m363_out_use_pwp;
    logic m363_out_fallback_bit_sparse;
    logic [15:0] m363_out_plus_mask;
    logic [15:0] m363_out_minus_mask;
    logic [2047:0] m363_catalog_flat;

    logic reset356_n = 1'b0;
    logic m356_cfg_valid;
    logic m356_cfg_ready;
    logic [2:0] m356_cfg_group;
    logic [255:0] m356_cfg_patterns_flat;
    logic m356_cfg_commit;
    logic m356_cfg_active;
    logic m356_cfg_protocol_error;
    logic m356_in_valid;
    logic m356_in_ready;
    logic [15:0] m356_in_original_pattern;
    logic m356_out_valid;
    logic m356_out_ready;
    logic [15:0] m356_out_original_pattern;
    logic [15:0] m356_out_best_center;
    logic [6:0] m356_out_best_center_id;
    logic [4:0] m356_out_best_distance;
    logic [4:0] m356_out_population;
    logic m356_out_use_pwp;
    logic m356_out_fallback_bit_sparse;
    logic [15:0] m356_out_plus_mask;
    logic [15:0] m356_out_minus_mask;

    logic [15:0] patterns [0:PATTERNS-1];
    logic [15:0] catalog_snapshot [0:PATTERNS-1];
    logic [15:0] stimulus [0:TOTAL-1];
    match_result_t expected [0:TOTAL-1];
    match_result_t observed363 [0:TOTAL-1];
    match_result_t observed356 [0:TOTAL-1];
    match_result_t attack_expected [0:3];
    match_result_t actual363;
    match_result_t actual356;
    integer accept_cycle363 [0:TOTAL-1];
    integer accept_cycle356 [0:TOTAL-1];

    integer main_tick;
    integer generated363;
    integer accepted363;
    integer retired363;
    integer generated356;
    integer accepted356;
    integer retired356;
    integer cycle363;
    integer cycle356;
    integer mismatch363;
    integer mismatch356;
    integer pairwise_mismatch;
    integer max_accept_run363;
    integer max_retire_run363;
    integer accept_run363;
    integer retire_run363;
    integer max_accept_run356;
    integer max_retire_run356;
    integer accept_run356;
    integer retire_run356;
    integer latency_min363;
    integer latency_max363;
    integer latency_min356;
    integer latency_max356;
    integer stalls363;
    integer stalls356;
    integer bubbles363;
    integer bubbles356;
    integer use363;
    integer fallback363;
    integer mixed363;
    integer exact363;
    integer cfg_block_cycles;
    integer sticky_attack_cycles;
    integer sticky_external_cfg_handshakes;
    integer sticky_input_handshakes;
    integer reset_mid_pipeline_count;
    integer reset_flushed_tokens;
    integer deferred_cfg_handshakes;
    logic main_run;

    always #1.5 core_clk = ~core_clk;

    function automatic integer popcount16(input logic [15:0] value);
        integer bit_index;
        begin
            popcount16 = 0;
            for (bit_index = 0; bit_index < 16; bit_index = bit_index + 1)
                popcount16 = popcount16 + value[bit_index];
        end
    endfunction

    task automatic calculate_reference(
        input logic [15:0] original,
        output match_result_t result
    );
        integer center_index;
        integer candidate_distance;
        integer best_distance;
        integer best_index;
        integer population;
        logic [15:0] best_center;
        begin
            best_index = 0;
            best_center = patterns[0];
            best_distance = popcount16(original ^ patterns[0]);
            for (center_index = 1; center_index < PATTERNS;
                    center_index = center_index + 1) begin
                candidate_distance = popcount16(
                    original ^ patterns[center_index]);
                if (candidate_distance < best_distance) begin
                    best_distance = candidate_distance;
                    best_index = center_index;
                    best_center = patterns[center_index];
                end
            end
            population = popcount16(original);
            result.original = original;
            result.center = best_center;
            result.center_id = best_index[6:0];
            result.distance = best_distance[4:0];
            result.population = population[4:0];
            result.use_pwp = (1 + best_distance) < population;
            result.fallback = !result.use_pwp;
            result.plus_mask = result.use_pwp ?
                (original & ~best_center) : original;
            result.minus_mask = result.use_pwp ?
                (best_center & ~original) : 16'h0000;
        end
    endtask

    task automatic reset_m363;
        integer reset_index;
        begin
            @(negedge core_clk);
            reset363_n = 1'b0;
            m363_cfg_valid = 1'b0;
            m363_cfg_commit = 1'b0;
            m363_in_valid = 1'b0;
            m363_out_ready = 1'b0;
            repeat (3) @(posedge core_clk);
            @(negedge core_clk);
            if (m363_cfg_active || m363_cfg_protocol_error ||
                    m363_out_valid || u_m363.stage0_valid_q ||
                    u_m363.stage1_valid_q || u_m363.stage2_valid_q ||
                    u_m363.cfg_next_group_q != 0)
                $fatal(1, "M364 M363 reset did not clear control state");
            for (reset_index = 0; reset_index < PATTERNS;
                    reset_index = reset_index + 1)
                if (u_m363.pattern_q[reset_index] !== 16'h0000)
                    $fatal(1, "M364 M363 reset did not clear catalog");
            reset363_n = 1'b1;
        end
    endtask

    task automatic reset_m356;
        begin
            @(negedge core_clk);
            reset356_n = 1'b0;
            m356_cfg_valid = 1'b0;
            m356_cfg_commit = 1'b0;
            m356_in_valid = 1'b0;
            m356_out_ready = 1'b0;
            repeat (3) @(posedge core_clk);
            @(negedge core_clk);
            if (m356_cfg_active || m356_cfg_protocol_error || m356_out_valid)
                $fatal(1, "M364 M356 reset did not clear state");
            reset356_n = 1'b1;
        end
    endtask

    task automatic load_m363_group(
        input integer group_index,
        input logic commit_value
    );
        integer lane;
        begin
            @(negedge core_clk);
            m363_cfg_patterns_flat = '0;
            for (lane = 0; lane < 16; lane = lane + 1)
                m363_cfg_patterns_flat[lane * 16 +: 16] =
                    patterns[group_index * 16 + lane];
            m363_cfg_group = group_index[2:0];
            m363_cfg_commit = commit_value;
            m363_cfg_valid = 1'b1;
            while (!m363_cfg_ready)
                @(negedge core_clk);
            @(posedge core_clk);
            @(negedge core_clk);
            m363_cfg_valid = 1'b0;
            m363_cfg_commit = 1'b0;
        end
    endtask

    task automatic configure_m363;
        integer group_index;
        begin
            for (group_index = 0; group_index < 8;
                    group_index = group_index + 1)
                load_m363_group(group_index, group_index == 7);
            @(posedge core_clk);
            @(negedge core_clk);
            if (!m363_cfg_active || m363_cfg_protocol_error ||
                    u_m363.cfg_next_group_q != 0)
                $fatal(1, "M364 M363 configuration failed");
        end
    endtask

    task automatic load_m356_group(
        input integer group_index,
        input logic commit_value
    );
        integer lane;
        begin
            @(negedge core_clk);
            m356_cfg_patterns_flat = '0;
            for (lane = 0; lane < 16; lane = lane + 1)
                m356_cfg_patterns_flat[lane * 16 +: 16] =
                    patterns[group_index * 16 + lane];
            m356_cfg_group = group_index[2:0];
            m356_cfg_commit = commit_value;
            m356_cfg_valid = 1'b1;
            while (!m356_cfg_ready)
                @(negedge core_clk);
            @(posedge core_clk);
            @(negedge core_clk);
            m356_cfg_valid = 1'b0;
            m356_cfg_commit = 1'b0;
        end
    endtask

    task automatic configure_m356;
        integer group_index;
        begin
            for (group_index = 0; group_index < 8;
                    group_index = group_index + 1)
                load_m356_group(group_index, group_index == 7);
            @(posedge core_clk);
            @(negedge core_clk);
            if (!m356_cfg_active || m356_cfg_protocol_error)
                $fatal(1, "M364 M356 configuration failed");
        end
    endtask

    task automatic fill_m363_four(input integer seed_offset);
        integer token_index;
        begin
            m363_out_ready = 1'b0;
            for (token_index = 0; token_index < 4;
                    token_index = token_index + 1) begin
                attack_expected[token_index] = expected[seed_offset + token_index];
                @(negedge core_clk);
                m363_in_original_pattern = stimulus[seed_offset + token_index];
                m363_in_valid = 1'b1;
                if (!m363_in_ready)
                    $fatal(1, "M364 could not inject four consecutive tokens");
                @(posedge core_clk);
            end
            @(negedge core_clk);
            m363_in_valid = 1'b0;
            if (!(u_m363.stage0_valid_q && u_m363.stage1_valid_q &&
                    u_m363.stage2_valid_q && m363_out_valid))
                $fatal(1, "M364 four elastic slots were not simultaneously full");
        end
    endtask

    task automatic compare_attack_output(input integer result_index);
        begin
            actual363.original = m363_out_original_pattern;
            actual363.center = m363_out_best_center;
            actual363.center_id = m363_out_best_center_id;
            actual363.distance = m363_out_best_distance;
            actual363.population = m363_out_population;
            actual363.use_pwp = m363_out_use_pwp;
            actual363.fallback = m363_out_fallback_bit_sparse;
            actual363.plus_mask = m363_out_plus_mask;
            actual363.minus_mask = m363_out_minus_mask;
            if (actual363 !== attack_expected[result_index])
                $fatal(1, "M364 directed attack output mismatch index=%0d",
                       result_index);
        end
    endtask

    task automatic attack_deferred_configuration;
        integer hold_cycle;
        integer direct_retired;
        integer lane;
        begin
            fill_m363_four(20);
            for (lane = 0; lane < PATTERNS; lane = lane + 1)
                catalog_snapshot[lane] = u_m363.pattern_q[lane];

            m363_cfg_patterns_flat = '0;
            for (lane = 0; lane < 16; lane = lane + 1)
                m363_cfg_patterns_flat[lane * 16 +: 16] = patterns[lane];
            m363_cfg_group = 0;
            m363_cfg_commit = 1'b0;
            m363_cfg_valid = 1'b1;
            m363_in_valid = 1'b1;
            m363_in_original_pattern = 16'hdeaf;

            for (hold_cycle = 0; hold_cycle < 24;
                    hold_cycle = hold_cycle + 1) begin
                @(posedge core_clk);
                @(negedge core_clk);
                if (m363_cfg_ready || m363_in_ready)
                    $fatal(1, "M364 config/input admitted while pipeline stalled");
                for (lane = 0; lane < PATTERNS; lane = lane + 1)
                    if (u_m363.pattern_q[lane] !== catalog_snapshot[lane])
                        $fatal(1, "M364 catalog changed before cfg handshake");
                cfg_block_cycles = cfg_block_cycles + 1;
            end

            m363_in_valid = 1'b0;
            m363_out_ready = 1'b1;
            direct_retired = 0;
            while (!m363_cfg_ready) begin
                @(posedge core_clk);
                if (m363_out_valid && m363_out_ready) begin
                    compare_attack_output(direct_retired);
                    direct_retired = direct_retired + 1;
                end
                @(negedge core_clk);
                if (!m363_cfg_ready)
                    cfg_block_cycles = cfg_block_cycles + 1;
            end
            if (direct_retired != 4 || u_m363.stage0_valid_q ||
                    u_m363.stage1_valid_q || u_m363.stage2_valid_q ||
                    m363_out_valid)
                $fatal(1, "M364 deferred config opened before true drain");
            @(posedge core_clk);
            deferred_cfg_handshakes = deferred_cfg_handshakes + 1;
            @(negedge core_clk);
            m363_cfg_valid = 1'b0;
            m363_cfg_commit = 1'b0;
            if (m363_cfg_active || u_m363.cfg_next_group_q != 1)
                $fatal(1, "M364 deferred group0 was not accepted exactly once");
            for (hold_cycle = 1; hold_cycle < 8;
                    hold_cycle = hold_cycle + 1)
                load_m363_group(hold_cycle, hold_cycle == 7);
            @(posedge core_clk);
            @(negedge core_clk);
            if (!m363_cfg_active || m363_cfg_protocol_error)
                $fatal(1, "M364 deferred reload did not reactivate catalog");
        end
    endtask

    task automatic attack_active_bad_reload_and_sticky_freeze;
        integer lane;
        integer attack_cycle;
        begin
            for (lane = 0; lane < PATTERNS; lane = lane + 1)
                catalog_snapshot[lane] = u_m363.pattern_q[lane];
            @(negedge core_clk);
            if (!m363_cfg_ready || !m363_cfg_active)
                $fatal(1, "M364 active-catalog attack precondition failed");
            m363_cfg_group = 1;
            m363_cfg_commit = 1'b0;
            m363_cfg_patterns_flat = {16{16'hbad1}};
            m363_cfg_valid = 1'b1;
            @(posedge core_clk);
            @(negedge core_clk);
            m363_cfg_valid = 1'b0;
            if (!m363_cfg_protocol_error || m363_cfg_active ||
                    m363_cfg_ready || m363_in_ready ||
                    u_m363.cfg_next_group_q != 0)
                $fatal(1, "M364 bad active reload did not enter quarantine");
            for (lane = 0; lane < PATTERNS; lane = lane + 1)
                if (u_m363.pattern_q[lane] !== catalog_snapshot[lane])
                    $fatal(1, "M364 bad reload mutated active catalog");

            for (attack_cycle = 0; attack_cycle < 64;
                    attack_cycle = attack_cycle + 1) begin
                m363_cfg_valid = 1'b1;
                m363_cfg_group = attack_cycle[2:0];
                m363_cfg_commit = (attack_cycle[2:0] == 7);
                m363_cfg_patterns_flat = {16{16'h6100 ^ attack_cycle[15:0]}};
                m363_in_valid = 1'b1;
                m363_in_original_pattern = 16'h9000 ^ attack_cycle[15:0];
                m363_out_ready = attack_cycle[0];
                @(posedge core_clk);
                if (m363_cfg_valid && m363_cfg_ready)
                    sticky_external_cfg_handshakes =
                        sticky_external_cfg_handshakes + 1;
                if (m363_in_valid && m363_in_ready)
                    sticky_input_handshakes = sticky_input_handshakes + 1;
                @(negedge core_clk);
                if (!m363_cfg_protocol_error || m363_cfg_active ||
                        m363_cfg_ready || m363_in_ready ||
                        u_m363.cfg_next_group_q != 0 ||
                        u_m363.stage0_valid_q || u_m363.stage1_valid_q ||
                        u_m363.stage2_valid_q || m363_out_valid)
                    $fatal(1, "M364 sticky error changed control/pipeline state");
                for (lane = 0; lane < PATTERNS; lane = lane + 1)
                    if (u_m363.pattern_q[lane] !== catalog_snapshot[lane])
                        $fatal(1, "M364 sticky error mutated catalog lane=%0d", lane);
                sticky_attack_cycles = sticky_attack_cycles + 1;
            end
            m363_cfg_valid = 1'b0;
            m363_cfg_commit = 1'b0;
            m363_in_valid = 1'b0;
            if (sticky_external_cfg_handshakes != 0 ||
                    sticky_input_handshakes != 0)
                $fatal(1, "M364 sticky quarantine admitted a handshake");
        end
    endtask

    task automatic attack_reset_mid_pipeline;
        integer lane;
        integer quiet_cycle;
        integer sentinel_latency;
        begin
            fill_m363_four(40);
            reset_mid_pipeline_count = reset_mid_pipeline_count + 1;
            reset_flushed_tokens = reset_flushed_tokens + 4;
            reset363_n = 1'b0;
            repeat (2) @(posedge core_clk);
            @(negedge core_clk);
            if (u_m363.stage0_valid_q || u_m363.stage1_valid_q ||
                    u_m363.stage2_valid_q || m363_out_valid ||
                    m363_cfg_active || m363_cfg_protocol_error ||
                    u_m363.cfg_next_group_q != 0)
                $fatal(1, "M364 mid-pipeline reset did not flush all slots");
            for (lane = 0; lane < PATTERNS; lane = lane + 1)
                if (u_m363.pattern_q[lane] !== 16'h0000)
                    $fatal(1, "M364 mid-pipeline reset retained catalog");
            reset363_n = 1'b1;
            m363_out_ready = 1'b1;
            for (quiet_cycle = 0; quiet_cycle < 10;
                    quiet_cycle = quiet_cycle + 1) begin
                @(posedge core_clk);
                if (m363_out_valid)
                    $fatal(1, "M364 stale pre-reset output escaped");
            end
            configure_m363();

            attack_expected[0] = expected[55];
            @(negedge core_clk);
            m363_in_original_pattern = stimulus[55];
            m363_in_valid = 1'b1;
            m363_out_ready = 1'b1;
            if (!m363_in_ready)
                $fatal(1, "M364 post-reset sentinel was not accepted");
            @(posedge core_clk);
            @(negedge core_clk);
            m363_in_valid = 1'b0;
            sentinel_latency = 0;
            while (sentinel_latency < 16) begin
                @(posedge core_clk);
                sentinel_latency = sentinel_latency + 1;
                if (m363_out_valid && m363_out_ready) begin
                    compare_attack_output(0);
                    break;
                end
            end
            if (sentinel_latency != 4)
                $fatal(1, "M364 post-reset sentinel latency=%0d", sentinel_latency);
            @(negedge core_clk);
            if (m363_out_valid)
                $fatal(1, "M364 extra stale output followed sentinel");
        end
    endtask

    genvar catalog_lane;
    generate
        for (catalog_lane = 0; catalog_lane < PATTERNS;
                catalog_lane = catalog_lane + 1) begin : g_catalog_observe
            assign m363_catalog_flat[catalog_lane * 16 +: 16] =
                u_m363.pattern_q[catalog_lane];
        end
    endgenerate

    m363_banked_q128_exact_signed_residual_matcher u_m363 (
        .core_clk(core_clk),
        .reset_n(reset363_n),
        .cfg_valid(m363_cfg_valid),
        .cfg_ready(m363_cfg_ready),
        .cfg_group(m363_cfg_group),
        .cfg_patterns_flat(m363_cfg_patterns_flat),
        .cfg_commit(m363_cfg_commit),
        .cfg_active(m363_cfg_active),
        .cfg_protocol_error(m363_cfg_protocol_error),
        .in_valid(m363_in_valid),
        .in_ready(m363_in_ready),
        .in_original_pattern(m363_in_original_pattern),
        .out_valid(m363_out_valid),
        .out_ready(m363_out_ready),
        .out_original_pattern(m363_out_original_pattern),
        .out_best_center(m363_out_best_center),
        .out_best_center_id(m363_out_best_center_id),
        .out_best_distance(m363_out_best_distance),
        .out_population(m363_out_population),
        .out_use_pwp(m363_out_use_pwp),
        .out_fallback_bit_sparse(m363_out_fallback_bit_sparse),
        .out_plus_mask(m363_out_plus_mask),
        .out_minus_mask(m363_out_minus_mask)
    );

    m356_failclosed_q128_signed_residual_matcher u_m356 (
        .core_clk(core_clk),
        .reset_n(reset356_n),
        .cfg_valid(m356_cfg_valid),
        .cfg_ready(m356_cfg_ready),
        .cfg_group(m356_cfg_group),
        .cfg_patterns_flat(m356_cfg_patterns_flat),
        .cfg_commit(m356_cfg_commit),
        .cfg_active(m356_cfg_active),
        .cfg_protocol_error(m356_cfg_protocol_error),
        .in_valid(m356_in_valid),
        .in_ready(m356_in_ready),
        .in_original_pattern(m356_in_original_pattern),
        .out_valid(m356_out_valid),
        .out_ready(m356_out_ready),
        .out_original_pattern(m356_out_original_pattern),
        .out_best_center(m356_out_best_center),
        .out_best_center_id(m356_out_best_center_id),
        .out_best_distance(m356_out_best_distance),
        .out_population(m356_out_population),
        .out_use_pwp(m356_out_use_pwp),
        .out_fallback_bit_sparse(m356_out_fallback_bit_sparse),
        .out_plus_mask(m356_out_plus_mask),
        .out_minus_mask(m356_out_minus_mask)
    );

    m363_banked_q128_exact_signed_residual_matcher_assertions u_m363_author_sva (
        .core_clk(core_clk),
        .reset_n(reset363_n),
        .cfg_valid(m363_cfg_valid),
        .cfg_ready(m363_cfg_ready),
        .cfg_active(m363_cfg_active),
        .cfg_protocol_error(m363_cfg_protocol_error),
        .in_valid(m363_in_valid),
        .in_ready(m363_in_ready),
        .out_valid(m363_out_valid),
        .out_ready(m363_out_ready),
        .out_original_pattern(m363_out_original_pattern),
        .out_best_center(m363_out_best_center),
        .out_best_center_id(m363_out_best_center_id),
        .out_best_distance(m363_out_best_distance),
        .out_population(m363_out_population),
        .out_use_pwp(m363_out_use_pwp),
        .out_fallback_bit_sparse(m363_out_fallback_bit_sparse),
        .out_plus_mask(m363_out_plus_mask),
        .out_minus_mask(m363_out_minus_mask)
    );

    m364_banked_q128_independent_hammer_assertions u_m364_sva (
        .core_clk(core_clk),
        .reset_n(reset363_n),
        .cfg_valid(m363_cfg_valid),
        .cfg_ready(m363_cfg_ready),
        .cfg_active(m363_cfg_active),
        .cfg_protocol_error(m363_cfg_protocol_error),
        .in_valid(m363_in_valid),
        .in_ready(m363_in_ready),
        .out_valid(m363_out_valid),
        .out_ready(m363_out_ready),
        .out_original_pattern(m363_out_original_pattern),
        .out_best_center(m363_out_best_center),
        .out_best_center_id(m363_out_best_center_id),
        .out_best_distance(m363_out_best_distance),
        .out_population(m363_out_population),
        .out_use_pwp(m363_out_use_pwp),
        .out_fallback_bit_sparse(m363_out_fallback_bit_sparse),
        .out_plus_mask(m363_out_plus_mask),
        .out_minus_mask(m363_out_minus_mask),
        .stage0_valid(u_m363.stage0_valid_q),
        .stage1_valid(u_m363.stage1_valid_q),
        .stage2_valid(u_m363.stage2_valid_q),
        .cfg_next_group(u_m363.cfg_next_group_q),
        .catalog_flat(m363_catalog_flat)
    );

    always @(posedge core_clk) begin
        if (reset363_n)
            cycle363 = cycle363 + 1;
        if (main_run && reset363_n) begin
            if (m363_in_valid && m363_in_ready) begin
                if (accepted363 >= TOTAL ||
                        m363_in_original_pattern !== stimulus[accepted363])
                    $fatal(1, "M364 M363 producer/order error accepted=%0d",
                           accepted363);
                accept_cycle363[accepted363] = cycle363;
                accepted363 = accepted363 + 1;
                accept_run363 = accept_run363 + 1;
                if (accept_run363 > max_accept_run363)
                    max_accept_run363 = accept_run363;
            end else begin
                accept_run363 = 0;
            end
            if (m363_out_valid && !m363_out_ready)
                stalls363 = stalls363 + 1;
            if (m363_out_valid && m363_out_ready) begin
                if (retired363 >= accepted363)
                    $fatal(1, "M364 M363 output without accepted input");
                actual363.original = m363_out_original_pattern;
                actual363.center = m363_out_best_center;
                actual363.center_id = m363_out_best_center_id;
                actual363.distance = m363_out_best_distance;
                actual363.population = m363_out_population;
                actual363.use_pwp = m363_out_use_pwp;
                actual363.fallback = m363_out_fallback_bit_sparse;
                actual363.plus_mask = m363_out_plus_mask;
                actual363.minus_mask = m363_out_minus_mask;
                if (actual363 !== expected[retired363]) begin
                    mismatch363 = mismatch363 + 1;
                    $fatal(1, "M364 M363 independent numerical/order mismatch index=%0d",
                           retired363);
                end
                observed363[retired363] = actual363;
                if (actual363.use_pwp)
                    use363 = use363 + 1;
                else
                    fallback363 = fallback363 + 1;
                if (actual363.use_pwp && actual363.plus_mask != 0 &&
                        actual363.minus_mask != 0)
                    mixed363 = mixed363 + 1;
                if (actual363.distance == 0)
                    exact363 = exact363 + 1;
                if (cycle363 - accept_cycle363[retired363] < latency_min363)
                    latency_min363 = cycle363 - accept_cycle363[retired363];
                if (cycle363 - accept_cycle363[retired363] > latency_max363)
                    latency_max363 = cycle363 - accept_cycle363[retired363];
                retired363 = retired363 + 1;
                retire_run363 = retire_run363 + 1;
                if (retire_run363 > max_retire_run363)
                    max_retire_run363 = retire_run363;
            end else begin
                retire_run363 = 0;
            end
        end
    end

    always @(posedge core_clk) begin
        if (reset356_n)
            cycle356 = cycle356 + 1;
        if (main_run && reset356_n) begin
            if (m356_in_valid && m356_in_ready) begin
                if (accepted356 >= TOTAL ||
                        m356_in_original_pattern !== stimulus[accepted356])
                    $fatal(1, "M364 M356 producer/order error accepted=%0d",
                           accepted356);
                accept_cycle356[accepted356] = cycle356;
                accepted356 = accepted356 + 1;
                accept_run356 = accept_run356 + 1;
                if (accept_run356 > max_accept_run356)
                    max_accept_run356 = accept_run356;
            end else begin
                accept_run356 = 0;
            end
            if (m356_out_valid && !m356_out_ready)
                stalls356 = stalls356 + 1;
            if (m356_out_valid && m356_out_ready) begin
                if (retired356 >= accepted356)
                    $fatal(1, "M364 M356 output without accepted input");
                actual356.original = m356_out_original_pattern;
                actual356.center = m356_out_best_center;
                actual356.center_id = m356_out_best_center_id;
                actual356.distance = m356_out_best_distance;
                actual356.population = m356_out_population;
                actual356.use_pwp = m356_out_use_pwp;
                actual356.fallback = m356_out_fallback_bit_sparse;
                actual356.plus_mask = m356_out_plus_mask;
                actual356.minus_mask = m356_out_minus_mask;
                if (actual356 !== expected[retired356]) begin
                    mismatch356 = mismatch356 + 1;
                    $fatal(1, "M364 M356 independent numerical/order mismatch index=%0d",
                           retired356);
                end
                observed356[retired356] = actual356;
                if (cycle356 - accept_cycle356[retired356] < latency_min356)
                    latency_min356 = cycle356 - accept_cycle356[retired356];
                if (cycle356 - accept_cycle356[retired356] > latency_max356)
                    latency_max356 = cycle356 - accept_cycle356[retired356];
                retired356 = retired356 + 1;
                retire_run356 = retire_run356 + 1;
                if (retire_run356 > max_retire_run356)
                    max_retire_run356 = retire_run356;
            end else begin
                retire_run356 = 0;
            end
        end
    end

    always @(negedge core_clk) begin
        if (main_run) begin
            main_tick = main_tick + 1;

            if (main_tick < 340)
                m363_out_ready = 1'b1;
            else if (main_tick < 404)
                m363_out_ready = 1'b0;
            else
                m363_out_ready = ((main_tick % 13) != 0) &&
                                 ((main_tick % 47) < 43);
            m356_out_ready = m363_out_ready;

            if (generated363 == accepted363) begin
                if (generated363 < TOTAL &&
                        (generated363 < 320 ||
                         (((main_tick % 7) != 0) &&
                          ((main_tick % 29) != 0)))) begin
                    m363_in_valid = 1'b1;
                    m363_in_original_pattern = stimulus[generated363];
                    generated363 = generated363 + 1;
                end else begin
                    m363_in_valid = 1'b0;
                    if (generated363 < TOTAL)
                        bubbles363 = bubbles363 + 1;
                end
            end

            if (generated356 == accepted356) begin
                if (generated356 < TOTAL &&
                        (generated356 < 320 ||
                         (((main_tick % 5) != 0) &&
                          ((main_tick % 31) != 0)))) begin
                    m356_in_valid = 1'b1;
                    m356_in_original_pattern = stimulus[generated356];
                    generated356 = generated356 + 1;
                end else begin
                    m356_in_valid = 1'b0;
                    if (generated356 < TOTAL)
                        bubbles356 = bubbles356 + 1;
                end
            end
        end
    end

    integer init_index;
    integer compare_index;
    integer watchdog;
    initial begin
        m363_cfg_valid = 0;
        m363_cfg_group = 0;
        m363_cfg_patterns_flat = 0;
        m363_cfg_commit = 0;
        m363_in_valid = 0;
        m363_in_original_pattern = 0;
        m363_out_ready = 0;
        m356_cfg_valid = 0;
        m356_cfg_group = 0;
        m356_cfg_patterns_flat = 0;
        m356_cfg_commit = 0;
        m356_in_valid = 0;
        m356_in_original_pattern = 0;
        m356_out_ready = 0;
        main_run = 0;
        main_tick = 0;
        generated363 = 0;
        accepted363 = 0;
        retired363 = 0;
        generated356 = 0;
        accepted356 = 0;
        retired356 = 0;
        cycle363 = 0;
        cycle356 = 0;
        mismatch363 = 0;
        mismatch356 = 0;
        pairwise_mismatch = 0;
        max_accept_run363 = 0;
        max_retire_run363 = 0;
        accept_run363 = 0;
        retire_run363 = 0;
        max_accept_run356 = 0;
        max_retire_run356 = 0;
        accept_run356 = 0;
        retire_run356 = 0;
        latency_min363 = 1 << 30;
        latency_max363 = 0;
        latency_min356 = 1 << 30;
        latency_max356 = 0;
        stalls363 = 0;
        stalls356 = 0;
        bubbles363 = 0;
        bubbles356 = 0;
        use363 = 0;
        fallback363 = 0;
        mixed363 = 0;
        exact363 = 0;
        cfg_block_cycles = 0;
        sticky_attack_cycles = 0;
        sticky_external_cfg_handshakes = 0;
        sticky_input_handshakes = 0;
        reset_mid_pipeline_count = 0;
        reset_flushed_tokens = 0;
        deferred_cfg_handshakes = 0;

        for (init_index = 0; init_index < PATTERNS;
                init_index = init_index + 1)
            patterns[init_index] =
                16'h2101 + init_index * 16'h01f3;
        patterns[2] = 16'ha55a;
        patterns[5] = 16'h0003;
        patterns[7] = 16'h00fe;
        patterns[9] = 16'h0005;
        patterns[37] = 16'h5a3c;
        patterns[73] = 16'ha55a;

        for (init_index = 0; init_index < TOTAL;
                init_index = init_index + 1) begin
            stimulus[init_index] =
                ((init_index * 16'h9e37) ^
                 ((init_index * 16'h1357) >> 3) ^ 16'h6d2b);
            if ((init_index % 17) == 0)
                stimulus[init_index] = patterns[init_index % PATTERNS];
            else if ((init_index % 19) == 0)
                stimulus[init_index] = patterns[init_index % PATTERNS] ^
                    (16'h0001 << (init_index % 16));
        end
        stimulus[0] = 16'ha55a;
        stimulus[1] = 16'h0000;
        stimulus[2] = 16'hffff;
        stimulus[3] = 16'h00fd;
        stimulus[4] = 16'h7ffe;
        for (init_index = 0; init_index < TOTAL;
                init_index = init_index + 1)
            calculate_reference(stimulus[init_index], expected[init_index]);
        if (expected[0].center_id != 2 || expected[0].distance != 0)
            $fatal(1, "M364 directed lowest-ID reference precondition failed");

        repeat (4) @(posedge core_clk);
        reset_m363();
        configure_m363();

        attack_deferred_configuration();
        attack_active_bad_reload_and_sticky_freeze();

        reset_m363();
        configure_m363();
        attack_reset_mid_pipeline();

        reset_m356();
        configure_m356();

        @(negedge core_clk);
        main_run = 1'b1;
        watchdog = 0;
        while ((retired363 < TOTAL || retired356 < TOTAL) &&
                watchdog < 100000) begin
            @(posedge core_clk);
            watchdog = watchdog + 1;
        end
        @(negedge core_clk);
        main_run = 1'b0;
        m363_in_valid = 1'b0;
        m356_in_valid = 1'b0;
        m363_out_ready = 1'b1;
        m356_out_ready = 1'b1;

        if (watchdog >= 100000)
            $fatal(1, "M364 main comparison watchdog timeout");
        for (compare_index = 0; compare_index < TOTAL;
                compare_index = compare_index + 1) begin
            if (observed363[compare_index] !== observed356[compare_index])
                pairwise_mismatch = pairwise_mismatch + 1;
        end

        if (accepted363 != TOTAL || retired363 != TOTAL ||
                accepted356 != TOTAL || retired356 != TOTAL ||
                mismatch363 != 0 || mismatch356 != 0 ||
                pairwise_mismatch != 0 || expected[0].center_id != 2 ||
                observed363[0].center_id != 2 ||
                max_accept_run363 < 256 || max_retire_run363 < 256 ||
                max_accept_run356 < 256 || max_retire_run356 < 128 ||
                latency_min363 != 4 || latency_min356 != 128 ||
                stalls363 < 64 || stalls356 < 64 ||
                bubbles363 == 0 || bubbles356 == 0 ||
                use363 == 0 || fallback363 == 0 || mixed363 == 0 ||
                exact363 == 0 || cfg_block_cycles < 24 ||
                sticky_attack_cycles != 64 ||
                sticky_external_cfg_handshakes != 0 ||
                sticky_input_handshakes != 0 ||
                reset_mid_pipeline_count != 1 || reset_flushed_tokens != 4 ||
                deferred_cfg_handshakes != 1)
            $fatal(1, "M364 coverage/termination gate failed");

        $display("PASS M364 independent hammer m363_transactions=%0d m356_transactions=%0d m363_mismatches=%0d m356_mismatches=%0d pairwise_mismatches=%0d lowest_id_tie_id=%0d use=%0d fallback=%0d mixed_signed=%0d exact=%0d m363_stalls=%0d m356_stalls=%0d m363_bubbles=%0d m356_bubbles=%0d m363_max_accept_run=%0d m363_max_retire_run=%0d m356_max_accept_run=%0d m356_max_retire_run=%0d m363_latency_min=%0d m363_latency_max=%0d m356_latency_min=%0d m356_latency_max=%0d cfg_block_cycles=%0d deferred_cfg_handshakes=%0d sticky_attack_cycles=%0d sticky_cfg_handshakes=%0d sticky_input_handshakes=%0d mid_pipeline_resets=%0d flushed_tokens=%0d numeric_signed_order_equivalent=true elastic_ii1=true system_speedup=false headline=false",
                 retired363, retired356, mismatch363, mismatch356,
                 pairwise_mismatch, observed363[0].center_id,
                 use363, fallback363, mixed363, exact363,
                 stalls363, stalls356, bubbles363, bubbles356,
                 max_accept_run363, max_retire_run363,
                 max_accept_run356, max_retire_run356,
                 latency_min363, latency_max363,
                 latency_min356, latency_max356,
                 cfg_block_cycles, deferred_cfg_handshakes,
                 sticky_attack_cycles, sticky_external_cfg_handshakes,
                 sticky_input_handshakes, reset_mid_pipeline_count,
                 reset_flushed_tokens);
        $finish;
    end

endmodule
