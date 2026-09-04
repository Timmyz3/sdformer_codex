module m356_failclosed_q128_signed_residual_matcher (
    input  logic          core_clk,
    input  logic          reset_n,
    input  logic          cfg_valid,
    output logic          cfg_ready,
    input  logic [2:0]    cfg_group,
    input  logic [255:0]  cfg_patterns_flat,
    input  logic          cfg_commit,
    output logic          cfg_active,
    output logic          cfg_protocol_error,
    input  logic          in_valid,
    output logic          in_ready,
    input  logic [15:0]   in_original_pattern,
    output logic          out_valid,
    input  logic          out_ready,
    output logic [15:0]   out_original_pattern,
    output logic [15:0]   out_best_center,
    output logic [6:0]    out_best_center_id,
    output logic [4:0]    out_best_distance,
    output logic [4:0]    out_population,
    output logic          out_use_pwp,
    output logic          out_fallback_bit_sparse,
    output logic [15:0]   out_plus_mask,
    output logic [15:0]   out_minus_mask
);

    logic core_cfg_ready;
    logic core_cfg_active;
    logic core_cfg_protocol_error;
    logic core_in_ready;

    // M350 found that the numeric M348 core could accept another complete
    // configuration after a sticky protocol error and reassert cfg_active.
    // Quarantine every configuration and input handshake until reset, and mask
    // the externally visible active state.  The underlying sticky error is
    // cleared only by reset_n, so the wrapper itself adds no recovery path.
    assign cfg_ready = core_cfg_ready && !core_cfg_protocol_error;
    assign cfg_active = core_cfg_active && !core_cfg_protocol_error;
    assign cfg_protocol_error = core_cfg_protocol_error;
    assign in_ready = core_in_ready && !core_cfg_protocol_error;

    m348_exact_q128_signed_residual_matcher u_numeric_core (
        .core_clk(core_clk),
        .reset_n(reset_n),
        .cfg_valid(cfg_valid && !core_cfg_protocol_error),
        .cfg_ready(core_cfg_ready),
        .cfg_group(cfg_group),
        .cfg_patterns_flat(cfg_patterns_flat),
        .cfg_commit(cfg_commit),
        .cfg_active(core_cfg_active),
        .cfg_protocol_error(core_cfg_protocol_error),
        .in_valid(in_valid && !core_cfg_protocol_error),
        .in_ready(core_in_ready),
        .in_original_pattern(in_original_pattern),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_original_pattern(out_original_pattern),
        .out_best_center(out_best_center),
        .out_best_center_id(out_best_center_id),
        .out_best_distance(out_best_distance),
        .out_population(out_population),
        .out_use_pwp(out_use_pwp),
        .out_fallback_bit_sparse(out_fallback_bit_sparse),
        .out_plus_mask(out_plus_mask),
        .out_minus_mask(out_minus_mask)
    );

endmodule
