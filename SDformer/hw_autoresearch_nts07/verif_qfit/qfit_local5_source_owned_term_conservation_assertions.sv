`timescale 1ns/1ps
`default_nettype none

module qfit_local5_source_owned_term_conservation_assertions (
    input logic        clk_core,
    input logic        rst_core,
    input logic        projection_start,
    input logic        projection_done,
    input logic [31:0] builder_terms,
    input logic [31:0] builder_updates,
    input logic [31:0] backend_terms,
    input logic [31:0] backend_updates
);
    logic [31:0] term_base_q;
    logic [31:0] update_base_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            term_base_q <= '0;
            update_base_q <= '0;
        end else begin
            if (projection_start) begin
                term_base_q <= builder_terms;
                update_base_q <= builder_updates;
            end
            if (projection_done) begin
                assert (builder_terms - term_base_q == backend_terms)
                    else $error(
                        "source-owned term conservation failed builder=%0d backend=%0d",
                        builder_terms - term_base_q,
                        backend_terms
                    );
                assert (builder_updates - update_base_q == backend_updates)
                    else $error(
                        "source-owned update conservation failed builder=%0d backend=%0d",
                        builder_updates - update_base_q,
                        backend_updates
                    );
            end
        end
    end
endmodule

bind qfit_local5_active_projection_tile
    qfit_local5_source_owned_term_conservation_assertions
    i_source_owned_term_conservation_assertions (.*);

`default_nettype wire
