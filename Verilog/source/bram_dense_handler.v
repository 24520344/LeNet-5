`timescale 1ns / 1ps

module bram_dense_handler #(
    parameter DATA_WIDTH = 16,
    parameter BRAM_SIZE = 196
) (
    input wire clk,
    input wire rst_n,
    input wire clear_bram_dense,
    input wire [$clog2(BRAM_SIZE) - 1:0] bram_dense_limit,
    input wire bram_in_sel,
    input wire valid_out_quad_pe,

    output reg [$clog2(BRAM_SIZE) - 1:0] w_addr_bram_dense,
    output reg full_bram_dense
);

    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            w_addr_bram_dense <= 0;
            full_bram_dense <= 0;
        end
		  
		  else if (clear_bram_dense) begin
            w_addr_bram_dense <= 0;
            full_bram_dense <= 0;
		  end
		  
        else if (bram_in_sel && valid_out_quad_pe) begin
            if (w_addr_bram_dense == bram_dense_limit - 1) begin
                full_bram_dense <= 1'b1;
                w_addr_bram_dense <= 0;
            end

            else
                w_addr_bram_dense <= w_addr_bram_dense +8'b1;
        end
    end

endmodule