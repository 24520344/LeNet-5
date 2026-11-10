`timescale 1ns / 1ps

//====================================================================================
// Organization:    University of Information Technology - VNU-HCM
// Author:          Dung Nguyen Viet
// Student ID:      24520344
//
// File Name:       bram.v
// Module Name:     bram
// Project:         Digit Recognition Using LeNet-5 CNN
// Software:        Vivado 2025.1 & Visual Studio Code
//
// Created:         Sunday, March 29, 2026
// Description:     
//
// History:        
// Date        By      Version    Change Description
// -----------------------------------------------------------------------------------
// 29-03-2026  DungNV  1.0        Initial release
//====================================================================================

module bram #(
    parameter DATA_WIDTH = 16,
    parameter SIZE = 196,
    parameter NUM_PE = 4
) (
    input wire clk,
    input wire rst_n,
    input wire wr_en,
    input wire rd_en,
    
    input wire [$clog2(SIZE) - 1:0] w_addr,
    input wire [$clog2(SIZE) - 1:0] r_addr,
    input wire signed [NUM_PE * DATA_WIDTH - 1:0] data_in,

    output reg valid_out,
    output reg signed [NUM_PE * DATA_WIDTH - 1:0] data_out
);

    (* ram_style = "block" *) reg signed [NUM_PE * DATA_WIDTH - 1:0] ram [SIZE - 1:0];

    always @(posedge clk) begin
        if (~rst_n) begin
            data_out <= 0;
        end

        else begin
            if (wr_en)
                ram[w_addr] <= data_in;

            if (rd_en) begin
                data_out <= ram[r_addr];
                valid_out <= 1'b1;
            end

            else
                valid_out <= 1'b0;
        end
    end
   
endmodule