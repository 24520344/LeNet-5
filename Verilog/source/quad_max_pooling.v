`timescale 1ns / 1ps

//====================================================================================
// Organization:    University of Information Technology - VNU-HCM
// Author:          Dung Nguyen Viet
// Student ID:      24520344
//
// File Name:       quad_max_pooling_modified.v
// Module Name:     quad_max_pooling_modified
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

module quad_max_pooling_modified #(
    parameter DATA_WIDTH = 16,
    parameter FEATURE_IN_WIDTH = 28,
    parameter BRAM_SIZE = 196,
    parameter NUM_PE = 4
) (
    input wire clk,
    input wire rst_n,
    input wire wr_en,
    input wire clear,
    input wire [$clog2(FEATURE_IN_WIDTH) - 1:0] buffer_limit,
    input wire signed [NUM_PE * DATA_WIDTH - 1:0] pixel_in,

    input wire [$clog2(BRAM_SIZE) - 1:0] bram_limit, 
    input wire clear_bram,

    output wire valid_out,
    output wire signed [NUM_PE * DATA_WIDTH - 1:0] pixel_out,
    
    output wire wr_en_bram,
    output wire [$clog2(BRAM_SIZE) - 1:0] w_addr_bram,
    output wire full_bram
);

    wire [3:0] valid_mp;
    assign valid_out = &valid_mp;

    wire [DATA_WIDTH - 1:0] pixel_out_0, pixel_out_1, pixel_out_2, pixel_out_3;
    assign pixel_out = {pixel_out_3, pixel_out_2, pixel_out_1, pixel_out_0};

    max_pooling_modified #(
        .DATA_WIDTH(DATA_WIDTH),
        .FEATURE_IN_WIDTH(FEATURE_IN_WIDTH),
        .BRAM_SIZE(BRAM_SIZE),
        .NUM_PE(NUM_PE)
    ) max_pooling_0 (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en),
        .clear(clear),
        .buffer_limit(buffer_limit),
        
        .bram_limit(bram_limit),
        .clear_bram(clear_bram),

        .pixel_in(pixel_in[DATA_WIDTH - 1 -: DATA_WIDTH]),

        .wr_en_bram(wr_en_bram),
        .w_addr_bram(w_addr_bram),
        .full_bram(full_bram),

        .valid_out(valid_mp[0]),
        .pixel_out(pixel_out_0)
    );

    max_pooling_modified #(
        .DATA_WIDTH(DATA_WIDTH),
        .FEATURE_IN_WIDTH(FEATURE_IN_WIDTH),
        .BRAM_SIZE(BRAM_SIZE),
        .NUM_PE(NUM_PE)
    ) max_pooling_1 (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en),
        .clear(clear),
        .buffer_limit(buffer_limit),

        .bram_limit(bram_limit),
        .clear_bram(clear_bram),

        .pixel_in(pixel_in[2 * DATA_WIDTH - 1 -: DATA_WIDTH]),

        .wr_en_bram(),
        .w_addr_bram(),
        .full_bram(),

        .valid_out(valid_mp[1]),
        .pixel_out(pixel_out_1)
    );

    max_pooling_modified #(
        .DATA_WIDTH(DATA_WIDTH),
        .FEATURE_IN_WIDTH(FEATURE_IN_WIDTH),
        .BRAM_SIZE(BRAM_SIZE),
        .NUM_PE(NUM_PE)
    ) max_pooling_2 (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en),
        .clear(clear),
        .buffer_limit(buffer_limit),

        .bram_limit(bram_limit),
        .clear_bram(clear_bram),

        .pixel_in(pixel_in[3 * DATA_WIDTH - 1 -: DATA_WIDTH]),

        .wr_en_bram(),
        .w_addr_bram(),
        .full_bram(),

        .valid_out(valid_mp[2]),
        .pixel_out(pixel_out_2)
    );

    max_pooling_modified #(
        .DATA_WIDTH(DATA_WIDTH),
        .FEATURE_IN_WIDTH(FEATURE_IN_WIDTH),
        .BRAM_SIZE(BRAM_SIZE),
        .NUM_PE(NUM_PE)
    ) max_pooling_3 (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en),
        .clear(clear),
        .buffer_limit(buffer_limit),

        .bram_limit(bram_limit),
        .clear_bram(clear_bram),

        .pixel_in(pixel_in[4 * DATA_WIDTH - 1 -: DATA_WIDTH]),

        .wr_en_bram(),
        .w_addr_bram(),
        .full_bram(),

        .valid_out(valid_mp[3]),
        .pixel_out(pixel_out_3)
    );
    
endmodule