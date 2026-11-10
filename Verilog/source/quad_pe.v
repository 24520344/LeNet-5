`timescale 1ns / 1ps

//====================================================================================
// Organization:    University of Information Technology - VNU-HCM
// Author:          Dung Nguyen Viet
// Student ID:      24520344
//
// File Name:       quad_pe.v
// Module Name:     quad_pe
// Project:         Digit Recognition Using LeNet-5 CNN
// Software:        Vivado 2025.1 & Visual Studio Code
//
// Created:         Sunday, April 5, 2026
// Description:     
//
// History:        
// Date        By      Version    Change Description
// -----------------------------------------------------------------------------------
// 05-04-2026  DungNV  1.0        Initial release
//====================================================================================

module quad_pe #(
    parameter DATA_WIDTH = 16,
    parameter FRACTIONAL_BITS = 10,
    parameter KERNEL_WIDTH = 5,
    parameter NUM_PE = 4
) (
    input wire clk,
    input wire rst_n,
    input wire en,
    input wire first_channel,
    input wire [8:0] mac_limit,

    input wire signed [DATA_WIDTH - 1:0] pixel_in,
    input wire signed [NUM_PE * DATA_WIDTH-1:0] weight_in,
    input wire signed [NUM_PE * DATA_WIDTH-1:0] bias_in,

    output wire valid_out,
    output wire signed [NUM_PE * DATA_WIDTH-1:0] pixel_out
);

    wire [3:0] valid_pe;
    assign valid_out = &valid_pe;

    wire [DATA_WIDTH - 1:0] pixel_out_0, pixel_out_1, pixel_out_2, pixel_out_3;
    assign pixel_out = {pixel_out_3, pixel_out_2, pixel_out_1, pixel_out_0};

    pe #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRACTIONAL_BITS(FRACTIONAL_BITS),
        .KERNEL_WIDTH(KERNEL_WIDTH)
    ) pe0 (
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .first_channel(first_channel),
        .mac_limit(mac_limit),

        .pixel_in(pixel_in),
        .weight_in(weight_in[DATA_WIDTH - 1 -: DATA_WIDTH]),
        .bias_in(bias_in[DATA_WIDTH - 1 -: DATA_WIDTH]),

        .valid_out(valid_pe[0]),
        .pixel_out(pixel_out_0)
    );

    pe #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRACTIONAL_BITS(FRACTIONAL_BITS),
        .KERNEL_WIDTH(KERNEL_WIDTH)
    ) pe1 (
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .first_channel(first_channel),
        .mac_limit(mac_limit),

        .pixel_in(pixel_in),
        .weight_in(weight_in[2 * DATA_WIDTH - 1 -: DATA_WIDTH]),
        .bias_in(bias_in[2 * DATA_WIDTH - 1 -: DATA_WIDTH]),

        .valid_out(valid_pe[1]),
        .pixel_out(pixel_out_1)
    );

    pe #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRACTIONAL_BITS(FRACTIONAL_BITS),
        .KERNEL_WIDTH(KERNEL_WIDTH)
    ) pe2 (
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .first_channel(first_channel),
        .mac_limit(mac_limit),

        .pixel_in(pixel_in),
        .weight_in(weight_in[3 * DATA_WIDTH - 1 -: DATA_WIDTH]),
        .bias_in(bias_in[3 * DATA_WIDTH - 1 -: DATA_WIDTH]),

        .valid_out(valid_pe[2]),
        .pixel_out(pixel_out_2)
    );

    pe #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRACTIONAL_BITS(FRACTIONAL_BITS),
        .KERNEL_WIDTH(KERNEL_WIDTH)
    ) pe3 (
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .first_channel(first_channel),
        .mac_limit(mac_limit),

        .pixel_in(pixel_in),
        .weight_in(weight_in[4 * DATA_WIDTH - 1 -: DATA_WIDTH]),
        .bias_in(bias_in[4 * DATA_WIDTH - 1 -: DATA_WIDTH]),

        .valid_out(valid_pe[3]),
        .pixel_out(pixel_out_3)
    );

endmodule