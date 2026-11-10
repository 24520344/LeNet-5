`timescale 1ns / 1ps

//====================================================================================
// Organization:    University of Information Technology - VNU-HCM
// Author:          Dung Nguyen Viet
// Student ID:      24520344
//
// File Name:       quad_relu.v
// Module Name:     quad_relu
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

module quad_relu #(
    parameter DATA_WIDTH = 16,
    parameter NUM_PE = 4
)(
    input wire signed [NUM_PE * DATA_WIDTH - 1:0] data_in,
    output wire signed [NUM_PE * DATA_WIDTH - 1:0] data_out
);

    relu #(
        .DATA_WIDTH(DATA_WIDTH)
    ) relu0 (
        .data_in(data_in[DATA_WIDTH - 1 -: DATA_WIDTH]),
        .data_out(data_out[DATA_WIDTH - 1 -: DATA_WIDTH])  
    );

    relu #(
        .DATA_WIDTH(DATA_WIDTH)
    ) relu1 (
        .data_in(data_in[2 * DATA_WIDTH - 1 -: DATA_WIDTH]),
        .data_out(data_out[2 * DATA_WIDTH - 1 -: DATA_WIDTH])  
    );

    relu #(
        .DATA_WIDTH(DATA_WIDTH)
    ) relu2 (
        .data_in(data_in[3 * DATA_WIDTH - 1 -: DATA_WIDTH]),
        .data_out(data_out[3 * DATA_WIDTH - 1 -: DATA_WIDTH])
    );

    relu #(
        .DATA_WIDTH(DATA_WIDTH)
    ) relu3 (
        .data_in(data_in[4 * DATA_WIDTH - 1 -: DATA_WIDTH]),
        .data_out(data_out[4 * DATA_WIDTH - 1 -: DATA_WIDTH])
    );

endmodule