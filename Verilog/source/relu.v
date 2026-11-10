`timescale 1ns / 1ps

//====================================================================================
// Organization:    University of Information Technology - VNU-HCM
// Author:          Dung Nguyen Viet
// Student ID:      24520344
//
// File Name:       relu.v
// Module Name:     relu
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

module relu #(
    parameter DATA_WIDTH = 16
)(
    input wire signed [DATA_WIDTH - 1:0] data_in,
    output wire signed [DATA_WIDTH - 1:0] data_out
);

    // Sử dụng bit dấu (MSB) làm tín hiệu điều khiển (Selector)
    // +) Giá trị dương -> trả về chính nó
    // +) Giá trị âm -> trả về 0
    assign data_out = (data_in[DATA_WIDTH-1]) ? {DATA_WIDTH{1'b0}} : data_in;


endmodule