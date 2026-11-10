`timescale 1ns / 1ps

//====================================================================================
// Organization:    University of Information Technology - VNU-HCM
// Author:          Dung Nguyen Viet
// Student ID:      24520344
//
// File Name:       top_module.v
// Module Name:     top_module
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

module top_module #(
    // Tham số chung
    parameter DATA_WIDTH = 16,
    parameter KERNEL_WIDTH = 5,
    parameter NUM_PE = 4,

    // Tham số PE/ QUAD PE
    parameter FRACTIONAL_BITS = 10,

    // Tham số riêng Max Pooling/ Quad Max Pooling
    parameter FEATURE_IN_WIDTH = 28,

    // Tham số riêng Line Buffer
    parameter INPUT_IMAGE_WIDTH = 32,

    // Tham số riêng Weight Buffer
    parameter NUM_FIFO_WB = 4,
    parameter NUM_PE_WB = 4,
    parameter BUFFER_SIZE_WB = 4800,

    // Tham số riêng Bias Buffer
    parameter NUM_FIFO_BB = 4,
    parameter NUM_PE_BB = 4,
    parameter BUFFER_SIZE_BB = 16,

    // Tham số riêng BRAM
    parameter BRAM_SIZE = 196

) (
    input wire clk,
    input wire rst_n,
    input wire start,

    input wire signed [DATA_WIDTH - 1:0] pixel_in,
    input wire signed [DATA_WIDTH * NUM_FIFO_WB - 1:0] weight_in,
    input wire signed [DATA_WIDTH * NUM_FIFO_BB - 1:0] bias_in,

    output wire done,
    output wire [1:0] curr_layer,
    output wire wr_en_bb,
    output wire wr_en_wb,
    output wire wr_en_lb,
    output wire [3:0] predicted_digit

);
    // Các ngõ internal nối controller với datapath
    wire bram_in_sel;

    wire bram_sel;
    
    // Giới hạn số lần chạy quad pe cho mỗi layer
    wire [8:0] mac_limit;

    wire [1:0] quad_pe_channels_cnt;



    // =============================================================================
    // PE / QUAD PE (khối tính toán tích chập)

    wire en_quad_pe;
    wire first_channel_quad_pe;
    

    // =============================================================================
    // MAX POOLING (khối nhóm và xuất pixel lớn nhất)

    wire clear_mp;
    wire [$clog2(FEATURE_IN_WIDTH) - 1:0] buffer_limit_mp;

    // Tín hiệu điều khiển BRAM từ controller
    wire clear_bram;
    wire [$clog2(BRAM_SIZE) - 1:0] bram_limit;
    

    // =============================================================================
    // LINE BUFFER (RAM lưu ảnh đầu vào)

    wire valid_out_lb;
    wire full_lb;

    //wire wr_en_lb;
    wire rd_en_lb;
    wire clear_lb;
    wire [$clog2(INPUT_IMAGE_WIDTH):0] buffer_limit_lb;
    wire [KERNEL_WIDTH - 1:0] output_index;


    // =============================================================================
    // BIAS BUFFER (RAM lưu bias)

    wire full_bb;

    //wire wr_en_bb;
    wire rd_en_bb;
    wire clear_bb;
    wire [$clog2(BUFFER_SIZE_BB) - 1:0] rd_addr_bb;
    wire [$clog2(BUFFER_SIZE_BB):0] buffer_limit_bb;


    // =============================================================================
    // WEIGHT BUFFER (RAM lưu weight)

    wire full_wb;

    //wire wr_en_wb;
    wire rd_en_wb;
    wire clear_wb;
    wire [$clog2(BUFFER_SIZE_WB) - 1:0] rd_addr_wb;
    wire [$clog2(BUFFER_SIZE_WB):0] buffer_limit_wb;


    // =============================================================================
    // BRAM DENSE HANDLER (Khối xử lý tín hiệu điều khiển với lớp dense)

    wire full_bram_dense;

    wire clear_bram_dense;
    wire [$clog2(BRAM_SIZE) - 1:0] bram_dense_limit;

    
    // =============================================================================
    // BRAM (RAM trung gian lưu feature maps)

    wire full_bram;

    wire rd_en_bram;
    wire [$clog2(BRAM_SIZE) - 1:0] rd_addr_bram;


    // ==============================================================================
    // ARGMAX (Khối tìm chỉ số nơ-ron có giá trị lớn nhất ~ SOFTMAX)

    wire valid_out_argmax;
    //wire [3:0] predicted_digit;

    wire en_argmax;
    wire [1:0] read_cycle_argmax;



    controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .KERNEL_WIDTH(KERNEL_WIDTH),

        .FEATURE_IN_WIDTH(FEATURE_IN_WIDTH),

        .INPUT_IMAGE_WIDTH(INPUT_IMAGE_WIDTH),

        .BUFFER_SIZE_BB(BUFFER_SIZE_BB),

        .BUFFER_SIZE_WB(BUFFER_SIZE_WB),

        .BRAM_SIZE(BRAM_SIZE)

    ) controller_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),

        .done(done),
        .curr_layer(curr_layer),
        .bram_in_sel(bram_in_sel),
        .bram_sel(bram_sel),
        .mac_limit(mac_limit),
        .quad_pe_channels_cnt(quad_pe_channels_cnt),

        .en_quad_pe(en_quad_pe),
        .first_channel_quad_pe(first_channel_quad_pe),

        .clear_mp(clear_mp),
        .buffer_limit_mp(buffer_limit_mp),
        .clear_bram(clear_bram),
        .bram_limit(bram_limit),

        .valid_out_lb(valid_out_lb),
        .full_lb(full_lb),
        .wr_en_lb(wr_en_lb),
        .rd_en_lb(rd_en_lb),
        .clear_lb(clear_lb),
        .buffer_limit_lb(buffer_limit_lb),
        .output_index(output_index),

        .full_bb(full_bb),
        .wr_en_bb(wr_en_bb),
        .rd_en_bb(rd_en_bb),
        .clear_bb(clear_bb),
        .rd_addr_bb(rd_addr_bb),
        .buffer_limit_bb(buffer_limit_bb),

        .full_wb(full_wb),
        .wr_en_wb(wr_en_wb),
        .rd_en_wb(rd_en_wb),
        .clear_wb(clear_wb),
        .rd_addr_wb(rd_addr_wb),
        .buffer_limit_wb(buffer_limit_wb),

        .full_bram_dense(full_bram_dense),
        .clear_bram_dense(clear_bram_dense),
        .bram_dense_limit(bram_dense_limit),

        .full_bram(full_bram),
        .rd_en_bram(rd_en_bram),
        .rd_addr_bram(rd_addr_bram),

        .valid_out_argmax(valid_out_argmax),
        .en_argmax(en_argmax),
        .read_cycle_argmax(read_cycle_argmax)
    );

    datapath #(
        .DATA_WIDTH(DATA_WIDTH),
        .KERNEL_WIDTH(KERNEL_WIDTH),
        .NUM_PE(NUM_PE),

        .FRACTIONAL_BITS(FRACTIONAL_BITS),

        .FEATURE_IN_WIDTH(FEATURE_IN_WIDTH),

        .INPUT_IMAGE_WIDTH(INPUT_IMAGE_WIDTH),

        .NUM_FIFO_WB(NUM_FIFO_WB),
        .NUM_PE_WB(NUM_PE_WB),
        .BUFFER_SIZE_WB(BUFFER_SIZE_WB),

        .NUM_FIFO_BB(NUM_FIFO_BB),
        .NUM_PE_BB(NUM_PE_BB),
        .BUFFER_SIZE_BB(BUFFER_SIZE_BB),

        .BRAM_SIZE(BRAM_SIZE)

    ) datapath_inst (
        .clk(clk),
        .rst_n(rst_n),
        .curr_layer(curr_layer),
        .bram_in_sel(bram_in_sel),
        .bram_sel(bram_sel),
        .mac_limit(mac_limit),
        .quad_pe_channels_cnt(quad_pe_channels_cnt),

        .pixel_in(pixel_in),
        .weight_in(weight_in),
        .bias_in(bias_in),

        .predicted_digit(predicted_digit),

        .en_quad_pe(en_quad_pe),
        .first_channel_quad_pe(first_channel_quad_pe),

        .clear_mp(clear_mp),
        .buffer_limit_mp(buffer_limit_mp),
        .clear_bram(clear_bram),
        .bram_limit(bram_limit),

        .wr_en_lb(wr_en_lb),
        .rd_en_lb(rd_en_lb),
        .clear_lb(clear_lb),
        .output_index(output_index),
        .buffer_limit_lb(buffer_limit_lb),
        .valid_out_lb(valid_out_lb),
        .full_lb(full_lb),
        .wr_en_bb(wr_en_bb),
        .rd_en_bb(rd_en_bb),
        .clear_bb(clear_bb),
        .rd_addr_bb(rd_addr_bb),
        .buffer_limit_bb(buffer_limit_bb),
        .full_bb(full_bb),

        .wr_en_wb(wr_en_wb),
        .rd_en_wb(rd_en_wb),
        .clear_wb(clear_wb),
        .rd_addr_wb(rd_addr_wb),
        .buffer_limit_wb(buffer_limit_wb),
        .full_wb(full_wb),

        .clear_bram_dense(clear_bram_dense),
        .bram_dense_limit(bram_dense_limit),
        .full_bram_dense(full_bram_dense),

        .rd_en_bram(rd_en_bram),
        .rd_addr_bram(rd_addr_bram),
        .full_bram(full_bram),

        .en_argmax(en_argmax),
        .read_cycle_argmax(read_cycle_argmax),
        .valid_out_argmax(valid_out_argmax)
        //.predicted_digit(predicted_digit)
    );

endmodule