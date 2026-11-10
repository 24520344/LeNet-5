`timescale 1ns / 1ps

//====================================================================================
// Organization:    University of Information Technology - VNU-HCM
// Author:          Dung Nguyen Viet
// Student ID:      24520344
//
// File Name:       datapath.v
// Module Name:     datapath
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

module datapath #(
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
    input wire [1:0] curr_layer,
    input wire bram_in_sel,

    input wire bram_sel,
    
    // Giới hạn số lần chạy quad pe cho mỗi layer
    input wire [8:0] mac_limit,

    input wire [1:0] quad_pe_channels_cnt,

    input wire signed [DATA_WIDTH - 1:0] pixel_in,
    input wire signed [DATA_WIDTH * NUM_FIFO_WB - 1:0] weight_in,
    input wire signed [DATA_WIDTH * NUM_FIFO_BB - 1:0] bias_in,

    output wire [3:0] predicted_digit,


    // =============================================================================
    // PE / QUAD PE (khối tính toán tích chập)

    input wire en_quad_pe,
    input wire first_channel_quad_pe,


    // =============================================================================
    // MAX POOLING (khối nhóm và xuất pixel lớn nhất)

    input wire clear_mp,
    input wire [$clog2(FEATURE_IN_WIDTH) - 1:0] buffer_limit_mp,

    // Tín hiệu điều khiển BRAM từ controller
    input wire clear_bram,
    input wire [$clog2(BRAM_SIZE) - 1:0] bram_limit,

    output wire full_bram,


    // =============================================================================
    // LINE BUFFER (RAM lưu ảnh đầu vào)

    input wire wr_en_lb,
    input wire rd_en_lb,
    input wire clear_lb,
    input wire [KERNEL_WIDTH - 1:0] output_index,
    input wire [$clog2(INPUT_IMAGE_WIDTH):0] buffer_limit_lb,

    output wire valid_out_lb,
    output wire full_lb,

    // =============================================================================
    // BIAS BUFFER (RAM lưu bias)

    input wire wr_en_bb,
    input wire rd_en_bb,
    input wire clear_bb,
    input wire [$clog2(BUFFER_SIZE_BB) - 1:0] rd_addr_bb,
    input wire [$clog2(BUFFER_SIZE_BB):0] buffer_limit_bb,

    output wire full_bb,


    // =============================================================================
    // WEIGHT BUFFER (RAM lưu weight)

    input wire wr_en_wb,
    input wire rd_en_wb,
    input wire clear_wb,
    input wire [$clog2(BUFFER_SIZE_WB) - 1:0] rd_addr_wb,
    input wire [$clog2(BUFFER_SIZE_WB):0] buffer_limit_wb,

    output wire full_wb,

    
    // =============================================================================
    // BRAM DENSE HANDLER (Khối xử lý tín hiệu điều khiển với lớp dense)

    input wire clear_bram_dense,
    input wire [$clog2(BRAM_SIZE) - 1:0] bram_dense_limit,

    output wire full_bram_dense,


    // =============================================================================
    // BRAM (RAM trung gian lưu feature maps)

    input wire rd_en_bram,
    input wire [$clog2(BRAM_SIZE) - 1:0] rd_addr_bram,


    // =============================================================================
    // ARGMAX (Khối tìm chỉ số 1 trong 10 neuron có giá trị lớn nhất ~ SOFTMAX)

    input wire en_argmax,
    input wire [1:0] read_cycle_argmax,

    output wire valid_out_argmax
    //output wire [3:0] predicted_digit
    
);

    wire valid_out_bb;
    wire signed [DATA_WIDTH * NUM_PE_BB - 1:0] bias_out_bb;

    wire valid_out_wb;
    wire signed [DATA_WIDTH * NUM_PE_WB - 1:0] weight_out_wb;

    wire signed [NUM_PE * DATA_WIDTH - 1:0] pixel_out_lb;

    wire valid_out_quad_pe;
    wire signed [DATA_WIDTH * NUM_PE - 1:0] pixel_out_quad_pe;

    wire signed [DATA_WIDTH * NUM_PE - 1:0] pixel_out_quad_relu;

    wire valid_out_mp, wr_en_bram;
    wire [$clog2(BRAM_SIZE) - 1:0] wr_addr_bram;
    wire signed [DATA_WIDTH * NUM_PE - 1:0] pixel_out_mp;

    wire wr_en_bram_dense;
    wire [$clog2(BRAM_SIZE) - 1:0] wr_addr_bram_dense;

    wire valid_out_bram0, valid_out_bram1;
    wire signed [NUM_PE * DATA_WIDTH - 1:0] pixel_out_bram0, pixel_out_bram1;


    single_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_FIFO(NUM_FIFO_BB),
        .NUM_PE(NUM_PE_BB),
        .BUFFER_SIZE(BUFFER_SIZE_BB)
    ) bias_buffer (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en_bb),
        .rd_en(rd_en_bb),
        .clear(clear_bb),
        .rd_addr(rd_addr_bb),
        .buffer_limit(buffer_limit_bb),
        .buffer_in(bias_in),    // internal port

        .valid_out(valid_out_bb),    // internal port
        .full(full_bb),
        .buffer_out(bias_out_bb)    // internal port
    );


    single_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_FIFO(NUM_FIFO_WB),
        .NUM_PE(NUM_PE_WB),
        .BUFFER_SIZE(BUFFER_SIZE_WB)
    ) weight_buffer (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en_wb),
        .rd_en(rd_en_wb),
        .clear(clear_wb),
        .rd_addr(rd_addr_wb),
        .buffer_limit(buffer_limit_wb),
        .buffer_in(weight_in),    // internal port

        .valid_out(valid_out_wb),    // internal port
        .full(full_wb),
        .buffer_out(weight_out_wb)    // internal port
    );

    
    line_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .INPUT_IMAGE_WIDTH(INPUT_IMAGE_WIDTH),
        .KERNEL_WIDTH(KERNEL_WIDTH)
    ) line_buffer_inst (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en_lb),
        .rd_en(rd_en_lb),
        .clear(clear_lb),
        .buffer_limit(buffer_limit_lb),
        .output_index(output_index),
        .pixel_in(curr_layer == 0 ? {48'd0, pixel_in} : ~bram_sel ? pixel_out_bram1 : pixel_out_bram0),    // internal port

        .valid_out(valid_out_lb),
        .full(full_lb),
        .pixel_out(pixel_out_lb)    // internal port
    );


    quad_pe #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRACTIONAL_BITS(FRACTIONAL_BITS),
        .KERNEL_WIDTH(KERNEL_WIDTH),
        .NUM_PE(NUM_PE)
    ) quad_pe_inst (
        .clk(clk),
        .rst_n(rst_n),
        .en(en_quad_pe),
        .first_channel(first_channel_quad_pe),
        .mac_limit(mac_limit),

        .pixel_in(bram_in_sel ? (~bram_sel ? pixel_out_bram1[quad_pe_channels_cnt * DATA_WIDTH +: DATA_WIDTH] : pixel_out_bram0[quad_pe_channels_cnt * DATA_WIDTH +: DATA_WIDTH]) : pixel_out_lb[quad_pe_channels_cnt * DATA_WIDTH +: DATA_WIDTH]),    // internal port
        .weight_in(weight_out_wb),  // internal port
        .bias_in(bias_out_bb),      // internal port

        .valid_out(valid_out_quad_pe),  // internal port
        .pixel_out(pixel_out_quad_pe)   // internal port
    );


    quad_relu #(
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_PE(NUM_PE)
    ) quad_relu_inst (
        .data_in(pixel_out_quad_pe),    // internal port
        .data_out(pixel_out_quad_relu)  // internal port
    );


    quad_max_pooling_modified #(
        .DATA_WIDTH(DATA_WIDTH),
        .FEATURE_IN_WIDTH(FEATURE_IN_WIDTH),
        .BRAM_SIZE(BRAM_SIZE),
        .NUM_PE(NUM_PE)
    ) quad_max_pooling_inst (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(valid_out_quad_pe),  // internal port
        .clear(clear_mp),
        .buffer_limit(buffer_limit_mp),
        .pixel_in(pixel_out_quad_relu),   // internal port

        .clear_bram(clear_bram),
        .bram_limit(bram_limit),

        .valid_out(valid_out_mp),   // internal port
        .pixel_out(pixel_out_mp),    // internal port

        .wr_en_bram(wr_en_bram),    // internal port
        .w_addr_bram(wr_addr_bram), // internal port
        .full_bram(full_bram)
    );


    // bram_dense_handler #(
    //     .DATA_WIDTH(DATA_WIDTH),
    //     .BRAM_SIZE(BRAM_SIZE)
    // ) bram_dense_handler_inst (
    //     .clk(clk),
    //     .rst_n(rst_n),
    //     .clear_bram_dense(clear_bram_dense),
    //     .bram_dense_limit(bram_dense_limit),
    //     .bram_in_sel(bram_in_sel),
    //     .valid_out_quad_pe(valid_out_quad_pe),

    //     .w_addr_bram_dense(wr_addr_bram_dense), // internal port
    //     .full_bram_dense(full_bram_dense)
    // );


    bram #(
        .DATA_WIDTH(DATA_WIDTH),
        .SIZE(BRAM_SIZE),
        .NUM_PE(NUM_PE)
    ) bram_inst0 (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(~bram_sel ? ((bram_in_sel) ? valid_out_quad_pe : wr_en_bram) : 1'b0),     // internal port
        .rd_en(rd_en_bram && bram_sel),
        .w_addr(bram_in_sel ? wr_addr_bram_dense : wr_addr_bram),  // internal port
        .r_addr(rd_addr_bram),
        .data_in(bram_in_sel ? pixel_out_quad_relu : pixel_out_mp),

        .valid_out(valid_out_bram0),     // internal port
        .data_out(pixel_out_bram0)       // internal port
    );


    bram #(
        .DATA_WIDTH(DATA_WIDTH),
        .SIZE(BRAM_SIZE),
        .NUM_PE(NUM_PE)
    ) bram_inst1 (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(bram_sel ? ((bram_in_sel) ? valid_out_quad_pe : wr_en_bram) : 1'b0),     // internal port
        .rd_en(rd_en_bram && ~bram_sel),
        .w_addr(bram_in_sel ? wr_addr_bram_dense : wr_addr_bram),  // internal port
        .r_addr(rd_addr_bram),
        .data_in(bram_in_sel ? pixel_out_quad_relu : pixel_out_mp),

        .valid_out(valid_out_bram1),     // internal port
        .data_out(pixel_out_bram1)       // internal port
    );


    argmax #(
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_PE(NUM_PE)
    ) argmax_inst (
        .clk(clk),
        .rst_n(rst_n),
        .en(en_argmax),
        .read_cycle(read_cycle_argmax),
        
        // Nhận trực tiếp ngõ ra của BRAM
        .data_in(pixel_out_bram1), 

        .valid_out(valid_out_argmax),
        .predicted_digit(predicted_digit)
    );

endmodule