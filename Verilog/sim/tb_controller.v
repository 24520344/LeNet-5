`timescale 1ns / 1ps

module tb_controller;

    localparam DATA_WIDTH = 16;
    localparam KERNEL_WIDTH = 5;
    localparam FEATURE_IN_WIDTH = 28;
    localparam INPUT_IMAGE_WIDTH = 32;
    localparam BUFFER_SIZE_BB = 16;
    localparam BUFFER_SIZE_WB = 4800;
    localparam BRAM_SIZE = 3136;

    // Inputs (Từ Testbench cấp vào Controller)
    reg clk;
    reg rst_n;
    reg start;
    reg full_bram;
    reg valid_out_lb;
    reg full_bb;
    reg full_wb;
    reg full_bram_dense;

    // Outputs (Từ Controller xuất ra)
    wire done;
    wire [1:0] curr_layer;
    wire bram_in_sel;
    wire [8:0] mac_limit;
    wire [1:0] quad_pe_channels_cnt;

    // Tín hiệu QUAD PE
    wire en_quad_pe;
    wire first_channel_quad_pe;

    // Tín hiệu MAX POOLING
    wire clear_mp;
    wire [$clog2(FEATURE_IN_WIDTH)-1:0] buffer_limit_mp;

    // Tín hiệu BRAM
    wire clear_bram;
    wire [$clog2(BRAM_SIZE)-1:0] bram_limit;
    wire rd_en_bram;
    wire [$clog2(BRAM_SIZE)-1:0] rd_addr_bram;

    // Tín hiệu LINE BUFFER
    wire wr_en_lb;
    wire rd_en_lb;
    wire clear_lb;
    wire [$clog2(INPUT_IMAGE_WIDTH):0] buffer_limit_lb;
    wire [KERNEL_WIDTH-1:0] output_index;

    // Tín hiệu BIAS BUFFER
    wire wr_en_bb;
    wire rd_en_bb;
    wire clear_bb;
    wire [$clog2(BUFFER_SIZE_BB)-1:0] rd_addr_bb;
    wire [$clog2(BUFFER_SIZE_BB):0] buffer_limit_bb;

    // Tín hiệu WEIGHT BUFFER
    wire wr_en_wb;
    wire rd_en_wb;
    wire clear_wb;
    wire [$clog2(BUFFER_SIZE_WB)-1:0] rd_addr_wb;
    wire [$clog2(BUFFER_SIZE_WB):0] buffer_limit_wb;

    // Tín hiệu BRAM DENSE
    wire clear_bram_dense;
    wire [$clog2(BRAM_SIZE)-1:0] bram_dense_limit;

    // =========================================================================
    // 3. KHỞI TẠO DUT (Device Under Test)
    // =========================================================================
    controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .KERNEL_WIDTH(KERNEL_WIDTH),
        .FEATURE_IN_WIDTH(FEATURE_IN_WIDTH),
        .INPUT_IMAGE_WIDTH(INPUT_IMAGE_WIDTH),
        .BUFFER_SIZE_BB(BUFFER_SIZE_BB),
        .BUFFER_SIZE_WB(BUFFER_SIZE_WB),
        .BRAM_SIZE(BRAM_SIZE)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .curr_layer(curr_layer),
        .bram_in_sel(bram_in_sel),
        .mac_limit(mac_limit),
        .quad_pe_channels_cnt(quad_pe_channels_cnt),
        .en_quad_pe(en_quad_pe),
        .first_channel_quad_pe(first_channel_quad_pe),
        .clear_mp(clear_mp),
        .buffer_limit_mp(buffer_limit_mp),
        .full_bram(full_bram),
        .clear_bram(clear_bram),
        .bram_limit(bram_limit),
        .valid_out_lb(valid_out_lb),
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
        .rd_en_bram(rd_en_bram),
        .rd_addr_bram(rd_addr_bram)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk; // Chu kỳ 10ns (100MHz)
    end

    initial begin
        #3000 $stop;
    end

    initial begin
        // Khởi tạo tất cả input = 0
        rst_n = 0;
        start = 0;
        full_bram = 0;
        valid_out_lb = 0;
        full_bb = 0;
        full_wb = 0;
        full_bram_dense = 0;

        // Xả reset
        #10 rst_n = 1;
        
        // Bắn xung start để đánh thức FSM
        #10 start = 1;
        #10 start = 0;

    end

endmodule