`timescale 1ns / 1ps

//====================================================================================
// Organization:    University of Information Technology - VNU-HCM
// Author:          Dung Nguyen Viet
// Student ID:      24520344
//
// File Name:       controller.v
// Module Name:     controller
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

module controller #(
    // Tham sÃ¡Â»â€˜ chung
    parameter DATA_WIDTH = 16,
    parameter KERNEL_WIDTH = 5,

    // Tham sÃ¡Â»â€˜ riÃƒÂªng Max Pooling
    parameter FEATURE_IN_WIDTH = 28,

    // Tham sÃ¡Â»â€˜ riÃƒÂªng Line Buffer
    parameter INPUT_IMAGE_WIDTH = 32,
    
    // Tham sÃ¡Â»â€˜ riÃƒÂªng Bias Buffer
    parameter BUFFER_SIZE_BB = 16,

    // Tham sÃ¡Â»â€˜ riÃƒÂªng Weight Buffer
    parameter BUFFER_SIZE_WB = 4800,

    // Tham sÃ¡Â»â€˜ riÃƒÂªng BRAM
    parameter BRAM_SIZE = 3136
) (

    input wire clk,
    input wire rst_n,
    input wire start,

    output wire done,
    output reg [1:0] curr_layer,

    output reg bram_in_sel,

    output reg bram_sel,

    // GiÃ¡Â»â€ºi hÃ¡ÂºÂ¡n sÃ¡Â»â€˜ lÃ¡ÂºÂ§n chÃ¡ÂºÂ¡y quad pe cho mÃ¡Â»â€”i layer
    output reg [8:0] mac_limit,

    output reg [1:0] quad_pe_channels_cnt,


    // =============================================================================
    // PE / QUAD PE (khÃ¡Â»â€˜i tÃƒÂ­nh toÃƒÂ¡n tÃƒÂ­ch chÃ¡ÂºÂ­p)

    output reg en_quad_pe,
    output reg first_channel_quad_pe,


    // =============================================================================
    // MAX POOLING (khÃ¡Â»â€˜i nhÃƒÂ³m vÃƒÂ  xuÃ¡ÂºÂ¥t pixel lÃ¡Â»â€ºn nhÃ¡ÂºÂ¥t)

    output reg clear_mp,
    output reg [$clog2(FEATURE_IN_WIDTH) - 1:0] buffer_limit_mp,

    // TÃƒÂ­n hiÃ¡Â»â€¡u Ã„â€˜iÃ¡Â»Âu khiÃ¡Â»Æ’n BRAM tÃ¡Â»Â« controller
    input wire full_bram,

    output reg clear_bram,
    output reg [$clog2(BRAM_SIZE) - 1:0] bram_limit,
    

    // =============================================================================
    // LINE BUFFER (RAM lÃ†Â°u Ã¡ÂºÂ£nh Ã„â€˜Ã¡ÂºÂ§u vÃƒÂ o)

    input wire valid_out_lb,
    input wire full_lb,

    output wire wr_en_lb,
    output wire rd_en_lb,
    output reg clear_lb,
    output reg [$clog2(INPUT_IMAGE_WIDTH):0] buffer_limit_lb,
    output wire [KERNEL_WIDTH - 1:0] output_index,


    // =============================================================================
    // BIAS BUFFER (RAM lÃ†Â°u bias)

    input wire full_bb,

    output wire wr_en_bb,
    output wire rd_en_bb,
    output reg clear_bb,
    output wire [$clog2(BUFFER_SIZE_BB) - 1:0] rd_addr_bb,
    output reg [$clog2(BUFFER_SIZE_BB):0] buffer_limit_bb,


    // =============================================================================
    // WEIGHT BUFFER (RAM lÃ†Â°u weight)

    input wire full_wb,

    output wire wr_en_wb,
    output wire rd_en_wb,
    output reg clear_wb,
    output wire [$clog2(BUFFER_SIZE_WB) - 1:0] rd_addr_wb,
    output reg [$clog2(BUFFER_SIZE_WB):0] buffer_limit_wb,


    // =============================================================================
    // BRAM DENSE HANDLER

    input wire full_bram_dense,

    output reg clear_bram_dense,
    output reg [$clog2(BRAM_SIZE) - 1:0] bram_dense_limit,

    
    // =============================================================================
    // BRAM (RAM trung gian lÃ†Â°u feature maps)

    output wire rd_en_bram,
    output wire [$clog2(BRAM_SIZE) - 1:0] rd_addr_bram,


    // =============================================================================
    // ARGMAX
    input wire valid_out_argmax,

    output reg en_argmax,
    output reg [1:0] read_cycle_argmax

);

    // =============================================================================
    // Khai bÃƒÂ¡o cÃƒÂ¡c biÃ¡ÂºÂ¿n layer vÃƒÂ  giÃ¡Â»â€ºi hÃ¡ÂºÂ¡n cho cÃƒÂ¡c buffer
    // =============================================================================

    localparam
        L0_conv1_pool1  = 0,
        L1_conv2_pool2  = 1,
        L2_dense1       = 2,
        L3_dense2       = 3;

    // reg [8:0] mac_limit; Ã„â€˜Ã†Â°Ã¡Â»Â£c khai bÃƒÂ¡o Ã¡Â»Å¸ trÃƒÂªn Ã„â€˜Ã¡Â»Æ’ truyÃ¡Â»Ân vÃƒÂ o datapath luÃƒÂ´n

    // GiÃ¡Â»â€ºi hÃ¡ÂºÂ¡n sÃ¡Â»â€˜ lÃ¡ÂºÂ§n dÃ¡Â»â€¹ch cÃ¡Â»Â­a sÃ¡Â»â€¢ (chÃ¡Â»â€° ÃƒÂ¡p dÃ¡Â»Â¥ng cho layer Conv)
    reg [9:0] slides_limit;

    // channel_cnt sÃ¡Â»Â­ dÃ¡Â»Â¥ng buffer_limit_bb lÃƒÂ m giÃ¡Â»â€ºi hÃ¡ÂºÂ¡n luÃƒÂ´n vÃƒÂ¬ bias cÃ…Â©ng chÃ¡Â»â€° sÃ¡Â»Â­ dÃ¡Â»Â¥ng tÃ†Â°Ã†Â¡ng Ã¡Â»Â©ng vÃ¡Â»â€ºi sÃ¡Â»â€˜ channel

    reg [2:0] quad_pe_channels_limit;



    // =============================================================================
    // Khai bÃƒÂ¡o cÃƒÂ¡c biÃ¡ÂºÂ¿n Ã„â€˜Ã¡ÂºÂ¡i diÃ¡Â»â€¡n cho tÃ¡Â»Â«ng state FSM
    // =============================================================================

    localparam
        S0_IDLE                 = 0,
        S1_LOAD_BIAS            = 1,
        S2_LOAD_WEIGHT          = 2,
        S3_LOAD_LINE_BUFFER     = 3,
        S4_CONV_POOL            = 4,
        S5_DENSE                = 5,
        S6_NEXT_LAYER           = 6,
        S7_ARGMAX               = 7,
        S8_DONE                 = 8;

    reg [3:0] curr_state, next_state;



    // =============================================================================
    // Khai bÃƒÂ¡o cÃƒÂ¡c biÃ¡ÂºÂ¿n Ã„â€˜Ã¡ÂºÂ¿m sÃ¡Â»â€˜ lÃ¡ÂºÂ§n chÃ¡ÂºÂ¡y MAC trong tÃ¡Â»Â«ng state FSM
    // =============================================================================

    // BiÃ¡ÂºÂ¿n Ã„â€˜Ã¡ÂºÂ¿m sÃ¡Â»â€˜ lÃ¡ÂºÂ§n chÃ¡ÂºÂ¡y MAC cÃ¡Â»Â§a QUAD PE cho cÃ¡ÂºÂ£ conv layer vÃƒÂ  dense layer
    reg [8:0] mac_cnt;

    // BiÃ¡ÂºÂ¿n Ã„â€˜Ã¡ÂºÂ¿m sÃ¡Â»â€˜ lÃ¡ÂºÂ§n trÃ†Â°Ã¡Â»Â£t cÃ¡Â»Â­a sÃ¡Â»â€¢ tuÃ¡Â»Â³ thuÃ¡Â»â„¢c vÃƒÂ o tÃ¡Â»Â«ng layer
    reg [9:0] slides_cnt;

    // BiÃ¡ÂºÂ¿n Ã„â€˜Ã¡ÂºÂ¿m sÃ¡Â»â€˜ kÃƒÂªnh cÃ¡Â»Â§a Ã¡ÂºÂ£nh input tuÃ¡Â»Â³ thuÃ¡Â»â„¢c vÃƒÂ o tÃ¡Â»Â«ng layer (sÃ¡Â»Â­ dÃ¡Â»Â¥ng buffer_limit_bb luÃƒÂ´n vÃƒÂ¬ bias cÃ…Â©ng chÃ¡Â»â€° xÃ¡Â»Â­ lÃƒÂ½ theo mÃ¡Â»â€”i kÃƒÂªnh 1 lÃ¡ÂºÂ§n)
    reg [3:0] filters_cnt;

    // reg [1:0] quad_pe_channels_cnt;

    // BiÃ¡ÂºÂ¿n cho biÃ¡ÂºÂ¿t Ã„â€˜Ã¡Â»Â§ Ã„â€˜iÃ¡Â»Âu kiÃ¡Â»â€¡n Ã„â€˜Ã¡Â»Æ’ dÃ¡Â»â€¹ch cÃ¡Â»Â­a sÃ¡Â»â€¢ (chÃ¡ÂºÂ¡y xong 1 cÃ¡Â»Â­a sÃ¡Â»â€¢ vÃƒÂ  chÃ¡ÂºÂ¡y hÃ¡ÂºÂ¿t cÃƒÂ¡c kÃƒÂªnh)




    reg valid_out_lb_delay;
    wire curr_valid_out_lb = (curr_layer == 0) ? valid_out_lb_delay : valid_out_lb;


    // =============================================================================
    // KhÃ¡Â»â€˜i FSM
    // =============================================================================

    assign done = (curr_state == S8_DONE);

    always @ (posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            curr_state <= S0_IDLE;
        end

        else begin
            curr_state <= next_state;
        end
    end

    always @ (*) begin
        next_state = curr_state;

        case (curr_state)

            S0_IDLE: 
                next_state = (start) ? S1_LOAD_BIAS : S0_IDLE;

            S1_LOAD_BIAS: 
                //next_state = (full_bb) ? S2_LOAD_WEIGHT : S1_LOAD_BIAS;
                next_state = S2_LOAD_WEIGHT;

            S2_LOAD_WEIGHT: 
                //if (full_wb) begin
                    if (curr_layer == L2_dense1 || curr_layer == L3_dense2)
                        next_state = S5_DENSE;
                    else
                        next_state = S3_LOAD_LINE_BUFFER;
                //end

            S3_LOAD_LINE_BUFFER: 
                next_state = (valid_out_lb) ? S4_CONV_POOL : S3_LOAD_LINE_BUFFER;

            S4_CONV_POOL:

            S5_DENSE: 

            S6_NEXT_LAYER:
                if (curr_layer == L3_dense2)
                    next_state = S7_ARGMAX;
                else if (full_bram || full_bram_dense)
                    next_state = S0_IDLE;

            S7_ARGMAX:
                next_state = (valid_out_argmax) ? S8_DONE : S7_ARGMAX;

            S8_DONE:
                next_state = S8_DONE;
        endcase
    end



    // =============================================================================
    // =============================================================================
    always @ (*) begin

        buffer_limit_lb = 0;
        buffer_limit_bb = 0;
        buffer_limit_wb = 0;
        buffer_limit_mp = 0;
        bram_limit = 0;
		bram_dense_limit = 0;
        mac_limit = 0;
        slides_limit = 0;
        quad_pe_channels_limit = 0;
        bram_sel = 0;

        case (curr_layer)

            L0_conv1_pool1: begin

            end

            L1_conv2_pool2: begin

            end

            L2_dense1: begin

            end

            L3_dense2: begin

            end

        endcase

    end



    // =============================================================================
    // KhÃ¡Â»â€˜i xÃ¡Â»Â­ lÃƒÂ½ cÃƒÂ¡c biÃ¡ÂºÂ¿n Ã„â€˜Ã¡ÂºÂ¿m cho cÃƒÂ¡c curr_state
    // =============================================================================

    always @(posedge clk or negedge rst_n) begin
        
        if (~rst_n) begin
            mac_cnt <= 0;
            filters_cnt <= 0;
            slides_cnt <= 0;
            curr_layer <= 0;
            quad_pe_channels_cnt <= 0;
        end
        
        else begin

            

        end

    end

    always @ (*) begin
        if (curr_layer >= L2_dense1)
            bram_in_sel = 1'b1;

        else
            bram_in_sel = 1'b0;
    end


    // =============================================================================
    // PE/ QUAD PE
    // =============================================================================

    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            en_quad_pe <= 1'b0;
            first_channel_quad_pe <= 1'b0;
        end
            
        else begin

        end
    end



    // =============================================================================
    // MAX POOLING
    // =============================================================================

    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            clear_mp <= 1'b0;
            clear_bram <= 1'b0;
        end

        else begin

        end
    end


    // =============================================================================
    // LINE BUFFER
    // =============================================================================

    // wr_en_lb must be delayed 1 cycle for BRAM fetching data latency (layer 1 - 3)
    reg raw_wr_en_lb;

    always @(posedge clk or negedge rst_n) begin

        if (~rst_n) begin
            raw_wr_en_lb <= 1'b0;
            clear_lb <= 1'b0;
            valid_out_lb_delay <= 1'b0;
        end
        
        else begin

        end

    end

    // wr_en_lb use combinational logic for layer 0 (reading input image) only
    assign wr_en_lb = 

    assign rd_en_lb =

    // - Conv1: Every mac_cnt is a new spatial pixel (0 to 24)
    // - Conv2: Every 4 mac_cnts is a new spatial pixel (mac_cnt / 4)
    assign output_index = 



    // =============================================================================
    // BIAS BUFFER
    // =============================================================================

    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            clear_bb <= 1'b0;
        end

        else begin
            clear_bb <= 
        end
    end

    assign wr_en_bb = /*(curr_state == S1_LOAD_BIAS) &&*/ !full_bb;
    assign rd_en_bb = 
    assign rd_addr_bb =



    // =============================================================================
    // WEIGHT BUFFER
    // =============================================================================

    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            clear_wb <= 1'b0;
        end
        
        else
            clear_wb <= 
    end

    assign wr_en_wb = /*(curr_state == S2_LOAD_WEIGHT) &&*/ !full_wb;
    assign rd_en_wb = 
    assign rd_addr_wb = 



    // =============================================================================
    // BRAM
    // =============================================================================

    reg [$clog2(BRAM_SIZE) - 1:0] bram_ptr;

    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            //rd_en_bram <= 1'b0;
            bram_ptr <= 0;
        end

        else begin

        end
    end

    assign rd_en_bram = 

    // - Conv Layers: Use the sliding pointer (bram_ptr)
    // - Dense Layers: Advance row only after extracting all 4 pixels from it (mac_cnt / 4)
    assign rd_addr_bram = 



    // =============================================================================
    // ARGMAX
    // =============================================================================

    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            read_cycle_argmax <= 2'd0;
            en_argmax <= 1'b0;
        end 

        else begin
            
        end 
    end
    
endmodule