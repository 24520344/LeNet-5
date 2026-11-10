`timescale 1ns / 1ps

//====================================================================================
// Organization:    University of Information Technology - VNU-HCM
// Author:          Dung Nguyen Viet
// Student ID:      24520344
//
// File Name:       line_buffer.v
// Module Name:     line_buffer
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

module line_buffer #(
    parameter DATA_WIDTH = 16,
    parameter INPUT_IMAGE_WIDTH = 32,
    parameter KERNEL_WIDTH = 5,
    parameter NUM_PE = 4
) (
    input wire clk,
    input wire rst_n,
    input wire wr_en,
    input wire rd_en,
    input wire clear,
    input wire [$clog2(INPUT_IMAGE_WIDTH):0] buffer_limit,
    input wire [KERNEL_WIDTH - 1:0] output_index,
    input wire signed [NUM_PE * DATA_WIDTH - 1:0] pixel_in,
    
    output reg valid_out,
    output reg full,
    output reg signed [NUM_PE * DATA_WIDTH - 1:0] pixel_out
);

    // Khai báo 5 hàng fifo (buffer) để lưu đủ số phần tử cho kernel 5x5
    (* ram_style = "register" *) reg signed [NUM_PE * DATA_WIDTH - 1:0] mem0 [0:INPUT_IMAGE_WIDTH - 1];
    (* ram_style = "register" *) reg signed [NUM_PE * DATA_WIDTH - 1:0] mem1 [0:INPUT_IMAGE_WIDTH - 1];
    (* ram_style = "register" *) reg signed [NUM_PE * DATA_WIDTH - 1:0] mem2 [0:INPUT_IMAGE_WIDTH - 1];
    (* ram_style = "register" *) reg signed [NUM_PE * DATA_WIDTH - 1:0] mem3 [0:INPUT_IMAGE_WIDTH - 1];
    (* ram_style = "register" *) reg signed [NUM_PE * DATA_WIDTH - 1:0] mem4 [0:INPUT_IMAGE_WIDTH - 1];
    


    // =============================================================================
    // Khối xử lý tín hiệu trạng thái nội bộ
    // =============================================================================

    reg [$clog2(INPUT_IMAGE_WIDTH) - 1:0] col_addr;
    reg [$clog2(INPUT_IMAGE_WIDTH) - 1:0] row_count;

    reg [$clog2(INPUT_IMAGE_WIDTH) - 1:0] size_count;

    always @ (posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            col_addr <= 0;
            row_count <= 0;
            full <= 1'b0;
        end

        else if (clear) begin
            col_addr <= 0;
            row_count <= 0;
            full <= 1'b0;
        end

        else if (wr_en) begin

            // Tăng dần địa chỉ cột cho đến khi đạt giá trị cuối hàng thì quay về đầu hàng
            if (col_addr == (buffer_limit - 1)) begin
                col_addr <= 0;

                if (row_count == (buffer_limit - 1)) begin
                    row_count <= 0;
                    full <= 1'b1; // Đánh dấu đã nạp đủ dữ liệu cho 1 layer
                end

                else
                    row_count <= row_count + 5'd1;
            end

            else
                col_addr <= col_addr + 5'd1;

        end
    end

    always @ (*) begin    
        // Chỉ cho phép valid_out = 1 ngay khi col_addr vừa đên hàng thứ 5
        if (wr_en && row_count >= 4 && col_addr >= 4)
            // Tại col_addr = 4, pixel_in đã nạp đủ 5 pixel ở hàng thứ 5 -> valid_out = 1
            valid_out = 1'b1; 

            // row_count < 5 hoặc col_addr < 4 thì window đang nạp từ RAM nên chưa được xuất dữ liệu
        else if ((wr_en && (row_count < 4 || col_addr < 4)) || clear)
            valid_out = 1'b0;
    end

    
    // =============================================================================
    // Khối xử lý logic ghi vào buffer
    // =============================================================================
    
    always @(posedge clk) begin
        if (wr_en) begin
            mem0[col_addr] <= pixel_in;
            mem1[col_addr] <= mem0[col_addr];
            mem2[col_addr] <= mem1[col_addr];
            mem3[col_addr] <= mem2[col_addr];
            mem4[col_addr] <= mem3[col_addr];
        end
    end



    // ==============================================================================
    // Khối đọc chọn pixel output để nạp vào pe
    // ==============================================================================

    // Logic tính địa chỉ pixel output dựa theo col_addr (do col_addr được cập nhật tuần tự nên khi ánh xạ ra địa chỉ đọc cần phải tính toán lại như dưới)
    wire [$clog2(INPUT_IMAGE_WIDTH) - 1:0] idx_1 = (col_addr >= 5'd1) ? (col_addr - 5'd1) : (col_addr + buffer_limit[4:0] - 5'd1);
    wire [$clog2(INPUT_IMAGE_WIDTH) - 1:0] idx_2 = (col_addr >= 5'd2) ? (col_addr - 5'd2) : (col_addr + buffer_limit[4:0] - 5'd2);
    wire [$clog2(INPUT_IMAGE_WIDTH) - 1:0] idx_3 = (col_addr >= 5'd3) ? (col_addr - 5'd3) : (col_addr + buffer_limit[4:0] - 5'd3);
    wire [$clog2(INPUT_IMAGE_WIDTH) - 1:0] idx_4 = (col_addr >= 5'd4) ? (col_addr - 5'd4) : (col_addr + buffer_limit[4:0] - 5'd4);
    wire [$clog2(INPUT_IMAGE_WIDTH) - 1:0] idx_5 = (col_addr >= 5'd5) ? (col_addr - 5'd5) : (col_addr + buffer_limit[4:0] - 5'd5);

    always @ (posedge clk or negedge rst_n) begin

        if (~rst_n)
            pixel_out <= 0;

        else if (rd_en) begin

            pixel_out <= 0;

            case (output_index)
                // Địa chỉ đọc luôn lùi lại 1 bước so với địa chỉ ghi do tại chu kỳ này col_addr đã + 1 làm lệch nhịp với con trỏ ghi

                // Hàng 1
                5'd0: pixel_out <= mem4[idx_5];
                5'd1: pixel_out <= mem4[idx_4];
                5'd2: pixel_out <= mem4[idx_3];
                5'd3: pixel_out <= mem4[idx_2];
                5'd4: pixel_out <= mem4[idx_1];

                // Hàng 2
                5'd5: pixel_out <= mem3[idx_5];
                5'd6: pixel_out <= mem3[idx_4];
                5'd7: pixel_out <= mem3[idx_3];
                5'd8: pixel_out <= mem3[idx_2];
                5'd9: pixel_out <= mem3[idx_1];

                // Hàng 3
                5'd10: pixel_out <= mem2[idx_5];
                5'd11: pixel_out <= mem2[idx_4];
                5'd12: pixel_out <= mem2[idx_3];
                5'd13: pixel_out <= mem2[idx_2];
                5'd14: pixel_out <= mem2[idx_1];

                // Hàng 4
                5'd15: pixel_out <= mem1[idx_5];
                5'd16: pixel_out <= mem1[idx_4];
                5'd17: pixel_out <= mem1[idx_3];
                5'd18: pixel_out <= mem1[idx_2];
                5'd19: pixel_out <= mem1[idx_1];

                // Hàng 5
                5'd20: pixel_out <= mem0[idx_5];
                5'd21: pixel_out <= mem0[idx_4];
                5'd22: pixel_out <= mem0[idx_3];
                5'd23: pixel_out <= mem0[idx_2];
                5'd24: pixel_out <= mem0[idx_1];
            endcase

        end

    end

endmodule