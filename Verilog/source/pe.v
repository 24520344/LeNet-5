`timescale 1ns / 1ps

//====================================================================================
// Organization:    University of Information Technology - VNU-HCM
// Author:          Dung Nguyen Viet
// Student ID:      24520344
//
// File Name:       pe.v
// Module Name:     pe
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

module pe #(
    parameter DATA_WIDTH = 16,
    parameter FRACTIONAL_BITS = 10,
    parameter KERNEL_WIDTH = 5 
) (
    input wire clk,
    input wire rst_n,
    input wire en,
    input wire first_channel,
    input wire cont_kernel,

    input wire signed [DATA_WIDTH-1:0] pixel_in,
    input wire signed [DATA_WIDTH-1:0] weight_in,
    input wire signed [DATA_WIDTH-1:0] bias_in,

    output reg valid_out,
    output reg signed [DATA_WIDTH-1:0] pixel_out
);

    //====================================================================================
    // Khối xử lý tín hiệu trạng thái và bộ đếm
    //====================================================================================
    reg [8:0] loop_amount;

    always @ (posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            loop_amount <= 0;
            valid_out <= 0;
        end

        // loop_amount là biến trạng thái đếm cho FSM
        // valid_out la biến trạng thái báo FSM đã chạy xong
        else if (en) begin
            if (loop_amount < mac_limit - 1) begin
                loop_amount <= loop_amount + 9'd1;
                valid_out <= 0;
            end

            else begin
                loop_amount <= 0;
                valid_out <= 1;
            end
        end

        else begin
            valid_out <= 0;
        end
    end



    //====================================================================================
    // Khối tính toán phép tích chập
    //====================================================================================
    wire signed [2*DATA_WIDTH-1:0] mult_result;

    // Tính phép nhân trong chu kỳ luôn thay vì pipeline phép nhân riêng và cộng dồn riêng (tránh đụng độ delay với trạng thái FSM)
    assign mult_result = pixel_in * weight_in;

    reg signed [2*DATA_WIDTH-1:0] acc;

    // Khối tính toán cộng dồn dựa theo trạng thái loop_amount FSM
    always @ (posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            acc <= 0;
            pixel_out <= 0;
        end

        else if (en) begin
            if (loop_amount == 0) begin
                // Nếu là kênh đầu tiên của bức ảnh thì khởi tạo cộng thêm luôn bias
                if (first_channel)
                    // Do tại lúc khởi tạo, acc có cộng thêm mult_result (cộng trước CALC 1 chu kỳ) vào nên chỉ cần lặp n - 1 lần 
                    acc <= {{6{bias_in[DATA_WIDTH-1]}}, bias_in, {FRACTIONAL_BITS{1'b0}}} + mult_result;

                // Các kênh còn lại thì cộng dồn mult_result bình thường
                else
                    acc <= acc + mult_result;
            end

            else if (loop_amount < mac_limit - 1 || )
                acc <= acc + mult_result;

            else begin
                // // Cập nhật acc để sử dụng cho kênh tiếp theo nếu có nhiều hơn 1 kênh
                if (cont_kernel)
                    acc <= acc + mult_result;

                else
                    // Cộng dồn lần cuối và dịch dấu phẩy khi tính xong tất cả các kênh
                    pixel_out <= (acc + mult_result) >>> FRACTIONAL_BITS;
            end
        end
    end

endmodule