`timescale 1ns / 1ps

//====================================================================================
// Organization:    University of Information Technology - VNU-HCM
// Author:          Dung Nguyen Viet
// Student ID:      24520344
//
// File Name:       max_pooling_modified.v
// Module Name:     max_pooling_modified
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

module max_pooling_modified #(
    parameter DATA_WIDTH = 16,
    parameter FEATURE_IN_WIDTH = 28,
    parameter BRAM_SIZE = 196,
    parameter NUM_PE = 4
)(
    input wire clk,
    input wire rst_n,
    input wire wr_en,
    input wire clear,
    input wire [$clog2(FEATURE_IN_WIDTH) - 1:0] buffer_limit,
    input wire signed [DATA_WIDTH - 1:0] pixel_in,

    // Tín hiệu điều khiển BRAM từ controller
    input wire clear_bram,
    input wire [$clog2(BRAM_SIZE) - 1:0] bram_limit, 

    output reg valid_out,
    output reg signed [DATA_WIDTH - 1:0] pixel_out,

    // Tín hiệu điều khiển BRAM nối tiếp trong datapath
    output wire wr_en_bram,
    output reg [$clog2(BRAM_SIZE) - 1:0] w_addr_bram,
    output reg full_bram
);

    // Line buffer
    (* ram_style = "register" *) reg signed [DATA_WIDTH - 1:0] mem [0:FEATURE_IN_WIDTH - 1];

    reg [$clog2(FEATURE_IN_WIDTH) - 1:0] col_addr;
    reg [$clog2(FEATURE_IN_WIDTH) - 1:0] row_addr;

    // ==============================================================================
    // Logic tìm giá trị lớn nhất trong 4 pixel
    // ==============================================================================

    // Chỉ cần khai báo 2 pixel của pool cho cột trái vì 2 pixel còn lại sẽ được lưu tạm từ trước (mem[col_addr]) và lấy trực tiếp từ pixel_in
    reg signed [DATA_WIDTH - 1:0] data_pool [0:1];
    wire signed [DATA_WIDTH - 1:0] max_01, max_23, max;

    assign max_01 = (data_pool[0] > data_pool[1]) ? data_pool[0] : data_pool[1];
    assign max_23 = (mem[col_addr] > pixel_in) ? mem[col_addr] : pixel_in;
    assign max = (max_01 > max_23) ? max_01 : max_23;


    // ==============================================================================
    // Khối nạp pixel vào max pool
    // ==============================================================================

    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin            
            col_addr <= 0;
            row_addr <= 0;
            data_pool[0] <= 16'd0;
            data_pool[1] <= 16'd0;
            pixel_out <= 16'd0;

            valid_out <= 0;
        end

        else if (clear) begin
            col_addr <= 0;
            row_addr <= 0;

            valid_out <= 0;
        end

        // Truyền dữ liệu tuần tự vào buffer khi dữ liệu ổn định
        else if (wr_en) begin

            // Khi đã nạp đủ 1 hàng
            if (col_addr == buffer_limit - 1) begin

                col_addr <= 0; // Reset con trỏ địa chỉ về 0

                if (row_addr == buffer_limit - 1)
                    row_addr <= 0; // Reset con trỏ địa chỉ về 0

                else
                    row_addr <= row_addr + 5'd1;

            end

            else
                col_addr <= col_addr + 5'd1;


            // Hàng chẵn: chỉ lưu vào buffer, chưa đủ 2D để tính max
            if (row_addr[0] == 1'b0) begin
                mem[col_addr] <= pixel_in;
                valid_out <= 1'b0;
            end

            // Hàng lẻ: bắt đầu gom ô 2x2 vì đã nạp đủ buffer hàng trên
            else begin

                // Cột chẵn: xử lý 2 pixel cột trái với giá đã lưu trong buffer trước đó (hàng trên) và pixel_in hiện tại (hàng dưới)
                if (col_addr[0] == 1'b0) begin
                    data_pool[0] <= mem[col_addr];
                    data_pool[1] <= pixel_in;
                    valid_out <= 1'b0;
                end 

                // Cột lẻ: 2 pixel bên phải còn lại chính là mem[col_addr] mới dịch và pixel_in
                else begin
                    // Xuất kết quả lớn nhất đã tìm được sau khi tính max
                    pixel_out <= max;
                    valid_out <= 1'b1;
                end

            end

        end

        else
            valid_out <= 0; // Reset cờ valid_out khi bắt đầu nạp dữ liệu mới

    end



    // ==============================================================================
    // Khối nạp dữ liệu vào BRAM
    // ==============================================================================

	assign wr_en_bram = valid_out;
	 
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            w_addr_bram <= 0;
            full_bram <= 0;
        end 

        // Reset bộ đếm khi FSM ra lệnh chuyển Layer
        else if (clear_bram) begin
            w_addr_bram <= 0;
            full_bram <= 0;
        end 

        // Khi Max Pooling tìm xong 1 điểm ảnh max và sẵn sàng ghi
        else if (wr_en_bram) begin     
            // Nếu đã ghi đến pixel cuối cùng của layer
            if (w_addr_bram == bram_limit - 1) begin
                full_bram <= 1'b1;
				w_addr_bram <= 0;
			end

            else
                w_addr_bram <= w_addr_bram + 8'd1;
        end
    end
	 
endmodule