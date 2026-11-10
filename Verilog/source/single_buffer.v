`timescale 1ns / 1ps

//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/10/2025 12:53:24 PM
// Design Name: 
// Module Name: buffer
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module single_buffer #(
    parameter DATA_WIDTH = 16,

    // Lượng pixel input song song
    parameter NUM_FIFO = 4,

    // Lượng pixel output song song
    parameter NUM_PE = 4,

    // Độ sâu (số hàng) của buffer
    parameter BUFFER_SIZE = 16

    // Tỉ lệ dịch địa chỉ giữa số lượng pixel input và output
    //parameter OFFSET_SIZE = 4
) (
    input wire clk,
    input wire rst_n,
    input wire wr_en,
    input wire rd_en,
    input wire clear,
    input wire [$clog2(BUFFER_SIZE) - 1:0] rd_addr,
    input wire [$clog2(BUFFER_SIZE):0] buffer_limit,
    input wire signed [DATA_WIDTH * NUM_FIFO - 1:0] buffer_in,

    //output reg [$clog2(BUFFER_SIZE) - 1:0] pixel_addr,
    output wire valid_out,
    output reg full,
    output reg signed [DATA_WIDTH * NUM_PE - 1:0] buffer_out
);
    // DATA_WIDTH (Độ rộng) của 1 hàng buffer
    localparam BUFFER_WIDTH = DATA_WIDTH * NUM_FIFO;

    reg [$clog2(BUFFER_SIZE) - 1:0] pixel_addr;

    // Cài đặt buffer sẽ synth thành thanh ghi (thay vì mặc định là RAM) trên FPGA
    (* ram_style = "block" *) reg signed [DATA_WIDTH*NUM_PE-1:0] mem [0:BUFFER_SIZE-1];
    
    //reg [$clog2(BUFFER_SIZE) - 1:0] pixel_addr;
    //reg [$clog2(OFFSET_SIZE)-1:0] pixel_offset;

    always @(posedge clk) begin
        if (~rst_n) begin
            // Write
            pixel_addr <= 0;
            //pixel_offset <= 0;
            
            // Read
            buffer_out <= 0;

            // Flags
            //valid_out <= 0;
            full <= 0;
        end

        else if (clear) begin
            // Reset con trỏ và cờ để truyền bias từ lớp khác vào
            pixel_addr <= 0;
            //pixel_offset <= 0;
            //valid_out <= 0;
            full <= 0;
        end

        // Logic ghi và đọc chính
        else begin

            // Write
            if (wr_en & !full) begin

                // Ghi vào buffer
                mem[pixel_addr]/*[(pixel_offset*BUFFER_WIDTH) +: BUFFER_WIDTH]*/ <= buffer_in;
                
                // Con trỏ ở hàng cuối cùng và ở slot cột pixel cuối cùng
                //if (pixel_offset == OFFSET_SIZE - 1) begin
                    // Quay về đầu hàng
                    //pixel_offset <= 0;

                    // Con trỏ tới hàng cuối cùng
                    if (pixel_addr == buffer_limit - 1) begin                    
                        // Reset về hàng đầu tiên
                        pixel_addr <= 0;

                        // Báo full
                        full <= 1'b1;
                    end
                    
                    // Chưa chạy đến hàng cuối cùng
                    else
                        pixel_addr <= pixel_addr + 1;
                    
                //end
                
                // Chưa chạy đến pixel cuối cùng của hàng cuối cùng
                //else
                //    pixel_offset <= pixel_offset + 1;
                    
            end
            
            // Read
            if (rd_en) begin
                // Đọc NUM_PE số pixel tại hàng buffer đó
                buffer_out <= mem[rd_addr];

                // Dữ liệu đã sẵn sàng để xuất
                //valid_out <= 1'b1;
            end
            
            //else
                //valid_out <= 1'b0;
        end
    end
    assign valid_out = rd_en; // Dữ liệu đã sẵn sàng để xuất ngay khi rd_en = 1 (đọc dữ liệu ra) vì đọc từ buffer là thao tác combinational

endmodule