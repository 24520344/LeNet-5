`timescale 1ns / 1ps

//====================================================================================
// Organization:    University of Information Technology - VNU-HCM
// Author:          Dung Nguyen Viet
// Student ID:      24520344
//
// File Name:       argmax.v
// Module Name:     argmax
// Project:         Digit Recognition Using LeNet-5 CNN
// Software:        Vivado 2025.1 & Visual Studio Code
//====================================================================================

module argmax #(
    parameter DATA_WIDTH = 16,
    parameter NUM_PE = 4
) (
    input wire clk,
    input wire rst_n,
    input wire en,
    input wire [1:0] read_cycle,
    input wire signed [NUM_PE * DATA_WIDTH - 1:0] data_in, 

    output reg [3:0] predicted_digit,
    output reg valid_out
);

    reg signed [DATA_WIDTH - 1:0] max_val;
    
    wire signed [DATA_WIDTH - 1:0] val_0 = data_in[DATA_WIDTH - 1 -: DATA_WIDTH];
    wire signed [DATA_WIDTH - 1:0] val_1 = data_in[2 * DATA_WIDTH - 1 -: DATA_WIDTH];
    wire signed [DATA_WIDTH - 1:0] val_2 = data_in[3 * DATA_WIDTH - 1 -: DATA_WIDTH];
    wire signed [DATA_WIDTH - 1:0] val_3 = data_in[4 * DATA_WIDTH - 1 -: DATA_WIDTH];

    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            predicted_digit <= 4'd0;
            max_val <= {1'b1, {(DATA_WIDTH-1){1'b0}}}; // Số âm nhỏ nhất
            valid_out <= 1'b0;
        end 

        else if (en) begin
            case (read_cycle)
                2'd0: begin // So sánh Nơ-ron 0, 1, 2, 3
                    if (val_0 > max_val) begin max_val <= val_0; predicted_digit <= 4'd0; end
                    if (val_1 > max_val && val_1 > val_0) begin max_val <= val_1; predicted_digit <= 4'd1; end
                    if (val_2 > max_val && val_2 > val_1 && val_2 > val_0) begin max_val <= val_2; predicted_digit <= 4'd2; end
                    if (val_3 > max_val && val_3 > val_2 && val_3 > val_1 && val_3 > val_0) begin max_val <= val_3; predicted_digit <= 4'd3; end
                end
                
                2'd1: begin // So sánh Nơ-ron 4, 5, 6, 7
                    if (val_0 > max_val) begin max_val <= val_0; predicted_digit <= 4'd4; end
                    if (val_1 > max_val && val_1 > val_0) begin max_val <= val_1; predicted_digit <= 4'd5; end
                    if (val_2 > max_val && val_2 > val_1 && val_2 > val_0) begin max_val <= val_2; predicted_digit <= 4'd6; end
                    if (val_3 > max_val && val_3 > val_2 && val_3 > val_1 && val_3 > val_0) begin max_val <= val_3; predicted_digit <= 4'd7; end
                end
                
                2'd2: begin // So sánh Nơ-ron 8, 9
                    if (val_0 > max_val) begin max_val <= val_0; predicted_digit <= 4'd8; end
                    if (val_1 > max_val && val_1 > val_0) begin max_val <= val_1; predicted_digit <= 4'd9; end
                    valid_out <= 1'b1; // Quét xong 10 class
                end
            endcase
        end

        else begin
            // Reset khi chuẩn bị chạy ảnh mới
            max_val <= {1'b1, {(DATA_WIDTH-1){1'b0}}};
            valid_out <= 1'b0;
        end
    end

endmodule