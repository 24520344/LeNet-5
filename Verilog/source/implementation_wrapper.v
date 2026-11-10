`timescale 1ns / 1ps

module implementation_wrapper (
    input wire clk,
    input wire rst_n,
    input wire start,

    output wire done
);
    // Tham số chung
    localparam DATA_WIDTH = 16;
    localparam KERNEL_WIDTH = 5;
    localparam NUM_PE = 4;

    // Tham số PE/ QUAD PE
    localparam FRACTIONAL_BITS = 10;

    // Tham số riêng Max Pooling/ Quad Max Pooling
    localparam FEATURE_IN_WIDTH = 28;

    // Tham số riêng Line Buffer
    localparam INPUT_IMAGE_WIDTH = 32;

    // Tham số riêng Weight Buffer
    localparam NUM_FIFO_WB = 4;
    localparam NUM_PE_WB = 4;
    localparam BUFFER_SIZE_WB = 4800;

    // Tham số riêng Bias Buffer
    localparam NUM_FIFO_BB = 4;
    localparam NUM_PE_BB = 4;
    localparam BUFFER_SIZE_BB = 16;

    // Tham số riêng BRAM
    localparam BRAM_SIZE = 196;


    // 1. Khai báo các dây nội bộ (Virtual Wires) - Không tốn chân IO
    wire signed [15:0] virtual_pixel_in;
    wire signed [63:0] virtual_weight_in;
    wire signed [63:0] virtual_bias_in;
    wire [1:0] curr_layer;
    wire wr_en_bb;
    wire wr_en_wb;
    wire wr_en_lb;
    wire bram_wr_en;
    wire signed [63:0] virtual_pixel_out;

    // 2. Tạo dữ liệu giả (Dummy Data) để tránh bị Vivado tối ưu xóa mất mạch
    // Chúng ta dùng một bộ đếm đơn giản để các dây luôn có tín hiệu thay đổi
    reg [63:0] dummy_counter;
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) dummy_counter <= 0;
        else dummy_counter <= dummy_counter + 1;
    end

    assign virtual_pixel_in  = dummy_counter;
    assign virtual_weight_in = ~dummy_counter; // Đảo bit để khác biệt
    assign virtual_bias_in   = dummy_counter ^ 64'h5555_5555_AAAA_AAAA;

    // 3. Kết nối vào lõi CNN thực tế của bạn
    top_module #(
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

    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .pixel_in(virtual_pixel_in),
        .weight_in(virtual_weight_in),
        .bias_in(virtual_bias_in),
        .done(done),
        .curr_layer(curr_layer),
        .wr_en_bb(wr_en_bb),
        .wr_en_wb(wr_en_wb),
        .wr_en_lb(wr_en_lb),
        .bram_wr_en(bram_wr_en),
        .pixel_out(virtual_pixel_out)
    );

endmodule