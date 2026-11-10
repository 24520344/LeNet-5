`timescale 1ns / 1ps

module tb_pe;

    // Tham số
    parameter DATA_WIDTH = 16;
    parameter FRACTIONAL_BITS = 10;

    // Khai báo tín hiệu Testbench
    reg clk;
    reg rst_n;
    reg en;
    reg first_channel;
    reg [8:0] mac_limit;
    reg signed [DATA_WIDTH - 1:0] pixel_in;
    reg signed [DATA_WIDTH - 1:0] weight_in;
    reg signed [DATA_WIDTH - 1:0] bias_in;

    // Tín hiệu ngõ ra
    wire valid_out;
    wire signed [DATA_WIDTH - 1:0] pixel_out;

    // Khởi tạo module PE (Device Under Test)
    pe #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRACTIONAL_BITS(FRACTIONAL_BITS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .first_channel(first_channel),
        .mac_limit(mac_limit),
        .pixel_in(pixel_in),
        .weight_in(weight_in),
        .bias_in(bias_in),
        .valid_out(valid_out),
        .pixel_out(pixel_out)
    );

    // Tạo xung nhịp Clock (Chu kỳ 10ns -> Tần số 100MHz)
    always #5 clk = ~clk;

    // Khối kịch bản mô phỏng
    initial begin
        // 1. Khởi tạo giá trị ban đầu
        clk = 0;
        rst_n = 0;
        en = 0;
        first_channel = 0;
        mac_limit = 4; // Giới hạn 4 phép nhân cộng (MAC) cho 1 luồng
        pixel_in = 0;
        weight_in = 0;
        bias_in = 0;

        // Reset hệ thống
        #15 rst_n = 1;
        #10;

        $display("=== BAT DAU MO PHONG PE ===");
        $display("Dinh dang Fixed-point Q6.10: 1.0 = 1024, 0.5 = 512, 0.25 = 256");

        // ---------------------------------------------------------
        // TEST CASE 1: Tích chập kênh đầu tiên (Có cộng Bias)
        // Số MAC = 4 nhịp. Bias = 1.0 (1024)
        // ---------------------------------------------------------
        @(posedge clk);
        en = 1;
        first_channel = 1; // Nhịp đầu tiên của channel đầu tiên sẽ nạp Bias
        bias_in = 16'd1024; // Bias = 1.0
        
        // Nhịp 1: 1.0 * 0.5 = 0.5. Tích lũy = Bias + 0.5 = 1.5 (1536)
        pixel_in = 16'd1024;  // 1.0
        weight_in = 16'd512;  // 0.5

        @(posedge clk);
        first_channel = 0; // Tắt cờ first_channel sau nhịp đầu
        // Nhịp 2: 1.0 * 1.0 = 1.0. Tích lũy = 1.5 + 1.0 = 2.5 (2560)
        pixel_in = 16'd1024;  // 1.0
        weight_in = 16'd1024; // 1.0

        @(posedge clk);
        // Nhịp 3: 0.5 * 0.5 = 0.25. Tích lũy = 2.5 + 0.25 = 2.75 (2816)
        pixel_in = 16'd512;   // 0.5
        weight_in = 16'd512;  // 0.5

        @(posedge clk);
        // Nhịp 4 (Cuối): -1.0 * 0.5 = -0.5. Tích lũy = 2.75 - 0.5 = 2.25 (2304)
        pixel_in = -16'd1024; // -1.0
        weight_in = 16'd512;  // 0.5

        // ---------------------------------------------------------
        // KIỂM TRA KẾT QUẢ TEST CASE 1
        // ---------------------------------------------------------
        @(posedge clk);
        en = 0; // Dừng cấp dữ liệu
        
        // Nhịp này valid_out phải bật lên 1 và pixel_out = 2304
        if (valid_out && pixel_out == 16'd2304) begin
            $display("[PASS] Test Case 1: Tich luy hoan hao! pixel_out = %d (Ky vong: 2304)", pixel_out);
        end else begin
            $display("[FAIL] Test Case 1: pixel_out = %d, valid_out = %b", pixel_out, valid_out);
        end

        #20;
        
        // ---------------------------------------------------------
        // TEST CASE 2: Kênh tiếp theo (Không cộng Bias, cộng dồn tiếp vào bộ nhớ)
        // ---------------------------------------------------------
        @(posedge clk);
        en = 1;
        first_channel = 0;  // KHÔNG phải kênh đầu -> Cần cộng dồn vào giá trị cũ (2304)
        mac_limit = 2;      // Chạy 2 nhịp

        // Nhịp 1: 0.5 * 1.0 = 0.5 (512). Tích lũy = 2.25 + 0.5 = 2.75 (2816)
        pixel_in = 16'd512;   // 0.5
        weight_in = 16'd1024; // 1.0

        @(posedge clk);
        // Nhịp 2: -0.25 * 1.0 = -0.25 (-256). Tích lũy = 2.75 - 0.25 = 2.5 (2560)
        pixel_in = -16'd256;  // -0.25
        weight_in = 16'd1024; // 1.0

        @(posedge clk);
        en = 0;
        
        if (valid_out && pixel_out == 16'd2560) begin
            $display("[PASS] Test Case 2: Cong don kenh ke tiep hoan hao! pixel_out = %d (Ky vong: 2560)", pixel_out);
        end else begin
            $display("[FAIL] Test Case 2: pixel_out = %d, valid_out = %b", pixel_out, valid_out);
        end

        #30;
        $display("=== KET THUC MO PHONG ===");
        $finish;
    end

endmodule