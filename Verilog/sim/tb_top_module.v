`timescale 1ns/1ps

module tb_top_module;

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
    


    // Input
    reg clk;
    reg rst_n;
    reg start;

    reg signed [DATA_WIDTH - 1:0] pixel_in;
    reg signed [DATA_WIDTH * NUM_FIFO_WB - 1:0] weight_in;
    reg signed [DATA_WIDTH * NUM_FIFO_BB - 1:0] bias_in;

    // Output
    wire done;
    wire [1:0] curr_layer;
    wire wr_en_bb;
    wire wr_en_wb;
    wire wr_en_lb;
    wire [3:0] predicted_digit;



    // Khởi tạo top module
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

        .pixel_in(pixel_in),
        .weight_in(weight_in),
        .bias_in(bias_in),

        .done(done),
        .curr_layer(curr_layer),
        .wr_en_bb(wr_en_bb),
        .wr_en_wb(wr_en_wb),
        .wr_en_lb(wr_en_lb),
        .predicted_digit(predicted_digit)
    );



    // =========================================================================
    // DEBUG
    // =========================================================================
    // 1. FSM & Counters (Theo dõi chuyển trạng thái và các vòng lặp)
    wire [2:0] debug_curr_state   = dut.controller_inst.curr_state;
    wire [2:0] debug_next_state   = dut.controller_inst.next_state;
    wire [8:0] debug_mac_cnt      = dut.controller_inst.mac_cnt;
    wire [9:0] debug_slides_cnt   = dut.controller_inst.slides_cnt;
    wire [4:0] debug_filters_cnt = dut.controller_inst.filters_cnt;
    wire [1:0] debug_quad_pe_channels_cnt   = dut.controller_inst.quad_pe_channels_cnt;

    // 2. Global & Layer Limits (Theo dõi cấu hình giới hạn của Layer hiện tại)
    wire [8:0] debug_mac_limit    = dut.controller_inst.mac_limit;
    wire       debug_bram_in_sel  = dut.controller_inst.bram_in_sel;
    wire debug_bram_sel     = dut.controller_inst.bram_sel;

    // 6. Bias Buffer (Độ lệch)
    wire       debug_wr_en_bb       = dut.controller_inst.wr_en_bb;
    wire       debug_rd_en_bb       = dut.controller_inst.rd_en_bb;
    wire       debug_clear_bb       = dut.controller_inst.clear_bb;
    wire [$clog2(BUFFER_SIZE_BB):0] debug_wr_addr_bb = dut.datapath_inst.bias_buffer.pixel_addr;
    wire [$clog2(BUFFER_SIZE_BB):0] debug_rd_addr_bb = dut.controller_inst.rd_addr_bb;
    wire [$clog2(BUFFER_SIZE_BB):0] buffer_limit_bb = dut.controller_inst.buffer_limit_bb;

    wire signed [DATA_WIDTH * NUM_FIFO_BB - 1:0] debug_bias_in             = dut.datapath_inst.bias_in;
    wire                                  debug_valid_out_bb        = dut.datapath_inst.valid_out_bb;
    wire signed [DATA_WIDTH * NUM_PE_BB - 1:0]   debug_bias_out_bb         = dut.datapath_inst.bias_out_bb;
    wire debug_full_bb              = dut.datapath_inst.full_bb;

    // 5. Weight Buffer (Trọng số)
    wire       debug_wr_en_wb       = dut.controller_inst.wr_en_wb;
    wire       debug_rd_en_wb       = dut.controller_inst.rd_en_wb;
    wire       debug_clear_wb       = dut.controller_inst.clear_wb;
    wire [$clog2(BUFFER_SIZE_WB):0] debug_wr_addr_wb = dut.datapath_inst.weight_buffer.pixel_addr;
    wire [$clog2(BUFFER_SIZE_WB):0] debug_rd_addr_wb = dut.controller_inst.rd_addr_wb;
    wire [$clog2(BUFFER_SIZE_WB):0] buffer_limit_wb = dut.controller_inst.buffer_limit_wb;

    wire signed [DATA_WIDTH * NUM_FIFO_WB - 1:0] debug_weight_in           = dut.datapath_inst.weight_in;
    wire                                  debug_valid_out_wb        = dut.datapath_inst.valid_out_wb;
    wire signed [DATA_WIDTH * NUM_PE_WB - 1:0]   debug_weight_out_wb       = dut.datapath_inst.weight_out_wb;
    wire debug_full_wb              = dut.datapath_inst.full_wb;

    // 4. Line Buffer (Ảnh đầu vào)
    wire debug_shift_window = dut.controller_inst.shift_window;
    wire debug_rewind_lb            = dut.controller_inst.rewind;
    wire       debug_wr_en_lb       = dut.controller_inst.wr_en_lb;
    wire       debug_rd_en_lb       = dut.controller_inst.rd_en_lb;
    wire [$clog2(INPUT_IMAGE_WIDTH):0] debug_col_addr_lb = dut.datapath_inst.line_buffer_inst.col_addr;
    wire [$clog2(INPUT_IMAGE_WIDTH):0] debug_row_count_lb = dut.datapath_inst.line_buffer_inst.row_count;
    wire [KERNEL_WIDTH-1:0] debug_output_index = dut.controller_inst.output_index;
    wire       debug_clear_lb       = dut.controller_inst.clear_lb;
    wire [$clog2(INPUT_IMAGE_WIDTH):0] buffer_limit_lb = dut.controller_inst.buffer_limit_lb;

    wire signed [NUM_PE * DATA_WIDTH - 1:0]      debug_lb_pixel_in         = dut.datapath_inst.line_buffer_inst.pixel_in;

    wire                                  debug_valid_out_lb        = dut.datapath_inst.valid_out_lb;
    wire debug_full_lb              = dut.datapath_inst.full_lb;
    wire signed [NUM_PE * DATA_WIDTH - 1:0]      debug_pixel_out_lb        = dut.datapath_inst.pixel_out_lb;

    // 3. QUAD PE (Xử lý tích chập)
    wire       debug_en_quad_pe         = dut.controller_inst.en_quad_pe;
    wire       debug_first_channel      = dut.controller_inst.first_channel_quad_pe;

    wire signed [DATA_WIDTH - 1:0]               debug_quad_pe_pixel_in    = dut.datapath_inst.quad_pe_inst.pixel_in;
    wire                                  debug_valid_out_quad_pe   = dut.datapath_inst.valid_out_quad_pe;
    wire signed [NUM_PE * DATA_WIDTH - 1:0]      debug_pixel_out_quad_pe   = dut.datapath_inst.pixel_out_quad_pe;

    wire signed [NUM_PE * DATA_WIDTH - 1:0]      debug_pixel_out_quad_relu = dut.datapath_inst.pixel_out_quad_relu;

    // 7. Max Pooling & BRAM (Tính toán sau tích chập & Lớp Dense)
    wire debug_wr_en_mp        = dut.datapath_inst.quad_max_pooling_inst.wr_en;
    wire       debug_clear_mp       = dut.controller_inst.clear_mp;
    wire [$clog2(FEATURE_IN_WIDTH) - 1:0] buffer_limit_mp = dut.controller_inst.buffer_limit_mp;
    wire                                  debug_valid_out_mp        = dut.datapath_inst.valid_out_mp;
    wire signed [NUM_PE * DATA_WIDTH - 1:0]      debug_pixel_out_mp        = dut.datapath_inst.pixel_out_mp;
    
    wire       debug_full_bram      = dut.controller_inst.full_bram;
    wire [$clog2(BRAM_SIZE) - 1:0]        debug_wr_addr_bram        = dut.datapath_inst.wr_addr_bram;
    wire       debug_clear_bram     = dut.controller_inst.clear_bram;
    wire [$clog2(BRAM_SIZE) - 1:0] bram_limit = dut.controller_inst.bram_limit;

    wire       debug_full_bram_dense = dut.controller_inst.full_bram_dense;
    wire [$clog2(BRAM_SIZE) - 1:0]        debug_wr_addr_bram_dense  = dut.datapath_inst.wr_addr_bram_dense;
    wire       debug_clear_bram_dense = dut.controller_inst.clear_bram_dense;
    wire [$clog2(BRAM_SIZE) - 1:0] bram_dense_limit = dut.controller_inst.bram_dense_limit;

    wire debug_wr_en_bram0 = dut.datapath_inst.bram_inst0.wr_en;
    wire debug_wr_en_bram1 = dut.datapath_inst.bram_inst1.wr_en;
    wire       debug_rd_en_bram0     = dut.datapath_inst.bram_inst0.rd_en;
    wire       debug_rd_en_bram1     = dut.datapath_inst.bram_inst1.rd_en;

    wire [$clog2(BRAM_SIZE) - 1:0]        debug_bram_inst_w_addr    = dut.datapath_inst.bram_inst0.w_addr;
    wire [$clog2(BRAM_SIZE) - 1:0]        debug_bram_inst_r_addr    = dut.datapath_inst.bram_inst0.r_addr;
    wire signed [NUM_PE * DATA_WIDTH - 1:0]      debug_bram_data_in        = dut.datapath_inst.bram_inst0.data_in;

    wire signed [NUM_PE * DATA_WIDTH - 1:0] debug_bram0_data_out          = dut.datapath_inst.bram_inst0.data_out;
    wire signed [NUM_PE * DATA_WIDTH - 1:0] debug_bram1_data_out          = dut.datapath_inst.bram_inst1.data_out;

    // ARGMAX
    wire       debug_en_argmax      = dut.controller_inst.en_argmax;
    wire [1:0] debug_read_cycle_argmax = dut.controller_inst.read_cycle_argmax;

    wire       debug_valid_out_argmax = dut.datapath_inst.valid_out_argmax;
    //wire [3:0] debug_predicted_digit = dut.datapath_inst.predicted_digit;



    // =========================================================================
    // KHAI BÁO BỘ NHỚ ẢO PHÂN TÁCH THEO TỪNG LAYER
    // =========================================================================
    reg signed [DATA_WIDTH - 1:0] image_mem [0:1023];
    reg [3:0] image_label [0:0];
    
    // Layer 0 (Conv1)
    reg signed [NUM_PE_WB * DATA_WIDTH - 1:0] w_mem_0 [0:24];      // 5x5 * 1 in * 1 group * 4 = 100
    reg signed [NUM_PE_BB * DATA_WIDTH - 1:0] b_mem_0 [0:0];       // 4 biases

    // Layer 1 (Conv2)
    reg signed [NUM_PE_WB * DATA_WIDTH - 1:0] w_mem_1 [0:299];    // 5x5 * 4 in * 3 groups * 4 = 1200
    reg signed [NUM_PE_BB * DATA_WIDTH - 1:0] b_mem_1 [0:2];      // 12 biases

    // Layer 2 (Dense1)
    reg signed [NUM_PE_WB * DATA_WIDTH - 1:0] w_mem_2 [0:4799];   // 300 in * 16 groups * 4 = 19200
    reg signed [NUM_PE_BB * DATA_WIDTH - 1:0] b_mem_2 [0:15];      // 64 biases

    // Layer 3 (Dense2)
    reg signed [NUM_PE_WB * DATA_WIDTH - 1:0] w_mem_3 [0:191];     // 64 in * 3 groups * 4 = 768 (Padding từ 10 thành 12)
    reg signed [NUM_PE_BB * DATA_WIDTH - 1:0] b_mem_3 [0:2];      // 12 biases (Padding từ 10 thành 12)



    // =========================================================================
    // TẠO CLOCK
    // =========================================================================
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end



    initial begin
        $readmemh("test_image.txt", image_mem);
        $readmemh("test_label.txt", image_label);

        $readmemh("layer1_conv1_weight.txt", w_mem_0);
        $readmemh("layer1_conv1_bias.txt", b_mem_0);

        $readmemh("layer3_conv2_weight.txt", w_mem_1);
        $readmemh("layer3_conv2_bias.txt", b_mem_1);

        $readmemh("layer6_dense1_weight.txt", w_mem_2);
        $readmemh("layer6_dense1_bias.txt", b_mem_2);

        $readmemh("layer7_dense2_weight.txt", w_mem_3);
        $readmemh("layer7_dense2_bias.txt", b_mem_3);

        //#10;
        start = 0;
        rst_n = 0;

        #10;
        rst_n = 1;
        start = 1;

        wait(done == 1'b1);
        if (predicted_digit === image_label[0]) begin
            $display("Test Passed: Predicted Digit = %0d, Expected Label = %0d", predicted_digit, image_label[0]);
        end else begin
            $display("Test Failed: Predicted Digit = %0d, Expected Label = %0d", predicted_digit, image_label[0]);
        end

        $finish;
    end



    // // =========================================================================
    // // BIẾN CHO QUÁ TRÌNH MULTI-TEST
    // // =========================================================================
    // integer i;
    // integer current_idx;
    // integer correct_count = 0;
    // localparam TOTAL_TESTS = 10; // Số lượng ảnh muốn test
    // integer test_indices [0:TOTAL_TESTS-1];

    // reg [8*256-1:0] image_filename;
    // reg [8*256-1:0] label_filename;
    // reg [3:0] expected_label [0:0]; 


    // // =========================================================================
    // // KỊCH BẢN MÔ PHỎNG CHÍNH (Đã được hợp nhất)
    // // =========================================================================
    // initial begin
    //     // 1. Nạp tệp trọng số (Chỉ cần nạp 1 lần ở đầu mô phỏng)
    //     $readmemh("layer1_conv1_weight.txt", w_mem_0);
    //     $readmemh("layer1_conv1_bias.txt", b_mem_0);

    //     $readmemh("layer3_conv2_weight.txt", w_mem_1);
    //     $readmemh("layer3_conv2_bias.txt", b_mem_1);

    //     $readmemh("layer6_dense1_weight.txt", w_mem_2);
    //     $readmemh("layer6_dense1_bias.txt", b_mem_2);

    //     $readmemh("layer7_dense2_weight.txt", w_mem_3);
    //     $readmemh("layer7_dense2_bias.txt", b_mem_3);

    //     // 2. Cấu hình danh sách các index muốn test
    //     test_indices[0] = 99;
    //     test_indices[1] = 88;
    //     test_indices[2] = 77;
    //     test_indices[3] = 66;
    //     test_indices[4] = 55;
    //     test_indices[5] = 44;
    //     test_indices[6] = 33;
    //     test_indices[7] = 22;
    //     test_indices[8] = 11;
    //     test_indices[9] = 1;

    //     // 3. Khởi tạo tín hiệu cơ bản
    //     start = 0;
    //     rst_n = 0;
    //     #100;
    //     rst_n = 1;

    //     $display("\n=======================================================");
    //     $display("   BAT DAU CHAY AUTOMATED MULTI-TEST CASES");
    //     $display("=======================================================\n");

    //     // 4. Vòng lặp nạp từng ảnh và chạy suy luận
    //     for (i = 0; i < TOTAL_TESTS; i = i + 1) begin
    //         current_idx = test_indices[i];

    //         // Định dạng tên file
    //         $sformat(image_filename, "test_image_%0d.txt", current_idx);
    //         $sformat(label_filename, "test_label_%0d.txt", current_idx);

    //         $display("[%0t] Dang nap Test Case Index: %0d", $time, current_idx);

    //         // Nạp dữ liệu ảnh VÀO image_mem của Testbench
    //         $readmemh(image_filename, image_mem); 
            
    //         // Nạp nhãn mong muốn
    //         $readmemh(label_filename, expected_label);

    //         // Reset hệ thống trước khi dự đoán ảnh mới (Reset cứng)
    //         // Việc reset rst_n = 0 sẽ tự động đưa img_ptr, w_ptr, b_ptr về 0 ở khối sequential
    //         rst_n = 0;
    //         #50;
    //         rst_n = 1;
    //         #50;

    //         // Kích hoạt tín hiệu Start
    //         @(posedge clk);
    //         start = 1;

    //         // Chờ mô hình tính toán xong
    //         wait(done == 1'b1);
    //         @(posedge clk);

    //         // Đối chiếu kết quả (Sửa biến predict_label -> predicted_digit)
    //         if (predicted_digit === expected_label[0]) begin
    //             $display(" -> [PASS] HW Doan: %0h | Nhan Dung: %0h", predicted_digit, expected_label[0]);
    //             correct_count = correct_count + 1;
    //         end else begin
    //             $display(" -> [FAIL] HW Doan: %0h | Nhan Dung: %0h", predicted_digit, expected_label[0]);
    //         end
            
    //         #100; // Đợi một chút trước khi sang ảnh tiếp theo
    //         start = 0;
    //     end

    //     // Báo cáo tổng kết
    //     $display("\n=======================================================");
    //     $display(" TONG KET: Tra loi dung %0d / %0d test cases", correct_count, TOTAL_TESTS);
    //     $display("=======================================================\n");

    //     $stop; 
    // end



    // =========================================================================
    // QUẢN LÝ DỮ LIỆU ĐƯA VÀO MODULE CHÍNH (Giữ nguyên của bạn)
    // =========================================================================
    integer img_ptr = 0, w_ptr = 0, b_ptr = 0;
    reg [1:0] prev_layer = 0;

    always @(*) begin
        if (curr_layer == 0)
            pixel_in = image_mem[img_ptr];
        else
            pixel_in = 0;

        case (curr_layer)
            2'd0: begin weight_in = w_mem_0[w_ptr]; bias_in = b_mem_0[b_ptr]; end
            2'd1: begin weight_in = w_mem_1[w_ptr]; bias_in = b_mem_1[b_ptr]; end
            2'd2: begin weight_in = w_mem_2[w_ptr]; bias_in = b_mem_2[b_ptr]; end
            2'd3: begin weight_in = w_mem_3[w_ptr]; bias_in = b_mem_3[b_ptr]; end
            default: begin weight_in = 0; bias_in = 0; end
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            img_ptr <= 0; w_ptr <= 0; b_ptr <= 0;
            prev_layer <= 0;
        end else begin
            // Layer Transition Reset
            if (curr_layer != prev_layer) begin
                w_ptr <= 0; 
                b_ptr <= 0;
                prev_layer <= curr_layer;
            end
            else begin
                if (wr_en_wb) w_ptr <= w_ptr + 1;
                if (wr_en_bb) b_ptr <= b_ptr + 1;
            end

            // Nạp Image 16-bit
            if (wr_en_lb && curr_layer == 0)
                img_ptr <= img_ptr + 1;
        end
    end

endmodule