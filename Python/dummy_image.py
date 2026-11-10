import math
import numpy as np
import os


def generate_dummy_fpga_data():
    print("=== TẠO DỮ LIỆU DEBUG CHO FPGA (DẠNG HEX 16-BIT & 64-BIT) ===")

    # =========================================================
    # 1. TẠO ẢNH ĐẦU VÀO (32x32)
    # Giá trị của mỗi pixel chính là địa chỉ (index) của nó (0 đến 1023)
    # Ảnh vẫn giữ dạng 16-bit 1 cột vì Line Buffer đọc vào từng pixel 1
    # =========================================================
    image_32x32 = np.arange(1024, dtype=int).reshape((32, 32))

    with open("test_image.txt", "w") as f:
        for val in image_32x32.flatten():
            # Ép sang int, lấy 16 bit cuối và format thành 4 chữ số Hex in hoa
            val_int = int(val)
            f.write(f"{val_int & 0xFFFF:04X}\n")
    print("-> Đã tạo test_image.txt (Dạng Hex 16-bit, Giá trị = Địa chỉ Index)")

    # =========================================================
    # 2. CẤU HÌNH CÁC LỚP TRỌNG SỐ (Khớp 100% với Testbench)
    # =========================================================
    layers_config = [
        {
            "name": "Conv1",
            "w_file": "layer1_conv1_weight.txt", "b_file": "layer1_conv1_bias.txt",
            "w_shape": (5, 5, 1, 4), "b_shape": (4,)
        },
        {
            "name": "Conv2",
            "w_file": "layer3_conv2_weight.txt", "b_file": "layer3_conv2_bias.txt",
            "w_shape": (5, 5, 4, 12), "b_shape": (12,)
        },
        {
            "name": "Dense1",
            "w_file": "layer6_dense1_weight.txt", "b_file": "layer6_dense1_bias.txt",
            "w_shape": (300, 64), "b_shape": (64,)
        },
        {
            "name": "Dense2",
            "w_file": "layer7_dense2_weight.txt", "b_file": "layer7_dense2_bias.txt",
            "w_shape": (64, 10), "b_shape": (10,)
        }
    ]

    # =========================================================
    # 3. LOGIC XUẤT RA DẠNG QUAD PE (Format Hex 64-bit)
    # =========================================================
    for layer in layers_config:
        w_shape = layer["w_shape"]
        b_shape = layer["b_shape"]

        total_w = np.prod(w_shape)
        w_kernel = np.arange(total_w, dtype=int).reshape(w_shape)

        total_b = np.prod(b_shape)
        w_bias = np.arange(total_b, dtype=int)

        # --- XUẤT WEIGHTS ---
        with open(layer["w_file"], "w", encoding='utf-8') as fw:
            if len(w_shape) == 4:  # Conv2D
                H, W, C_in, C_out = w_shape
                num_groups = math.ceil(C_out / 4)
                for g in range(num_groups):
                    for h in range(H):
                        for w in range(W):
                            for c in range(C_in):
                                row_vals = []
                                for k in range(4):
                                    out_c = g * 4 + k
                                    val = w_kernel[h, w, c, out_c] if out_c < C_out else 0

                                    # Format Hex 16-bit
                                    val_int = int(val)
                                    row_vals.append(f"{val_int & 0xFFFF:04X}")

                                # Nối 4 giá trị lại thành 1 chuỗi 64-bit (16 ký tự Hex)
                                # Phải đảo ngược mảng (reversed) để LSB (k=0) nằm bên phải và MSB (k=3) nằm bên trái
                                fw.write("".join(reversed(row_vals)) + "\n")

            elif len(w_shape) == 2:  # Dense
                In_Dim, Out_Dim = w_shape
                num_groups = math.ceil(Out_Dim / 4)
                for g in range(num_groups):
                    for in_node in range(In_Dim):
                        row_vals = []
                        for k in range(4):
                            out_c = g * 4 + k
                            val = w_kernel[in_node, out_c] if out_c < Out_Dim else 0

                            val_int = int(val)
                            row_vals.append(f"{val_int & 0xFFFF:04X}")

                        # Nối 4 giá trị lại thành 1 chuỗi 64-bit (16 ký tự Hex)
                        fw.write("".join(reversed(row_vals)) + "\n")

        # --- XUẤT BIAS ---
        with open(layer["b_file"], "w", encoding='utf-8') as fb:
            num_groups = math.ceil(len(w_bias) / 4)
            for g in range(num_groups):
                row_vals = []
                for k in range(4):
                    out_c = g * 4 + k
                    val = w_bias[out_c] if out_c < len(w_bias) else 0

                    val_int = int(val)
                    row_vals.append(f"{val_int & 0xFFFF:04X}")

                # Nối 4 giá trị lại thành 1 chuỗi 64-bit (16 ký tự Hex)
                fb.write("".join(reversed(row_vals)) + "\n")

        print(f"-> Đã tạo {layer['name']}: {layer['w_file']} và {layer['b_file']} (Dạng 64-bit)")

    print("=== HOÀN TẤT! SẴN SÀNG ĐỂ NẠP BẰNG $READMEMH ===")


generate_dummy_fpga_data()