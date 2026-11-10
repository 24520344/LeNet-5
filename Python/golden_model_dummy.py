import numpy as np
import math
import os

# --- CẤU HÌNH PHẦN CỨNG ---
FRACTIONAL_BITS = 10


def hex_to_signed(hex_str):
    v = int(hex_str, 16)
    if v >= 0x8000:
        v -= 0x10000
    return v


def load_test_image(filename):
    vals = []
    if not os.path.exists(filename):
        print(f"LỖI: Không tìm thấy '{filename}'")
        return np.zeros((32, 32, 1), dtype=int)

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: vals.append(hex_to_signed(line))

    return np.array(vals[:1024]).reshape((32, 32, 1))


def load_hw_64bit_file(filename):
    vals = []
    if not os.path.exists(filename):
        print(f"LỖI: Không tìm thấy '{filename}'")
        return vals

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue

            h3, h2, h1, h0 = line[0:4], line[4:8], line[8:12], line[12:16]
            for hx in [h0, h1, h2, h3]:
                vals.append(hex_to_signed(hx))
    return vals


def build_conv_weights(flat_vals, shape):
    H, W, C_in, C_out = shape
    w = np.zeros(shape, dtype=int)
    idx = 0
    num_groups = math.ceil(C_out / 4)
    for g in range(num_groups):
        for h in range(H):
            for w_idx in range(W):
                for c in range(C_in):
                    for k in range(4):
                        out_c = g * 4 + k
                        if out_c < C_out and idx < len(flat_vals):
                            w[h, w_idx, c, out_c] = flat_vals[idx]
                        idx += 1
    return w


def build_dense_weights(flat_vals, shape):
    In_Dim, Out_Dim = shape
    w = np.zeros(shape, dtype=int)
    idx = 0
    num_groups = math.ceil(Out_Dim / 4)
    for g in range(num_groups):
        for in_node in range(In_Dim):
            for k in range(4):
                out_c = g * 4 + k
                if out_c < Out_Dim and idx < len(flat_vals):
                    w[in_node, out_c] = flat_vals[idx]
                idx += 1
    return w


# --- CÁC HÀM TÍNH TOÁN TOÁN HỌC ---

def conv2d(img, weights, biases):
    H, W, _ = img.shape
    KH, KW, _, C_out = weights.shape
    OUT_H, OUT_W = H - KH + 1, W - KW + 1
    out = np.zeros((OUT_H, OUT_W, C_out), dtype=int)

    for out_c in range(C_out):
        b_shifted = biases[out_c] << FRACTIONAL_BITS
        kernel = weights[:, :, :, out_c]
        for h in range(OUT_H):
            for w in range(OUT_W):
                patch = img[h:h + KH, w:w + KW, :]
                mac = np.sum(patch * kernel)
                acc = b_shifted + mac
                out[h, w, out_c] = acc >> FRACTIONAL_BITS
    return out


def dense(inputs, weights, biases):
    In_Dim, Out_Dim = weights.shape
    out = np.zeros((Out_Dim,), dtype=int)

    for out_c in range(Out_Dim):
        b_shifted = biases[out_c] << FRACTIONAL_BITS
        mac = np.sum(inputs * weights[:, out_c])
        acc = b_shifted + mac
        out[out_c] = acc >> FRACTIONAL_BITS
    return out


def maxpool2d(fmap, size=2, stride=2):
    H, W, C = fmap.shape
    OUT_H, OUT_W = H // stride, W // stride
    out = np.zeros((OUT_H, OUT_W, C), dtype=int)
    for c in range(C):
        for h in range(OUT_H):
            for w in range(OUT_W):
                patch = fmap[h * stride: h * stride + size, w * stride: w * stride + size, c]
                out[h, w, c] = np.max(patch)
    return out


def relu(x):
    return np.maximum(0, x)


# --- CÁC HÀM XUẤT DỮ LIỆU ---

def print_64bit_fmap(fmap, name):
    H, W, C = fmap.shape
    num_groups = math.ceil(C / 4)
    print(f"\n{'=' * 70}\n {name} ({H}x{W}x{C}) \n{'=' * 70}")

    for g in range(num_groups):
        print(f"\n--- Channel Group {g} (Channels {g * 4} to {g * 4 + 3}) ---")
        for h in range(H):
            row_hex = []
            for w in range(W):
                v0 = fmap[h, w, g * 4 + 0] if g * 4 + 0 < C else 0
                v1 = fmap[h, w, g * 4 + 1] if g * 4 + 1 < C else 0
                v2 = fmap[h, w, g * 4 + 2] if g * 4 + 2 < C else 0
                v3 = fmap[h, w, g * 4 + 3] if g * 4 + 3 < C else 0

                hx0, hx1, hx2, hx3 = f"{v0 & 0xFFFF:04X}", f"{v1 & 0xFFFF:04X}", f"{v2 & 0xFFFF:04X}", f"{v3 & 0xFFFF:04X}"
                row_hex.append(f"{hx3}{hx2}{hx1}{hx0}")
            print(" ".join(row_hex))


def export_fmap_to_txt(fmap, filename):
    """
    Xuất feature map ra file .txt định dạng Hex 64-bit (4 kênh/dòng)
    Phù hợp để nạp vào Verilog qua $readmemh.
    """
    H, W, C = fmap.shape
    num_groups = math.ceil(C / 4)

    with open(filename, 'w', encoding='utf-8') as f:
        for g in range(num_groups):
            for h in range(H):
                for w in range(W):
                    v0 = fmap[h, w, g * 4 + 0] if g * 4 + 0 < C else 0
                    v1 = fmap[h, w, g * 4 + 1] if g * 4 + 1 < C else 0
                    v2 = fmap[h, w, g * 4 + 2] if g * 4 + 2 < C else 0
                    v3 = fmap[h, w, g * 4 + 3] if g * 4 + 3 < C else 0

                    hx0 = f"{v0 & 0xFFFF:04X}"
                    hx1 = f"{v1 & 0xFFFF:04X}"
                    hx2 = f"{v2 & 0xFFFF:04X}"
                    hx3 = f"{v3 & 0xFFFF:04X}"

                    # Nối thành 64-bit hex (H3 H2 H1 H0)
                    f.write(f"{hx3}{hx2}{hx1}{hx0}\n")

    print(f" -> Đã lưu kết quả vào: {filename}")


# =====================================================================
# CHẠY TOÀN BỘ PIPELINE
# =====================================================================
def run_full_pipeline():
    print("=== ĐANG NẠP DỮ LIỆU TỪ CÁC FILE HEX 64-BIT ===")

    img = load_test_image("test_image.txt")

    w1_vals = load_hw_64bit_file("layer1_conv1_weight.txt")
    b1_vals = load_hw_64bit_file("layer1_conv1_bias.txt")
    w1 = build_conv_weights(w1_vals, (5, 5, 1, 4))

    w2_vals = load_hw_64bit_file("layer3_conv2_weight.txt")
    b2_vals = load_hw_64bit_file("layer3_conv2_bias.txt")
    w2 = build_conv_weights(w2_vals, (5, 5, 4, 12))

    b6_vals = load_hw_64bit_file("layer6_dense1_bias.txt")
    w6_vals = load_hw_64bit_file("layer6_dense1_weight.txt")
    l6_out_dim = len(b6_vals)
    w6 = build_dense_weights(w6_vals, (300, l6_out_dim))

    b7_vals = load_hw_64bit_file("layer7_dense2_bias.txt")
    w7_vals = load_hw_64bit_file("layer7_dense2_weight.txt")
    l7_out_dim = len(b7_vals)
    w7 = build_dense_weights(w7_vals, (l6_out_dim, l7_out_dim))

    print("=== DỮ LIỆU ĐÃ NẠP XONG. BẮT ĐẦU TÍNH TOÁN... ===\n")

    # --- TÍNH TOÁN CÁC LAYER ---
    l1_conv = conv2d(img, w1, b1_vals)
    l1_relu = relu(l1_conv)
    l1_pool = maxpool2d(l1_relu)

    l2_conv = conv2d(l1_pool, w2, b2_vals)
    l2_relu = relu(l2_conv)
    l2_pool = maxpool2d(l2_relu)

    l2_flat = l2_pool.flatten()
    l6_dense = dense(l2_flat, w6, b6_vals)
    l6_relu = relu(l6_dense)

    l7_dense = dense(l6_relu, w7, b7_vals)

    l6_dense_3d = l6_dense.reshape((1, 1, -1))
    l6_relu_3d = l6_relu.reshape((1, 1, -1))
    l7_dense_3d = l7_dense.reshape((1, 1, -1))

    # --- IN KẾT QUẢ GOLDEN MODEL LÊN MÀN HÌNH ---
    print("\n\n>>>>>>>>>> KẾT QUẢ GOLDEN MODEL CHO LAYER 1 <<<<<<<<<<")
    print_64bit_fmap(l1_conv, "LAYER 1: RAW MAC (Pre-ReLU)")
    print_64bit_fmap(l1_pool, "LAYER 1: MAX POOLING")

    print("\n\n>>>>>>>>>> KẾT QUẢ GOLDEN MODEL CHO LAYER 2 <<<<<<<<<<")
    print_64bit_fmap(l2_conv, "LAYER 2: RAW MAC (Pre-ReLU)")
    print_64bit_fmap(l2_pool, "LAYER 2: MAX POOLING")

    print("\n\n>>>>>>>>>> KẾT QUẢ GOLDEN MODEL CHO DENSE LAYERS <<<<<<<<<<")
    print_64bit_fmap(l7_dense_3d, "LAYER 7: DENSE OUTPUT (Final)")

    # --- XUẤT KẾT QUẢ RA FILE .TXT ---
    print("\n=== ĐANG XUẤT KẾT QUẢ RA FILE .TXT ===")

    # Layer 1
    export_fmap_to_txt(l1_conv, "golden_l1_conv_out.txt")
    export_fmap_to_txt(l1_relu, "golden_l1_relu_out.txt")
    export_fmap_to_txt(l1_pool, "golden_l1_pool_out.txt")

    # Layer 2
    export_fmap_to_txt(l2_conv, "golden_l2_conv_out.txt")
    export_fmap_to_txt(l2_relu, "golden_l2_relu_out.txt")
    export_fmap_to_txt(l2_pool, "golden_l2_pool_out.txt")

    # Dense Layers
    export_fmap_to_txt(l6_dense_3d, "golden_l6_dense_out.txt")
    export_fmap_to_txt(l6_relu_3d, "golden_l6_relu_out.txt")
    export_fmap_to_txt(l7_dense_3d, "golden_l7_dense_out.txt")

    print("=== HOÀN TẤT! ===")


if __name__ == "__main__":
    run_full_pipeline()