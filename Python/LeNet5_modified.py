#                                                   KHAI BÁO THƯ VIỆN
# tensorflow: Chứa thư viện keras sở hữu các class để xây dựng model ANN/ CNN
# numpy: Thư viện toán học chuyên về xử lý ma trận, phục vụ tính toán khi train model
# matplotlib: Thư viện phục vụ tạo biểu đồ
import os
import math
import tensorflow as tf                 # Gán tên riêng "tf" cho để dễ dàng khai báo thư viện
import numpy as np                      #               "np"
import matplotlib.pyplot as plt         #               "plt"

#                                             TIỀN XỬ LÝ DỮ LIỆU ẢNH ĐẦU VÀO
# --- 1. Khởi tạo biến mnist chứa dữ liệu hình ảnh đầu vào từ MNIST database ---
mnist = tf.keras.datasets.mnist
# Dòng lệnh mnist.load_data() tải về 4 file ảnh (tổng 70000 ảnh) và mặc định truyền vào lần lượt
# biến train nhận input train 60000 ảnh - 60000 nhãn và biến test còn lại nhận input test 10000 ảnh - 10000 nhãn
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Khởi tạo biến rows = cols = 28
rows, cols = 28, 28

# Lấy biến tạm cho y_test để sử dụng cho test thủ công
# Do y_test sẽ chuyển từ label dạng số nguyên thành vector 10 phần tử chữ số
y_test_original = y_test.copy()

# --- 2. Chuyển thành không gian 4D (Batch_Size, Height, Width, Channels) ---
# Vì dữ liệu ảnh MNIST truyền vào ở dạng (60000, 28, 28) mà LeNet-5 còn thao tác với ảnh RGB (3 channels)
# -> Phải mở rộng lên 4D cho tham số channels
x_train = x_train.reshape(x_train.shape[0], rows, cols, 1)
x_test = x_test.reshape(x_test.shape[0], rows, cols, 1)

# --- 3. Chuẩn hoá ---
# Kiểu dữ liệu gốc từ MNIST database là int nhưng model ANN/ CNN cần phải tính vi tích phân/ đạo hàm
# nhất là giai đoạn Back Propagation để train parameters -> Chuyển sang kiểu dữ liệu float
# Việc / 255.0 là nhằm chuẩn hoá phạm vi thay vì 0 -> 255 thì từ 0 -> 1 sẽ lợi ích hơn:
# +) Tránh cho ra số quá lớn khi nhân trọng số
# +) Tốc độ nhanh hơn
# +) Kết hợp với hàm ReLU
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# --- 4. Mã hoá nhãn theo kiểu Hot-One ---
# Việc mã hoá này cho người dùng quan sát model dễ dàng hơn bằng cách gán 1 cho chữ số có xác suất cao nhất
# thay vì hiện xác suất của cả 10 chữ số
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

#                                  XÂY DỰNG KIẾN TRÚC LENET-5 CNN (ĐÃ TUỲ CHỈNH)
# --- 0. Khởi tạo biến nền tảng chứa kiến trúc ---
model = tf.keras.models.Sequential()    # Sequential() là class xây dựng các layers chồng lên nhau 1 cách tuần tự

# --- 1. Input Layer ---
# Vì kiến trúc LeNet-5 nhận input 32x32, mà ảnh mẫu từ MNIST là 28x28
# => Thực hiện padding 2 pixels vào mỗi cạnh
model.add(tf.keras.layers.Input(shape=(28, 28, 1)))
model.add(tf.keras.layers.ZeroPadding2D(padding=(2, 2)))

# --- 2. Conv Layer 1 ---
# 4 filters, kernel size 5x5, stride = 1x1, no padding
# -> Parameters: (5 x 5 x 1 + 1) x 4 = 104
# Kiến trúc LeNet-5 sử dụng hàm activation là ReLU cho các Conv + Dense Layer trừ Output Layer
model.add(tf.keras.layers.Conv2D(4, (5, 5), (1, 1), 'valid', activation='relu', name='conv1'))
# 32x32x1 -> 28x28x4

# --- 3. Pooling Layer 1 ---
# Sử dụng Max Pooling size 2x2, stride = 2x2
# -> Parameters: 0 (vì bản chất Pooling chỉ là rút gọn từ ma trận có sẵn chứ không tính toán gì thêm
model.add(tf.keras.layers.MaxPooling2D((2, 2), (2, 2), name='pool1'))
# 28x28x4 -> 14x14x4

# --- 4. Conv Layer 2 ---
# 12 filters, kernel size 5x5, stride = 1x1, no padding
# -> Parameters: (5 x 5 x 4 + 1) x 12 = 1212
model.add(tf.keras.layers.Conv2D(12, (5, 5), (1, 1), 'valid', activation='relu', name='conv2'))
# 14x14x4 -> 10x10x12

# --- 5. Pooling Layer 2 ----
# Max Pooling size 2x2, stride = 2x2
# -> Parameters: 0
model.add(tf.keras.layers.MaxPooling2D((2, 2), (2, 2), name='pool2'))
# 10x10x12 -> 5x5x12

# --- 6. Flatten (làm phẳng thành vector 1D) ---
# -> Parameters: 0
model.add(tf.keras.layers.Flatten(name='flatten'))
# 5x5x12 -> 300 neurons (thoả mãn để nạp vào Dense Layer)

# --- 7. Dense Layer 1 (Fully Connected) ---
# 64 neurons
# -> Parameters: (300 + 1) x 64 = 19264
model.add(tf.keras.layers.Dense(64, activation='relu', name='dense1'))
# 300 neurons -> 64 neurons

# --- 8. Dense Layer 2 (Output Layer) ---
# 10 neurons
# -> Parameters: (64 + 1) x 10 = 650
# Hàm activation softmax cho ra phân phối xác suất từng chữ số 0 -> 9
model.add(tf.keras.layers.Dense(10, activation='softmax', name='dense2'))
# 84 neurons -> 10 neurons đại diện cho 10 chữ số
# Tổng cộng: 21230 params

#                                                   TỔNG QUAN MODEL
model.summary()

#                                                  GIAI ĐOẠN TRAINING
# --- 1. Biên dịch model ---
# Sau khi thực hiện Forward Propagation (Feed Forward)
# -> Cho ra kết quả activation từng chữ số
# -> Tính toán sai số Loss Funtion ((a_i - y_i) ^ 2 với a_i là giá trị activation, y_i = 1 khi i là chữ số cần tìm)
# -> Thực hiện Back Propagation, tìm Gradient (bằng đạo hàm chuỗi) tham số nào ảnh hưởng lớn nhất để giảm sai số Loss
# -> Tính toán vi tích phân/ đạo hàm và điều chỉnh giá trị các tham số để gradient giẩm nhanh và nhiều nhất
# -> Cập nhật parameters sau mỗi lần chạy xong 1 batch
model.compile(
    optimizer='adam',                   # Thuật toán dùng để cập nhật các trọng số (weights)
    loss='categorical_crossentropy',    # Tính toán sai số (Loss Function) = hàm categorical_crossentropy
    metrics=['accuracy']                # Độ chính xác phát hiện chữ số
)

# --- 2. Training tìm giá trị parameters ---
# Sau mỗi lần lặp training với từng dataset, lưu chúng vào history để hiển thị lên biểu đồ
history = model.fit(
    x_train, y_train,           # Dữ liệu để train và output
    epochs=5,                  # Số lần lặp training với mỗi dataset khác nhau
    batch_size=64,              # Mỗi dataset gồm 64 ảnh để train trước khi cập nhật parameters
    validation_split=0.1        # Trích ra 10% số ảnh của training data (6000 ảnh từ 60000 ảnh train) để kiểm tra trước
                                # sau mỗi lần lặp (epoch) -> Tránh hiện tượng Overfitting
)

# --- 3. Tạo biểu đồ --- (Không quan trọng)
# Biểu đồ trực quan hoá quá trình học của model
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



#                               HÀM LƯỢNG TỬ HOÁ (QUANTIZATION)
def quantize_to_int16(fp32_array, fractional_bits=10):
    # Chuyển đổi mảng số thực float32 sang số nguyên int16 (Fixed-point Q6.10)
    scale_factor = 2 ** fractional_bits     # 2^10

    # Nhân với hệ số scale và làm tròn
    scaled = np.round(fp32_array * scale_factor)

    # Cắt (Clip) giá trị để chống tràn số 16-bit có dấu [-32768, 32767]
    clipped = np.clip(scaled, -32768, 32767)

    # Ép kiểu về số nguyên int16
    return clipped.astype(np.int16)



#                                LƯU GIÁ TRỊ PARAMETERS TRAIN ĐƯỢC THÀNH FILE HEADER C
def save_model_to_header(model, filename="model_data.h", fractional_bits=10):
    # Khởi tạo file và viết ('w' = write)
    with open(filename, "w", encoding="utf-8") as f:
        # --- PHẦN 1: Ghi phần đầu của file Header C++ ---
        f.write(f"/* File trọng số đã được lượng tử hoá"
                f"(Fixed-point 16-bit, {fractional_bits} bit phân số) cho: {model.name} */\n")
        # Tạo 'Include Guard' để tránh lỗi import lặp lại trong C++
        f.write("#ifndef MODEL_DATA_H\n")
        f.write("#define MODEL_DATA_H\n\n")

        # --- PHẦN 2: Duyệt qua từng lớp (Layer) của mô hình ---
        for i, layer in enumerate(model.layers):
            # Lấy danh sách trọng số của lớp hiện tại
            weights = layer.get_weights()

            # Nếu lớp này không có trọng số (ví dụ: Flatten, Pooling) thì bỏ qua
            if len(weights) == 0:
                continue

            # Theo cài đặt của Keras: phần tử [0] là Ma trận trọng số (Kernel/Filter), [1] là Bias
            w_kernel_fp32 = weights[0]
            w_bias_fp32 = weights[1]

            # --- THỰC HIỆN QUANTIZATION TẠI ĐÂY ---
            w_kernel_q = quantize_to_int16(w_kernel_fp32, fractional_bits)
            w_bias_q = quantize_to_int16(w_bias_fp32, fractional_bits)

            # === XỬ LÝ MA TRẬN TRỌNG SỐ (WEIGHTS/KERNEL) ===

            # Duỗi phẳng mảng nhiều chiều thành mảng 1 chiều (1D Array)
            # Điều này giúp C++ truy cập bộ nhớ nhanh hơn và đơn giản hơn
            flat_w = w_kernel_q.flatten()

            # Tạo tên biến để viêt vào file C++ (ví dụ: conv2d_1_weights) với lệnh f"    "
            name_w = f"{layer.name}_weights"

            # Ghi chú thích về tên lớp và kích thước gốc để dễ theo dõi
            f.write(f"// Layer: {layer.name} | Shape gốc: {w_kernel_fp32.shape}\n")

            # Bắt đầu khai báo mảng C++: const float ten_bien[kich_thuoc] = { ...
            f.write(f"const int16_t {name_w}[{len(flat_w)}] = {{\n    ")

            # Ghi từng con số vào mảng
            for idx, val in enumerate(flat_w):
                # Ghi số thực với độ chính xác 8 số sau dấu phẩy, thêm hậu tố 'f' (ví dụ: 0.12345678f)
                f.write(f"{val}, ")

                # Cứ viết được 10 số thì xuống dòng một lần cho file dễ đọc
                if (idx + 1) % 10 == 0:
                    f.write("\n    ")

            # Đóng ngoặc mảng
            f.write("\n};\n\n")

            # === XỬ LÝ BIAS ===
            # Quy trình y hệt như trên, nhưng dành cho Bias
            flat_b = w_bias_q.flatten()
            name_b = f"{layer.name}_bias"

            f.write(f"// Layer: {layer.name} (Bias) | Shape gốc: {w_bias_fp32.shape}\n")
            f.write(f"const int16_t {name_b}[{len(flat_b)}] = {{\n    ")

            for idx, val in enumerate(flat_b):
                f.write(f"{val}, ")
                if (idx + 1) % 10 == 0:
                    f.write("\n    ")
            f.write("\n};\n\n")

        # --- PHẦN 3: Kết thúc file Header ---
        f.write("#endif // MODEL_DATA_H\n")

    print(f"Thành công! Toàn bộ tham số (đã lượng tử hoá) đã được lưu vào file: {filename}")

# --- Thực thi hàm lưu sau khi hoàn thành training ---
# save_model_to_header(model, "model_data.h")



#                       LƯU GIÁ TRỊ PARAMETERS ĐÃ TRAIN RA FILE .TXT TRỰC QUAN
def save_model_to_fpga_txt(model, output_dir="fpga_weights", to_fixed_point=True, fractional_bits=10):
    """
    Lưu trọng số mô hình ra từng file .txt riêng biệt cho từng Layer.
    Cấu trúc quét: Output Channel Group (mỗi group 4 channels) -> Spatial (H, W) -> Input Channel.
    """
    os.makedirs(output_dir, exist_ok=True)
    scale_factor = 2 ** fractional_bits if to_fixed_point else 1.0

    print(f"=== ĐANG XUẤT TRỌNG SỐ MODEL: {model.name} ===")

    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if len(weights) == 0:
            continue

        w_kernel = weights[0]
        w_bias = weights[1]

        w_filename = os.path.join(output_dir, f"layer{i}_{layer.name}_weight.txt")
        b_filename = os.path.join(output_dir, f"layer{i}_{layer.name}_bias.txt")

        shape = w_kernel.shape

        # ==========================================
        # 1. XỬ LÝ WEIGHTS
        # ==========================================
        with open(w_filename, "w", encoding='utf-8') as fw:
            # --- TRƯỜNG HỢP CONV2D ---
            if len(shape) == 4:
                H, W, C_in, C_out = shape
                num_groups = math.ceil(C_out / 4)

                for g in range(num_groups):
                    for h in range(H):
                        for w in range(W):
                            for c in range(C_in):
                                row_vals = []
                                for k in range(4):
                                    out_c = g * 4 + k
                                    val = w_kernel[h, w, c, out_c] * scale_factor if out_c < C_out else 0.0

                                    if to_fixed_point:
                                        val_int = int(np.round(val))
                                        row_vals.append(f"{val_int & 0xFFFF:04X}")
                                    else:
                                        row_vals.append(f"{val:10.6f}")

                                # Đảo ngược array để Channel 3 nằm ở MSB (trái), Channel 0 nằm ở LSB (phải)
                                # Nối không có dấu cách để tạo thành 1 chuỗi 64-bit duy nhất
                                fw.write("".join(reversed(row_vals)) + "\n")

            # --- TRƯỜNG HỢP DENSE ---
            elif len(shape) == 2:
                In_Dim, Out_Dim = shape
                num_groups = math.ceil(Out_Dim / 4)

                # ==========================================================
                # ĐẶC TRỊ CHO DENSE 1 (FLATTEN LAYER)
                # BRAM lưu dữ liệu theo dạng Khối (Channels Group -> Spatial)
                # ==========================================================
                if In_Dim == 300:  # Nhận diện Dense 1 (5x5x12 = 300)
                    H, W, C_in = 5, 5, 12
                    fpga_input_sequence = []

                    # Mô phỏng chính xác thứ tự mac_cnt (0 -> 299) của FPGA
                    for g_in in range(C_in // 4):  # Batch 0, 1, 2
                        for h in range(H):  # Spatial Y
                            for w in range(W):  # Spatial X
                                for k in range(4):  # 4 Channels trong Batch
                                    in_c = g_in * 4 + k
                                    # Công thức Flatten chuẩn của Keras (HWC)
                                    keras_node_idx = h * (W * C_in) + w * C_in + in_c
                                    fpga_input_sequence.append(keras_node_idx)
                else:
                    # Dense 2 (1D -> 1D) đọc tuyến tính bình thường
                    fpga_input_sequence = list(range(In_Dim))

                # Ghi file theo sequence đã được ánh xạ lại
                for g in range(num_groups):
                    for fpga_idx in range(In_Dim):
                        # Lấy index thực tế của Keras dựa trên tiến trình của FPGA
                        keras_node = fpga_input_sequence[fpga_idx]

                        row_vals = []
                        for k in range(4):
                            out_c = g * 4 + k
                            val = w_kernel[keras_node, out_c] * scale_factor if out_c < Out_Dim else 0.0

                            if to_fixed_point:
                                val_int = int(np.round(val))
                                row_vals.append(f"{val_int & 0xFFFF:04X}")
                            else:
                                row_vals.append(f"{val:10.6f}")

                        fw.write("".join(reversed(row_vals)) + "\n")

        # ==========================================
        # 2. XỬ LÝ BIAS
        # ==========================================
        with open(b_filename, "w", encoding='utf-8') as fb:
            num_groups = math.ceil(len(w_bias) / 4)
            for g in range(num_groups):
                row_vals = []
                for k in range(4):
                    out_c = g * 4 + k
                    val = w_bias[out_c] * scale_factor if out_c < len(w_bias) else 0.0

                    if to_fixed_point:
                        val_int = int(np.round(val))
                        row_vals.append(f"{val_int & 0xFFFF:04X}")
                    else:
                        row_vals.append(f"{val:10.6f}")

                fb.write("".join(reversed(row_vals)) + "\n")

        print(f"Đã xuất: {w_filename} | {b_filename}")

    print(f"\n✅ Hoàn tất! Toàn bộ file đã được lưu trong thư mục: '{output_dir}'")


# CÁCH GỌI HÀM:
save_model_to_fpga_txt(model, output_dir="weights_for_verilog", to_fixed_point=True)

import numpy as np


def run_test_and_export(index=0):
    print(f"\n--- ĐANG XỬ LÝ ẢNH TEST (Index: {index}) ---")

    # 1. Lấy ảnh và nhãn thực tế
    test_image = x_test[index]  # Shape gốc: (28, 28, 1)
    true_label = y_test_original[index]  # Số nguyên (ví dụ: 7)

    # FIX: Padding ảnh từ 28x28 lên 32x32 cho LeNet-5 Hardware
    # Thêm 2 pixel giá trị 0 vào các viền: Top, Bottom, Left, Right
    # Cấu trúc pad cho mảng 3D: ((top, bottom), (left, right), (channel_front, channel_back))
    padded_image = np.pad(test_image, ((2, 2), (2, 2), (0, 0)), mode='constant', constant_values=0)

    # 2. Lượng tử hoá ảnh và lưu ra file .txt (để C++ hoặc Testbench đọc)
    # Dùng padded_image (32x32) để xuất ra hardware
    quantized_image = quantize_to_int16(padded_image, fractional_bits=10)

    # Flatten thành 1024 dòng (32x32) thay vì 784 dòng
    flat_image = quantized_image.flatten()

    # Đặt tên file động theo index để tiện theo dõi nếu chạy nhiều ảnh
    image_filename = f"test_image_{index}.txt"
    with open(image_filename, "w") as f:
        for val in flat_image:
            f.write(f"{val}\n")
    print(f"-> Đã lưu ảnh lượng tử hoá 32x32 vào '{image_filename}'")

    # 3. Lưu nhãn ra file .txt (để C++ đối chiếu)
    label_filename = f"test_label_{index}.txt"
    with open(label_filename, "w") as f:
        f.write(str(true_label))
    print(f"-> Đã lưu nhãn vào '{label_filename}'")

    # 4. CHẠY DỰ ĐOÁN BẰNG PYTHON (Inference)
    # Lưu ý: Code gốc của bạn ghi Keras model nhận input (1, 28, 28, 1).
    # Nếu Keras model cũng được train với ảnh 32x32, hãy thay `test_image` bằng `padded_image` ở dòng dưới.
    # Hiện tại tôi đang giữ nguyên `test_image` để không làm crash Keras model của bạn.
    input_sample = np.expand_dims(test_image, axis=0)

    # Dự đoán
    predictions = model.predict(input_sample, verbose=0)  # Trả về mảng (1, 10)
    probabilities = predictions[0]  # Lấy vector xác suất đầu tiên

    # 5. In kết quả (Format giống C++)
    print("\n--- DỰ ĐOÁN PYTHON ---")
    best_digit = 0
    max_prob = -1.0

    for i in range(10):
        prob = probabilities[i]
        # In format % giống C++
        print(f"Digit {i}: {prob * 100:.2f}%", end="")

        if i == true_label:
            print(" <--- (Label)", end="")
        print()  # Xuống dòng

        if prob > max_prob:
            max_prob = prob
            best_digit = i

    print("\n-----------------------")
    print(f"Chữ số model đoán: {best_digit}")
    print(f"Chữ số nhãn:       {true_label}")

    if best_digit == true_label:
        print("Kết quả: ĐÚNG")
    else:
        print("Kết quả: SAI")


# --- GỌI HÀM TEST ---
for i in [45, 1209]:
    run_test_and_export(index=i)