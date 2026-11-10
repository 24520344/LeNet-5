import numpy as np
from tensorflow.keras.datasets import mnist

# ==========================================
# 1. LOAD DATASET MNIST
# ==========================================
(_, _), (x_test, y_test_original) = mnist.load_data()
x_test = np.expand_dims(x_test, axis=-1)
x_test = x_test.astype('float32') / 255.0


# ==========================================
# 2. HÀM LƯỢNG TỬ HOÁ
# ==========================================
def quantize_to_int16(data, fractional_bits=10):
    scale_factor = 2 ** fractional_bits
    quantized = np.round(data * scale_factor)
    quantized = np.clip(quantized, -32768, 32767)
    return quantized.astype(np.int16)


# ==========================================
# 3. HÀM XỬ LÝ VÀ XUẤT RA FILE TXT
# ==========================================
def run_test_and_export(index=0):
    print(f"\n--- ĐANG XỬ LÝ ẢNH TEST (Index: {index}) ---")

    test_image = x_test[index]
    true_label = y_test_original[index]

    # Padding ảnh từ 28x28 lên 32x32
    padded_image = np.pad(test_image, ((2, 2), (2, 2), (0, 0)), mode='constant', constant_values=0)

    # Lượng tử hoá ảnh
    quantized_image = quantize_to_int16(padded_image, fractional_bits=10)
    flat_image = quantized_image.flatten()

    # Lưu ảnh ra file .txt dưới dạng HEX 4 ký tự (16-bit)
    image_filename = f"test_image_{index}.txt"
    with open(image_filename, "w") as f:
        for val in flat_image:
            # Mask & 0xFFFF lấy chuẩn 16-bit bù 2.
            # :04X in ra 4 ký tự Hex (mỗi ký tự 4 bit -> tổng 16 bit)
            hex_val = f"{int(val) & 0xFFFF:04X}"
            f.write(f"{hex_val}\n")

    print(f"-> Đã lưu ảnh lượng tử hoá (HEX 16-bit, 4 ký tự) vào '{image_filename}'")

    # Lưu nhãn ra file .txt
    label_filename = f"test_label_{index}.txt"
    with open(label_filename, "w") as f:
        f.write(f"{int(true_label):X}\n")

    print(f"-> Đã lưu nhãn (HEX) vào '{label_filename}'")


# ==========================================
# 4. CHẠY SCRIPT
# ==========================================
if __name__ == "__main__":
    for i in [99, 88, 77, 66, 55, 44, 33, 22, 11, 1]:
        run_test_and_export(index=i)