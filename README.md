# LeNet-5 CNN Hardware Accelerator (Verilog)

A high-performance, pipelined hardware accelerator implementation of the **LeNet-5 Convolutional Neural Network** written in synthesizable Verilog. Designed for FPGA deployment (Kria KV260), this project features a custom Finite State Machine (FSM) controller, parallel processing elements (Quad-PE), single buffer memory interfaces, and robust on-the-fly Max Pooling.

---

## 🚀 Key Architectural Features

- **Quad-PE Processing Engine:** Accelerates convolution operations by computing multiple multiply-accumulate (MAC) operations in parallel per clock cycle.
- **Precision Control:** Fixed-point arithmetic support configurable via parameterizable fractional bits (`FRACTIONAL_BITS = 10`).
- **Optimized Dataflow & Buffering:**
  - **Weight & Bias Buffers:** Synchronized single-buffer architectures feeding weights dynamically using cycle-accurate address mapping (`channel_cnt` and `mac_cnt`).
  - **Line Buffers:** On-chip sliding window generation for efficient $5 \times 5$ convolution streaming.
  - **BRAM Management:** Intermediate feature map storage with automated dual-bank or multi-layer address routing.
- **Robust FSM Controller:** Clean separation of control and datapath logic, ensuring glitch-free enables (`en_quad_pe`) and exact pipeline synchronization without redundant latency overhead.
- **Fully Automated Testbench:** Includes a software-like DMA model (`tb_top_module.v`) that automatically streams images, weights, and biases layer-by-layer from text files (`$readmemh`) into the accelerator.

---

## 📈 Architecture Performance & Hardware Report

Based on implementation results targeting the **Kria KV260** FPGA platform, the accelerator achieves exceptional throughput, low power consumption, and highly efficient resource utilization.

### 1. Hardware Resource Utilization (Post-Implementation)
| Resource | Utilization | Available | Utilization % |
| :--- | :--- | :--- | :--- |
| **LUT** | 16,392 | 117,120 | 14.00% |
| **FF (Flip-Flops)** | 12,660 | 234,240 | 5.40% |
| **BRAM** | 13.50 | 144 | 9.38% |
| **DSP** | 8 | 1,248 | 0.64% |
| **IO** | 157 | 189 | 83.07% |

### 2. Performance Metrics & Timing ($F_{max} = 166.89\text{ MHz}$)
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Max Frequency ($F_{max}$)** | $\approx 166.89\text{ MHz}$ | Highest achieved operating clock frequency |
| **Clock Cycle ($T_{cycle}$)** | $\approx 5.992\text{ ns}$ | Period of each clock cycle |
| **Total Cycles / Image** | $\approx 79,573\text{ cycles}$ | Cycles required to process one complete digit inference |
| **Processing Time / Image** | $\approx 476.96\text{ }\mu\text{s}$ | Execution latency per image |
| **Throughput** | $\approx 2,096\text{ images/s}$ | Total classification rate |
| **Total On-Chip Power** | $0.338\text{ W}$ | Dynamic ($14\%$) + Static ($86\%$) |

---

## 🌐 Comparison with Existing Works

| Model | [1] | [2] | [3] | [4] (Rounded Mac) | **This Work** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FPGA Platform** | Zynq VC7VX485T | Zynq XCZU9EG | Artix XC7A20 | Zynq XCZU9EG | **Kria KV260** |
| **Precision** | 16-bit fixed | 16-bit floating | 8-bit fixed | 12/8-bit fixed | **16-bit fixed** |
| **Clock (MHz)** | - | 150 | 50 | 100 | **166.89** |
| **Power (W)** | 0.676 | - | 14.13 | 1.673 | **0.338** |
| **GOPs** | 20.3 | 28.8 | 164.1 | 0.141 | **1.77** |
| **LUT** | 75,221 | 25,276 | 88,756 | 61,713 | **16,392** |
| **FF** | 38,577 | 66,569 | 42,038 | 27,863 | **12,660** |
| **BRAM** | 101 | 55 | 218 | 102 | **13.5** |
| **DSP** | 406 | 204 | 571 | 123 | **8** |

---

## 📁 Repository Structure

```tree
├── rtl/
│   ├── controller.v          # Main FSM and memory control logic
│   ├── datapath.v            # Interconnection of PEs, Line Buffers, and RAMs
│   ├── quad_pe.v             # Quad Processing Element (MAC array)
│   ├── line_buffer.v         # Sliding window image buffer
│   ├── quad_max_pooling.v    # Max pooling hardware block
│   └── ...                   # Activation (ReLU), Argmax, and BRAM modules
├── sim/
│   ├── tb_top_module.v       # Comprehensive system-level testbench & DMA
│   └── data/                 # Hex-formatted weights, biases, and test images
└── README.md
```

---

## 🧠 FSM Layer Pipeline & States

The controller coordinates inference across multiple sequential layers (`conv1`, `conv2`, `dense1`, `dense2`) using a structured state sequence:

1. **`S0_IDLE`**: Awaiting the global `start` signal and resetting address pointers.
2. **`S1_LOAD_BIAS`**: Streaming layer biases into the Bias Buffer.
3. **`S2_LOAD_WEIGHT`**: Streaming kernel weights into the Weight Buffer.
4. **`S3_LOAD_LINE_BUFFER`**: Initializing the line buffer with initial pixel rows from BRAM.
5. **`S4_CONV_POOL_WR_FEATURE`**: Executing MAC operations (`mac_cnt` 0 to 24), applying bias and ReLU, and writing feature maps.
6. **`S5_SHIFT_WINDOW`**: Sliding the convolution window across the feature map.
7. **`S6_NEXT_LAYER`**: Managing layer transitions and resetting memory pointers.

---

## 📊 Getting Started & Simulation

### Prerequisites
- An HDL simulator supporting Verilog-2001 / SystemVerilog (e.g., **Xilinx Vivado**, **ModelSim**, **Icarus Verilog**, or **Verilator**).

### Running the Simulation
1. Clone the repository and add all RTL source files into your project compilation set.
2. Ensure data files (`test_image.txt`, `test_label.txt`, and layer weight/bias `.txt` files) are located in your simulation working directory.
3. Set `tb_top_module.v` as the top-level testbench.
4. Run behavioral simulation. The testbench will automatically parse hex memory files, pulse `start`, monitor execution, and output verification results to the console:
   ```text
   Test Passed: Predicted Digit = 7, Expected Label = 7
   ```

---

## 🛠️ Configuration Parameters

You can scale the accelerator architecture by modifying local parameters inside `tb_top_module.v` or top module instantiations:

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `DATA_WIDTH` | `16` | Bit-width for fixed-point data representation |
| `KERNEL_WIDTH` | `5` | Spatial dimension of convolution kernels ($5 \times 5$) |
| `NUM_PE` | `4` | Parallel processing elements per cycle |
| `FRACTIONAL_BITS` | `10` | Fixed-point binary point location |
| `BRAM_SIZE` | `196` | Internal feature map memory depth |

---

## 💡 Engineering Highlights & Lessons Learned

- **Timing Alignment:** Leveraging non-blocking assignments (`<=`) with `curr_state` instead of `next_state` for enable flags naturally absorbs the 1-cycle RAM read latency, eliminating pipeline bubbles.
- **Combinational Address Mapping:** Separating control flags (clocked registers) from address generation (combinational logic) avoids multiple-driver errors and prevents clock-cycle skew when addressing multi-channel weights.

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).