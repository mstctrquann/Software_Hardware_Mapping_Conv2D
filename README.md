# Conv2D Accelerator Dataflow Exploration

## 1. Overview

This project models the computation hardware of a Conv2D layer and evaluates the impact of different dataflow strategies on memory access patterns and execution performance.

The objective is to compare the following dataflow strategies under the same hardware architecture and workload configuration:
- **Baseline** (No Stationary)
- **Weight Stationary** (WS)
- **Input Stationary** (IS)
- **Output Stationary** (OS)

---

## 2. Workload Configuration

**Dimensions & Kernel:**
- `M = 16`, `H = 32`, `W = 32`
- `C_in = 64`
- `K_H = 3`, `K_W = 3`

**Output Dimensions:**
- `H_out` = H - K_H + 1 = **30**
- `W_out` = W - K_W + 1 = **30**

---

## 3. Hardware Architecture

### Processing Element (PE) Array
- `PE_ROWS = 16`
- `PE_COLS = 16`

**PE Internal Structure:**
```text
PE
├── Weight Register
├── Input Register
├── Accumulator Register
└── MAC Unit
```

### Memory Hierarchy
```text
DRAM
 ├── Weight Buffer (Implemented as PingPongSRAM)
 ├── Input Buffer (Implemented as PingPongSRAM)
 └── Output Buffer
```

**Ping-Pong Buffering (Double Buffering):**
To optimize execution cycles and hide DRAM latency, both the Weight Buffer and Input Buffer are physically implemented using `PingPongSRAM`. This double-buffering architecture allows the PE array to continuously compute data from `SRAM_0` while new data is pre-fetched from DRAM into `SRAM_1` (and vice versa).

### Data Path Overview

The physical architecture remains unchanged for all experiments; only the dataflow policy is modified.

- **Weight Path:** `DRAM → Weight PingPongSRAM (Prefetch) → PE Array (Compute)`
- **Input Path:** `DRAM → Input PingPongSRAM (Prefetch) → PE Array (Compute)`
- **Output Path:** `PE Array → Adder Tree → Output Buffer → DRAM`

---

### 4.1 Baseline Dataflow (No Stationary)
**Motivation:** Serves as a neutral reference point. No data type is intentionally kept stationary inside the Processing Elements. All operands are continuously streamed from SRAM buffers through the $16 \times 16$ PE array at every clock cycle.

- **Weight Path:** DRAM → Weight Bank (Load) → Swap → Weight Bank (Compute) → PE Array (Fetched whenever needed; no explicit register reuse inside the PE).
- **Input Path:** DRAM → Line Buffer → Input Bank → PE Array (Streamed continuously without local temporal reuse).
- **Output Path:** PE Array → Adder Tree → Global Accumulator → DRAM (Partial sums are immediately pushed out, reduced, and written back to memory).

**Characteristics for $16 \times 576 \times 900$ Matrix Multiplication:**
- ✅ **Advantages:** Simplest control logic (FSM), minimal local register storage inside individual PEs.
- ❌ **Disadvantages:** Extreme memory bandwidth pressure. Since $16 \times 16$ PEs process data on every cycle without local storage, the same Weight and Input elements must be re-fetched from SRAM banks multiple times, causing poor energy efficiency.

### 4.2 Weight Stationary (WS)
**Principle:** Weights are pre-loaded and locked inside the PE local registers as long as possible. Input activations are streamed across the array, and partial sums are accumulated across the spatial grid.

- **Weight Path (Configuration Phase):** DRAM → Weight Bank (Load) → Swap → Weight Bank (Compute) → PE Weight Register (A tile of weights is loaded once into the PE array and kept stationary for `H_out * W_out` cycles).
- **Input Path (Streaming Phase):** DRAM → Line Buffer → Input Bank → PE Array (The corresponding rows of Input matrix elements are streamed through the array sequentially).
- **Output Path:** PE Array → Adder Tree → Global Accumulator → DRAM (Partial sums travel horizontally/vertically through PEs, aggregated by the Adder Tree, and accumulated in the Global Accumulator across temporal tiles).

**Expected Benefit:**
- ⬇️ **Reduce:** Weight DRAM/SRAM Access (Weights are only read once per tile calculation).
- ⬆️ **Increase:** Weight Reuse Factor.
- ⚠️ **Hardware Mapping Note:** The FSM must loop $\frac{576}{16} = 36$ times to process the entire kernel length for each output channel.

### 4.3 Input Stationary (IS)
**Principle:** Input activation blocks remain stationary inside the PE registers. Weight matrices are streamed through the array, and the generated partial sums are continuously moved out to be reduced.

- **Input Path (Configuration Phase):** DRAM → Line Buffer → Input Bank → PE Input Register (A spatial tile of input pixels is pre-filled and locked inside the PE array).
- **Weight Path (Streaming Phase):** DRAM → Weight Bank (Load) → Swap → Weight Bank (Compute) → PE Array (Weight vectors for all 16 kernels are streamed across the stationary input grid).
- **Output Path:** PE Array → Adder Tree → Global Accumulator → DRAM (PEs continuously emit fragmented partial sums every cycle).

**Expected Benefit:**
- ⬇️ **Reduce:** Input DRAM Access, Input SRAM Bank Reads.
- ⬆️ **Increase:** Input Reuse Factor.
- ⚙️ **Hardware Mapping Note:** Highly efficient for a $16 \times 16$ grid since the input matrix is large. The input tile remains stationary while weights for all 16 kernels stream past.

### 4.4 Output Stationary (OS)
**Principle:** Partial sums remain locked inside the PE accumulator registers until the final output pixel value is fully computed ($3 \times 3 \times 64 = 576$ accumulation steps). This strategy completely eliminates intermediate partial sum movement.

- **Weight Path (Streaming Phase):** DRAM → Weight Bank (Load) → Swap → Weight Bank (Compute) → PE Array (Weight elements are streamed sequentially into the array).
- **Input Path (Streaming Phase):** DRAM → Line Buffer → Input Bank → PE Array (Broadcast) (Input pixels are broadcasted to match the streaming weights).
- **Output Path (Readout Phase):** PE Accumulator → Global Accumulator (Bypass/ReLU) → DRAM (Partial sums never leave the PE during the 576 calculation cycles. Only final computed outputs are read out to the buffer).

**Expected Benefit:**
- ⬇️ **Reduce:** Partial Sum SRAM Writes/Reads, Output Memory Traffic.
- ⬆️ **Increase:** Accumulator Utilization.
- ⚙️ **Hardware Mapping Note:** To calculate the $16 \times 900$ output matrix on a $16 \times 16$ PE array, the FSM allocates a row of 16 output spatial pixels to the PEs and accumulates them over 576 cycles before committing to DRAM.

## 5. Evaluation Metrics

To ensure a fair comparison, all experiments use the same physical parameters. Only the dataflow policy changes.

**Hardware & Workload:**
```python
PE_ROWS, PE_COLS = 16, 16
Clock = 200 MHz
M, H, W = 16, 32, 32
C_in = 64
Kernel = 3x3
```

### Metrics Collected

| Category | Metrics |
| :--- | :--- |
| **Compute** | `total_mac`, `compute_cycles`, `pe_utilization`, `throughput_mac_per_cycle` |
| **Load** | `dram_weight_reads`, `dram_input_reads`, `sram_weight_reads`, `sram_input_reads` |
| **Store** | `dram_output_writes`, `partial_sum_reads`, `partial_sum_writes` |
| **Reuse** | `weight_reuse_factor`, `input_reuse_factor`, `output_reuse_factor` |

---

## 6. Fair Comparison Methodology

The following parameters **MUST** remain identical across all tests:
- PE array size
- Clock frequency
- DRAM bandwidth
- SRAM capacity
- Conv2D workload dimensions

Only the mapping policy changes (`Dataflow.BASELINE`, `Dataflow.WS`, `Dataflow.IS`, `Dataflow.OS`). This ensures that any observed performance difference is caused solely by the dataflow strategy.

---

## 7. Expected Outcome

| Dataflow | Primary Reuse Target | Reduced Traffic |
| :--- | :--- | :--- |
| **Baseline** | None | None |
| **WS** | Weight | Weight Access |
| **IS** | Input Activation | Input Access |
| **OS** | Partial Sum | Output / Psum Access |

The study aims to quantify how each dataflow affects memory traffic, hardware utilization, and execution cycles for Conv2D workloads.
