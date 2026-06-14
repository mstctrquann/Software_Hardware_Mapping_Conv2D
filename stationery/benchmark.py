import os
import sys
import io
import time
import numpy as np
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import our 2D modules
from convolution2d_mapping_baseline import HardwareConfig2D, generate_data_2d
from conv2d_baseline import Accelerator2D_Baseline
from conv2d_ws import Accelerator2D_WS
from conv2d_is import Accelerator2D_IS
from conv2d_os import Accelerator2D_OS

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def autolabel(rects, ax, format_str='%.2e'):
    for rect in rects:
        height = rect.get_height()
        if height == 0:
            continue
        ax.annotate(format_str % height if format_str else str(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

def plot_metric(labels, values, title, ylabel, filepath, format_str='%.2e'):
    plt.figure(figsize=(8, 6))
    x = np.arange(len(labels))
    colors = ['gray', 'royalblue', 'forestgreen', 'firebrick']
    
    bars = plt.bar(x, values, color=colors, edgecolor='black')
    plt.ylabel(ylabel, fontsize=11)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(x, labels, fontsize=11)
    
    # Use log scale if values are very large and have a wide spread
    max_val = max(values) if values else 0
    min_val = min([v for v in values if v > 0]) if any(v > 0 for v in values) else 0
    if max_val > 0 and min_val > 0 and max_val / min_val > 100:
        plt.yscale('log')
        
    autolabel(bars, plt.gca(), format_str)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

def main():
    print("[BENCHMARK 2D] Starting Conv2D Dataflow Comparison...")
    
    # Create subfolders for results
    base_dir = "Results"
    dirs = {
        "compute": os.path.join(base_dir, "compute"),
        "load": os.path.join(base_dir, "load"),
        "store": os.path.join(base_dir, "store"),
        "reuse": os.path.join(base_dir, "reuse")
    }
    
    for d in dirs.values():
        ensure_dir(d)
        
    cfg = HardwareConfig2D()
    d_in, d_w = generate_data_2d(cfg)
    
    accel_base = Accelerator2D_Baseline(cfg)
    accel_ws = Accelerator2D_WS(cfg)
    accel_is = Accelerator2D_IS(cfg)
    accel_os = Accelerator2D_OS(cfg)
    
    _, stats_base = accel_base.run(d_in, d_w)
    _, stats_ws = accel_ws.run(d_in, d_w)
    _, stats_is = accel_is.run(d_in, d_w)
    _, stats_os = accel_os.run(d_in, d_w)
    
    labels = ['Baseline', 'WS', 'IS', 'OS']
    stats = [stats_base, stats_ws, stats_is, stats_os]
    
    print("\n[BENCHMARK 2D] Generating individual metric plots...")
    
    # ---------------------------------------------------------
    # 1. COMPUTE METRICS
    # ---------------------------------------------------------
    v_mac = [s.total_mac for s in stats]
    plot_metric(labels, v_mac, "Total MAC Operations", "Count", os.path.join(dirs["compute"], "total_mac.png"), "%.2e")
    
    v_cycles = [s.compute_cycles for s in stats]
    plot_metric(labels, v_cycles, "Compute Cycles", "Cycles", os.path.join(dirs["compute"], "compute_cycles.png"), "%.2e")
    
    v_util = [s.pe_utilization(cfg.PE_ROWS, cfg.PE_COLS) * 100 for s in stats]
    plot_metric(labels, v_util, "PE Utilization", "Percentage (%)", os.path.join(dirs["compute"], "pe_utilization.png"), "%.1f%%")
    
    v_tpc = [s.throughput_mac_per_cycle() for s in stats]
    plot_metric(labels, v_tpc, "Throughput (MAC per Cycle)", "MAC/Cycle", os.path.join(dirs["compute"], "throughput_mac_per_cycle.png"), "%.2f")

    # ---------------------------------------------------------
    # 2. LOAD METRICS
    # ---------------------------------------------------------
    v_dw = [s.dram_weight_reads for s in stats]
    plot_metric(labels, v_dw, "DRAM Weight Reads", "Words", os.path.join(dirs["load"], "dram_weight_reads.png"), "%.2e")
    
    v_di = [s.dram_input_reads for s in stats]
    plot_metric(labels, v_di, "DRAM Input Reads", "Words", os.path.join(dirs["load"], "dram_input_reads.png"), "%.2e")
    
    v_sw = [s.sram_weight_reads for s in stats]
    plot_metric(labels, v_sw, "SRAM Weight Reads", "Words", os.path.join(dirs["load"], "sram_weight_reads.png"), "%.2e")
    
    v_si = [s.sram_input_reads for s in stats]
    plot_metric(labels, v_si, "SRAM Input Reads", "Words", os.path.join(dirs["load"], "sram_input_reads.png"), "%.2e")

    # ---------------------------------------------------------
    # 3. STORE METRICS
    # ---------------------------------------------------------
    v_do = [s.dram_output_writes for s in stats]
    plot_metric(labels, v_do, "DRAM Output Writes", "Words", os.path.join(dirs["store"], "dram_output_writes.png"), "%.2e")
    
    v_pr = [s.partial_sum_reads for s in stats]
    plot_metric(labels, v_pr, "Partial Sum Reads (SRAM/DRAM)", "Words", os.path.join(dirs["store"], "partial_sum_reads.png"), "%.2e")
    
    v_pw = [s.partial_sum_writes for s in stats]
    plot_metric(labels, v_pw, "Partial Sum Writes (SRAM/DRAM)", "Words", os.path.join(dirs["store"], "partial_sum_writes.png"), "%.2e")

    # ---------------------------------------------------------
    # 4. REUSE METRICS
    # ---------------------------------------------------------
    v_rw = [s.weight_reuse_factor() for s in stats]
    plot_metric(labels, v_rw, "Weight Reuse Factor", "MACs / DRAM Read", os.path.join(dirs["reuse"], "weight_reuse_factor.png"), "%.1f")
    
    v_ri = [s.input_reuse_factor() for s in stats]
    plot_metric(labels, v_ri, "Input Reuse Factor", "MACs / DRAM Read", os.path.join(dirs["reuse"], "input_reuse_factor.png"), "%.1f")
    
    v_ro = [s.output_reuse_factor() for s in stats]
    plot_metric(labels, v_ro, "Output/Psum Reuse Factor", "MACs / Traffic", os.path.join(dirs["reuse"], "output_reuse_factor.png"), "%.1f")

    print("[BENCHMARK 2D] All plots saved successfully in subfolders: compute, load, store, reuse.")

if __name__ == "__main__":
    main()
