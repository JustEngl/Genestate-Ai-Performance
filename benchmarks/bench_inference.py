import os
import gc
import time
import torch
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from genestate.engine.sdxlengine import SDXLEngine

console = Console()

def cleanup_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def generate_benchmark_plot(base_lats, cuda_lats, save_path="benchmark_results.png"):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 5), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')

    runs = list(range(1, len(base_lats) + 1))
    
    ax.plot(runs, base_lats, marker='o', color='#f43f5e', linewidth=2.5, label='PyTorch Baseline (ms)')
    ax.plot(runs, cuda_lats, marker='s', color='#10b981', linewidth=2.5, label='CUDA Accelerated Engine (ms)')

    avg_base = sum(base_lats) / len(base_lats)
    avg_cuda = sum(cuda_lats) / len(cuda_lats)

    ax.axhline(avg_base, color='#f43f5e', linestyle='--', alpha=0.4)
    ax.axhline(avg_cuda, color='#10b981', linestyle='--', alpha=0.4)

    ax.set_title('GenEstate Inference Latency: Baseline vs CUDA Engine', fontsize=14, fontweight='bold', pad=15, color='#f8fafc')
    ax.set_xlabel('Run Index', fontsize=11, color='#94a3b8')
    ax.set_ylabel('Latency (ms)', fontsize=11, color='#94a3b8')
    ax.set_xticks(runs)
    ax.grid(True, linestyle=':', alpha=0.3, color='#475569')
    ax.legend(facecolor='#0f172a', edgecolor='#334155')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    console.print(f"\n[bold green]✓ Benchmark plot saved to {save_path}[/bold green]")


def run_benchmark_suite(num_runs: int = 3):
    console.print(Panel("[bold cyan]GENESTATE ENGINE BENCHMARK HARNESS[/bold cyan]\n[dim]testing baseline vs cuda accelerated engine[/dim]", style="on #0f172a"))

    prompt = "eye level exterior shot, photograph of a modern house, dark slate roof, clear sky"
    neg_prompt = "blurry, distortion, low quality"

    # 1. Base Engine Pass
    console.print("\n[bold yellow]► Running PyTorch Baseline Engine...[/bold yellow]")
    engine_base = SDXLEngine(mode="baseline")
    engine_base.load_model()
    
    base_lats = []
    for i in range(num_runs):
        _, lat = engine_base.generate(prompt, neg_prompt, seed=42 + i)
        base_lats.append(lat)
        console.print(f"  [dim]run {i+1}: {lat:.2f} ms[/dim]")

    # CRITICAL: Clean VRAM completely before second engine allocation
    del engine_base
    cleanup_vram()

    # 2. CUDA Accelerated Pass
    console.print("\n[bold green]► Running CUDA Accelerated Engine...[/bold green]")
    engine_cuda = SDXLEngine(mode="cuda_optimized")
    engine_cuda.load_model()

    cuda_lats = []
    for i in range(num_runs):
        _, lat = engine_cuda.generate(prompt, neg_prompt, seed=42 + i)
        cuda_lats.append(lat)
        console.print(f"  [dim]run {i+1}: {lat:.2f} ms[/dim]")

    del engine_cuda
    cleanup_vram()

    avg_base = sum(base_lats) / len(base_lats)
    avg_cuda = sum(cuda_lats) / len(cuda_lats)
    speedup = ((avg_base - avg_cuda) / avg_base) * 100

    # Summary Table
    table = Table(title="ENGINE BENCHMARK SUMMARY", style="bold white on #1e293b", header_style="bold magenta")
    table.add_column("Engine Mode", style="cyan")
    table.add_column("Avg Latency (ms)", justify="right", style="bold")
    table.add_column("Min Latency (ms)", justify="right")
    table.add_column("Status", justify="center")

    table.add_row("PyTorch Baseline", f"{avg_base:.2f}", f"{min(base_lats):.2f}", "[red]BASELINE[/red]")
    table.add_row("CUDA Accelerated", f"{avg_cuda:.2f}", f"{min(cuda_lats):.2f}", "[green]OPTIMIZED[/green]")

    console.print("\n")
    console.print(table)
    
    console.print(Panel(
        f"[bold green]PERFORMANCE DELTA:[/bold green] CUDA Engine is [bold yellow]{speedup:.2f}% faster[/bold yellow] ({avg_base/1000:.2f}s -> {avg_cuda/1000:.2f}s)",
        style="on #0f172a"
    ))

    generate_benchmark_plot(base_lats, cuda_lats)