import os
import time
import torch
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from genestate.engine.sdxlengine import SDXLEngine

console = Console()

def generate_benchmark_plot(base_lats, triton_lats, save_path="benchmark_results.png"):
    # dark theme plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 5), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')

    runs = list(range(1, len(base_lats) + 1))
    
    ax.plot(runs, base_lats, marker='o', color='#f43f5e', linewidth=2.5, label='PyTorch Baseline (ms)')
    ax.plot(runs, triton_lats, marker='s', color='#10b981', linewidth=2.5, label='Triton Fused Ops (ms)')

    avg_base = sum(base_lats) / len(base_lats)
    avg_triton = sum(triton_lats) / len(triton_lats)

    ax.axhline(avg_base, color='#f43f5e', linestyle='--', alpha=0.5)
    ax.axhline(avg_triton, color='#10b981', linestyle='--', alpha=0.5)

    ax.set_title('GenEstate Inference Latency: PyTorch Native vs Triton Fused', fontsize=14, fontweight='bold', pad=15, color='#f8fafc')
    ax.set_xlabel('Run Index', fontsize=11, color='#94a3b8')
    ax.set_ylabel('Latency (ms)', fontsize=11, color='#94a3b8')
    ax.set_xticks(runs)
    ax.grid(True, linestyle=':', alpha=0.3, color='#475569')
    ax.legend(facecolor='#0f172a', edgecolor='#334155')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    console.print(f"[bold green]✓ Benchmark graph saved to {save_path}[/bold green]")


def run_benchmark_suite(num_runs: int = 3):
    console.print(Panel("[bold cyan]GENESTATE ENGINE BENCHMARK HARNESS[/bold cyan]\n[dim]testing torch base vs triton fused pipeline[/dim]", style="on #0f172a"))

    prompt = "eye level exterior shot, photograph of a modern house, dark slate roof, clear sky"
    neg_prompt = "blurry, distortion, low quality"

    # 1. pytorch base pass
    console.print("\n[bold yellow]► Running PyTorch Native Engine...[/bold yellow]")
    engine_base = SDXLEngine(use_triton=False)
    engine_base.load_model()
    
    base_latencies = []
    for i in range(num_runs):
        _, lat = engine_base.generate(prompt, neg_prompt, seed=42 + i)
        base_latencies.append(lat)
        console.print(f"  [dim]run {i+1}: {lat:.2f} ms[/dim]")

    avg_base = sum(base_latencies) / len(base_latencies)

    del engine_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. triton pass
    console.print("\n[bold green]► Running Triton Fused Engine...[/bold green]")
    engine_triton = SDXLEngine(use_triton=True)
    engine_triton.load_model()

    triton_latencies = []
    for i in range(num_runs):
        _, lat = engine_triton.generate(prompt, neg_prompt, seed=42 + i)
        triton_latencies.append(lat)
        console.print(f"  [dim]run {i+1}: {lat:.2f} ms[/dim]")

    avg_triton = sum(triton_latencies) / len(triton_latencies)
    speedup = ((avg_base - avg_triton) / avg_base) * 100

    # terminal summary table
    table = Table(title="BENCHMARK RESULTS SUMMARY", style="bold white on #1e293b", header_style="bold magenta")
    table.add_column("Engine Mode", style="cyan")
    table.add_column("Avg Latency (ms)", justify="right", style="bold")
    table.add_column("Min Latency (ms)", justify="right")
    table.add_column("Status", justify="center")

    table.add_row("PyTorch Baseline", f"{avg_base:.2f}", f"{min(base_latencies):.2f}", "[red]BASELINE[/red]")
    table.add_row("Triton Accelerated", f"{avg_triton:.2f}", f"{min(triton_latencies):.2f}", "[green]OPTIMIZED[/green]")

    console.print("\n")
    console.print(table)
    
    console.print(Panel(
        f"[bold green]PERFORMANCE DELTA:[/bold green] Triton is [bold yellow]{abs(speedup):.2f}%[/bold yellow] {'faster' if speedup > 0 else 'delta registered'}",
        style="on #0f172a"
    ))

    # plot output graph
    generate_benchmark_plot(base_latencies, triton_latencies)