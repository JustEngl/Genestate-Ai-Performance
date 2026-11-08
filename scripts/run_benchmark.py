import os
import torch
from rich.console import Console
from rich.table import Table
from genestate.engine.sdxlengine import SDXLEngine

console = Console()

def run_benchmark():
    console.print("[bold cyan]GENESTATE TENSORRT BENCHMARK HARNESS[/bold cyan]")
    
    modes = ["tensorrt"]
    results = {}
    prompt = "high-end luxury modern villa architecture, cinematic lighting"
    negative_prompt = "blurry, low quality, distorted"

    for mode in modes:
        console.print(f"\n► Running [bold yellow]{mode.upper()}[/bold yellow] Engine...")
        engine = SDXLEngine(mode=mode)
        engine.load_model()

        latencies = []
        for i in range(3):
            _, latency = engine.generate(prompt=prompt, negative_prompt=negative_prompt, seed=42 + i)
            latencies.append(latency)
            console.print(f"  run {i+1}: {latency:.2f} ms")

        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        results[mode] = {"avg": avg_latency, "min": min_latency}

        del engine
        torch.cuda.empty_cache()

    table = Table(title="TENSORRT BENCHMARK SUMMARY")
    table.add_column("Engine Mode", style="cyan")
    table.add_column("Avg Latency (ms)", justify="right")
    table.add_column("Min Latency (ms)", justify="right")

    for mode, metrics in results.items():
        table.add_row(mode, f"{metrics['avg']:.2f}", f"{metrics['min']:.2f}")

    console.print(table)

if __name__ == "__main__":
    run_benchmark()