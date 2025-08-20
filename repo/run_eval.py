"""
Comprehensive evaluation runner for OVOD
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

from metrics.coco_eval import COCOEvaluator, EvalConfig, print_evaluation_summary
from metrics.benchmark import LatencyBenchmark, BenchmarkConfig, print_benchmark_summary


def run_coco_evaluation(args) -> Dict[str, Any]:
    """Run COCO mAP evaluation"""
    
    print("🎯 Running COCO mAP Evaluation")
    print("=" * 50)
    
    # Create evaluation config
    eval_config = EvalConfig(
        coco_path=args.coco_path,
        subset=args.subset,
        max_images=args.max_images,
        device=args.device,
        image_size=args.image_size,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_threshold=args.nms_threshold,
        results_dir=args.results_dir
    )
    
    # Run evaluation
    evaluator = COCOEvaluator(eval_config)
    results = evaluator.run_evaluation()
    
    # Print summary
    print_evaluation_summary(results)
    
    return results


def run_latency_benchmark(args) -> Dict[str, Any]:
    """Run latency benchmarking"""
    
    print("\n⚡ Running Latency Benchmark")
    print("=" * 50)
    
    # Create benchmark config
    benchmark_config = BenchmarkConfig(
        device=args.device,
        num_runs=args.benchmark_runs,
        warmup_runs=args.warmup_runs,
        results_dir=args.results_dir
    )
    
    # Quick mode for faster testing
    if args.quick:
        benchmark_config.image_sizes = [(640, 640)]
        benchmark_config.prompts = ["person", "person, car"]
        benchmark_config.num_runs = 5
    
    # Run benchmark
    benchmark = LatencyBenchmark(benchmark_config)
    results = benchmark.run_benchmark()
    
    # Print summary
    print_benchmark_summary(results)
    
    return results


def generate_final_report(coco_results: Dict, benchmark_results: Dict, args) -> Dict[str, Any]:
    """Generate comprehensive evaluation report"""
    
    report = {
        "evaluation_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": args.device,
            "image_size": args.image_size,
            "parameters": {
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
                "nms_threshold": args.nms_threshold
            }
        },
        "coco_evaluation": coco_results,
        "latency_benchmark": benchmark_results
    }
    
    # Extract key metrics for summary
    summary = {
        "device": args.device,
        "timestamp": report["evaluation_info"]["timestamp"]
    }
    
    # COCO metrics
    if "evaluation" in coco_results and "bbox" in coco_results["evaluation"]:
        bbox_metrics = coco_results["evaluation"]["bbox"]
        summary["coco_metrics"] = {
            "mAP": bbox_metrics.get("mAP", 0.0),
            "mAP_50": bbox_metrics.get("mAP_50", 0.0),
            "mAP_75": bbox_metrics.get("mAP_75", 0.0)
        }
    
    # Latency metrics  
    if "model_results" in benchmark_results and benchmark_results["model_results"]:
        model_result = benchmark_results["model_results"][0]  # First model
        if "summary" in model_result and "overall" in model_result["summary"]:
            overall = model_result["summary"]["overall"]
            summary["latency_metrics"] = {
                "mean_ms": overall.get("mean_inference_ms", 0.0),
                "p95_ms": overall.get("p95_inference_ms", 0.0),
                "fastest_ms": overall.get("fastest_ms", 0.0)
            }
    
    # Performance assessment
    assessment = assess_performance(summary, args.device)
    summary["assessment"] = assessment
    
    report["summary"] = summary
    
    return report


def assess_performance(summary: Dict, device: str) -> Dict[str, str]:
    """Assess performance against targets"""
    
    assessment = {}
    
    # mAP assessment (target: competitive with YOLOv5s)
    if "coco_metrics" in summary:
        map_score = summary["coco_metrics"].get("mAP", 0.0)
        
        if map_score >= 0.35:  # YOLOv5s typically ~0.37
            assessment["mAP"] = "✅ Excellent (≥0.35)"
        elif map_score >= 0.25:
            assessment["mAP"] = "⚠️ Good (≥0.25)" 
        elif map_score >= 0.15:
            assessment["mAP"] = "⚠️ Fair (≥0.15)"
        else:
            assessment["mAP"] = "❌ Needs improvement (<0.15)"
    
    # Latency assessment
    if "latency_metrics" in summary:
        mean_ms = summary["latency_metrics"].get("mean_ms", float('inf'))
        
        # Targets vary by device
        if "cuda" in device.lower() or "gpu" in device.lower():
            # GPU targets (RTX 3070: ≤120ms @ 640px)
            if mean_ms <= 120:
                assessment["latency"] = "✅ Excellent (≤120ms)"
            elif mean_ms <= 200:
                assessment["latency"] = "⚠️ Good (≤200ms)"
            elif mean_ms <= 400:
                assessment["latency"] = "⚠️ Acceptable (≤400ms)"
            else:
                assessment["latency"] = "❌ Too slow (>400ms)"
        else:
            # CPU targets (≤600ms @ 640px)
            if mean_ms <= 600:
                assessment["latency"] = "✅ Excellent (≤600ms)"
            elif mean_ms <= 1000:
                assessment["latency"] = "⚠️ Good (≤1000ms)"
            elif mean_ms <= 2000:
                assessment["latency"] = "⚠️ Acceptable (≤2000ms)"
            else:
                assessment["latency"] = "❌ Too slow (>2000ms)"
    
    return assessment


def print_final_report(report: Dict):
    """Print formatted final report"""
    
    print("\n" + "="*80)
    print("📊 OVOD Comprehensive Evaluation Report")
    print("="*80)
    
    info = report["evaluation_info"]
    summary = report.get("summary", {})
    
    print(f"\n📅 Evaluation Info:")
    print(f"   Timestamp: {info['timestamp']}")
    print(f"   Device: {info['device']}")
    print(f"   Image Size: {info['image_size']}px")
    print(f"   Parameters: box={info['parameters']['box_threshold']}, "
          f"text={info['parameters']['text_threshold']}, "
          f"nms={info['parameters']['nms_threshold']}")
    
    # Performance summary
    if "coco_metrics" in summary:
        coco = summary["coco_metrics"]
        print(f"\n🎯 COCO Performance:")
        print(f"   mAP@[.50:.95]: {coco['mAP']:.3f}")
        print(f"   mAP@.50:      {coco['mAP_50']:.3f}")
        print(f"   mAP@.75:      {coco['mAP_75']:.3f}")
    
    if "latency_metrics" in summary:
        latency = summary["latency_metrics"]
        print(f"\n⚡ Latency Performance:")
        print(f"   Mean:    {latency['mean_ms']:.1f}ms")
        print(f"   95th %:  {latency['p95_ms']:.1f}ms")
        print(f"   Fastest: {latency['fastest_ms']:.1f}ms")
    
    # Assessment
    if "assessment" in summary:
        assessment = summary["assessment"]
        print(f"\n📈 Performance Assessment:")
        for metric, result in assessment.items():
            print(f"   {metric.upper()}: {result}")
    
    print("\n" + "="*80)


def save_report(report: Dict, output_path: str):
    """Save evaluation report to file"""
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types for JSON serialization
    def convert_numpy_types(obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    json_report = convert_numpy_types(report)
    
    with open(output_file, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    print(f"\n💾 Saved comprehensive report to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="OVOD Comprehensive Evaluation")
    
    # General settings
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--results-dir", default="metrics/eval_results", help="Results directory")
    parser.add_argument("--output", default="metrics/eval_results/comprehensive_report.json", 
                       help="Output report file")
    
    # COCO evaluation settings
    parser.add_argument("--coco-path", default="data/coco", help="Path to COCO dataset")
    parser.add_argument("--subset", default="val2017", choices=["val2017", "test2017"])
    parser.add_argument("--max-images", type=int, default=1000, help="Max images for COCO eval")
    parser.add_argument("--image-size", type=int, default=640, help="Input image size")
    
    # Model parameters
    parser.add_argument("--box-threshold", type=float, default=0.35, help="Box confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="Text confidence threshold")
    parser.add_argument("--nms-threshold", type=float, default=0.5, help="NMS IoU threshold")
    
    # Benchmark settings
    parser.add_argument("--benchmark-runs", type=int, default=10, help="Benchmark runs per test")
    parser.add_argument("--warmup-runs", type=int, default=3, help="Warmup runs")
    
    # Mode settings
    parser.add_argument("--skip-coco", action="store_true", help="Skip COCO evaluation")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip latency benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick evaluation mode")
    
    args = parser.parse_args()
    
    print("🚀 OVOD Comprehensive Evaluation")
    print(f"📍 Device: {args.device}")
    print(f"📁 Results: {args.results_dir}")
    print(f"⚡ Quick mode: {args.quick}")
    
    # Create results directory
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    
    # Run evaluations
    coco_results = {}
    benchmark_results = {}
    
    try:
        if not args.skip_coco:
            coco_results = run_coco_evaluation(args)
        else:
            print("⏭️  Skipping COCO evaluation")
        
        if not args.skip_benchmark:
            benchmark_results = run_latency_benchmark(args)
        else:
            print("⏭️  Skipping latency benchmark")
        
        # Generate final report
        report = generate_final_report(coco_results, benchmark_results, args)
        
        # Print and save report
        print_final_report(report)
        save_report(report, args.output)
        
        print(f"\n🎉 Evaluation complete!")
        
        # Return summary for programmatic use
        return report.get("summary", {})
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        
        # Save error report
        error_report = {
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "args": vars(args)
        }
        
        error_file = Path(args.results_dir) / "error_report.json"
        with open(error_file, 'w') as f:
            json.dump(error_report, f, indent=2)
        
        raise


if __name__ == "__main__":
    main()