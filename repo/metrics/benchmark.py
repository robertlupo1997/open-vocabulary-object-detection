"""
Latency benchmarking for OVOD pipeline
"""
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from dataclasses import dataclass
import platform
import psutil
import subprocess

import torch
import cv2
from PIL import Image

from ovod.pipeline import OVODPipeline


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    device: str = "cuda"
    image_sizes: List[Tuple[int, int]] = None
    prompts: List[str] = None
    num_runs: int = 10
    warmup_runs: int = 3
    batch_size: int = 1
    
    # Model configurations to test
    model_configs: List[Dict] = None
    
    # Output
    results_dir: str = "metrics/benchmark_results"
    save_results: bool = True


class LatencyBenchmark:
    """Comprehensive latency benchmarking for OVOD"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        
        # Default configurations
        if self.config.image_sizes is None:
            self.config.image_sizes = [(640, 640), (1024, 1024), (1280, 1280)]
            
        if self.config.prompts is None:
            self.config.prompts = [
                "person",
                "car",
                "person, car, dog",
                "construction worker with helmet",
                "laptop, phone, coffee cup",
                "person, bicycle, car, motorcycle, airplane, bus, train, truck"  # Multi-object
            ]
            
        if self.config.model_configs is None:
            self.config.model_configs = [
                {
                    "name": "grounding_dino_sam2_small",
                    "sam2_config": "sam2_hiera_s.yaml",
                    "description": "Lightweight: Grounding DINO + SAM 2 Small"
                },
                {
                    "name": "grounding_dino_sam2_base", 
                    "sam2_config": "sam2_hiera_b+.yaml",
                    "description": "Balanced: Grounding DINO + SAM 2 Base+"
                }
            ]
    
    def get_system_info(self) -> Dict:
        """Get system information for benchmarking context"""
        
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": platform.python_version()
        }
        
        # GPU information
        if torch.cuda.is_available():
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__
            }
        else:
            info["gpu"] = None
            
        return info
    
    def create_test_images(self) -> Dict[Tuple[int, int], np.ndarray]:
        """Create test images for benchmarking"""
        
        test_images = {}
        
        for h, w in self.config.image_sizes:
            # Create realistic test image with various patterns
            image = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Add some patterns to make it realistic
            # Gradient background
            for i in range(h):
                for j in range(w):
                    image[i, j] = [
                        int(128 + 127 * np.sin(i * 0.01)),
                        int(128 + 127 * np.sin(j * 0.01)), 
                        int(128 + 127 * np.sin((i+j) * 0.01))
                    ]
            
            # Add some rectangular shapes (simulate objects)
            for _ in range(5):
                x1 = np.random.randint(0, w//2)
                y1 = np.random.randint(0, h//2)
                x2 = np.random.randint(x1, min(x1 + w//4, w))
                y2 = np.random.randint(y1, min(y1 + h//4, h))
                color = tuple(np.random.randint(0, 256, 3).tolist())
                cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
            
            test_images[(h, w)] = image
            
        return test_images
    
    def warmup_pipeline(self, pipeline: OVODPipeline, test_images: Dict):
        """Warmup the pipeline with test runs"""
        
        print(f"🔥 Warming up pipeline ({self.config.warmup_runs} runs)...")
        
        # Use smallest image for warmup
        smallest_size = min(self.config.image_sizes, key=lambda x: x[0] * x[1])
        test_image = test_images[smallest_size]
        test_prompt = self.config.prompts[0]
        
        for i in range(self.config.warmup_runs):
            _ = pipeline.predict(test_image, test_prompt, return_masks=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        print("✅ Warmup complete")
    
    def benchmark_configuration(self, 
                               model_config: Dict,
                               test_images: Dict) -> Dict:
        """Benchmark a specific model configuration"""
        
        print(f"\n📊 Benchmarking: {model_config['description']}")
        
        # Initialize pipeline
        try:
            pipeline = OVODPipeline(
                device=self.config.device,
                sam2_config=model_config.get("sam2_config", "sam2_hiera_s.yaml")
            )
        except Exception as e:
            return {"error": f"Failed to initialize pipeline: {str(e)}"}
        
        # Warmup
        self.warmup_pipeline(pipeline, test_images)
        
        results = {
            "config": model_config,
            "measurements": []
        }
        
        # Test each combination of image size and prompt
        total_tests = len(self.config.image_sizes) * len(self.config.prompts)
        test_count = 0
        
        for image_size in self.config.image_sizes:
            test_image = test_images[image_size]
            
            for prompt in self.config.prompts:
                test_count += 1
                print(f"  Testing {image_size} with '{prompt}' ({test_count}/{total_tests})")
                
                # Run multiple measurements
                times = []
                memory_usage = []
                detection_counts = []
                component_times = []
                
                for run in range(self.config.num_runs):
                    # Clear cache before each run
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Measure memory before
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                        mem_before = torch.cuda.memory_allocated()
                    
                    # Time the inference
                    start_time = time.perf_counter()
                    
                    result = pipeline.predict(
                        test_image, 
                        prompt,
                        return_masks=True,
                        max_detections=100
                    )
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    end_time = time.perf_counter()
                    
                    # Record measurements
                    inference_time = (end_time - start_time) * 1000  # Convert to ms
                    times.append(inference_time)
                    
                    if torch.cuda.is_available():
                        mem_after = torch.cuda.memory_allocated() 
                        peak_mem = torch.cuda.max_memory_allocated()
                        memory_usage.append({
                            "allocated_mb": (mem_after - mem_before) / (1024**2),
                            "peak_mb": peak_mem / (1024**2)
                        })
                    
                    detection_counts.append(len(result["boxes"]))
                    component_times.append(result["timings"])
                
                # Calculate statistics
                measurement = {
                    "image_size": image_size,
                    "prompt": prompt,
                    "num_runs": self.config.num_runs,
                    "inference_time_ms": {
                        "mean": np.mean(times),
                        "std": np.std(times),
                        "min": np.min(times),
                        "max": np.max(times),
                        "median": np.median(times),
                        "p95": np.percentile(times, 95),
                        "p99": np.percentile(times, 99)
                    },
                    "detections": {
                        "mean": np.mean(detection_counts),
                        "std": np.std(detection_counts),
                        "min": np.min(detection_counts),
                        "max": np.max(detection_counts)
                    }
                }
                
                # Add component timing statistics
                if component_times:
                    components = ["detection_ms", "segmentation_ms", "nms_ms", "prompt_processing_ms"]
                    for component in components:
                        values = [ct.get(component, 0) for ct in component_times]
                        if any(v > 0 for v in values):  # Only include if component has meaningful times
                            measurement[component] = {
                                "mean": np.mean(values),
                                "std": np.std(values),
                                "min": np.min(values),
                                "max": np.max(values)
                            }
                
                # Add memory statistics
                if memory_usage and torch.cuda.is_available():
                    allocated_mbs = [m["allocated_mb"] for m in memory_usage]
                    peak_mbs = [m["peak_mb"] for m in memory_usage]
                    
                    measurement["memory_usage_mb"] = {
                        "allocated_mean": np.mean(allocated_mbs),
                        "allocated_max": np.max(allocated_mbs),
                        "peak_mean": np.mean(peak_mbs),
                        "peak_max": np.max(peak_mbs)
                    }
                
                results["measurements"].append(measurement)
        
        # Calculate overall statistics
        results["summary"] = self._calculate_summary_stats(results["measurements"])
        
        return results
    
    def _calculate_summary_stats(self, measurements: List[Dict]) -> Dict:
        """Calculate overall summary statistics"""
        
        if not measurements:
            return {}
        
        # Aggregate across all measurements
        all_times = []
        all_detections = []
        
        by_image_size = {}
        by_prompt_complexity = {"simple": [], "complex": []}
        
        for m in measurements:
            mean_time = m["inference_time_ms"]["mean"]
            all_times.append(mean_time)
            all_detections.append(m["detections"]["mean"])
            
            # Group by image size
            size_key = f"{m['image_size'][0]}x{m['image_size'][1]}"
            if size_key not in by_image_size:
                by_image_size[size_key] = []
            by_image_size[size_key].append(mean_time)
            
            # Group by prompt complexity (simple = single object, complex = multiple objects)
            if "," in m["prompt"] or len(m["prompt"].split()) > 2:
                by_prompt_complexity["complex"].append(mean_time)
            else:
                by_prompt_complexity["simple"].append(mean_time)
        
        summary = {
            "overall": {
                "mean_inference_ms": np.mean(all_times),
                "median_inference_ms": np.median(all_times),
                "p95_inference_ms": np.percentile(all_times, 95),
                "fastest_ms": np.min(all_times),
                "slowest_ms": np.max(all_times),
                "mean_detections": np.mean(all_detections)
            },
            "by_image_size": {
                size: {
                    "mean_ms": np.mean(times),
                    "median_ms": np.median(times),
                    "min_ms": np.min(times),
                    "max_ms": np.max(times)
                }
                for size, times in by_image_size.items()
            },
            "by_prompt_complexity": {
                "simple_prompts_ms": {
                    "mean": np.mean(by_prompt_complexity["simple"]) if by_prompt_complexity["simple"] else 0,
                    "count": len(by_prompt_complexity["simple"])
                },
                "complex_prompts_ms": {
                    "mean": np.mean(by_prompt_complexity["complex"]) if by_prompt_complexity["complex"] else 0,
                    "count": len(by_prompt_complexity["complex"])
                }
            }
        }
        
        return summary
    
    def run_benchmark(self) -> Dict:
        """Run complete benchmark suite"""
        
        print("🚀 Starting OVOD Latency Benchmark")
        print("=" * 60)
        
        # Get system info
        system_info = self.get_system_info()
        print(f"🖥️  System: {system_info['platform']}")
        if system_info.get("gpu"):
            print(f"🔧 GPU: {system_info['gpu']['name']} ({system_info['gpu']['memory_gb']:.1f}GB)")
        else:
            print("🔧 Device: CPU only")
        
        # Create test images
        print(f"\n🖼️  Creating test images: {self.config.image_sizes}")
        test_images = self.create_test_images()
        
        # Run benchmarks for each configuration
        all_results = {
            "system_info": system_info,
            "config": {
                "image_sizes": self.config.image_sizes,
                "prompts": self.config.prompts,
                "num_runs": self.config.num_runs,
                "warmup_runs": self.config.warmup_runs,
                "device": self.config.device
            },
            "model_results": []
        }
        
        for model_config in self.config.model_configs:
            result = self.benchmark_configuration(model_config, test_images)
            all_results["model_results"].append(result)
        
        # Save results
        if self.config.save_results:
            self._save_results(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict):
        """Save benchmark results"""
        
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        device_name = self.config.device
        if torch.cuda.is_available() and device_name == "cuda":
            gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
            device_name = f"cuda_{gpu_name}"
        
        filename = f"benchmark_{device_name}_{timestamp}.json"
        filepath = Path(self.config.results_dir) / filename
        
        # Convert numpy types for JSON serialization
        json_results = self._convert_numpy_types(results)
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Saved benchmark results to: {filepath}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj


def print_benchmark_summary(results: Dict):
    """Print formatted benchmark summary"""
    
    print("\n" + "="*80)
    print("⚡ OVOD Latency Benchmark Summary")
    print("="*80)
    
    # System info
    system = results["system_info"]
    print(f"\n🖥️  System Information:")
    print(f"   Platform: {system['platform']}")
    print(f"   CPU: {system['processor']} ({system['cpu_count']} cores)")
    print(f"   Memory: {system['memory_gb']:.1f}GB")
    
    if system.get("gpu"):
        gpu = system["gpu"]
        print(f"   GPU: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
        print(f"   CUDA: {gpu['cuda_version']}, PyTorch: {gpu['pytorch_version']}")
    
    # Model results
    for model_result in results["model_results"]:
        if "error" in model_result:
            print(f"\n❌ {model_result['config']['name']}: {model_result['error']}")
            continue
            
        config = model_result["config"]
        summary = model_result.get("summary", {})
        
        print(f"\n📊 {config['description']}:")
        
        if "overall" in summary:
            overall = summary["overall"]
            print(f"   Overall Performance:")
            print(f"   ├── Mean latency: {overall['mean_inference_ms']:.1f}ms")
            print(f"   ├── Median latency: {overall['median_inference_ms']:.1f}ms")
            print(f"   ├── 95th percentile: {overall['p95_inference_ms']:.1f}ms")
            print(f"   ├── Fastest: {overall['fastest_ms']:.1f}ms")
            print(f"   ├── Slowest: {overall['slowest_ms']:.1f}ms")
            print(f"   └── Avg detections: {overall['mean_detections']:.1f}")
        
        if "by_image_size" in summary:
            print(f"   By Image Size:")
            for size, stats in summary["by_image_size"].items():
                print(f"   ├── {size}: {stats['mean_ms']:.1f}ms (median: {stats['median_ms']:.1f}ms)")
        
        if "by_prompt_complexity" in summary:
            complexity = summary["by_prompt_complexity"]
            simple = complexity["simple_prompts_ms"]
            complex = complexity["complex_prompts_ms"]
            print(f"   By Prompt Complexity:")
            print(f"   ├── Simple prompts: {simple['mean']:.1f}ms ({simple['count']} tests)")
            print(f"   └── Complex prompts: {complex['mean']:.1f}ms ({complex['count']} tests)")
    
    # Performance targets
    print(f"\n🎯 Performance Assessment:")
    for model_result in results["model_results"]:
        if "error" in model_result or "summary" not in model_result:
            continue
            
        config = model_result["config"]
        overall = model_result["summary"].get("overall", {})
        
        if "mean_inference_ms" in overall:
            mean_ms = overall["mean_inference_ms"]
            print(f"   {config['name']}:")
            
            # Check against targets (RTX 3070 target: ≤120ms @ 640px)
            if mean_ms <= 120:
                print(f"   ├── ✅ Meets target (≤120ms): {mean_ms:.1f}ms")
            elif mean_ms <= 200:
                print(f"   ├── ⚠️  Close to target: {mean_ms:.1f}ms")
            else:
                print(f"   ├── ❌ Above target: {mean_ms:.1f}ms")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="OVOD Latency Benchmark")
    parser.add_argument("--device", default="cuda", help="Device to benchmark")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs per test")
    parser.add_argument("--warmup-runs", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--results-dir", default="metrics/benchmark_results", help="Results directory")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark with fewer tests")
    
    args = parser.parse_args()
    
    # Configure benchmark
    config = BenchmarkConfig(
        device=args.device,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
        results_dir=args.results_dir
    )
    
    # Quick mode: fewer image sizes and prompts
    if args.quick:
        config.image_sizes = [(640, 640)]
        config.prompts = ["person", "person, car, dog"]
        config.num_runs = 5
    
    # Run benchmark
    benchmark = LatencyBenchmark(config)
    results = benchmark.run_benchmark()
    
    # Print summary
    print_benchmark_summary(results)


if __name__ == "__main__":
    main()