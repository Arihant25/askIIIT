#!/usr/bin/env python3
"""
Configuration utility for askIIIT parallel embedding processing
"""

import os
import psutil
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def detect_optimal_settings():
    """Detect optimal parallel processing settings based on system resources"""
    
    settings = {
        "cpu_cores": os.cpu_count() or 4,
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "has_cuda": torch.cuda.is_available(),
        "has_mps": torch.backends.mps.is_available(),
        "gpu_memory_gb": 0,
        "recommended_workers": 2,
        "recommended_batch_size": 4,
        "recommended_device": "cpu"
    }
    
    # GPU information
    if settings["has_cuda"]:
        try:
            settings["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)
            settings["gpu_name"] = gpu_name
            
            if settings["gpu_memory_gb"] >= 8:
                settings["recommended_device"] = "cuda"
                settings["recommended_workers"] = min(4, settings["cpu_cores"])
                settings["recommended_batch_size"] = 2
            elif settings["gpu_memory_gb"] >= 6:
                settings["recommended_device"] = "cuda"
                settings["recommended_workers"] = min(3, settings["cpu_cores"])
                settings["recommended_batch_size"] = 1
            else:
                settings["recommended_device"] = "cpu"
                settings["recommended_workers"] = min(6, settings["cpu_cores"])
                settings["recommended_batch_size"] = 8
                
        except Exception as e:
            logger.warning(f"Could not detect GPU details: {e}")
            settings["recommended_device"] = "cpu"
    
    elif settings["has_mps"]:
        settings["recommended_device"] = "mps"
        settings["recommended_workers"] = min(3, settings["cpu_cores"])
        settings["recommended_batch_size"] = 4
    
    else:
        # CPU only
        if settings["memory_gb"] >= 16:
            settings["recommended_workers"] = min(6, settings["cpu_cores"])
            settings["recommended_batch_size"] = 8
        elif settings["memory_gb"] >= 8:
            settings["recommended_workers"] = min(4, settings["cpu_cores"])
            settings["recommended_batch_size"] = 6
        else:
            settings["recommended_workers"] = min(2, settings["cpu_cores"])
            settings["recommended_batch_size"] = 4
    
    return settings

def generate_env_recommendations():
    """Generate .env file recommendations based on system capabilities"""
    
    settings = detect_optimal_settings()
    
    recommendations = f"""
# Recommended settings for your system:
# CPU Cores: {settings['cpu_cores']}
# Memory: {settings['memory_gb']:.1f} GB
# GPU: {'Yes' if settings['has_cuda'] else 'No'}
{f"# GPU Memory: {settings['gpu_memory_gb']:.1f} GB" if settings['has_cuda'] else ""}
{f"# GPU Name: {settings.get('gpu_name', 'Unknown')}" if settings['has_cuda'] else ""}

# Parallel Processing Configuration (RECOMMENDED)
FORCE_CPU_EMBEDDINGS={str(settings['recommended_device'] == 'cpu').lower()}
EMBEDDING_MAX_WORKERS={settings['recommended_workers']}
EMBEDDING_BATCH_SIZE={settings['recommended_batch_size']}

# Memory Optimization
MAX_FILE_SIZE_MB=25
CHUNK_SIZE=400
CHUNK_OVERLAP=50
CHUNK_PROCESSING_BATCH_SIZE=20
"""
    
    return recommendations, settings

def print_system_info():
    """Print system information and recommendations"""
    
    settings = detect_optimal_settings()
    
    print("🖥️  System Information:")
    print(f"   CPU Cores: {settings['cpu_cores']}")
    print(f"   Memory: {settings['memory_gb']:.1f} GB")
    print(f"   CUDA Available: {settings['has_cuda']}")
    if settings['has_cuda']:
        print(f"   GPU Memory: {settings['gpu_memory_gb']:.1f} GB")
        print(f"   GPU Name: {settings.get('gpu_name', 'Unknown')}")
    print(f"   MPS Available: {settings['has_mps']}")
    
    print("\n⚡ Recommended Parallel Processing Settings:")
    print(f"   Device: {settings['recommended_device']}")
    print(f"   Max Workers: {settings['recommended_workers']}")
    print(f"   Batch Size: {settings['recommended_batch_size']}")
    
    print("\n📝 Add these to your .env file:")
    recommendations, _ = generate_env_recommendations()
    print(recommendations)
    
    # Performance estimates
    print("\n📊 Estimated Performance:")
    baseline_time = 100  # seconds for 100 texts
    parallel_speedup = min(settings['recommended_workers'], 4)  # Realistic speedup
    
    if settings['recommended_device'] == 'cuda':
        device_speedup = 3.0
    elif settings['recommended_device'] == 'mps':
        device_speedup = 2.0
    else:
        device_speedup = 1.0
    
    estimated_time = baseline_time / (parallel_speedup * device_speedup)
    speedup_factor = baseline_time / estimated_time
    
    print(f"   Estimated speedup: {speedup_factor:.1f}x faster than single-threaded CPU")
    print(f"   Time for 100 embeddings: ~{estimated_time:.0f} seconds")

def validate_current_settings():
    """Validate current environment settings"""
    
    print("🔍 Validating current .env settings...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ No .env file found. Copy .env.example to .env first.")
        return
    
    # Load current settings
    current_workers = int(os.getenv("EMBEDDING_MAX_WORKERS", "0"))
    current_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "0"))
    force_cpu = os.getenv("FORCE_CPU_EMBEDDINGS", "false").lower() == "true"
    
    # Get recommendations
    settings = detect_optimal_settings()
    
    print(f"   Current workers: {current_workers} (recommended: {settings['recommended_workers']})")
    print(f"   Current batch size: {current_batch_size} (recommended: {settings['recommended_batch_size']})")
    print(f"   Force CPU: {force_cpu} (recommended: {settings['recommended_device'] == 'cpu'})")
    
    # Warnings
    if current_workers > settings['recommended_workers']:
        print(f"⚠️  WARNING: Too many workers ({current_workers}) may cause memory issues")
    
    if current_batch_size > settings['recommended_batch_size']:
        print(f"⚠️  WARNING: Batch size ({current_batch_size}) may be too large for your system")
    
    if force_cpu and settings['recommended_device'] != 'cpu':
        print(f"ℹ️  INFO: You're using CPU when {settings['recommended_device']} is available")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="askIIIT Parallel Processing Configuration Utility")
    parser.add_argument("--info", action="store_true", help="Show system info and recommendations")
    parser.add_argument("--validate", action="store_true", help="Validate current .env settings")
    parser.add_argument("--generate", action="store_true", help="Generate recommended .env settings")
    
    args = parser.parse_args()
    
    if args.info:
        print_system_info()
    elif args.validate:
        validate_current_settings()
    elif args.generate:
        recommendations, settings = generate_env_recommendations()
        
        output_file = ".env.recommended"
        with open(output_file, "w") as f:
            f.write(recommendations)
        
        print(f"✅ Recommendations written to {output_file}")
        print("Copy the relevant settings to your .env file")
    else:
        print("Usage:")
        print("  python config_parallel.py --info      # Show system info")
        print("  python config_parallel.py --validate  # Validate current settings")
        print("  python config_parallel.py --generate  # Generate recommended settings")

if __name__ == "__main__":
    main()
