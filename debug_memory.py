"""Debug GPU memory usage for a vLLM model configuration.

Loads the model from a llm-tuna study YAML and prints a detailed
memory breakdown before KV cache allocation. Helps diagnose OOM
errors by showing exactly what's consuming VRAM.

Usage:
    python debug_memory.py --config studies/gemma4_31b_dspark_math.yaml
    python debug_memory.py --config studies/gemma4_31b_dspark_math.yaml --snapshot mem.pickle
"""

import argparse
import sys

import torch
import yaml


def load_study_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_llm_kwargs(config):
    kwargs = {"model": config["server"]["model"]}

    server_keys = {
        "tensor_parallel_size": int,
        "startup_timeout": None,
    }
    for key, cast in server_keys.items():
        if key in config["server"] and key != "startup_timeout":
            val = config["server"][key]
            kwargs[key] = cast(val) if cast else val

    flag_map = {
        "gpu-memory-utilization": ("gpu_memory_utilization", float),
        "max-model-len": ("max_model_len", int),
        "max-num-batched-tokens": ("max_num_batched_tokens", int),
        "max-num-seqs": ("max_num_seqs", int),
        "spec-model": ("speculative_model", str),
        "spec-method": ("speculative_method", str),
        "spec-tokens": ("num_speculative_tokens", int),
        "enforce-eager": ("enforce_eager", bool),
        "no-enable-prefix-caching": ("enable_prefix_caching", lambda v: not v),
        "trust-remote-code": ("trust_remote_code", bool),
        "compilation-config": ("compilation_config", str),
    }

    for flag, (kwarg, cast) in flag_map.items():
        val = config.get("static_params", {}).get(flag)
        if val is None:
            val = config.get("server", {}).get(flag)
        if val is not None:
            kwargs[kwarg] = cast(val) if callable(cast) else val

    return kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to llm-tuna study YAML")
    parser.add_argument("--snapshot", default=None,
                        help="Save memory snapshot to this file (visualize with "
                             "python -m torch.utils.viz._memory_viz trace_plot <file> -o mem.html)")
    args = parser.parse_args()

    config = load_study_config(args.config)

    for key, val in config.get("env_vars", {}).items():
        import os
        os.environ[key] = str(val)

    torch.cuda.memory._record_memory_history(max_entries=100000)

    kwargs = build_llm_kwargs(config)
    kwargs["enforce_eager"] = True

    print(f"Loading model with kwargs: {kwargs}\n")

    from vllm import LLM
    try:
        llm = LLM(**kwargs)
    except Exception as e:
        print(f"\nModel load failed: {e}\n")

    for i in range(torch.cuda.device_count()):
        print(f"\n{'=' * 70}")
        print(f"  GPU {i} Memory Summary")
        print(f"{'=' * 70}")
        print(torch.cuda.memory_summary(device=i))

    if args.snapshot:
        torch.cuda.memory._dump_snapshot(args.snapshot)
        print(f"\nSnapshot saved to {args.snapshot}")
        print(f"Visualize: python -m torch.utils.viz._memory_viz trace_plot {args.snapshot} -o mem.html")


if __name__ == "__main__":
    main()
