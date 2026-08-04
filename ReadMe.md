# llm-tuna

Bayesian optimization of vLLM serving parameters using Optuna. Searches over `max_num_batched_tokens` and `enable_cuda_graphs` to find the best config for your model, hardware, and workload — then hands you a copy-pasteable `vllm serve` command.

## How it works

1. Runs a **baseline** trial with vLLM defaults
2. Optuna's TPE sampler suggests parameter combinations across N trials
3. Each trial: spawns vLLM → waits for `/health` → runs GuideLLM → extracts metric → kills vLLM
4. Checkpoints the best config every time a new best is found (`configs/checkpoints/`)
5. Prints the final recipe and saves it to `configs/{study_name}.yaml`

Three strategies: **high_throughput** (max tokens/sec), **low_latency** (min p95 TTFT), **balanced** (max throughput with a TTFT cap).

## Install

```bash
pip install optuna pyyaml requests
# vLLM and GuideLLM must be installed separately on the target machine
pip install vllm guidellm
```

## Run

```bash
# Dry run — prints 3 sample vLLM commands without launching anything
python llm_tuna.py --config studies/high_throughput.yaml --dry-run

# Real run
python llm_tuna.py --config studies/high_throughput.yaml
```

Edit `studies/*.yaml` to match your model, GPU count, and workload shape before running.

## Example output

After 30 trials, llm-tuna prints:

```
============================================================
  llm-tuna results: qwen3_30b_throughput
  strategy: high_throughput
============================================================
  Baseline:  420.0 output_tokens_per_second
  Best:      525.0 output_tokens_per_second  (+25.0%)
  Trials:    30 total, 27 completed, 1 pruned, 2 failed

  vllm serve RedHatAI/Qwen3-30B-A3B-FP8-dynamic \
    --tensor-parallel-size 2 \
    --max-num-batched-tokens 73728 \
    --no-enable-cuda-graphs \
    --disable-log-requests

============================================================
```

Copy the recipe and run it:

```bash
vllm serve RedHatAI/Qwen3-30B-A3B-FP8-dynamic \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 73728 \
  --no-enable-cuda-graphs \
  --disable-log-requests
```

## Project layout

```
studies/                         <- pick a strategy, edit model/workload
  high_throughput.yaml
  low_latency.yaml
  balanced.yaml
configs/                         <- created at runtime
  {study_name}.yaml              <- final best result + recipe
  checkpoints/
    {study_name}_trial_{n}.yaml  <- saved each time a new best is found
llm_tuna.py                      <- all the code
```
