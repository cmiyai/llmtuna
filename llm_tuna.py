"""llm-tuna: Bayesian optimization of vLLM serving parameters.

Searches over max_num_batched_tokens and enable_cuda_graphs using Optuna's
TPE sampler. For each trial, spawns a fresh vLLM server subprocess, benchmarks
it with GuideLLM, extracts the target metric, and feeds it back to Optuna.

Three strategy presets control what gets optimized:
  high_throughput — maximize median output tokens/sec
  low_latency    — minimize p95 TTFT
  balanced       — maximize throughput with a latency cap

No Ray, no database, no plugin system — just a config-driven optimize loop.

Usage:
    python llm_tuna.py --config studies/high_throughput.yaml
    python llm_tuna.py --config studies/balanced.yaml --dry-run

Output:
    configs/{study_name}.yaml              — final best result + recipe
    configs/checkpoints/{study_name}_trial_{n}.yaml — saved whenever a new best is found
"""

import argparse
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import optuna
import requests
import yaml

log = logging.getLogger("llm-tuna")

STRATEGIES = {
    "high_throughput": ("output_tokens_per_second", "maximize", "median"),
    "low_latency":     ("time_to_first_token_ms",   "minimize", "p95"),
    "balanced":        ("output_tokens_per_second", "maximize", "median"),
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    for key in ("study", "server", "benchmark", "parameters"):
        if key not in config:
            sys.exit(f"Missing required config key: {key}")
    config.setdefault("static_params", {})
    config.setdefault("env_vars", {})
    return config


def resolve_strategy(config):
    strategy = config["study"].get("strategy", "high_throughput")
    if strategy not in STRATEGIES:
        sys.exit(f"Unknown strategy: {strategy}. Choose: {', '.join(STRATEGIES)}")
    metric, direction, percentile = STRATEGIES[strategy]
    config["study"]["metric"] = metric
    config["study"]["direction"] = direction
    config["study"]["percentile"] = percentile


# ---------------------------------------------------------------------------
# Port allocation
# ---------------------------------------------------------------------------

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# vLLM server lifecycle
# ---------------------------------------------------------------------------

def build_vllm_cmd(config, trial_params, port):
    cmd = ["vllm", "serve", config["server"]["model"], "--port", str(port)]

    for key, val in config["server"].items():
        if key in ("model", "startup_timeout"):
            continue
        cmd.extend([f"--{key.replace('_', '-')}", str(val)])

    for key, val in trial_params.items():
        flag = key.replace("_", "-")
        if isinstance(val, bool):
            cmd.append(f"--{flag}" if val else f"--no-{flag}")
        else:
            cmd.extend([f"--{flag}", str(val)])

    for key, val in config.get("static_params", {}).items():
        if isinstance(val, bool):
            if val:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(val)])

    return cmd


def start_vllm(cmd, env_vars=None, log_dir=None, verbose=False):
    env = os.environ.copy()
    if env_vars:
        env.update({k: str(v) for k, v in env_vars.items()})
    if verbose:
        stderr_dest = None
    elif log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"vllm_{cmd[cmd.index('--port') + 1]}.log")
        stderr_dest = open(log_path, "w")
    else:
        stderr_dest = subprocess.DEVNULL
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.DEVNULL, stderr=stderr_dest,
        start_new_session=True,
    )
    proc._log_path = (log_dir and not verbose) and log_path or None
    return proc


def wait_healthy(port, proc, timeout=300, interval=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            if requests.get(f"http://localhost:{port}/health", timeout=2).status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(interval)
    return False


def kill_vllm(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def run_guidellm(config, port):
    bench = config["benchmark"]
    data = bench["data"]
    max_seconds = bench.get("max_seconds", 120)
    rate_type = bench.get("rate_type", "concurrent")
    rate = bench.get("rate", 10)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name

    profile_key = "streams" if rate_type == "concurrent" else "rate"

    cmd = [
        "guidellm", "run",
        "--backend", f"kind=openai_http,target=http://localhost:{port}/v1,model={config['server']['model']}",
        "--data", f"kind=synthetic_text,prompt_tokens={data['prompt_tokens']},output_tokens={data['output_tokens']}",
        "--profile", f"kind={rate_type},{profile_key}={rate}",
        "--constraint", f"kind=max_duration,seconds={max_seconds}",
        "--output", f"kind=json,path={output_path}",
    ]

    try:
        timeout = bench.get("max_seconds", 120) + 120
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"GuideLLM exit {result.returncode}: {result.stderr[:500]}"
            )
        with open(output_path) as f:
            return json.load(f)
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def extract_metric(results, metric, percentile):
    benchmarks = results.get("benchmarks", [])
    if not benchmarks:
        raise KeyError("No benchmarks in GuideLLM output")
    bench = benchmarks[0]

    stats = None
    if "metrics" in bench and metric in bench["metrics"]:
        stats = bench["metrics"][metric]
    elif metric in bench:
        stats = bench[metric]
    else:
        for val in bench.values():
            if isinstance(val, dict) and metric in val:
                stats = val[metric]
                break

    if stats is None:
        available = list(bench.get("metrics", bench).keys())
        raise KeyError(f"Metric '{metric}' not found. Available: {available}")

    if isinstance(stats, (int, float)):
        return float(stats)
    if percentile not in stats:
        raise KeyError(
            f"Percentile '{percentile}' not in {metric}. "
            f"Available: {list(stats.keys())}"
        )
    return float(stats[percentile])


# ---------------------------------------------------------------------------
# Optuna integration
# ---------------------------------------------------------------------------

def suggest_params(trial, config):
    params = {}
    for name, spec in config["parameters"].items():
        if isinstance(spec, bool):
            params[name] = trial.suggest_categorical(name, [True, False])
        elif isinstance(spec, dict):
            params[name] = trial.suggest_int(
                name, spec["min"], spec["max"], step=spec.get("step", 1)
            )
    return params


def check_latency_cap(results, config):
    cap = config["study"].get("latency_cap_ms")
    if cap is None:
        return True
    try:
        return extract_metric(results, "time_to_first_token_ms", "p95") <= cap
    except KeyError:
        return True


def _execute_trial(config, trial_params, log_dir=None, verbose=False):
    port = find_free_port()
    cmd = build_vllm_cmd(config, trial_params, port)
    log.info("vllm cmd: %s", " ".join(cmd))
    proc = start_vllm(cmd, config.get("env_vars"), log_dir=log_dir, verbose=verbose)
    try:
        timeout = config["server"].get("startup_timeout", 300)
        if not wait_healthy(port, proc, timeout):
            hint = ""
            if proc._log_path and os.path.exists(proc._log_path):
                with open(proc._log_path) as f:
                    tail = f.read()[-1000:]
                hint = f"\nvLLM stderr (last 1000 chars):\n{tail}"
            raise RuntimeError(f"vLLM not healthy after {timeout}s{hint}")
        return run_guidellm(config, port)
    finally:
        kill_vllm(proc)


def run_baseline(config, log_dir=None, verbose=False):
    log.info("Running baseline (vLLM defaults)...")
    results = _execute_trial(config, {}, log_dir=log_dir, verbose=verbose)
    val = extract_metric(
        results, config["study"]["metric"], config["study"]["percentile"]
    )
    log.info("Baseline %s = %.2f", config["study"]["metric"], val)
    return val


def objective(trial, config, baseline_metric, log_dir=None, verbose=False):
    params = suggest_params(trial, config)
    log.info("Trial %d: %s", trial.number, params)

    try:
        results = _execute_trial(config, params, log_dir=log_dir, verbose=verbose)
    except RuntimeError as e:
        log.warning("Trial %d failed: %s", trial.number, e)
        raise optuna.TrialPruned(str(e))

    if config["study"]["strategy"] == "balanced":
        if not check_latency_cap(results, config):
            log.info("Trial %d pruned: TTFT exceeded cap", trial.number)
            raise optuna.TrialPruned("p95 TTFT exceeded latency cap")

    val = extract_metric(
        results, config["study"]["metric"], config["study"]["percentile"]
    )
    delta = ((val - baseline_metric) / baseline_metric) * 100 if baseline_metric else 0
    log.info("Trial %d: %.2f (%+.1f%% vs baseline)", trial.number, val, delta)
    return val


# ---------------------------------------------------------------------------
# Output — configs/ and configs/checkpoints/
# ---------------------------------------------------------------------------

def build_recipe(config, best_params):
    lines = [f"vllm serve {config['server']['model']}"]

    for key, val in config["server"].items():
        if key in ("model", "startup_timeout"):
            continue
        lines.append(f"  --{key.replace('_', '-')} {val}")

    for key, val in best_params.items():
        flag = key.replace("_", "-")
        if isinstance(val, bool):
            lines.append(f"  --{flag}" if val else f"  --no-{flag}")
        else:
            lines.append(f"  --{flag} {val}")

    for key, val in config.get("static_params", {}).items():
        if isinstance(val, bool):
            if val:
                lines.append(f"  --{key}")
        else:
            lines.append(f"  --{key} {val}")

    return " \\\n".join(lines)


def save_checkpoint(config, study, checkpoint_dir):
    """Write configs/checkpoints/{name}_trial_{n}.yaml whenever a new best is found."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    best = study.best_trial
    path = os.path.join(
        checkpoint_dir,
        f"{config['study']['name']}_trial_{best.number}.yaml",
    )

    checkpoint = {
        "trial_number": best.number,
        "best_params": dict(best.params),
        "metric_value": round(best.value, 2),
        "metric": config["study"]["metric"],
        "recipe": build_recipe(config, best.params),
    }

    with open(path, "w") as f:
        yaml.dump(checkpoint, f, default_flow_style=False, sort_keys=False)
    log.info("Checkpoint saved: %s", path)


def save_result(config, study, baseline_metric, output_dir):
    """Write configs/{name}.yaml with the final best result."""
    os.makedirs(output_dir, exist_ok=True)
    best = study.best_trial
    delta = (
        ((best.value - baseline_metric) / baseline_metric) * 100
        if baseline_metric else 0
    )
    completed = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    pruned = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    )

    result = {
        "study": {
            "name": config["study"]["name"],
            "strategy": config["study"]["strategy"],
            "metric": config["study"]["metric"],
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        },
        "best_params": dict(best.params),
        "results": {
            "baseline": round(baseline_metric, 2) if baseline_metric else None,
            "best": round(best.value, 2),
            "improvement": f"{delta:+.1f}%",
            "trials_total": len(study.trials),
            "trials_completed": completed,
            "trials_pruned": pruned,
            "trials_failed": len(study.trials) - completed - pruned,
        },
        "recipe": build_recipe(config, best.params),
    }

    path = os.path.join(output_dir, f"{config['study']['name']}.yaml")
    with open(path, "w") as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False)
    log.info("Results saved to %s", path)


def make_checkpoint_callback(config, checkpoint_dir):
    best_so_far = [None]

    def callback(study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        if best_so_far[0] is None or study.best_trial.number == trial.number:
            if best_so_far[0] != study.best_value:
                best_so_far[0] = study.best_value
                save_checkpoint(config, study, checkpoint_dir)

    return callback


def print_results(study, config, baseline_metric):
    best = study.best_trial
    strategy = config["study"]["strategy"]
    metric = config["study"]["metric"]
    delta = (
        ((best.value - baseline_metric) / baseline_metric) * 100
        if baseline_metric else 0
    )
    completed = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    pruned = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    )
    failed = len(study.trials) - completed - pruned

    cap_info = ""
    if strategy == "balanced":
        cap_info = f" (latency cap: {config['study'].get('latency_cap_ms', '?')}ms)"

    print()
    print("=" * 60)
    print(f"  llm-tuna results: {config['study']['name']}")
    print(f"  strategy: {strategy}{cap_info}")
    print("=" * 60)
    if baseline_metric:
        print(f"  Baseline:  {baseline_metric:.1f} {metric}")
    print(f"  Best:      {best.value:.1f} {metric}  ({delta:+.1f}%)")
    print(
        f"  Trials:    {len(study.trials)} total, "
        f"{completed} completed, {pruned} pruned, {failed} failed"
    )
    print()
    print(f"  {build_recipe(config, best.params)}")
    print()
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="llm-tuna: optimize vLLM serving parameters"
    )
    parser.add_argument("--config", required=True, help="Study config YAML")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sample vLLM commands without running",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Results directory (default: configs/ next to this script)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Stream vLLM stderr to terminal (see startup progress)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)
    resolve_strategy(config)

    base = Path(__file__).parent
    output_dir = args.output_dir or str(base / "configs")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    vllm_log_dir = os.path.join(output_dir, "logs")

    if args.dry_run:
        study = optuna.create_study(direction=config["study"]["direction"])
        print(f"\nStrategy: {config['study']['strategy']}")
        print(f"Metric:   {config['study']['metric']} ({config['study']['direction']})\n")
        for i in range(3):
            trial = study.ask()
            params = suggest_params(trial, config)
            cmd = build_vllm_cmd(config, params, 8000 + i)
            print(f"Trial {i}: {' '.join(cmd)}\n")
            study.tell(trial, 0.0)
        return

    baseline = run_baseline(config, log_dir=vllm_log_dir, verbose=args.verbose)

    study = optuna.create_study(
        study_name=config["study"]["name"],
        direction=config["study"]["direction"],
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=config["study"].get("n_startup_trials", 5),
        ),
    )

    try:
        study.optimize(
            partial(objective, config=config, baseline_metric=baseline, log_dir=vllm_log_dir, verbose=args.verbose),
            n_trials=config["study"].get("n_trials", 30),
            callbacks=[make_checkpoint_callback(config, checkpoint_dir)],
        )
    except KeyboardInterrupt:
        log.info("Interrupted — saving best result so far...")

    has_completed = any(
        t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
    )
    if has_completed:
        save_result(config, study, baseline, output_dir)
        print_results(study, config, baseline)
    else:
        log.warning("No completed trials.")


if __name__ == "__main__":
    main()
