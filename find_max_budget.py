#!/usr/bin/env python3
"""
find_max_budget.py: Find the largest precision budget that fits in memory for a given batch size.

Usage:
  python find_max_budget.py --csv <csv> --batch-size <N> --model-args ...

This script will:
- Use config_gen.py to generate a precision map YAML for different budgets
- Run qserve_benchmark.py with the generated YAML and batch size
- Use binary search to find the largest budget that does not OOM
"""
import argparse
import subprocess
import sys
import os
import tempfile


def run_config_gen(csv_path, budget, output_yaml, extra_args=None):
    cmd = [sys.executable, 'config_gen.py', '--csv', csv_path, '--budget', str(budget), '--output', output_yaml]
    if extra_args:
        cmd += extra_args
    subprocess.check_call(cmd)


def run_benchmark(
    batch_size,
    precision_map,
    model_path,
    precision,
    group_size,
    kv_quant_granularity,
    cuda_visible_devices,
    extra_args=None,
    env=None,
):
    env = env or os.environ.copy()
    env['GLOBAL_BATCH_SIZE'] = str(batch_size)
    env['NUM_RETRIEVAL_GPU_PAGE_BLOCKS'] = str(batch_size * 25)
    env['NUM_STREAMING_GPU_PAGE_BLOCKS'] = str(0)
    env['CHUNK_PREFILL_SIZE'] = str(2147000000)
    if cuda_visible_devices is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)

    base_args = [
        '--model', model_path,
        '--benchmarking',
        '--precision', precision,
        '--precision-map', precision_map,
        '--group-size', str(group_size),
        '--kv-quant-granularity', kv_quant_granularity,
    ]
    if extra_args:
        base_args += extra_args

    cmd = [sys.executable, 'qserve_benchmark.py'] + base_args
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode, proc.stdout.decode(), proc.stderr.decode()


def is_oom(stderr):
    return 'out of memory' in stderr.lower() or 'cuda error' in stderr.lower()


def find_max_budget(
    csv_path,
    batch_size,
    model_path,
    precision,
    group_size,
    kv_quant_granularity,
    cuda_visible_devices,
    model_extra_args=None,
    extra_config_args=None,
    tol=1e-3,
):
    low, high = 0.0, 1.0
    best = 0.0
    with tempfile.TemporaryDirectory() as tmpdir:
        for _ in range(15):  # up to 15 iterations
            mid = (low + high) / 2
            yaml_path = os.path.join(tmpdir, f'precision_{mid:.4f}.yaml')
            try:
                run_config_gen(csv_path, mid, yaml_path, extra_config_args)
            except Exception as e:
                print(f"Config gen failed for budget {mid:.4f}: {e}")
                high = mid - tol
                continue
            code, out, err = run_benchmark(
                batch_size=batch_size,
                precision_map=yaml_path,
                model_path=model_path,
                precision=precision,
                group_size=group_size,
                kv_quant_granularity=kv_quant_granularity,
                cuda_visible_devices=cuda_visible_devices,
                extra_args=model_extra_args,
            )
            if code == 0 and not is_oom(err):
                best = mid
                low = mid + tol
                print(f"Budget {mid:.4f} OK")
            else:
                print(f"Budget {mid:.4f} OOM or failed")
                high = mid - tol
            if high - low < tol:
                break
    return best


def main():
    parser = argparse.ArgumentParser(description="Find max precision budget for given batch size.")
    parser.add_argument('--csv', required=True, help='CSV for config_gen.py')
    parser.add_argument('--batch-size', type=int, required=True)
    # Encoded defaults from benchmark.sh, but overridable
    parser.add_argument('--model', default='./QServe-benchmarks/Llama-2-7B', help='Model path for qserve_benchmark.py')
    parser.add_argument('--precision', default='w4a8kv8')
    parser.add_argument('--group-size', type=int, default=128)
    parser.add_argument('--kv-quant-granularity', default='fine_grained')
    parser.add_argument('--cuda-visible-devices', default='0')
    parser.add_argument('--model-extra-args', nargs='*', help='Additional args for qserve_benchmark.py')
    parser.add_argument('--config-gen-args', nargs='*', default=None, help='Extra args for config_gen.py')
    args = parser.parse_args()
    best_budget = find_max_budget(
        csv_path=args.csv,
        batch_size=args.batch_size,
        model_path=args.model,
        precision=args.precision,
        group_size=args.group_size,
        kv_quant_granularity=args.kv_quant_granularity,
        cuda_visible_devices=args.cuda_visible_devices,
        model_extra_args=args.model_extra_args,
        extra_config_args=args.config_gen_args,
    )
    print(f"Max budget for batch size {args.batch_size}: {best_budget:.4f}")

if __name__ == '__main__':
    main()
