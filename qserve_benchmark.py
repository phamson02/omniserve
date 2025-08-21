# File authors: Haotian Tang, Shang Yang, Yujun Lin, Song Han
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }
# @article{yang2025lserve,
#   title={LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention},
#   author={Yang*, Shang and Guo*, Junxian and Tang, Haotian and Hu, Qinghao and Xiao, Guangxuan and Tang, Jiaming and Lin, Yujun and Liu, Zhijian and Lu, Yao and Han, Song},
#   year={2025}
# }

import argparse
import time
import gc
import torch

import omniserve.utils.constants
from omniserve import EngineArgs, LLMEngine, SamplingParams
from omniserve.config import ProfilingConfig

max_seq_len = omniserve.utils.constants.max_seq_len

import os


def process_requests(
    engine: LLMEngine, batch_size: int, prompt_len: int, generation_len: int
):
    """Continuously process a list of prompts and handle the outputs."""
    request_key = 0
    profiling_config = ProfilingConfig(
        prompt_len=prompt_len, generation_len=generation_len
    )
    for b in range(batch_size):
        engine.add_request(
            str(b),
            prompt=None,
            profiling_config=profiling_config,
            sampling_params=SamplingParams(top_p=0.5, top_k=1, temperature=0),
        )

    if engine.ifb_mode == False:
        # We need to pre-caulcate the block table size for initialization
        block_size = engine.cache_config.block_size
        tot_length = prompt_len + generation_len
        init_num_blocks = (tot_length + block_size - 1) // block_size
        engine.update_init_num_blocks(init_num_blocks)

    # seq_group_metadata_list, scheduler_outputs = engine.step()
    iter = 1

    time_lis = []
    # Track TTFT per sequence id (time from start to first observed token)
    first_token_seen = set()
    ttft_list = []
    # Track per-sequence output lengths and last token emission time for TPT
    prev_out_len = {}
    last_token_ts = {}
    per_token_latencies = []  # inter-token latencies excluding the first token
    num_tokens = 0
    torch.cuda.synchronize()
    st = time.time()

    while engine.has_unfinished_requests():
        ### Schedule iteration 1 (context stage)
        requests_outputs = engine.step()
        # torch.cuda.synchronize()
        if len(requests_outputs) == 0:
            break

        # Record TTFT when we first see output_len >= 1; also accumulate new tokens
        now = time.time()
        if isinstance(requests_outputs, list) and requests_outputs:
            first_elem = requests_outputs[0]
            # Path 1: IFB mode returns list of dicts with "id" and "tokens"
            if isinstance(first_elem, dict) and "id" in first_elem:
                for out in requests_outputs:
                    sid = out.get("id")
                    tokens = out.get("tokens", [])
                    # Derive output_len from total tokens minus known prompt_len
                    curr_out_len = max(0, len(tokens) - prompt_len)
                    # TTFT when first generated token appears
                    if sid is not None and sid not in first_token_seen and curr_out_len >= 1:
                        first_token_seen.add(sid)
                        ttft_list.append(now - st)
                        # initialize last_token_ts at first token emission; exclude first token from TPT
                        last_token_ts[sid] = now
                    # Accumulate new tokens for throughput
                    if sid is not None:
                        prev = prev_out_len.get(sid, 0)
                        delta = curr_out_len - prev
                        if delta > 0:
                            # per-token latency: measure time since last token, if we have one
                            if sid in last_token_ts:
                                per_token_latencies.append(now - last_token_ts[sid])
                            # update trackers
                            num_tokens += delta
                            prev_out_len[sid] = curr_out_len
                            last_token_ts[sid] = now
            else:
                # Path 2: Benchmarking (non-IFB) returns SequenceGroup objects.
                # Detect first generated token by checking when output_len > 0.
                for sg in requests_outputs:
                    # Duck-typing to avoid importing SequenceGroup here.
                    get_seqs = getattr(sg, "get_seqs", None)
                    if callable(get_seqs):
                        for seq in get_seqs():
                            sid = getattr(seq, "seq_id", None)
                            get_output_len = getattr(seq, "get_output_len", None)
                            if sid is None or not callable(get_output_len):
                                continue
                            curr_out_len = get_output_len()
                            if sid not in first_token_seen and curr_out_len >= 1:
                                first_token_seen.add(sid)
                                ttft_list.append(now - st)
                                # initialize last_token_ts at first token emission; exclude first token from TPT
                                last_token_ts[sid] = now
                            prev = prev_out_len.get(sid, 0)
                            delta = curr_out_len - prev
                            if delta > 0:
                                if sid in last_token_ts:
                                    per_token_latencies.append(now - last_token_ts[sid])
                                num_tokens += delta
                                prev_out_len[sid] = curr_out_len
                                last_token_ts[sid] = now

        iter += 1
        if engine.profiling_mode and iter == generation_len + 1:
            break
    torch.cuda.synchronize()
    ed = time.time()
    time_lis.append(ed - st)
    # Compute p95 of TTFT if we saw any tokens
    ttft_p95 = None
    if ttft_list:
        ttft_list.sort()
        # p95 index using nearest-rank method
        k = max(1, int(0.95 * len(ttft_list) + 0.999999)) - 1
        ttft_p95 = ttft_list[min(k, len(ttft_list) - 1)]
    # Compute p95 of per-token latency (TPT) excluding first token per seq)
    tpt_p95 = None
    if per_token_latencies:
        per_token_latencies.sort()
        k2 = max(1, int(0.95 * len(per_token_latencies) + 0.999999)) - 1
        tpt_p95 = per_token_latencies[min(k2, len(per_token_latencies) - 1)]
    return time_lis, num_tokens, ttft_p95, tpt_p95


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""

    # Default to batch_size=1 if env var is not provided
    batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE", "1"))
    prompt_len = 3072
    generation_len = 256
    rounds = 3  # measured rounds
    warmup_rounds = 1

    with open("results.csv", "a") as file:
        print("=" * 50, file=file)
        print(
            f"{args.model}: Batch={batch_size}, Input={prompt_len}, Output={generation_len}",
            file=file,
        )

    with torch.no_grad():
        # Dedicated warmup runs (not recorded)
        for w in range(warmup_rounds):
            print(f"[Warmup Round {w}]")
            engine = initialize_engine(args)
            engine.profiling_mode = True
            _ = process_requests(
                engine,
                batch_size=batch_size,
                prompt_len=prompt_len,
                generation_len=generation_len,
            )
            del engine
            torch.cuda.empty_cache()
            gc.collect()

        # Measured rounds
        for rnd in range(rounds):
            engine = initialize_engine(args)
            engine.profiling_mode = True

            # Reset CUDA peak memory stats across all visible devices
            if torch.cuda.is_available():
                for d in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(d)

            time_lis, num_tokens, ttft_p95, tpt_p95 = process_requests(
                engine,
                batch_size=batch_size,
                prompt_len=prompt_len,
                generation_len=generation_len,
            )

            # Synchronize before reading memory stats
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Collect peak memory across devices
            peak_alloc_bytes = 0
            peak_reserved_bytes = 0
            if torch.cuda.is_available():
                for d in range(torch.cuda.device_count()):
                    peak_alloc_bytes = max(peak_alloc_bytes, torch.cuda.max_memory_allocated(d))
                    peak_reserved_bytes = max(peak_reserved_bytes, torch.cuda.max_memory_reserved(d))

            del engine
            torch.cuda.empty_cache()
            gc.collect()

            throughput = num_tokens / max(1e-9, sum(time_lis))
            ttft_ms = (ttft_p95 * 1000.0) if ttft_p95 is not None else None
            tpt_ms = (tpt_p95 * 1000.0) if tpt_p95 is not None else None
            ttft_str = f"{ttft_ms:.1f} ms" if ttft_ms is not None else "N/A"
            tpt_str = f"{tpt_ms:.1f} ms" if tpt_ms is not None else "N/A"
            peak_alloc_gb = peak_alloc_bytes / (1024 ** 3)
            peak_reserved_gb = peak_reserved_bytes / (1024 ** 3)

            print(
                f"Round {rnd} Throughput: {throughput:.3f} tokens / second. | "
                f"p95 TTFT: {ttft_str} | p95 TPT: {tpt_str} | "
                f"Peak Alloc: {peak_alloc_gb:.3f} GB | Peak Reserved: {peak_reserved_gb:.3f} GB"
            )
            with open("results.csv", "a") as file:
                print(
                    f"Round {rnd} Throughput: {throughput:.3f} tokens / second.",
                    file=file,
                )
                print(
                    (
                        f"Round {rnd} p95 TTFT: {ttft_ms:.1f} ms"
                        if ttft_ms is not None
                        else f"Round {rnd} p95 TTFT: N/A"
                    ),
                    file=file,
                )
                print(
                    (
                        f"Round {rnd} p95 TPT: {tpt_ms:.1f} ms"
                        if tpt_ms is not None
                        else f"Round {rnd} p95 TPT: N/A"
                    ),
                    file=file,
                )
                print(
                    f"Round {rnd} Peak Alloc: {peak_alloc_gb:.3f} GB | Peak Reserved: {peak_reserved_gb:.3f} GB",
                    file=file,
                )

    with open("results.csv", "a") as file:
        print("=" * 50, file=file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
