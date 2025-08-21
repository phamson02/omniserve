MODEL=./QServe-benchmarks/Llama-2-7B
GLOBAL_BATCH_SIZE=16 NUM_RETRIEVAL_GPU_PAGE_BLOCKS=400 NUM_STREAMING_GPU_PAGE_BLOCKS=0 \
CHUNK_PREFILL_SIZE=2147000000 \
CUDA_VISIBLE_DEVICES=0 python qserve_benchmark.py --model $MODEL --benchmarking --precision w8a8 --group-size -1 --kv-quant-granularity fine_grained $common_args
