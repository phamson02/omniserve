# original file: https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
# modified by: Haotian Tang and Shang Yang
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
import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import yaml

from omniserve.config import (
    CacheConfig,
    DeviceConfig,
    IFBConfig,
    ModelConfig,
    ParallelConfig,
    PrecisionMap,
    SchedulerConfig,
)

from omniserve.attn_config import sparse_attn_init

_STR_DTYPE_TO_TORCH_DTYPE = {
    "int8": torch.int8,
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""

    model: str
    tokenizer: Optional[str] = None
    tokenizer_mode: str = "auto"
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    load_format: str = "auto"
    dtype: str = "auto"
    kv_cache_dtype: str = "int8"
    seed: int = 0
    max_model_len: Optional[int] = None
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    max_parallel_loading_workers: Optional[int] = None
    block_size: int = 64
    swap_space: int = 4  # GiB
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: int = 262144
    max_num_seqs: int = 256
    max_paddings: int = 256
    disable_log_stats: bool = False
    revision: Optional[str] = None
    code_revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None
    enforce_eager: bool = False
    max_context_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    device: str = "cuda"
    ifb_mode: bool = False
    benchmarking: bool = False
    precision: str = "w4a8kv4"
    precision_map: Optional[str] = None
    # int4_kv: bool = False
    # kv_zp: bool = True
    quant_path: Optional[str] = None
    group_size: int = -1
    omit_prompt: bool = False
    kv_quant_granularity: Optional[str] = None  # str = "per_tensor"
    chunk_prefill_size: int = 32000
    sparse_context_mode: bool = False
    sparse_decode_mode: int = 1
    static_sparse_attn_load_dir: Optional[str] = None
    static_sparsity: float = 0.0
    ctx_sink_token: int = 128
    ctx_local_token: int = 8192
    dec_sink_token: int = 128
    dec_local_token: int = 256
    sub_chunk_per_block: int = 4
    dynamic_sparse_token_budget: int = 4096
    selector_update_interval: int = 4
    multiblock_switch: int = 2048

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Shared CLI arguments for vLLM engine."""

        # NOTE: If you update any of the arguments below, please also
        # make sure to update docs/source/models/engine_args.rst

        # Model arguments
        parser.add_argument(
            "--model",
            type=str,
            default="facebook/opt-125m",
            help="name or path of the huggingface model to use",
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            default=EngineArgs.tokenizer,
            help="name or path of the huggingface tokenizer to use",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=None,
            help="the specific model version to use. It can be a branch "
            "name, a tag name, or a commit id. If unspecified, will use "
            "the default version.",
        )
        parser.add_argument(
            "--code-revision",
            type=str,
            default=None,
            help="the specific revision to use for the model code on "
            "Hugging Face Hub. It can be a branch name, a tag name, or a "
            "commit id. If unspecified, will use the default version.",
        )
        parser.add_argument(
            "--tokenizer-revision",
            type=str,
            default=None,
            help="the specific tokenizer version to use. It can be a branch "
            "name, a tag name, or a commit id. If unspecified, will use "
            "the default version.",
        )
        parser.add_argument(
            "--tokenizer-mode",
            type=str,
            default=EngineArgs.tokenizer_mode,
            choices=["auto", "slow"],
            help='tokenizer mode. "auto" will use the fast '
            'tokenizer if available, and "slow" will '
            "always use the slow tokenizer.",
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="trust remote code from huggingface",
        )
        parser.add_argument(
            "--download-dir",
            type=str,
            default=EngineArgs.download_dir,
            help="directory to download and load the weights, "
            "default to the default cache dir of "
            "huggingface",
        )
        parser.add_argument(
            "--load-format",
            type=str,
            default=EngineArgs.load_format,
            choices=["auto", "pt", "safetensors", "npcache", "dummy"],
            help="The format of the model weights to load. "
            '"auto" will try to load the weights in the safetensors format '
            "and fall back to the pytorch bin format if safetensors format "
            "is not available. "
            '"pt" will load the weights in the pytorch bin format. '
            '"safetensors" will load the weights in the safetensors format. '
            '"npcache" will load the weights in pytorch format and store '
            "a numpy cache to speed up the loading. "
            '"dummy" will initialize the weights with random values, '
            "which is mainly for profiling.",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default=EngineArgs.dtype,
            choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
            help="data type for model weights and activations. "
            'The "auto" option will use FP16 precision '
            "for FP32 and FP16 models, and BF16 precision "
            "for BF16 models.",
        )
        parser.add_argument(
            "--kv-cache-dtype",
            type=str,
            choices=["int8", "fp16", "fp8_e5m2"],
            default=EngineArgs.kv_cache_dtype,
            help='Data type for kv cache storage. If "auto", will use model '
            "data type. Note FP8 is not supported when cuda version is "
            "lower than 11.8.",
        )
        parser.add_argument(
            "--max-model-len",
            type=int,
            default=EngineArgs.max_model_len,
            help="model context length. If unspecified, "
            "will be automatically derived from the model.",
        )
        # Parallel arguments
        parser.add_argument(
            "--pipeline-parallel-size",
            "-pp",
            type=int,
            default=EngineArgs.pipeline_parallel_size,
            help="number of pipeline stages",
        )
        parser.add_argument(
            "--tensor-parallel-size",
            "-tp",
            type=int,
            default=EngineArgs.tensor_parallel_size,
            help="number of tensor parallel replicas",
        )
        parser.add_argument(
            "--max-parallel-loading-workers",
            type=int,
            default=EngineArgs.max_parallel_loading_workers,
            help="load model sequentially in multiple batches, "
            "to avoid RAM OOM when using tensor "
            "parallel and large models",
        )
        # KV cache arguments
        parser.add_argument(
            "--block-size",
            type=int,
            default=EngineArgs.block_size,
            choices=[64],
            help="token block size",
        )
        # TODO(woosuk): Support fine-grained seeds (e.g., seed per request).
        parser.add_argument(
            "--seed", type=int, default=EngineArgs.seed, help="random seed"
        )
        parser.add_argument(
            "--swap-space",
            type=int,
            default=EngineArgs.swap_space,
            help="CPU swap space size (GiB) per GPU",
        )
        parser.add_argument(
            "--gpu-memory-utilization",
            type=float,
            default=EngineArgs.gpu_memory_utilization,
            help="the fraction of GPU memory to be used for "
            "the model executor, which can range from 0 to 1."
            "If unspecified, will use the default value of 0.9.",
        )
        parser.add_argument(
            "--max-num-batched-tokens",
            type=int,
            default=EngineArgs.max_num_batched_tokens,
            help="maximum number of batched tokens per iteration",
        )
        parser.add_argument(
            "--max-num-seqs",
            type=int,
            default=EngineArgs.max_num_seqs,
            help="maximum number of sequences per iteration",
        )
        parser.add_argument(
            "--max-paddings",
            type=int,
            default=EngineArgs.max_paddings,
            help="maximum number of paddings in a batch",
        )
        parser.add_argument(
            "--disable-log-stats",
            action="store_true",
            help="disable logging statistics",
        )
        # Quantization settings.
        parser.add_argument(
            "--quantization",
            "-q",
            type=str,
            choices=["awq", "gptq", "squeezellm", None],
            default=EngineArgs.quantization,
            help="Method used to quantize the weights. If "
            "None, we first check the `quantization_config` "
            "attribute in the model config file. If that is "
            "None, we assume the model weights are not "
            "quantized and use `dtype` to determine the data "
            "type of the weights.",
        )
        parser.add_argument(
            "--enforce-eager",
            action="store_true",
            help="Always use eager-mode PyTorch. If False, "
            "will use eager mode and CUDA graph in hybrid "
            "for maximal performance and flexibility.",
        )
        parser.add_argument(
            "--max-context-len-to-capture",
            type=int,
            default=EngineArgs.max_context_len_to_capture,
            help="maximum context length covered by CUDA "
            "graphs. When a sequence has context length "
            "larger than this, we fall back to eager mode.",
        )
        parser.add_argument(
            "--disable-custom-all-reduce",
            action="store_true",
            default=EngineArgs.disable_custom_all_reduce,
            help="See ParallelConfig",
        )
        parser.add_argument(
            "--device",
            type=str,
            default=EngineArgs.device,
            choices=["cuda"],
            help=(
                "Device type for vLLM execution. "
                "Currently, only CUDA-compatible devices are supported."
            ),
        )
        parser.add_argument(
            "--ifb-mode",
            action="store_true",
            help="Enable In-flight Batching mode.",
        )
        parser.add_argument(
            "--benchmarking",
            action="store_true",
            help="Enable Profiling mode.",
        )
        parser.add_argument(
            "--precision",
            type=str,
            default="w4a8kv4",
            help="Model precision. Select from [w4a8kv4, w4a8kv8, w8a8kv4, w8a8kv8]. If kv precision is not specified, it will be the same as the activation.",
        )
        parser.add_argument(
            "--precision-map",
            type=str,
            default=None,
            metavar="YAML",
            help="Path to the precision map YAML file.",
        )
        # parser.add_argument(
        #     "--int4-kv",
        #     action="store_true",
        #     help="Use 4-bit quantization for key-value cache",
        # )
        # parser.add_argument(
        #     "--kv-zp",
        #     action="store_true",
        #     help="Use zero-point quantization for key-value cache",
        # )
        parser.add_argument(
            "--quant-path",
            type=str,
            default=None,
            help="Path to the quantized checkpoint",
        )
        parser.add_argument(
            "--group-size",
            type=int,
            default=-1,
            help="Group size for weight quantization, -1 means per-channel",
        )
        parser.add_argument(
            "--omit-prompt",
            action="store_true",
            help="Whether to omit the prompt in the final output",
        )
        # for lserve
        parser.add_argument(
            "--kv-quant-granularity",
            type=str,
            default=EngineArgs.kv_quant_granularity,
            help="per_tensor or fine_grained (per_token + per_head)",
        )
        parser.add_argument(
            "--chunk-prefill-size",
            type=int,
            default=EngineArgs.chunk_prefill_size,
            help="Number of tokens in one chunk.",
        )
        parser.add_argument(
            "--static-sparse-attn-load-dir",
            type=str,
            default=EngineArgs.static_sparse_attn_load_dir,
            help="Directory to load static sparse attention alpha",
        )
        parser.add_argument(
            "--static-sparsity",  # todo: change this to static sparsity
            type=float,
            default=EngineArgs.static_sparsity,
            help="Sparsity of the attention pattern.",
        )
        parser.add_argument(
            "--sparse-context-mode",
            action="store_true",
            help="Enable sparse prefilling",
        )
        parser.add_argument(
            "--ctx-sink-token",
            type=int,
            default=EngineArgs.ctx_sink_token,
            help="Number of sink tokens for ctx attn.",
        )
        parser.add_argument(
            "--ctx-local-token",
            type=int,
            default=EngineArgs.ctx_local_token,
            help="Number of local tokens for ctx attn.",
        )
        parser.add_argument(
            "--dec-sink-token",
            type=int,
            default=EngineArgs.dec_sink_token,
            help="Number of sink tokens for dec attn.",
        )
        parser.add_argument(
            "--dec-local-token",
            type=int,
            default=EngineArgs.dec_local_token,
            help="Number of local tokens for dec attn.",
        )
        # for decoding
        parser.add_argument(
            "--sparse-decode-mode",  # todo: use this to replace dynamic_sparse-mode
            type=int,
            default=EngineArgs.sparse_decode_mode,
            help="Mode for sparse decoding.",
        )
        parser.add_argument(
            "--sub-chunk-per-block",  # todo: use this to replace N_SUB_CHUNK_PER_BLOCK
            type=int,
            default=EngineArgs.sub_chunk_per_block,
            help="Number of logical pages in one physical page.",
        )
        parser.add_argument(
            "--dynamic-sparse-token-budget",
            type=int,
            default=EngineArgs.dynamic_sparse_token_budget,
            help="Number of tokens reserved for dynamic sparse heads (decoding stage).",
        )
        parser.add_argument(
            "--selector-update-interval",
            type=int,
            default=EngineArgs.selector_update_interval,
            help="Number of intervals between two selecting operation.",
        )
        parser.add_argument(
            "--multiblock-switch",
            type=int,
            default=EngineArgs.multiblock_switch,
            help="The sequence length at witch using multiblock.",
        )

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "EngineArgs":
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    def parse_precision_map(self) -> PrecisionMap | None:
        """Parse the precision map from the YAML file."""
        if self.precision_map is None:
            return None

        with open(self.precision_map, "r") as f:
            precision_map = yaml.safe_load(f)
        return PrecisionMap(**precision_map)

    def create_engine_configs(
        self,
    ) -> Tuple[
        ModelConfig,
        CacheConfig,
        ParallelConfig,
        SchedulerConfig,
        DeviceConfig,
        IFBConfig,
        bool,  # benchmarking_mode
        str,  # precision
        bool,  # int4_kv
        bool,  # kv_zp
        str,  # quant_path
        int,  # group_size
        bool,  # omit_prompt
    ]:
        assert self.precision in [
            "w4a8",
            "w4a8kv4",
            "w4a8kv8",
            "w8a8",
            "w8a8kv4",
            "w8a8kv8",
            "w16a16kv8",
            "w16a16kv4",
        ], (
            f"Invalid precision {self.precision} specified. Please choose from w4a8, w4a8kv4, w4a8kv8, w8a8, w8a8kv4, w8a8kv8, w16a16kv8, w16a16kv4."
        )

        if "kv4" in self.precision:
            self.kv_cache_bits = 4
            self.int4_kv = True
        else:
            self.kv_cache_bits = 8
            self.int4_kv = False
        precision = self.precision
        precision_map = self.parse_precision_map()
        # self.kv_zp = True
        # Note (kentang): per-tensor kv8 does not have zero point.

        if self.kv_quant_granularity == "per_tensor":
            self.kv_zp = False
        elif self.kv_quant_granularity == "fine_grained":
            self.kv_zp = True
        else:
            raise NotImplementedError(
                f"Unsupported kv_quant_granularity {self.kv_quant_granularity}"
            )

        kv_zp = self.kv_zp
        int4_kv = self.int4_kv

        device_config = DeviceConfig(self.device)
        model_config = ModelConfig(
            self.model,
            self.tokenizer,
            self.tokenizer_mode,
            self.trust_remote_code,
            self.download_dir,
            self.load_format,
            self.dtype,
            self.seed,
            self.block_size,
            self.revision,
            self.code_revision,
            self.tokenizer_revision,
            self.max_model_len,
            self.quantization,
            self.enforce_eager,
            self.max_context_len_to_capture,
            self.kv_quant_granularity,
            self.chunk_prefill_size,
            self.multiblock_switch,
            precision,
            precision_map,
        )
        sp_attn_config = sparse_attn_init(
            total_num_kv_heads=model_config.get_total_num_kv_heads(),
            total_num_layers=model_config.hf_config.num_hidden_layers,
            cache_block_size=self.block_size,
            sparse_context_mode=self.sparse_context_mode,
            sparse_decode_mode=self.sparse_decode_mode,
            static_sparse_attn_load_dir=self.static_sparse_attn_load_dir,
            static_sparsity=self.static_sparsity,
            ctx_sink_token=self.ctx_sink_token,
            ctx_local_token=self.ctx_local_token,
            dec_sink_token=self.dec_sink_token,
            dec_local_token=self.dec_local_token,
            sub_chunk_per_block=self.sub_chunk_per_block,
            dynamic_sparse_token_budget=self.dynamic_sparse_token_budget,
            selector_update_interval=self.selector_update_interval,
        )
        self.kv_cache_bits = _get_dtype_size(
            _STR_DTYPE_TO_TORCH_DTYPE[self.kv_cache_dtype]
        )
        if self.int4_kv:
            # The storage of kv cache is still in int8 format, so we do not change kv_cache_dtype here
            self.kv_cache_bits = 4

        cache_config = CacheConfig(
            self.block_size,
            self.gpu_memory_utilization,
            self.swap_space,
            self.kv_cache_dtype,
            self.kv_cache_bits,
            model_config.get_sliding_window(),
        )

        # add sp_attn_config to cache_config and model_config
        model_config.sp_attn_config = sp_attn_config
        cache_config.sp_attn_config = sp_attn_config

        parallel_config = ParallelConfig(
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
            self.max_parallel_loading_workers,
            self.disable_custom_all_reduce,
        )
        scheduler_config = SchedulerConfig(
            self.max_num_batched_tokens,
            self.max_num_seqs,
            model_config.max_model_len,
            self.max_paddings,
        )
        ifb_config = IFBConfig(self.ifb_mode)
        benchmarking_mode = self.benchmarking

        quant_path = self.quant_path
        group_size = self.group_size
        omit_prompt = self.omit_prompt
        return (
            model_config,
            cache_config,
            parallel_config,
            scheduler_config,
            device_config,
            ifb_config,
            benchmarking_mode,
            precision,
            int4_kv,
            kv_zp,
            quant_path,
            group_size,
            omit_prompt,
        )


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous vLLM engine."""

    disable_log_requests: bool = False
    max_log_len: Optional[int] = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = EngineArgs.add_cli_args(parser)
        parser.add_argument(
            "--disable-log-requests",
            action="store_true",
            help="disable logging requests",
        )
        parser.add_argument(
            "--max-log-len",
            type=int,
            default=None,
            help="max number of prompt characters or prompt "
            "ID numbers being printed in log. "
            "Default: unlimited.",
        )
        return parser


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
