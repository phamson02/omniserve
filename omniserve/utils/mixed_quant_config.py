from omniserve.utils.quant_config import QServeQuantConfig


class MixedQuantConfig:
    """
    Delegates every query to the *right* QServeQuantConfig
    according to the suffix in the tensor name.
    """

    def __init__(self):
        self.cfg4 = QServeQuantConfig(weight_bits=4)  # knows about qweight / s1_ / s2_
        self.cfg8 = QServeQuantConfig(
            weight_bits=8
        )  # knows about weight / dequant_scale

        # pre-compute the unions once
        self._col_suffixes = list(
            set(
                self.cfg4.get_col_parallel_tensor_names()
                + self.cfg8.get_col_parallel_tensor_names()
            )
        )
        self._row_suffixes = list(
            set(
                self.cfg4.get_row_parallel_tensor_names()
                + self.cfg8.get_row_parallel_tensor_names()
            )
        )

    def get_col_parallel_tensor_names(self):
        return self._col_suffixes

    def get_row_parallel_tensor_names(self):
        return self._row_suffixes

    def get_packed_dim(self, name: str):
        cfg = self._select_cfg(name)
        return cfg.get_packed_dim(name) if cfg else None

    def is_transposed(self, name: str):
        cfg = self._select_cfg(name)
        return cfg.is_transposed(name) if cfg else False

    def pack_factor(self, name: str) -> int:
        cfg = self._select_cfg(name)
        if cfg and cfg.weight_bits == 4:
            return 2
        # if cfg and cfg.weight_bits == 8:
        return 1

    # ---------- internal ----------
    def _select_cfg(self, name: str):
        # quick heuristic: INT4 tensors always have "qweight" or "s1_" / "s2_"
        if any(tag in name for tag in ("qweight", "s1_", "s2_")):
            return self.cfg4
        return self.cfg8
