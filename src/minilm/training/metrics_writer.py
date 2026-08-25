import json
from pathlib import Path
from typing import Any
from torch.utils.tensorboard import SummaryWriter

import torch


class MetricsWriter:
    def __init__(
        self, run_name: str, tb: bool = True, wandb: bool = False, jsonl: bool = True
    ) -> None:
        self.run_name = run_name
        self.tb = tb
        self.jsonl = jsonl

        self.log_dir = Path("runs") / run_name
        self._tb_writer = None
        self._jsonl_file = None
        self._closed = False

        if tb or jsonl:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        if tb:
            self._tb_writer = SummaryWriter(log_dir=str(self.log_dir))

        if jsonl:
            self._jsonl_file = (self.log_dir / "metrics.jsonl").open(
                "a", encoding="utf-8", buffering=1
            )

    def add_scalar(
        self,
        tag: str,
        scalar_value: Any,
        global_step: int | None = None,
    ) -> None:

        if self._tb_writer is not None:
            self._tb_writer.add_scalar(
                tag,
                scalar_value,
                global_step=global_step,
            )

        if self._jsonl_file is not None:
            value = self._as_json_scalar(scalar_value)
            record = {"tag": tag, "value": value, "step": global_step}
            self._jsonl_file.write(json.dumps(record, allow_nan=False) + "\n")

    def add_text(
        self,
        tag: str,
        text_string: str,
        global_step: int | None = None,
    ) -> None:
        if self._tb_writer is not None:
            self._tb_writer.add_text(
                tag,
                text_string,
                global_step=global_step,
            )

    def add_hparams(
        self,
        hparam_dict: dict[str, Any],
        metric_dict: dict[str, Any],
        hparam_domain_discrete: dict[str, list[Any]] | None = None,
        run_name: str | None = None,
        global_step: int | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("Cannot write metrics after MetricsWriter.close()")
        if self._tb_writer is not None:
            self._tb_writer.add_hparams(
                self._flatten_hparams(hparam_dict),
                metric_dict,
                hparam_domain_discrete=hparam_domain_discrete,
                run_name=run_name,
                global_step=global_step,
            )

    def flush(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.flush()
        if self._jsonl_file is not None and not self._jsonl_file.closed:
            self._jsonl_file.flush()

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._jsonl_file is not None:
            self._jsonl_file.close()
        self._closed = True

    @classmethod
    def _flatten_hparams(cls, hparams: dict[str:Any], prefix=""):
        flat = {}
        for k, v in hparams.items():
            if isinstance(v, dict):
                flat.update(cls._flatten_hparams(v, f"{k}/"))
            elif isinstance(v, (bool, int, float, str, torch.Tensor)):
                flat[k] = v
            else:
                flat[k] = str(v)
        return flat

    @staticmethod
    def _as_json_scalar(value: Any) -> bool | int | float:
        item = getattr(value, "item", None)
        if callable(item):
            value = item()
        if not isinstance(value, (bool, int, float)):
            raise TypeError(
                f"scalar_value must be a scalar, got {type(value).__name__}"
            )
        return value
