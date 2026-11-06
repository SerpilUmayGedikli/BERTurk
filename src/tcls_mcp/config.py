"""Server settings, read from environment variables once at startup."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# Hard limits of the model / protocol, not user-tunable.
MODEL_MAX_TOKENS = 512  # BERT position embeddings
MAX_BATCH_CEILING = 64  # the documented upper bound of classify_batch
MAX_TEXT_CHARS = 100_000  # rejects pathological inputs before tokenization
VALID_DEVICES = ("auto", "cpu", "cuda")


class ConfigError(ValueError):
    """Invalid server configuration (raised at startup, never per call)."""


@dataclass(frozen=True)
class Settings:
    model_path: Path
    max_batch: int = MAX_BATCH_CEILING
    device: str = "auto"
    model_report: Path | None = None  # run JSON written by tcls.train (for model://card)
    explain_batch_size: int = 32  # forward-pass batch size for occlusion
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not 1 <= self.max_batch <= MAX_BATCH_CEILING:
            raise ConfigError(f"MAX_BATCH must be in 1..{MAX_BATCH_CEILING}, got {self.max_batch}")
        if self.device not in VALID_DEVICES and not self.device.startswith("cuda:"):
            raise ConfigError(
                f"DEVICE must be one of {VALID_DEVICES} or cuda:N, got {self.device!r}"
            )
        if self.explain_batch_size < 1:
            raise ConfigError("EXPLAIN_BATCH_SIZE must be >= 1")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        model_path = env.get("MODEL_PATH", "").strip()
        if not model_path:
            raise ConfigError(
                "MODEL_PATH is not set. Point it at a checkpoint directory that contains "
                "config.json, model.safetensors and tokenizer.json."
            )
        report = env.get("MODEL_REPORT", "").strip()
        return cls(
            model_path=Path(model_path),
            max_batch=_int(env, "MAX_BATCH", MAX_BATCH_CEILING),
            device=env.get("DEVICE", "auto").strip().lower() or "auto",
            model_report=Path(report) if report else None,
            explain_batch_size=_int(env, "EXPLAIN_BATCH_SIZE", 32),
            log_level=env.get("LOG_LEVEL", "INFO").strip().upper() or "INFO",
        )


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
