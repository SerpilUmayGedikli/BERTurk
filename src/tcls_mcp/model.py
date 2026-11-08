"""Model loading, input validation, tokenization with truncation, and batched inference.

Kept free of MCP imports so it can be tested and benchmarked on its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tcls_mcp.config import MAX_TEXT_CHARS, MODEL_MAX_TOKENS, ConfigError
from tcls_mcp.schemas import ClassificationResult

PROB_DECIMALS = 6


class InputError(ValueError):
    """Invalid tool input. The message is shown to the client, so it must be actionable."""


def validate_text(text: object, where: str = "text") -> str:
    if not isinstance(text, str):
        raise InputError(f"{where} must be a string, got {type(text).__name__}.")
    if text == "":
        raise InputError(f"{where} is empty. Pass a non-empty Turkish text.")
    if not text.strip():
        raise InputError(f"{where} contains only whitespace. Pass a text with real content.")
    if len(text) > MAX_TEXT_CHARS:
        raise InputError(
            f"{where} has {len(text)} characters; the limit is {MAX_TEXT_CHARS}. "
            "Split it into smaller pieces."
        )
    return text


def validate_batch(texts: object, max_batch: int) -> list[str]:
    if not isinstance(texts, list):
        raise InputError(f"texts must be a list of strings, got {type(texts).__name__}.")
    if not texts:
        raise InputError(f"texts is an empty list. Pass between 1 and {max_batch} texts.")
    if len(texts) > max_batch:
        raise InputError(
            f"texts has {len(texts)} items; the limit is {max_batch} per call. "
            f"Split the list into chunks of at most {max_batch} and call classify_batch per chunk."
        )
    return [validate_text(t, where=f"texts[{i}]") for i, t in enumerate(texts)]


@dataclass(frozen=True)
class Encoded:
    """One text after tokenization and (if needed) truncation."""

    input_ids: list[int]  # with [CLS]/[SEP], at most max_tokens long
    input_tokens: int  # length of the untruncated encoding
    word_ids: list[int | None]  # per token; None for special tokens
    offsets: list[tuple[int, int]]  # per token character span in the original text

    @property
    def truncated(self) -> bool:
        return self.input_tokens > len(self.input_ids)


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ConfigError(
            f"DEVICE={requested} but CUDA is not available (torch {torch.__version__}). "
            "Use DEVICE=cpu or install a CUDA build of torch."
        )
    return requested


class Classifier:
    """A sequence-classification model plus its fast tokenizer, loaded once."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cpu",
        max_tokens: int = MODEL_MAX_TOKENS,
        inference_batch_size: int = 64,
    ) -> None:
        if not getattr(tokenizer, "is_fast", False):
            raise ConfigError("A fast tokenizer (tokenizer.json) is required for offsets/word ids.")
        if tokenizer.mask_token_id is None or tokenizer.pad_token_id is None:
            raise ConfigError("The tokenizer needs [MASK] and [PAD] tokens.")
        positions = int(getattr(model.config, "max_position_embeddings", max_tokens))
        self.max_tokens = min(max_tokens, positions)
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.inference_batch_size = inference_batch_size
        id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
        if sorted(id2label) != list(range(len(id2label))):
            raise ConfigError(f"config.json id2label must map 0..K-1, got {id2label}")
        self.labels = [id2label[i] for i in range(len(id2label))]

    @classmethod
    def load(cls, path: Path, device: str = "auto", **kwargs: Any) -> Classifier:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not (path / "config.json").is_file():
            raise ConfigError(
                f"MODEL_PATH={path} has no config.json. Point it at a checkpoint directory "
                "(for example .../checkpoint-NNN)."
            )
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        return cls(model, tokenizer, device=resolve_device(device), **kwargs)

    # -- tokenization --------------------------------------------------------------------------

    def encode(self, text: str) -> Encoded:
        # verbose=False: we truncate ourselves, so the ">512 tokens" warning is noise.
        enc = self.tokenizer(
            text, add_special_tokens=True, return_offsets_mapping=True, verbose=False
        )
        ids: list[int] = list(enc["input_ids"])
        word_ids: list[int | None] = list(enc.word_ids())
        offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
        n = len(ids)
        if n > self.max_tokens:
            # Keep the head and the closing [SEP]: same as truncation="head" in training.
            keep = self.max_tokens - 1
            ids = [*ids[:keep], ids[-1]]
            word_ids = [*word_ids[:keep], None]
            offsets = [*offsets[:keep], (0, 0)]
        return Encoded(ids, n, word_ids, offsets)

    # -- inference -----------------------------------------------------------------------------

    def probabilities(
        self, batch_ids: Sequence[Sequence[int]], batch_size: int | None = None
    ) -> np.ndarray:
        """Softmax probabilities, shape (len(batch_ids), n_labels). Chunks internally."""
        import torch

        step = batch_size or self.inference_batch_size
        out: list[np.ndarray] = []
        pad = int(self.tokenizer.pad_token_id)
        for start in range(0, len(batch_ids), step):
            chunk = batch_ids[start : start + step]
            width = max(len(x) for x in chunk)
            ids = torch.full((len(chunk), width), pad, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for i, row in enumerate(chunk):
                ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
                mask[i, : len(row)] = 1
            with torch.inference_mode():
                logits = self.model(
                    input_ids=ids.to(self.device),
                    attention_mask=mask.to(self.device),
                    token_type_ids=torch.zeros_like(ids).to(self.device),
                ).logits
            out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0)

    def classify(self, texts: Sequence[str]) -> list[ClassificationResult]:
        encoded = [self.encode(t) for t in texts]
        probs = self.probabilities([e.input_ids for e in encoded])
        return [self._result(e, p) for e, p in zip(encoded, probs, strict=True)]

    def _result(self, enc: Encoded, probs: np.ndarray) -> ClassificationResult:
        best = int(np.argmax(probs))
        warning = None
        if enc.truncated:
            warning = (
                f"Text has {enc.input_tokens} tokens; only the first {len(enc.input_ids)} "
                f"(model limit {self.max_tokens}) were classified."
            )
        return ClassificationResult(
            label=self.labels[best],
            confidence=round(float(probs[best]), PROB_DECIMALS),
            probabilities={
                lab: round(float(p), PROB_DECIMALS)
                for lab, p in zip(self.labels, probs, strict=True)
            },
            truncated=enc.truncated,
            input_tokens=enc.input_tokens,
            used_tokens=len(enc.input_ids),
            warning=warning,
        )
