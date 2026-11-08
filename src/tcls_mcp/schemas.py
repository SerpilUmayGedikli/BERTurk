"""Tool output models. The SDK turns each return annotation into the tool's outputSchema,
so the field descriptions here are what an LLM client reads."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    label: str = Field(description="Predicted label (the class with the highest probability).")
    confidence: float = Field(
        description="Probability of the predicted label, 0..1. Same as probabilities[label]."
    )
    probabilities: dict[str, float] = Field(
        description="Softmax probability for every label; the values sum to 1."
    )
    truncated: bool = Field(
        description="True if the text was longer than the model limit and only its beginning "
        "was classified."
    )
    input_tokens: int = Field(
        description="Length of the full text in model tokens, including [CLS] and [SEP]."
    )
    used_tokens: int = Field(description="Tokens actually given to the model (at most 512).")
    warning: str | None = Field(
        default=None, description="Human-readable note, e.g. about truncation; null if none."
    )


class BatchClassificationResult(BaseModel):
    count: int = Field(description="Number of texts classified (equals len(results)).")
    label_counts: dict[str, int] = Field(
        description="How many texts got each label; every label is present, possibly with 0."
    )
    truncated_count: int = Field(description="How many texts were truncated to 512 tokens.")
    results: list[ClassificationResult] = Field(
        description="One result per input text, in the same order as the input."
    )


class TokenContribution(BaseModel):
    token: str = Field(description="The word (as split by the BERTurk pre-tokenizer).")
    start: int = Field(description="Start character offset of the word in the input text.")
    end: int = Field(description="End character offset (exclusive).")
    contribution: float = Field(
        description="Drop in the predicted label's probability when this word is masked. "
        "Positive = the word supports the prediction; negative = it argues against it."
    )


class Explanation(BaseModel):
    label: str = Field(description="Predicted label for the unmodified text.")
    confidence: float = Field(description="Probability of that label for the unmodified text.")
    method: Literal["occlusion"] = Field(
        description="Word-level occlusion: each word's sub-tokens are replaced with [MASK] "
        "and the change in the predicted label's probability is measured."
    )
    top_tokens: list[TokenContribution] = Field(
        description="Up to 5 words with the largest contribution, highest first."
    )
    words_evaluated: int = Field(description="Number of words that were occluded one by one.")
    truncated: bool = Field(
        description="True if the text was truncated to 512 tokens; words after the cut "
        "were not evaluated."
    )
    warning: str | None = Field(default=None, description="Note about truncation; null if none.")
