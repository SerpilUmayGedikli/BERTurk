"""Word-level occlusion explanations.

For each word (as split by the tokenizer's pre-tokenizer, so punctuation is its own word)
all of its WordPiece sub-tokens are replaced with [MASK], and the drop in the predicted
label's probability is that word's contribution. One extra forward pass per word, batched.
See README "explain_prediction" for why occlusion was chosen and its limits.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from tcls_mcp.model import PROB_DECIMALS, Classifier
from tcls_mcp.schemas import Explanation, TokenContribution

TOP_K = 5


def explain(clf: Classifier, text: str, top_k: int = TOP_K, batch_size: int = 32) -> Explanation:
    enc = clf.encode(text)
    base = clf.probabilities([enc.input_ids])[0]
    pred = int(np.argmax(base))

    # word id -> token positions (special tokens have word id None and are never masked)
    words: OrderedDict[int, list[int]] = OrderedDict()
    for pos, wid in enumerate(enc.word_ids):
        if wid is not None:
            words.setdefault(wid, []).append(pos)

    mask_id = int(clf.tokenizer.mask_token_id)
    variants: list[list[int]] = []
    for positions in words.values():
        ids = list(enc.input_ids)
        for pos in positions:
            ids[pos] = mask_id
        variants.append(ids)

    contributions: list[TokenContribution] = []
    if variants:
        probs = clf.probabilities(variants, batch_size=batch_size)
        drops = base[pred] - probs[:, pred]
        for positions, drop in zip(words.values(), drops, strict=True):
            start, end = enc.offsets[positions[0]][0], enc.offsets[positions[-1]][1]
            contributions.append(
                TokenContribution(
                    token=text[start:end],
                    start=start,
                    end=end,
                    contribution=round(float(drop), PROB_DECIMALS),
                )
            )
    # Stable sort: ties keep text order, so the output is deterministic.
    contributions.sort(key=lambda c: c.contribution, reverse=True)

    warning = None
    if enc.truncated:
        warning = (
            f"Text has {enc.input_tokens} tokens; only words in the first "
            f"{len(enc.input_ids)} tokens were evaluated."
        )
    return Explanation(
        label=clf.labels[pred],
        confidence=round(float(base[pred]), PROB_DECIMALS),
        method="occlusion",
        top_tokens=contributions[:top_k],
        words_evaluated=len(words),
        truncated=enc.truncated,
        warning=warning,
    )
