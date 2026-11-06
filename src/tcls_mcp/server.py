"""MCP server exposing the BERTurk classifier.

Run over stdio:  ``MODEL_PATH=... python -m tcls_mcp``  (or the ``tcls-mcp`` script).
stdout carries the MCP protocol, so every log line goes to stderr.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from tcls_mcp import __version__
from tcls_mcp.card import ModelCard, build_card
from tcls_mcp.config import MAX_BATCH_CEILING, Settings
from tcls_mcp.explain import TOP_K, explain
from tcls_mcp.model import Classifier, InputError, validate_batch, validate_text
from tcls_mcp.schemas import BatchClassificationResult, ClassificationResult, Explanation

log = logging.getLogger("tcls_mcp")

INSTRUCTIONS = """\
Turkish text classifier (fine-tuned BERTurk, dbmdz/bert-base-turkish-cased).
- classify_text: one text -> label, per-label probabilities, confidence.
- classify_batch: up to MAX_BATCH (default 64) texts in one call; prefer it over many
  classify_text calls. Longer lists must be split into chunks.
- explain_prediction: the 5 words that most support the prediction (occlusion).
Read the model://card resource for the label set, test metrics and known limitations
before drawing conclusions. Texts longer than 512 tokens are truncated (truncated=true).
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)


@dataclass
class AppState:
    settings: Settings
    classifier: Classifier
    card: ModelCard
    # Serialises model access: sync tools run in worker threads, and one forward pass at a
    # time keeps peak memory bounded (batching inside a call is where the speed comes from).
    lock: threading.Lock = field(default_factory=threading.Lock)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.handlers[:] = [handler]
    log.setLevel(level)
    log.propagate = False
    # transformers prints load-time warnings via its own logger (stderr); keep them quiet.
    logging.getLogger("transformers").setLevel(logging.ERROR)


@contextmanager
def call_log(tool: str, **fields: Any) -> Iterator[None]:
    """Log duration and status of one tool call; map input errors to ToolError."""
    t0 = time.perf_counter()
    extra = " ".join(f"{k}={v}" for k, v in fields.items())

    def ms() -> float:
        return (time.perf_counter() - t0) * 1e3

    try:
        yield
    except InputError as exc:
        log.warning("tool=%s status=invalid_input ms=%.1f %s error=%r", tool, ms(), extra, str(exc))
        raise ToolError(str(exc)) from exc
    except ToolError:
        log.warning("tool=%s status=error ms=%.1f %s", tool, ms(), extra)
        raise
    except Exception as exc:
        log.exception("tool=%s status=internal_error ms=%.1f %s", tool, ms(), extra)
        raise ToolError(f"Internal error ({type(exc).__name__}); see the server log.") from exc
    else:
        log.info("tool=%s status=ok ms=%.1f %s", tool, ms(), extra)


def build_server(
    settings: Settings | None = None,
    classifier_factory: Callable[[Settings], Classifier] | None = None,
) -> MCPServer[AppState]:
    """Create the server. Settings are read from the environment at startup if not given;
    ``classifier_factory`` lets tests inject a tiny model."""
    # Static resources get no Context in this SDK, so the card is also kept here.
    # The lifespan runs once per server, so this holder has exactly one value while running.
    loaded: dict[str, ModelCard] = {}

    @asynccontextmanager
    async def lifespan(_: MCPServer[AppState]) -> AsyncIterator[AppState]:
        s = settings or Settings.from_env()
        setup_logging(s.log_level)
        t0 = time.perf_counter()
        if classifier_factory is not None:
            clf = classifier_factory(s)
        else:
            clf = Classifier.load(s.model_path, device=s.device, inference_batch_size=s.max_batch)
        card = build_card(s.model_path, clf.labels, clf.max_tokens, clf.device, s.model_report)
        log.info(
            "model loaded in %.1fs: path=%s device=%s labels=%s status=%s max_batch=%d",
            time.perf_counter() - t0, s.model_path, clf.device, clf.labels, card.status,
            s.max_batch,
        )  # fmt: skip
        for w in card.warnings:
            log.warning("model card: %s", w)
        loaded["card"] = card
        try:
            yield AppState(settings=s, classifier=clf, card=card)
        finally:
            loaded.clear()
            log.info("server shutting down")

    mcp: MCPServer[AppState] = MCPServer(
        "berturk-classifier",
        title="BERTurk Turkish text classifier",
        instructions=INSTRUCTIONS,
        version=__version__,
        lifespan=lifespan,
    )

    def state(ctx: Context[AppState, Any]) -> AppState:
        st: AppState = ctx.request_context.lifespan_context
        return st

    @mcp.tool(title="Classify one Turkish text", annotations=READ_ONLY)
    def classify_text(
        text: Annotated[
            str,
            Field(description="The Turkish text to classify (a review, tweet, sentence...)."),
        ],
        ctx: Context[AppState, Any],
    ) -> ClassificationResult:
        """Classify a single Turkish text with the fine-tuned BERTurk model.

        Returns the predicted label, the probability of every label and the confidence
        (probability of the predicted label). Texts over 512 tokens are truncated to their
        beginning; the result then has truncated=true and a warning. For more than one
        text, use classify_batch instead.
        """
        st = state(ctx)
        with call_log("classify_text", chars=len(text) if isinstance(text, str) else None):
            validate_text(text)
            with st.lock:
                return st.classifier.classify([text])[0]

    @mcp.tool(title="Classify many Turkish texts", annotations=READ_ONLY)
    def classify_batch(
        texts: Annotated[
            list[str],
            Field(
                description=f"1 to {MAX_BATCH_CEILING} Turkish texts (the server limit "
                "MAX_BATCH may be lower). Results keep the input order."
            ),
        ],
        ctx: Context[AppState, Any],
    ) -> BatchClassificationResult:
        """Classify a list of Turkish texts in one call (much faster than one call per text).

        Each result has the same fields as classify_text, in input order. Also returns
        label_counts (the label distribution) and how many texts were truncated. Lists
        longer than the limit are rejected: split them into chunks and call once per chunk.
        """
        st = state(ctx)
        n = len(texts) if isinstance(texts, list) else None
        with call_log("classify_batch", n=n):
            valid = validate_batch(texts, st.settings.max_batch)
            with st.lock:
                results = st.classifier.classify(valid)
            counts = dict.fromkeys(st.classifier.labels, 0)
            for r in results:
                counts[r.label] += 1
            return BatchClassificationResult(
                count=len(results),
                label_counts=counts,
                truncated_count=sum(r.truncated for r in results),
                results=results,
            )

    @mcp.tool(title="Explain a prediction", annotations=READ_ONLY)
    def explain_prediction(
        text: Annotated[str, Field(description="The Turkish text whose prediction to explain.")],
        ctx: Context[AppState, Any],
    ) -> Explanation:
        """Show which words drove the model's prediction for one text.

        Uses word-level occlusion: each word is masked in turn and the drop in the
        predicted label's probability is its contribution. Returns the top 5 words,
        highest contribution first. A positive contribution supports the prediction; a
        negative one argues against it. Cost grows with text length (one model pass per
        word), so use it on individual texts, not on whole batches.
        """
        st = state(ctx)
        with call_log("explain_prediction", chars=len(text) if isinstance(text, str) else None):
            validate_text(text)
            with st.lock:
                return explain(
                    st.classifier, text, top_k=TOP_K, batch_size=st.settings.explain_batch_size
                )

    @mcp.resource(
        "model://card",
        name="model_card",
        title="Model card",
        description="Model name, version, labels, training data summary, test metrics and "
        "known limitations of the classifier behind the tools.",
        mime_type="application/json",
    )
    def model_card() -> str:
        card = loaded.get("card")
        if card is None:  # only possible before the lifespan has started
            raise RuntimeError("model is not loaded yet")
        return card.model_dump_json(indent=2)

    return mcp


mcp = build_server()


def main() -> None:
    mcp.run()  # stdio


if __name__ == "__main__":
    main()
