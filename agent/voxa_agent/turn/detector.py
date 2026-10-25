"""Semantic end-of-utterance detection. Wraps the LiveKit turn-detector ONNX model (English)."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod


class TurnDetector(ABC):
    @abstractmethod
    def p_complete(self, text: str, last_agent_text: str = "") -> float:
        """Probability that the user has finished their turn."""


class HeuristicTurnDetector(TurnDetector):
    """Dependency-free fallback: trailing conjunction/preposition/filler or a digit run in progress
    means 'not done'. Also used when the ONNX model is unavailable."""

    _OPEN = re.compile(r"(?:\b(?:and|or|but|so|because|is|are|my|the|a|an|to|of|for|with|about|that|"
                       r"if|when|um|uh|like|its|it's|number|at|on|in)\b|[,:-])\s*$", re.I)

    def p_complete(self, text: str, last_agent_text: str = "") -> float:
        t = text.strip()
        if not t:
            return 0.0
        if self._OPEN.search(t):
            return 0.02
        if t[-1] in ".?!":
            return 0.9
        return 0.5


class LiveKitTurnDetector(TurnDetector):
    """livekit/turn-detector (English, int8 ONNX, ~1-7 ms on CPU), run directly with onnxruntime + the model's tokenizer.
    (The livekit plugin wrapper needs the agents inference executor and main-thread registration, which breaks the
    thread-executor deployment; the model itself needs neither.) The English model is trained on un-normalized text,
    so it trusts punctuation: see `is_complete` for the guard. `threshold` is the model card's tuned value."""

    threshold = 0.0289
    REPO, REVISION, ONNX = "livekit/turn-detector", "v1.2.2-en", "model_q8.onnx"

    def __init__(self) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        path = hf_hub_download(self.REPO, self.ONNX, subfolder="onnx", revision=self.REVISION, local_files_only=True)
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = 1, 1
        self._sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=so)
        self._tok = AutoTokenizer.from_pretrained(self.REPO, revision=self.REVISION, local_files_only=True, truncation_side="left")

    def p_complete(self, text: str, last_agent_text: str = "") -> float:
        msgs = ([{"role": "assistant", "content": last_agent_text}] if last_agent_text else []) + [{"role": "user", "content": text}]
        merged: list[dict[str, str]] = []
        for m in msgs:
            if not m["content"]:
                continue
            if merged and merged[-1]["role"] == m["role"]:
                merged[-1] = {"role": m["role"], "content": merged[-1]["content"] + " " + m["content"]}
            else:
                merged.append(dict(m))
        convo = self._tok.apply_chat_template(merged, add_generation_prompt=False, add_special_tokens=False, tokenize=False)
        convo = convo[: convo.rfind("<|im_end|>")]            # drop the end-of-utterance token: its probability is the output
        ids = self._tok(convo, add_special_tokens=False, return_tensors="np", max_length=128, truncation=True)["input_ids"]
        return float(self._sess.run(None, {"input_ids": ids.astype("int64")})[0].flatten()[-1])

    def is_complete(self, text: str, last_agent_text: str = "") -> bool:
        return self.p_complete(text, last_agent_text) >= self.threshold


def is_complete(det: TurnDetector, text: str, last_agent_text: str = "") -> bool:
    """Single decision point so thresholds live with the detector that produced the probability.
    ASR punctuates fragments ("My order number is."), and the model trusts punctuation, so a
    trailing function word / filler vetoes 'complete' regardless of the model."""
    if HeuristicTurnDetector._OPEN.search(text.strip().rstrip(" .?!…")):
        return False
    if isinstance(det, LiveKitTurnDetector):
        return det.is_complete(text, last_agent_text)
    return det.p_complete(text, last_agent_text) >= 0.5
