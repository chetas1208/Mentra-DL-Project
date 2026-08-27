"""P1.9 -- generic ASR evaluation interface + local backend + transcript cache.

Deliberately NOT hard-coded to one commercial provider. The contract is:

    backend.transcribe(pcm: float32 mono @ sr) -> ASRResult
        .text        normalised-ish raw transcript string
        .words       [ASRWord(word, start_s, end_s)]  (may be empty if a
                     backend cannot produce timestamps -- callers must
                     degrade, see `ASRResult.has_timestamps`)
        .latency_s   wall-clock decode time (excludes model load)
        .rtf         latency_s / audio_duration_s
        .backend_id  stable string identifying backend+model+config

Every downstream metric in this package consumes `ASRResult` only, so
swapping in a cloud provider later is a new `ASRBackend` subclass and
nothing else.

BACKENDS
--------
`SherpaOnnxStreamingASR`  -- REAL, available in this environment. Uses the
    already-downloaded streaming Zipformer transducer at
    `models/sherpa-asr/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06`
    via `sherpa_onnx` (both already present -- no download needed). Emits
    per-token timestamps which are folded into word timestamps. CPU-only,
    measured RTF ~0.05 at num_threads=1.
`NullASR` -- explicit no-op backend that returns empty transcripts and marks
    results `available=False`. Lets the whole harness run (and its
    non-ASR metrics stay valid) on a machine with no ASR, instead of
    crashing. Never silently substituted for a real backend: the harness
    records `asr_backend_available` in every artifact.

CACHING
-------
Transcripts are cached by sha256(pcm bytes) + backend_id, under
`evaluation/agent_audio/cache/asr/`. Identical audio is never transcribed
twice -- important because the P1.5 pipeline matrix runs six pipelines over
the same windows and RAW/ORACLE variants frequently produce byte-identical
audio.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "evaluation/agent_audio/cache/asr"
DEFAULT_MODEL_DIR = REPO_ROOT / "models/sherpa-asr/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# data types
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ASRWord:
    word: str
    start_s: float
    end_s: float

    def to_json(self) -> dict:
        return {"word": self.word, "start_s": self.start_s, "end_s": self.end_s}


@dataclasses.dataclass
class ASRResult:
    text: str
    words: List[ASRWord]
    latency_s: float
    audio_duration_s: float
    backend_id: str
    available: bool = True
    cached: bool = False

    @property
    def rtf(self) -> float:
        return self.latency_s / max(self.audio_duration_s, 1e-9)

    @property
    def has_timestamps(self) -> bool:
        return bool(self.words)

    def to_json(self) -> dict:
        return {
            "text": self.text,
            "words": [w.to_json() for w in self.words],
            "latency_s": self.latency_s,
            "audio_duration_s": self.audio_duration_s,
            "rtf": self.rtf,
            "backend_id": self.backend_id,
            "available": self.available,
        }

    @staticmethod
    def from_json(d: dict, cached: bool = True) -> "ASRResult":
        return ASRResult(
            text=d["text"],
            words=[ASRWord(**w) for w in d.get("words", [])],
            latency_s=d.get("latency_s", 0.0),
            audio_duration_s=d.get("audio_duration_s", 0.0),
            backend_id=d.get("backend_id", "?"),
            available=d.get("available", True),
            cached=cached,
        )


# ---------------------------------------------------------------------------
# text normalisation (shared by every metric so WER is comparable)
# ---------------------------------------------------------------------------
_PUNCT = re.compile(r"[^\w\s']")
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation (keeping intra-word apostrophes), collapse
    whitespace. Applied identically to references and hypotheses so a
    backend that emits punctuation/casing (like the Zipformer here) is not
    penalised against a reference that does not."""
    s = s.lower().replace("-", " ")
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def tokenize(s: str) -> List[str]:
    n = normalize_text(s)
    return n.split() if n else []


# ---------------------------------------------------------------------------
# backend interface
# ---------------------------------------------------------------------------
class ASRBackend:
    """Provider-agnostic ASR contract. Subclass this for a cloud provider."""

    backend_id: str = "abstract"
    available: bool = False

    def transcribe(self, pcm: np.ndarray, sr: int = SAMPLE_RATE) -> ASRResult:
        raise NotImplementedError

    # -- caching wrapper -----------------------------------------------------
    def transcribe_cached(self, pcm: np.ndarray, sr: int = SAMPLE_RATE,
                          cache_dir: Path = CACHE_DIR) -> ASRResult:
        key = audio_key(pcm, sr, self.backend_id)
        p = Path(cache_dir) / f"{key}.json"
        if p.exists():
            try:
                return ASRResult.from_json(json.loads(p.read_text()), cached=True)
            except Exception:
                pass  # corrupt cache entry -> re-transcribe, don't crash
        res = self.transcribe(pcm, sr)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(res.to_json()))
        os.replace(tmp, p)
        return res


def audio_key(pcm: np.ndarray, sr: int, backend_id: str) -> str:
    """Cache key = content hash of the exact PCM + sample rate + backend id.
    Uses float32 bytes so a 1-LSB difference from a different pipeline is a
    different key (correct: it may transcribe differently)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(pcm, dtype=np.float32).tobytes())
    h.update(str(int(sr)).encode())
    h.update(backend_id.encode())
    return h.hexdigest()[:40]


# ---------------------------------------------------------------------------
# real local backend: sherpa-onnx streaming zipformer
# ---------------------------------------------------------------------------
class SherpaOnnxStreamingASR(ASRBackend):
    """Local streaming-transducer ASR. Model + runtime are already in-repo.

    `num_threads` defaults to 1 on purpose: this is a heavily shared box and
    the GeoWearNet GPU campaign's dataloaders are already using most cores.
    """

    _lock = threading.Lock()

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, num_threads: int = 1,
                 decoding_method: str = "greedy_search", tail_padding_s: float = 0.5):
        self.model_dir = Path(model_dir)
        self.num_threads = num_threads
        self.decoding_method = decoding_method
        self.tail_padding_s = tail_padding_s
        self._rec = None
        self.available = self.model_dir.is_dir() and (self.model_dir / "encoder.onnx").exists()
        self.backend_id = (f"sherpa-onnx/{self.model_dir.name}/"
                           f"{decoding_method}/t{num_threads}")

    def _recognizer(self):
        if self._rec is None:
            import sherpa_onnx
            d = self.model_dir
            self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(d / "tokens.txt"),
                encoder=str(d / "encoder.onnx"),
                decoder=str(d / "decoder.onnx"),
                joiner=str(d / "joiner.onnx"),
                num_threads=self.num_threads,
                sample_rate=SAMPLE_RATE,
                feature_dim=80,
                decoding_method=self.decoding_method,
            )
        return self._rec

    def transcribe(self, pcm: np.ndarray, sr: int = SAMPLE_RATE) -> ASRResult:
        if not self.available:
            return ASRResult("", [], 0.0, len(pcm) / max(sr, 1), self.backend_id, available=False)
        pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        dur = len(pcm) / float(sr)
        rec = self._recognizer()
        t0 = time.perf_counter()
        with self._lock:  # sherpa streams are not shared, but keep decode serial on a loaded box
            s = rec.create_stream()
            s.accept_waveform(sr, pcm)
            if self.tail_padding_s > 0:
                s.accept_waveform(sr, np.zeros(int(sr * self.tail_padding_s), dtype=np.float32))
            s.input_finished()
            while rec.is_ready(s):
                rec.decode_stream(s)
            res = rec.get_result_all(s)
        latency = time.perf_counter() - t0
        words = _tokens_to_words(list(res.tokens), list(res.timestamps))
        return ASRResult(text=res.text, words=words, latency_s=latency,
                         audio_duration_s=dur, backend_id=self.backend_id)


def _tokens_to_words(tokens: Sequence[str], timestamps: Sequence[float]) -> List[ASRWord]:
    """Fold BPE-ish sub-tokens into words. This model emits a leading space on
    word-initial tokens (' Ask', 's', 'k'), the standard sherpa convention.
    A word's start is its first sub-token's timestamp; its end is the next
    word's start (or last token + a small tail)."""
    words: List[ASRWord] = []
    cur, cur_start = "", None
    for tok, ts in zip(tokens, timestamps):
        if tok.startswith(" ") or cur == "":
            if cur.strip():
                words.append(ASRWord(cur.strip(), cur_start, float(ts)))
            cur, cur_start = tok, float(ts)
        else:
            cur += tok
    if cur.strip():
        end = (float(timestamps[-1]) + 0.20) if len(timestamps) else (cur_start or 0.0)
        words.append(ASRWord(cur.strip(), cur_start if cur_start is not None else 0.0, end))
    # normalise word text, drop anything that normalises away
    out = []
    for w in words:
        n = normalize_text(w.word)
        if n:
            out.append(ASRWord(n, w.start_s, max(w.end_s, w.start_s)))
    return out


class NullASR(ASRBackend):
    """Explicit unavailable-ASR backend. Returns empty transcripts and
    `available=False` so the harness can record BLOCKED rather than pretend
    a 0% WER."""

    backend_id = "null-asr/none"
    available = False

    def transcribe(self, pcm: np.ndarray, sr: int = SAMPLE_RATE) -> ASRResult:
        return ASRResult("", [], 0.0, len(pcm) / max(sr, 1), self.backend_id, available=False)


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------
_DEFAULT: Optional[ASRBackend] = None


def get_backend(name: str = "auto", **kw) -> ASRBackend:
    """`auto` -> local sherpa-onnx if its model is present, else NullASR."""
    global _DEFAULT
    if name in ("auto", "sherpa", "sherpa-onnx"):
        try:
            import sherpa_onnx  # noqa: F401
            b = SherpaOnnxStreamingASR(**kw)
            if b.available:
                return b
        except Exception:
            pass
        if name != "auto":
            raise RuntimeError("sherpa-onnx ASR requested but unavailable")
        return NullASR()
    if name in ("null", "none"):
        return NullASR()
    raise ValueError(f"unknown ASR backend {name!r}")


def default_backend() -> ASRBackend:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = get_backend("auto")
    return _DEFAULT


def availability_report() -> Dict[str, object]:
    b = get_backend("auto")
    return {
        "backend_id": b.backend_id,
        "available": bool(b.available),
        "model_dir": str(getattr(b, "model_dir", "")),
        "status": "AVAILABLE" if b.available else "BLOCKED_NO_LOCAL_ASR",
        "cache_dir": str(CACHE_DIR),
    }


if __name__ == "__main__":
    print(json.dumps(availability_report(), indent=2))
