#!/usr/bin/env python3
"""
tts-ref-picker: semantic reference-audio matcher for TTS
Matches user input text against 205 cleaned single-speaker Twi prompt audios
using a lightweight CPU-ready sentence embedding model (all-MiniLM-L6-v2).

Input in any language (Twi, Ewe, Ga, ...) is first translated to English with the
free Google Translate endpoint, then matched against the English prompt texts.
If translation is unavailable (offline / rate-limited) the raw text is used.

Usage via CLI:
    python3 match_prompt.py "May God bless you"
    python3 match_prompt.py "Get out of my face!" --top-k 3
    python3 match_prompt.py "How much does it cost?" --audio-only
    python3 match_prompt.py "I am so heartbroken" --json
    python3 match_prompt.py "Fi me dan mu seesei ara!"     # Twi input, auto-translated

Usage in Python:
    from match_prompt import PromptMatcher
    matcher = PromptMatcher()
    best = matcher.match("Good morning!")
    print(best.audio_file, best.english_text, best.twi_text)
"""

import os
import sys
import json
import argparse
import urllib.parse
import urllib.request
from typing import List, Dict, Any, Union
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_DIR = os.path.join(MODELS_DIR, "all-MiniLM-L6-v2")
EMBEDDINGS_FILE = os.path.join(MODELS_DIR, "embeddings.npy")
INDEX_FILE = os.path.join(MODELS_DIR, "prompts_index.json")

TRANSLATE_ENDPOINTS = [
    "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q=",
    "https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl=auto&tl=en&q=",
]


def translate_to_english(text: str, timeout: float = 5.0):
    """Translate text to English via the free Google Translate endpoints.
    Returns (english_text, detected_lang); falls back to (text, None) on failure."""
    for url in TRANSLATE_ENDPOINTS:
        try:
            req = urllib.request.Request(url + urllib.parse.quote(text), headers={"User-Agent": "Mozilla/5.0"})
            data = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8"))
            if "client=gtx" in url:
                english = "".join(seg[0] for seg in data[0] if seg[0])
                lang = data[2]
            else:
                english, lang = data[0] if isinstance(data[0], list) else (data[0], None)
            if english.strip():
                return english.strip(), lang
        except Exception:
            continue
    return text, None


class _SuppressCStderr:
    """Temporarily suppresses C-level stderr (e.g. ONNX DRM device discovery notice)."""
    def __enter__(self):
        try:
            self.errnull = os.open(os.devnull, os.O_WRONLY)
            self.old_stderr = os.dup(2)
            os.dup2(self.errnull, 2)
            self.active = True
        except Exception:
            self.active = False
        return self

    def __exit__(self, *args):
        if getattr(self, "active", False):
            try:
                os.dup2(self.old_stderr, 2)
                os.close(self.errnull)
                os.close(self.old_stderr)
            except Exception:
                pass


class PromptMatch:
    def __init__(self, data: Dict[str, Any], score: float):
        self.id = data["id"]
        self.prompt_id = data["prompt_id"]
        self.audio_file = os.path.join(PROJECT_ROOT, data["audio_file"])
        self.audio_rel = data["audio_file"]
        self.english_text = data["english_text"]
        self.twi_text = data["twi_text"]
        self.primary_category = data["primary_category"]
        self.emotion = data["emotion"]
        self.delivery_style = data["delivery_style"]
        self.sub_labels = data["sub_labels"]
        self.duration_sec = data["duration_sec"]
        self.score = float(score)
        self.query_english = None
        self.query_lang = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt_id": self.prompt_id,
            "score": round(self.score, 4),
            "audio_file": self.audio_rel,
            "english_text": self.english_text,
            "twi_text": self.twi_text,
            "primary_category": self.primary_category,
            "emotion": self.emotion,
            "delivery_style": self.delivery_style,
            "sub_labels": self.sub_labels,
            "duration_sec": self.duration_sec,
            "query_english": self.query_english,
            "query_lang": self.query_lang
        }

    def __repr__(self) -> str:
        return f"<PromptMatch id={self.id} score={self.score:.4f} en='{self.english_text}'>"


class PromptMatcher:
    def __init__(self, model_dir: str = MODEL_DIR, embeddings_file: str = EMBEDDINGS_FILE, index_file: str = INDEX_FILE):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")

        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        model_onnx_path = os.path.join(model_dir, "model.onnx")

        with _SuppressCStderr():
            os.environ["ORT_LOGGING_LEVEL"] = "3"
            import onnxruntime as ort
            from tokenizers import Tokenizer

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self.session = ort.InferenceSession(model_onnx_path, sess_options, providers=['CPUExecutionProvider'])

        # Load precomputed embeddings (normalized Nx384 float32)
        self.embeddings = np.load(embeddings_file).astype(np.float32)

        # Load metadata index
        with open(index_file, "r", encoding="utf-8") as f:
            self.index = json.load(f)

    def embed_text(self, text: str) -> np.ndarray:
        enc = self.tokenizer.encode(text.strip())
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)
        token_type_ids = np.array([enc.type_ids], dtype=np.int64)

        outputs = self.session.run(None, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        })
        last_hidden = outputs[0]  # (1, seq_len, 384)

        # Mean pooling with attention mask
        mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(last_hidden * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        emb = sum_embeddings / sum_mask

        # L2 normalize
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        normed = emb / np.clip(norm, a_min=1e-9, a_max=None)
        return normed[0].astype(np.float32)

    def match(self, text: str, top_k: int = 1, translate: bool = True) -> Union[PromptMatch, List[PromptMatch]]:
        """
        Translates input text to English (unless translate=False), embeds it and retrieves
        the top-k most semantically and emotionally matching TTS prompt audios.
        """
        english, lang = translate_to_english(text) if translate else (text, None)
        query_vec = self.embed_text(english)
        sims = np.dot(self.embeddings, query_vec)
        top_indices = np.argsort(-sims)[:top_k]

        matches = [PromptMatch(self.index[idx], sims[idx]) for idx in top_indices]
        for m in matches:
            m.query_english, m.query_lang = english, lang
        return matches[0] if top_k == 1 else matches


def main():
    parser = argparse.ArgumentParser(description="Find optimal Ghanaian Twi prompt audio for TTS.")
    parser.add_argument("text", type=str, help="Text you want to synthesize or match against.")
    parser.add_argument("--top-k", "-k", type=int, default=1, help="Number of top candidates to return (default: 1)")
    parser.add_argument("--audio-only", action="store_true", help="Print only the audio file path (for shell piping into TTS)")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--no-translate", action="store_true", help="Skip Google Translate (input must be English)")

    args = parser.parse_args()

    matcher = PromptMatcher()
    matches = matcher.match(args.text, top_k=args.top_k, translate=not args.no_translate)
    if not isinstance(matches, list):
        matches = [matches]

    if args.audio_only:
        print(matches[0].audio_file)
        return

    if args.json:
        data = [m.to_dict() for m in matches]
        print(json.dumps(data if len(data) > 1 else data[0], indent=2, ensure_ascii=False))
        return

    print("=" * 70)
    print(f"Query: \"{args.text}\"")
    if matches[0].query_english != args.text:
        print(f"Translated ({matches[0].query_lang}): \"{matches[0].query_english}\"")
    print(f"Matched {len(matches)} Prompt Audio(s):")
    print("=" * 70)

    for rank, m in enumerate(matches, 1):
        print(f"#{rank} [Score: {m.score:.4f}] - {m.prompt_id}")
        print(f"  English:  \"{m.english_text}\"")
        print(f"  Twi:      \"{m.twi_text}\"")
        print(f"  Category: {m.primary_category} | Emotion: {m.emotion} | Style: {m.delivery_style}")
        print(f"  Tags:     {', '.join(m.sub_labels)}")
        print(f"  Audio:    {m.audio_rel} ({m.duration_sec}s, 24kHz mono clean vocal)")
        if rank < len(matches):
            print("-" * 70)
    print("=" * 70)


if __name__ == "__main__":
    main()
