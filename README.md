# tts-ref-picker: Pick the Best Twi Reference Voice for Zero-Shot TTS

**tts-ref-picker** is a self-contained repo: a curated dataset of **205 expressive single-speaker Twi (Akan) audio prompts**, specifically engineered for **zero-shot voice cloning, emotional style transfer, and reference audio prompting** in modern TTS models (e.g., **F5-TTS**, **CosyVoice**, **XTTS-v2**, **Chatterbox**, and **StyleTTS2**).

All audio samples have been processed through **Demucs v4 (`htdemucs`)** (music removal) followed by the **resemble-enhance speech denoiser** (ambient noise / SFX removal) on an NVIDIA H200 GPU, and screened with **pyannote speaker-diarization-3.1** so every clip contains exactly one speaker. Clips are normalized and ready to plug directly into any TTS pipeline.

---

## Quick Start

```bash
git clone https://github.com/GhanaOpenAI/tts-ref-picker.git
cd tts-ref-picker
pip install -r requirements.txt        # numpy, onnxruntime, tokenizers

python3 match_prompt.py "Fi me dan mu seesei ara!"
```

Everything needed is in the repo: the clean audio, metadata, the ONNX embedding model, pre-computed embeddings, and the translation step (built into `match_prompt.py`, standard library only, no API key). The only network call is to Google Translate; without internet the raw text is matched directly.

---

## Key Highlights

- **Vocal Isolation:** Demucs v4 (`htdemucs`) + resemble-enhance denoiser on NVIDIA H200 GPU — median speech-to-noise-floor 48 dB.
- **Studio Audio Format:** **24,000 Hz, 16-bit Mono PCM WAV**, peak-normalized to **-1.0 dBFS** (the universal standard for modern zero-shot TTS reference prompts).
- **Single Speaker Only:** 54 source clips were dropped (42 with a second voice detected by pyannote, 3 with no speech left after cleaning, 9 with residual background noise — speech-to-noise-floor < 25 dB).
- **Multi-Level Tagging:** Broadened beyond game dialogue into a rich, hierarchical speech taxonomy with **Primary Categories**, **Sub-Labels**, **Emotion Classes**, **Delivery Styles**, and **Target TTS Use Cases**.
- **CPU Prompt Matcher (`match_prompt.py`):** Includes a local, offline, CPU-ready semantic search engine powered by `all-MiniLM-L6-v2` (ONNX runtime) that translates any user input to English (free Google Translate endpoint), embeds it, and picks the best matching prompt audio.

---

## Dataset Summary

| Specification | Details |
| :--- | :--- |
| **Total Samples** | 205 prompt audios (single speaker) |
| **Audio Format** | 24,000 Hz, 16-bit Mono PCM WAV |
| **Audio Processing** | Demucs v4 (`htdemucs`) + resemble-enhance denoiser + pyannote single-speaker filter + Peak Normalization (-1.0 dBFS) |
| **Language Pair** | Spoken Twi (Akan) with English Translations & Subtitles |
| **Duration Range** | 1.04s - 8.40s (Average: 3.42s — optimal for reference prompts) |
| **Audio Quality Tier** | Excellent (Composite Score ≥ 80, SNR > 20 dB, 0% clipping) |

---

## Speech Taxonomy & Labels

Users have complete flexibility to filter by broad speech acts or drill down into granular emotional sub-labels:

### 1. Primary Categories (8 Classes)
- **`inquiry_investigation`** (51 samples): Questions, curiosity, investigative, clarification, suspicion.
- **`conversational_banter`** (41 samples): Colloquial talk, witty remarks, casual back-and-forth, humor.
- **`emotional_expressive`** (35 samples): Deep feeling, love, heartache, vulnerability, remorse, grief.
- **`wisdom_reflection`** (24 samples): Proverbs, life counsel, patience, philosophical commentary.
- **`conflict_confrontation`** (21 samples): Heated disputes, threats, indignations, moral outrage.
- **`command_warning`** (19 samples): Directives, silencing orders, firm boundaries, cautions.
- **`greeting_social`** (11 samples): Warm welcomes, blessings, gratitude, respectful greetings.
- **`negotiation_trade`** (3 samples): Pricing, commercial bargaining, persuasion, agreements.

### 2. Emotion Classes (15 Nuanced Tones)
`curious_inquisitive` (48) | `conversational_neutral` (40) | `calm_reassurance` (25) | `affection_warmth` (20) | `authoritative_stern` (18) | `anger_aggression` (10) | `playful_humor` (8) | `indignation_outrage` (7) | `gratitude_reverence` (6) | `joy_warmth` (6) | `fear_anxiety` (5) | `defensive_disbelief` (4) | `sorrow_grief` (4) | `remorse_contrition` (3) | `urgent_alarmed` (1)

### 3. Delivery Styles
`casual_conversational` | `inquisitive_probing` | `warm_gentle` | `reflective_measured` | `indignant_sharp` | `authoritative_commanding` | `sorrowful_somber` | `playful_witty` | `distressed_breathless` | `urgent_fast`

### 4. Target TTS Use Cases
- `conversational_ai`: Chatbots, voice assistants, customer service, daily dialogue.
- `audiobook_storytelling`: Audiobooks, cultural narratives, podcasts, documentary narration.
- `drama_film_dubbing`: Cinematic dubbing, high-emotion dramatic acting, animation.
- `interactive_character`: Video game NPCs, virtual companions, responsive audio.
- `public_announcements`: System alerts, public notices, authoritative directives.
- `commercial_advertising`: Voiceovers, product persuasion, promo reads.

---

## Instant CPU Prompt Matcher

The repository includes a ready-to-use semantic matcher (`match_prompt.py`) that uses pre-computed 384-dimensional embeddings and an offline CPU ONNX model. When a user enters text **in any language** (Twi, Ewe, Ga, English, ...), it is first translated to English with the free Google Translate endpoint (no API key), then embedded in ~3 ms and matched against the English prompt texts to pick the best Twi reference audio. If translation is unavailable (offline or rate-limited), the raw text is matched directly; pass `--no-translate` / `translate=False` to skip it.

### CLI Usage

```bash
# 1. Standard search (top match)
python3 match_prompt.py "Thank you and may God bless your family"

# Output:
# [Score: 0.7470] - ghana_prompt_0001
# English:  "Thank you and may God bless you."
# Twi:      "nyame nhyira wo nyame mma wo nyɛ kɛseɛ ɔmma"
# Category: greeting_social | Emotion: gratitude_reverence | Style: warm_gentle
# Audio:    audio/prompt_0001_greeting_social_blessing.wav (4.88s, 24kHz mono clean vocal)

# 2. Twi (or Ewe) input is auto-translated before matching
python3 match_prompt.py "Fi me dan mu seesei ara!"
# Translated (ak): "Get out of my room right now!"
# Audio:    audio/prompt_0249_command_warning_warning.wav

# 3. Retrieve top-3 candidates
python3 match_prompt.py "Get out of my room immediately!" --top-k 3

# 4. Direct pipe into TTS scripts (outputs audio file path only)
python3 match_prompt.py "Please help me, I am begging you" --audio-only
# /path/to/tts-ref-picker/audio/prompt_0065_command_warning_shelter_eviction.wav

# 5. JSON output for automated pipelines
python3 match_prompt.py "How much is this cloth?" --json
```

### Python API

```python
from match_prompt import PromptMatcher

matcher = PromptMatcher()

# Find the best reference audio for an input text
match = matcher.match("Nyame nkyɛ wo")   # any language; translated to English first

print("Matched Audio:", match.audio_file)
print("English Text: ", match.english_text)
print("Twi Spoken:   ", match.twi_text)
print("Emotion:      ", match.emotion)
print("Similarity:   ", match.score)
print("Translated:   ", match.query_english, f"({match.query_lang})")
```

---

## Plugging into Zero-Shot TTS Pipelines

### 1. F5-TTS / E2-TTS
```bash
python -m f5_tts.infer_cli \
    --ref_audio $(python3 match_prompt.py "Good morning my elders" --audio-only) \
    --ref_text "wo maame aka biribi akyerɛ me" \
    --gen_text "Me daase pii, me nua baa." \
    --output_dir output/
```

### 2. CosyVoice / XTTS-v2
```python
from match_prompt import PromptMatcher

matcher = PromptMatcher()
prompt = matcher.match("We must be very patient")

# XTTS-v2 inference
tts.tts_to_file(
    text="Mmerɛ no bɛso.",
    speaker_wav=prompt.audio_file,
    language="en", # or fine-tuned twi
    file_path="output.wav"
)
```

---

## File Structure

```
tts-ref-picker/
├── audio/                                  # 205 cleaned single-speaker vocal files (24kHz mono WAV)
│   ├── prompt_0001_greeting_social_blessing.wav
│   ├── prompt_0002_conflict_confrontation_kill_denial.wav
│   └── ...
├── models/                                 # Offline CPU semantic matcher assets
│   ├── all-MiniLM-L6-v2/                   # ONNX CPU embedding model & tokenizer
│   ├── embeddings.npy                      # Pre-computed 205x384 unit-normalized embeddings
│   └── prompts_index.json                  # Metadata lookup index
├── ghana_tones_tts_prompts.json            # Complete structured dataset with all tags & metrics
├── ghana_tones_tts_prompts.csv             # Tabular spreadsheet format
├── match_prompt.py                         # CLI & Python matcher (translate -> embed -> pick audio)
├── requirements.txt                        # numpy, onnxruntime, tokenizers
└── README.md                               # Dataset documentation
```

---

## Provenance & Attribution

- Source Material: Ghanaian movie speech transcription dataset (`ghananlpcommunity/kumawood-speech-transcriptions`).
- Vocal Isolation: **Demucs v4 (`htdemucs`)** + **resemble-enhance** denoiser on the **Ghana NLP H200 GPU**.
- Speaker Screening: **pyannote/speaker-diarization-3.1** (clips with a second speaker ≥ 0.1 s removed).
- Linguistic Alignment & Quality Screening: Google Gemini 2.5 Flash + Signal-to-Noise Ratio (SNR > 20 dB) acoustic filter.
