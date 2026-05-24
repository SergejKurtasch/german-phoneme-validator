# German Phoneme Pronunciation Validator

A Python module for acoustic validation of German phoneme pronunciation, designed as a second-stage verifier in L2 German speech assessment pipelines.

## Live Demo

**[German Pronunciation Trainer](https://sergejkurt-german-pronunciation-trainer.hf.space/)**

Upload audio and check your German pronunciation without any local setup.

---

## What It Does

The validator takes a full audio waveform, a timestamp, and a phoneme pair (e.g., what the learner said vs. what they should have said), extracts acoustic features, and returns a binary verdict — correct or incorrect — along with a confidence score.

It covers 22 phoneme confusion pairs common in L2 German:

| Category | Pairs |
|----------|-------|
| Plosives | `b-p`, `d-t`, `g-k`, `kʰ-g`, `tʰ-d` |
| Fricatives | `s-ʃ`, `ç-ʃ`, `ç-x`, `z-s`, `ts-s`, `x-k` |
| Vowels | `a-ɛ`, `aː-a`, `aɪ̯-aː`, `aʊ̯-aː`, `eː-ɛ`, `iː-ɪ`, `uː-ʊ`, `oː-ɔ`, `ə-ɛ` |
| Sonorants | `ŋ-n`, `ʁ-ɐ` |

Each pair has a dedicated trained model. Models (~1.74 GB total) are downloaded from [Hugging Face Hub](https://huggingface.co/SergejKurt/german-phoneme-models) on first use and cached locally.

---

## Installation

```bash
pip install git+https://github.com/SergejKurtasch/german-phoneme-validator.git
```

Or in editable mode from a local clone:

```bash
git clone https://github.com/SergejKurtasch/german-phoneme-validator.git
cd german-phoneme-validator
pip install -e .
```

**Requirements:** Python 3.8+, PyTorch 2.0+.

---

## Quick Start

```python
from german_phoneme_validator import validate_phoneme
import numpy as np

# audio: numpy array (16 kHz mono) or path to a WAV file
audio = np.random.randn(3 * 16000).astype(np.float32)

result = validate_phoneme(
    audio=audio,
    phoneme="b",           # what the learner produced
    position_ms=1500.0,    # timestamp in milliseconds
    expected_phoneme="p"   # what they should have said
)

print(result["is_correct"])   # True / False / None
print(result["confidence"])   # 0.0 – 1.0
print(result["explanation"])
```

Models load on first call. Subsequent calls within the same session are fast.

### Using the class directly

```python
from german_phoneme_validator import PhonemeValidator

validator = PhonemeValidator()
print(validator.get_available_pairs())  # lists all supported pairs

result = validator.validate_phoneme(
    audio="recording.wav",
    phoneme="b",
    position_ms=1500.0,
    expected_phoneme="p"
)
```

---

## How It Works

1. The caller passes the full audio waveform and the timestamp where the phoneme occurs.
2. The module extracts acoustic features around that position (MFCC, formants, VOT, spectral features).
3. A pair-specific classifier (one per phoneme confusion pair) returns a probability score.
4. The result is mapped to a binary verdict with confidence.

No forced alignment or transcription is required — the caller supplies the position.

---

## License

MIT — see `LICENSE`.
