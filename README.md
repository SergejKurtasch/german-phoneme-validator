# German Phoneme Pronunciation Validator

A production-ready Python module for acoustic validation of German phoneme pronunciation. This module implements the Research Brief specification for L2 German pronunciation assessment.

## Overview

This module provides acoustic feature-based validation to confirm whether a German phoneme was pronounced correctly by a second language learner, using only acoustic evidence from the audio signal.

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+ (install via conda recommended for better compatibility)

### Quick Install

```bash
# Install package (recommended)
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

**Note**: For best compatibility, install PyTorch via conda:
```bash
conda install pytorch torchaudio -c pytorch
```

For optional dependencies (advanced formant extraction):
```bash
pip install -e ".[optional]"
```

## Quick Start

**If installed as package:**
```python
from german_phoneme_validator import validate_phoneme
import numpy as np

# Using numpy array
audio_array = np.random.randn(3 * 16000).astype(np.float32)  # 3 seconds at 16kHz
result = validate_phoneme(
    audio=audio_array,
    phoneme="/b/",
    position_ms=1500.0,
    expected_phoneme="/b/"
)

print(f"Correct: {result['is_correct']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Explanation: {result['explanation']}")
```

**If running from project directory (without installation):**
```python
# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import from local package
from __init__ import validate_phoneme
# or
from core.validator import validate_phoneme
```

### Using WAV File

```python
from german_phoneme_validator import validate_phoneme

result = validate_phoneme(
    audio="path/to/audio.wav",
    phoneme="/p/",
    position_ms=1200.0,
    expected_phoneme="/b/"
)
```

### Using PhonemeValidator Class

```python
from german_phoneme_validator import PhonemeValidator

validator = PhonemeValidator()
available_pairs = validator.get_available_pairs()
print(f"Available pairs: {available_pairs}")

result = validator.validate_phoneme(
    audio="audio.wav",
    phoneme="/b/",
    position_ms=1500.0,
    expected_phoneme="/b/"
)
```

## API Reference

### `validate_phoneme()`

Main function for phoneme validation.

**Parameters:**
- `audio`: Path to WAV file (str/Path) or numpy array (16kHz, mono)
- `phoneme`: Target phoneme in IPA notation (e.g., `/b/` or `b`)
- `position_ms`: Timestamp in milliseconds where the phoneme occurs
- `expected_phoneme`: (Optional) Expected correct phoneme
- `artifacts_dir`: (Optional) Path to artifacts directory

**Returns:**
```python
{
    'is_correct': bool,      # True/False/None (error)
    'confidence': float,     # 0.0 to 1.0
    'features': dict,        # Extracted acoustic features
    'explanation': str        # Human-readable explanation
}
```

## Input/Output

**Audio Format:**
- WAV file or numpy array
- 16kHz sample rate (auto-resampled)
- Mono channel (auto-converted)
- 3-5 seconds recommended

**Phoneme Notation:**
- IPA notation with or without brackets: `/b/`, `b`, `/p/`, `p`
- Case-insensitive

**Output:**
- `is_correct`: True (correct), False (incorrect), None (error)
- `confidence`: Model confidence (0.0-1.0)
- `features`: Dictionary of acoustic features (MFCC, formants, VOT, etc.)
- `explanation`: Human-readable result description

## Supported Phoneme Pairs

The system supports 22 phoneme pairs including:
- Plosives: `b-p`, `d-t`, `g-k`, `kʰ-g`, `tʰ-d`
- Fricatives: `s-ʃ`, `ç-ʃ`, `ç-x`, `z-s`, `ts-s`, `x-k`
- Vowels: `a-ɛ`, `aː-a`, `aɪ̯-aː`, `aʊ̯-aː`, `eː-ɛ`, `iː-ɪ`, `uː-ʊ`, `oː-ɔ`, `ə-ɛ`
- Others: `ŋ-n`, `ʁ-ɐ`

Use `validator.get_available_pairs()` to see available pairs in your installation.

## Setup

1. **Copy trained models**: Copy the `artifacts/` directory from the main SpeechRec-German project
2. **Install**: `pip install -e .` or `pip install -r requirements.txt`
3. **Verify**: `python -c "from german_phoneme_validator import validate_phoneme; print('OK')"`

## Documentation

- **SETUP.md** - Detailed setup instructions
- **PROJECT_STRUCTURE.md** - Project structure and components
- **TECHNICAL_REPORT.md** - Technical documentation and methodology
- **example_usage.py** - Complete usage examples

## Error Handling

The function handles errors gracefully:
- File not found → `is_correct=None` with error message
- Invalid audio format → Error description in `explanation`
- Position out of bounds → Error message
- Unsupported phoneme pair → List of available pairs
- Model loading errors → Error description

## Performance

- Models loaded lazily and cached in memory
- Optimized feature extraction for numpy arrays
- Automatic audio resampling
- First call slower due to model loading

## License

MIT License - see LICENSE file for details.

This module is part of the German Speech Recognition project.
