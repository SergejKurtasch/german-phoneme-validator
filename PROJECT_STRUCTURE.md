# Project Structure

## Overview

This project contains all necessary components for implementing the backend according to the Research Brief specification. It is a standalone module extracted from the main SpeechRec-German project.

## Directory Structure

```
german-phoneme-validator/
├── core/                          # Core modules
│   ├── __init__.py               # Core module exports
│   ├── models.py                 # Model architectures (HybridCNNMLP_V4_3)
│   ├── feature_extraction.py     # Feature extraction functions
│   └── validator.py              # Main validation logic
├── tools/                         # Utility scripts
│   ├── __init__.py
│   └── upload_models.py          # Script to upload models to Hugging Face Hub
├── core/
│   ├── downloader.py             # Model download from Hugging Face Hub
│   ├── ...
├── notebooks/                     # Development notebooks (for reference)
├── __init__.py                    # Main package exports
├── example_usage.py              # Usage examples
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation script
├── README.md                      # Main documentation
├── SETUP.md                       # Setup instructions
├── PROJECT_STRUCTURE.md          # This file
├── TECHNICAL_REPORT.md           # Technical documentation
└── .gitignore                    # Git ignore rules
```

## Component Description

### core/models.py
- Contains the `HybridCNNMLP_V4_3` model architecture
- Includes all model components (ResidualBlock2D, EnhancedChannelAttention, etc.)
- Extracted from `hybrid_model_utils.py`

### core/feature_extraction.py
- Contains all feature extraction functions:
  - `extract_mfcc_features()` - MFCC coefficients
  - `extract_energy_features()` - Energy and ZCR
  - `extract_spectral_features()` - Spectral features
  - `extract_formants_lpc()` / `extract_formants_parselmouth()` - Formants
  - `extract_vot()` - Voice Onset Time
  - `extract_all_features()` - Main feature extraction function
  - `extract_spectrogram_window()` - Mel-spectrogram extraction
- Extracted from `utils/dl_data_preparation.py`

### core/validator.py
- Contains the `PhonemeValidator` class
- Implements `validate_phoneme()` function according to Research Brief spec
- Handles model loading, feature extraction, and validation
- Supports both local artifacts and Hugging Face Hub downloads
- Extracted and adapted from `gradio_modules/phoneme_validator.py`

### core/downloader.py
- Handles model downloads from Hugging Face Hub
- Implements `get_model_assets()` function for on-demand model loading
- Uses ETag-based caching to check for updates without re-downloading
- Automatically caches models in `~/.cache/huggingface/hub/`

### __init__.py
- Main package entry point
- Exports `validate_phoneme()`, `PhonemeValidator`, and other public APIs
- Matches the Research Brief specification exactly

## Key Features

1. **Standalone Module**: All dependencies are self-contained
2. **Production Ready**: Error handling, input validation, and clear API
3. **Flexible Input**: Supports WAV files or numpy arrays
4. **Automatic Model Download**: Models are downloaded from Hugging Face Hub on first use
5. **Model Caching**: Models are loaded lazily, cached in memory, and stored locally
6. **Update Checking**: Automatically checks for model updates via ETag without full re-download
7. **Comprehensive Features**: Extracts MFCC, formants, spectral features, VOT, etc.

## Usage

**Recommended (when installed as package):**
```python
from german_phoneme_validator import validate_phoneme

result = validate_phoneme(
    audio="audio.wav",
    phoneme="/b/",
    position_ms=1500.0,
    expected_phoneme="/b/"
)
```

**Alternative (when running from project directory):**
```python
from __init__ import validate_phoneme
# or
from core.validator import validate_phoneme
```

## Next Steps

1. Clone the repository: `git clone https://github.com/SergejKurtasch/german-phoneme-validator.git`
2. Install package: `pip install -e .` or dependencies: `pip install -r requirements.txt`
3. Test with `python example_usage.py`

**Note**: Models are automatically downloaded from Hugging Face Hub on first use. 
If you have a local `artifacts/` directory (for development), it will be used instead of downloading from Hugging Face Hub.
