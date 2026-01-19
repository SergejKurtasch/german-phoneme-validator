# Setup Instructions

## Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SergejKurtasch/german-phoneme-validator.git
   cd german-phoneme-validator
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   # or
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```python
   from german_phoneme_validator import validate_phoneme
   print("Installation successful!")
   ```

**Note**: Model artifacts are automatically downloaded from Hugging Face Hub on first use. 
No manual setup is required. An internet connection is needed for the first run to download the models.

Models are cached locally in `~/.cache/huggingface/hub/` after the first download. 
Subsequent runs will check for updates automatically but won't re-download unchanged files.

The model structure (downloaded from Hugging Face Hub) contains subdirectories for each phoneme pair 
(with normalized names, avoiding special IPA characters), for example:
   ```
   {phoneme_pair}_model/
   ├── best_model.pt
   ├── config.json              # Contains original phoneme_pair and class_mapping
   ├── feature_scaler.joblib
   └── feature_cols.json
   ```
   
Note: Folder names use normalized phoneme representations (e.g., 'schwa-E_model' for 'ə-ɛ', 
'aaa-a_model' for 'aː-a') to avoid special characters in filesystem paths. All model files 
are stored directly in the model folder root.


## Testing

Run the example script to test the installation:
```bash
python example_usage.py
```

## Directory Structure

After setup, your directory should look like:
```
german-phoneme-validator/
├── core/
│   ├── __init__.py
│   ├── models.py
│   ├── feature_extraction.py
│   └── validator.py
├── tools/                  # Upload scripts (for maintainers)
│   └── upload_models.py    # Script to upload models to Hugging Face Hub
├── __init__.py
├── validate_phoneme.py
├── requirements.txt
├── README.md
├── SETUP.md
├── example_usage.py
└── .gitignore
```

## Notes

- Models are automatically downloaded from Hugging Face Hub on first use
- Models are loaded lazily (on first use) and cached in memory
- The module automatically detects available phoneme pairs
- Currently, the repository provides trained models for 22 phoneme pairs
- Models are cached in `~/.cache/huggingface/hub/` after first download
- Updates are checked automatically via ETag without re-downloading unchanged files
- For offline use, ensure models are downloaded first or use `local_files_only=True` (advanced)

## Local Development

If you're developing locally and have a local `artifacts/` directory, the validator will 
use it instead of downloading from Hugging Face Hub. This allows for local testing without 
requiring internet access.

