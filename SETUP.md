# Setup Instructions

## Quick Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Copy trained models**:
   Copy the `artifacts/` directory from the main SpeechRec-German project to this directory:
   ```bash
   cp -r /path/to/SpeechRec-German/artifacts ./artifacts
   ```

   The artifacts directory should contain subdirectories for each phoneme pair, for example:
   ```
   artifacts/
   ├── b-p_dl_models_with_context_v2/
   │   ├── improved_models/
   │   │   └── hybrid_cnn_mlp_v4_3_enhanced/
   │   │       ├── best_model.pt
   │   │       └── config.json
   │   ├── feature_scaler.joblib
   │   └── feature_cols.json
   ├── d-t_dl_models_with_context_v2/
   │   └── ...
   └── ...
   ```

3. **Verify installation**:
   ```python
   from validate_phoneme import validate_phoneme
   print("Installation successful!")
   ```

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
├── artifacts/              # Copied from main project
│   ├── b-p_dl_models_with_context_v2/
│   ├── d-t_dl_models_with_context_v2/
│   └── ...
├── __init__.py
├── validate_phoneme.py
├── requirements.txt
├── README.md
├── SETUP.md
├── example_usage.py
└── .gitignore
```

## Notes

- The `artifacts/` directory contains trained models and should be copied from the main SpeechRec-German project
- Models are loaded lazily (on first use) and cached in memory
- The module automatically detects available phoneme pairs from the artifacts directory
- If no models are found, the validator will still work but will return errors when trying to validate phonemes

