"""
Phoneme validation module using trained models.
Adapted from gradio_modules/phoneme_validator.py for standalone use.
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
import joblib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore')

from .models import HybridCNNMLP_V4_3
from .feature_extraction import (
    extract_all_features,
    extract_spectrogram_window,
    SAMPLE_RATE,
    HOP_LENGTH,
    SPECTROGRAM_WINDOW_MS,
    N_MELS
)


class PhonemeValidator:
    """Validator for phoneme pairs using trained models."""
    
    def __init__(self, artifacts_dir: Optional[Path] = None):
        """
        Initialize phoneme validator.
        
        Args:
            artifacts_dir: Path to artifacts directory. If None, uses default.
        """
        if artifacts_dir is None:
            # Default: look for artifacts in parent directory or current directory
            current_dir = Path(__file__).parent.parent
            artifacts_dir = current_dir / 'artifacts'
            if not artifacts_dir.exists():
                # Try parent directory
                artifacts_dir = current_dir.parent / 'artifacts'
        
        self.artifacts_dir = Path(artifacts_dir)
        self.models_cache = {}
        self.scalers_cache = {}
        self.feature_cols_cache = {}
        self.available_pairs = self._discover_phoneme_pairs()
        self.device = self._get_device()
    
    def _get_device(self) -> str:
        """Auto-detect device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _discover_phoneme_pairs(self) -> List[str]:
        """Discover available phoneme pairs from artifacts directory."""
        pairs = []
        if not self.artifacts_dir.exists():
            return pairs
        
        for item in self.artifacts_dir.iterdir():
            if item.is_dir() and item.name.endswith('_dl_models_with_context_v2'):
                pair_name = item.name.replace('_dl_models_with_context_v2', '')
                model_path = item / 'improved_models' / 'hybrid_cnn_mlp_v4_3_enhanced' / 'best_model.pt'
                if model_path.exists():
                    pairs.append(pair_name)
        return sorted(pairs)
    
    def get_phoneme_pair(self, phoneme1: str, phoneme2: str) -> Optional[str]:
        """
        Determine phoneme pair from two phonemes.
        
        Args:
            phoneme1: First phoneme (expected)
            phoneme2: Second phoneme (recognized/suspected)
            
        Returns:
            Pair name (e.g., 'b-p') or None if not found
        """
        # Normalize phonemes
        p1 = phoneme1.strip().lower()
        p2 = phoneme2.strip().lower()
        
        # If phonemes are the same, check if phoneme is part of any available pair
        if p1 == p2:
            for pair in self.available_pairs:
                pair_phonemes = pair.split('-')
                if len(pair_phonemes) == 2:
                    pair_p1 = pair_phonemes[0].strip().lower()
                    pair_p2 = pair_phonemes[1].strip().lower()
                    if p1 == pair_p1 or p1 == pair_p2:
                        return pair
            return None
        
        # Try both orders
        pair1 = f"{p1}-{p2}"
        pair2 = f"{p2}-{p1}"
        
        if pair1 in self.available_pairs:
            return pair1
        elif pair2 in self.available_pairs:
            return pair2
        
        return None
    
    def _load_model(self, phoneme_pair: str) -> Optional[Tuple[nn.Module, Any, List[str]]]:
        """
        Load model, scaler, and feature columns for a phoneme pair.
        
        Args:
            phoneme_pair: Phoneme pair name (e.g., 'b-p')
            
        Returns:
            Tuple of (model, scaler, feature_cols) or None if failed
        """
        if phoneme_pair in self.models_cache:
            return self.models_cache[phoneme_pair]
        
        model_dir = (
            self.artifacts_dir / 
            f"{phoneme_pair}_dl_models_with_context_v2" / 
            'improved_models' / 
            'hybrid_cnn_mlp_v4_3_enhanced'
        )
        
        if not model_dir.exists():
            return None
        
        try:
            # Load model
            model_path = model_dir / 'best_model.pt'
            if not model_path.exists():
                return None
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load config to get n_features
            config_path = model_dir / 'config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    n_features = config.get('n_features', 129)
            else:
                n_features = 129  # Default
            
            # Create model
            model = HybridCNNMLP_V4_3(n_features=n_features, num_classes=2)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            # Load scaler
            scaler_path = self.artifacts_dir / f"{phoneme_pair}_dl_models_with_context_v2" / 'feature_scaler.joblib'
            scaler = None
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
            
            # Load feature columns
            feature_cols_path = self.artifacts_dir / f"{phoneme_pair}_dl_models_with_context_v2" / 'feature_cols.json'
            feature_cols = []
            if feature_cols_path.exists():
                with open(feature_cols_path, 'r') as f:
                    feature_cols = json.load(f)
            
            # Exclude metadata columns
            metadata_cols = ['phoneme_id', 'class', 'duration_ms', 'phoneme', 'utterance_id', 
                           'duration_ms_features', 'start_ms', 'end_ms', 'split', 'class_encoded']
            feature_cols = [col for col in feature_cols if col not in metadata_cols]
            
            # Trim to n_features if needed
            if len(feature_cols) > n_features:
                feature_cols = feature_cols[:n_features]
            
            result = (model, scaler, feature_cols)
            self.models_cache[phoneme_pair] = result
            return result
            
        except Exception as e:
            print(f"Error loading model for {phoneme_pair}: {e}")
            return None
    
    def extract_audio_segment(
        self,
        audio: np.ndarray,
        start_ms: float,
        end_ms: float,
        sr: int = SAMPLE_RATE,
        context_ms: float = 100.0
    ) -> np.ndarray:
        """
        Extract audio segment with context.
        
        Args:
            audio: Full audio array
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            sr: Sample rate
            context_ms: Context to add before and after (milliseconds)
            
        Returns:
            Audio segment with context
        """
        start_sample = max(0, int((start_ms - context_ms) / 1000 * sr))
        end_sample = min(len(audio), int((end_ms + context_ms) / 1000 * sr))
        return audio[start_sample:end_sample]
    
    def validate_phoneme_segment(
        self,
        audio_segment: np.ndarray,
        phoneme_pair: str,
        expected_phoneme: str,
        suspected_phoneme: Optional[str] = None,
        sr: int = SAMPLE_RATE,
        is_missing: bool = False
    ) -> Dict[str, Any]:
        """
        Validate phoneme segment using trained model.
        
        Args:
            audio_segment: Audio segment (numpy array)
            phoneme_pair: Phoneme pair name (e.g., 'b-p')
            expected_phoneme: Expected correct phoneme
            suspected_phoneme: Suspected phoneme (if different from expected)
            sr: Sample rate
            is_missing: Whether the phoneme is marked as missing
            
        Returns:
            Dictionary with validation results
        """
        # Handle missing phonemes
        if is_missing:
            return {
                'expected_phoneme': expected_phoneme,
                'recognized_phoneme': suspected_phoneme,
                'is_correct': False,
                'confidence': 0.0,
                'predicted_phoneme': None,
                'probabilities': {},
                'is_missing': True
            }
        
        # Load model
        model_data = self._load_model(phoneme_pair)
        if model_data is None:
            return {
                'expected_phoneme': expected_phoneme,
                'recognized_phoneme': suspected_phoneme,
                'is_correct': None,
                'confidence': 0.0,
                'predicted_phoneme': None,
                'probabilities': {},
                'features': {},
                'error': f'Model not found for pair {phoneme_pair}'
            }
        
        model, scaler, feature_cols = model_data
        
        try:
            # Extract features directly from numpy array
            features_dict = extract_all_features(audio_segment, sr=sr, phoneme_type=phoneme_pair)
            
            # Extract spectrogram directly from numpy array
            spectrogram = extract_spectrogram_window(
                audio_segment,
                target_duration_ms=SPECTROGRAM_WINDOW_MS,
                sr=sr,
                n_mels=N_MELS,
                hop_length=HOP_LENGTH
            )
            
            if features_dict is None or spectrogram is None:
                return {
                    'is_correct': None,
                    'confidence': 0.0,
                    'predicted_phoneme': None,
                    'probabilities': {},
                    'features': {},
                    'error': 'Failed to extract features'
                }
            
            # Prepare features vector
            # First, flatten all array features in features_dict
            flattened_features = {}
            for key, val in features_dict.items():
                if isinstance(val, np.ndarray):
                    # Flatten array features and create multiple keys
                    for i, v in enumerate(val.flatten()):
                        flattened_features[f"{key}_{i}"] = float(v)
                else:
                    flattened_features[key] = float(val)
            
            # Now build features_vector from feature_cols
            features_vector = []
            for col in feature_cols:
                if col in flattened_features:
                    features_vector.append(flattened_features[col])
                else:
                    features_vector.append(0.0)
            
            features_vector = np.array(features_vector, dtype=np.float32)
            
            # Prepare spectrogram
            if len(spectrogram.shape) == 2:
                spectrogram = np.expand_dims(spectrogram, axis=0)
            
            spectrogram = spectrogram.astype(np.float32)
            
            # Normalize features
            if scaler is not None:
                # Ensure features_vector matches scaler's expected input size
                if hasattr(scaler, 'n_features_in_'):
                    if len(features_vector) != scaler.n_features_in_:
                        if len(features_vector) > scaler.n_features_in_:
                            features_vector = features_vector[:scaler.n_features_in_]
                        else:
                            features_vector = np.pad(features_vector, (0, scaler.n_features_in_ - len(features_vector)), 'constant')
                
                features_vector = scaler.transform([features_vector])[0]
                
                # Ensure features_vector matches model's n_features
                if hasattr(model, 'n_features'):
                    if len(features_vector) > model.n_features:
                        features_vector = features_vector[:model.n_features]
                    elif len(features_vector) < model.n_features:
                        features_vector = np.pad(features_vector, (0, model.n_features - len(features_vector)), 'constant')
                
                features_vector = features_vector.astype(np.float32)
            
            # Convert to tensors
            spectrogram_tensor = torch.from_numpy(spectrogram).unsqueeze(0).to(self.device)
            features_tensor = torch.from_numpy(features_vector).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = model((spectrogram_tensor, features_tensor))
                probabilities = torch.softmax(logits, dim=-1)[0]
                predicted_class = torch.argmax(logits, dim=-1)[0].item()
            
            # Get class mapping
            phonemes = phoneme_pair.split('-')
            predicted_phoneme = phonemes[predicted_class] if predicted_class < len(phonemes) else None
            
            # Check if correct
            is_correct = (predicted_phoneme == expected_phoneme)
            confidence = float(probabilities[predicted_class])
            
            # Get probabilities for both classes
            prob_dict = {}
            for i, phoneme in enumerate(phonemes):
                if i < len(probabilities):
                    prob_dict[phoneme] = float(probabilities[i])
            
            result = {
                'expected_phoneme': expected_phoneme,
                'recognized_phoneme': suspected_phoneme,
                'is_correct': is_correct,
                'confidence': confidence,
                'predicted_phoneme': predicted_phoneme,
                'probabilities': prob_dict,
                'features': features_dict,
                'is_missing': False
            }
            
            return result
            
        except Exception as e:
            import traceback
            return {
                'expected_phoneme': expected_phoneme,
                'recognized_phoneme': suspected_phoneme,
                'is_correct': None,
                'confidence': 0.0,
                'predicted_phoneme': None,
                'probabilities': {},
                'features': {},
                'error': str(e)
            }
    
    def get_available_pairs(self) -> List[str]:
        """Get list of available phoneme pairs."""
        return self.available_pairs.copy()
    
    def has_trained_model(self, expected_phoneme: str, recognized_phoneme: str) -> bool:
        """
        Check if a trained model exists for a phoneme pair.
        
        Args:
            expected_phoneme: Expected phoneme (correct one)
            recognized_phoneme: Recognized phoneme (potentially incorrect)
            
        Returns:
            True if trained model exists for this pair, False otherwise
        """
        exp_ph = expected_phoneme.strip()
        rec_ph = recognized_phoneme.strip()
        
        if exp_ph == rec_ph:
            return False
        
        pair1 = f"{exp_ph}-{rec_ph}"
        pair2 = f"{rec_ph}-{exp_ph}"
        
        if pair1 in self.available_pairs or pair2 in self.available_pairs:
            return True
        
        return False
    
    def _normalize_phoneme_notation(self, phoneme: str) -> str:
        """
        Normalize IPA phoneme notation.
        Removes '/' brackets and converts to lowercase.
        
        Args:
            phoneme: Phoneme in IPA notation (e.g., '/b/' or 'b')
            
        Returns:
            Normalized phoneme (e.g., 'b')
        """
        phoneme = phoneme.strip()
        if phoneme.startswith('/') and phoneme.endswith('/'):
            phoneme = phoneme[1:-1]
        return phoneme.lower()
    
    def _generate_explanation(
        self,
        is_correct: Optional[bool],
        predicted_phoneme: Optional[str],
        expected_phoneme: Optional[str],
        confidence: float,
        probabilities: Dict[str, float]
    ) -> str:
        """
        Generate human-readable explanation of the validation result.
        """
        if predicted_phoneme is None:
            return (
                "Unable to predict phoneme. The model could not determine the phoneme "
                "from the acoustic features."
            )
        
        if expected_phoneme is None:
            return (
                f"The model predicted phoneme '{predicted_phoneme}' "
                f"with {confidence:.1%} confidence. "
                f"Probabilities: {', '.join([f'{k}={v:.1%}' for k, v in probabilities.items()])}."
            )
        
        if is_correct is None:
            return (
                f"Validation result unclear. The model predicted '{predicted_phoneme}' "
                f"(expected '{expected_phoneme}') with {confidence:.1%} confidence. "
                f"Probabilities: {', '.join([f'{k}={v:.1%}' for k, v in probabilities.items()])}."
            )
        
        if is_correct:
            return (
                f"Correct pronunciation detected. The model identified phoneme '{predicted_phoneme}' "
                f"(expected '{expected_phoneme}') with {confidence:.1%} confidence. "
                f"The acoustic features match the expected phoneme."
            )
        else:
            return (
                f"Incorrect pronunciation detected. The model predicted '{predicted_phoneme}' "
                f"but expected '{expected_phoneme}'. Confidence: {confidence:.1%}. "
                f"Probabilities: {', '.join([f'{k}={v:.1%}' for k, v in probabilities.items()])}. "
                f"The acoustic features suggest a different phoneme than expected."
            )
    
    def validate_phoneme(
        self,
        audio: Union[str, Path, np.ndarray],
        phoneme: str,
        position_ms: float,
        expected_phoneme: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate phoneme pronunciation according to Research Brief specification.
        
        This is the main API function that matches the documentation requirements.
        
        Args:
            audio: Audio segment - can be:
                - Path to WAV file (str or Path)
                - numpy array of audio samples (16kHz, mono)
            phoneme: Target phoneme in IPA notation (e.g., '/b/' or 'b')
            position_ms: Timestamp in milliseconds indicating where the phoneme occurs
            expected_phoneme: Expected correct phoneme in IPA notation (optional)
            
        Returns:
            Dictionary with validation results:
            {
                'is_correct': bool,
                'confidence': float (0-1),
                'features': dict,
                'explanation': str
            }
        """
        try:
            # Normalize phoneme notation
            phoneme_normalized = self._normalize_phoneme_notation(phoneme)
            expected_phoneme_normalized = None
            if expected_phoneme is not None:
                expected_phoneme_normalized = self._normalize_phoneme_notation(expected_phoneme)
            
            # Load audio if it's a file path
            if isinstance(audio, (str, Path)):
                audio_path = Path(audio)
                if not audio_path.exists():
                    return {
                        'is_correct': None,
                        'confidence': 0.0,
                        'features': {},
                        'explanation': f'Audio file not found: {audio_path}',
                        'error': f'File not found: {audio_path}'
                    }
                
                audio_array, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
                
                if sr != SAMPLE_RATE:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SAMPLE_RATE)
                    sr = SAMPLE_RATE
            elif isinstance(audio, np.ndarray):
                audio_array = audio.copy()
                sr = SAMPLE_RATE
            else:
                return {
                    'is_correct': None,
                    'confidence': 0.0,
                    'features': {},
                    'explanation': f'Unsupported audio type: {type(audio)}. Expected str, Path, or numpy.ndarray.',
                    'error': f'Unsupported audio type: {type(audio)}'
                }
            
            # Validate audio duration (3-5 seconds as per spec)
            duration_ms = len(audio_array) / sr * 1000
            
            # Validate position_ms is within audio bounds
            if position_ms < 0 or position_ms > duration_ms:
                return {
                    'is_correct': None,
                    'confidence': 0.0,
                    'features': {},
                    'explanation': f'Position {position_ms}ms is outside audio bounds (0-{duration_ms:.1f}ms)',
                    'error': f'Position out of bounds: {position_ms}ms'
                }
            
            # Extract audio segment around position_ms
            window_ms = 100.0  # Default window size
            start_ms = max(0, position_ms - window_ms)
            end_ms = min(duration_ms, position_ms + window_ms)
            
            audio_segment = self.extract_audio_segment(
                audio_array,
                start_ms,
                end_ms,
                sr=sr,
                context_ms=0
            )
            
            # Determine phoneme pair
            if expected_phoneme_normalized is None:
                expected_phoneme_normalized = phoneme_normalized
            
            phoneme_pair = self.get_phoneme_pair(
                expected_phoneme_normalized,
                phoneme_normalized
            )
            
            if phoneme_pair is None:
                return {
                    'is_correct': None,
                    'confidence': 0.0,
                    'features': {},
                    'explanation': (
                        f'No model available for phoneme pair involving '
                        f'"{phoneme_normalized}" and "{expected_phoneme_normalized}". '
                        f'Available pairs: {", ".join(self.available_pairs)}.'
                    ),
                    'error': f'No model for phoneme pair'
                }
            
            # Validate using existing method
            validation_result = self.validate_phoneme_segment(
                audio_segment,
                phoneme_pair,
                expected_phoneme_normalized,
                suspected_phoneme=phoneme_normalized,
                sr=sr
            )
            
            # Extract features if available
            features = validation_result.get('features', {})
            
            # Generate explanation
            explanation = self._generate_explanation(
                validation_result.get('is_correct'),
                validation_result.get('predicted_phoneme'),
                expected_phoneme_normalized,
                validation_result.get('confidence', 0.0),
                validation_result.get('probabilities', {})
            )
            
            # Format result according to specification
            result = {
                'is_correct': validation_result.get('is_correct'),
                'confidence': validation_result.get('confidence', 0.0),
                'features': features,
                'explanation': explanation
            }
            
            # Add error if present
            if 'error' in validation_result:
                result['error'] = validation_result['error']
            
            return result
            
        except Exception as e:
            import traceback
            return {
                'is_correct': None,
                'confidence': 0.0,
                'features': {},
                'explanation': f'Error during validation: {str(e)}',
                'error': str(e),
                'traceback': traceback.format_exc()
            }


# Global instance
_validator = None


def get_validator(artifacts_dir: Optional[Path] = None) -> PhonemeValidator:
    """Get or create global validator instance."""
    global _validator
    if _validator is None:
        _validator = PhonemeValidator(artifacts_dir=artifacts_dir)
    return _validator


def validate_phoneme(
    audio: Union[str, Path, np.ndarray],
    phoneme: str,
    position_ms: float,
    expected_phoneme: Optional[str] = None,
    artifacts_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Standalone function for phoneme validation (matches Research Brief specification).
    
    This is a convenience wrapper around PhonemeValidator.validate_phoneme().
    
    Args:
        audio: Audio segment - can be path to WAV file (str/Path) or numpy array (16kHz, mono)
        phoneme: Target phoneme in IPA notation (e.g., '/b/' or 'b')
        position_ms: Timestamp in milliseconds indicating where the phoneme occurs
        expected_phoneme: Expected correct phoneme in IPA notation (optional)
        artifacts_dir: Path to artifacts directory (optional, uses default if None)
        
    Returns:
        Dictionary with validation results:
        {
            'is_correct': bool,
            'confidence': float (0-1),
            'features': dict,
            'explanation': str
        }
    """
    validator = get_validator(artifacts_dir=artifacts_dir)
    return validator.validate_phoneme(audio, phoneme, position_ms, expected_phoneme)

