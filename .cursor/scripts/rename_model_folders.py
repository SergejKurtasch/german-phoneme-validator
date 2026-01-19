#!/usr/bin/env python3
"""
Script for renaming model folders in artifacts.
Replaces IPA special characters with text and updates config.json.
"""

import json
import shutil
from pathlib import Path

# Mapping of IPA special characters to text replacements
PHONEME_NORMALIZATION = {
    ':': 'aa',      # long vowel
    'ɪ̯': 'Ij',      # non-syllabic i
    'ʊ': 'U',       # near-close back vowel
    'ɐ': 'A',       # near-open central vowel
    'ʁ': 'R',       # voiced uvular fricative
    'ŋ': 'N',       # velar nasal
    'ə': 'schwa',   # schwa
    'ɛ': 'E',       # open-mid front vowel
    'ɔ': 'O',       # open-mid back vowel
    'ç': 'C',       # voiceless palatal fricative
    'ʃ': 'S',       # voiceless postalveolar fricative
    'ʰ': 'h',       # aspiration
    'a': 'a',
    'b': 'b',
    'd': 'd',
    'e': 'e',
    'g': 'g',
    'i': 'i',
    'k': 'k',
    'n': 'n',
    'o': 'o',
    'p': 'p',
    's': 's',
    't': 't',
    'u': 'u',
    'x': 'x',
    'z': 'z',
    'ɪ': 'I',
}

# Original class mappings from validator.py
CLASS_MAPPING = {
    'a-ɛ': {0: 'a', 1: 'ɛ'},
    'aː-a': {0: 'a', 1: 'aː'},
    'aɪ̯-aː': {0: 'aː', 1: 'aɪ̯'},
    'aʊ̯-aː': {0: 'aː', 1: 'aʊ̯'},
    'b-p': {0: 'b', 1: 'p'},
    'd-t': {0: 'd', 1: 't'},
    'eː-ɛ': {0: 'ɛ', 1: 'eː'},
    'g-k': {0: 'g', 1: 'k'},
    'iː-ɪ': {0: 'ɪ', 1: 'iː'},
    'kʰ-g': {0: 'kʰ', 1: 'ɡ'},
    'oː-ɔ': {0: 'ɔ', 1: 'oː'},
    's-ʃ': {0: 's', 1: 'ʃ'},
    'ts-s': {0: 's', 1: 'ts'},
    'tʰ-d': {0: 'd', 1: 'tʰ'},
    'uː-ʊ': {0: 'ʊ', 1: 'uː'},
    'x-k': {0: 'k', 1: 'x'},
    'z-s': {0: 's', 1: 'z'},
    'ç-x': {0: 'x', 1: 'ç'},
    'ç-ʃ': {0: 'ç', 1: 'ʃ'},
    'ŋ-n': {0: 'n', 1: 'ŋ'},
    'ə-ɛ': {0: 'ɛ', 1: 'ə'},
    'ʁ-ɐ': {0: 'ɐ', 1: 'ʁ'},
}


def normalize_phoneme(phoneme: str) -> str:
    """Normalizes phoneme by replacing special characters with text."""
    result = []
    i = 0
    while i < len(phoneme):
        # Check multi-character sequences first
        found = False
        for multi_char in ['aɪ̯', 'aʊ̯', 'kʰ', 'tʰ', 'aː', 'eː', 'iː', 'oː', 'uː']:
            if phoneme[i:].startswith(multi_char):
                # Normalize each character
                for char in multi_char:
                    result.append(PHONEME_NORMALIZATION.get(char, char))
                i += len(multi_char)
                found = True
                break
        
        if not found:
            char = phoneme[i]
            result.append(PHONEME_NORMALIZATION.get(char, char))
            i += 1
    
    return ''.join(result)


def normalize_pair_name(pair_name: str) -> str:
    """Normalizes phoneme pair name."""
    parts = pair_name.split('-')
    if len(parts) != 2:
        return pair_name
    
    normalized_parts = [normalize_phoneme(part) for part in parts]
    return '-'.join(normalized_parts)


def update_config_json(config_path: Path, original_pair: str, class_mapping: dict):
    """Updates config.json by adding phoneme_pair and class_mapping."""
    if not config_path.exists():
        print(f"  ⚠️  Config not found: {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Add new information
    config['phoneme_pair'] = original_pair
    config['class_mapping'] = class_mapping
    config['class_mapping_description'] = {
        '0': f"Class 0 corresponds to phoneme '{class_mapping[0]}'",
        '1': f"Class 1 corresponds to phoneme '{class_mapping[1]}'"
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Updated config.json")


def rename_model_folders(artifacts_dir: Path):
    """Renames model folders and updates config.json."""
    if not artifacts_dir.exists():
        print(f"❌ Artifacts directory not found: {artifacts_dir}")
        return
    
    # Find all model folders
    model_folders = [
        d for d in artifacts_dir.iterdir()
        if d.is_dir() and d.name.endswith('_dl_models_with_context_v2')
    ]
    
    if not model_folders:
        print("❌ No model folders found")
        return
    
    print(f"📁 Found {len(model_folders)} model folders\n")
    
    rename_mapping = {}  # Old name -> new name for updating references
    
    for old_folder in sorted(model_folders):
        # Extract original pair name
        original_pair = old_folder.name.replace('_dl_models_with_context_v2', '')
        
        # Normalize name
        normalized_pair = normalize_pair_name(original_pair)
        new_folder_name = f"{normalized_pair}_model"
        new_folder = old_folder.parent / new_folder_name
        
        print(f"📦 {old_folder.name}")
        print(f"   → {new_folder_name}")
        
        # Rename folder
        if new_folder.exists():
            print(f"  ⚠️  New folder already exists, skipping")
            continue
        
        old_folder.rename(new_folder)
        print(f"  ✓ Folder renamed")
        
        # Save mapping
        rename_mapping[original_pair] = normalized_pair
        
        # Update config.json (after flattening structure, files are in folder root)
        # Note: at the time this script was executed, files were still in subfolder,
        # but now they are moved to root. For compatibility, check both locations.
        config_path = new_folder / 'config.json'
        if not config_path.exists():
            # Fallback for case when script is run before flattening
            config_path = new_folder / 'improved_models' / 'hybrid_cnn_mlp_v4_3_enhanced' / 'config.json'
        
        if original_pair in CLASS_MAPPING:
            if config_path.exists():
                update_config_json(config_path, original_pair, CLASS_MAPPING[original_pair])
            else:
                print(f"  ⚠️  Config.json not found at path: {config_path}")
        else:
            print(f"  ⚠️  Class mapping not found for {original_pair}")
        
        print()
    
    # Save mapping for code updates
    mapping_file = artifacts_dir.parent / '.cursor' / 'scripts' / 'folder_rename_mapping.json'
    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(rename_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Renaming completed!")
    print(f"📝 Mapping saved to: {mapping_file}")
    print(f"\n📋 Renaming summary:")
    for old, new in sorted(rename_mapping.items()):
        print(f"   {old} → {new}")


if __name__ == '__main__':
    # Determine path to artifacts
    script_dir = Path(__file__).parent
    artifacts_dir = script_dir.parent.parent / 'artifacts'
    
    print("🚀 Starting model folder renaming...\n")
    rename_model_folders(artifacts_dir)
