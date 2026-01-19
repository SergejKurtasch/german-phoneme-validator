#!/usr/bin/env python3
"""
Script to translate class_mapping_description in all config.json files to English.
"""

import json
from pathlib import Path


def translate_config_description(config_path: Path):
    """Translates class_mapping_description to English."""
    if not config_path.exists():
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    if 'class_mapping' not in config:
        return False
    
    class_mapping = config.get('class_mapping', {})
    
    # Update description to English
    config['class_mapping_description'] = {
        '0': f"Class 0 corresponds to phoneme '{class_mapping.get('0', '')}'",
        '1': f"Class 1 corresponds to phoneme '{class_mapping.get('1', '')}'"
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return True


def translate_all_configs(artifacts_dir: Path):
    """Translates all config.json files in model folders."""
    if not artifacts_dir.exists():
        print(f"❌ Artifacts directory not found: {artifacts_dir}")
        return
    
    model_folders = [
        d for d in artifacts_dir.iterdir()
        if d.is_dir() and d.name.endswith('_model')
    ]
    
    if not model_folders:
        print("❌ No model folders found")
        return
    
    print(f"📁 Found {len(model_folders)} model folders\n")
    
    translated_count = 0
    for model_folder in sorted(model_folders):
        config_path = model_folder / 'config.json'
        if translate_config_description(config_path):
            print(f"✓ Translated: {model_folder.name}")
            translated_count += 1
        else:
            print(f"⚠️  Skipped: {model_folder.name}")
    
    print(f"\n✅ Translation completed!")
    print(f"📊 Translated: {translated_count} of {len(model_folders)} configs")


if __name__ == '__main__':
    artifacts_dir = Path('/Volumes/SSanDisk/german-phoneme-validator/artifacts')
    print("🚀 Translating config.json descriptions...\n")
    translate_all_configs(artifacts_dir)
