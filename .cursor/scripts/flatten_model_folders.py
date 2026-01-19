#!/usr/bin/env python3
"""
Script for flattening model folder structure.
Moves files from improved_models/hybrid_cnn_mlp_v4_3_enhanced/ to model folder root.
"""

import shutil
from pathlib import Path


def flatten_model_folder(model_folder: Path):
    """Simplifies model folder structure by moving files to root."""
    print(f"📦 Processing: {model_folder.name}")
    
    # Path to nested folder
    nested_dir = model_folder / 'improved_models' / 'hybrid_cnn_mlp_v4_3_enhanced'
    
    if not nested_dir.exists():
        print(f"  ⚠️  Subfolder not found, skipping")
        return False
    
    # Files to move
    files_to_move = {
        'best_model.pt': model_folder / 'best_model.pt',
        'config.json': model_folder / 'config.json'
    }
    
    moved = False
    for filename, target_path in files_to_move.items():
        source_path = nested_dir / filename
        if source_path.exists():
            if target_path.exists():
                print(f"  ⚠️  File {filename} already exists in root, skipping")
            else:
                shutil.move(str(source_path), str(target_path))
                print(f"  ✓ Moved {filename}")
                moved = True
        else:
            print(f"  ⚠️  File {filename} not found in subfolder")
    
    # Remove empty subfolders
    if moved:
        try:
            shutil.rmtree(model_folder / 'improved_models')
            print(f"  ✓ Removed nested folders")
        except Exception as e:
            print(f"  ⚠️  Failed to remove nested folders: {e}")
    
    return moved


def flatten_all_model_folders(artifacts_dir: Path):
    """Simplifies structure of all model folders."""
    if not artifacts_dir.exists():
        print(f"❌ Artifacts directory not found: {artifacts_dir}")
        return
    
    # Find all model folders
    model_folders = [
        d for d in artifacts_dir.iterdir()
        if d.is_dir() and d.name.endswith('_model')
    ]
    
    if not model_folders:
        print("❌ No model folders found")
        return
    
    print(f"📁 Found {len(model_folders)} model folders\n")
    
    flattened_count = 0
    for model_folder in sorted(model_folders):
        if flatten_model_folder(model_folder):
            flattened_count += 1
        print()
    
    print(f"✅ Processing completed!")
    print(f"📊 Flattened structures: {flattened_count} of {len(model_folders)}")


if __name__ == '__main__':
    artifacts_dir = Path('/Volumes/SSanDisk/german-phoneme-validator/artifacts')
    print("🚀 Flattening model folder structure...\n")
    flatten_all_model_folders(artifacts_dir)
