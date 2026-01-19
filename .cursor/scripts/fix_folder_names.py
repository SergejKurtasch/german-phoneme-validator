#!/usr/bin/env python3
"""
Fixes remaining special characters in folder names.
"""

from pathlib import Path

def fix_folder_names(artifacts_dir: Path):
    """Fixes folder names by replacing remaining special characters."""
    
    # Mapping for replacing remaining characters
    replacements = {
        'ː': 'aa',  # long vowel
        '̯': '',     # non-syllabic marker - remove
    }
    
    folders_to_rename = {
        'aI̯-aː_model': 'aI-aaa_model',
        'aU̯-aː_model': 'aU-aaa_model',
        'aː-a_model': 'aaa-a_model',
        'eː-E_model': 'eaa-E_model',
        'iː-I_model': 'iaa-I_model',
        'oː-O_model': 'oaa-O_model',
        'uː-U_model': 'uaa-U_model',
    }
    
    for old_name, new_name in folders_to_rename.items():
        old_path = artifacts_dir / old_name
        new_path = artifacts_dir / new_name
        
        if old_path.exists() and not new_path.exists():
            print(f"📦 {old_name} → {new_name}")
            old_path.rename(new_path)
            print(f"  ✓ Renamed")
        elif old_path.exists():
            print(f"⚠️  {old_name} already renamed to {new_name}")
        else:
            print(f"⚠️  {old_name} not found")


if __name__ == '__main__':
    artifacts_dir = Path('/Volumes/SSanDisk/german-phoneme-validator/artifacts')
    print("🔧 Fixing folder names...\n")
    fix_folder_names(artifacts_dir)
    print("\n✅ Done!")
