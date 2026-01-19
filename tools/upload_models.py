"""
Script for uploading model artifacts to Hugging Face Hub.

This script uploads all models from the local artifacts/ directory to a Hugging Face repository.
It checks for problematic characters in folder names that may cause issues on Windows.

Usage:
    python tools/upload_models.py --repo-id SergejKurtasch/german-phoneme-models --artifacts-dir artifacts/
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import warnings

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("ERROR: huggingface_hub is not installed. Install it with: pip install huggingface_hub")
    sys.exit(1)


# Windows-unsafe characters that should be warned about
WINDOWS_UNSAFE_CHARS = [':', '\\', '/', '*', '?', '"', '<', '>', '|']


def check_folder_names(artifacts_dir: Path) -> List[str]:
    """
    Check folder names for Windows-unsafe characters.
    
    Args:
        artifacts_dir: Path to artifacts directory
        
    Returns:
        List of folder names with problematic characters
    """
    problematic = []
    
    if not artifacts_dir.exists():
        return problematic
    
    for item in artifacts_dir.iterdir():
        if item.is_dir():
            folder_name = item.name
            for char in WINDOWS_UNSAFE_CHARS:
                if char in folder_name:
                    problematic.append(folder_name)
                    break
    
    return problematic


def upload(
    repo_id: str,
    artifacts_dir: Path,
    token: Optional[str] = None,
    repo_type: str = "model"
) -> None:
    """
    Upload models to Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repository ID (e.g., 'username/repo-name')
        artifacts_dir: Path to local artifacts directory
        token: Hugging Face token (optional, will use cached token if not provided)
        repo_type: Type of repository ('model', 'dataset', etc.)
    """
    artifacts_dir = Path(artifacts_dir)
    
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
    
    # Check for problematic folder names
    problematic = check_folder_names(artifacts_dir)
    if problematic:
        warnings.warn(
            f"Found {len(problematic)} folder(s) with Windows-unsafe characters: {problematic}. "
            f"These may cause issues on Windows systems. Consider renaming them to use safe "
            f"characters (e.g., replace ':' with '_').",
            UserWarning
        )
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Upload cancelled.")
            return
    
    # Initialize HF API
    api = HfApi()
    
    # Login if token provided
    if token:
        login(token=token)
    elif not api.token:
        # Try to use cached token
        print("No token provided. Attempting to use cached Hugging Face token...")
        try:
            login()
        except Exception as e:
            print(f"ERROR: Failed to authenticate. Please provide a token or run 'huggingface-cli login'")
            raise
    
    print(f"Uploading models from {artifacts_dir} to {repo_id}...")
    print(f"Repository type: {repo_type}")
    
    # Count model folders
    model_folders = [d for d in artifacts_dir.iterdir() if d.is_dir() and d.name.endswith('_model')]
    print(f"Found {len(model_folders)} model folder(s) to upload.")
    
    # Upload entire artifacts directory
    try:
        api.upload_folder(
            folder_path=str(artifacts_dir),
            repo_id=repo_id,
            repo_type=repo_type,
            ignore_patterns=[".git*", "__pycache__", "*.pyc"],
        )
        print(f"✓ Successfully uploaded models to {repo_id}")
    except Exception as e:
        print(f"ERROR: Failed to upload models: {e}")
        raise


def main():
    """Main entry point for the upload script."""
    parser = argparse.ArgumentParser(
        description="Upload model artifacts to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload with explicit token
  python tools/upload_models.py --repo-id SergejKurtasch/german-phoneme-models --artifacts-dir artifacts/ --token YOUR_TOKEN

  # Upload using cached token (after running 'huggingface-cli login')
  python tools/upload_models.py --repo-id SergejKurtasch/german-phoneme-models --artifacts-dir artifacts/
        """
    )
    
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repository ID (e.g., 'username/repo-name')"
    )
    
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Path to local artifacts directory (default: artifacts/)"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (optional, will use cached token if not provided)"
    )
    
    parser.add_argument(
        "--repo-type",
        type=str,
        default="model",
        choices=["model", "dataset", "space"],
        help="Type of repository (default: model)"
    )
    
    args = parser.parse_args()
    
    try:
        upload(
            repo_id=args.repo_id,
            artifacts_dir=args.artifacts_dir,
            token=args.token,
            repo_type=args.repo_type
        )
    except KeyboardInterrupt:
        print("\nUpload cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
