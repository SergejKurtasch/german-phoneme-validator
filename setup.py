"""
Setup script for German Phoneme Pronunciation Validator package.
"""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, comments, and section headers
            if not line or line.startswith("#") or line.startswith("="):
                continue
            # Extract package name and version (remove inline comments)
            match = re.match(r"^([^#]+)", line)
            if match:
                req = match.group(1).strip()
                if req:
                    requirements.append(req)

# Core requirements (exclude optional dependencies)
core_requirements = [
    req for req in requirements
    if not any(opt in req.lower() for opt in ["parselmouth", "webrtcvad", "pandas", "tqdm", "torchaudio"])
]

setup(
    name="german-phoneme-validator",
    version="1.0.0",
    description="A production-ready Python module for acoustic validation of German phoneme pronunciation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sergej Kurtasch",
    author_email="sergej.kurtasch@gmail.com",
    url="https://github.com/SergejKurtasch/german-phoneme-validator",
    license="MIT",
    # Map Python package name (with underscores) to source directory (with hyphens)
    # This allows: from german_phoneme_validator import ...
    package_dir={"german_phoneme_validator": "."},
    packages=["german_phoneme_validator", "german_phoneme_validator.core"],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        "optional": [
            "parselmouth>=0.4.0",
            "webrtcvad>=2.0.10",
            "pandas>=1.3.0",
            "tqdm>=4.64.0",
            "torchaudio>=2.0.0",
        ],
        "all": requirements,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="phoneme validation, speech recognition, german language, acoustic analysis, pronunciation assessment",
    project_urls={
        "Documentation": "https://github.com/SergejKurtasch/german-phoneme-validator#readme",
        "Source": "https://github.com/SergejKurtasch/german-phoneme-validator",
        "Tracker": "https://github.com/SergejKurtasch/german-phoneme-validator/issues",
    },
)
