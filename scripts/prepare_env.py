#!/usr/bin/env python3
"""
Environment check and setup script.
Verifies Python, PyTorch, MPS, and all dependencies.
"""

import sys


def check_env():
    """Check all environment requirements and print summary."""
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)

    # Python
    print(f"\nPython: {sys.version}")
    assert sys.version_info >= (3, 9), "Python 3.9+ required"

    # PyTorch
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    if not torch.backends.mps.is_available():
        print("WARNING: MPS not available, will use CPU")

    # Ultralytics
    import ultralytics
    print(f"Ultralytics: {ultralytics.__version__}")

    # OpenCV
    import cv2
    print(f"OpenCV: {cv2.__version__}")

    # Other deps
    import lxml
    print(f"lxml: {lxml.__version__}")

    import matplotlib
    print(f"matplotlib: {matplotlib.__version__}")

    import tqdm
    print(f"tqdm: {tqdm.__version__}")

    import yaml
    print(f"PyYAML: {yaml.__version__}")

    # Device selection
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nSelected device: {device}")

    # Memory info
    if device == "mps":
        print("Apple Silicon GPU detected - using Metal Performance Shaders")

    print("\n" + "=" * 60)
    print("Environment check passed!")
    print("=" * 60)

    return device


if __name__ == "__main__":
    device = check_env()
    print(f"\nDevice for training: {device}")
