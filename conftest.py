"""
conftest.py — pytest configuration for ClarifAI-Med.
Adds the project root to sys.path so imports work correctly.
"""

import sys
from pathlib import Path

# Add the project root to sys.path so `from src.xxx import yyy` works
sys.path.insert(0, str(Path(__file__).resolve().parent))
