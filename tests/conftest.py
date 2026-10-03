"""Make the project root importable as ``src`` when running ``pytest`` from anywhere."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
