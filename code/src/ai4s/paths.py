"""Where code, data and run outputs live. Override the data folder with the AI4S_DATA environment variable
(for example on Kaggle or a rented machine)."""
import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("AI4S_DATA", CODE_ROOT.parent / "data"))
RUNS_ROOT = Path(os.environ.get("AI4S_RUNS", CODE_ROOT / "runs"))
SPLITS_ROOT = CODE_ROOT / "splits"

OOC_DIR = DATA_ROOT / "ooc"
BBBC039_DIR = DATA_ROOT / "bbbc039"
