import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 256))
