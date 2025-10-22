from pathlib import Path
from src import PROJECT_ROOT

# Data paths
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_PATH = DATA_DIR / 'raw' / 'Career Mode player datasets - FIFA 15-22.xlsx'
PROCESSED_DATA_PATH = DATA_DIR / 'processed' / 'players_cleaned.pkl'

# Create directories if missing
PROCESSED_DATA_PATH.parent.mkdir(exist_ok=True, parents=True)
