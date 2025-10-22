# src/data_loader.py
import pandas as pd
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_fifa_data(file_path: str, sample_frac: float = None) -> pd.DataFrame:
    """Main data loading function"""
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        logger.info(f"Loading data from: {path}")
        
        all_sheets = pd.read_excel(path, sheet_name=None, engine='openpyxl')
        dfs = []
        
        for sheet_name, df in all_sheets.items():
            year_match = re.search(r'(\d{2})', sheet_name)
            if not year_match:
                continue
            year = 2000 + int(year_match.group(1))
            df['year'] = year
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        if sample_frac:
            combined_df = combined_df.sample(frac=sample_frac, random_state=42)
        
        logger.info(f"Loaded {len(combined_df)} rows")
        return combined_df
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    test_data = load_fifa_data(
        r"..\data\raw\Career Mode player datasets - FIFA 15-22.xlsx",
        sample_frac=0.01
    )
    print(test_data.head())