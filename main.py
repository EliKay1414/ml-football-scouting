from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

print(f"Raw data path: {RAW_DATA_PATH}")
print(f"Path exists: {RAW_DATA_PATH.exists()}")
print(f"Will save processed data to: {PROCESSED_DATA_PATH}")