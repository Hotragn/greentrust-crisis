"""Download the multilingual disaster-response dataset from Hugging Face Hub.

The dataset is `community-datasets/disaster_response_messages`, originally
curated by Figure-Eight (now Appen). It contains 26,248 multilingual disaster-
response messages spanning English, French, Haitian Creole, Spanish, and Urdu,
annotated across 36 binary aid-related categories.

Outputs (under data/):
    train.parquet       (~2.8 MB)
    validation.parquet  (~380 KB)
    test.parquet        (~380 KB)
"""
from __future__ import annotations
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR

DATA_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_URLS = {
    "train.parquet":      "https://huggingface.co/datasets/community-datasets/disaster_response_messages/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "validation.parquet": "https://huggingface.co/datasets/community-datasets/disaster_response_messages/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
    "test.parquet":       "https://huggingface.co/datasets/community-datasets/disaster_response_messages/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
}


def main():
    print("Downloading multilingual disaster-response dataset ...")
    for filename, url in PARQUET_URLS.items():
        dest = DATA_DIR / filename
        if dest.exists() and dest.stat().st_size > 100_000:
            print(f"  [skip] {filename} already exists ({dest.stat().st_size:,} bytes)")
            continue
        print(f"  [fetch] {filename} from Hugging Face Hub ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print(f"          saved {len(data):,} bytes -> {dest}")
    print("\nDataset download complete. Verify by running:")
    print("  python -c \"import pandas as pd; df=pd.read_parquet('data/train.parquet'); print(df.shape, df.columns.tolist()[:5])\"")


if __name__ == "__main__":
    main()
