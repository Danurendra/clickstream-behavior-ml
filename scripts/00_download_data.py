"""
Download Clickstream Data for Online Shopping directly from UCI file URL.

Reason:
The dataset exists in UCI, but it is not available through ucimlrepo fetch_ucirepo().
Therefore, we download the ZIP file directly from the UCI static file server.

Run:
    python scripts/00_download_data.py
"""

from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import shutil


DATA_URL = "https://archive.ics.uci.edu/static/public/553/clickstream%2Bdata%2Bfor%2Bonline%2Bshopping.zip"

RAW_DIR = Path("data/raw")
ZIP_PATH = RAW_DIR / "clickstream_data_for_online_shopping.zip"
CSV_OUTPUT_PATH = RAW_DIR / "clickstream_raw.csv"
DESCRIPTION_OUTPUT_PATH = RAW_DIR / "clickstream_data_description.txt"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset ZIP file from UCI...")
    urlretrieve(DATA_URL, ZIP_PATH)
    print(f"Downloaded ZIP file to: {ZIP_PATH}")

    print("Extracting ZIP file...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(RAW_DIR)

    csv_files = list(RAW_DIR.glob("*.csv"))
    txt_files = list(RAW_DIR.glob("*.txt"))

    if not csv_files:
        raise FileNotFoundError("No CSV file found after extracting the dataset ZIP.")

    original_csv = csv_files[0]
    shutil.copy(original_csv, CSV_OUTPUT_PATH)

    print(f"CSV dataset saved to: {CSV_OUTPUT_PATH}")

    if txt_files:
        original_description = txt_files[0]
        shutil.copy(original_description, DESCRIPTION_OUTPUT_PATH)
        print(f"Data description saved to: {DESCRIPTION_OUTPUT_PATH}")

    print("Dataset download completed successfully.")


if __name__ == "__main__":
    main()