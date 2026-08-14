import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

import kaggle


def download_dataset(
    dataset="naserabdullahalam/phishing-email-dataset",
    file_name="CEAS_08.csv",
    path="data/raw",
):
    """
    Downloads only `file_name` from the Kaggle dataset into `path`,
    skipping the download if it's already there.
    """
    expected_path = os.path.join(path, file_name)

    if os.path.exists(expected_path):
        print(f"'{file_name}' already exists at '{expected_path}', skipping download.")
        return

    print(f"Downloading '{file_name}' from '{dataset}' into '{path}'...")
    kaggle.api.dataset_download_file(dataset, file_name=file_name, path=path)
    print("Download complete.")

    # Kaggle sometimes delivers single-file downloads as a .zip — unzip if needed
    zip_path = expected_path + ".zip"
    if os.path.exists(zip_path):
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(path)
        os.remove(zip_path)
        print(f"Unzipped and cleaned up '{zip_path}'.")

    if os.path.exists(expected_path):
        print(f"Confirmed: '{file_name}' is now present.")
    else:
        print(f"Warning: '{file_name}' not found at '{expected_path}' after download.")

# Download the dataset when this script is run directly, not when imported
if __name__ == "__main__":
    download_dataset()


df = pd.read_csv("data/raw/CEAS_08.csv")

df.info()
