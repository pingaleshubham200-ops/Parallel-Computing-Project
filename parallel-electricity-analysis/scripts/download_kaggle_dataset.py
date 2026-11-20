#!/usr/bin/env python3
"""
Download script for the Smart Grid Real-Time Load Monitoring Dataset from Kaggle.
Uses the official kagglehub library as specified:
    kagglehub.dataset_download("ziya07/smart-grid-real-time-load-monitoring-dataset")
"""

import os
import shutil
import sys

def main():
    try:
        import kagglehub
    except ImportError:
        print("[!] kagglehub is not installed. Installing it now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        import kagglehub

    dataset_handle = "ziya07/smart-grid-real-time-load-monitoring-dataset"
    print(f"[*] Downloading dataset using kagglehub: {dataset_handle} ...")
    download_dir = kagglehub.dataset_download(dataset_handle)
    print(f"[+] Downloaded dataset to cache: {download_dir}")

    # Determine target path inside project
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Search for smart_grid_dataset.csv in downloaded directory
    found_csv = False
    for root, _, files in os.walk(download_dir):
        for f in files:
            if f.endswith(".csv"):
                src_path = os.path.join(root, f)
                dst_path = os.path.join(data_dir, f)
                shutil.copy2(src_path, dst_path)
                print(f"[+] Successfully copied {f} to: {dst_path}")
                found_csv = True

    if not found_csv:
        print(f"[-] No CSV file found in download directory: {download_dir}")
        sys.exit(1)

    print("[+] Dataset setup complete and ready for parallel analysis!")

if __name__ == "__main__":
    main()
