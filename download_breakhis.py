"""
Download the BreakHis dataset archive with resume + retry support.

Usage:
    python download_breakhis.py

If the download stalls or your connection drops, just re-run the script -
it will resume from where it left off (as long as the server supports
range requests, which most static file servers do).
"""

import os
import tarfile
import time
import requests

URL = "http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"
OUT_PATH = "BreaKHis_v1.tar.gz"
EXTRACT_DIR = "."
CHUNK_SIZE = 1024 * 1024  # 1 MB
MAX_RETRIES = 10
TIMEOUT = 60  # seconds per request

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def get_remote_size():
    resp = requests.head(URL, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
    resp.raise_for_status()
    return int(resp.headers.get("Content-Length", 0))


def download():
    remote_size = None
    try:
        remote_size = get_remote_size()
        print(f"Remote file size: {remote_size / (1024**3):.2f} GB")
    except Exception as e:
        print(f"Could not get remote size via HEAD request ({e}); continuing anyway.")

    attempt = 0
    while attempt < MAX_RETRIES:
        attempt += 1
        existing = os.path.getsize(OUT_PATH) if os.path.exists(OUT_PATH) else 0

        if remote_size and existing >= remote_size:
            print("Download already complete.")
            return

        headers = dict(HEADERS)
        mode = "wb"
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            mode = "ab"
            print(f"Resuming from byte {existing}...")

        try:
            with requests.get(URL, headers=headers, stream=True, timeout=TIMEOUT) as r:
                if r.status_code not in (200, 206):
                    print(f"Unexpected status code {r.status_code}, retrying...")
                    time.sleep(5)
                    continue

                downloaded = existing
                with open(OUT_PATH, mode) as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if remote_size:
                            pct = downloaded / remote_size * 100
                            print(f"\r{downloaded / (1024**2):.1f} MB / "
                                  f"{remote_size / (1024**2):.1f} MB ({pct:.1f}%)",
                                  end="", flush=True)
                        else:
                            print(f"\r{downloaded / (1024**2):.1f} MB downloaded",
                                  end="", flush=True)

            print("\nDownload finished.")
            if remote_size is None or os.path.getsize(OUT_PATH) >= remote_size:
                return

        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout) as e:
            print(f"\nConnection issue ({e}). Retrying in 5s "
                  f"(attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(5)

    print("Max retries reached. Re-run the script to keep resuming.")


def extract():
    """Extract the downloaded archive so index_dataset() can find the .png files."""
    if not os.path.exists(OUT_PATH):
        print(f"{OUT_PATH} not found; nothing to extract.")
        return
    marker = os.path.join(EXTRACT_DIR, "BreaKHis_v1")
    if os.path.isdir(marker):
        print(f"{marker} already exists; skipping extraction.")
        return
    print(f"Extracting {OUT_PATH} to {EXTRACT_DIR}/ ...")
    with tarfile.open(OUT_PATH, "r:gz") as tar:
        tar.extractall(EXTRACT_DIR)
    print("Extraction finished.")


if __name__ == "__main__":
    download()
    extract()