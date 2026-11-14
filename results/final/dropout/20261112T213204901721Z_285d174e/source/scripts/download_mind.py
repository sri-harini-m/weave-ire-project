import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

HF_BASE = "https://huggingface.co/datasets/huyva/MIND-small/resolve/main"
URLS = {
    "train": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
    "dev": "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
}


def download_file(url: str, dest: Path):
    print(f"Downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r  {downloaded / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="", flush=True)
        print()


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    raw = Path(cfg["paths"]["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev"):
        out_dir = raw / split
        out_dir.mkdir(parents=True, exist_ok=True)
        files = ("news.tsv", "behaviors.tsv")
        if all((out_dir / f).exists() for f in files):
            print(f"[{split}] already present, skipping")
            continue

        # Try Azure zip first; fallback to HF direct files
        zip_path = raw / f"MINDsmall_{split}.zip"
        url = URLS[split]
        success = False
        try:
            print(f"[{split}] attempting download from Azure: {url}")
            download_file(url, zip_path)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(out_dir)
            zip_path.unlink()
            success = True
        except Exception as e:
            print(f"\nAzure download failed: {e}. Falling back to Hugging Face mirror...")
            if zip_path.exists():
                zip_path.unlink()

        if not success:
            for f in files:
                target = out_dir / f
                if target.exists():
                    continue
                hf_url = f"{HF_BASE}/{split}/{f}"
                download_file(hf_url, target)

    for split in URLS:
        for f in ("news.tsv", "behaviors.tsv"):
            assert (raw / split / f).exists(), f"missing {raw / split / f}"
    print("MIND-small ready in", raw)


if __name__ == "__main__":
    main()
