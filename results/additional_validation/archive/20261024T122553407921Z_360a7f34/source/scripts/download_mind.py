import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

URLS = {
    "train": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
    "dev": "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
}


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    raw = Path(cfg["paths"]["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)

    for split, url in URLS.items():
        out_dir = raw / split
        if (out_dir / "news.tsv").exists():
            print(f"[{split}] already present, skipping")
            continue
        zip_path = raw / f"MINDsmall_{split}.zip"
        print(f"[{split}] downloading {url}")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            print(f"\nDownload failed: {e}\n"
                  "Get the two zips manually (see README, 'If the download fails'),\n"
                  f"put them in {raw}/ and rerun this script.")
            if not zip_path.exists():
                sys.exit(1)
        print(f"[{split}] unzipping")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
        zip_path.unlink()

    for split in URLS:
        for f in ("news.tsv", "behaviors.tsv"):
            assert (raw / split / f).exists(), f"missing {raw / split / f}"
    print("MIND-small ready in", raw)


if __name__ == "__main__":
    main()
