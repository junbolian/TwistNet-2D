import argparse
import os
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

DTD_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"

def download(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[download] Exists: {out_path}")
        return

    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))

def extract(tar_path: Path, out_dir: Path):
    print(f"[extract] {tar_path} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(out_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data", help="Folder to store downloaded archives and extracted dataset")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    tar_path = data_dir / "dtd-r1.0.1.tar.gz"
    download(DTD_URL, tar_path)
    extract(tar_path, data_dir)

    dtd_root = data_dir / "dtd"
    ok = (dtd_root / "images").exists() and (dtd_root / "labels").exists()
    print(f"[done] DTD root: {dtd_root} | OK={ok}")
    if not ok:
        print("Expected structure: data/dtd/images and data/dtd/labels")

if __name__ == "__main__":
    main()
