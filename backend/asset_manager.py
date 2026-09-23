"""
Fetch the prebuilt index at boot when ASSET_MODE=remote.

HF_CACHE_URL points at a .tar.gz of the index_cache folder (files at the root
or under one top-level folder — both work). Download and extraction happen in
a temp dir beside the target, which is renamed into place only when complete,
so a crash mid-download never leaves a half-populated cache that a later boot
mistakes for a real one.
"""

import os
import shutil
import tarfile
import tempfile

import requests

from config import settings


def extract_cache(tarball: str, dest: str) -> None:
    # filter="data" rejects absolute paths, "..", links out of dest and device
    # files — without it a hostile tarball can write anywhere (CVE-2007-4559).
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(dest, filter="data")


def ensure_index_cache(cache_dir: str) -> None:
    if settings.ASSET_MODE == "local":
        print("Using local assets.")
        return
    if os.path.exists(os.path.join(cache_dir, "faiss.index")):
        print(f"{cache_dir} already present.")
        return
    if not settings.HF_CACHE_URL:
        raise ValueError("ASSET_MODE=remote but HF_CACHE_URL is not set")

    parent = os.path.dirname(os.path.abspath(cache_dir))
    os.makedirs(parent, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as tmp:
        tarball = os.path.join(tmp, "index_cache.tar.gz")
        print(f"Downloading index cache from {settings.HF_CACHE_URL} ...")
        with requests.get(settings.HF_CACHE_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tarball, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

        print("Extracting index cache...")
        root = os.path.join(tmp, "extracted")
        extract_cache(tarball, root)
        if not os.path.exists(os.path.join(root, "faiss.index")):
            entries = os.listdir(root)
            if len(entries) == 1:
                root = os.path.join(root, entries[0])

        shutil.rmtree(cache_dir, ignore_errors=True)  # leftovers without faiss.index
        os.replace(root, cache_dir)
    print("Index cache ready.")
