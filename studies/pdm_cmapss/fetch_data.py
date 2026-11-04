"""Download and verify the NASA C-MAPSS turbofan data (Prognostics Center of Excellence repository) into data/.

Everything downloaded is treated as untrusted: each archive lands in its own new directory, files are extracted by explicit name
only (never by archive-supplied path), nothing is executed, and every byte is checked against the SHA-256 pins below.

    python3 -I studies/pdm_cmapss/fetch_data.py                     # download if needed, verify, extract
    python3 -I studies/pdm_cmapss/fetch_data.py --verify            # verify only, never touch the network
    python3 -I studies/pdm_cmapss/fetch_data.py --from-zip FILE     # use a local copy of the outer archive instead of downloading
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
OUTER_SHA = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"       # 12,429,152 bytes
NESTED_NAME = "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip"                       # member name inside the outer archive
NESTED_SHA = "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f"                         # 12,425,978 bytes
FILES = {
    "readme.txt": "4f5270554b775c67e73aff383c5436fd329d6e4cc3d3a116913276fae511269b",
    "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
    "test_FD001.txt": "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851",
    "RUL_FD001.txt": "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca",
    "train_FD002.txt": "dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f",
    "test_FD002.txt": "de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02",
    "RUL_FD002.txt": "c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b",
    "train_FD003.txt": "2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2",
    "test_FD003.txt": "299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790",
    "RUL_FD003.txt": "df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb",
    "train_FD004.txt": "27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218",
    "test_FD004.txt": "1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5",
    "RUL_FD004.txt": "196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1",
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_files(root: Path) -> list[str]:
    """Names of expected files that are missing or whose SHA-256 differs (empty list = the data set is intact)."""
    bad = []
    for name, sha in FILES.items():
        p = root / name
        if not p.is_file() or sha256_file(p) != sha:
            bad.append(name)
    return bad


def _extract(outer: bytes, root: Path) -> None:
    if sha256_bytes(outer) != OUTER_SHA:
        raise SystemExit(f"outer archive hash mismatch: got {sha256_bytes(outer)}, expected {OUTER_SHA}")
    with zipfile.ZipFile(io.BytesIO(outer)) as z:
        if NESTED_NAME not in z.namelist():
            raise SystemExit(f"{NESTED_NAME} not found in the outer archive")
        nested = z.read(NESTED_NAME)
    if sha256_bytes(nested) != NESTED_SHA:
        raise SystemExit(f"nested archive hash mismatch: got {sha256_bytes(nested)}, expected {NESTED_SHA}")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(nested)) as z:
        names = set(z.namelist())
        for name, sha in FILES.items():
            if name not in names:
                raise SystemExit(f"{name} missing from the nested archive")
            data = z.read(name)                              # explicit name only: archive-supplied paths are never used
            if sha256_bytes(data) != sha:
                raise SystemExit(f"{name}: hash mismatch (got {sha256_bytes(data)})")
            (root / name).write_bytes(data)
    (root / "SHA256SUMS.json").write_text(json.dumps(FILES, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2] / "data" / "cmapss"))
    ap.add_argument("--verify", action="store_true", help="verify only; never download")
    ap.add_argument("--from-zip", default=None, help="local copy of the outer archive (skips the download)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    bad = verify_files(root)
    if not bad:
        print(f"OK: {len(FILES)} files in {root} match the pinned SHA-256 values")
        return 0
    if a.verify:
        print("FAILED: missing or modified:", ", ".join(bad))
        return 1
    if a.from_zip:
        outer = Path(a.from_zip).read_bytes()
    else:
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL, timeout=120) as r:
            outer = r.read()
    _extract(outer, root)
    bad = verify_files(root)
    if bad:
        print("FAILED after extraction:", ", ".join(bad))
        return 1
    print(f"OK: extracted and verified {len(FILES)} files into {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
