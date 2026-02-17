import argparse
import sys
import os
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

PDB_URL = "https://files.rcsb.org/download/{id}.pdb"
MMCIF_URL = "https://files.rcsb.org/download/{id}.cif"

def fetch(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "python-urllib"})
    with urlopen(req, timeout=60) as resp:
        return resp.read()

def is_placeholder(p: Path) -> bool:
    return str(p) in {"/path/to", "/path", "/path/to/rcsb"}

def ensure_out_dir(out_dir: Path) -> None:
    if is_placeholder(out_dir):
        raise OSError("Output path looks like a placeholder. Pick a real folder, e.g., ./rcsb or ~/Downloads/rcsb")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise OSError(f"Cannot create directory '{out_dir}': Permission denied. "
                      f"Try a folder in your home, e.g., ./rcsb") from e
    except OSError as e:
        raise OSError(f"Cannot create directory '{out_dir}': {e}") from e
    testfile = out_dir / ".write_test"
    try:
        testfile.write_text("ok")
        testfile.unlink(missing_ok=True)
    except Exception as e:
        raise OSError(f"Directory '{out_dir}' is not writable: {e}") from e

def load_ids(path: Path):
    with path.open() as f:
        ids = [line.strip() for line in f if line.strip()]
    clean = []
    for pid in ids:
        pid_up = pid.upper()
        if not pid_up.isalnum():
            print(f"WARNING: skipping invalid ID '{pid}'", file=sys.stderr)
            continue
        clean.append(pid_up)
    if not clean:
        raise ValueError("No valid IDs found in ids file.")
    return clean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="Text file with one PDB ID per line")
    ap.add_argument("--out-dir", required=True, help="Folder to save files (use quotes if path has spaces)")
    ap.add_argument("--format", choices=["pdb","mmcif"], default="pdb")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ids_path = Path(os.path.expanduser(args.ids)).resolve()
    out_dir = Path(os.path.expanduser(args.out_dir)).resolve()

    if not ids_path.exists():
        sys.exit(f"IDs file not found: {ids_path}")

    try:
        ensure_out_dir(out_dir)
    except OSError as e:
        sys.exit(f"ERROR: {e}")

    ids = load_ids(ids_path)

    n_ok = 0; n_fail = 0
    for pid in ids:
        if args.format == "pdb":
            url = PDB_URL.format(id=pid)
            out = out_dir / f"{pid}.pdb"
        else:
            url = MMCIF_URL.format(id=pid)
            out = out_dir / f"{pid}.cif"

        if out.exists() and not args.overwrite:
            print(f"SKIP (exists): {out}")
            n_ok += 1
            continue
        try:
            data = fetch(url)
            out.write_bytes(data)
            print(f"OK  {pid} -> {out}")
            n_ok += 1
        except (HTTPError, URLError) as e:
            print(f"FAIL {pid}: {e}", file=sys.stderr)
            n_fail += 1
        except Exception as e:
            print(f"FAIL {pid}: {e}", file=sys.stderr)
            n_fail += 1

    print(f"Done. ok={n_ok} fail={n_fail}")

if __name__ == "__main__":
    main()