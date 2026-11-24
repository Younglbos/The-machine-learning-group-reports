"""Import one pinned author's project per process; never mix its generic module names."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def author_path(name):
    path = ROOT / 'third_party' / name
    if not path.is_dir():
        raise FileNotFoundError(path)
    sys.path.insert(0, str(path))
    return path
