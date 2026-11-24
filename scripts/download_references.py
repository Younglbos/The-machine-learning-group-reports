"""Download public papers and pinned author-code snapshots without executing code."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "references"


def fetch(item):
    folder = REFS / ("papers" if item["kind"] == "paper" else "archives")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / item["name"]
    if path.exists():
        content = path.read_bytes()
    else:
        request = Request(item["url"], headers={"User-Agent": "course-reproduction-preparation/1.0"})
        with urlopen(request, timeout=45) as response:
            content = response.read()
    if item["kind"] == "paper":
        if not content.startswith(b"%PDF-"):
            raise ValueError(f"Not a PDF: {item['name']}")
    else:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"ZIP integrity failure: {item['name']}")
            dest = ROOT / "third_party" / item["extract_to"]
            dest.mkdir(parents=True, exist_ok=True)
            for member in archive.infolist():
                relative = Path(*Path(member.filename).parts[1:])
                if not relative.parts or member.is_dir():
                    continue
                target = (dest / relative).resolve()
                if not target.is_relative_to(dest.resolve()):
                    raise ValueError("Unsafe ZIP member")
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    continue
                with archive.open(member) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".part")
        partial.write_bytes(content)
        partial.replace(path)
    result = {**item, "local_path": path.relative_to(ROOT).as_posix(),
              "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    print(f"Ready: {item['name']} ({len(content):,} bytes)", flush=True)
    return result


def main():
    items = json.loads((REFS / "download_sources.json").read_text(encoding="utf-8"))
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(fetch, item): item for item in items}
        for future in as_completed(jobs):
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"name": jobs[future]["name"], "error": str(exc)})
                print(f"FAILED: {jobs[future]['name']}: {exc}", flush=True)
    (REFS / "download_report.json").write_text(
        json.dumps({"assets": results, "failures": failures}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
