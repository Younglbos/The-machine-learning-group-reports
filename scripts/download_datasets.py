"""Download course datasets and verify them against their GitHub blob hashes."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "dataset_sources.json"
RAW = ROOT / "data" / "raw"


def verify(path, item):
    content = path.read_bytes()
    blob_hash = hashlib.sha1(
        b"blob " + str(len(content)).encode("ascii") + b"\0" + content
    ).hexdigest()
    if len(content) != item["expected_bytes"] or blob_hash != item["git_blob_sha1"]:
        raise ValueError(f"Size or Git blob hash mismatch: {path.name}")
    return {
        **item,
        "local_path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "verified": True,
    }


def download(item):
    target = RAW / item["name"]
    if target.exists():
        result = verify(target, item)
        print(f"Verified existing: {target.name}", flush=True)
        return result
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(3):
        try:
            request = Request(item["download_url"], headers={"User-Agent": "course-dataset-downloader/1.0"})
            with urlopen(request, timeout=30) as response, partial.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            result = verify(partial, item)
            partial.replace(target)
            result["local_path"] = target.relative_to(ROOT).as_posix()
            print(f"Downloaded and verified: {target.name} ({target.stat().st_size:,} bytes)", flush=True)
            return result
        except Exception as exc:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}: {target.name}: {exc}", flush=True)
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    items = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {
            pool.submit(verify, RAW / item["name"], item) if args.verify_only
            else pool.submit(download, item): item for item in items
        }
        for job in as_completed(jobs):
            try:
                results.append(job.result())
            except Exception as exc:
                failures.append({"name": jobs[job]["name"], "error": str(exc)})
                print(f"FAILED: {jobs[job]['name']}: {exc}", flush=True)
    report = {"datasets": sorted(results, key=lambda x: x["name"]), "failures": failures}
    (ROOT / "data" / "download_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Verified {len(results)}/{len(items)} datasets.", flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
