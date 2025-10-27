import os
import sys
import math
import time
import argparse
import requests


DEFAULT_DATASET_PATH = "hf-audio/esb-datasets-test-only-sorted"
DEFAULT_SPLITS = {
    "ami": ["test"],
    "earnings22": ["test"],
    "gigaspeech": ["test"],
    "librispeech": ["test.clean", "test.other"],
    "spgispeech": ["test"],
    "tedlium": ["test"],
    "voxpopuli": ["test"],
}

ROWS_API = "https://datasets-server.huggingface.co/rows"
SIZE_API = "https://datasets-server.huggingface.co/size"


def http_headers() -> dict:
    token = os.getenv("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _parse_retry_after(value: str) -> float:
    try:
        # Numeric seconds
        return float(value)
    except Exception:
        pass
    # HTTP-date fallback
    try:
        from email.utils import parsedate_to_datetime
        import datetime
        dt = parsedate_to_datetime(value)
        return max(1.0, (dt - datetime.datetime.now(datetime.timezone.utc)).total_seconds())
    except Exception:
        return 0.0


def _get_json_with_retries(url: str, params: dict, timeout: int, max_retries: int, initial_backoff: float) -> dict:
    attempt = 0
    backoff = max(0.1, float(initial_backoff))
    headers = http_headers()
    while True:
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
                wait = _parse_retry_after(retry_after) if retry_after else min(60.0, backoff)
                attempt += 1
                if attempt > max_retries:
                    resp.raise_for_status()
                print(f"429 rate-limited; retrying in {wait:.1f}s (attempt {attempt}/{max_retries}) for {url}")
                time.sleep(wait)
                backoff = min(60.0, backoff * 2)
                continue
            if 500 <= resp.status_code < 600:
                attempt += 1
                if attempt > max_retries:
                    resp.raise_for_status()
                print(f"{resp.status_code} server error; retrying in {backoff:.1f}s (attempt {attempt}/{max_retries}) for {url}")
                time.sleep(backoff)
                backoff = min(60.0, backoff * 2)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            attempt += 1
            if attempt > max_retries:
                raise
            print(f"HTTP error {e}; retrying in {backoff:.1f}s (attempt {attempt}/{max_retries}) for {url}")
            time.sleep(backoff)
            backoff = min(60.0, backoff * 2)


def get_num_rows(dataset_path: str, dataset: str, split: str, timeout: int = 30, max_retries: int = 6, initial_backoff: float = 1.0) -> int:
    params = {"dataset": dataset_path, "config": dataset, "split": split}
    js = _get_json_with_retries(SIZE_API, params=params, timeout=timeout, max_retries=max_retries, initial_backoff=initial_backoff)
    if "size" not in js or "config" not in js["size"] or "num_rows" not in js["size"]["config"]:
        raise RuntimeError(f"Malformed size response for {dataset}/{split}: {js}")
    return int(js["size"]["config"]["num_rows"])


def sum_duration_seconds(
    dataset_path: str,
    dataset: str,
    split: str,
    total_rows: int,
    batch: int = 100,
    timeout: int = 60,
    progress_every: int = 20,
    rows_timeout: int = 60,
    max_retries: int = 6,
    initial_backoff: float = 1.0,
    sleep_between_pages: float = 0.0,
) -> float:
    total = 0.0
    if total_rows <= 0:
        return 0.0
    pages = math.ceil(total_rows / float(batch))
    print(f"[{dataset}:{split}] fetching durations: rows={total_rows} pages={pages} batch={batch}")
    start = time.time()
    page_idx = 0
    for offset in range(0, total_rows, batch):
        params = {
            "dataset": dataset_path,
            "config": dataset,
            "split": split,
            "offset": offset,
            "length": min(batch, total_rows - offset),
        }
        data = _get_json_with_retries(ROWS_API, params=params, timeout=rows_timeout, max_retries=max_retries, initial_backoff=initial_backoff)
        rows = data.get("rows", [])
        for row in rows:
            total += float(row.get("audio_length_s", 0.0))
        page_idx += 1
        if sleep_between_pages > 0:
            time.sleep(sleep_between_pages)
        if page_idx == 1 or page_idx % max(1, progress_every) == 0 or page_idx == pages:
            elapsed = time.time() - start
            rate = page_idx / elapsed if elapsed > 0 else 0.0
            remaining_pages = pages - page_idx
            eta_s = remaining_pages / rate if rate > 0 else 0.0
            print(
                f"[{dataset}:{split}] pages {page_idx}/{pages} "
                f"elapsed={elapsed:.1f}s "
                f"eta={eta_s:.1f}s"
            )
    elapsed = time.time() - start
    print(f"[{dataset}:{split}] done in {elapsed:.1f}s; seconds≈{total:.1f} hours≈{total/3600.0:.2f}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Report dataset sample counts and audio hours from HF Datasets Server")
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH, help="Datasets path (default: hf-audio/esb-datasets-test-only-sorted)")
    parser.add_argument("--datasets", type=str, nargs="*", default=list(DEFAULT_SPLITS.keys()), help="Subset of dataset configs to query")
    parser.add_argument("--include_splits", type=str, nargs="*", default=None, help="Optional overrides for splits, e.g. librispeech:test.clean librispeech:test.other")
    parser.add_argument("--batch", type=int, default=100, help="Batch size for rows pagination")
    parser.add_argument("--no_duration", action="store_true", help="Skip duration summation and only print sample counts")
    parser.add_argument("--size_timeout", type=int, default=30, help="Timeout (s) for size calls")
    parser.add_argument("--rows_timeout", type=int, default=60, help="Timeout (s) for rows calls")
    parser.add_argument("--progress_every", type=int, default=20, help="Print progress every N pages")
    parser.add_argument("--max_retries", type=int, default=6, help="Max retries for HTTP errors including 429/5xx")
    parser.add_argument("--initial_backoff", type=float, default=1.0, help="Initial backoff seconds for retries")
    parser.add_argument("--sleep_between_pages", type=float, default=0.0, help="Optional sleep seconds between page fetches")
    parser.add_argument("--estimate_duration", action="store_true", help="Estimate total hours using only the first N rows (see --sample_rows)")
    parser.add_argument("--sample_rows", type=int, default=2000, help="Number of rows to sample when --estimate_duration is enabled")
    args = parser.parse_args()

    # Build splits mapping
    splits_map = {k: v[:] for k, v in DEFAULT_SPLITS.items()}
    if args.include_splits:
        for spec in args.include_splits:
            if ":" not in spec:
                print(f"Ignoring malformed split spec: {spec}")
                continue
            ds, sp = spec.split(":", 1)
            splits_map.setdefault(ds, [])
            if sp not in splits_map[ds]:
                splits_map[ds].append(sp)

    header = ["dataset", "split", "samples", "hours"] if not args.no_duration else ["dataset", "split", "samples"]
    print("\t".join(header))

    for ds in args.datasets:
        if ds not in splits_map:
            print(f"Skipping unknown dataset config: {ds}")
            continue
        for sp in splits_map[ds]:
            try:
                print(f"[{ds}:{sp}] querying size...")
                n = get_num_rows(
                    args.dataset_path,
                    ds,
                    sp,
                    timeout=args.size_timeout,
                    max_retries=args.max_retries,
                    initial_backoff=args.initial_backoff,
                )
                if args.no_duration:
                    print(f"{ds}\t{sp}\t{n}")
                    continue
                if args.estimate_duration:
                    sample_n = max(1, min(args.sample_rows, n))
                    print(f"[{ds}:{sp}] estimating hours from first {sample_n} rows (of {n})...")
                    seconds_sample = sum_duration_seconds(
                        args.dataset_path,
                        ds,
                        sp,
                        sample_n,
                        batch=args.batch,
                        rows_timeout=args.rows_timeout,
                        max_retries=args.max_retries,
                        initial_backoff=args.initial_backoff,
                        sleep_between_pages=args.sleep_between_pages,
                        progress_every=args.progress_every,
                    )
                    avg_sec = seconds_sample / float(sample_n)
                    total_seconds_est = avg_sec * float(n)
                    hours_est = total_seconds_est / 3600.0
                    print(f"{ds}\t{sp}\t{n}\t≈{hours_est:.2f} (estimated)")
                else:
                    seconds = sum_duration_seconds(
                        args.dataset_path,
                        ds,
                        sp,
                        n,
                        batch=args.batch,
                        rows_timeout=args.rows_timeout,
                        max_retries=args.max_retries,
                        initial_backoff=args.initial_backoff,
                        sleep_between_pages=args.sleep_between_pages,
                        progress_every=args.progress_every,
                    )
                    hours = seconds / 3600.0
                    print(f"{ds}\t{sp}\t{n}\t{hours:.2f}")
            except Exception as e:
                print(f"{ds}\t{sp}\terror: {e}")


if __name__ == "__main__":
    main()


