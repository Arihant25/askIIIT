#!/usr/bin/env python3
"""
download_pdfs.py

Simple script to download all PDF links from a given page.

Features:
- Finds <a> elements with hrefs
- Normalizes relative links using base URL
- Filters by extension (default .pdf)
- Concurrent downloads using ThreadPoolExecutor
- Progress bar and basic retry
- Saves files into a folder

Dependencies: requests, beautifulsoup4, tqdm
"""

import argparse
import os
import sys
import time
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry
from tqdm import tqdm


def make_session(retries=3, backoff_factor=0.5, status_forcelist=(500, 502, 503, 504)):
    s = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=("HEAD", "GET", "OPTIONS"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({
        "User-Agent": "askIIIT-downloader/1.0 (+https://github.com)"
    })
    return s


def find_links(url, session, extensions):
    r = session.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    anchors = soup.find_all("a", href=True)
    urls = []
    for a in anchors:
        href = a["href"].strip()
        # Normalize relative links
        abs_url = urljoin(url, href)

        if any(abs_url.lower().split("?")[0].endswith(ext) for ext in extensions):
            urls.append(abs_url)

    # remove duplicates while keeping order
    seen = set()
    unique_urls = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)
    return unique_urls


def safe_filename(url):
    # Try to get filename from URL path
    parsed = urlparse(url)
    name = os.path.basename(parsed.path)
    name = unquote(name)
    if not name:
        name = parsed.netloc.replace('.', '_')
    return name


def download_file(url, session, out_dir, timeout=30):
    filename = safe_filename(url)
    target = os.path.join(out_dir, filename)

    # Avoid name collisions
    base, ext = os.path.splitext(target)
    i = 1
    while os.path.exists(target) and os.path.getsize(target) > 0:
        target = f"{base}({i}){ext}"
        i += 1

    try:
        with session.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            # If server returns Content-Disposition, prefer it
            cd = r.headers.get("content-disposition")
            if cd and 'filename=' in cd:
                import re
                # Match both filename and filename* (RFC 5987 style)
                # Example: filename*=UTF-8''myfile.pdf or filename="myfile.pdf"
                m = re.search(r"filename\*?=(?:UTF-8''?)?(?P<name>.+)", cd, flags=re.IGNORECASE)
                if m:
                    # Strip surrounding quotes/spaces and trailing semicolons
                    newname = m.group('name').strip(' \"\'\;')
                    target = os.path.join(out_dir, unquote(newname))

            with open(target, "wb") as fd:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        fd.write(chunk)
        return target, None
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Download all PDFs (or other extensions) found on a page"
    )
    parser.add_argument("url", help="Page URL to scan for files")
    parser.add_argument("-o", "--out", default="pdfs", help="Output folder (default: pdfs)")
    parser.add_argument("-c", "--concurrency", type=int, default=6, help="Concurrent downloads")
    parser.add_argument("-e", "--ext", default=".pdf", help="Comma separated extensions to download (default: .pdf)")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    args = parser.parse_args()

    extensions = [e.strip().lower() for e in args.ext.split(",") if e.strip()]
    # Ensure dotted ext
    extensions = [e if e.startswith(".") else "." + e for e in extensions]

    os.makedirs(args.out, exist_ok=True)

    session = make_session()

    print(f"Scanning {args.url} for {extensions} ...")

    try:
        urls = find_links(args.url, session, extensions)
    except RequestException as e:
        print("Failed to fetch page:", e)
        sys.exit(1)

    if not urls:
        print("No files found.")
        return

    print(f"Found {len(urls)} file(s). Downloading to {args.out} ...")

    results = []
    pbar = None
    if not args.no_progress:
        pbar = tqdm(total=len(urls), unit="file")

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(download_file, u, session, args.out): u for u in urls}
        for fut in as_completed(futures):
            u = futures[fut]
            try:
                path, err = fut.result()
                if err:
                    results.append((u, False, err))
                else:
                    results.append((u, True, path))
            except Exception as e:
                results.append((u, False, str(e)))
            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()

    # Summarize
    succeeded = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print(f"Done. {len(succeeded)} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed downloads:")
        for url, ok, err in failed:
            print(" -", url, "=>", err)


if __name__ == "__main__":
    main()
