"""Collect linked PDFs from the National Archives 2025 JFK release listing."""
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
import re
import time
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.archives.gov/research/jfk/release-2025"


def collect(output: Path, max_pages: int = 3, delay: float = 1.0) -> int:
    if max_pages < 1 or delay < 0:
        raise ValueError("max_pages must be >= 1 and delay must be >= 0")
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "JFKReaderResearch/1.0 (public archives research)"})
    downloaded = 0
    seen = set()
    for page in range(max_pages):
        response = session.get(BASE_URL, params={"page": page}, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links = []
        for a in soup.select("a[href]"):
            link = urljoin(response.url, a["href"])
            if urlparse(link).path.lower().endswith(".pdf") and link not in seen:
                seen.add(link)
                links.append(link)
        print(f"Listing page {page}: {len(links)} new PDF links")
        if not links:
            break
        for link in links:
            basename = Path(unquote(urlparse(link).path)).name
            safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", basename)
            # Prefix with a URL hash to prevent unrelated PDFs with identical names overwriting.
            import hashlib
            filename = hashlib.sha256(link.encode()).hexdigest()[:10] + "_" + safe_name
            target = output / filename
            if target.exists():
                continue
            try:
                with session.get(link, stream=True, timeout=60) as result:
                    result.raise_for_status()
                    content_type = result.headers.get("Content-Type", "").lower()
                    if content_type and "pdf" not in content_type and "octet-stream" not in content_type:
                        print(f"Skipping non-PDF response: {link}")
                        continue
                    temp = target.with_suffix(".part")
                    try:
                        with temp.open("wb") as handle:
                            for block in result.iter_content(chunk_size=65536):
                                if block:
                                    handle.write(block)
                        with temp.open("rb") as handle:
                            if handle.read(4) != b"%PDF":
                                print(f"Skipping invalid PDF: {link}")
                                continue
                        temp.replace(target)
                        downloaded += 1
                    finally:
                        temp.unlink(missing_ok=True)
            except requests.RequestException as exc:
                print(f"Download failed ({link}): {exc}")
            time.sleep(delay)
        time.sleep(delay)
    return downloaded
