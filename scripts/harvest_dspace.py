"""Harvest PDFs from the Parliament of Ghana Library Repository (DSpace).

The repository publishes OAI-PMH, an open protocol designed for exactly this —
so instead of scraping HTML we enumerate records properly and fetch each item's
PDF through the REST API. Its robots.txt permits /server/; only /search, /admin,
/submit and similar are off limits.

    python scripts/harvest_dspace.py --list          # count records, no download
    python scripts/harvest_dspace.py                 # harvest everything
    python scripts/harvest_dspace.py --max-files 100
    python scripts/harvest_dspace.py --match "act"   # only titles containing "act"
"""

import argparse
import hashlib
import json
import os
import re
import time
import urllib.parse

import requests

BASE = "https://repository.parliament.gh"
OAI = f"{BASE}/server/oai/request"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "Legal_documents", "repository")
STATE_FILE = os.path.join(OUT_DIR, ".harvest_state.json")

USER_AGENT = "LawyerFriendBot/1.0 (free legal information for Ghanaians; +https://github.com/jean-on-hub/Lawyer-freind)"
DELAY = 1.0  # be gentle; the repository is a small library server

RECORD_RE = re.compile(r"<record>(.*?)</record>", re.S)
TITLE_RE = re.compile(r"<dc:title>(.*?)</dc:title>", re.S)
HANDLE_RE = re.compile(r"(?:hdl\.handle\.net|/handle)/(\d+/\d+)")
TOKEN_RE = re.compile(r"<resumptionToken[^>]*>([^<]+)</resumptionToken>")
SIZE_RE = re.compile(r'completeListSize="(\d+)"')


def get(url: str, **kw):
    return requests.get(url, headers={"User-Agent": USER_AGENT, **kw.pop("headers", {})},
                        timeout=kw.pop("timeout", 60), **kw)


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"handles": [], "hashes": []}


def save_state(state: dict) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({"handles": sorted(set(state["handles"])),
                   "hashes": sorted(set(state["hashes"]))}, f)


def iter_records():
    """Yield (handle, title) for every record, following resumption tokens."""
    params = {"verb": "ListRecords", "metadataPrefix": "oai_dc"}
    total = None
    while True:
        r = get(f"{OAI}?{urllib.parse.urlencode(params)}")
        if r.status_code != 200:
            print(f"  ! OAI HTTP {r.status_code}")
            return
        xml = r.text
        if total is None:
            m = SIZE_RE.search(xml)
            total = m.group(1) if m else "?"
            print(f"  repository reports {total} records")

        for chunk in RECORD_RE.findall(xml):
            handle = HANDLE_RE.search(chunk)
            title = TITLE_RE.search(chunk)
            if handle:
                yield handle.group(1), re.sub(r"\s+", " ", title.group(1)).strip() if title else ""

        token = TOKEN_RE.search(xml)
        if not token or not token.group(1).strip():
            return
        params = {"verb": "ListRecords", "resumptionToken": token.group(1).strip()}
        time.sleep(DELAY)


def pdf_urls_for(handle: str) -> list[str]:
    """Find downloadable PDF bitstreams for one item via the REST API."""
    try:
        r = get(f"{BASE}/server/api/pid/find", params={"id": f"hdl:{handle}"},
                headers={"Accept": "application/json"}, allow_redirects=True)
        if r.status_code != 200:
            return []
        item = r.json()
        uuid = item.get("uuid")
        if not uuid:
            return []
        b = get(f"{BASE}/server/api/core/items/{uuid}/bundles",
                headers={"Accept": "application/json"})
        urls = []
        for bundle in b.json().get("_embedded", {}).get("bundles", []):
            link = bundle.get("_links", {}).get("bitstreams", {}).get("href")
            if not link:
                continue
            bs = get(link, headers={"Accept": "application/json"})
            for item_bs in bs.json().get("_embedded", {}).get("bitstreams", []):
                name = (item_bs.get("name") or "").lower()
                content = item_bs.get("_links", {}).get("content", {}).get("href")
                if content and (name.endswith(".pdf") or "pdf" in
                                str(item_bs.get("metadata", {}).get("dc.format.mimetype", ""))):
                    urls.append(content)
        return urls
    except Exception as e:
        print(f"    ! {str(e)[:90]}")
        return []


def safe_name(title: str, handle: str) -> str:
    name = re.sub(r"[^A-Za-z0-9 ._-]", "", title).strip() or f"item_{handle.replace('/', '_')}"
    return (name[:110] + ".pdf")


def download(url: str, title: str, handle: str, state: dict) -> str | None:
    try:
        r = get(url, timeout=120)
        r.raise_for_status()
    except Exception as e:
        print(f"    ! download failed: {str(e)[:90]}")
        return None
    if not r.content.startswith(b"%PDF"):
        return None

    digest = hashlib.sha256(r.content).hexdigest()
    if digest in state["hashes"]:
        return None
    state["hashes"].append(digest)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, safe_name(title, handle))
    if os.path.exists(path):
        path = os.path.join(OUT_DIR, f"{digest[:8]}_{safe_name(title, handle)}")
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-files", type=int, default=5000)
    ap.add_argument("--match", help="only harvest records whose title contains this")
    ap.add_argument("--list", action="store_true", help="list records without downloading")
    args = ap.parse_args()

    state = load_state()
    saved, seen = 0, 0
    print(f"Harvesting {BASE}")
    try:
        for handle, title in iter_records():
            seen += 1
            if args.match and args.match.lower() not in title.lower():
                continue
            if args.list:
                print(f"  {handle}  {title[:90]}")
                continue
            if handle in state["handles"]:
                continue
            state["handles"].append(handle)

            for url in pdf_urls_for(handle):
                time.sleep(DELAY)
                path = download(url, title, handle, state)
                if path:
                    saved += 1
                    print(f"  [{saved}] {os.path.basename(path)}")
                    if saved >= args.max_files:
                        return
            if seen % 50 == 0:
                save_state(state)
                print(f"  ... {seen} records scanned, {saved} PDFs saved")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        save_state(state)
        print(f"\nScanned {seen} records, saved {saved} PDFs into {os.path.normpath(OUT_DIR)}")
        if saved:
            print("Rebuild the index with: python scripts/new_file_loader.py")


if __name__ == "__main__":
    main()
