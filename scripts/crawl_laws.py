"""Crawl public Ghanaian legal sources for PDFs, obeying each site's own rules.

Ghanaian statutes and judgments are government edicts and free to reuse, but that
says nothing about whether a given server wants automated traffic. This crawler
therefore checks two things before touching a host:

  1. robots.txt paths and Crawl-delay, via the standard parser.
  2. The Content-Signal header some sites publish. GhaLII and AfricanLII declare
     "ai-input=no", and their own preamble defines ai-input as feeding content to
     a model for retrieval augmented generation — precisely what this project
     does. Those hosts are refused, and the reason is printed.

Usage:
    python scripts/crawl_laws.py                  # crawl the default sources
    python scripts/crawl_laws.py --max-files 50   # stop after 50 PDFs
    python scripts/crawl_laws.py --url URL        # crawl one source
"""

import argparse
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.robotparser

import requests

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "Legal_documents", "crawled")
STATE_FILE = os.path.join(OUT_DIR, ".crawl_state.json")

# Identify honestly, so an operator who dislikes the traffic can identify and block it.
USER_AGENT = "LawyerFriendBot/1.0 (free legal information for Ghanaians; +https://github.com/jean-on-hub/Lawyer-freind)"

SEEDS = [
    "https://www.parliament.gh/acts",
    "https://www.parliament.gh/bills",
    "https://judicial.gov.gh/index.php/media-center/publications",
    "https://laws.africa/",
]

DEFAULT_DELAY = 2.0          # seconds between requests when a site states none
MAX_PAGES_PER_HOST = 5000    # override with --max-pages
LINK_RE = re.compile(r'href=["\']([^"\']+)["\']', re.I)
PDF_RE = re.compile(r"\.pdf($|\?)", re.I)


class HostRules:
    """robots.txt plus Content-Signal for one host."""

    def __init__(self, base: str):
        self.base = base
        self.delay = DEFAULT_DELAY
        self.refuses_ai_input = False
        # Collected by hand as well as by the stdlib parser: a site may publish
        # several "User-agent: *" groups, and RobotFileParser honours only the
        # first, which silently drops the Disallow rules in the later ones.
        self.disallowed: list[str] = []
        self.parser = urllib.robotparser.RobotFileParser()
        self._load()

    def _load(self):
        robots_url = urllib.parse.urljoin(self.base, "/robots.txt")
        try:
            r = requests.get(robots_url, headers={"User-Agent": USER_AGENT}, timeout=20)
            text = r.text if r.status_code == 200 else ""
        except Exception:
            text = ""

        self.parser.parse(text.splitlines())

        # Content-Signal is not part of robots.txt, so parse it ourselves. Only the
        # blocks addressed to "*" matter to us.
        applies = False
        for line in text.splitlines():
            stripped = line.strip()
            low = stripped.lower()
            if low.startswith("user-agent:"):
                applies = stripped.split(":", 1)[1].strip() == "*"
            elif applies and low.startswith("content-signal:"):
                if re.search(r"ai-input\s*=\s*no", low):
                    self.refuses_ai_input = True
            elif applies and low.startswith("crawl-delay:"):
                try:
                    self.delay = max(self.delay, float(stripped.split(":", 1)[1].strip()))
                except ValueError:
                    pass
            elif applies and low.startswith("disallow:"):
                path = stripped.split(":", 1)[1].strip()
                if path:
                    self.disallowed.append(path)

    def allows(self, url: str) -> bool:
        path = urllib.parse.urlparse(url).path or "/"
        if any(path.startswith(rule) for rule in self.disallowed):
            return False
        return self.parser.can_fetch(USER_AGENT, url)


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"visited": [], "hashes": []}


def save_state(state: dict) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({"visited": sorted(state["visited"]), "hashes": sorted(state["hashes"])}, f)


def safe_name(url: str) -> str:
    name = os.path.basename(urllib.parse.urlparse(url).path) or "document.pdf"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", urllib.parse.unquote(name))
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name[:120]


def download_pdf(url: str, state: dict) -> str | None:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"    ! download failed: {e}")
        return None

    if not r.content.startswith(b"%PDF"):
        return None  # HTML error page served with a .pdf URL

    digest = hashlib.sha256(r.content).hexdigest()
    if digest in state["hashes"]:
        return None  # same document under a different URL
    state["hashes"].append(digest)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, safe_name(url))
    if os.path.exists(path):
        path = os.path.join(OUT_DIR, f"{digest[:8]}_{safe_name(url)}")
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def crawl(seed: str, state: dict, max_files: int, downloaded: list, max_pages: int = MAX_PAGES_PER_HOST) -> None:
    parts = urllib.parse.urlparse(seed)
    base = f"{parts.scheme}://{parts.netloc}"
    rules = HostRules(base)

    print(f"\n=== {base} ===")
    if rules.refuses_ai_input:
        print("  REFUSED: this site's Content-Signal sets ai-input=no, which its own")
        print("  preamble defines as retrieval augmented generation. Skipping.")
        return
    print(f"  crawl-delay: {rules.delay}s")

    queue, seen_here = [seed], set()
    while queue and len(downloaded) < max_files and len(seen_here) < max_pages:
        url = queue.pop(0)
        if url in seen_here or url in state["visited"]:
            continue
        seen_here.add(url)

        if not rules.allows(url):
            print(f"  robots.txt disallows: {url}")
            continue

        time.sleep(rules.delay)
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=40)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}: {url}")
                continue
        except Exception as e:
            print(f"  ! {e}")
            continue

        state["visited"].append(url)
        ctype = r.headers.get("Content-Type", "")

        for raw in LINK_RE.findall(r.text if "html" in ctype else ""):
            link = urllib.parse.urljoin(url, raw.split("#")[0])
            if urllib.parse.urlparse(link).netloc != parts.netloc:
                continue
            if PDF_RE.search(link):
                if link in state["visited"] or not rules.allows(link):
                    continue
                state["visited"].append(link)
                time.sleep(rules.delay)
                saved = download_pdf(link, state)
                if saved:
                    downloaded.append(saved)
                    print(f"  [{len(downloaded)}] {os.path.basename(saved)}")
                    if len(downloaded) >= max_files:
                        return
            elif link not in seen_here:
                queue.append(link)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", action="append", help="crawl this URL instead of the defaults")
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--max-pages", type=int, default=MAX_PAGES_PER_HOST,
                    help="pages to walk per host before moving on")
    args = ap.parse_args()

    state = load_state()
    downloaded: list[str] = []
    try:
        for seed in (args.url or SEEDS):
            if len(downloaded) >= args.max_files:
                break
            crawl(seed, state, args.max_files, downloaded, args.max_pages)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        save_state(state)

    print(f"\nDownloaded {len(downloaded)} new PDFs into {os.path.normpath(OUT_DIR)}")
    if downloaded:
        print("Rebuild the index with: python scripts/new_file_loader.py")


if __name__ == "__main__":
    main()
