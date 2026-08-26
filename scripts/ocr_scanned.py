"""OCR the scanned Acts so they stop being invisible to the bot.

About a third of the corpus is scanned images with no text layer. Those files look
present — they sit in the folder, they pass the legislation filter — but they
contribute nothing to the index, so the bot silently has no source for whole areas
of law. The Criminal Offences Act, the Road Traffic Act and the Conveyancing Act
were all in this state.

Each unreadable PDF is rendered page by page and run through Tesseract, and the
recovered text is written beside it as a .txt file that the indexer picks up. The
original PDF is never modified.

    python scripts/ocr_scanned.py --list        # show what would be processed
    python scripts/ocr_scanned.py               # OCR everything unreadable
    python scripts/ocr_scanned.py --limit 20    # just the biggest few
"""

import argparse
import os
import sys
import time

import fitz  # pymupdf

sys.path.insert(0, os.path.dirname(__file__))
from build_index import find_sources  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "Legal_documents", "ocr")
MIN_CHARS = 2000     # below this a document is treated as unreadable
DPI = 200            # enough for statute text without making pages huge


def text_length(path: str) -> tuple[int, int]:
    try:
        with fitz.open(path) as doc:
            return len(doc), sum(len(p.get_text().strip()) for p in doc)
    except Exception:
        return 0, 0


def ocr_pdf(path: str) -> str:
    """Return OCR text for a whole PDF."""
    out = []
    with fitz.open(path) as doc:
        for page in doc:
            try:
                tp = page.get_textpage_ocr(dpi=DPI, full=True)
                out.append(page.get_text(textpage=tp))
            except Exception as e:
                print(f"      ! page {page.number}: {str(e)[:60]}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS)
    args = ap.parse_args()

    todo = []
    for path in find_sources(None, True):
        if not path.lower().endswith(".pdf"):
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        if os.path.exists(os.path.join(OUT_DIR, stem + ".txt")):
            continue  # already recovered
        pages, chars = text_length(path)
        if pages and chars < args.min_chars:
            todo.append((path, pages, chars))

    todo.sort(key=lambda t: -t[1])
    if args.limit:
        todo = todo[:args.limit]

    total_pages = sum(t[1] for t in todo)
    print(f"{len(todo)} unreadable documents, {total_pages:,} pages")
    if args.list:
        for p, pages, chars in todo:
            print(f"  {pages:4}p {chars:6,} chars  {os.path.basename(p)[:60]}")
        return 0
    if not todo:
        print("Nothing to do.")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    started, done_pages, recovered = time.time(), 0, 0
    for i, (path, pages, _) in enumerate(todo, 1):
        name = os.path.basename(path)
        try:
            text = ocr_pdf(path)
        except Exception as e:
            print(f"  ! {name[:50]}: {str(e)[:70]}")
            continue
        done_pages += pages
        if len(text.strip()) < 200:
            print(f"  ~ {name[:50]}: OCR produced almost nothing, skipping")
            continue
        with open(os.path.join(OUT_DIR, os.path.splitext(name)[0] + ".txt"), "w") as fh:
            fh.write(text)
        recovered += 1
        rate = done_pages / max(time.time() - started, 1)
        eta = (total_pages - done_pages) / max(rate, 0.01) / 60
        print(f"  [{i}/{len(todo)}] {len(text):,} chars from {pages}p · "
              f"{rate:.1f} pages/s · ~{eta:.0f} min left · {name[:42]}")

    print(f"\nRecovered {recovered} documents into {os.path.normpath(OUT_DIR)}")
    print("Rebuild the index to include them: python scripts/build_index.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
