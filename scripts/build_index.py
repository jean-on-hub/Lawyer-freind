"""Build the vector index from the harvested PDFs.

Two things differ from the old FAISS build:

  * It filters to legislation. Roughly 40% of the Parliament repository is
    committee reports and 4% is loan agreements; indexing those means a tenancy
    question can retrieve a mining investigation.
  * It writes to LanceDB, which searches from disk. FAISS loads the whole index
    into RAM, and the server has 916 MB — fine for 21 Acts, impossible for
    hundreds. Disk is the resource the free tier has spare.

Documents are embedded and written file by file, so peak memory stays flat no
matter how large the corpus grows.

    python scripts/build_index.py                 # build from every source dir
    python scripts/build_index.py --no-filter     # index everything, not just laws
    python scripts/build_index.py --limit 50      # first 50 files, for a quick test
"""

import argparse
import json
import os
import re
import sys
import time

import lancedb
import pyarrow as pa
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = os.path.join(os.path.dirname(__file__), "..")
SOURCE_DIRS = [
    os.path.join(ROOT, "Legal_documents"),
    os.path.join(ROOT, "Legal_documents", "repository"),
    os.path.join(ROOT, "Legal_documents", "crawled"),
    # Text recovered by OCR from scanned Acts, which extract nothing on their own
    os.path.join(ROOT, "Legal_documents", "ocr"),
    # Hand-written referral notes: where to actually go, which statutes never say.
    # Always indexed, never filtered. "pending_verification" is deliberately not
    # listed — a draft with unconfirmed phone numbers must not reach users.
    os.path.join(ROOT, "Legal_documents", "practical"),
]
TEXT_EXTS = (".md", ".txt")
DB_DIR = os.path.join(ROOT, "ghana_law_lancedb")
TABLE = "ghana_law"
EMBED_MODEL = "all-MiniLM-L6-v2"
DIM = 384

# Titles that are legislation. Kept deliberately broad — a missed Act costs more
# than an extra report, so anything act-like is admitted.
LAW_RE = re.compile(
    r"\b(act|acts|bill|bills|constitution|regulation|regulations|"
    r"l\.?\s?i\.?|instrument|decree|ordinance|code|rules|"
    r"statute|enactment|amendment)\b", re.I)

# Titles that are clearly not legislation, checked first so "Report of the
# Committee on the X Act" is excluded rather than admitted by the word "Act".
NOT_LAW_RE = re.compile(
    r"\b(report|committee|hansard|votes\s+and\s+proceedings|minutes|"
    r"agreement|loan|facility|credit|financing|budget\s+statement|"
    r"communiqu|memorandum|speech|programme|manual|newsletter)\b", re.I)


def is_legislation(name: str) -> bool:
    stem = os.path.splitext(os.path.basename(name))[0]
    # Underscores and hyphens are word characters to the regex engine, so
    # "Marriages_Act_1884" would never match \bact\b. Normalise separators first.
    stem = re.sub(r"[_\-]+", " ", stem)
    if NOT_LAW_RE.search(stem):
        return False
    return bool(LAW_RE.search(stem))


AMENDMENT_RE = re.compile(r"amendment", re.I)
AMENDMENTS_FILE = "amendments.json"


def _law_key(name: str) -> str:
    """Normalised identity of a law, so an Act and its amendments collapse together."""
    s = os.path.splitext(os.path.basename(name))[0]
    s = re.sub(r"^[0-9a-f]{6,}_", "", s)                       # harvest dedupe prefix
    s = AMENDMENT_RE.sub(" ", s)
    s = re.sub(r"\b(no\.?\s*\d+|act|bill|rules|instrument|revised|edition)\b", " ", s, flags=re.I)
    s = re.sub(r"\b(1[89]|20)\d{2}\b", " ", s)                 # years
    s = re.sub(r"[^a-z]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def build_amendment_map(names: list[str]) -> dict[str, list[str]]:
    """Map each base law to the amending laws held alongside it.

    The corpus holds Acts as enacted, with no record of what was later changed —
    the Road Traffic Act 2004 sits next to its 2025 amendment with nothing linking
    them. Quoting superseded text confidently is the worst failure this bot can
    have, so retrieval carries a warning when an amendment exists.
    """
    by_key: dict[str, list[str]] = {}
    for n in names:
        if AMENDMENT_RE.search(n):
            by_key.setdefault(_law_key(n), []).append(os.path.splitext(n)[0])

    out: dict[str, list[str]] = {}
    for n in names:
        if AMENDMENT_RE.search(n):
            continue
        hits = by_key.get(_law_key(n))
        if hits:
            out[n] = sorted(set(hits))
    return out


def _dedupe_key(name: str) -> str:
    """Same document harvested twice differs only by the hash prefix."""
    return re.sub(r"^[0-9a-f]{6,}_", "", name).lower().strip()


def find_sources(limit: int | None, apply_filter: bool) -> list[str]:
    seen, out = set(), []
    for d in SOURCE_DIRS:
        if not os.path.isdir(d):
            continue
        for entry in sorted(os.listdir(d)):
            path = os.path.join(d, entry)
            low = entry.lower()
            if not os.path.isfile(path) or _dedupe_key(entry) in seen:
                continue
            if not (low.endswith(".pdf") or low.endswith(TEXT_EXTS)):
                continue
            seen.add(_dedupe_key(entry))
            # Curated notes are always kept; the filter only judges harvested PDFs
            if apply_filter and low.endswith(".pdf") and not is_legislation(entry):
                continue
            out.append(path)
            if limit and len(out) >= limit:
                return out
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-filter", action="store_true", help="index every document, not just legislation")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=150)
    args = ap.parse_args()

    pdfs = find_sources(args.limit, not args.no_filter)
    if not pdfs:
        print("No PDFs found. Harvest first: python scripts/harvest_dspace.py")
        return 1
    print(f"{len(pdfs)} documents to index"
          f"{'' if args.no_filter else ' (legislation only)'}")

    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size,
                                              chunk_overlap=args.overlap)

    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), DIM)),
        pa.field("text", pa.string()),
        pa.field("source", pa.string()),
        pa.field("page", pa.int32()),
    ])
    db = lancedb.connect(DB_DIR)
    if TABLE in db.table_names():
        db.drop_table(TABLE)
    table = db.create_table(TABLE, schema=schema)

    total_chunks, started, failed = 0, time.time(), 0
    for i, path in enumerate(pdfs, 1):
        name = os.path.basename(path)
        try:
            if path.lower().endswith(TEXT_EXTS):
                from langchain_core.documents import Document as _Doc
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    docs = [_Doc(page_content=fh.read(), metadata={"page": 0})]
            else:
                docs = PyMuPDFLoader(path).load()
            chunks = splitter.split_documents(docs)
            chunks = [c for c in chunks if len(c.page_content.strip()) > 50]
            if not chunks:
                continue
            vectors = embedder.embed_documents([c.page_content for c in chunks])
            table.add([
                {"vector": v,
                 "text": c.page_content,
                 "source": name,
                 "page": int(c.metadata.get("page", 0) or 0)}
                for v, c in zip(vectors, chunks)
            ])
            total_chunks += len(chunks)
        except Exception as e:
            failed += 1
            print(f"  ! {name[:60]}: {str(e)[:80]}")
            continue

        if i % 10 == 0 or i == len(pdfs):
            rate = i / max(time.time() - started, 1)
            eta = (len(pdfs) - i) / max(rate, 0.01) / 60
            print(f"  {i}/{len(pdfs)} files · {total_chunks:,} chunks · "
                  f"{rate:.1f} files/s · ~{eta:.0f} min left")

    # Written next to the index so the server can warn when a law was later amended
    amendments = build_amendment_map([os.path.basename(p) for p in pdfs])
    with open(os.path.join(DB_DIR, AMENDMENTS_FILE), "w") as fh:
        json.dump(amendments, fh, indent=1)

    print(f"\nIndexed {total_chunks:,} chunks from {len(pdfs) - failed} files "
          f"({failed} failed)")
    print(f"{len(amendments)} laws flagged as having amendments in the corpus")
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(DB_DIR) for f in fs)
    print(f"Index on disk: {size / 1e6:.0f} MB at {os.path.normpath(DB_DIR)}")
    print("\nThe server reads this automatically when the directory is present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
