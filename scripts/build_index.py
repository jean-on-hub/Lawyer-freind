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
MIN_CHUNK = 300
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


# A section starts on a line holding just its number — "20." or, from OCR, "124," —
# or with the number running into "(1)" or a capitalised first word.
SECTION_START = re.compile(r"(?m)^[ \t]*(\d{1,3}[A-Z]?)[ \t]*[.,][ \t]*(?:$|(?=\(1\)|[A-Z\u201c\"]))")


def _section_starts(text: str) -> list[int]:
    """Offsets where sections begin, moved back to include the margin title.

    Statutes print a short title above the number ("Leave entitlement" / "20."),
    and that title is often the only place the words a user would search for
    appear, so it must travel with its section rather than end the previous one.
    """
    starts = []
    for m in SECTION_START.finditer(text):
        begin = m.start()
        for _ in range(2):
            line_start = text.rfind("\n", 0, max(begin - 1, 0)) + 1
            line = text[line_start:begin].strip()
            if (0 < len(line) <= 70 and not line.endswith((".", ";", ":", ","))
                    and not re.match(r"^\(?\d", line)):
                begin = line_start
            else:
                break
        if not starts or begin > starts[-1]:
            starts.append(begin)
    return starts


def section_chunks(text: str, target: int, splitter) -> list[str]:
    """Chunks that begin at section boundaries.

    Fixed 1,000-character windows cut across sections, so a short provision like
    "every worker is entitled to not less than fifteen working days leave" ended
    up sharing a chunk with the tail of the previous section and ranked too low
    to be found. Here small sections are packed together but never split, and a
    section is only cut when it is long on its own — each piece then repeats the
    section's heading so it still says what it is about.
    """
    starts = _section_starts(text)
    if len(starts) < 3:
        return splitter.split_text(text)

    pieces = [text[:starts[0]]] + [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]
    chunks, buf = [], ""
    for sec in pieces:
        sec = sec.strip()
        if not sec:
            continue
        if len(sec) > target * 1.4:
            if buf:
                chunks.append(buf); buf = ""
            heading = " ".join(sec.split("\n")[:3])[:100]
            parts = splitter.split_text(sec)
            chunks.append(parts[0])
            chunks += [f"{heading} (continued)\n{p}" for p in parts[1:]]
        elif buf and (len(buf) + len(sec) > target or len(buf) >= MIN_CHUNK):
            # Once the buffer holds a real section, the next one starts fresh;
            # only fragments such as table-of-contents lines are packed together.
            chunks.append(buf); buf = sec
        else:
            buf = f"{buf}\n{sec}" if buf else sec
    if buf:
        chunks.append(buf)
    return chunks


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
    ap.add_argument("--table", default=TABLE, help="table to write (build a test table without touching production)")
    ap.add_argument("--fixed-chunks", action="store_true", help="old fixed-size chunking instead of section-aware")
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
    if args.table in db.table_names():
        db.drop_table(args.table)
    table = db.create_table(args.table, schema=schema)

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
            if args.fixed_chunks:
                split = splitter.split_documents(docs)
                chunks = [(c.page_content, int(c.metadata.get("page", 0) or 0)) for c in split]
            else:
                # Join pages so sections that cross a page break stay whole,
                # remembering where each page starts to label chunks.
                text, page_at = "", []
                for d in docs:
                    page_at.append((len(text), int(d.metadata.get("page", 0) or 0)))
                    text += d.page_content + "\n"
                chunks, cursor = [], 0
                for c in section_chunks(text, args.chunk_size, splitter):
                    pos = text.find(c[:60], cursor)
                    if pos >= 0:
                        cursor = pos
                    page = max((pg for off, pg in page_at if off <= max(pos, 0)), default=0)
                    chunks.append((c, page))
            chunks = [(t, p) for t, p in chunks if len(t.strip()) > 50]
            if not chunks:
                continue
            vectors = embedder.embed_documents([t for t, _ in chunks])
            table.add([
                {"vector": v, "text": t, "source": name, "page": p}
                for v, (t, p) in zip(vectors, chunks)
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

    # Search indexes are built here because the server mounts the table read-only.
    # The vector index turns a 300 ms scan into ~10 ms; the full-text index
    # supplies the keyword half of hybrid retrieval.
    table.create_index(metric="l2", num_partitions=64, num_sub_vectors=16, replace=True)
    table.create_fts_index("text", use_tantivy=False, replace=True, with_position=False)

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
