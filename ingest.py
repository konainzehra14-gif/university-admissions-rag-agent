"""
ingest.py
---------
Builds the search index from documents in knowledge_base/.

Supported input files:
    .pdf
    .txt
    .md

Run:
    python ingest.py

Generated files:
    index/chunks.pkl
    index/metadata.pkl
    index/tokens.pkl
    index/keywords.pkl

Method:
    - Section-based chunking
    - FAQ Q/A splitting
    - Paragraph-based splitting for long sections
    - BM25 token index
    - Domain/acronym keyword extraction

No PyTorch or neural embedding dependency is required.
"""

import os
import re
import glob
import pickle

import pymupdf


KB_DIR = "knowledge_base"
INDEX_DIR = "index"

MAX_CHUNK_CHARS = 900
MIN_CHUNK_CHARS = 40


# ============================================================
# HEADING / FAQ REGEX
# ============================================================

NUMBERED_HEADER_RE = re.compile(
    r"^\s*\d+(?:\.\d+)*[\.\)]?\s+[A-Z][A-Z0-9 \-/&(),:]{3,}\s*$"
)

KNOWN_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"eligibility criteria|"
    r"eligibility requirements|"
    r"admission requirements|"
    r"admission process|"
    r"application process|"
    r"entry test|"
    r"entry tests|"
    r"test requirements|"
    r"documents required|"
    r"required documents|"
    r"important dates|"
    r"admission dates|"
    r"deadlines?|"
    r"fee structure|"
    r"fees?|"
    r"scholarships?|"
    r"financial aid|"
    r"programs offered|"
    r"programs|"
    r"frequently asked questions|"
    r"faq|"
    r"faqs"
    r")\s*:?\s*$",
    re.IGNORECASE
)

FAQ_Q_RE = re.compile(
    r"^\s*Q\s*:\s*",
    re.IGNORECASE | re.MULTILINE
)


# ============================================================
# FILE READERS
# ============================================================

def read_pdf(path):
    pages = []

    with pymupdf.open(path) as document:
        for page in document:
            text = page.get_text()
            if text:
                pages.append(text)

    return "\n".join(pages)


def read_text_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


def normalize_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():
    paths = sorted(
        glob.glob(os.path.join(KB_DIR, "*.pdf"))
        + glob.glob(os.path.join(KB_DIR, "*.txt"))
        + glob.glob(os.path.join(KB_DIR, "*.md"))
    )

    if not paths:
        raise SystemExit(
            f"\nNo .pdf/.txt/.md files found in '{KB_DIR}'.\n"
            f"Add your admissions documents there and run:\n\n"
            f"    python ingest.py\n"
        )

    documents = []

    for path in paths:
        extension = os.path.splitext(path)[1].lower()

        if extension == ".pdf":
            text = read_pdf(path)
        else:
            text = read_text_file(path)

        text = normalize_text(text)

        if not text:
            print(f"WARNING: Empty document skipped: {path}")
            continue

        documents.append({
            "source": os.path.basename(path),
            "text": text,
        })

    if not documents:
        raise SystemExit("All knowledge-base documents were empty.")

    print(
        f"Loaded {len(documents)} document(s): "
        f"{[d['source'] for d in documents]}"
    )

    return documents


# ============================================================
# SECTION DETECTION
# ============================================================

def is_heading(line):
    stripped = line.strip()

    if not stripped:
        return False

    if NUMBERED_HEADER_RE.match(stripped):
        return True

    if KNOWN_HEADING_RE.match(stripped):
        return True

    return False


def split_into_sections(text):
    lines = text.splitlines()

    sections = []
    current_title = "Document"
    current_lines = []

    def flush_current():
        nonlocal current_lines

        body = "\n".join(current_lines).strip()

        if len(body) >= MIN_CHUNK_CHARS:
            sections.append((current_title, body))

        current_lines = []

    for line in lines:
        if is_heading(line):
            flush_current()
            current_title = line.strip()
        else:
            current_lines.append(line)

    flush_current()

    if not sections:
        return [("Document", text.strip())]

    return sections


# ============================================================
# FAQ SPLITTING
# ============================================================

def split_faq_section(title, body):
    if not FAQ_Q_RE.search(body):
        return [(title, body)]

    parts = FAQ_Q_RE.split(body)
    results = []

    preamble = parts[0].strip()

    if preamble and len(preamble) >= MIN_CHUNK_CHARS:
        results.append((title, preamble))

    for part in parts[1:]:
        part = part.strip()

        if part:
            results.append((f"{title} — FAQ", "Q: " + part))

    return results


# ============================================================
# LONG SECTION SPLITTING
# ============================================================

def split_long_section(body, max_chars=MAX_CHUNK_CHARS):
    body = body.strip()

    if len(body) <= max_chars:
        return [body]

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", body)
        if p.strip()
    ]

    if len(paragraphs) == 1:
        paragraphs = [
            p.strip()
            for p in body.splitlines()
            if p.strip()
        ]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars and not current:
            chunks.append(paragraph)
            continue

        if not current:
            current = paragraph
            continue

        proposed = current + "\n\n" + paragraph

        if len(proposed) <= max_chars:
            current = proposed
        else:
            chunks.append(current.strip())
            current = paragraph

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    return [
        token
        for token in text.split()
        if token
    ]


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

KEYWORD_RE = re.compile(r"\b[A-Z][A-Z0-9&/-]{1,}\b")


def extract_keywords(text):
    return set(KEYWORD_RE.findall(text))


# ============================================================
# BUILD INDEX
# ============================================================

def build_index():
    documents = load_documents()

    all_chunks = []
    all_metadata = []
    all_tokens = []
    all_keywords = []

    for document in documents:
        source = document["source"]
        sections = split_into_sections(document["text"])

        for section_title, section_body in sections:
            faq_sections = split_faq_section(
                section_title,
                section_body
            )

            for faq_title, faq_body in faq_sections:
                sub_chunks = split_long_section(
                    faq_body,
                    MAX_CHUNK_CHARS
                )

                for chunk_number, sub_chunk in enumerate(
                    sub_chunks,
                    start=1
                ):
                    sub_chunk = sub_chunk.strip()

                    if len(sub_chunk) < MIN_CHUNK_CHARS:
                        continue

                    metadata_section = faq_title

                    if len(sub_chunks) > 1:
                        metadata_section = (
                            f"{faq_title} — Part {chunk_number}"
                        )

                    all_chunks.append(sub_chunk)

                    all_metadata.append({
                        "source": source,
                        "section": metadata_section,
                    })

                    all_tokens.append(
                        tokenize(sub_chunk)
                    )

                    all_keywords.append(
                        extract_keywords(sub_chunk)
                    )

    if not all_chunks:
        raise SystemExit(
            "No chunks were produced. "
            "Check the files inside knowledge_base/."
        )

    print()
    print("=" * 60)
    print(f"Created {len(all_chunks)} chunks.")
    print("=" * 60)

    os.makedirs(INDEX_DIR, exist_ok=True)

    output_files = {
        "chunks.pkl": all_chunks,
        "metadata.pkl": all_metadata,
        "tokens.pkl": all_tokens,
        "keywords.pkl": all_keywords,
    }

    for filename, data in output_files.items():
        output_path = os.path.join(INDEX_DIR, filename)

        with open(output_path, "wb") as file:
            pickle.dump(data, file)

        print(f"Saved: {output_path}")

    print()
    print("Index build completed successfully.")
    print()
    print("Next step:")
    print("    streamlit run app.py")
    print()


if __name__ == "__main__":
    build_index()
